"""
Name / address normalisation.

Every function is pure (str -> dict of str) so it can be mapped over the
unique values of a column with multiprocessing.
"""
from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

from . import patterns as P

try:  # best transliterator for all Indic scripts + accent folding
    from anyascii import anyascii as _anyascii
except ImportError:  # pragma: no cover - fallback keeps the pipeline running
    _anyascii = None

# ---------------------------------------------------------------------------
# Unicode helpers
# ---------------------------------------------------------------------------
_ZERO_WIDTH = re.compile(r"[​-‍﻿­]")
INDIC_RE = re.compile(r"[ऀ-෿]")
_NON_ASCII = re.compile(r"[^\x00-\x7F]")
_WS = re.compile(r"\s+")

_INDIC_LEGAL_SORTED = sorted(P.INDIC_LEGAL.items(), key=lambda kv: -len(kv[0]))
_NATIVE_STATES_SORTED = sorted(P.INDIA_STATES_NATIVE.items(), key=lambda kv: -len(kv[0]))


def _fallback_translit(text: str) -> str:
    """Transliterate without anyascii: accents via NFKD, Brahmic scripts via
    Unicode character names ("DEVANAGARI LETTER RA" -> "ra")."""
    out = []
    for ch in unicodedata.normalize("NFKD", text):
        if ord(ch) < 128:
            out.append(ch)
            continue
        if unicodedata.combining(ch) and not INDIC_RE.match(ch):
            continue
        name = unicodedata.name(ch, "")
        if " LETTER " in name:
            syl = name.split(" LETTER ")[-1].lower().split()[0]
            out.append(syl if len(syl) <= 3 else syl[:3])
        elif " VOWEL SIGN " in name:
            out.append(name.split(" VOWEL SIGN ")[-1].lower().split()[0][:2])
        elif " DIGIT " in name:
            out.append(str(unicodedata.digit(ch, 0)))
        elif "VIRAMA" in name and out and out[-1].endswith("a"):
            out[-1] = out[-1][:-1]
        else:
            out.append(" ")
    return "".join(out)


def to_ascii(text: str) -> str:
    if not _NON_ASCII.search(text):
        return text
    if _anyascii is not None:
        return _anyascii(text)
    return _fallback_translit(text)


def clean_unicode(text) -> str:
    if text is None:
        return ""
    text = unicodedata.normalize("NFKC", str(text))
    text = _ZERO_WIDTH.sub("", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Phonetic skeleton - makes transliterated Indic names comparable with the
# English spelling ("sliusns" ~ "solutions", "piraivet" ~ "private").
# ---------------------------------------------------------------------------
_SKEL_SUBS = [
    ("tion", "sn"), ("sion", "sn"), ("tian", "sn"), ("ph", "f"), ("sh", "s"),
    ("ch", "c"), ("th", "t"), ("kh", "k"), ("gh", "g"), ("bh", "b"),
    ("dh", "d"), ("jh", "j"), ("ck", "k"), ("qu", "k"), ("x", "ks"),
    ("w", "v"), ("z", "s"), ("ce", "se"), ("ci", "si"), ("cy", "sy"),
    ("ge", "je"), ("gi", "ji"), ("gy", "jy"),
    ("c", "k"), ("q", "k"), ("b", "p"), ("d", "t"),
    ("g", "k"), ("m", "n"),
]
_VOWELS = set("aeiouyh")


@lru_cache(maxsize=100_000)
def skeleton(token: str) -> str:
    if not token:
        return ""
    if token.isdigit():
        return token.lstrip("0") or "0"
    t = token
    for a, b in _SKEL_SUBS:
        t = t.replace(a, b)
    head, rest = t[0], [c for c in t[1:] if c not in _VOWELS]
    out = [head]
    for c in rest:
        if c != out[-1]:
            out.append(c)
    return "".join(out)


_LEGAL_SKELETONS = {skeleton(w) for w in P.LEGAL_SKELETON_WORDS}


def ocr_fold(text: str) -> str:
    """Symmetric OCR folding for comparison keys only."""
    out = []
    for tok in text.split():
        if any(c.isalpha() for c in tok):
            tok = "".join(P.OCR_DIGIT_TO_CHAR.get(c, c) for c in tok)
        for a, b in P.OCR_CHAR_FOLD:
            tok = tok.replace(a, b)
        out.append(tok)
    return " ".join(out)


# ---------------------------------------------------------------------------
# Business names
# ---------------------------------------------------------------------------
_ALIAS_RE = re.compile("|".join(P.ALIAS_MARKERS))
_JUNK_RES = [re.compile(r) for r in P.JUNK_REGEXES]
_DOTTED_ABBR = re.compile(r"\b(?:[a-z]\.){2,}[a-z]?\.?")
_DOMAIN_RE = re.compile(
    r"^(?:www\.)?([a-z0-9][a-z0-9\-]*?)\.(" + "|".join(
        re.escape(t) for t in sorted(P.TLDS, key=len, reverse=True)) + r")\.?$")
_NAME_PUNCT = re.compile(r"[^a-z0-9\s]")
_LEGAL_MULTI = [(re.compile(r"\b" + re.escape(a) + r"\b"), b) for a, b in P.LEGAL_MULTIWORD]
_LEGAL_TOKENS = dict(P.LEGAL_TOKENS, lnc="inc", ilc="llc", iic="llc", ltdd="ltd")


def _fix_ocr_token(tok: str) -> str:
    # digits inside a word are OCR noise: 5outh, washingt0n, dermato1ogy
    if any(c.isdigit() for c in tok) and sum(c.isalpha() for c in tok) >= 2:
        return "".join(P.OCR_DIGIT_TO_CHAR.get(c, c) for c in tok)
    return tok


def _clean_single_name(s: str, from_indic: bool):
    flags = {"domain": 0, "junk": 0}
    s = s.strip()
    for rx in _JUNK_RES:
        new = rx.sub(" ", s)
        if new != s:
            flags["junk"] = 1
            s = new
    s = s.strip()
    # whole name written as a web domain: aahanarealty.com / silvergrill.c0m
    m = _DOMAIN_RE.match(s.replace(" ", ""))
    if m and " " not in s.strip():
        s = m.group(1).replace("-", " ")
        flags["domain"] = 1
    s = _DOTTED_ABBR.sub(lambda mm: mm.group(0).replace(".", ""), s)
    s = s.replace("&", " and ").replace("+", " and ").replace("'s ", "s ")
    s = s.replace("'", "")
    s = _NAME_PUNCT.sub(" ", s)
    s = _WS.sub(" ", s).strip()
    for rx, rep in _LEGAL_MULTI:
        s = rx.sub(rep, s)

    core, legal = [], set()
    for tok in s.split():
        tok = _fix_ocr_token(tok)
        if tok in _LEGAL_TOKENS:
            for lg in _LEGAL_TOKENS[tok].split():
                legal.add(lg)
            continue
        if from_indic and len(tok) > 2 and skeleton(tok) in _LEGAL_SKELETONS:
            legal.add(tok)
            continue
        if tok in P.NAME_STOPWORDS or tok in P.HONORIFICS:
            continue
        if core and core[-1] == tok:  # repeated word: "bright bright"
            continue
        core.append(tok)
    if not core and legal:  # name was only legal words - keep something
        core = sorted(legal)
    return core, legal, flags


@lru_cache(maxsize=20_000)
def normalize_name(raw) -> dict:
    s = clean_unicode(raw)
    has_indic = bool(INDIC_RE.search(s))
    if has_indic:
        for native, rep in _INDIC_LEGAL_SORTED:
            s = s.replace(native, rep)
    s = to_ascii(s).lower()

    parts = [p for p in _ALIAS_RE.split(s) if p and p.strip()]
    if not parts:
        parts = [s]
    alts = []
    legal_all: set = set()
    flags = {"domain": 0, "junk": 0}
    for p in parts:
        core, legal, fl = _clean_single_name(p, has_indic)
        if core:
            alts.append(core)
        legal_all |= legal
        for k in flags:
            flags[k] = max(flags[k], fl[k])
    if not alts:
        alts = [[]]
    # primary = the longest alternative (usually the real business name)
    main = max(alts, key=lambda c: len(" ".join(c)))
    core_str = " ".join(main)
    skel = " ".join(skeleton(t) for t in main)
    return {
        "core": core_str,
        "full": (core_str + " " + " ".join(sorted(legal_all))).strip(),
        "ocr": ocr_fold(core_str),
        "skel": skel,
        "nospace": core_str.replace(" ", ""),
        "alts": "||".join(dict.fromkeys(" ".join(a) for a in alts if a)) if len(alts) > 1 else "",
        "legal": " ".join(sorted(legal_all)),
        "acr": "".join(t[0] for t in main) if len(main) >= 2 else "",
        "indic": int(has_indic),
        "domain": flags["domain"],
        "junk": int(flags["junk"] or len(alts) > 1),
    }


# ---------------------------------------------------------------------------
# Addresses
# ---------------------------------------------------------------------------
_NULL_RE = re.compile(r"<\s*null\s*>|\b(?:null|none|nan|n/a)\b", re.I)
_ADDR_PUNCT = re.compile(r"[^a-z0-9\s/\-#,]")
_ORD_SUFFIX = re.compile(r"^(\d+)(st|nd|rd|th|er|e|eme|re)$")
_DIGITS = re.compile(r"\d+")
_LABEL_GLUE = re.compile(r"\b(h|d|f|s|sy|kh|r|p)\s*\.\s*no(?=\b|\d)")  # h.no -> hno
_LABEL_DIGIT = re.compile(r"\b(no|hno|dno|fno|sno|plot|flat|door|shop|unit|apt|suite|ste|room|office|house)(?=\d)")
_WORD_HYPHEN = re.compile(r"(?<=[a-z])-(?=[a-z])")
_FRANCE_ONLY = {"r", "ch", "all", "che", "imp", "ld", "res"}
_NO_GLUE = re.compile(r"\bno\s*[.:]\s*")
_HASH = re.compile(r"#+\s*")

_US_CODES = set(P.US_STATES.values())
_IN_CODES = set(P.INDIA_STATES.values()) | set(P.INDIA_STATE_CODE_ALIASES)
_ALL_STATE_NAMES = sorted(
    list(P.US_STATES.items()) + list(P.INDIA_STATES.items()) + list(P.FRANCE_REGIONS.items()),
    key=lambda kv: -len(kv[0]))
_STATE_NAME_RES = [(re.compile(r"\b" + re.escape(n) + r"\b"), c) for n, c in _ALL_STATE_NAMES]
_CITY_ALIAS_RES = [(re.compile(r"\b" + re.escape(a) + r"\b"), c)
                   for a, c in sorted(P.CITY_ALIASES.items(), key=lambda kv: -len(kv[0]))]
_STATE_CODES = _US_CODES | _IN_CODES | set(P.FRANCE_REGIONS.values())


def canon_number(tok: str) -> str:
    """0271a -> 271a, 4Nd -> 4, 05131 -> 5131, b-204 -> b-204."""
    m = _ORD_SUFFIX.match(tok)
    if m:
        return m.group(1).lstrip("0") or "0"
    return _DIGITS.sub(lambda mm: mm.group(0).lstrip("0") or "0", tok).strip("-/")


@lru_cache(maxsize=20_000)
def normalize_address(raw, country: str = "") -> dict:
    s = clean_unicode(raw)
    for native, code in _NATIVE_STATES_SORTED:
        if native in s:
            s = s.replace(native, f" {code} ")
    s = to_ascii(s).lower()
    s = _NULL_RE.sub(" ", s)
    s = _LABEL_GLUE.sub(lambda m: m.group(1) + "no ", s)
    s = _NO_GLUE.sub("no ", s)
    s = _LABEL_DIGIT.sub(lambda m: m.group(1) + " ", s)
    s = _WORD_HYPHEN.sub(" ", s)
    s = _HASH.sub(" # ", s)
    s = s.replace(".", " ").replace(";", ",").replace("(", " ").replace(")", " ")
    s = _ADDR_PUNCT.sub(" ", s)
    for rx, c in _CITY_ALIAS_RES:
        s = rx.sub(c, s)

    alpha, nums, numparts, numstreet, states = [], [], set(), [], set()
    ordered = []
    for comp in s.split(","):
        comp = _WS.sub(" ", comp).strip(" -/")
        if not comp:
            continue
        if comp in _STATE_CODES and (
                (country == "US" and comp in _US_CODES)
                or (country == "India" and (comp in _IN_CODES))
                or country not in ("US", "India")):
            states.add(P.INDIA_STATE_CODE_ALIASES.get(comp, comp) if country == "India" else comp)
            continue
        for rx, c in _STATE_NAME_RES:
            if rx.search(comp):
                states.add(c)
                comp = rx.sub(" ", comp)
        toks = comp.split()
        prev_num = None
        for i, tok in enumerate(toks):
            tok = tok.strip("-/")
            if not tok or tok == "#":
                continue
            if tok in P.ORDINAL_WORDS and i + 1 < len(toks) and toks[i + 1] in ("floor", "fl", "flr", "etage"):
                tok = P.ORDINAL_WORDS[tok]
            if any(c.isdigit() for c in tok):
                cn = canon_number(tok)
                if cn:
                    nums.append(cn)
                    numparts.update(d.lstrip("0") or "0" for d in _DIGITS.findall(tok))
                    ordered.append(cn)
                    prev_num = cn
                continue
            if tok in P.UNIT_LABELS or tok in P.ADDRESS_STOPWORDS:
                continue
            if country == "France" or tok not in _FRANCE_ONLY:
                tok = P.STREET_TYPES.get(tok, tok)
            tok = P.DIRECTIONALS.get(tok, tok)
            if not tok:
                continue
            if len(tok) == 2 and tok in _STATE_CODES and i == len(toks) - 1 and len(toks) <= 2:
                states.add(tok)
                continue
            alpha.append(tok)
            ordered.append(tok)
            if prev_num is not None:
                numstreet.append(f"{prev_num}_{tok}")
                prev_num = None

    postal = sorted(n for n in nums if n.isdigit() and len(n) in (5, 6))
    return {
        "atext": " ".join(ordered),
        "alpha": " ".join(alpha),
        "nums": " ".join(nums),
        "numparts": " ".join(sorted(numparts)),
        "numstreet": " ".join(numstreet),
        "first_num": nums[0] if nums else "",
        "state": " ".join(sorted(states)),
        "postal": " ".join(postal),
        "missing": int(not ordered),
    }
