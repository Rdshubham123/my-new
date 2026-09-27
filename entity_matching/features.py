"""
Pairwise comparison features for (S1, target) candidate pairs.

Groups
------
name      : fuzzy ratios on several normalised views (core / OCR-folded /
            phonetic skeleton / space-less), IDF-weighted token overlap,
            alias (DBA) best match, domain containment, acronym, legal form.
address   : fuzzy ratios, IDF-weighted word overlap, number agreement
            (exact / compound / parts / house number / number+street),
            state agreement, missing flags.
rules     : hard-coded boolean patterns (exact core + shared number, ...)
context   : rank / gap of the pair among the S1's candidates and among the
            S1 records competing for the same target (one-owner, cell 8).
"""
from __future__ import annotations

import math
from collections import Counter
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

from .utils import run_pool, worker_view

PAIR_FEATURES = [
    # name
    "n_ratio_full", "n_ratio_core", "n_tsort_core", "n_tset_core", "n_partial_core",
    "n_wratio_core", "n_jw_nospace", "n_ratio_ocr", "n_tset_ocr", "n_ratio_skel",
    "n_tset_skel", "n_idf_jacc", "n_idf_contain", "n_idf_shared_max",
    "n_idf_miss_s", "n_idf_miss_t", "n_alt_best", "n_nospace_contain",
    "n_nospace_partial", "n_acr", "n_legal_both", "n_legal_eq", "n_legal_conflict",
    "n_ntok_s", "n_ntok_t", "n_ntok_diff", "n_exact_core", "n_exact_skel",
    "n_core_df_t", "s_indic", "t_indic", "t_domain", "t_junk",
    # address
    "a_missing_t", "a_ratio", "a_tset", "a_tsort", "a_partial", "a_idf_jacc",
    "a_idf_contain", "a_ratio_alpha_nospace", "num_s", "num_t", "num_shared",
    "num_jacc", "num_t_only", "num_s_only", "num_compound_shared", "numparts_shared",
    "numparts_jacc", "first_num_eq", "numstreet_shared", "state_eq", "postal_eq",
    "num_idf_shared", "num_idf_max", "num_idf_t_only", "first_num_conflict",
    # rules
    "r_exact_core_num", "r_skel_num", "r_domain_num", "r_alias_num",
    "r_name_only_strong", "r_addr_only_strong",
    # misc
    "is_s3",
]
BLOCK_FEATURES = ["blk_score", "blk_name", "blk_addr", "blk_nkeys"]
CONTEXT_FEATURES = [
    "ctx_ncand_s", "ctx_rank_s", "ctx_gap_s", "ctx_ncand_t", "ctx_rank_t", "ctx_gap_t",
    "ctx_blk_rank_s", "ctx_same_core_in_s", "ctx_same_num_in_s",
]
ALL_FEATURES = PAIR_FEATURES + BLOCK_FEATURES + CONTEXT_FEATURES

_LEGAL_EQUIV = [{"llc", "pllc", "lc"}, {"inc", "corp"}, {"ltd", "pvt", "public", "plc"},
                {"llp", "lp"}, {"pc", "pa"}, {"sarl", "eurl"}, {"sas", "sasu", "sa"}]

# globals shared with forked workers
_G: dict = {}


def build_idf(tgt: pd.DataFrame) -> dict:
    """Per-country IDF tables from TARGET records (never from S1/test labels)."""
    idf = {}
    for country, grp in tgt.groupby(tgt["country"].astype(str), observed=True):
        n = len(grp)
        cn, ca, cu = Counter(), Counter(), Counter()
        for s in grp["n_skel"]:
            cn.update(set(s.split()))
        for s in grp["a_alpha"]:
            ca.update(set(s.split()))
        for s in grp["a_nums"]:
            cu.update(set(s.split()))
        core_df = Counter(grp["n_skel"])
        idf[country] = {
            "n": {k: math.log1p(n / v) for k, v in cn.items()},
            "a": {k: math.log1p(n / v) for k, v in ca.items()},
            "u": {k: math.log1p(n / v) for k, v in cu.items()},
            "core_df": dict(core_df),
            "default": math.log1p(n),
        }
    return idf


def _idf_overlap(a_toks, b_toks, table, default):
    if not a_toks or not b_toks:
        return 0.0, 0.0, 0.0, 1.0, 1.0
    sa, sb = set(a_toks), set(b_toks)
    w = lambda t: table.get(t, default)  # noqa: E731
    inter = sa & sb
    wi = sum(w(t) for t in inter)
    wa = sum(w(t) for t in sa)
    wb = sum(w(t) for t in sb)
    wu = wa + wb - wi
    return (wi / wu if wu else 0.0,
            wi / min(wa, wb) if min(wa, wb) else 0.0,
            max((w(t) for t in inter), default=0.0),
            (wa - wi) / wa if wa else 1.0,
            (wb - wi) / wb if wb else 1.0)


def _legal_rel(a, b):
    sa, sb = set(a.split()), set(b.split())
    both = int(bool(sa) and bool(sb))
    eq = int(both and sa == sb)
    conflict = 0
    if both and not (sa & sb):
        conflict = 1
        for grp in _LEGAL_EQUIV:
            if sa & grp and sb & grp:
                conflict = 0
    return both, eq, conflict


def _pair(i, j):
    S, T = _G["S"], _G["T"]
    country = str(S["country"][i])
    tab = _G["idf"].get(country) or {"n": {}, "a": {}, "u": {}, "core_df": {}, "default": 10.0}
    d = tab["default"]

    sc, tc = S["n_core"][i], T["n_core"][j]
    sf, tf = S["n_full"][i], T["n_full"][j]
    so, to = S["n_ocr"][i], T["n_ocr"][j]
    ss, ts = S["n_skel"][i], T["n_skel"][j]
    sn, tn = S["n_nospace"][i], T["n_nospace"][j]
    stoks, ttoks = ss.split(), ts.split()

    jacc, contain, smax, miss_s, miss_t = _idf_overlap(stoks, ttoks, tab["n"], d)

    alt_best = 0.0
    for alt in (T["n_alts"][j].split("||") if T["n_alts"][j] else []):
        alt_best = max(alt_best, fuzz.token_set_ratio(sc, alt))
    for alt in (S["n_alts"][i].split("||") if S["n_alts"][i] else []):
        alt_best = max(alt_best, fuzz.token_set_ratio(alt, tc))

    short, long_ = (sn, tn) if len(sn) <= len(tn) else (tn, sn)
    nospace_contain = int(len(short) >= 4 and short in long_)
    acr = int(bool(S["n_acr"][i] and S["n_acr"][i] == tn) or bool(T["n_acr"][j] and T["n_acr"][j] == sn))
    lboth, leq, lconf = _legal_rel(S["n_legal"][i], T["n_legal"][j])

    # address
    sa_txt, ta_txt = S["a_atext"][i], T["a_atext"][j]
    salpha, talpha = S["a_alpha"][i].split(), T["a_alpha"][j].split()
    ajacc, acontain, _, _, _ = _idf_overlap(salpha, talpha, tab["a"], d)
    snum, tnum = set(S["a_nums"][i].split()), set(T["a_nums"][j].split())
    shared = snum & tnum
    union = snum | tnum
    comp_shared = sum(1 for x in shared if ("/" in x or "-" in x or not x.isdigit()))
    sp, tp = set(S["a_numparts"][i].split()), set(T["a_numparts"][j].split())
    sp_sh = len(sp & tp)
    sp_un = len(sp | tp)
    sns, tns = set(S["a_numstreet"][i].split()), set(T["a_numstreet"][j].split())
    sst, tst = S["a_state"][i], T["a_state"][j]
    state_eq = (1 if set(sst.split()) & set(tst.split()) else -1) if sst and tst else 0
    spo, tpo = S["a_postal"][i], T["a_postal"][j]
    postal_eq = (1 if set(spo.split()) & set(tpo.split()) else -1) if spo and tpo else 0
    a_missing = int(T["a_missing"][j])

    n_tset_skel = fuzz.token_set_ratio(ss, ts)
    a_tset = fuzz.token_set_ratio(sa_txt, ta_txt) if not a_missing else -1.0
    exact_core = int(sc == tc and sc != "")
    exact_skel = int(ss == ts and ss != "")
    n_shared = len(shared)
    # rarity-weighted number evidence: sharing "63/1/2" >> sharing "12"
    ut = tab["u"]
    num_idf = [ut.get(x, d) for x in shared]
    num_idf_t_only = sum(ut.get(x, d) for x in (tnum - snum))
    f_s, f_t = S["a_first_num"][i], T["a_first_num"][j]
    first_conflict = int(bool(f_s) and bool(f_t) and f_s != f_t and f_t not in snum and f_s not in tnum)

    return (
        fuzz.ratio(sf, tf), fuzz.ratio(sc, tc), fuzz.token_sort_ratio(sc, tc),
        fuzz.token_set_ratio(sc, tc), fuzz.partial_ratio(sc, tc), fuzz.WRatio(sc, tc),
        JaroWinkler.normalized_similarity(sn, tn) * 100, fuzz.ratio(so, to),
        fuzz.token_set_ratio(so, to), fuzz.ratio(ss, ts), n_tset_skel,
        jacc, contain, smax, miss_s, miss_t, alt_best, nospace_contain,
        fuzz.partial_ratio(sn, tn), acr, lboth, leq, lconf,
        len(stoks), len(ttoks), abs(len(stoks) - len(ttoks)), exact_core, exact_skel,
        math.log1p(tab["core_df"].get(ts, 0)), S["n_indic"][i], T["n_indic"][j],
        T["n_domain"][j], T["n_junk"][j],
        a_missing,
        fuzz.ratio(sa_txt, ta_txt) if not a_missing else -1.0,
        a_tset,
        fuzz.token_sort_ratio(sa_txt, ta_txt) if not a_missing else -1.0,
        fuzz.partial_ratio(sa_txt, ta_txt) if not a_missing else -1.0,
        ajacc, acontain,
        fuzz.ratio("".join(sorted(salpha)), "".join(sorted(talpha))) if not a_missing else -1.0,
        len(snum), len(tnum), n_shared, n_shared / len(union) if union else 0.0,
        len(tnum - snum), len(snum - tnum), comp_shared, sp_sh, sp_sh / sp_un if sp_un else 0.0,
        int(bool(S["a_first_num"][i]) and S["a_first_num"][i] == T["a_first_num"][j]),
        len(sns & tns), state_eq, postal_eq,
        sum(num_idf), max(num_idf, default=0.0), num_idf_t_only, first_conflict,
        # rules
        int(exact_core and n_shared > 0), int(exact_skel and n_shared > 0),
        int(T["n_domain"][j] and nospace_contain and n_shared > 0),
        int(alt_best >= 95 and n_shared > 0),
        int(n_tset_skel >= 95 and contain >= 0.9),
        int((a_tset >= 90) and n_shared >= 2),
        int(str(T["source"][j]) == "S3"),
    )


def _worker(bounds):
    lo, hi = bounds
    si, ti = _G["si"], _G["ti"]
    out = np.empty((hi - lo, len(PAIR_FEATURES)), np.float32)
    for r in range(lo, hi):
        out[r - lo] = _pair(si[r], ti[r])
    return lo, out


_S_COLS = ["country", "n_core", "n_full", "n_ocr", "n_skel", "n_nospace", "n_alts",
           "n_acr", "n_legal", "n_indic", "a_atext", "a_alpha", "a_nums", "a_numparts",
           "a_numstreet", "a_state", "a_postal", "a_first_num", "a_missing"]
_T_COLS = _S_COLS + ["n_domain", "n_junk", "source"]


def compute_pair_features(cands: pd.DataFrame, s1: pd.DataFrame, tgt: pd.DataFrame,
                          idf: dict, n_jobs: int = 4, chunk: int = 20000) -> pd.DataFrame:
    # zero-copy numpy views, shared with forked workers (nothing is pickled)
    _G["S"] = worker_view(s1, _S_COLS)   # COW-safe string access in workers
    _G["T"] = worker_view(tgt, _T_COLS)
    _G["idf"] = idf
    _G["si"] = cands["s_row"].to_numpy()
    _G["ti"] = cands["t_row"].to_numpy()
    n = len(cands)
    X = np.empty((n, len(PAIR_FEATURES)), np.float32)
    for lo, arr in run_pool(_worker, [(lo, min(lo + chunk, n)) for lo in range(0, n, chunk)], n_jobs):
        X[lo:lo + len(arr)] = arr
    _G.clear()
    feats = pd.DataFrame(X, columns=PAIR_FEATURES, index=cands.index)
    return pd.concat([cands, feats], axis=1)


def add_target_context(cands: pd.DataFrame) -> pd.DataFrame:
    """Competition for the same target across ALL S1 records (one-owner
    constraint).  Uses blocking scores so it can be computed globally before
    the batched feature pass."""
    g = cands.groupby("t_row")["blk_score"]
    cands["ctx_ncand_t"] = g.transform("size").astype(np.float32)
    cands["ctx_rank_t"] = g.rank(ascending=False, method="min").astype(np.float32)
    cands["ctx_gap_t"] = (g.transform("max") - cands["blk_score"]).astype(np.float32)
    return cands


def add_s1_context(df: pd.DataFrame, tgt: pd.DataFrame) -> pd.DataFrame:
    """Ranking features within each S1's candidate list (needs all candidates
    of an S1 in the same batch)."""
    comb = (df["n_tset_skel"].to_numpy() + df["n_idf_contain"].to_numpy() * 100
            + np.clip(df["a_tset"].to_numpy(), 0, None) + 20 * df["num_jacc"].to_numpy())
    df["_comb"] = comb
    gs = df.groupby("s_row")["_comb"]
    df["ctx_ncand_s"] = gs.transform("size").astype(np.float32)
    df["ctx_rank_s"] = gs.rank(ascending=False, method="min").astype(np.float32)
    df["ctx_gap_s"] = (gs.transform("max") - comb).astype(np.float32)
    df["ctx_blk_rank_s"] = df.groupby("s_row")["blk_score"].rank(
        ascending=False, method="min").astype(np.float32)
    # siblings: an S1 usually has several S2/S3 duplicates of itself
    # group on integer codes (categorical columns) - no string materialisation
    tr = df["t_row"].to_numpy()
    df["_core"] = _codes(tgt["n_skel"])[tr]
    fcol = tgt["a_first_num"]
    df["_fnum"] = _codes(fcol)[tr]
    df["ctx_same_core_in_s"] = df.groupby(["s_row", "_core"])["_comb"].transform("size").astype(np.float32)
    df["ctx_same_num_in_s"] = df.groupby(["s_row", "_fnum"])["_comb"].transform("size").astype(np.float32)
    df.loc[df["_fnum"] == _empty_code(fcol), "ctx_same_num_in_s"] = 0
    return df.drop(columns=["_comb", "_core", "_fnum"])


def _codes(col):
    if isinstance(col.dtype, pd.CategoricalDtype):
        return col.cat.codes.to_numpy()
    return pd.factorize(col)[0]


def _empty_code(col):
    if isinstance(col.dtype, pd.CategoricalDtype):
        cats = col.cat.categories
        return int(cats.get_loc("")) if "" in cats else -99
    codes, uniq = pd.factorize(col)
    return int(np.flatnonzero(uniq == "")[0]) if (uniq == "").any() else -99
