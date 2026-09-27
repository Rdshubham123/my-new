"""
Candidate generation (blocking).

Country is a HARD block (cell 9: 100 % of true links are same-country).
Inside a country every record emits hashed keys of several types; an S1
record is paired with every target sharing one of its rarest keys and the
pair is scored by the IDF-weighted sum of shared keys.  We keep the top-K
pairs per S1 by total score plus the top-K2 by name-only and address-only
score, so a record whose name is unrecognisable (cell 17: names replaced by
"Kelopyrahalo") can still be found through its address, and vice versa.

EDA motivation: name token OR number gives 96.9 % recall and adding rare
address tokens 99.7 % (cells 21-23) - but with capped fan-out.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
import pandas as pd

from .normalize import skeleton
from .utils import free, run_pool

_G: dict = {}

# key type -> (weight, max document frequency, max keys per record, group)
# group: 0 = name evidence, 1 = address evidence
KEY_TYPES = {
    "N": (1.0, 2000, 6, 0),   # skeleton name token
    "B": (1.6, 2000, 4, 0),   # skeleton name bigram
    "P": (1.6, 1000, 2, 0),   # first 8 chars of space-less core (domains, spacing)
    "F": (3.0, 2000, 1, 0),   # full skeleton name
    "A": (0.6, 3000, 6, 1),   # address word
    "U": (1.0, 3000, 6, 1),   # address number (canonical)
    "S": (2.2, 1000, 4, 1),   # number + following street word
    "C": (2.5, 1000, 6, 1),   # name token x address number ("arihant|417")
}
TYPE_CODE = {k: i for i, k in enumerate(KEY_TYPES)}


@dataclass
class BlockConfig:
    top_k: int = 60          # k_max of the adaptive forward list
    k_min: int = 12          # always keep this many per S1
    adapt_ratio: float = 0.25  # beyond k_min keep only score >= ratio * top1 (SABER-style)
    top_k_name: int = 10
    top_k_addr: int = 10
    reverse_k: int = 5       # bidirectional: each target keeps its top-R S1 records
    posting_budget: int = 2500  # max target postings expanded per S1 record
    min_keys: int = 6        # ... but always use the 6 rarest keys
    pair_budget: int = 5_000_000  # max expanded postings per chunk (~0.6 GB temp)
    max_df_frac: float = 0.01
    n_jobs: int = 4
    key_types: dict = field(default_factory=lambda: dict(KEY_TYPES))


def _keys_for(skel, alts, nospace, alpha, nums, numstreet):
    out = []
    toks = [t for t in skel.split() if len(t) >= 2]
    for t in dict.fromkeys(toks):
        out.append(("N", t))
    for a, b in zip(toks, toks[1:]):
        out.append(("B", a + "_" + b))
    if alts:
        for alt in alts.split("||"):
            for t in alt.split():
                s = skeleton(t)
                if len(s) >= 2:
                    out.append(("N", s))
            ns = alt.replace(" ", "")
            if len(ns) >= 5:
                out.append(("P", ns[:8]))
    if len(nospace) >= 5:
        out.append(("P", nospace[:8]))
    if skel:
        out.append(("F", skel.replace(" ", "")))
    for t in dict.fromkeys(alpha.split()):
        if len(t) >= 3:
            out.append(("A", t))
    for t in dict.fromkeys(nums.split()):
        out.append(("U", t))
    for t in dict.fromkeys(numstreet.split()):
        out.append(("S", t))
    # composite keys stay rare even when the name word and the number are
    # both common (generic names such as "pediatric group", EDA cell 20)
    # 2 longest (≈ rarest) name tokens x first 3 numbers = at most 6 keys
    num_list = list(dict.fromkeys(nums.split()))[:3]
    for t in sorted(dict.fromkeys(toks), key=len, reverse=True)[:2]:
        for n in num_list:
            out.append(("C", t + "|" + n))
    return out


def _key_worker(b):
    lo, hi = b
    cols = [c[lo:hi] for c in _G["cols"]]
    idx, codes, strs = [], [], []
    for i, row in enumerate(zip(*cols)):
        seen = set()
        for typ, val in _keys_for(*row):
            k = typ + ":" + val
            if k in seen:  # de-duplicate per record here: no pandas pass later
                continue
            seen.add(k)
            idx.append(lo + i)
            codes.append(TYPE_CODE[typ])
            strs.append(k)
    h = pd.util.hash_array(np.array(strs, dtype=object)) if strs else np.zeros(0, np.uint64)
    return (np.asarray(idx, np.int32), np.asarray(codes, np.int8), h)


def build_keys(df: pd.DataFrame, n_jobs: int = 4, chunk: int = 50000):
    """Hashed blocking keys -> (record index int32, key type int8, hash uint64)."""
    cols_names = ["n_skel", "n_alts", "n_nospace", "a_alpha", "a_nums", "a_numstreet"]
    _G["cols"] = [df[c].to_numpy() for c in cols_names]  # zero-copy, fork-shared
    res = run_pool(_key_worker, [(s, min(s + chunk, len(df))) for s in range(0, len(df), chunk)], n_jobs)
    _G.clear()
    if not res:
        return np.zeros(0, np.int32), np.zeros(0, np.int8), np.zeros(0, np.uint64)
    out = (np.concatenate([r[0] for r in res]),
           np.concatenate([r[1] for r in res]),
           np.concatenate([r[2] for r in res]))
    del res
    return out


def _topk_mask(s, score, k):
    """Boolean mask of the top-k rows by score within each s (s sorted not required)."""
    order = np.lexsort((-score, s))
    s_sorted = s[order]
    first = np.r_[True, s_sorted[1:] != s_sorted[:-1]]
    grp_start = np.maximum.accumulate(np.where(first, np.arange(len(s_sorted)), 0))
    rank = np.arange(len(s_sorted)) - grp_start
    mask = np.zeros(len(s), bool)
    mask[order[rank < k]] = True
    return mask


def generate_candidates(s1: pd.DataFrame, tgt: pd.DataFrame, cfg: BlockConfig,
                        s1_rows: np.ndarray | None = None, log=print) -> pd.DataFrame:
    """Returns DataFrame[s_row, t_row, blk_score, blk_name, blk_addr, blk_nkeys]."""
    kt = cfg.key_types
    weights = np.array([v[0] for v in kt.values()])
    caps = np.array([v[1] for v in kt.values()])
    per_rec = np.array([v[2] for v in kt.values()])
    group = np.array([v[3] for v in kt.values()])

    if s1_rows is None:
        s1_rows = np.arange(len(s1))
    out, rev_pool = [], []
    s_ctry = s1["country"].astype(str).to_numpy()
    t_ctry = tgt["country"].astype(str).to_numpy()
    for country in sorted(pd.unique(s_ctry[s1_rows])):
        s_rows = s1_rows[s_ctry[s1_rows] == country]
        t_rows = np.flatnonzero(t_ctry == country)
        if len(s_rows) == 0 or len(t_rows) == 0:
            continue
        log(f"  [{country}] S1={len(s_rows):,} targets={len(t_rows):,}")
        t_idx, t_typ, t_h = build_keys(tgt.iloc[t_rows], cfg.n_jobs)
        # one sort gives document frequencies AND posting lists (no pandas,
        # no np.unique inverse: peak RAM ~ 3 x 8 bytes per key)
        order = np.argsort(t_h, kind="stable")
        post_h = t_h[order]
        del t_h
        post_t = t_idx[order]
        del t_idx
        typ_sorted = t_typ[order]
        del t_typ, order
        free()
        brk = np.flatnonzero(np.r_[True, post_h[1:] != post_h[:-1]])
        klen = np.diff(np.r_[brk, len(post_h)])
        # caps are absolute for the full data and shrink with small target sets
        eff_caps = np.maximum(20, np.minimum(caps, cfg.max_df_frac * len(t_rows))).astype(np.int64)
        keep_key = klen <= eff_caps[typ_sorted[brk]]
        del typ_sorted
        keep_el = np.repeat(keep_key, klen)
        post_h, post_t = post_h[keep_el], post_t[keep_el]
        del keep_el
        ph_len = klen[keep_key]
        ph_start = np.r_[0, np.cumsum(ph_len)[:-1]] if len(ph_len) else np.zeros(0, np.int64)
        ph_uniq = post_h[ph_start] if len(ph_len) else np.zeros(0, np.uint64)
        del post_h, brk, klen, keep_key
        free()
        n_t = len(t_rows)

        if len(ph_uniq) == 0:
            continue
        s_idx, s_typ, s_h = build_keys(s1.iloc[s_rows], cfg.n_jobs)
        pos = np.clip(np.searchsorted(ph_uniq, s_h), 0, len(ph_uniq) - 1)
        found = ph_uniq[pos] == s_h
        s_idx, s_typ, pos = s_idx[found], s_typ[found], pos[found]
        del s_h, found
        dfk = ph_len[pos]
        # keep the per_rec rarest keys of each type per S1 record
        o = np.lexsort((dfk, s_typ, s_idx))
        s_idx, s_typ, pos, dfk = s_idx[o], s_typ[o], pos[o], dfk[o]
        grp = np.r_[True, (s_idx[1:] != s_idx[:-1]) | (s_typ[1:] != s_typ[:-1])]
        gstart = np.maximum.accumulate(np.where(grp, np.arange(len(grp)), 0))
        keep = (np.arange(len(grp)) - gstart) < per_rec[s_typ]
        s_idx, s_typ, pos, dfk = s_idx[keep], s_typ[keep], pos[keep], dfk[keep]
        # per-S1 posting budget: rarest keys first, stop once ~budget target
        # postings are covered (always keep the min_keys rarest) -> bounded time
        o = np.lexsort((dfk, s_idx))
        s_idx, s_typ, pos, dfk = s_idx[o], s_typ[o], pos[o], dfk[o]
        first = np.r_[True, s_idx[1:] != s_idx[:-1]] if len(s_idx) else np.zeros(0, bool)
        gstart = np.maximum.accumulate(np.where(first, np.arange(len(s_idx)), 0))
        cum = np.cumsum(dfk)
        before = cum - dfk - (cum[gstart] - dfk[gstart])
        rank = np.arange(len(s_idx)) - gstart
        keep = (rank < cfg.min_keys) | (before < cfg.posting_budget)
        sk = pd.DataFrame({"i": s_idx[keep], "p": pos[keep], "c": s_typ[keep], "df": dfk[keep]})
        del s_idx, s_typ, pos, dfk, o, grp, gstart, keep, first, cum, before, rank
        sk["w"] = weights[sk["c"].to_numpy()] * np.log1p(n_t / sk["df"].to_numpy())
        sk["g"] = group[sk["c"].to_numpy()]

        # chunk S1 records so each chunk expands to <= pair_budget postings:
        # RAM stays bounded whatever the key frequencies are
        si_all = sk["i"].to_numpy()
        cum = np.cumsum(ph_len[sk["p"].to_numpy()])
        rec_start = np.flatnonzero(np.r_[True, si_all[1:] != si_all[:-1]])
        rec_cum = np.r_[0, cum][rec_start]
        cuts = rec_start[np.unique(np.searchsorted(
            rec_cum, np.arange(0, (cum[-1] if len(cum) else 0) + cfg.pair_budget, cfg.pair_budget)))
            .clip(0, len(rec_start) - 1)] if len(rec_start) else np.zeros(0, np.int64)
        bounds = np.unique(np.r_[cuts, len(sk)])
        bounds = np.r_[0, bounds[bounds > 0]] if len(bounds) else np.zeros(1, np.int64)
        for b0, b1 in zip(bounds[:-1], bounds[1:]):
            if b1 <= b0:
                continue
            ch = sk.iloc[b0:b1]
            starts = ph_start[ch["p"].to_numpy()]
            lens = ph_len[ch["p"].to_numpy()]
            total = int(lens.sum())
            if total == 0:
                continue
            rep = np.repeat(np.arange(len(ch)), lens)
            offs = np.arange(total) - np.repeat(np.cumsum(lens) - lens, lens)
            tt = post_t[np.repeat(starts, lens) + offs]
            ss = ch["i"].to_numpy()[rep]
            w = ch["w"].to_numpy()[rep]
            g = ch["g"].to_numpy()[rep]
            pair = ss.astype(np.int64) * (n_t + 1) + tt
            o = np.argsort(pair, kind="quicksort")
            pair_s = pair[o]
            brk = np.flatnonzero(np.r_[True, pair_s[1:] != pair_s[:-1]])
            up = pair_s[brk]
            wo, go = w[o], g[o]
            score = np.add.reduceat(wo, brk)
            sname = np.add.reduceat(wo * (go == 0), brk)
            saddr = score - sname
            nkeys = np.diff(np.r_[brk, len(pair_s)])
            del o, pair_s, wo, go, pair, ss, tt, w, g, rep, offs
            ps = up // (n_t + 1)
            pt = up % (n_t + 1)
            # forward, adaptive K: rank < k_min OR (rank < k_max AND score >= ratio*top1)
            rank, top1 = _rank_top1(ps, score)
            fwd = (rank < cfg.k_min) | ((rank < cfg.top_k) & (score >= cfg.adapt_ratio * top1))
            m = (fwd | (_rank_top1(ps, sname)[0] < cfg.top_k_name)
                 | (_rank_top1(ps, saddr)[0] < cfg.top_k_addr))
            # reverse direction (target -> S1): chunk-local top-R is a superset
            # of the global top-R, merged and re-pruned below
            rev = (_rank_top1(pt, score)[0] < cfg.reverse_k) if cfg.reverse_k else np.zeros(len(ps), bool)
            keep = m | rev
            frame = pd.DataFrame({
                "s_row": s_rows[ps[keep]].astype(np.int32), "t_row": t_rows[pt[keep]].astype(np.int32),
                "blk_score": score[keep].astype(np.float32),
                "blk_name": sname[keep].astype(np.float32),
                "blk_addr": saddr[keep].astype(np.float32),
                "blk_nkeys": nkeys[keep].astype(np.int16),
            })
            out.append(frame[m[keep]])
            if cfg.reverse_k:
                rev_pool.append(frame[rev[keep]])
                if sum(len(r) for r in rev_pool) > 4 * cfg.reverse_k * n_t:
                    rev_pool = [_prune_reverse(pd.concat(rev_pool, ignore_index=True), cfg.reverse_k)]
        if cfg.reverse_k and rev_pool:
            out.append(_prune_reverse(pd.concat(rev_pool, ignore_index=True), cfg.reverse_k))
        rev_pool = []
        del post_t, ph_uniq, ph_start, ph_len, sk
        free()
    if not out:
        return pd.DataFrame(columns=["s_row", "t_row", "blk_score", "blk_name", "blk_addr", "blk_nkeys"])
    res = pd.concat(out, ignore_index=True).drop_duplicates(["s_row", "t_row"])
    return res.reset_index(drop=True)


def _rank_top1(g, v):
    """Rank of each row within its group g by descending v, and the group max.
    One float argsort on a composite key (much faster than lexsort)."""
    n = len(v)
    if n == 0:
        return np.zeros(0, np.int64), np.zeros(0)
    v = np.asarray(v, np.float64)
    vmax = float(v.max()) + 1.0
    scale = 2.0 ** np.ceil(np.log2(vmax + 1.0))
    comp = g.astype(np.float64) * scale + (vmax - v)
    o = np.argsort(comp, kind="quicksort")
    gs = g[o]
    first = np.r_[True, gs[1:] != gs[:-1]]
    start = np.maximum.accumulate(np.where(first, np.arange(n), 0))
    rank = np.empty(n, np.int64)
    rank[o] = np.arange(n) - start
    top = np.empty(n)
    top[o] = v[o][start]
    return rank, top


def _prune_reverse(df, k):
    """Keep each target's top-k S1 records by blocking score."""
    df = df.sort_values(["t_row", "blk_score"], ascending=[True, False])
    return df[df.groupby("t_row").cumcount().to_numpy() < k]
