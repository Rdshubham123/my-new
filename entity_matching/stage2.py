"""
Stage 2: graph / cluster features built from stage-1 probabilities.

An S1 entity is usually matched by several S2/S3 duplicates of itself
(EDA cell 7: 3.46 links per S1 on average), and every target belongs to at
most one S1 (cell 8).  Stage 2 turns those structural facts into features:

sibling   : how similar is candidate t to the most probable OTHER candidates
            of the same S1?  A weak candidate that looks like a confident
            sibling is probably a duplicate of it (label propagation over the
            candidate graph, cf. multi-source clustering / TransClean).
s-context : rank, gap, mass of stage-1 probabilities inside the S1's list
t-context : strongest competing S1 for the same target (one-owner)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from .utils import run_pool, worker_view

CARRY = ["n_tset_skel", "n_idf_contain", "n_alt_best", "n_exact_skel", "a_tset",
         "num_shared", "num_jacc", "a_missing_t", "n_core_df_t", "blk_score",
         "ctx_ncand_s", "t_indic", "t_domain", "is_s3"]
STAGE2_FEATURES = CARRY + [
    "p1", "p1_logit", "p1_rank_s", "p1_gap_s", "p1_second_s", "p1_mass_other_s",
    "p1_n50_s", "p1_other_max_t", "p1_gap_t", "p1_n10_t",
    "sib_name_max", "sib_addr_max", "sib_num_max", "sib_score", "sib_p1_best",
]
_G: dict = {}
SIB_TOP = 3        # compare each candidate with the 3 most probable others
SIB_MIN_P1 = 0.02  # candidates below this are hopeless - skip the fuzz work


def _sib_worker(b):
    lo, hi = b
    ta, tb = _G["ta"], _G["tb"]
    skel, atext, nums = _G["skel"], _G["atext"], _G["nums"]
    out = np.zeros((hi - lo, 3), np.float32)
    for r in range(lo, hi):
        i, j = ta[r], tb[r]
        out[r - lo, 0] = fuzz.token_set_ratio(skel[i], skel[j])
        out[r - lo, 1] = fuzz.token_set_ratio(atext[i], atext[j]) if atext[i] and atext[j] else 0.0
        ni, nj = set(nums[i].split()), set(nums[j].split())
        out[r - lo, 2] = len(ni & nj) / len(ni | nj) if (ni and nj) else 0.0
    return lo, out


def top2_by_group(keys, vals):
    """Per row: (max of its group, max of its group excluding this row)."""
    n = len(vals)
    order = np.lexsort((-vals, keys))
    ks, vs = keys[order], vals[order]
    first = np.r_[True, ks[1:] != ks[:-1]] if n else np.zeros(0, bool)
    start = np.maximum.accumulate(np.where(first, np.arange(n), 0))
    rank = np.arange(n) - start
    top1 = vs[start]
    same_next = np.r_[ks[1:] == ks[:-1], False]
    second_at = np.where(same_next, np.r_[vs[1:], 0.0], 0.0)[start]
    other = np.where(rank == 0, second_at, top1)
    out1, out2 = np.empty(n, np.float32), np.empty(n, np.float32)
    out1[order], out2[order] = top1, other
    return out1, out2


def _rank_desc(group_keys, values):
    return (pd.Series(values).groupby(group_keys).rank(ascending=False, method="first")
            .to_numpy(np.float32))


def build_stage2(s_row, t_row, p1, carry: dict, tgt: pd.DataFrame, n_jobs: int = 4) -> np.ndarray:
    """Returns the stage-2 feature matrix aligned with the pair arrays."""
    n = len(p1)
    p1 = np.asarray(p1, np.float32)
    F = {c: np.asarray(carry[c], np.float32) for c in CARRY}
    F["p1"] = p1
    pc = np.clip(p1, 1e-6, 1 - 1e-6)
    F["p1_logit"] = np.log(pc / (1 - pc)).astype(np.float32)

    # ---- S1 context on p1
    mx, other_s = top2_by_group(s_row, p1)
    F["p1_rank_s"] = _rank_desc(s_row, p1)
    F["p1_gap_s"] = (mx - p1).astype(np.float32)
    F["p1_second_s"] = other_s
    F["p1_mass_other_s"] = (pd.Series(p1).groupby(s_row).transform("sum").to_numpy() - p1).astype(np.float32)
    F["p1_n50_s"] = pd.Series(p1 >= 0.5).groupby(s_row).transform("sum").to_numpy(np.float32)

    # ---- target competition on p1 (one-owner)
    _, F["p1_other_max_t"] = top2_by_group(t_row, p1)
    F["p1_gap_t"] = (p1 - F["p1_other_max_t"]).astype(np.float32)
    F["p1_n10_t"] = pd.Series(p1 >= 0.1).groupby(t_row).transform("sum").to_numpy(np.float32)

    # ---- sibling similarity
    for c in ("sib_name_max", "sib_addr_max", "sib_num_max", "sib_score", "sib_p1_best"):
        F[c] = np.zeros(n, np.float32)
    top = pd.DataFrame({"s": s_row, "t": t_row, "p": p1, "rk": F["p1_rank_s"]})
    top = top[(top["rk"] <= SIB_TOP) & (top["p"] >= 0.3)][["s", "t", "p"]]
    cand = pd.DataFrame({"row": np.arange(n), "s": s_row, "t": t_row})[p1 >= SIB_MIN_P1]
    pairs = cand.merge(top, on="s", suffixes=("", "_sib"))
    pairs = pairs[pairs["t"] != pairs["t_sib"]]
    if len(pairs):
        v = worker_view(tgt, ["n_skel", "a_atext", "a_nums"])
        _G.update(ta=pairs["t"].to_numpy(), tb=pairs["t_sib"].to_numpy(),
                  skel=v["n_skel"], atext=v["a_atext"], nums=v["a_nums"])
        m = len(pairs)
        sims = np.empty((m, 3), np.float32)
        for lo, arr in run_pool(_sib_worker, [(lo, min(lo + 50000, m)) for lo in range(0, m, 50000)], n_jobs):
            sims[lo:lo + len(arr)] = arr
        _G.clear()
        pairs = pairs.assign(sn=sims[:, 0], sa=sims[:, 1], su=sims[:, 2])
        pairs["score"] = pairs["p"] * (0.5 * pairs["sn"] + 0.5 * pairs["sa"]) / 100.0
        g = pairs.groupby("row")
        agg = g.agg(sn=("sn", "max"), sa=("sa", "max"), su=("su", "max"), sc=("score", "max"))
        best = pairs.loc[g["score"].idxmax(), ["row", "p"]].set_index("row")["p"]
        rows = agg.index.to_numpy()
        F["sib_name_max"][rows] = agg["sn"].to_numpy()
        F["sib_addr_max"][rows] = agg["sa"].to_numpy()
        F["sib_num_max"][rows] = agg["su"].to_numpy()
        F["sib_score"][rows] = agg["sc"].to_numpy()
        F["sib_p1_best"][best.index.to_numpy()] = best.to_numpy()
    return np.column_stack([F[c] for c in STAGE2_FEATURES]).astype(np.float32)
