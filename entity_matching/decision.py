"""
Decision layer: calibration + per-S1 expected-F0.5 set selection.

The official metric (per public write-ups) is F0.5 computed per Source-1
entity and averaged (macro), with an empty prediction for an S1 without
true matches scoring 1.  For one S1 with calibrated, one-owner-resolved
candidate probabilities q1 >= q2 >= ..., predicting the top-k set has

    E[F_b | top-k] ~= (1+b^2) * sum_{i<=k} q_i / (b^2 * E|T| + k)
    E[F_b | empty] =  P(T empty) ~= prod_i (1 - q_i) * exp(-lam)

with E|T| = sum_i q_i + lam, lam = expected true links missed by blocking.
We pick the k (0 included) with the largest value - the plug-in version of
the General F-measure Maximizer (Dembczynski et al., NIPS 2011; Waegeman
et al., JMLR 2014).  `empty_bias` scales E[F|empty] and is tuned on OOF.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression


class Calibrator:
    """Isotonic calibration stored as monotone knots (np.interp at predict)."""

    def __init__(self, x=None, y=None):
        self.x, self.y = x, y

    def fit(self, p, y):
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(np.asarray(p, float), np.asarray(y, float))
        self.x, self.y = iso.X_thresholds_, iso.y_thresholds_
        return self

    def __call__(self, p):
        return np.interp(np.asarray(p, float), self.x, self.y).astype(np.float32)

    def to_dict(self):
        return {"x": np.asarray(self.x).tolist(), "y": np.asarray(self.y).tolist()}

    @staticmethod
    def from_dict(d):
        return Calibrator(np.asarray(d["x"]), np.asarray(d["y"]))


def one_owner(t_row, q):
    """Mask keeping, for every target, only its most probable S1."""
    order = np.lexsort((-q, t_row))
    keep = np.zeros(len(q), bool)
    ts = t_row[order]
    keep[order[np.r_[True, ts[1:] != ts[:-1]]]] = True
    return keep


def expected_f_select(s_row, t_row, q, beta=0.5, lam=0.0, empty_bias=1.0,
                      min_q=0.0, n_s=None) -> np.ndarray:
    """Boolean mask of selected pairs maximising per-S1 expected F_beta."""
    b2 = beta * beta
    n = len(q)
    owned = one_owner(t_row, q)
    # expected truth size uses ALL candidates of the S1 (owned or not)
    exp_t = pd.Series(q).groupby(s_row).transform("sum").to_numpy() + lam
    log_empty = pd.Series(np.log1p(-np.clip(q, 0, 1 - 1e-7))).groupby(s_row).transform("sum").to_numpy()
    p_empty = np.exp(log_empty - lam) * empty_bias

    idx = np.flatnonzero(owned & (q > min_q))
    sel = np.zeros(n, bool)
    if not len(idx):
        return sel
    d = pd.DataFrame({"s": s_row[idx], "q": q[idx], "i": idx,
                      "et": exp_t[idx], "pe": p_empty[idx]}).sort_values(["s", "q"], ascending=[True, False])
    d["k"] = d.groupby("s").cumcount() + 1
    d["cum"] = d.groupby("s")["q"].cumsum()
    d["ef"] = (1 + b2) * d["cum"] / (b2 * d["et"] + d["k"])
    best = d.groupby("s")["ef"].transform("max")
    kbest = d["k"].where(d["ef"] == best).groupby(d["s"]).transform("min")
    take = (d["k"] <= kbest) & (best > d["pe"])
    sel[d["i"].to_numpy()[take.to_numpy()]] = True
    return sel


def threshold_select(s_row, t_row, q, thr, alpha=0.0):
    """Baseline rule: q >= thr, one owner per target, q >= alpha * max q of S1."""
    mask = (q >= thr) & one_owner(t_row, q)
    if alpha > 0 and mask.any():
        idx = np.flatnonzero(mask)
        mx = pd.Series(q[idx]).groupby(s_row[idx]).transform("max").to_numpy()
        mask[idx[q[idx] < alpha * mx]] = False
    return mask
