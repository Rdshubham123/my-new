"""F-beta evaluation (default beta = 0.5: precision weighted 2x recall)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def fbeta(p: float, r: float, beta: float = 0.5) -> float:
    b2 = beta * beta
    return (1 + b2) * p * r / (b2 * p + r) if (p + r) > 0 else 0.0


def evaluate(pred: pd.DataFrame, truth: pd.DataFrame, s1_ids, beta: float = 0.5) -> dict:
    """pred / truth: DataFrames[source1_entity_id, matched_id].
    Only S1 ids in `s1_ids` are scored.  Returns micro (pair-level) and macro
    (per-S1 averaged, empty-vs-empty = 1) scores."""
    s1_ids = pd.Index(pd.unique(np.asarray(s1_ids)))
    pred = pred[pred["source1_entity_id"].isin(s1_ids)]
    truth = truth[truth["source1_entity_id"].isin(s1_ids)]
    pk = set(zip(pred["source1_entity_id"], pred["matched_id"]))
    tk = set(zip(truth["source1_entity_id"], truth["matched_id"]))
    tp = len(pk & tk)
    p = tp / len(pk) if pk else 1.0
    r = tp / len(tk) if tk else 1.0

    # per-S1
    tp_s = pd.Series([a for a, _ in pk & tk], dtype=object).value_counts()
    np_s = pred["source1_entity_id"].value_counts()
    nt_s = truth["source1_entity_id"].value_counts()
    tp_v = tp_s.reindex(s1_ids).fillna(0).to_numpy()
    np_v = np_s.reindex(s1_ids).fillna(0).to_numpy()
    nt_v = nt_s.reindex(s1_ids).fillna(0).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        ps = np.where(np_v > 0, tp_v / np_v, np.where(nt_v > 0, 0.0, 1.0))
        rs = np.where(nt_v > 0, tp_v / nt_v, np.where(np_v > 0, 0.0, 1.0))
        b2 = beta * beta
        fs = np.where((ps + rs) > 0, (1 + b2) * ps * rs / (b2 * ps + rs), 0.0)
    both_empty = (np_v == 0) & (nt_v == 0)
    fs = np.where(both_empty, 1.0, fs)
    return {
        "micro_precision": p, "micro_recall": r, "micro_fbeta": fbeta(p, r, beta),
        "macro_precision": float(ps.mean()), "macro_recall": float(rs.mean()),
        "macro_fbeta": float(fs.mean()), "tp": tp, "n_pred": len(pk), "n_true": len(tk),
    }
