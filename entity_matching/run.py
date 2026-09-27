"""
End-to-end pipeline (Kaggle-ready).

    python -m entity_matching.run kaggle                 # train + predict test, auto paths
    python -m entity_matching.run train   [--data-dir D] [--out-dir O]
    python -m entity_matching.run predict [--data-dir D] [--out-dir O]
    python -m entity_matching.run synth   --out-dir SYNTH_DATA

`--data-dir` is optional: the dataset is auto-detected under /kaggle/input
(any slug / nesting) or local folders.  Outputs go to /kaggle/working/em_model
on Kaggle (./em_output elsewhere); the submission is also copied to
/kaggle/working/submission.tsv.

`train` produces out-of-fold (GroupKFold by S1) predictions, tunes the
decision rule for F-beta (beta=0.5 by default), reports honest validation
scores on ALL held-out S1 records (singletons and doubtful pairs included,
true links missed by blocking counted as false negatives), and saves the
final model.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from .blocking import BlockConfig, generate_candidates
from .data import gt_pairs, load_normalized
from .features import (ALL_FEATURES, add_s1_context, add_target_context,
                       build_idf, compute_pair_features)
from .metrics import evaluate, fbeta
from .utils import (auto_data_dir, default_out_dir, free, is_kaggle, log,
                    n_workers)

LGB_PARAMS = dict(
    objective="binary", learning_rate=0.05, num_leaves=127, min_data_in_leaf=40,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
    max_bin=255, verbose=-1, seed=42, force_col_wise=True,
)
DOUBT_COLS = ["n_tset_skel", "n_idf_contain", "a_tset", "num_shared", "n_alt_best",
              "n_exact_skel", "a_missing_t", "num_jacc", "num_s"]


# ---------------------------------------------------------------------------
# Shared steps
# ---------------------------------------------------------------------------
def build_pairs(s1, tgt, cfg, n_jobs, keep_s_rows=None, predict_fn=None,
                batch_s1=100_000, extra_fn=None):
    """Blocking over ALL S1 records (so target-competition features see every
    competitor, as at test time), then features in S1 batches.

    keep_s_rows : only these S1 rows get features (training sample)
    predict_fn  : part -> probabilities; then only (s_row, t_row, p) is kept
    extra_fn    : part -> dict of extra arrays to keep (used for training)"""
    log("blocking ...")
    cands = generate_candidates(s1, tgt, cfg, log=log)
    log(f"  candidates: {len(cands):,} ({len(cands) / max(len(s1), 1):.1f} per S1)")
    cands = add_target_context(cands)
    if keep_s_rows is not None:
        cands = cands[np.isin(cands["s_row"].to_numpy(), keep_s_rows)]
    log("building IDF tables from targets ...")
    idf = build_idf(tgt)
    log("pair features ...")
    cands = cands.sort_values("s_row", kind="stable").reset_index(drop=True)
    free()
    srow = cands["s_row"].to_numpy()
    edges = np.r_[np.searchsorted(srow, np.unique(srow)[::batch_s1]), len(srow)]
    parts = []
    for b, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        part = compute_pair_features(cands.iloc[lo:hi], s1, tgt, idf, n_jobs=n_jobs)
        part = add_s1_context(part, tgt)
        if predict_fn is not None:
            keep = {"s_row": part["s_row"].to_numpy(), "t_row": part["t_row"].to_numpy(),
                    "p": predict_fn(part).astype(np.float32)}
        else:
            keep = {"s_row": part["s_row"].to_numpy(), "t_row": part["t_row"].to_numpy(),
                    "X": part[ALL_FEATURES].to_numpy(np.float32)}
            keep.update(extra_fn(part) if extra_fn else {})
        parts.append(keep)
        del part
        free()
        log(f"  batch {b + 1}/{len(edges) - 1}: {hi - lo:,} pairs")
    del cands
    if not parts:
        return {"s_row": np.zeros(0, np.int32), "t_row": np.zeros(0, np.int32),
                "p": np.zeros(0, np.float32), "X": np.zeros((0, len(ALL_FEATURES)), np.float32)}
    out = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    del parts
    free()
    return out


def select_mask(s_row, t_row, p, thr, alpha, one_owner=True) -> np.ndarray:
    """Decision rule: p >= thr; each target kept only for its best S1 (the
    one-owner constraint verified in EDA cell 8); p >= alpha * best p of the
    S1.  Returns a boolean mask over the pairs."""
    mask = p >= thr
    idx = np.flatnonzero(mask)
    if one_owner and len(idx):
        d = pd.DataFrame({"t": t_row[idx], "p": p[idx], "i": idx}).sort_values(
            ["t", "p"], ascending=[True, False])
        best = d.drop_duplicates("t")["i"].to_numpy()
        mask = np.zeros(len(p), bool)
        mask[best] = True
        idx = best
    if alpha > 0 and len(idx):
        mx = pd.Series(p[idx]).groupby(s_row[idx]).transform("max").to_numpy()
        mask[idx[p[idx] < alpha * mx]] = False
    return mask


def fast_scores(s_row, y, mask, n_true_per_s, beta):
    """Micro and macro F-beta from aligned arrays (for the tuning grid).
    n_true_per_s[s] = number of true links of S1 row s (incl. those missed
    by blocking)."""
    n = len(n_true_per_s)
    tp_s = np.bincount(s_row[mask], weights=y[mask], minlength=n)
    np_s = np.bincount(s_row[mask], minlength=n).astype(float)
    tp, n_pred, n_true = tp_s.sum(), np_s.sum(), n_true_per_s.sum()
    p = tp / n_pred if n_pred else 1.0
    r = tp / n_true if n_true else 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        ps = np.where(np_s > 0, tp_s / np_s, np.where(n_true_per_s > 0, 0.0, 1.0))
        rs = np.where(n_true_per_s > 0, tp_s / n_true_per_s, np.where(np_s > 0, 0.0, 1.0))
        b2 = beta * beta
        fs = np.where((ps + rs) > 0, (1 + b2) * ps * rs / (b2 * ps + rs), 0.0)
    return {"micro_fbeta": fbeta(p, r, beta), "macro_fbeta": float(fs.mean())}


def doubtful_mask(d: dict, y: np.ndarray) -> np.ndarray:
    """Pairs whose label cannot be learnt from the evidence available.
    Removed ONLY from training folds - validation always scores everything.
      weak positive: true link with no name, address or number evidence
      twin negative: non-link indistinguishable from a link (same phonetic
                     name + same address) -> label noise / true duplicate"""
    weak_pos = ((y == 1) & (d["n_tset_skel"] < 35) & (d["n_idf_contain"] < 0.2)
                & (d["a_tset"] < 40) & (d["num_shared"] == 0) & (d["n_alt_best"] < 60))
    twin_neg = ((y == 0) & (d["n_exact_skel"] == 1) & (d["a_missing_t"] == 0)
                & ((d["a_tset"] >= 97) | ((d["num_jacc"] >= 0.999) & (d["num_s"] > 0))))
    return weak_pos | twin_neg


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------
def cmd_train(a):
    import lightgbm as lgb
    from sklearn.model_selection import GroupKFold

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    n_jobs = n_workers(a.n_jobs)
    cfg = BlockConfig(top_k=a.top_k, top_k_name=a.top_k_name, top_k_addr=a.top_k_addr, n_jobs=n_jobs)
    data_dir = auto_data_dir(a.data_dir, "train")
    log(f"data dir: {data_dir} | out dir: {out} | workers: {n_jobs}")

    s1, tgt, gt = load_normalized(data_dir, "train", None if a.no_cache else out / "cache", n_jobs)
    truth = gt_pairs(gt)
    del gt
    rng = np.random.default_rng(42)
    if a.train_s1_sample and a.train_s1_sample < len(s1):
        keep_rows = np.sort(rng.choice(len(s1), a.train_s1_sample, replace=False)).astype(np.int32)
    else:
        keep_rows = np.arange(len(s1), dtype=np.int32)
    s1_ids = s1["entity_id"].to_numpy()
    sub_ids = s1_ids[keep_rows]
    log(f"S1 used for training/validation: {len(keep_rows):,} / {len(s1):,}")

    d = build_pairs(s1, tgt, cfg, n_jobs, keep_s_rows=keep_rows,
                    extra_fn=lambda part: {c: part[c].to_numpy(np.float32) for c in DOUBT_COLS})
    # labels
    t_ids = tgt["entity_id"].to_numpy()
    truth_sub = truth[truth["source1_entity_id"].isin(pd.Index(sub_ids))]
    pair_key = pd.MultiIndex.from_arrays([s1_ids[d["s_row"]], t_ids[d["t_row"]]])
    y = pair_key.isin(pd.MultiIndex.from_frame(truth_sub)).astype(np.int8)
    del pair_key
    recall_ceiling = y.sum() / max(len(truth_sub), 1)
    log(f"blocking recall (ceiling): {recall_ceiling:.4%} "
        f"({int(y.sum()):,} / {len(truth_sub):,} true links in candidates)")

    n_true_per_s = (truth_sub["source1_entity_id"].map(pd.Series(np.arange(len(s1)), index=s1_ids))
                    .value_counts().reindex(np.arange(len(s1)), fill_value=0).to_numpy().astype(float))
    eval_rows = np.zeros(len(s1), bool)
    eval_rows[keep_rows] = True

    # training-set cleaning (validation always sees everything)
    is_singleton = n_true_per_s[d["s_row"]] == 0
    doubt = doubtful_mask(d, y)
    train_ok = ~doubt
    if not a.keep_singletons:
        train_ok &= ~is_singleton
    log(f"train cleaning: doubtful pairs={int(doubt.sum()):,} "
        f"(pos {int((doubt & (y == 1)).sum()):,} / neg {int((doubt & (y == 0)).sum()):,}); "
        f"singleton-S1 pairs dropped={0 if a.keep_singletons else int(is_singleton.sum()):,}")
    for c in DOUBT_COLS:
        d.pop(c)
    free()

    X, s_row, t_row = d["X"], d["s_row"], d["t_row"]
    params = dict(LGB_PARAMS, num_threads=n_jobs)
    full = lgb.Dataset(X, y, feature_name=ALL_FEATURES, free_raw_data=False,
                       params={"max_bin": 255}).construct()
    oof = np.zeros(len(y), np.float32)
    best_iters = []
    for fold, (tr, va) in enumerate(GroupKFold(n_splits=a.folds).split(s_row, y, s_row)):
        tr = tr[train_ok[tr]]
        bst = lgb.train(params, full.subset(np.sort(tr)), num_boost_round=a.rounds,
                        valid_sets=[full.subset(np.sort(va))],
                        callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[va] = bst.predict(X[va], num_iteration=bst.best_iteration, num_threads=n_jobs)
        best_iters.append(bst.best_iteration or a.rounds)
        log(f"  fold {fold}: best_iter={best_iters[-1]}")
        del bst
        free()

    metric = f"{a.metric}_fbeta"
    ntps = np.where(eval_rows, n_true_per_s, 0.0)  # score only the sampled S1s
    best = None
    for thr in np.round(np.arange(0.10, 0.96, 0.025), 3):
        for alpha in (0.0, 0.2, 0.4, 0.6, 0.8):
            m = fast_scores(s_row, y, select_mask(s_row, t_row, oof, thr, alpha), ntps, a.beta)
            if best is None or m[metric] > best[2][metric]:
                best = (float(thr), float(alpha), m)
    thr, alpha, _ = best
    sel = select_mask(s_row, t_row, oof, thr, alpha)
    pred = pd.DataFrame({"source1_entity_id": s1_ids[s_row[sel]], "matched_id": t_ids[t_row[sel]]})
    m = evaluate(pred, truth_sub, sub_ids, a.beta)
    log(f"OOF best rule: thr={thr} alpha={alpha}")
    for k, v in m.items():
        log(f"  {k:16s} {v:.5f}" if isinstance(v, float) else f"  {k:16s} {v:,}")

    n_rounds = int(np.mean(best_iters) * 1.1) + 1
    log(f"final model on all clean rows ({int(train_ok.sum()):,}), rounds={n_rounds}")
    bst = lgb.train(params, full.subset(np.flatnonzero(train_ok)), num_boost_round=n_rounds)
    bst.save_model(str(out / "model.txt"))
    imp = pd.Series(bst.feature_importance("gain"), index=ALL_FEATURES).sort_values(ascending=False)
    imp.to_csv(out / "feature_importance.csv", header=["gain"])
    report = {"threshold": thr, "alpha": alpha, "beta": a.beta, "metric": metric,
              "oof": m, "blocking_recall": float(recall_ceiling),
              "block_cfg": {"top_k": a.top_k, "top_k_name": a.top_k_name, "top_k_addr": a.top_k_addr},
              "n_s1": int(len(keep_rows)), "n_pairs": int(len(y)),
              "n_doubtful": int(doubt.sum()), "features": ALL_FEATURES}
    (out / "config.json").write_text(json.dumps(report, indent=2, default=float))
    log(f"saved model + config to {out}")
    return report


# ---------------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------------
def cmd_predict(a):
    import lightgbm as lgb

    out = Path(a.out_dir)
    n_jobs = n_workers(a.n_jobs)
    conf = json.loads((out / "config.json").read_text())
    bc = conf["block_cfg"]
    cfg = BlockConfig(top_k=bc["top_k"], top_k_name=bc["top_k_name"],
                      top_k_addr=bc["top_k_addr"], n_jobs=n_jobs)
    bst = lgb.Booster(model_file=str(out / "model.txt"))
    data_dir = auto_data_dir(a.data_dir, a.split)
    log(f"data dir: {data_dir} | split: {a.split}")

    s1, tgt, gt = load_normalized(data_dir, a.split, None if a.no_cache else out / "cache", n_jobs)
    feats = conf["features"]
    d = build_pairs(s1, tgt, cfg, n_jobs, predict_fn=lambda part: bst.predict(
        part[feats].to_numpy(np.float32), num_threads=n_jobs))
    sel = select_mask(d["s_row"], d["t_row"], d["p"], conf["threshold"], conf["alpha"])
    pairs = pd.DataFrame({"source1_entity_id": s1["entity_id"].to_numpy()[d["s_row"][sel]],
                          "matched_id": tgt["entity_id"].to_numpy()[d["t_row"][sel]]})

    agg = pairs.groupby("source1_entity_id")["matched_id"].agg(",".join)
    sub = pd.DataFrame({"source1_entity_id": s1["entity_id"]})
    sub["matched_entity_ids"] = sub["source1_entity_id"].map(agg).fillna("")
    path = Path(a.submission or out / f"submission_{a.split}.tsv")
    sub.to_csv(path, sep="\t", index=False)
    log(f"wrote {path} ({len(pairs):,} links for {len(sub):,} S1 records)")
    if is_kaggle() and a.split == "test":
        shutil.copy(path, "/kaggle/working/submission.tsv")
        log("copied to /kaggle/working/submission.tsv")
    if gt is not None:
        m = evaluate(pairs, gt_pairs(gt), s1["entity_id"], conf["beta"])
        log("score on this split: " + json.dumps({k: round(v, 5) if isinstance(v, float) else v
                                                  for k, v in m.items()}))
        return m
    return None


def cmd_kaggle(a):
    cmd_train(a)
    free()
    a.split = "test"
    a.submission = None
    return cmd_predict(a)


def cmd_synth(a):
    from .synth import write_synthetic
    write_synthetic(a.out_dir, n_train=a.n_train, n_test=a.n_test, seed=a.seed)


def _add_train_args(t):
    t.add_argument("--train-s1-sample", type=int, default=250_000,
                   help="S1 records used for training/validation (0 = all). "
                        "250k fits the 30 GB Kaggle RAM limit comfortably")
    t.add_argument("--folds", type=int, default=5)
    t.add_argument("--rounds", type=int, default=3000)
    t.add_argument("--beta", type=float, default=0.5)
    t.add_argument("--metric", choices=["micro", "macro"], default="micro")
    t.add_argument("--keep-singletons", action="store_true",
                   help="keep S1 records without any true match in the TRAINING folds")
    t.add_argument("--top-k", type=int, default=40)
    t.add_argument("--top-k-name", type=int, default=10)
    t.add_argument("--top-k-addr", type=int, default=10)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="entity_matching")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("train", "predict", "kaggle"):
        p = sub.add_parser(name)
        p.add_argument("--data-dir", default=None, help="auto-detected when omitted")
        p.add_argument("--out-dir", default=str(default_out_dir()))
        p.add_argument("--n-jobs", type=int, default=None, help="default: all available cores")
        p.add_argument("--no-cache", action="store_true",
                       help="do not cache normalised data in OUT/cache")
    _add_train_args(sub.choices["train"])
    _add_train_args(sub.choices["kaggle"])
    for name in ("predict", "kaggle"):
        sub.choices[name].add_argument("--split", default="test")
        sub.choices[name].add_argument("--submission", default=None)
    s = sub.add_parser("synth")
    s.add_argument("--out-dir", required=True)
    s.add_argument("--n-train", type=int, default=6000)
    s.add_argument("--n-test", type=int, default=3000)
    s.add_argument("--seed", type=int, default=0)
    a, _ = ap.parse_known_args(argv)  # tolerate Jupyter's -f kernel.json
    {"train": cmd_train, "predict": cmd_predict, "kaggle": cmd_kaggle, "synth": cmd_synth}[a.cmd](a)


if __name__ == "__main__":
    main()
