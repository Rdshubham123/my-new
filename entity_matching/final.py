"""
FINAL competition pipeline (v3): learned two-stage matcher + GPU ensemble.

Phase A  train  (CPU, fork pools)  normalise -> block -> features ->
                                    stage-1 LightGBM (GroupKFold OOF) ->
                                    learned pruning rule chosen on OOF
Phase B  test   (CPU, fork pools)  normalise -> block -> features ->
                                    stage-1 score -> prune -> candidate_pairs.tsv
Phase C  models (GPU allowed)      LightGBM + XGBoost + CatBoost on the pruned
                                    pairs (OOF), blend chosen on OOF, isotonic
                                    calibration, macro-F0.5 decision rule
                                    -> matching_results.tsv

Deadlock safety: every fork happens in phases A/B.  CUDA (XGBoost/CatBoost
GPU) is first touched in phase C, after which nothing forks again.  Forking
a process that already holds a CUDA context is the classic hang.

Metric (problem statement): F0.5 per Source-1 entity, macro-averaged over
ALL S1 entities; a singleton predicted empty scores 1.0.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .blocking import BlockConfig, generate_candidates
from .data import gt_pairs, load_split, normalize_frame
from .decision import Calibrator, expected_f_select, threshold_select
from .features import ALL_FEATURES, add_s1_context, build_idf, compute_pair_features
from .utils import auto_data_dir, free, is_kaggle, log, n_workers

# blocking-based target-competition features need blocking over ALL S1
# records; training blocks only the sampled S1s, so they are left out
# (one-owner is still enforced by the decision rule)
FEATS = [f for f in ALL_FEATURES if f not in ("ctx_ncand_t", "ctx_rank_t", "ctx_gap_t")]
DOUBT_COLS = ["n_tset_skel", "n_idf_contain", "a_tset", "num_shared", "n_alt_best",
              "n_exact_skel", "a_missing_t", "num_jacc", "num_s"]


@dataclass
class FinalConfig:
    data_dir: str | None = None
    out_dir: str | None = None
    train_s1_sample: int = 250_000     # S1 records used for training (0 = all)
    folds: int = 3
    beta: float = 0.5
    n_jobs: int | None = None
    prune_keep: float = 0.998          # pruning keeps >= 99.8 % of reachable true links
    gpu: str = "auto"                  # auto | on | off
    use_xgboost: bool = True
    use_catboost: bool = True
    batch_s1: int = 100_000
    seed: int = 42
    team_name: str = "team"
    make_zip: bool = True


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def gpu_available() -> bool:
    try:
        r = subprocess.run(["nvidia-smi", "-L"], capture_output=True, timeout=20)
        return r.returncode == 0 and b"GPU" in r.stdout
    except Exception:
        return False


def rank_in_s(s_row, v):
    order = np.lexsort((-v, s_row))
    ss = s_row[order]
    first = np.r_[True, ss[1:] != ss[:-1]] if len(ss) else np.zeros(0, bool)
    start = np.maximum.accumulate(np.where(first, np.arange(len(ss)), 0))
    r = np.empty(len(v), np.int64)
    r[order] = np.arange(len(ss)) - start
    return r


def macro_micro(s_row, y, sel, n_true, beta=0.5):
    """F-beta per S1 averaged over ALL S1 in the universe (len(n_true)).
    n_true[s] counts every true link of S1 s, including ones blocking or
    pruning lost (they are false negatives)."""
    n = len(n_true)
    tp = np.bincount(s_row[sel], weights=y[sel].astype(float), minlength=n)
    npred = np.bincount(s_row[sel], minlength=n).astype(float)
    b2 = beta * beta
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(npred > 0, tp / npred, np.where(n_true > 0, 0.0, 1.0))
        r = np.where(n_true > 0, tp / n_true, np.where(npred > 0, 0.0, 1.0))
        f = np.where((p + r) > 0, (1 + b2) * p * r / (b2 * p + r), 0.0)
    f = np.where((npred == 0) & (n_true == 0), 1.0, f)
    P = tp.sum() / npred.sum() if npred.sum() else 1.0
    R = tp.sum() / n_true.sum() if n_true.sum() else 1.0
    micro = (1 + b2) * P * R / (b2 * P + R) if (P + R) else 0.0
    return {"macro_f05": float(f.mean()), "macro_p": float(p.mean()), "macro_r": float(r.mean()),
            "micro_f05": float(micro), "micro_p": float(P), "micro_r": float(R)}


def apply_rule(rule, s_row, t_row, q, beta):
    if rule["rule"] == "threshold":
        return threshold_select(s_row, t_row, q, rule["thr"], rule["alpha"])
    return expected_f_select(s_row, t_row, q, beta, rule["lam"], rule["empty_bias"], rule["min_q"])


def tune_rule(s_row, t_row, y, q, n_true, beta, lam_hat):
    best = None
    grid = [{"rule": "threshold", "thr": float(t), "alpha": a}
            for t in np.round(np.arange(0.15, 0.96, 0.025), 3) for a in (0.0, 0.3, 0.6)]
    grid += [{"rule": "expected_f", "lam": lam, "empty_bias": eb, "min_q": mq}
             for lam in (0.0, lam_hat) for eb in (0.5, 0.75, 1.0, 1.5, 2.0, 3.0) for mq in (0.0, 0.05)]
    for rule in grid:
        m = macro_micro(s_row, y, apply_rule(rule, s_row, t_row, q, beta), n_true, beta)
        if best is None or m["macro_f05"] > best[1]["macro_f05"]:
            best = (rule, m)
    return best


def featurize(s1, tgt, s_rows, bcfg, n_jobs, idf, batch_s1, scorer=None):
    """Blocking + features in S1 batches.
    scorer(Xpart, s_row_part) -> keep-mask and p1; used at test time to prune
    each batch right away so only the pruned pairs are kept in memory."""
    t0 = time.time()
    cands = generate_candidates(s1, tgt, bcfg, s1_rows=s_rows, log=log)
    n_block = len(cands)
    log(f"  blocking: {n_block:,} pairs ({n_block / max(len(s_rows), 1):.1f} per S1) "
        f"in {time.time() - t0:.0f}s")
    cands = cands.sort_values("s_row", kind="stable").reset_index(drop=True)
    srow = cands["s_row"].to_numpy()
    edges = np.r_[np.searchsorted(srow, np.unique(srow)[::batch_s1]), len(srow)]
    out = {"s": [], "t": [], "X": [], "p1": [], "doubt": []}
    for b, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        part = compute_pair_features(cands.iloc[lo:hi], s1, tgt, idf, n_jobs=n_jobs)
        part = add_s1_context(part, tgt)
        X = part[FEATS].to_numpy(np.float32)
        s_p, t_p = part["s_row"].to_numpy(np.int32), part["t_row"].to_numpy(np.int32)
        if scorer is not None:
            keep, p1 = scorer(X, s_p)
            out["p1"].append(p1[keep])
            X, s_p, t_p = X[keep], s_p[keep], t_p[keep]
        else:
            out["doubt"].append(part[DOUBT_COLS].to_numpy(np.float32))
        out["s"].append(s_p)
        out["t"].append(t_p)
        out["X"].append(X)
        del part
        free()
        log(f"  features batch {b + 1}/{len(edges) - 1}: {hi - lo:,} pairs -> kept {len(s_p):,}")
    del cands
    res = {k: (np.concatenate(v) if v else None) for k, v in out.items()}
    if res["X"] is None:
        res = {"s": np.zeros(0, np.int32), "t": np.zeros(0, np.int32),
               "X": np.zeros((0, len(FEATS)), np.float32), "p1": np.zeros(0, np.float32),
               "doubt": np.zeros((0, len(DOUBT_COLS)), np.float32)}
    res["n_block"] = n_block
    return res


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------
def lgb_oof(X, y, folds, ok, params, rounds, tag, n_jobs):
    import lightgbm as lgb
    full = lgb.Dataset(X, y, free_raw_data=False, params={"max_bin": 255}).construct()
    oof = np.zeros(len(y), np.float32)
    its = []
    for k, (tr, va) in enumerate(folds):
        tr = np.sort(tr[ok[tr]])
        bst = lgb.train(params, full.subset(tr), num_boost_round=rounds,
                        valid_sets=[full.subset(np.sort(va))],
                        callbacks=[lgb.early_stopping(50, verbose=False)])
        oof[va] = bst.predict(X[va], num_iteration=bst.best_iteration, num_threads=n_jobs)
        its.append(bst.best_iteration or rounds)
        log(f"    [{tag}] fold {k}: {its[-1]} iters")
        del bst
        free()
    final = lgb.train(params, full.subset(np.flatnonzero(ok)), num_boost_round=int(np.mean(its) * 1.1) + 1)
    del full
    free()
    return oof, final


def xgb_oof(X, y, folds, ok, Xtest, gpu, n_jobs, seed):
    import xgboost as xgb
    params = {"objective": "binary:logistic", "eval_metric": "logloss", "eta": 0.08,
              "max_depth": 8, "min_child_weight": 5, "subsample": 0.8,
              "colsample_bytree": 0.8, "lambda": 1.0, "tree_method": "hist",
              "max_bin": 256, "seed": seed, "nthread": n_jobs}
    params["device"] = "cuda" if gpu else "cpu"
    oof = np.zeros(len(y), np.float32)
    its = []
    for k, (tr, va) in enumerate(folds):
        tr = tr[ok[tr]]
        dtr = xgb.DMatrix(X[tr], label=y[tr])
        dva = xgb.DMatrix(X[va], label=y[va])
        bst = xgb.train(params, dtr, 3000, evals=[(dva, "va")], early_stopping_rounds=50,
                        verbose_eval=False)
        oof[va] = bst.predict(dva, iteration_range=(0, bst.best_iteration + 1))
        its.append(bst.best_iteration + 1)
        log(f"    [xgb{'-gpu' if gpu else ''}] fold {k}: {its[-1]} iters")
        del dtr, dva, bst
        free()
    dall = xgb.DMatrix(X[ok], label=y[ok])
    bst = xgb.train(params, dall, int(np.mean(its) * 1.1) + 1, verbose_eval=False)
    pred = bst.predict(xgb.DMatrix(Xtest))
    del dall, bst
    free()
    return oof, pred.astype(np.float32)


def cat_oof(X, y, folds, ok, Xtest, gpu, n_jobs, seed):
    from catboost import CatBoostClassifier
    kw = dict(loss_function="Logloss", learning_rate=0.1 if gpu else 0.15,
              depth=8 if gpu else 6, iterations=3000 if gpu else 500,
              random_seed=seed, verbose=0, border_count=128,
              od_type="Iter", od_wait=50, allow_writing_files=False)
    if gpu:
        kw.update(task_type="GPU", devices="0")
    else:
        kw.update(thread_count=n_jobs)
    oof = np.zeros(len(y), np.float32)
    its = []
    for k, (tr, va) in enumerate(folds):
        tr = tr[ok[tr]]
        m = CatBoostClassifier(**kw)
        m.fit(X[tr], y[tr], eval_set=(X[va], y[va]), use_best_model=True)
        oof[va] = m.predict_proba(X[va])[:, 1]
        its.append(max(m.get_best_iteration() or 0, 1) + 1)
        log(f"    [cat{'-gpu' if gpu else ''}] fold {k}: {its[-1]} iters")
        del m
        free()
    kw.pop("od_type"), kw.pop("od_wait")
    kw["iterations"] = int(np.mean(its) * 1.1) + 1
    m = CatBoostClassifier(**kw)
    m.fit(X[ok], y[ok])
    pred = m.predict_proba(Xtest)[:, 1]
    del m
    free()
    return oof, pred.astype(np.float32)


# ---------------------------------------------------------------------------
# outputs
# ---------------------------------------------------------------------------
def write_lists(path, s1_ids, s_idx, t_ids_arr, col):
    df = pd.DataFrame({"s": s1_ids[s_idx], "t": t_ids_arr})
    df = df.drop_duplicates()
    agg = df.groupby("s", sort=False)["t"].agg(",".join)
    out = pd.DataFrame({"source1_entity_id": s1_ids})
    out[col] = out["source1_entity_id"].map(agg).fillna("")
    out.to_csv(path, sep="\t", index=False)
    return out


def validate_outputs(match_path, cand_path, test_s1_ids, test_target_ids):
    """Same rules as utils/validate_submission.py (problem statement)."""
    issues = []
    s1_set = set(test_s1_ids)
    tgt_set = set(test_target_ids)
    lists = {}
    for path, col in [(match_path, "matched_entity_ids"), (cand_path, "candidate_entity_ids")]:
        df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
        if list(df.columns) != ["source1_entity_id", col]:
            issues.append(f"{path.name}: bad header {list(df.columns)}")
        if df["source1_entity_id"].duplicated().any():
            issues.append(f"{path.name}: duplicate source1_entity_id rows")
        if set(df["source1_entity_id"]) != s1_set:
            issues.append(f"{path.name}: S1 ids differ from test_source1 "
                          f"(missing {len(s1_set - set(df['source1_entity_id']))})")
        d = {}
        for sid, ids in zip(df["source1_entity_id"], df[col]):
            li = [x for x in ids.split(",") if x] if ids else []
            if len(li) != len(set(li)):
                issues.append(f"{path.name}: duplicate ids in list of {sid}")
                break
            bad = [x for x in li if x not in tgt_set or not (x.startswith("S2-") or x.startswith("S3-"))]
            if bad:
                issues.append(f"{path.name}: unknown / non S2-S3 ids for {sid}: {bad[:3]}")
                break
            d[sid] = set(li)
        lists[col] = d
    m, c = lists.get("matched_entity_ids", {}), lists.get("candidate_entity_ids", {})
    not_sub = sum(1 for k, v in m.items() if not v <= c.get(k, set()))
    if not_sub:
        issues.append(f"{not_sub} S1 rows have matches that are not candidates")
    return issues


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def run_final(cfg: FinalConfig):
    import lightgbm as lgb  # noqa: F401  (import early: fail fast)
    from sklearn.model_selection import GroupKFold

    T0 = time.time()
    n_jobs = n_workers(cfg.n_jobs)
    out_dir = Path(cfg.out_dir or ("/kaggle/working" if is_kaggle() else "final_out"))
    (out_dir / "output").mkdir(parents=True, exist_ok=True)
    data_dir = auto_data_dir(cfg.data_dir, "train")
    log(f"data: {data_dir} | out: {out_dir} | workers: {n_jobs}")
    bcfg = BlockConfig(n_jobs=n_jobs)
    report = {"config": asdict(cfg)}

    # ===================== PHASE A: train (CPU) =====================
    log("PHASE A - train data")
    s1, tgt, gt = load_split(data_dir, "train")
    truth = gt_pairs(gt)
    del gt
    n_s1_total = len(s1)
    rng = np.random.default_rng(cfg.seed)
    if cfg.train_s1_sample and cfg.train_s1_sample < len(s1):
        s1 = s1.iloc[np.sort(rng.choice(len(s1), cfg.train_s1_sample, replace=False))]
    s1 = s1.reset_index(drop=True)
    log(f"  S1 used: {len(s1):,}  targets: {len(tgt):,}")
    normalize_frame(s1, n_jobs)
    normalize_frame(tgt, n_jobs)
    idf = build_idf(tgt)
    # reverse search (target -> its top-R S1) is only meaningful when ALL S1
    # compete; on a sample every target would pick among far fewer S1 records
    # and flood training with extra negatives -> off for sampled training
    sampled = bool(cfg.train_s1_sample) and len(s1) < n_s1_total
    tcfg = BlockConfig(n_jobs=n_jobs, reverse_k=0 if sampled else bcfg.reverse_k)
    A = featurize(s1, tgt, np.arange(len(s1)), tcfg, n_jobs, idf, cfg.batch_s1)
    s1_ids = s1["entity_id"].to_numpy()
    t_ids = tgt["entity_id"].to_numpy()
    del s1, tgt, idf
    free()

    truth = truth[truth["source1_entity_id"].isin(pd.Index(s1_ids))]
    s_index = pd.Series(np.arange(len(s1_ids)), index=s1_ids)
    n_true = np.bincount(s_index[truth["source1_entity_id"]].to_numpy(), minlength=len(s1_ids)).astype(float)
    key = pd.MultiIndex.from_arrays([s1_ids[A["s"]], t_ids[A["t"]]])
    y = key.isin(pd.MultiIndex.from_frame(truth)).astype(np.int8)
    del key, truth
    X, s_row, t_row = A["X"], A["s"], A["t"]
    reach = y.sum() / max(n_true.sum(), 1)
    log(f"  blocking recall ceiling: {reach:.4%} ({int(y.sum()):,}/{int(n_true.sum()):,}), "
        f"{len(y) / len(s1_ids):.1f} pairs per S1")
    dd = dict(zip(DOUBT_COLS, A["doubt"].T))
    weak_pos = ((y == 1) & (dd["n_tset_skel"] < 35) & (dd["n_idf_contain"] < 0.2) & (dd["a_tset"] < 40)
                & (dd["num_shared"] == 0) & (dd["n_alt_best"] < 60))
    twin_neg = ((y == 0) & (dd["n_exact_skel"] == 1) & (dd["a_missing_t"] == 0)
                & ((dd["a_tset"] >= 97) | ((dd["num_jacc"] >= 0.999) & (dd["num_s"] > 0))))
    ok = ~(weak_pos | twin_neg)
    log(f"  noise separation: {int(weak_pos.sum()):,} unlearnable positives and "
        f"{int(twin_neg.sum()):,} twin negatives removed from TRAINING only")
    del A, dd
    free()

    folds = list(GroupKFold(n_splits=cfg.folds).split(s_row, y, s_row))
    log("stage 1: LightGBM on all blocked pairs (OOF)")
    p_lgb1 = dict(objective="binary", learning_rate=0.1, num_leaves=127, min_data_in_leaf=50,
                  feature_fraction=0.7, bagging_fraction=0.7, bagging_freq=1, lambda_l2=1.0,
                  verbose=-1, seed=cfg.seed, num_threads=n_jobs, force_col_wise=True)
    oof1, bst1 = lgb_oof(X, y, folds, ok, p_lgb1, 3000, "lgb-stage1", n_jobs)
    m1 = macro_micro(s_row, y, threshold_select(s_row, t_row, oof1, 0.5), n_true, cfg.beta)
    log(f"  stage-1 OOF @0.5: macro F0.5 {m1['macro_f05']:.5f}")

    # learned pruning: smallest candidate set keeping >= prune_keep of reachable links
    r1 = rank_in_s(s_row, oof1)
    best = None
    for K in (3, 4, 5, 6, 8, 10, 12, 16):
        for tau in (0.001, 0.003, 0.01, 0.02, 0.05):
            keep = (oof1 >= tau) & (r1 < K)
            if y[keep].sum() >= cfg.prune_keep * y.sum():
                if best is None or keep.sum() < best[2]:
                    best = (K, tau, int(keep.sum()))
    if best is None:
        best = (16, 0.001, int(((oof1 >= 0.001) & (r1 < 16)).sum()))
    K, tau, _ = best
    P = (oof1 >= tau) & (r1 < K)
    log(f"  pruning rule: rank < {K} and p1 >= {tau} -> {P.sum() / len(s1_ids):.2f} candidates per S1, "
        f"keeps {y[P].sum() / max(y.sum(), 1):.4%} of reachable links "
        f"(recall ceiling {y[P].sum() / max(n_true.sum(), 1):.4%})")
    report["train"] = {"blocking_recall": float(reach), "pairs_per_s1_blocking": float(len(y) / len(s1_ids)),
                       "prune_K": K, "prune_tau": tau, "cands_per_s1": float(P.sum() / len(s1_ids)),
                       "recall_after_prune": float(y[P].sum() / max(n_true.sum(), 1)),
                       "stage1_oof": m1}
    Xp, yp, sp, tp_, okp, p1p = X[P], y[P], s_row[P], t_row[P], ok[P], oof1[P]
    del X, y, s_row, t_row, ok, oof1, r1, folds
    free()

    # ===================== PHASE B: test (CPU) =====================
    log("PHASE B - test data")
    ts1, ttgt, _ = load_split(data_dir, "test")
    log(f"  S1={len(ts1):,} targets={len(ttgt):,}  countries={sorted(ts1['country'].astype(str).unique())}")
    normalize_frame(ts1, n_jobs)
    normalize_frame(ttgt, n_jobs)
    tidf = build_idf(ttgt)

    def scorer(Xb, sb):
        p = bst1.predict(Xb, num_threads=n_jobs).astype(np.float32)
        return (p >= tau) & (rank_in_s(sb, p) < K), p

    B = featurize(ts1, ttgt, np.arange(len(ts1)), bcfg, n_jobs, tidf, cfg.batch_s1, scorer=scorer)
    ts1_ids = ts1["entity_id"].to_numpy()
    tt_ids = ttgt["entity_id"].to_numpy()
    del ts1, ttgt, tidf
    free()
    Xt, st, tt, p1t = B["X"], B["s"], B["t"], B["p1"]
    log(f"  test: {B['n_block']:,} blocked pairs -> {len(st):,} candidates "
        f"({len(st) / len(ts1_ids):.2f} per S1)")
    cand_path = out_dir / "output" / "candidate_pairs.tsv"
    write_lists(cand_path, ts1_ids, st, tt_ids[tt], "candidate_entity_ids")
    report["test"] = {"n_s1": int(len(ts1_ids)), "blocked_pairs": int(B["n_block"]),
                      "cands_per_s1": float(len(st) / len(ts1_ids))}
    del B
    free()

    # ===================== PHASE C: ensemble (GPU allowed) =====================
    use_gpu = cfg.gpu == "on" or (cfg.gpu == "auto" and gpu_available())
    log(f"PHASE C - ensemble on pruned pairs (train {len(yp):,}, test {len(st):,}); GPU={use_gpu}")
    folds = list(GroupKFold(n_splits=cfg.folds).split(sp, yp, sp))
    members_oof = {"lgb1": p1p}
    members_test = {"lgb1": p1t}
    p_lgb2 = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=30,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=2.0,
                  verbose=-1, seed=cfg.seed + 1, num_threads=n_jobs, force_col_wise=True)
    o, b2 = lgb_oof(Xp, yp, folds, okp, p_lgb2, 4000, "lgb-stage2", n_jobs)
    members_oof["lgb2"], members_test["lgb2"] = o, b2.predict(Xt, num_threads=n_jobs).astype(np.float32)
    b2.save_model(str(out_dir / "lgb_stage2.txt"))
    bst1.save_model(str(out_dir / "lgb_stage1.txt"))
    for name, fn, flag in (("xgb", xgb_oof, cfg.use_xgboost), ("cat", cat_oof, cfg.use_catboost)):
        if not flag:
            continue
        try:
            members_oof[name], members_test[name] = fn(Xp, yp, folds, okp, Xt, use_gpu, n_jobs, cfg.seed)
        except Exception as e:  # GPU OOM / missing lib -> retry on CPU, else skip
            log(f"    {name} failed on {'GPU' if use_gpu else 'CPU'} ({type(e).__name__}: {e})")
            if use_gpu:
                try:
                    members_oof[name], members_test[name] = fn(Xp, yp, folds, okp, Xt, False, n_jobs, cfg.seed)
                except Exception as e2:
                    log(f"    {name} skipped ({type(e2).__name__}: {e2})")
    names = list(members_oof)
    for nm in names:
        mm = macro_micro(sp, yp, threshold_select(sp, tp_, members_oof[nm], 0.5), n_true, cfg.beta)
        log(f"  member {nm:5s} OOF @0.5 macro F0.5 = {mm['macro_f05']:.5f}")

    def logit(p):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))

    blends = {nm: [nm] for nm in names}
    if len(names) > 1:
        blends["mean_all"] = names
        second = [n for n in names if n != "lgb1"]
        if len(second) > 1:
            blends["mean_stage2"] = second
    missed = max(float(n_true.sum() - yp.sum()), 0.0)
    lam_hat = missed / len(s1_ids)
    results = {}
    for bname, mem in blends.items():
        z = np.mean([logit(members_oof[m]) for m in mem], axis=0)
        pb = 1 / (1 + np.exp(-z))
        cal = Calibrator().fit(pb, yp)
        q = cal(pb)
        rule, m = tune_rule(sp, tp_, yp, q, n_true, cfg.beta, lam_hat)
        results[bname] = (m["macro_f05"], rule, cal, mem, m)
        log(f"  blend {bname:12s} OOF macro F0.5 = {m['macro_f05']:.5f}  rule={rule}")
    bname = max(results, key=lambda k: results[k][0])
    score, rule, cal, mem, m = results[bname]
    log(f"CHOSEN: blend={bname} {mem} rule={rule}")
    log(f"  OOF (all sampled train S1, blocking + pruning losses counted): "
        f"macro F0.5={m['macro_f05']:.5f}  macro P={m['macro_p']:.5f}  macro R={m['macro_r']:.5f}  "
        f"micro F0.5={m['micro_f05']:.5f}")
    report["oof"] = {"chosen_blend": bname, "members": mem, "rule": rule, **m,
                     "all_blends": {k: v[0] for k, v in results.items()}}

    zt = np.mean([logit(members_test[mm]) for mm in mem], axis=0)
    qt = cal(1 / (1 + np.exp(-zt)))
    sel = apply_rule(rule, st, tt, qt, cfg.beta)
    match_path = out_dir / "output" / "matching_results.tsv"
    res = write_lists(match_path, ts1_ids, st[sel], tt_ids[tt[sel]], "matched_entity_ids")
    n_links = int(sel.sum())
    empty = float((res["matched_entity_ids"] == "").mean())
    log(f"  test: {n_links:,} links, {empty:.2%} S1 predicted as singletons")
    report["test"].update({"links": n_links, "pred_singleton_rate": empty})

    issues = validate_outputs(match_path, cand_path, ts1_ids, tt_ids)
    report["validation"] = issues or "PASS"
    log("VALIDATION: " + ("PASS" if not issues else "; ".join(issues)))
    if is_kaggle():
        shutil.copy(match_path, "/kaggle/working/submission.tsv")
    report["runtime_min"] = round((time.time() - T0) / 60, 1)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, default=str))
    log(f"done in {report['runtime_min']} min -> {match_path}")
    return report
