"""Loading the challenge TSVs and attaching normalised columns."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .normalize import normalize_address, normalize_name
from .utils import free, log as _log, n_workers, run_pool

USECOLS = ["entity_id", "business_name", "business_address", "country"]
_G: dict = {}

NAME_FIELDS = ["core", "full", "ocr", "skel", "nospace", "alts", "legal", "acr",
               "indic", "domain", "junk"]
ADDR_FIELDS = ["atext", "alpha", "nums", "numparts", "numstreet", "first_num",
               "state", "postal", "missing"]


def find_files(data_dir: str | Path) -> dict:
    found = {p.name: p for p in Path(data_dir).rglob("*.tsv")}
    return found


def read_tsv(path, usecols=None) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False,
                       usecols=usecols, engine="c", na_filter=False)


def load_split(data_dir, split: str):
    """Returns (s1, targets, ground_truth or None). targets = S2 + S3."""
    f = find_files(data_dir)
    s1 = read_tsv(f[f"{split}_source1.tsv"], USECOLS)
    s2 = read_tsv(f[f"{split}_source2.tsv"], USECOLS)
    s3 = read_tsv(f[f"{split}_source3.tsv"], USECOLS)
    s2["source"] = "S2"
    s3["source"] = "S3"
    tgt = pd.concat([s2, s3], ignore_index=True)
    del s2, s3
    free()
    gt = None
    if f"{split}_ground_truth.tsv" in f:
        gt = read_tsv(f[f"{split}_ground_truth.tsv"])
    for df in (s1, tgt):
        df["country"] = df["country"].astype(str).str.strip().astype("category")
    tgt["source"] = tgt["source"].astype("category")
    return s1, tgt, gt


def gt_pairs(gt: pd.DataFrame) -> pd.DataFrame:
    """Explode 'S2-1,S3-2' -> one row per (source1_entity_id, matched_id)."""
    g = gt[["source1_entity_id", "matched_entity_ids"]].copy()
    g["matched_id"] = g["matched_entity_ids"].astype(str).str.split(",")
    g = g.explode("matched_id")
    g["matched_id"] = g["matched_id"].astype(str).str.strip()
    g = g[g["matched_id"].ne("") & g["matched_id"].ne("nan")]
    return g[["source1_entity_id", "matched_id"]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Parallel normalisation over unique values (fork-shared input, no pickling)
# ---------------------------------------------------------------------------
def _name_worker(b):
    vals = _G["vals"]
    return [tuple(normalize_name(v)[k] for k in NAME_FIELDS) for v in vals[b[0]:b[1]]]


def _addr_worker(b):
    vals, ctry = _G["vals"], _G["ctry"]
    return [tuple(normalize_address(v, c)[k] for k in ADDR_FIELDS)
            for v, c in zip(vals[b[0]:b[1]], ctry[b[0]:b[1]])]


def _run(fn, n, n_jobs, chunk=20000):
    out = []
    for res in run_pool(fn, [(i, min(i + chunk, n)) for i in range(0, n, chunk)], n_jobs):
        out.extend(res)
    return out


def _assign(df, prefix, fields, int_fields, res, codes):
    for j, k in enumerate(fields):
        col = np.array([r[j] for r in res], dtype=object)[codes]
        df[prefix + k] = col.astype(np.int8) if k in int_fields else col


def normalize_frame(df: pd.DataFrame, n_jobs: int | None = None, log=_log,
                    drop_raw: bool = True) -> pd.DataFrame:
    """Adds n_* (name) and a_* (address) columns.  Work is done once per
    UNIQUE value; the resulting object arrays share string objects, so
    duplicated names cost one pointer each.  Raw text is dropped afterwards."""
    n_jobs = n_workers(n_jobs)
    names = df["business_name"].astype(str)
    codes, uniq = pd.factorize(names)
    log(f"  normalising {len(uniq):,} unique names")
    _G["vals"] = np.asarray(uniq, dtype=object)
    res = _run(_name_worker, len(uniq), n_jobs)
    _assign(df, "n_", NAME_FIELDS, ("indic", "domain", "junk"), res, codes)
    del res, codes, uniq, names

    key = pd.MultiIndex.from_arrays([df["business_address"].astype(str),
                                     df["country"].astype(str)])
    codes, uniq = pd.factorize(key)
    log(f"  normalising {len(uniq):,} unique addresses")
    _G["vals"] = np.asarray(uniq.get_level_values(0), dtype=object)
    _G["ctry"] = np.asarray(uniq.get_level_values(1), dtype=object)
    res = _run(_addr_worker, len(uniq), n_jobs)
    _assign(df, "a_", ADDR_FIELDS, ("missing",), res, codes)
    _G.clear()
    if drop_raw:
        df.drop(columns=["business_name", "business_address"], inplace=True)
    free()
    return df


def load_normalized(data_dir, split, cache_dir=None, n_jobs=None, s1_ids=None, log=_log):
    """load_split + normalize_frame with an optional parquet/pickle cache so a
    re-run of the notebook skips the expensive normalisation."""
    cache = Path(cache_dir) / f"norm_{split}" if cache_dir else None
    if cache is not None and (cache / "tgt.pkl").exists():
        log(f"loading normalised {split} from cache {cache}")
        s1 = pd.read_pickle(cache / "s1.pkl")
        tgt = pd.read_pickle(cache / "tgt.pkl")
        gt = pd.read_pickle(cache / "gt.pkl") if (cache / "gt.pkl").exists() else None
    else:
        log(f"loading {split} from {data_dir}")
        s1, tgt, gt = load_split(data_dir, split)
        log(f"  S1={len(s1):,} targets={len(tgt):,}")
        log("normalising S1 ...")
        normalize_frame(s1, n_jobs, log)
        log("normalising targets ...")
        normalize_frame(tgt, n_jobs, log)
        if cache is not None:
            cache.mkdir(parents=True, exist_ok=True)
            s1.to_pickle(cache / "s1.pkl")
            tgt.to_pickle(cache / "tgt.pkl")
            if gt is not None:
                gt.to_pickle(cache / "gt.pkl")
    if s1_ids is not None:
        s1 = s1[s1["entity_id"].isin(s1_ids)].reset_index(drop=True)
    return s1, tgt, gt
