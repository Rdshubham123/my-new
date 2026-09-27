"""Kaggle helpers: path auto-detection, deadlock-safe process pools, memory logs."""
from __future__ import annotations

import gc
import os
import sys
import time
from array import array
from multiprocessing import get_context
from multiprocessing.context import TimeoutError as MPTimeout
from pathlib import Path

import numpy as np

_T0 = time.time()

# Keep native thread pools single-threaded inside forked workers: forking a
# process whose OpenMP/BLAS pool is live and then using that pool in the
# child is the classic multiprocessing deadlock.  Set before numpy/lightgbm
# spin their pools up in the children.
_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")

REQUIRED = ("train_source1.tsv", "train_source2.tsv", "train_source3.tsv")
SEARCH_ROOTS = ("/kaggle/input", "/kaggle/working", ".", "./dataset", "./data", "..")


def rss_gb() -> float:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024 ** 2
    except OSError:
        pass
    return float("nan")


def log(msg: str) -> None:
    print(f"[{time.time() - _T0:7.1f}s | {rss_gb():5.1f} GB] {msg}", flush=True)


def free() -> None:
    gc.collect()


def is_kaggle() -> bool:
    return Path("/kaggle/working").is_dir() or "KAGGLE_KERNEL_RUN_TYPE" in os.environ


def auto_data_dir(explicit: str | None = None, split: str = "train") -> Path:
    """Find the folder tree holding the challenge TSVs.

    Checks the explicit path, $EM_DATA_DIR, then searches /kaggle/input (any
    dataset/competition slug, any nesting depth) and local folders for
    `{split}_source1.tsv`.  Returns the common root that contains both the
    train and test files when possible."""
    cands = [explicit, os.environ.get("EM_DATA_DIR")] + list(SEARCH_ROOTS)
    for root in cands:
        if not root:
            continue
        root = Path(root)
        if not root.is_dir():
            continue
        hits = sorted(root.rglob(f"{split}_source1.tsv"))
        if not hits:
            continue
        # prefer a root that also contains the other split
        for h in hits:
            for parent in [h.parent, h.parent.parent]:
                names = {p.name for p in parent.rglob("*.tsv")}
                if {"train_source1.tsv", "test_source1.tsv"} <= names:
                    return parent
        return hits[0].parent.parent if hits[0].parent.name == split else hits[0].parent
    raise FileNotFoundError(
        f"Could not find {split}_source1.tsv under {', '.join(str(c) for c in cands if c)}. "
        "Attach the dataset to the notebook or pass --data-dir.")


def default_out_dir() -> Path:
    return Path("/kaggle/working/em_model") if is_kaggle() else Path("em_output")


def n_workers(requested: int | None = None) -> int:
    if requested:
        return max(1, int(requested))
    if hasattr(os, "sched_getaffinity"):
        return max(1, len(os.sched_getaffinity(0)))
    return max(1, os.cpu_count() or 1)


def _init_worker():
    for v in _THREAD_VARS:
        os.environ[v] = "1"
    try:  # workers must not handle Ctrl-C / kernel interrupts themselves
        import signal
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    except Exception:
        pass


def run_pool(fn, tasks, n_jobs: int, ordered: bool = True, timeout: float = 600.0):
    """Map `fn` over `tasks` with a fork pool.

    - fork shares the big read-only arrays with workers (no pickling, no copy)
    - every result has a timeout: a stuck/killed worker (e.g. OOM-killed by
      the kernel) can no longer hang the notebook forever; remaining tasks are
      finished serially instead
    - falls back to serial execution on platforms without fork"""
    tasks = list(tasks)
    if n_jobs <= 1 or len(tasks) <= 1 or sys.platform == "win32":
        return [fn(t) for t in tasks]
    results = [None] * len(tasks)
    done = [False] * len(tasks)
    ctx = get_context("fork")
    # freeze the parent's objects out of the cyclic GC: a worker's GC pass
    # would otherwise write to every inherited container header (more COW)
    gc.collect()
    gc.freeze()
    # maxtasksperchild recycles workers: any memory a worker accumulated
    # (copy-on-write pages, caches) is returned to the OS regularly
    pool = ctx.Pool(min(n_jobs, len(tasks)), initializer=_init_worker, maxtasksperchild=25)
    try:
        asyncs = [pool.apply_async(fn, (t,)) for t in tasks]
        for i, ar in enumerate(asyncs):
            try:
                results[i] = ar.get(timeout=timeout)
                done[i] = True
            except MPTimeout:
                log(f"  worker timeout on task {i}; finishing remaining tasks serially")
                break
        pool.close()
    except BaseException:
        pool.terminate()
        raise
    finally:
        if not all(done):
            pool.terminate()
        pool.join()
        gc.unfreeze()
    for i, ok in enumerate(done):
        if not ok:
            results[i] = fn(tasks[i])
    return results


# ---------------------------------------------------------------------------
# Copy-on-write-safe strings for forked workers
# ---------------------------------------------------------------------------
# A forked worker that merely READS a Python str still writes its refcount,
# so the kernel copies every memory page it touches: with millions of
# normalised strings and 4 workers the parent's text is duplicated up to 4x
# (the classic "copy-on-read" OOM).  Workers therefore read strings from ONE
# bytes buffer + offsets (only that object's header page is ever written).
class StrCol:
    __slots__ = ("buf", "off")

    def __init__(self, values):
        enc = [("" if v is None else str(v)).encode("utf-8", "surrogatepass") for v in values]
        off = np.zeros(len(enc) + 1, np.int64)
        if enc:
            np.cumsum(np.fromiter((len(e) for e in enc), np.int64, len(enc)), out=off[1:])
        self.buf = b"".join(enc)
        self.off = array("q", off.tobytes())

    def __len__(self):
        return len(self.off) - 1

    def __getitem__(self, i):
        return self.buf[self.off[i]:self.off[i + 1]].decode("utf-8", "surrogatepass")


class CodedCol:
    """Row access to a categorical column: codes (numpy) -> StrCol of categories."""
    __slots__ = ("codes", "cats")

    def __init__(self, codes, cats):
        self.codes, self.cats = codes, cats

    def __len__(self):
        return len(self.codes)

    def __getitem__(self, i):
        c = self.codes[i]
        return self.cats[c] if c >= 0 else ""


_STRCOL_CACHE: dict = {}


def _strcol_for(categories):
    key = id(categories)
    hit = _STRCOL_CACHE.get(key)
    if hit is None or hit[0] is not categories:
        hit = (categories, StrCol(categories))
        _STRCOL_CACHE[key] = hit
    return hit[1]


def clear_strcol_cache():
    _STRCOL_CACHE.clear()


def worker_view(df, cols):
    """dict col -> COW-safe row accessor (CodedCol for text, numpy otherwise)."""
    import pandas as pd
    out = {}
    for c in cols:
        s = df[c]
        if isinstance(s.dtype, pd.CategoricalDtype):
            out[c] = CodedCol(s.cat.codes.to_numpy(), _strcol_for(s.cat.categories))
        elif s.dtype == object:
            cat = s.astype("category")
            out[c] = CodedCol(cat.cat.codes.to_numpy(), StrCol(cat.cat.categories))
        else:
            out[c] = s.to_numpy()
    return out
