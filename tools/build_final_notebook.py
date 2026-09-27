"""Builds final_submission.ipynb: self-contained Kaggle notebook for the final run."""
import nbformat as nbf
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PKG = REPO / "entity_matching"
MODULES = ["__init__.py", "patterns.py", "normalize.py", "utils.py", "data.py", "blocking.py",
           "features.py", "decision.py", "stage2.py", "metrics.py", "synth.py", "charnn.py", "final.py"]

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = []

cells.append(md("""# Amazon ML Challenge 2026 — Business Entity Resolution · FINAL (v3)

**Two-stage learned matcher + GPU ensemble, tuned for the official metric** (F0.5 per Source-1 entity,
macro-averaged, with an empty prediction for a singleton scoring 1.0).

**Run:** attach the dataset, pick the **GPU T4×2** accelerator (CPU also works), set `TEAM_NAME` in the config cell,
then **Run All**. The notebook writes:
- `/kaggle/working/output/matching_results.tsv` → upload this to the portal
- `/kaggle/working/output/candidate_pairs.tsv`
- `/kaggle/working/<TEAM_NAME>_submission.zip` → the final package (outputs, code, README, requirements,
  methodology)

### Strategy
| Phase | Where | What |
|---|---|---|
| A · train | CPU, fork pools | Hardcoded noise normalisation (Indic scripts, legal forms, OCR digits, junk, DBA, NULLs, abbreviations, FR/US/IN states) → IDF-weighted multi-key blocking, both directions, adaptive K, per-record posting budget → 75 pair/rule/context features → **stage-1 LightGBM** (GroupKFold OOF) → **learned pruning** (smallest candidate set keeping ≥ 99.8% of reachable true links) |
| B · test | CPU, fork pools | Same features; stage-1 scores each batch and prunes it immediately → **`candidate_pairs.tsv` averages ~4 candidates per S1** (a smaller set ranks higher in the final evaluation) |
| C · ensemble | **GPU**, no more forks | LightGBM + **XGBoost (CUDA)** + **CatBoost (GPU)** + **charnn**, a deep character-level cross-attention matcher (ESIM-style soft alignment over name and address bytes, fused with the tabular features, mixed precision). All are trained on the pruned pairs with the same OOF folds. The blend (single, logit-mean of all / top-3, or best GBDT + charnn) and the decision rule (threshold, or per-S1 **expected-F0.5 set**, optionally **per country**) are chosen on OOF macro F0.5. Isotonic calibration; one owner per target |

**Noise separation:** true pairs with no name, address or number evidence ("unlearnable") and non-links
indistinguishable from a link ("twins") are removed from **training only**. Validation always scores every held-out S1,
and true links lost to blocking or pruning count as misses, so the OOF number is honest.

**Deadlock and RAM safety:**
- All multiprocessing happens before CUDA is first used, and the whole pipeline runs in a fresh subprocess.
- Workers read fork-shared arrays instead of pickled copies. Each pool task has a timeout with a serial fallback.
- Blocking work per record is bounded. Test features are computed and pruned in S1 batches.

**Rules compliance:** only the provided training data is used; no external lookups. The models (LightGBM,
XGBoost, CatBoost) are MIT/Apache-2.0 and far below 8B parameters."""))

cells.append(md("## 0 · Configuration"))
cells.append(code("""TEAM_NAME = "team"          # <- your team name (used for the zip file name)
TRAIN_S1_SAMPLE = 100_000   # S1 records used for training (plenty for the models; keeps RAM and time safe)
FOLDS = 3
GPU = "auto"                # auto | on | off
RUN_SMOKE_TEST = True       # ~1 min synthetic end-to-end check before the real run
DATA_DIR = None             # None = auto-detect under /kaggle/input"""))

cells.append(md("## 1 · Dependencies"))
cells.append(code("""import importlib, os, subprocess, sys
def ensure(pkg, mod=None):
    try:
        importlib.import_module(mod or pkg); print(f"ok        {pkg}")
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], capture_output=True)
        try:
            importlib.import_module(mod or pkg); print(f"installed {pkg}")
        except ImportError:
            print(f"MISSING   {pkg}" + (" (built-in transliterator fallback)" if pkg == "anyascii" else ""))
for p in ["pandas", "numpy", "scikit-learn:sklearn", "lightgbm", "xgboost", "catboost", "rapidfuzz", "anyascii", "torch"]:
    n, _, m = p.partition(":"); ensure(n, m or None)

WORK = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.getcwd()
os.chdir(WORK)
os.makedirs("entity_matching", exist_ok=True)
print("working dir:", WORK)
print(subprocess.run(["bash", "-c", "nvidia-smi -L 2>/dev/null || echo 'no GPU (CPU mode)'; nproc; free -g | head -2"],
                     capture_output=True, text=True).stdout)"""))

cells.append(md("## 2 · Pipeline source (written to `entity_matching/`)"))
for m in MODULES:
    src = (PKG / m).read_text()
    if not src.strip():
        src = '"""Business entity resolution pipeline (Amazon ML Challenge 2026)."""\n'
    cells.append(code(f"%%writefile entity_matching/{m}\n" + src.rstrip("\n")))
cells.append(code("%%writefile run_final.py\n" + (REPO / "run_final.py").read_text().rstrip("\n")))

cells.append(md("## 3 · Sanity check: hardcoded noise patterns on real true pairs from the EDA"))
cells.append(code("""import pandas as pd
from entity_matching.normalize import normalize_name
pairs = [("Gold Solutions", "গোল্ড সলিউশনস"), ("Jain Builders Private Limited", "जैन बिल्डर्स प्राइवेट लिमिटेड"),
         ("Creative Builders", "ಕ್ರಿಯೇಟಿವ್ ಬಿಲ್ಡರ್ಸ್"), ("Arihant Technology Limited", "Arihant टेक्नोलॉजी लिमिटेड"),
         ("Creative Intermediate LLC", "Llc Creative Intermediate"), ("South Academy", "5outh Academy"),
         ("Washington, Harris and Burrell", "WASHINGT0N, HARRIS AND BURRELL"),
         ("Midwest Applied, LLC", "Midwest Applied, ([LLC])"),
         ("Fort Worth Telecommunication", "Fort Worth Telecommunication L.L.C. (ID: 49166)"),
         ("Summit Ministries", "Summit Ministries lnc"), ("Bright Automation Enterprises", "Bright Bright Automation Enterprises")]
df = pd.DataFrame([{"S1": a, "S2/S3": b, "key S1": normalize_name(a)["skel"], "key S2/S3": normalize_name(b)["skel"]}
                   for a, b in pairs])
df["match"] = df["key S1"] == df["key S2/S3"]
print(f"{df['match'].sum()}/{len(df)} real true pairs normalise to the same key"); df"""))

cells.append(md("""## 4 · Smoke test (synthetic data, separate process)
Runs the complete pipeline on 2k synthetic S1 records with the same folder layout, including France in test. It exercises
blocking, pruning, the GPU/CPU ensemble, validation and output writing in about a minute, so an environment problem
shows up **before** the long run. It runs in a subprocess, so the notebook kernel never touches CUDA."""))
cells.append(code("""import shutil
if RUN_SMOKE_TEST:
    shutil.rmtree("_smoke", ignore_errors=True)
    r = subprocess.run([sys.executable, "-c",
        "from entity_matching.synth import write_synthetic; write_synthetic('_smoke/data', 2000, 1000, 7)"],
        capture_output=True, text=True); print(r.stdout[-500:], r.stderr[-2000:])
    r = subprocess.run([sys.executable, "run_final.py", "--data-dir", "_smoke/data", "--out-dir", "_smoke/out",
                        "--train-s1-sample", "0", "--gpu", GPU], capture_output=True, text=True)
    tail = "\\n".join(l for l in r.stdout.splitlines() if "features batch" not in l)
    print(tail[-3500:]); print(r.stderr[-3000:])
    assert r.returncode == 0 and "VALIDATION: PASS" in r.stdout, "SMOKE TEST FAILED - fix before the real run"
    shutil.rmtree("_smoke", ignore_errors=True)
    for f in ("submission.tsv",):  # never leave a synthetic file behind
        if os.path.exists(f): os.remove(f)
    print("SMOKE TEST PASSED")"""))

cells.append(md("""## 5 · Real run (fresh subprocess, live log)
The pipeline runs as **three fresh processes**: A (train), B (test), then C (ensemble, GPU). Each starts with clean memory,
and CUDA is initialised only in C. Rough runtime on Kaggle: A ~25 min, B ~35 min, C ~10 min.
The log shows RAM per step, the blocking recall ceiling, candidates per S1, the OOF score of every member and blend,
and the final validation."""))
cells.append(code("""cmd = [sys.executable, "-u", "run_final.py", "--train-s1-sample", str(TRAIN_S1_SAMPLE),
       "--folds", str(FOLDS), "--gpu", GPU, "--team", TEAM_NAME, "--out-dir", WORK]
if DATA_DIR: cmd += ["--data-dir", DATA_DIR]
with open("run_log.txt", "w") as logf:
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in proc.stdout:
        logf.write(line)
        print(line, end="")  # every line: progress is always visible
    proc.wait()
print("exit code:", proc.returncode)
assert proc.returncode == 0, "run failed - see run_log.txt\""""))

cells.append(md("## 6 · Results"))
cells.append(code("""import json
rep = json.load(open(os.path.join(WORK, "report.json")))
print("VALIDATION:", rep["validation"])
print(f"train: blocking recall {rep['train']['blocking_recall']:.4%} | after pruning {rep['train']['recall_after_prune']:.4%} "
      f"| candidates/S1 {rep['train']['cands_per_s1']:.2f}")
print(f"test : {rep['test']['n_s1']:,} S1 | candidates/S1 {rep['test']['cands_per_s1']:.2f} | links {rep['test']['links']:,} "
      f"| predicted singletons {rep['test']['pred_singleton_rate']:.2%}")
o = rep["oof"]
print(f"OOF  : blend={o['chosen_blend']} rule={o['rule']}")
print(f"       macro F0.5 {o['macro_f05']:.5f} | macro P {o['macro_p']:.5f} | macro R {o['macro_r']:.5f}")
display(pd.Series(o["all_blends"], name="OOF macro F0.5").sort_values(ascending=False).to_frame())
for f in ("matching_results.tsv", "candidate_pairs.tsv"):
    display(pd.read_csv(os.path.join(WORK, "output", f), sep="\\t", dtype=str, keep_default_na=False).head(5))"""))

cells.append(md("## 7 · Final submission package (zip)"))
cells.append(code("""import importlib.metadata as im, zipfile, datetime
pkg_root = os.path.join(WORK, "code", "business_entity_resolution")
shutil.rmtree(os.path.join(WORK, "code"), ignore_errors=True)
os.makedirs(os.path.join(pkg_root, "src"), exist_ok=True)
shutil.copytree("entity_matching", os.path.join(pkg_root, "src", "entity_matching"),
                ignore=shutil.ignore_patterns("__pycache__"))
shutil.copy("run_final.py", os.path.join(pkg_root, "src", "run_final.py"))
vers = []
for p in ["pandas", "numpy", "scikit-learn", "lightgbm", "xgboost", "catboost", "rapidfuzz", "anyascii", "torch"]:
    try: vers.append(f"{p}=={im.version(p)}")
    except Exception: pass
open(os.path.join(pkg_root, "requirements.txt"), "w").write("\\n".join(vers) + "\\n")
open(os.path.join(pkg_root, "README.md"), "w").write(f'''# Business Entity Resolution — reproduction

```bash
pip install -r requirements.txt
cd src
python run_final.py --data-dir <path to dataset/> --out-dir <out> --train-s1-sample {TRAIN_S1_SAMPLE} --folds {FOLDS} --gpu auto
```
Writes `<out>/output/matching_results.tsv`, `<out>/output/candidate_pairs.tsv`, `<out>/report.json`.
`--gpu off` runs everything on CPU. The pipeline is deterministic given the seed (42).
Pipeline: normalise -> block -> features -> stage-1 LightGBM -> learned pruning (candidate_pairs.tsv)
-> LightGBM + XGBoost + CatBoost ensemble -> calibration -> macro-F0.5 decision (matching_results.tsv).
''')
t, o = rep["train"], rep["oof"]
doc = f'''# Methodology — Amazon ML Challenge 2026: Business Entity Resolution
Team: {TEAM_NAME} · generated {datetime.date.today()}

## 1. Methodology
Source 1 is the deduplicated reference. For each S1 entity we (1) generate candidates from S2/S3, (2) score them with
a learned pairwise model, (3) prune to a small candidate set, (4) re-score with a gradient-boosting ensemble and
(5) choose the final set per S1 to maximise the **macro F0.5** (the official metric; singletons count as 1.0 when
predicted empty).

Data facts used (EDA): country is a hard boundary (100% of true links are same-country); every S2/S3 record belongs
to at most one S1 (one-owner constraint, enforced at decision time); 5.6% of S1 entities are singletons; France appears
only in test, so no rule depends on the countries seen in training.

**Noise normalisation (hardcoded patterns):**
- Indic scripts: native legal words are mapped (प्राइवेट→private …), the rest is transliterated, then a
  phonetic consonant skeleton is taken so `গোল্ড সলিউশনস` = `Gold Solutions`.
- Legal forms (US/IN/FR, dotted, bracketed, moved) are canonicalised and separated from the core name.
- OCR digits (5outh, WASHINGT0N, lnc) are repaired; accents folded.
- Junk (`***`, `(ID: 123)`, `M/s`), domains (aahanarealty.com) and DBA/formerly aliases are handled.
- Addresses: `NULL` placeholders, state names/codes (including native script), street and unit abbreviations,
  city aliases, canonical numbers (leading zeros, ordinals, compound numbers).

**Noise separation:** unlearnable positives (no name, address or number evidence) and twin negatives
(indistinguishable from a link) are removed from training folds only.

## 2. Candidate generation / blocking
- Country-partitioned inverted index over hashed keys, weighted by IDF. Key types: name skeleton tokens, bigrams,
  8-character space-less prefix, full name key, address words, canonical numbers, number+street, and
  **name-token × number** composites (rare even for generic names like "Pediatric Group").
- Each key has a document-frequency cap. Per S1, keys are used rarest-first under a posting budget.
- Forward adaptive-K (keep 12, up to 60 while score ≥ 0.25 × best), plus top-10 by name-only and address-only
  score, plus **reverse search** (each target keeps its top-5 S1).
- Blocking recall on the training sample: **{t["blocking_recall"]:.4%}** at {t["pairs_per_s1_blocking"]:.1f} pairs per S1.
- **Learned pruning:** stage-1 LightGBM scores all blocked pairs. We keep rank < {t["prune_K"]} and
  p ≥ {t["prune_tau"]}, the smallest set retaining ≥ 99.8% of reachable true links, chosen on out-of-fold
  predictions. Result: **{t["cands_per_s1"]:.2f} candidates per S1 (train, OOF)** and
  **{rep["test"]["cands_per_s1"]:.2f} per S1 on test**, recall ceiling {t["recall_after_prune"]:.4%}.
  `candidate_pairs.tsv` is exactly this pruned set, which is what the final ensemble scores.

## 3. Model architecture and features
- 75 features: fuzzy ratios over five name views (core / OCR-folded / phonetic / space-less / full), IDF-weighted
  token overlap, alias, domain, acronym and legal-form agreement, name frequency; address fuzzy and IDF overlap,
  number agreement (exact / compound / parts / house number / number+street, **rarity-weighted shared and
  conflicting numbers**), state and postal agreement; hardcoded rule flags; per-S1 rank/gap/sibling context.
- Stage 1: LightGBM (GroupKFold by S1). Stage 2 on pruned pairs: LightGBM + XGBoost (CUDA) + CatBoost (GPU)
  + **charnn**, a deep character-level cross-attention matcher: 96-byte name+address sequences, a shared residual
  char-CNN, ESIM-style soft alignment with [h−a, h⊙a] comparison, mean/max pooling, fused with the 75
  standardised tabular features. It is trained fold-wise with mixed precision and a wall-clock budget. All members
  produce OOF predictions; the blend (single, logit-mean of all / top-3, best GBDT + charnn) is chosen on OOF
  macro F0.5, then isotonic calibration.
- Decision: best on OOF of a threshold rule or the per-S1 **expected-F0.5 set** (plug-in General F-measure Maximizer:
  choose k maximising 1.25·Σq/(0.25·E|T|+k) against P(empty)), with one owner per target. A country's own rule
  is used only when it beats the global rule on that country's OOF by > 0.0002; France (unseen) uses the global rule.
- Country rules used: {o.get("country_rules", "none (global rule)")}.
- Chosen: blend **{o["chosen_blend"]}** ({", ".join(o["members"])}), rule {o["rule"]}.

## 4. Validation
GroupKFold by S1 on {rep["config"]["train_s1_sample"] or "all"} training S1 records. Every held-out S1 is scored, with
true links lost to blocking or pruning counted as false negatives.
**OOF macro F0.5 = {o["macro_f05"]:.5f}** (macro P {o["macro_p"]:.5f}, macro R {o["macro_r"]:.5f}).
Blend comparison: {", ".join(f"{k} {v:.5f}" for k, v in o["all_blends"].items())}.

## 5. Other
No external data, APIs or lookups. Models are LightGBM (MIT), XGBoost (Apache-2.0), CatBoost (Apache-2.0),
and a small PyTorch network (BSD-style licence, trained from scratch, about 0.37M parameters, far below 8B).
The pipeline is deadlock-safe (all forks before CUDA; pool timeouts) and bounded in RAM (batching, posting budget).
'''
open(os.path.join(WORK, "Documentation_template.md"), "w").write(doc)
zpath = os.path.join(WORK, f"{TEAM_NAME}_submission.zip")
with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
    for rel in ["output/matching_results.tsv", "output/candidate_pairs.tsv", "Documentation_template.md"]:
        z.write(os.path.join(WORK, rel), rel)
    for root, _, files in os.walk(os.path.join(WORK, "code")):
        for f in files:
            full = os.path.join(root, f); z.write(full, os.path.relpath(full, WORK))
print("package:", zpath, f"{os.path.getsize(zpath) / 1e6:.1f} MB")
with zipfile.ZipFile(zpath) as z:
    print("\\n".join(z.namelist()[:40]))"""))

nb = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"}})
out = REPO / "final_submission.ipynb"
nbf.write(nb, out)
print("wrote", out, len(cells), "cells")
