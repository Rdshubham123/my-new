import nbformat as nbf
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PKG = REPO / "entity_matching"
MODULES = ["__init__.py", "patterns.py", "normalize.py", "utils.py", "data.py",
           "blocking.py", "features.py", "decision.py", "stage2.py", "metrics.py",
           "synth.py", "run.py"]

md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell
cells = []

cells.append(md("""# Business entity matching — Amazon ML Challenge 2026 (S1 → S2/S3) · v2

A self-contained, precision-first entity-resolution notebook built from the findings of `dataesplore.ipynb`.
For every **Source-1** business it predicts which **Source-2 / Source-3** records describe the same business.
It is tuned for **macro F0.5**: F0.5 is computed per Source-1 entity and then averaged, which is the metric the
public write-ups describe for this challenge.

**How to use on Kaggle:**
1. Attach the competition dataset (for example `satwiksps/amazon-ml-challenge-2026`).
2. Click **Run All**.
3. The submission is written to `/kaggle/working/submission.tsv`.

The dataset path is detected automatically under `/kaggle/input`. Internet is optional: if `anyascii` can't be
installed, a built-in transliterator is used instead.

| Step | What happens |
|---|---|
| 0 | Install / check dependencies |
| 1 | Write the pipeline package (`entity_matching/`) from the cells below |
| 2 | Sanity-check the hardcoded noise patterns on real examples from the EDA |
| 3 | **Synthetic demo:** the v1 and v2 methods side by side on a 6k-S1 train and a 3k-S1 test that includes the unseen France |
| 4 | **Real competition data:** train with 5-fold OOF, calibrate, choose the F0.5 decision rule, predict test, write `submission.tsv` |
| 5 | Inspect the OOF scores, the chosen rule, feature importance and the submission |

### What changed in v2 (methodology)
| Component | v1 | v2 |
|---|---|---|
| Candidate search | Fixed top-40 per S1 | **Adaptive K** (keep 12, then up to 60 while score ≥ 0.25 × best) **+ reverse search**: each S2/S3 record also keeps its top-5 S1 records |
| Tuning metric | Micro F0.5 (pooled pairs) | **Macro F0.5 per S1** (the challenge metric) |
| Probabilities | Raw LightGBM | **Isotonic calibration** on out-of-fold predictions |
| Decision rule | Global threshold | Best on OOF of: threshold rule, or the **per-S1 expected-F0.5 set** (plug-in General F-measure Maximizer). Both enforce one owner per target |
| No-match S1 in training | Dropped | **Kept** (`--drop-singletons` to drop them) |
| Stage-2 sibling model | – | Available with `--stage2` (no gain in the ablation, so off by default) |

**Expected-F0.5 rule.** For one S1 with calibrated probabilities q₁ ≥ q₂ ≥ …, choose the k (0 included) that
maximises `E[F0.5 | top-k] ≈ 1.25·Σᵢ≤ₖ qᵢ / (0.25·E|T| + k)`, versus `E[F0.5 | empty] ≈ Πᵢ(1−qᵢ)·e^(−λ)`.
Here E|T| = Σ q + λ, and λ is the expected number of true links that blocking misses.

### Ablation (100k-S1 synthetic train, 50k-S1 test including France)
| Setup | Blocking recall | Test precision | Test recall | **Test macro F0.5** |
|---|---|---|---|---|
| v1 | 92.3% | 0.9984 | 0.8861 | 0.9610 |
| v2 without keeping singletons | 94.8% | 0.9963 | 0.9211 | 0.9722 |
| **v2 (default)** | **94.8%** | **0.9965** | **0.9223** | **0.9728** |
| v2 + stage 2 | 94.8% | 0.9959 | 0.9205 | 0.9715 |

These are synthetic scores. They check correctness and compare methods; they do **not** predict the leaderboard.

**Pipeline:**
1. Hardcoded normalisation.
2. Country-blocked, IDF-weighted key search in both directions, with adaptive K.
3. 73 pair, rule and context features.
4. LightGBM with GroupKFold by S1.
5. Isotonic calibration.
6. Decision rule chosen for macro F0.5, with one owner per target.

Doubtful pairs are removed from **training only**; validation scores every held-out S1."""))

cells.append(md("## Step 0 — dependencies"))
cells.append(code("""import importlib, os, subprocess, sys

def ensure(pkg, import_name=None):
    try:
        importlib.import_module(import_name or pkg)
        print(f"ok        {pkg}")
    except ImportError:
        r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg],
                           capture_output=True, text=True)
        try:
            importlib.import_module(import_name or pkg)
            print(f"installed {pkg}")
        except ImportError:
            print(f"MISSING   {pkg} (no internet?) " + ("- built-in fallback will be used" if pkg == "anyascii" else ""))

for p in ["pandas", "numpy", "scikit-learn:sklearn", "lightgbm", "rapidfuzz", "anyascii"]:
    name, _, mod = p.partition(":")
    ensure(name, mod or None)

# work in /kaggle/working on Kaggle, the current folder elsewhere
WORK = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.getcwd()
os.chdir(WORK)
os.makedirs("entity_matching", exist_ok=True)
if WORK not in sys.path:
    sys.path.insert(0, WORK)
print("working dir:", WORK)"""))

cells.append(md("""## Step 1 — the pipeline package
Each cell writes one module of `entity_matching/`. The modules are kept as real files, not inline code,
so the fork-based worker pools can share memory with the workers.

| Module | Role |
|---|---|
| `patterns.py` | **Hardcoded pattern dictionaries**: legal forms (US/IN/FR), native-script Indic legal words, DBA markers, junk regexes, OCR confusions, US/Indian/French states (incl. native script), street types, city aliases, unit labels |
| `normalize.py` | Name + address normalisers: transliteration, phonetic skeleton, OCR fold, canonical numbers |
| `utils.py` | Kaggle path detection, deadlock-safe fork pool with timeouts, RAM logging |
| `data.py` | TSV loading, parallel normalisation over unique values, cache |
| `blocking.py` | Candidate generation: hashed IDF-weighted keys, country hard block, **adaptive K + reverse search (v2)**, bounded-RAM chunks |
| `features.py` | 73 pair features, rule flags, context/rank features |
| `decision.py` | **v2:** isotonic calibration, one-owner rule, per-S1 expected-F0.5 set selection, threshold rule |
| `stage2.py` | **v2 (optional):** sibling-similarity and competition features for a stacked second model |
| `metrics.py` | Micro / macro F-beta |
| `synth.py` | Synthetic data with the same noise as the real data (used for the demo) |
| `run.py` | CLI / entry points: `kaggle`, `train`, `predict`, `synth` |"""))

for m in MODULES:
    src = (PKG / m).read_text()
    if not src.strip():
        src = '"""Business entity matching pipeline (Amazon ML Challenge 2026)."""\n'
    cells.append(code(f"%%writefile entity_matching/{m}\n" + src.rstrip("\n")))

cells.append(md("""## Step 2 — sanity check of the hardcoded patterns
These pairs are **real true matches taken from the EDA notebook**. After normalisation both sides should give the
same phonetic skeleton, whatever the script, OCR noise, brackets or legal-form position."""))
cells.append(code("""import importlib, entity_matching
for m in ["patterns", "normalize", "utils", "data", "blocking", "features", "decision", "stage2",
          "metrics", "synth", "run"]:
    importlib.reload(importlib.import_module(f"entity_matching.{m}"))

import pandas as pd
from entity_matching.normalize import normalize_name, normalize_address

pairs = [
    ("Gold Solutions", "গোল্ড সলিউশনস"),
    ("Jain Builders Private Limited", "जैन बिल्डर्स प्राइवेट लिमिटेड"),
    ("Creative Builders", "ಕ್ರಿಯೇಟಿವ್ ಬಿಲ್ಡರ್ಸ್"),
    ("Arihant Technology Limited", "Arihant टेक्नोलॉजी लिमिटेड"),
    ("Creative Intermediate LLC", "Llc Creative Intermediate"),
    ("Bright Seafood Inc", "Bright Séafood Inc"),
    ("South Academy", "5outh Academy"),
    ("Washington, Harris and Burrell", "WASHINGT0N, HARRIS AND BURRELL"),
    ("Midwest Applied, LLC", "Midwest Applied, ([LLC])"),
    ("Fort Worth Telecommunication", "Fort Worth Telecommunication L.L.C. (ID: 49166)"),
    ("Summit Ministries", "Summit Ministries lnc"),
    ("Bright Automation Enterprises", "Bright Bright Automation Enterprises"),
    ("Integrity Services (India) Limited", "*** Integrity Services (India)  Limited"),
]
rows = []
for a, b in pairs:
    na, nb = normalize_name(a), normalize_name(b)
    rows.append({"S1 name": a, "S2/S3 name": b, "S1 skeleton": na["skel"], "S2/S3 skeleton": nb["skel"],
                 "legal": f"{na['legal']} / {nb['legal']}", "match": na["skel"] == nb["skel"]})
df_names = pd.DataFrame(rows)
print(f"{df_names['match'].sum()} / {len(df_names)} real true pairs reduced to the same skeleton")
df_names"""))
cells.append(code("""addr = [
    ("##16978 Moore Rd, <NULL>, Andalusia, Alabama", "US"),
    ("16978 MOORE ROAD, ANDALUSIA, AL", "US"),
    ("PLOT NO B-78/1, ADDITIONAL MIDC ANAND NAGAR, AMBERNATH EAST, THANE, महाराष्ट्र", "India"),
    ("Thane, MH, Plot No. B-78/1, Additional Midc Anand Nagar, Ambernath East", "India"),
    ("05131 COPPER MEADOW LN, WEST JORDAN CITY, UT", "US"),
    ("H.No1-8-67/P Apiic Kamalanagar, Kushaiguda, Hyderabad, Rangareddy, Telangana", "India"),
    ("20 bis RUE jules lefebvre, Lille, Hauts-de-France", "France"),
]
pd.DataFrame([{"raw address": a, "country": c, **{k: v for k, v in normalize_address(a, c).items()
               if k in ("alpha", "nums", "numstreet", "state")}} for a, c in addr])"""))

cells.append(md("""## Step 3 — synthetic demo: v1 vs v2 methodology
This generates data in the exact competition layout, with every noise type found in the EDA:
- Indic scripts; legal-form swaps, brackets and moves; OCR digits; accents
- `***` / `(ID: …)` junk; domains; DBA aliases; repeated or shuffled words
- `NULL` placeholders, shuffled address components, abbreviations, missing addresses
- hard-negative generic names and singletons

Train has US + India. **Test adds the unseen France**, as in the competition.

Both methods are trained on the same data and scored on the same test. **v1** is the earlier notebook's settings;
**v2** is the current default. On 3k test S1 the two land within noise of each other, because the remaining errors on
this small set are mostly unresolvable (missing address plus a generic name). The 100k ablation above is where v2's
gain shows. Set `RUN_DEMO = False` to skip this step."""))
cells.append(code("""RUN_DEMO = True
V1_ARGS = ["--metric", "micro", "--top-k", "40", "--k-min", "40", "--reverse-k", "0", "--drop-singletons"]

if RUN_DEMO:
    from entity_matching.run import main
    main(["synth", "--out-dir", "synth_demo", "--n-train", "6000", "--n-test", "3000", "--seed", "0"])
    print("\\n==================== v1 methodology ====================")
    main(["kaggle", "--data-dir", "synth_demo", "--out-dir", "em_demo_v1",
          "--train-s1-sample", "0", "--no-cache"] + V1_ARGS)
    print("\\n==================== v2 methodology (default) ====================")
    main(["kaggle", "--data-dir", "synth_demo", "--out-dir", "em_demo",
          "--train-s1-sample", "0", "--no-cache"])"""))
cells.append(code("""if RUN_DEMO:
    import json
    from entity_matching.data import gt_pairs, load_split
    from entity_matching.metrics import evaluate
    s1_test, _, gt_test = load_split("synth_demo", "test")
    truth_test = gt_pairs(gt_test)
    ctry = s1_test.set_index("entity_id")["country"].astype(str)
    rows, by_c = [], []
    for label, d in [("v1", "em_demo_v1"), ("v2 (default)", "em_demo")]:
        conf = json.load(open(f"{d}/config.json"))
        sub = pd.read_csv(f"{d}/submission_test.tsv", sep="\\t", dtype=str, keep_default_na=False)
        pred = gt_pairs(sub)
        res = evaluate(pred, truth_test, s1_test["entity_id"], beta=0.5)
        rows.append({"method": label, "blocking recall (train)": conf["blocking_recall"],
                     "OOF macro F0.5": conf["oof"]["macro_fbeta"],
                     "test precision": res["micro_precision"], "test recall": res["micro_recall"],
                     "test micro F0.5": res["micro_fbeta"], "test macro F0.5": res["macro_fbeta"]})
        for c in sorted(ctry.unique()):
            m = evaluate(pred, truth_test, ctry.index[ctry == c], beta=0.5)
            by_c.append({"method": label, "country": c, "precision": m["micro_precision"],
                         "recall": m["micro_recall"], "macro F0.5": m["macro_fbeta"]})
    print("3k-S1 synthetic test (includes the unseen France):")
    display(pd.DataFrame(rows).round(4))
    display(pd.DataFrame(by_c).round(4))
    print("v2 decision rule chosen on OOF:", json.load(open("em_demo/config.json"))["rule"])"""))

cells.append(md("""## Step 4 — real competition data
This runs automatically when the dataset is attached: the code searches `/kaggle/input/**` for `train_source1.tsv`.
To point at a specific folder, set `DATA_DIR`.

- Training uses `TRAIN_S1_SAMPLE` S1 records (250k fits Kaggle's 30 GB RAM). Blocking and the
  target-competition features still use **all** S1 records.
- The v2 defaults apply automatically: adaptive K + reverse search, macro F0.5, calibration, and the best rule
  chosen on OOF. Add `"--stage2"` or `"--drop-singletons"` to `EXTRA_ARGS` to try the variants.
- The log prints elapsed time and RAM for every step, plus the **blocking recall ceiling**, which is the best
  recall any model can reach, and the OOF precision / recall / F0.5.
- Normalised data is cached in `em_model/cache/`, so a re-run skips normalisation.

Expected runtime on a 4-core Kaggle CPU notebook, extrapolated from the synthetic scale test: roughly 30–60 min."""))
cells.append(code("""from pathlib import Path

DATA_DIR = None            # e.g. "/kaggle/input/amazon-ml-challenge-2026/dataset"; None = auto-detect
TRAIN_S1_SAMPLE = 250_000  # 0 = all S1 records (needs more RAM)
OUT_DIR = "/kaggle/working/em_model" if Path("/kaggle/working").is_dir() else "em_model"
EXTRA_ARGS = []            # e.g. ["--stage2"] or ["--drop-singletons"]

def find_real_data():
    if DATA_DIR:
        return DATA_DIR
    root = Path("/kaggle/input")
    hits = sorted(root.rglob("train_source1.tsv")) if root.is_dir() else []
    if not hits:
        return None
    d = hits[0].parent
    return str(d.parent if d.name == "train" else d)

real_dir = find_real_data()
if real_dir is None:
    print("Competition dataset not found under /kaggle/input - attach it (or set DATA_DIR) and re-run this cell.")
else:
    print("dataset:", real_dir)
    from entity_matching.run import main
    main(["kaggle", "--data-dir", real_dir, "--out-dir", OUT_DIR,
          "--train-s1-sample", str(TRAIN_S1_SAMPLE)] + EXTRA_ARGS)"""))

cells.append(md("## Step 5 — inspect results"))
cells.append(code("""import json
cfg_path = Path(OUT_DIR) / "config.json"
if cfg_path.exists():
    conf = json.loads(cfg_path.read_text())
    print(f"decision rule: {conf['rule']}  (tuned for {conf['metric']}, stage2={conf['stage2']})")
    print(f"blocking recall ceiling: {conf['blocking_recall']:.4%}")
    display(pd.Series(conf["oof"]).to_frame("OOF").T)
    display(pd.read_csv(Path(OUT_DIR) / "feature_importance.csv", index_col=0).head(20))
    sub_path = Path("/kaggle/working/submission.tsv")
    if sub_path.exists():
        sub = pd.read_csv(sub_path, sep="\\t", dtype=str, keep_default_na=False)
        n_links = sub["matched_entity_ids"].str.count(",").add(1).where(sub["matched_entity_ids"] != "", 0)
        print(f"submission: {len(sub):,} S1 rows, {int(n_links.sum()):,} links, "
              f"{(n_links == 0).mean():.2%} S1 with no match")
        display(sub.head(10))
else:
    print("No real-data model yet - run Step 4 with the dataset attached.")"""))

nb = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"}})
out = REPO / "entity_matching_full.ipynb"
nbf.write(nb, out)
print("wrote", out, len(cells), "cells")
