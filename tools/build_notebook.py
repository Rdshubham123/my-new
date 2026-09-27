import nbformat as nbf
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PKG = REPO / "entity_matching"
MODULES = ["__init__.py", "patterns.py", "normalize.py", "utils.py", "data.py",
           "blocking.py", "features.py", "metrics.py", "synth.py", "run.py"]

md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell
cells = []

cells.append(md("""# Business entity matching — Amazon ML Challenge 2026 (S1 → S2/S3)

A self-contained, precision-first entity-resolution notebook built from the findings of `dataesplore.ipynb`.
For every **Source-1** business it predicts which **Source-2 / Source-3** records describe the same business.
The decision rule is tuned for **F0.5**, which weights precision twice as much as recall.

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
| 3 | **Synthetic demo:** 6k-S1 train and 3k-S1 test including the unseen France. Last run: P 0.997, R 0.957, **F0.5 0.989** |
| 4 | **Real competition data:** train with 5-fold OOF, tune F0.5, predict test, write `submission.tsv` |
| 5 | Inspect the OOF scores, feature importance and submission |

**Pipeline:** hardcoded normalisation → country-blocked IDF key blocking → 73 pair/rule/context features →
LightGBM (GroupKFold by S1) → F0.5-tuned threshold + one-owner constraint.
Singletons and doubtful pairs are removed from **training only**; validation scores every held-out S1."""))

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
| `blocking.py` | Candidate generation: hashed IDF-weighted keys, country hard block, bounded-RAM chunks |
| `features.py` | 73 pair features, rule flags, context/rank features |
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
for m in ["patterns", "normalize", "utils", "data", "blocking", "features", "metrics", "synth", "run"]:
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

cells.append(md("""## Step 3 — synthetic demo (reproduces the 3k-S1 test result including France)
This generates data in the exact competition layout, with every noise type found in the EDA:
- Indic scripts; legal-form swaps, brackets and moves; OCR digits; accents
- `***` / `(ID: …)` junk; domains; DBA aliases; repeated or shuffled words
- `NULL` placeholders, shuffled address components, abbreviations, missing addresses
- hard-negative generic names and singletons

Train has US + India. **Test adds the unseen France**, as in the competition.

Last run: 3k-S1 test including France → **precision 0.997, recall 0.957, F0.5 0.989**.
These are synthetic scores. They check correctness, **not** the leaderboard. Set `RUN_DEMO = False` to skip."""))
cells.append(code("""RUN_DEMO = True

if RUN_DEMO:
    from entity_matching.run import main
    main(["synth", "--out-dir", "synth_demo", "--n-train", "6000", "--n-test", "3000", "--seed", "0"])
    main(["kaggle", "--data-dir", "synth_demo", "--out-dir", "em_demo",
          "--train-s1-sample", "0", "--no-cache"])"""))
cells.append(code("""if RUN_DEMO:
    import json
    from entity_matching.data import gt_pairs, load_split
    from entity_matching.metrics import evaluate
    conf = json.load(open("em_demo/config.json"))
    sub = pd.read_csv("em_demo/submission_test.tsv", sep="\\t", dtype=str, keep_default_na=False)
    s1_test, _, gt_test = load_split("synth_demo", "test")
    res = evaluate(gt_pairs(sub), gt_pairs(gt_test), s1_test["entity_id"], beta=0.5)
    summary = pd.DataFrame([
        {"run": "6k S1 train (OOF)", "blocking recall": conf["blocking_recall"],
         "precision": conf["oof"]["micro_precision"], "recall": conf["oof"]["micro_recall"],
         "F0.5 micro": conf["oof"]["micro_fbeta"], "F0.5 macro": conf["oof"]["macro_fbeta"]},
        {"run": "3k S1 test incl. France", "blocking recall": None,
         "precision": res["micro_precision"], "recall": res["micro_recall"],
         "F0.5 micro": res["micro_fbeta"], "F0.5 macro": res["macro_fbeta"]},
    ])
    # per-country breakdown of the test result
    by_c = []
    ctry = s1_test.set_index("entity_id")["country"].astype(str)
    for c in sorted(ctry.unique()):
        ids = ctry.index[ctry == c]
        m = evaluate(gt_pairs(sub), gt_pairs(gt_test), ids, beta=0.5)
        by_c.append({"country": c, "S1": len(ids), "precision": m["micro_precision"],
                     "recall": m["micro_recall"], "F0.5": m["micro_fbeta"]})
    display(summary.round(4))
    display(pd.DataFrame(by_c).round(4))"""))

cells.append(md("""## Step 4 — real competition data
This runs automatically when the dataset is attached: the code searches `/kaggle/input/**` for `train_source1.tsv`.
To point at a specific folder, set `DATA_DIR`.

- Training uses `TRAIN_S1_SAMPLE` S1 records (250k fits Kaggle's 30 GB RAM). Blocking and the
  target-competition features still use **all** S1 records.
- The log prints elapsed time and RAM for every step, plus the **blocking recall ceiling**, which is the best
  recall any model can reach, and the OOF precision / recall / F0.5.
- Normalised data is cached in `em_model/cache/`, so a re-run skips normalisation.

Expected runtime on a 4-core Kaggle CPU notebook, extrapolated from the synthetic scale test: roughly 30–60 min."""))
cells.append(code("""from pathlib import Path

DATA_DIR = None            # e.g. "/kaggle/input/amazon-ml-challenge-2026/dataset"; None = auto-detect
TRAIN_S1_SAMPLE = 250_000  # 0 = all S1 records (needs more RAM)
OUT_DIR = "/kaggle/working/em_model" if Path("/kaggle/working").is_dir() else "em_model"

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
          "--train-s1-sample", str(TRAIN_S1_SAMPLE)])"""))

cells.append(md("## Step 5 — inspect results"))
cells.append(code("""import json
cfg_path = Path(OUT_DIR) / "config.json"
if cfg_path.exists():
    conf = json.loads(cfg_path.read_text())
    print(f"decision rule: p >= {conf['threshold']}  alpha = {conf['alpha']}  (tuned for {conf['metric']})")
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
