# Business entity matching — Amazon ML Challenge 2026 (S1 → S2/S3)

A precision-first entity-resolution pipeline built from the findings of
`dataesplore.ipynb`. For every Source-1 business it predicts which Source-2 and
Source-3 records describe the same business. It is tuned for **F0.5**, which
weights precision twice as much as recall.

```
entity_matching/
  patterns.py   hardcoded pattern dictionaries (legal forms, Indic legal words, DBA markers,
                junk, OCR confusions, US/India/France states, street types, city aliases, unit labels)
  normalize.py  name + address normalisers (transliteration, phonetic skeleton, OCR fold, numbers)
  data.py       TSV loading, parallel normalisation over unique values, cache
  blocking.py   candidate generation (hashed IDF-weighted keys, country hard block)
  features.py   73 pair features + rule flags + context/rank features
  metrics.py    micro / macro F-beta
  run.py        CLI: kaggle | train | predict | synth
  synth.py      synthetic data generator with the same noise (used for testing)
  utils.py      Kaggle path detection, deadlock-safe process pool, RAM logging
entity_matching_full.ipynb  self-contained notebook: all code inline, synthetic demo with outputs, real-data run
kaggle_run.ipynb  3-cell notebook that clones this repo instead
tools/build_notebook.py     regenerates the full notebook from the package sources
tests/            unit tests built from real examples in the EDA
```

## Run on Kaggle

**Easiest option:** upload `entity_matching_full.ipynb` to Kaggle, attach the dataset and choose **Run All**. It needs no clone and no internet (if `anyascii` can't be installed, a built-in transliterator is used).

Or from a notebook cell:

Attach the dataset (for example `satwiksps/amazon-ml-challenge-2026`), then run:

```python
!pip -q install rapidfuzz anyascii lightgbm
!git clone -b claude/adoring-newton-axw8an https://github.com/rdshubham123/my-new.git
%cd my-new
!python -m entity_matching.run kaggle
```

- **Paths are detected automatically.** The code searches `/kaggle/input/**` for
  `train_source1.tsv` / `test_source1.tsv` at any depth. You can override this with
  `--data-dir` or the `EM_DATA_DIR` environment variable.
- **Outputs.** Everything is written to `/kaggle/working/em_model/`: `model.txt`,
  `config.json` (tuned threshold and OOF scores), `feature_importance.csv`, and
  `submission_test.tsv`. The submission is also copied to `/kaggle/working/submission.tsv`.
- **Cache.** Normalised data is cached in `em_model/cache/`, so re-running `predict` or
  `train` skips normalisation. Pass `--no-cache` to turn this off.
- **Useful flags.** `--train-s1-sample 250000` (the default; `0` uses every S1),
  `--metric micro|macro`, `--beta 0.5`, `--keep-singletons`, `--top-k 40`, `--n-jobs N`.

### Kaggle engineering

| Concern | What the code does |
|---|---|
| Deadlocks | All parallelism goes through one `run_pool` helper. It uses a fork pool whose workers run BLAS/OpenMP single-threaded and ignore SIGINT. Every task has a timeout, so a stuck or OOM-killed worker cannot hang the notebook: the remaining tasks finish serially. LightGBM runs only in the parent process, with explicit `num_threads`. |
| RAM | Workers read fork-shared arrays by `(lo, hi)` range instead of receiving pickled copies. Each unique name/address is normalised once, and duplicates share the same string objects. Raw text columns are dropped after normalisation. `country`/`source` are categoricals and indices are int32. Blocking expands at most 5M postings per chunk. Test features are computed and scored in S1 batches, keeping only `(s, t, p)`. Fold datasets are LightGBM `subset()` views of one binned dataset. |
| Speed | Uses all available cores by default. Normalisation runs over unique values only, blocking is vectorised numpy, and there is a parquet-free pickle cache. |
| Measured | 100k S1 / 450k targets synthetic: full train and predict in about 5 min on 4 cores; blocking peaks at 1.6 GB RSS. |

## FINAL (v3): `final_submission.ipynb`, the competition notebook

Upload `final_submission.ipynb` to Kaggle, choose the **GPU T4×2** accelerator (CPU also works), attach the
dataset, set `TEAM_NAME`, and Run All. It writes:
- `/kaggle/working/output/matching_results.tsv`: upload this to the portal
- `/kaggle/working/output/candidate_pairs.tsv`
- `/kaggle/working/<TEAM_NAME>_submission.zip`: outputs, code (`src/`), README, pinned requirements, and the
  filled-in methodology

What v3 adds on top of v2 (`entity_matching/final.py`, CLI `run_final.py`):
- **Learned pruning.** Stage-1 LightGBM scores all blocked pairs, and a (rank, probability) cut is chosen on OOF
  to keep ≥ 99.8% of reachable true links. This takes candidates from ~25–45 per S1 down to **~4 per S1**, which
  helps the "smaller candidate set" criterion. `candidate_pairs.tsv` is exactly the pruned set the ensemble scores.
- **GPU ensemble** on the pruned pairs: LightGBM + XGBoost (CUDA) + CatBoost (GPU), with automatic CPU fallback.
  The blend and decision rule are chosen on OOF macro F0.5, with isotonic calibration.
- **New features:** rarity-weighted shared and conflicting address numbers, and a first house number conflict.
- **Scaling:** blocking with a single sort (no pandas dedupe), a per-record posting budget, and composite-key
  ranking. On a sampled train set only the sampled S1 records are blocked.
- **Deadlock safety:** every fork happens before CUDA is used. The notebook runs the pipeline in a fresh
  subprocess and runs a 1-minute synthetic smoke test first.

## v2 methodology (current default)

| Component | v1 | v2 |
|---|---|---|
| Candidate search | Fixed top-40 per S1 | Adaptive K (keep 12, then up to 60 while score ≥ 0.25 × best) **plus** reverse search, where each target keeps its top-5 S1 records |
| Tuning metric | Micro F0.5 | **Macro F0.5 per S1** (the metric the public write-ups describe) |
| Probabilities | Raw | Isotonic calibration on OOF predictions |
| Decision | Global threshold | Best on OOF of: threshold rule, or the per-S1 expected-F0.5 set (plug-in GFM), both with one owner per target |
| No-match S1 in training | Dropped | Kept (`--drop-singletons` restores v1 behaviour) |
| Stage-2 sibling model | – | Opt-in with `--stage2` (no gain in the ablation) |

Ablation on a 100k-S1 synthetic train, scored on a 50k-S1 test that includes France:

| Setup | Blocking recall | Test precision | Test recall | Test macro F0.5 |
|---|---|---|---|---|
| v1 | 92.3% | 0.9984 | 0.8861 | 0.9610 |
| v2 without keeping singletons | 94.8% | 0.9963 | 0.9211 | 0.9722 |
| **v2 (default)** | **94.8%** | 0.9965 | 0.9223 | **0.9728** |
| v2 + stage 2 | 94.8% | 0.9959 | 0.9205 | 0.9715 |

## Approach

### 1. Facts from the EDA used as hard constraints
- **Country is a hard block.** 100% of true links are same-country (cell 9).
- **One owner per target.** Each S2/S3 record belongs to at most one S1 (cell 8). At
  decision time every target is kept only for its highest-scoring S1.
- **France appears only in test** (15% of test S1). Nothing is learnt per country: the
  features are language-agnostic, and the French legal forms, street types and regions are
  hardcoded in `patterns.py`.

### 2. Removing noise with hardcoded patterns (`patterns.py`, `normalize.py`)

| Noise seen in the notebook | Example | Handling |
|---|---|---|
| Indic scripts (7 scripts, ~15% of India names) | `গোল্ড সলিউশনস` ↔ `Gold Solutions` | Native legal words replaced first, then `anyascii` transliteration, then a **phonetic skeleton** (`sliusns` → `slsns` = `solutions`) |
| Legal forms moved, added, dropped, bracketed, dotted | `Llc Creative…`, `([LLC])`, `L.L.P.` | Collapse dotted abbreviations, map ~60 variants to canonical forms, keep them apart from the core name, add a legal-conflict feature |
| OCR / leet digits | `5outh`, `WASHINGT0N`, `lnc`, `.c0m` | Letter-context digit repair plus a symmetric OCR-folded comparison key |
| Accents | `Séafoeod`, `Ínc` | ASCII folding |
| Junk prefixes / suffixes | `***`, `--`, `>>`, `M/s`, `(ID: 49166)`, `#98587` | Regex removal plus a `junk` flag |
| Domains | `aahanarealty.com` | TLD stripped, space-less containment feature |
| DBA / aliases | `X D.B.A. Y`, `X formerly: Y`, `X \| www.x.com` | Split into alternative names; best alternative match used as a feature |
| Repeated words, word order | `Bright Bright…`, `Kern, Zarah` | Adjacent duplicates removed; token-set, IDF and skeleton comparisons |
| Address `NULL` placeholders | `NULL`, `<NULL>`, `nan` | Removed |
| State names / codes / native script | `Texas`↔`TX`, `महाराष्ट्र`↔`MH` | All mapped to one code (US, IN, FR) |
| Street and unit abbreviations | `Road`↔`Rd`, `H.No`, `Plot No`, `Unit Unit` | USPS / Indian / French tables; unit labels dropped, numbers kept |
| Number formatting | `05131`, `4Nd`, `No1-8-67/P` | Canonical numbers (leading zeros removed, ordinals stripped, labels unglued), with compound numbers kept intact |
| City renames | Bombay/Mumbai, Bengaluru/Bangalore, Gurugram/Gurgaon | Alias table |

### 3. Blocking (candidate generation)
Every record emits hashed keys:

- name skeleton tokens and bigrams
- the first 8 characters of the space-less name
- the full name skeleton
- address words and canonical numbers
- number + following street word
- **name token × number** (stays rare even for generic names like "Pediatric Group")

For each S1, the targets sharing its rarest keys are scored by IDF-weighted overlap. The
pipeline keeps the top 40 overall, plus the top 10 by name evidence and the top 10 by
address evidence. That covers the cases the notebook found where the name was replaced
entirely and only the address matches.

### 4. Features and model
- **73 features**: fuzzy ratios over five name views; IDF-weighted token overlap; alias,
  domain and acronym features; legal-form agreement; address fuzzy and IDF overlap;
  number agreement (exact, compound, parts, house number, number+street); state and postal
  agreement.
- **Hardcoded rule flags**: for example `exact core name + shared number`, or
  `domain contained + shared number`.
- **Context features**: the pair's rank and gap among the S1's candidates, target
  competition across all S1 records, and sibling counts (an S1 usually has several S2/S3
  duplicates).
- **Model**: LightGBM with 5-fold **GroupKFold by S1**, so there is no leakage between
  pairs of the same S1.

### 5. Training-set cleaning (your request)
These records are removed **from the training folds only**:

- **Singleton S1 records** (no true match) are dropped by default. Use
  `--keep-singletons` to keep them; it is worth comparing both.
- **Doubtful pairs** are dropped:
  - *weak positives*: true links with no name, address or number evidence at all, which
    cannot be learnt.
  - *twin negatives*: non-links whose name skeleton and address are identical to the S1.
    These are label noise or true duplicates.

**Validation always scores every held-out S1**, singletons and doubtful pairs included.
True links that blocking misses count as false negatives, so the reported F0.5 is honest
rather than inflated by the cleaning.

### 6. Decision rule tuned for F0.5
`p ≥ threshold`, then one owner per target, then `p ≥ alpha · max p of the S1`. Threshold
and alpha are grid-searched on out-of-fold predictions to maximise micro F0.5 (or macro,
with `--metric macro`).

## About "F0.5 = 1"
A perfect score cannot be guaranteed on this data. The notebook shows true pairs where both
the name and the address carry no shared evidence. Examples are `Suryaika Foundation Group`
↔ `Viovera | www.viovera.com`, and names replaced entirely (`Kelopyrahalo`) whose target
address is missing. No rule or model can recover those without leaking labels.

The pipeline therefore reports:

- the **blocking recall ceiling**, which is the best recall any model can reach, and
- the **OOF precision/recall/F0.5**.

Because it is precision-first, precision stays around 0.996–0.998 on synthetic data.

Synthetic results, which check correctness only and do **not** predict leaderboard scores:

| Data | Blocking recall | Precision | Recall | F0.5 (micro) |
|---|---|---|---|---|
| 6k S1 train, OOF | 97.0% | 0.996 | 0.956 | 0.988 |
| 3k S1 test incl. France | – | 0.997 | 0.957 | 0.989 |
| 100k S1 train, OOF | 92.3% | 0.996 | – | 0.977 |
| 50k S1 test incl. France | – | 0.998 | 0.889 | 0.974 |

At 100k the synthetic vocabulary is tiny (about 400 name combinations), so it is harder than
real data. The notebook shows real name tokens have a median frequency of 1.

## Research and open-source work this draws on
- **Fellegi–Sunter record linkage** / **Splink** (UK MoJ): blocking with
  frequency-adjusted (IDF) agreement weights.
- **Dedupe** (dedupeio): learned blocking, and pairwise classification with many string
  comparators.
- **Magellan / py_entitymatching** (Konda et al., VLDB 2016): the blocking → feature
  generation → ML matcher workflow, with blocking-recall debugging.
- **DeepMatcher** (Mudgal et al., SIGMOD 2018) and **Ditto** (Li et al., VLDB 2021):
  deep and pre-trained-LM matchers. Ditto's domain-knowledge injection (normalising spans
  such as numbers and legal forms) is what `patterns.py` does in hardcoded form.
- **JedAI** (Papadakis et al.): token blocking with meta-blocking, i.e. pruning pairs by
  shared-key weights. That is the top-K by IDF score used here.
- **RapidFuzz** (string metrics), **anyascii** (transliteration), **cleanco**
  (legal-form stripping), **libpostal / usaddress** (address parsing, whose component
  vocabularies inspired the tables).
- Transliteration-robust matching for Indic names: phonetic keys in the Soundex/Metaphone
  family, adapted into the consonant skeleton.

**Possible next step if more accuracy is needed:** add a multilingual sentence-embedding
similarity (for example LaBSE or `paraphrase-multilingual-MiniLM`) as an extra feature and
as an ANN blocker via FAISS. This helps the remaining cross-script cases, at the cost of
GPU time.

## Local usage
```bash
pip install -r requirements.txt
python -m entity_matching.run synth --out-dir synth          # synthetic data, same layout
python -m entity_matching.run kaggle --data-dir synth --out-dir model --train-s1-sample 0
python -m pytest -q tests
```
