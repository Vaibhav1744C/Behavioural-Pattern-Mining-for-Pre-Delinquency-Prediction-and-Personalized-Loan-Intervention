# Behavioural Pattern Mining for Explainable Pre-Delinquency Prediction

Group 4, CSE (AI), VIT — Semester 1, AY 2026-27
Guide: Dr. Sangita Maheshwar Jaybhaye

## What this project does

Extends Monje, Carrasco & Sanchez-Montanes (2025), *Machine Learning XAI for
Early Loan Default Prediction* (Computational Economics 67, 4033-4062), from a
static-snapshot XGBoost + fuzzy-surrogate model to a temporal, behavioural
model — with an explainability layer and an intervention-recommendation layer
the base paper does not attempt.

Full synopsis: see `reports/synopsis.docx` (or the FF-180 form submitted to
the department).

---

## Setup Guide for Teammates (read this first after cloning)

### Step 1 — Clone the repo

```bash
git clone https://github.com/Vaibhav1744C/Behavioural-Pattern-Mining-for-Pre-Delinquency-Prediction-and-Personalized-Loan-Intervention.git
cd Behavioural-Pattern-Mining-for-Pre-Delinquency-Prediction-and-Personalized-Loan-Intervention
```

### Step 2 — Create a virtual environment and install dependencies

```bash
python -m venv venv

# Windows:
venv\Scripts\activate

# Mac/Linux:
source venv/bin/activate

pip install -r requirements.txt
pip install pyarrow openpyxl scipy matplotlib
```

> Confirm your setup works:
> ```bash
> python -c "import xgboost, torch, shap, pandas, pyarrow; print('All good')"
> ```

---

### Step 3 — Download the raw dataset (NOT in the repo — do this yourself)

The dataset is too large to commit to git. Each teammate downloads their own copy.

1. Go to: https://www.kaggle.com/datasets/ethon0426/lending-club-20072020q1
2. Log in to Kaggle (free account) and click **Download**
3. Extract the zip. You need exactly these two files:
   - `Loan_status_2007-2020Q3.gzip`
   - `LCDataDictionary.xlsx`
4. Place **both files** into `data/raw/`

Your `data/raw/` folder should look like:
```
data/raw/
  Loan_status_2007-2020Q3.gzip   ← main dataset (~3 GB)
  LCDataDictionary.xlsx          ← column reference
  README.md                      ← already in repo
```

---

### Step 4 — Convert raw CSV to parquet (run once, takes ~5 min)

```bash
python src/data_generation/load_raw.py
```

This reads the CSV in 100k-row chunks (avoids RAM issues) and saves:
- `data/raw/lending_club_raw.parquet` (~476 MB, fast to reload)

You'll see chunk-by-chunk progress: 30 chunks × 100k rows = 2,925,493 rows total.

---

### Step 5 — Run Phase 1 cleaning (run once, takes ~2 min)

```bash
python src/data_generation/clean_lending_club.py
```

Outputs:
- `data/processed/lending_club_clean.parquet` — 1,879,234 rows × 71 columns, ready for modelling
- `reports/cleaning_audit.csv` — every dropped column with reason

Expected output:
```
early-default rate: 20.19%  (paper reports 20.20%)
columns: 142 -> 71
```

---

### Step 6 — Reproduce the XGBoost baseline (optional but recommended)

```bash
python src/models/baseline_xgboost.py
```

Trains the base paper's XGBoost on the cleaned data. Takes ~10-15 min on a laptop.

Expected output:
```
AUC-ROC: 0.7345  (paper reports 0.731)
```

Outputs:
- `src/models/xgb_baseline_pipeline.pkl` — fitted pipeline for Phase 6
- `data/processed/X_test.parquet` + `y_test.parquet` — held-out test set
- `reports/baseline_results.json` — full metrics

---

### Step 7 — Generate synthetic behavioural sequences (long — runs for hours)

> **Only one person needs to run this** and share the output files.
> If someone has already generated them, skip to Step 8.

```bash
python src/data_generation/generate_behavioural_sequences.py
```

Generates monthly transaction sequences for all 1.87M borrowers.
Logs progress to `logs/phase3a_generation.log`.

Outputs (both ~1.1 GB — not committed to git):
- `data/synthetic/behavioural_sequences.parquet` — main run (alpha=0.6)
- `data/synthetic/behavioural_sequences_alpha0.parquet` — ablation run
- `reports/generator_manifest.json`
- `reports/generator_validation_plots/` — 30 trajectory plots

---

### Step 8 — Build LSTM features (long — runs for hours, needs Step 7 first)

> **Only one person needs to run this** and share the output files.

```bash
python src/features/build_features.py
```

Transforms sequences into LSTM-ready format. Logs to `logs/phase4_features.log`.

Outputs (not committed to git — large files):
- `data/processed/lstm_sequences.parquet` — 45M rows × 9 cols (per-month features, padded to 24)
- `data/processed/sequence_context_features.parquet` — 1.87M rows × 8 cols (one per borrower)
- `reports/feature_engineering_report.json`

---

### Step 9 — Check what's been done

```bash
# Read the full progress log
# (open SUMMARY.md in your editor)
```

`SUMMARY.md` has a detailed record of every phase: what was built, what numbers were verified, what's next.

---

## Files NOT in the repo (too large — generate or download yourself)

| File | How to get it |
|---|---|
| `data/raw/Loan_status_2007-2020Q3.gzip` | Download from Kaggle (Step 3) |
| `data/raw/lending_club_raw.parquet` | Run `load_raw.py` (Step 4) |
| `data/processed/lending_club_clean.parquet` | Run `clean_lending_club.py` (Step 5) |
| `data/processed/X_test.parquet` | Run `baseline_xgboost.py` (Step 6) |
| `data/processed/y_test.parquet` | Run `baseline_xgboost.py` (Step 6) |
| `data/synthetic/behavioural_sequences.parquet` | Run `generate_behavioural_sequences.py` (Step 7) |
| `data/synthetic/behavioural_sequences_alpha0.parquet` | Run `generate_behavioural_sequences.py` (Step 7) |
| `data/processed/lstm_sequences.parquet` | Run `build_features.py` (Step 8) |
| `data/processed/sequence_context_features.parquet` | Run `build_features.py` (Step 8) |
| `src/models/xgb_baseline_pipeline.pkl` | Run `baseline_xgboost.py` (Step 6) |

> Tip: Steps 7 and 8 are slow. Coordinate with the team so only one person runs them and shares the output files via a shared drive.

---

## Repo structure

```
data/
  raw/              raw dataset + dictionary (not committed)
  processed/        cleaned + feature tables (not committed)
  synthetic/        generated sequences (not committed)
src/
  data_generation/  load_raw.py, clean_lending_club.py,
                    generate_behavioural_sequences.py
  features/         build_features.py
  models/           baseline_xgboost.py, (lstm_model.py — Phase 5)
  explainability/   (Phase 6)
  intervention/     (Phase 7)
notebooks/          exploratory work
reports/            cleaning_audit.csv, baseline_results.json,
                    generator_manifest.json, trajectory plots
dashboard/          (Phase 7)
```

---

## Current Phase Status

| Phase | Description | Status |
|---|---|---|
| Phase 1 | Data ingestion + cleaning | ✅ Complete |
| Phase 2 | Baseline XGBoost (AUC 0.7345) | ✅ Complete |
| Phase 3A | Synthetic behavioural sequence generator | ✅ Complete |
| Phase 4 | Feature engineering for LSTM | ✅ Complete |
| Phase 5 | LSTM + attention temporal model | ⏳ Next |
| Phase 6 | Explainability — SHAP + fuzzy surrogate | ⏳ |
| Phase 7 | Intervention layer + dashboard | ⏳ |

See `SUMMARY.md` for full details on each phase.

---

## Team

| Area | Owner | Status |
|---|---|---|
| Baseline reproduction (XGBoost + surrogate) | TBD | ✅ Done |
| Synthetic behavioural generator | TBD | ✅ Done |
| Feature engineering | TBD | ✅ Done |
| Temporal model (LSTM + attention) | TBD | ⏳ Next |
| Explainability (surrogate + SHAP) | TBD | ⏳ |
| Intervention layer + dashboard | TBD | ⏳ |

Fill in owners during your kickoff meeting — see `PHASE0_CHECKLIST.md`.

---

## Base Paper

Monje, Carrasco & Sanchez-Montanes (2025). *Machine Learning XAI for Early Loan Default Prediction*. Computational Economics, 67, 4033–4062.
