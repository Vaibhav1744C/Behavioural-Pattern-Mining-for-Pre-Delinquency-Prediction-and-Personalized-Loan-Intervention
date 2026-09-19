# Project Progress Summary

> Keep this file updated as each phase completes. Teammates: read this first
> before diving into any code.

---

## Current Status — Phase 1 (Data Ingestion & Cleaning) — In Progress

---

## What Has Been Done

### Phase 0 — Scaffold & Setup ✅

- Repo created and pushed to GitHub:
  `https://github.com/Vaibhav1744C/Behavioural-Pattern-Mining-for-Pre-Delinquency-Prediction-and-Personalized-Loan-Intervention.git`
- Project directory structure set up (`data/`, `src/`, `notebooks/`, `reports/`, `dashboard/`)
- `requirements.txt` written with all core dependencies (pandas, numpy, scikit-learn, xgboost, torch, pytorch-lightning, shap, scikit-fuzzy, kagglehub, etc.)
- `README.md` and `PHASE0_CHECKLIST.md` in place

---

### Phase 1 — Data Ingestion & Cleaning 🔄 (in progress)

#### Dataset
- Source: Kaggle — `ethon0426/lending-club-20072020q1`
- Two files manually downloaded and placed in `data/raw/` (NOT committed to git):
  - `Loan_status_2007-2020Q3.gzip` — main dataset (~2.9M rows, 141 columns, plain CSV despite the `.gzip` extension)
  - `LCDataDictionary.xlsx` — column reference dictionary

#### Scripts written (both in `src/data_generation/`)

| Script | Purpose | Status |
|---|---|---|
| `load_raw.py` | Reads the raw CSV in 100k-row chunks, casts to a fixed Arrow schema, saves `data/raw/lending_club_raw.parquet` | ✅ Ready to run |
| `clean_lending_club.py` | Phase 1 cleaning — filters to resolved loans, builds `early_default` target, drops leaky + sparse columns, saves `data/processed/lending_club_clean.parquet` | ✅ Ready (run after load_raw.py) |

#### What `clean_lending_club.py` does (replicating the base paper)
1. **Target construction** — keeps only loans with a known outcome:
   - `Charged Off` → `early_default = 1`
   - `Late (31-120 days)` → `early_default = 1`
   - `Fully Paid` → `early_default = 0`
   - Excludes: Current, In Grace Period, <30 days late (unresolved)
   - Paper reports 806,161 loans, 20.20% default rate — we expect close to this
2. **Leaky column removal** — drops ~35 post-origination fields:
   - Repayment activity (`total_pymnt`, `recoveries`, `last_pymnt_d`, etc.)
   - Post-origination credit data (`last_fico_range_*`, `last_credit_pull_d`)
   - Hardship programme fields (only populated for already-distressed loans)
   - Debt settlement fields (by definition post-default)
   - Identifiers and free text (`id`, `url`, `desc`, `emp_title`, etc.)
3. **Sparse column removal** — drops columns with >40% missing values
4. **Audit trail** — every dropped column logged with reason to `reports/cleaning_audit.csv`
5. **Output** — `data/processed/lending_club_clean.parquet` (target: ~57 columns per base paper)

---

## What's Next

| Phase | Description | Status |
|---|---|---|
| Phase 1 (finish) | Run `load_raw.py` then `clean_lending_club.py`, verify row/column counts match base paper | 🔄 |
| Phase 2 | Baseline reproduction — XGBoost on the 57 cleaned features, replicate paper's AUC/F1 | ⏳ |
| Phase 3 | Synthetic behavioural data generator (`src/data_generation/`) | ⏳ |
| Phase 4 | Feature engineering — behavioural + static features (`src/features/`) | ⏳ |
| Phase 5 | Temporal model — LSTM + attention (`src/models/`) | ⏳ |
| Phase 6 | Explainability — SHAP + fuzzy surrogate (`src/explainability/`) | ⏳ |
| Phase 7 | Intervention layer + dashboard (`src/intervention/`, `dashboard/`) | ⏳ |

---

## Key Decisions & Notes

- Raw dataset is **not committed to git** — each teammate downloads their own copy (see README for instructions)
- The base paper (Monje et al., 2025) does not publish its final column list — our 57-column target is a reconstruction from the three criteria they state; the audit CSV documents every decision
- All numeric columns stored as `float64` (not int) to handle NaN values correctly throughout the pipeline
- Parquet format used throughout for fast I/O — never re-read the raw CSV after `load_raw.py` runs

---

## Base Paper Reference

Monje, Carrasco & Sanchez-Montanes (2025). *Machine Learning XAI for Early Loan Default Prediction*. Computational Economics, 67, 4033–4062.
