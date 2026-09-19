# Project Progress Summary

> Keep this file updated as each phase completes. Teammates: read this first
> before diving into any code.

---

## Current Status — Phase 1 (Data Ingestion & Cleaning) — ✅ Complete

---

## What Has Been Done

### Phase 0 — Scaffold & Setup ✅

- Repo created and pushed to GitHub:
  `https://github.com/Vaibhav1744C/Behavioural-Pattern-Mining-for-Pre-Delinquency-Prediction-and-Personalized-Loan-Intervention.git`
- Project directory structure set up (`data/`, `src/`, `notebooks/`, `reports/`, `dashboard/`)
- `requirements.txt` written with all core dependencies (pandas, numpy, scikit-learn, xgboost, torch, pytorch-lightning, shap, scikit-fuzzy, kagglehub, etc.)
- `README.md` and `PHASE0_CHECKLIST.md` in place

---

### Phase 1 — Data Ingestion & Cleaning ✅ Complete

#### Dataset
- Source: Kaggle — `ethon0426/lending-club-20072020q1`
- Two files manually downloaded and placed in `data/raw/` (NOT committed to git):
  - `Loan_status_2007-2020Q3.gzip` — main dataset (plain CSV despite the `.gzip` extension)
  - `LCDataDictionary.xlsx` — column reference dictionary (153 entries)

#### Scripts written (both in `src/data_generation/`)

| Script | Purpose | Status |
|---|---|---|
| `load_raw.py` | Reads the raw CSV in 100k-row chunks, casts to a fixed Arrow schema, saves `data/raw/lending_club_raw.parquet` | ✅ Done |
| `clean_lending_club.py` | Phase 1 cleaning — filters to resolved loans, builds `early_default` target, drops leaky + sparse columns, saves `data/processed/lending_club_clean.parquet` | ✅ Done |

#### Verified output numbers

| Metric | Our Result | Paper Reports |
|---|---|---|
| Raw rows | 2,925,493 | 2,925,493 ✅ |
| Raw columns | 142 | 141 (paper had no `Unnamed: 0` index col) ✅ |
| Rows after filtering to resolved loans | 1,879,234 | 806,161 (paper used older subset) |
| Early-default rate | **20.19%** | **20.20%** ✅ near-exact match |
| Columns after cleaning | 71 | 57 |

> **On the row count gap:** The paper used a 2007–2018 subset; our dataset runs to 2020 Q1, so we have more resolved loans. The default rate match (20.19% vs 20.20%) confirms our target construction logic is correct.

> **On the column gap (71 vs 57):** The paper never publishes its final column list. We reconstructed their 3 filtering criteria faithfully. The 14-column gap is expected and documented in `reports/cleaning_audit.csv`. Every dropped column has a logged reason. We report this gap honestly in the paper rather than force-fitting to 57.

#### What `clean_lending_club.py` does (replicating the base paper)
1. **Target construction** — keeps only loans with a known outcome:
   - `Charged Off` → `early_default = 1`
   - `Late (31-120 days)` → `early_default = 1`
   - `Fully Paid` → `early_default = 0`
   - Excludes: Current, In Grace Period, <30 days late (unresolved)
2. **Leaky column removal** — dropped 38 post-origination fields:
   - Repayment activity (`total_pymnt`, `recoveries`, `last_pymnt_d`, etc.)
   - Post-origination credit data (`last_fico_range_*`, `last_credit_pull_d`)
   - Hardship programme fields (only populated for already-distressed loans)
   - Debt settlement fields (by definition post-default)
   - Identifiers and free text (`id`, `url`, `desc`, `emp_title`, etc.)
3. **Sparse column removal** — dropped 33 columns with >40% missing values
4. **Audit trail** — every dropped column logged with reason to `reports/cleaning_audit.csv`
5. **Output** — `data/processed/lending_club_clean.parquet` — 1,879,234 rows × 71 columns

---

## What's Next

| Phase | Description | Status |
|---|---|---|
| Phase 1 | Data ingestion (`load_raw.py`) + cleaning (`clean_lending_club.py`) | ✅ Complete |
| Phase 2 | Baseline reproduction — XGBoost on the 71 cleaned features, replicate paper's AUC/F1 | ⏳ |
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
