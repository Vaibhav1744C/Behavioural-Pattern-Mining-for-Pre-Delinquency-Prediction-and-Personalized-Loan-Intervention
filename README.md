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

## Repo structure

```
data/
  raw/            Lending Club CSVs (NOT committed — see .gitignore)
  synthetic/      Generated behavioural sequences (NOT committed if large)
  processed/      Cleaned/merged feature tables ready for modelling
notebooks/        Exploratory work — one notebook per person is fine here,
                   move stable code into src/ once it works
src/
  data_generation/  the synthetic behavioural generator
  features/         behavioural + static feature engineering
  models/           baseline XGBoost + temporal (LSTM/TFT) models
  explainability/    surrogate tree, fuzzy linguistic, SHAP
  intervention/       rule-based recommendation layer
reports/           write-ups, the synopsis, figures for the paper
dashboard/          the analyst-facing UI
```

## Getting Started (for teammates cloning this repo)

### 1. Clone & install dependencies
```bash
git clone https://github.com/Vaibhav1744C/Behavioural-Pattern-Mining-for-Pre-Delinquency-Prediction-and-Personalized-Loan-Intervention.git
cd Behavioural-Pattern-Mining-for-Pre-Delinquency-Prediction-and-Personalized-Loan-Intervention

python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Get the dataset (NOT in the repo — download it yourself)

The raw dataset is too large to commit. Download it manually:

1. Go to: https://www.kaggle.com/datasets/ethon0426/lending-club-20072020q1
2. Download and extract the zip — you need these two files:
   - `Loan_status_2007-2020Q3.gzip`
   - `LCDataDictionary.xlsx`
3. Place both files in `data/raw/`

See `data/raw/README.md` for full details.

### 3. Convert raw data to parquet (do this once)
```bash
pip install pyarrow openpyxl   # if not already installed
python src/data_generation/load_raw.py
```
This reads the raw CSV in chunks and saves `data/raw/lending_club_raw.parquet`.
Takes a few minutes — the file is several GB.

### 4. Run Phase 1 cleaning
```bash
python src/data_generation/clean_lending_club.py
```
Outputs:
- `data/processed/lending_club_clean.parquet` — cleaned dataset ready for modelling
- `reports/cleaning_audit.csv` — log of every dropped column with reason

### 5. Check progress
Read `SUMMARY.md` for a full up-to-date picture of what's done and what's next.

## Team

| Area | Owner | Status |
|---|---|---|
| Baseline reproduction (XGBoost + surrogate) | TBD | not started |
| Synthetic behavioural generator | TBD | not started |
| Feature engineering | TBD | not started |
| Temporal model (LSTM + attention) | TBD | not started |
| Explainability (surrogate + SHAP) | TBD | not started |
| Intervention layer + dashboard | TBD | not started |

Fill this in during your Phase 0 kickoff meeting — see `PHASE0_CHECKLIST.md`.
