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

## Setup

1. Clone the repo, then create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate   # venv\Scripts\activate on Windows
   pip install -r requirements.txt
   ```
2. Get the dataset (see `data/raw/README.md` for the exact steps).
3. Run `notebooks/00_data_check.ipynb` to confirm your environment and data
   are set up correctly before doing anything else.

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
