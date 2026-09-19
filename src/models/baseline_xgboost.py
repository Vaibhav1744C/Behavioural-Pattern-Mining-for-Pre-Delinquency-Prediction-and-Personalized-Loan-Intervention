"""
baseline_xgboost.py
-------------------
Phase 2: Faithful reproduction of the XGBoost baseline from:

  Monje, Carrasco & Sanchez-Montanes (2025). "Machine Learning XAI for Early
  Loan Default Prediction". Computational Economics, 67, 4033-4062.

All design choices below are tied to explicit paper citations so deviations
are visible and reportable.

Corrections incorporated (per code review):
  1. 70/30 train/test split  — paper Section 4.4 states this explicitly
  2. Mixed encoding scheme   — ordinal cols label-encoded, nominals one-hot
                               (paper Section 4.5; ordinality preserved for
                               grade/sub_grade matters for surrogate rules)
  3. No imbalance correction — paper Table 4: plain XGBoost (AUC 0.731) beats
                               SMOTE (0.729), oversampling (0.662),
                               undersampling (0.663). Imbalance methods run as
                               a secondary labeled comparison only.
  4. Paper's hyperparameters — max_depth=6, min_child_weight=5, gamma=10,
                               objective='binary:logistic' (paper Section 4.4)
  5. Date columns handled    — issue_d / earliest_cr_line → numeric months;
                               not one-hot (would create one col per date)
  6. Full pipeline serialized — fitted Pipeline (imputer+encoder+model) saved,
                               not raw XGBoost model, for leak-free Phase 6
                               SHAP/surrogate use

Usage:
    python src/models/baseline_xgboost.py

Outputs:
    data/processed/X_test.parquet          — held-out features (for Phase 6)
    data/processed/y_test.parquet          — held-out labels   (for Phase 6)
    src/models/xgb_baseline_pipeline.pkl   — full fitted pipeline
    reports/baseline_results.json          — metrics vs paper targets
"""

import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from xgboost import XGBClassifier

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
CLEAN_PARQUET = ROOT / "data" / "processed" / "lending_club_clean.parquet"
X_TEST_OUT    = ROOT / "data" / "processed" / "X_test.parquet"
Y_TEST_OUT    = ROOT / "data" / "processed" / "y_test.parquet"
PIPELINE_OUT  = ROOT / "src" / "models" / "xgb_baseline_pipeline.pkl"
RESULTS_OUT   = ROOT / "reports" / "baseline_results.json"

RANDOM_STATE = 42

# ── Paper hyperparameters (Section 4.4) ───────────────────────────────────────
XGB_PARAMS = dict(
    max_depth=6,
    min_child_weight=5,
    gamma=10,
    objective="binary:logistic",
    eval_metric="auc",
    n_estimators=300,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

# ── Encoding scheme (paper Section 4.5) ───────────────────────────────────────
# Ordinal: preserve natural ordering via integer mapping
ORDINAL_MAPPINGS = {
    "grade": {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7},
    "sub_grade": {
        f"{g}{n}": (ord(g) - ord("A")) * 5 + n
        for g in "ABCDEFG"
        for n in range(1, 6)
    },
    "emp_length": {
        "< 1 year": 0, "1 year": 1, "2 years": 2, "3 years": 3,
        "4 years": 4,  "5 years": 5, "6 years": 6, "7 years": 7,
        "8 years": 8,  "9 years": 9, "10+ years": 10,
    },
}

# Nominal: one-hot encoded
NOMINAL_COLS = [
    "home_ownership", "verification_status", "purpose",
    "application_type", "initial_list_status", "pymnt_plan",
    "addr_state", "term",
]

# Date columns: convert to numeric (months since a reference date)
DATE_COLS = ["issue_d", "earliest_cr_line"]
DATE_REFERENCE = pd.Timestamp("2007-01-01")  # earliest loan in the dataset


# ── Helper functions ──────────────────────────────────────────────────────────

def months_since(series: pd.Series, ref: pd.Timestamp) -> pd.Series:
    """Convert a month-year string column (e.g. 'Jan-2015') to months since ref."""
    parsed = pd.to_datetime(series, format="%b-%Y", errors="coerce")
    delta = (parsed.dt.year - ref.year) * 12 + (parsed.dt.month - ref.month)
    return delta.astype("float64")


def apply_ordinal_mappings(df: pd.DataFrame) -> pd.DataFrame:
    """Label-encode ordinal columns in-place, unknown values → NaN."""
    df = df.copy()
    for col, mapping in ORDINAL_MAPPINGS.items():
        if col in df.columns:
            df[col] = df[col].map(mapping).astype("float64")
    return df


def convert_date_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Replace date string columns with months-since-reference numeric cols."""
    df = df.copy()
    for col in DATE_COLS:
        if col in df.columns:
            df[col] = months_since(df[col], DATE_REFERENCE)
    return df


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Build a ColumnTransformer for the remaining columns after ordinal encoding
    and date conversion have already been applied.
    Numeric cols: median imputation.
    Nominal cols: constant imputation ('missing') + one-hot encoding.
    """
    # After ordinal + date conversion, these are the remaining column groups
    nominal_present = [c for c in NOMINAL_COLS if c in X.columns]
    numeric_cols = [
        c for c in X.columns
        if c not in nominal_present
    ]

    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
    ])

    nominal_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("nom", nominal_pipe, nominal_present),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def evaluate(y_true, y_pred, y_prob, label: str) -> dict:
    """Return a metrics dict for one model run."""
    return {
        "label": label,
        "auc_roc":   round(float(roc_auc_score(y_true, y_prob)), 4),
        "f1":        round(float(f1_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred)), 4),
        "recall":    round(float(recall_score(y_true, y_pred)), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 64)
    print("Phase 2 — Baseline XGBoost Reproduction")
    print("=" * 64)

    # ── Load ──────────────────────────────────────────────────────────────────
    print("\nLoading cleaned parquet...")
    df = pd.read_parquet(CLEAN_PARQUET)
    print(f"  shape: {df.shape}")

    # ── Target ────────────────────────────────────────────────────────────────
    y = df["early_default"].astype("int8")
    X = df.drop(columns=["early_default"])

    # ── Pre-encode ordinals and dates (outside pipeline — simple map ops) ─────
    print("\nApplying ordinal encoding and date conversion...")
    X = apply_ordinal_mappings(X)
    X = convert_date_cols(X)

    # ── 70 / 30 split (paper Section 4.4) ────────────────────────────────────
    print("Splitting 70/30 (stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y
    )
    print(f"  train: {X_train.shape[0]:,}  |  test: {X_test.shape[0]:,}")
    print(f"  default rate — train: {y_train.mean():.2%}  test: {y_test.mean():.2%}")

    # ── Build preprocessor ────────────────────────────────────────────────────
    preprocessor = build_preprocessor(X_train)

    # ── PRIMARY: vanilla XGBoost — paper's best (Table 4, AUC 0.731) ─────────
    print("\n[1/4] Training primary baseline (vanilla XGBoost — no imbalance correction)...")
    pipeline_vanilla = Pipeline([
        ("prep", preprocessor),
        ("clf",  XGBClassifier(**XGB_PARAMS)),
    ])
    pipeline_vanilla.fit(X_train, y_train)
    prob_vanilla = pipeline_vanilla.predict_proba(X_test)[:, 1]
    pred_vanilla = (prob_vanilla >= 0.5).astype(int)
    metrics_vanilla = evaluate(y_test, pred_vanilla, prob_vanilla,
                               "XGBoost — vanilla (paper primary)")
    print(f"  AUC-ROC : {metrics_vanilla['auc_roc']}  (paper reports 0.731)")
    print(f"  F1      : {metrics_vanilla['f1']}")

    # ── SECONDARY comparisons (paper Table 4) ─────────────────────────────────
    results = [metrics_vanilla]

    # scale_pos_weight equivalent to oversampling minority
    print("\n[2/4] Training secondary: XGBoost + scale_pos_weight (oversampling proxy)...")
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    spw_params = {**XGB_PARAMS, "scale_pos_weight": neg / pos}
    pipeline_spw = Pipeline([
        ("prep", build_preprocessor(X_train)),
        ("clf",  XGBClassifier(**spw_params)),
    ])
    pipeline_spw.fit(X_train, y_train)
    prob_spw = pipeline_spw.predict_proba(X_test)[:, 1]
    pred_spw = (prob_spw >= 0.5).astype(int)
    metrics_spw = evaluate(y_test, pred_spw, prob_spw,
                           "XGBoost + scale_pos_weight")
    print(f"  AUC-ROC : {metrics_spw['auc_roc']}  (paper SMOTE: 0.729, oversampling: 0.662)")
    results.append(metrics_spw)

    # ── Save test set for Phase 6 (SHAP / surrogate) ─────────────────────────
    print("\n[3/4] Saving held-out test set for Phase 6...")
    X_test.to_parquet(X_TEST_OUT, index=False)
    pd.DataFrame({"early_default": y_test}).reset_index(drop=True).to_parquet(
        Y_TEST_OUT, index=False
    )
    print(f"  X_test → {X_TEST_OUT.relative_to(ROOT)}")
    print(f"  y_test → {Y_TEST_OUT.relative_to(ROOT)}")

    # ── Serialize full pipeline (vanilla — primary) ───────────────────────────
    print("\n[4/4] Serializing full pipeline...")
    with open(PIPELINE_OUT, "wb") as f:
        pickle.dump(pipeline_vanilla, f)
    print(f"  saved → {PIPELINE_OUT.relative_to(ROOT)}")

    # ── Save results ──────────────────────────────────────────────────────────
    output = {
        "paper_target_auc": 0.731,
        "paper_split": "70/30",
        "our_split": "70/30",
        "results": results,
    }
    RESULTS_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_OUT, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults → {RESULTS_OUT.relative_to(ROOT)}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("RESULTS SUMMARY")
    print("=" * 64)
    for m in results:
        print(f"\n  {m['label']}")
        print(f"    AUC-ROC  : {m['auc_roc']}")
        print(f"    F1       : {m['f1']}")
        print(f"    Precision: {m['precision']}")
        print(f"    Recall   : {m['recall']}")
    print(f"\n  Paper target AUC: 0.731")
    print("=" * 64)


if __name__ == "__main__":
    main()
