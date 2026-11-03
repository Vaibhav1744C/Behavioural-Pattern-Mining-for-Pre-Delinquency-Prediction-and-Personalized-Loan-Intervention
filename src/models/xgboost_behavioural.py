"""
xgboost_behavioural.py
----------------------
Missing comparison: XGBoost trained on static features + 7 behavioural
context features (salary_stability_idx, salary_delay_trend, savings_slope,
savings_volatility, cashflow_compression, emi_stress_count, seq_len).

This is the fast, cheap evidence that behavioural signal helps a tree model
BEFORE the expensive LSTM training finishes. It answers:
  "Do hand-engineered behavioural features lift a tree model?"
  (vs. LSTM answering "do raw sequences help a temporal model?")

Uses the exact same preprocessing pipeline as Phase 2 (baseline_xgboost.py)
so the AUC is directly comparable.

Same 70/30 split, same hyperparameters, same encoding — only the feature set
changes (static 71 cols + 7 behavioural context cols = 78 cols total).

Usage:
    python src/models/xgboost_behavioural.py

Outputs:
    src/models/xgb_behavioural_pipeline.pkl
    reports/xgboost_behavioural_results.json

Requirements:
    pip install xgboost scikit-learn pandas pyarrow
"""

import json
import pickle
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).resolve().parents[2]
CLEAN_PATH    = ROOT / "data" / "processed" / "lending_club_clean.parquet"
CONTEXT_PATH  = ROOT / "data" / "processed" / "sequence_context_features.parquet"
PIPELINE_OUT  = ROOT / "src"  / "models"    / "xgb_behavioural_pipeline.pkl"
RESULTS_OUT   = ROOT / "reports"            / "xgboost_behavioural_results.json"

RANDOM_STATE  = 42

# ── Paper hyperparameters (same as Phase 2 baseline) ─────────────────────────
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
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

# ── Encoding (same as Phase 2) ────────────────────────────────────────────────
ORDINAL_MAPPINGS = {
    "grade": {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7},
    "sub_grade": {
        f"{g}{n}": (ord(g) - ord("A")) * 5 + n
        for g in "ABCDEFG" for n in range(1, 6)
    },
    "emp_length": {
        "< 1 year": 0, "1 year": 1, "2 years": 2, "3 years": 3,
        "4 years": 4,  "5 years": 5, "6 years": 6, "7 years": 7,
        "8 years": 8,  "9 years": 9, "10+ years": 10,
    },
}

NOMINAL_COLS = [
    "home_ownership", "verification_status", "purpose",
    "application_type", "initial_list_status", "pymnt_plan", "addr_state",
]

DATE_COLS      = ["issue_d", "earliest_cr_line", "sec_app_earliest_cr_line"]
DATE_REFERENCE = pd.Timestamp("2007-01-01")
PCT_COLS       = ["int_rate", "revol_util"]
TERM_COL       = "term"
DROP_COLS      = ["Unnamed: 0"]

# Behavioural context features to join (seq_len included as a feature here)
BEHAVIOURAL_COLS = [
    "salary_stability_idx", "salary_delay_trend", "savings_slope",
    "savings_volatility", "cashflow_compression", "emi_stress_count", "seq_len",
]


# ── Pre-processing helpers (reused from baseline_xgboost.py) ─────────────────

def apply_ordinal_mappings(df):
    df = df.copy()
    for col, mapping in ORDINAL_MAPPINGS.items():
        if col in df.columns:
            df[col] = df[col].map(mapping).astype("float64")
    return df

def convert_date_cols(df):
    df = df.copy()
    for col in DATE_COLS:
        if col in df.columns:
            cleaned = df[col].astype(str).replace({"None": None, "nan": None})
            parsed  = pd.to_datetime(cleaned, format="%b-%Y", errors="coerce")
            df[col] = ((parsed.dt.year  - DATE_REFERENCE.year) * 12 +
                       (parsed.dt.month - DATE_REFERENCE.month)).astype("float64")
    return df

def convert_pct_cols(df):
    df = df.copy()
    for col in PCT_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace("%","",regex=False).str.strip(),
                errors="coerce"
            )
    return df

def convert_term_col(df):
    df = df.copy()
    if TERM_COL in df.columns:
        df[TERM_COL] = pd.to_numeric(
            df[TERM_COL].astype(str).str.replace("months","",regex=False).str.strip(),
            errors="coerce"
        )
    return df

def drop_junk_cols(df):
    return df.drop(columns=[c for c in DROP_COLS if c in df.columns])

def preprocess_features(X):
    X = drop_junk_cols(X)
    X = apply_ordinal_mappings(X)
    X = convert_date_cols(X)
    X = convert_pct_cols(X)
    X = convert_term_col(X)
    return X

def build_preprocessor(X):
    nominal_present = [c for c in NOMINAL_COLS if c in X.columns]
    numeric_cols    = [c for c in X.columns if c not in nominal_present]
    numeric_pipe = Pipeline([("impute", SimpleImputer(strategy="median"))])
    nominal_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
        ("ohe",    OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("nom", nominal_pipe, nominal_present),
        ],
        remainder="drop", verbose_feature_names_out=False,
    )

def evaluate(y_true, y_pred, y_prob, label):
    return {
        "label":            label,
        "auc_roc":          round(float(roc_auc_score(y_true, y_prob)), 4),
        "pr_auc":           round(float(average_precision_score(y_true, y_prob)), 4),
        "f1":               round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "precision":        round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall":           round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train static + behavioural XGBoost")
    parser.add_argument("--loan-ids-file", type=Path, default=None,
                        help="Parquet file with fixed loan_id IDs and train/test membership")
    parser.add_argument("--context-file", type=Path, default=None,
                        help="Context parquet to use (defaults to the full or subset context file)")
    parser.add_argument("--output-tag", type=str, default=None,
                        help="Tag pipeline and results filenames, e.g. null")
    parser.add_argument("--vanilla-only", action="store_true",
                        help="Train only the vanilla model; skip scale_pos_weight comparison")
    args = parser.parse_args()
    output_tag = args.output_tag or ("subsample" if args.loan_ids_file else None)
    if output_tag and not re.fullmatch(r"[A-Za-z0-9_-]+", output_tag):
        raise ValueError("--output-tag may contain only letters, digits, '_' and '-'")
    pipeline_out = ROOT / "src" / "models" / (
        f"xgb_behavioural_pipeline_{output_tag}.pkl" if output_tag else PIPELINE_OUT.name
    )
    results_out = ROOT / "reports" / (
        f"xgboost_behavioural_results_{output_tag}.json" if output_tag else RESULTS_OUT.name
    )
    default_context = (
        f"sequence_context_features_{output_tag}.parquet"
        if output_tag else CONTEXT_PATH.name
    )
    context_path = args.context_file or (ROOT / "data" / "processed" / default_context)

    print("=" * 64)
    print("XGBoost — Static + Behavioural Context Features")
    print("=" * 64)

    # ── Load static data ──────────────────────────────────────────────────────
    print("\nLoading cleaned static data...")
    df = pd.read_parquet(CLEAN_PATH)
    print(f"  shape: {df.shape}")

    # ── Load behavioural context features ─────────────────────────────────────
    print("Loading behavioural context features...")
    ctx = pd.read_parquet(context_path)
    # loan_id may be index or column
    if "loan_id" not in ctx.columns:
        ctx = ctx.reset_index().rename(columns={"index": "loan_id"})
    ctx = ctx[["loan_id"] + BEHAVIOURAL_COLS]
    print(f"  context shape: {ctx.shape}")

    # ── Build loan_id for static data ─────────────────────────────────────────
    if "loan_id" not in df.columns:
        df = df.reset_index().rename(columns={"index": "loan_id"})

    # ── Merge static + behavioural ────────────────────────────────────────────
    merged = df.merge(ctx, on="loan_id", how="inner")
    if args.loan_ids_file:
        selected = pd.read_parquet(args.loan_ids_file)
        if not {"loan_id", "split"}.issubset(selected.columns):
            raise KeyError("--loan-ids-file must contain loan_id and split columns")
        if selected["loan_id"].duplicated().any():
            raise ValueError("--loan-ids-file contains duplicate loan_id values")
        if set(selected["split"].unique()) - {"train", "test"}:
            raise ValueError("split values in --loan-ids-file must be 'train' or 'test'")
        split_by_id = dict(zip(selected["loan_id"].astype("int64"), selected["split"]))
        merged = merged[merged["loan_id"].isin(split_by_id)].copy()
        if len(merged) != len(selected):
            raise ValueError("Not all requested IDs have both clean and context rows")
        split_labels = merged["loan_id"].map(split_by_id)
        if split_labels.isna().any():
            raise ValueError("A selected loan_id has no train/test assignment")
    else:
        split_labels = None
    print(f"  merged shape: {merged.shape}  "
          f"(inner join — {len(df) - len(merged):,} borrowers dropped — no sequences)")

    y = merged["early_default"].astype("int8")
    X = merged.drop(columns=["early_default", "loan_id"])
    print(f"  Features: {X.shape[1]}  ({df.shape[1]-2} static + {len(BEHAVIOURAL_COLS)} behavioural)")

    # ── Pre-pipeline transforms ───────────────────────────────────────────────
    print("\nApplying pre-pipeline transforms...")
    X = preprocess_features(X)

    # ── 70/30 split (same as Phase 2) ─────────────────────────────────────────
    print("Splitting 70/30 (stratified)...")
    if split_labels is None:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y
        )
    else:
        train_mask = split_labels.to_numpy() == "train"
        test_mask = split_labels.to_numpy() == "test"
        if not train_mask.any() or not test_mask.any():
            raise ValueError("Fixed-ID file must include both train and test borrowers")
        X_train, X_test = X.iloc[train_mask], X.iloc[test_mask]
        y_train, y_test = y.iloc[train_mask], y.iloc[test_mask]
    print(f"  train: {X_train.shape[0]:,}  test: {X_test.shape[0]:,}")
    print(f"  default rate — train: {y_train.mean():.2%}  test: {y_test.mean():.2%}")

    # ── Train primary (vanilla) ───────────────────────────────────────────────
    print("\n[1/2] Training XGBoost static+behavioural (vanilla)...")
    pipeline = Pipeline([
        ("prep", build_preprocessor(X_train)),
        ("clf",  XGBClassifier(**XGB_PARAMS)),
    ])
    pipeline.fit(X_train, y_train)
    prob = pipeline.predict_proba(X_test)[:, 1]
    pred = (prob >= 0.5).astype(int)
    m    = evaluate(y_test, pred, prob, "XGBoost static+behavioural (vanilla)")
    print(f"  AUC-ROC: {m['auc_roc']}  (static-only baseline: 0.7345)")
    print(f"  F1     : {m['f1']}")

    m_s = None
    if args.vanilla_only:
        print("\nScale-pos-weight comparison skipped by --vanilla-only.")
    else:
        # ── Train secondary (scale_pos_weight) ───────────────────────────────
        print("\n[2/2] Training XGBoost static+behavioural + scale_pos_weight...")
        neg, pos   = (y_train == 0).sum(), (y_train == 1).sum()
        spw_params = {**XGB_PARAMS, "scale_pos_weight": neg / pos}
        pipeline_spw = Pipeline([
            ("prep", build_preprocessor(X_train)),
            ("clf",  XGBClassifier(**spw_params)),
        ])
        pipeline_spw.fit(X_train, y_train)
        prob_s = pipeline_spw.predict_proba(X_test)[:, 1]
        pred_s = (prob_s >= 0.5).astype(int)
        m_s    = evaluate(y_test, pred_s, prob_s,
                          "XGBoost static+behavioural + scale_pos_weight")
        print(f"  AUC-ROC: {m_s['auc_roc']}")

    # ── Save pipeline (vanilla — used by shap_explainer.py) ───────────────────
    with open(pipeline_out, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"\nPipeline saved -> {pipeline_out.relative_to(ROOT)}")

    # ── Save results ──────────────────────────────────────────────────────────
    output = {
        "model":                   "XGBoost static+behavioural",
        "train_rows":              int(len(y_train)),
        "test_rows":               int(len(y_test)),
        "static_only_baseline_auc": 0.7345,
        "n_features":              X.shape[1],
        "n_static":                df.shape[1] - 2,
        "n_behavioural":           len(BEHAVIOURAL_COLS),
        "results":                 [m] if m_s is None else [m, m_s],
    }
    results_out.parent.mkdir(parents=True, exist_ok=True)
    with open(results_out, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved -> {results_out.relative_to(ROOT)}")

    print("\n" + "=" * 64)
    print("COMPARISON TABLE")
    print("=" * 64)
    print(f"  Static-only XGBoost (Phase 2)       : AUC 0.7345")
    print(f"  Static+Behavioural XGBoost (vanilla): AUC {m['auc_roc']}")
    if m_s is not None:
        print(f"  Static+Behavioural + pos_weight     : AUC {m_s['auc_roc']}")
    print("=" * 64)


if __name__ == "__main__":
    main()
