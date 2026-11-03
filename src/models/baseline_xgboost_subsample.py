"""Retrain the static baseline on the approved fixed 200k borrower subset.

The model, preprocessing, encoding, and hyperparameters are imported directly
from baseline_xgboost.py. The fixed ID file supplies the original train/test
membership (140k train, 60k test). Outputs are new files and are never
overwritten.

Run only when approved:
    python src/models/baseline_xgboost_subsample.py
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from baseline_xgboost import (
    CLEAN_PARQUET,
    XGB_PARAMS,
    build_preprocessor,
    evaluate,
    preprocess_features,
)


ROOT = Path(__file__).resolve().parents[2]
IDS_PATH = ROOT / "data" / "processed" / "subsample_loan_ids.parquet"
PIPELINE_OUT = ROOT / "src" / "models" / "xgb_baseline_subsample_pipeline.pkl"
RESULTS_OUT = ROOT / "reports" / "baseline_subsample_results.json"


def main() -> None:
    for output_path in (PIPELINE_OUT, RESULTS_OUT):
        if output_path.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing artifact: {output_path}"
            )
    if not IDS_PATH.exists():
        raise FileNotFoundError(f"Subsample ID file not found: {IDS_PATH}")

    selected = pd.read_parquet(IDS_PATH)
    if not {"loan_id", "split"}.issubset(selected.columns):
        raise KeyError("Subsample file must contain loan_id and split columns")
    if selected["loan_id"].duplicated().any():
        raise ValueError("Subsample file contains duplicate loan_id values")
    if set(selected["split"].unique()) - {"train", "test"}:
        raise ValueError("split values must be 'train' or 'test'")

    clean = pd.read_parquet(CLEAN_PARQUET)
    if "loan_id" not in clean.columns:
        clean = clean.reset_index().rename(columns={"index": "loan_id"})
    clean["loan_id"] = clean["loan_id"].astype("int64")
    selected["loan_id"] = selected["loan_id"].astype("int64")

    selected_ids = set(selected["loan_id"])
    if not selected_ids.issubset(set(clean["loan_id"])):
        raise ValueError("Subsample file contains IDs absent from cleaned data")
    # The ID file also records early_default for stratification; use the target
    # from the cleaned table to avoid duplicate/suffixed target columns.
    data = clean.merge(
        selected[["loan_id", "split"]],
        on="loan_id",
        how="inner",
        validate="one_to_one",
    )
    if len(data) != len(selected):
        raise ValueError("Not all selected borrowers matched cleaned data")

    y = data["early_default"].astype("int8")
    X = data.drop(columns=["early_default", "loan_id", "split"])
    X = preprocess_features(X)
    train_mask = data["split"].to_numpy() == "train"
    test_mask = data["split"].to_numpy() == "test"
    if int(train_mask.sum()) != 140_000 or int(test_mask.sum()) != 60_000:
        raise ValueError(
            f"Expected 140,000 train and 60,000 test IDs; got "
            f"{train_mask.sum():,} and {test_mask.sum():,}"
        )

    X_train, X_test = X.iloc[train_mask], X.iloc[test_mask]
    y_train, y_test = y.iloc[train_mask], y.iloc[test_mask]

    pipeline = Pipeline([
        ("prep", build_preprocessor(X_train)),
        ("clf", XGBClassifier(**XGB_PARAMS)),
    ])
    pipeline.fit(X_train, y_train)
    prob = pipeline.predict_proba(X_test)[:, 1]
    pred = (prob >= 0.5).astype(int)
    vanilla_metrics = evaluate(
        y_test, pred, prob, "XGBoost vanilla — 200k fixed-ID subset"
    )
    vanilla_metrics["pr_auc"] = round(float(average_precision_score(y_test, prob)), 4)

    # Keep the original baseline's secondary scale_pos_weight comparison.
    negative_count = int((y_train == 0).sum())
    positive_count = int((y_train == 1).sum())
    weighted_params = {
        **XGB_PARAMS,
        "scale_pos_weight": negative_count / positive_count,
    }
    weighted_pipeline = Pipeline([
        ("prep", build_preprocessor(X_train)),
        ("clf", XGBClassifier(**weighted_params)),
    ])
    weighted_pipeline.fit(X_train, y_train)
    weighted_prob = weighted_pipeline.predict_proba(X_test)[:, 1]
    weighted_pred = (weighted_prob >= 0.5).astype(int)
    weighted_metrics = evaluate(
        y_test, weighted_pred, weighted_prob,
        "XGBoost scale_pos_weight — 200k fixed-ID subset",
    )
    weighted_metrics["pr_auc"] = round(
        float(average_precision_score(y_test, weighted_prob)), 4
    )

    PIPELINE_OUT.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_OUT.parent.mkdir(parents=True, exist_ok=True)
    with PIPELINE_OUT.open("xb") as f:
        pickle.dump(pipeline, f)
    result = {
        "model": "Static XGBoost — 200,000 borrower subsample",
        "train_rows": int(train_mask.sum()),
        "test_rows": int(test_mask.sum()),
        "id_file": str(IDS_PATH.relative_to(ROOT)),
        "encoding": "baseline_xgboost.preprocess_features",
        "hyperparameters": XGB_PARAMS,
        "results": [vanilla_metrics, weighted_metrics],
    }
    with RESULTS_OUT.open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"Vanilla ROC-AUC: {vanilla_metrics['auc_roc']}")
    print(f"Scale-pos-weight ROC-AUC: {weighted_metrics['auc_roc']}")
    print(f"Pipeline saved: {PIPELINE_OUT.relative_to(ROOT)}")
    print(f"Results saved: {RESULTS_OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
