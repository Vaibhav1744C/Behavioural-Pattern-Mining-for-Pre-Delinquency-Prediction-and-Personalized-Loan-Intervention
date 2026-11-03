"""Build dashboard-ready files from the completed fixed-ID subsample artifacts.

This script performs inference only. It does not fit models or change existing
artifacts. Sequence rows are scanned through a PyArrow loan_id filter.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import DATA_ROOT
OUT = ROOT / "dashboard" / "data"
IDS_PATH = DATA_ROOT / "processed" / "subsample_loan_ids.parquet"
CLEAN_PATH = DATA_ROOT / "processed" / "lending_club_clean.parquet"
CONTEXT_PATH = DATA_ROOT / "processed" / "sequence_context_features_subsample.parquet"
SEQUENCE_PATH = DATA_ROOT / "synthetic" / "behavioural_sequences_subsample.parquet"
STATIC_MODEL_PATH = ROOT / "src" / "models" / "xgb_baseline_subsample_pipeline.pkl"
BEHAV_MODEL_PATH = ROOT / "src" / "models" / "xgb_behavioural_pipeline_subsample.pkl"
SHAP_LOCAL_PATH = ROOT / "reports" / "shap_local_samples_subsample.json"
SENSITIVITY_PATH = ROOT / "reports" / "sensitivity_sweep.json"
FIDELITY_PATH = ROOT / "reports" / "surrogate_fidelity_subsample.json"

CONTEXT_COLS = [
    "salary_stability_idx", "salary_delay_trend", "savings_slope",
    "savings_volatility", "cashflow_compression", "emi_stress_count", "seq_len",
]
BASE_BORROWER_COLS = [
    "grade", "dti", "annual_inc", "loan_amnt", "term", "int_rate",
]
TIER_LABELS = ["Very Low", "Low", "Medium", "High", "Very High"]
TIER_CENTERS = np.array([0.0, 0.25, 0.5, 0.75, 1.0])


def _require_inputs() -> None:
    required = [
        IDS_PATH, CLEAN_PATH, CONTEXT_PATH, SEQUENCE_PATH,
        STATIC_MODEL_PATH, BEHAV_MODEL_PATH, SHAP_LOCAL_PATH,
        SENSITIVITY_PATH, FIDELITY_PATH,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing dashboard inputs:\n  " + "\n  ".join(missing))
    outputs = [
        OUT / "test_predictions.parquet", OUT / "sample_borrowers.parquet",
        OUT / "sample_sequences.parquet", OUT / "sensitivity.json",
        OUT / "reference.json", OUT / "model_metrics.json",
    ]
    existing = [str(path) for path in outputs if path.exists()]
    if existing:
        raise FileExistsError("Refusing to overwrite dashboard data:\n  " + "\n  ".join(existing))


def _tier_from_percentile(percentile: float) -> str:
    memberships = np.maximum(0.0, 1.0 - np.abs(TIER_CENTERS - percentile) / 0.25)
    return TIER_LABELS[int(np.argmax(memberships))]


def _is_date_derived(feature: str) -> bool:
    name = feature.casefold()
    date_tokens = (
        "issue_d", "earliest_cr_line", "sec_app_earliest_cr_line",
        "months_since_issue", "months_since_earliest", "months_since_",
        "mths_since_", "mo_sin_", "credit_history_age", "cr_line_age",
        "loan_vintage", "date_derived", "last_pymnt_d", "last_credit_pull_d",
    )
    return any(token in name for token in date_tokens)


def _serialize_driver_list(drivers: list[dict]) -> list[dict]:
    return [
        {"feature": str(item["feature"]), "shap_value": float(item["shap_value"])}
        for item in drivers
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build dashboard data for a named run profile")
    parser.add_argument("--run", choices=("subsample", "full"), default="subsample")
    args = parser.parse_args()
    if args.run == "full":
        from build_full_dashboard_data import build
        build()
        return
    _require_inputs()
    OUT.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(ROOT / "src" / "models"))
    sys.path.insert(0, str(ROOT / "src" / "explainability"))
    from baseline_xgboost import preprocess_features as preprocess_static
    from xgboost_behavioural import preprocess_features as preprocess_behavioural

    ids = pd.read_parquet(IDS_PATH)
    if ids["loan_id"].duplicated().any():
        raise ValueError("Subsample ID artifact contains duplicate IDs")
    test_ids = ids.loc[ids["split"] == "test", "loan_id"].astype("int64").tolist()
    if len(test_ids) != 60_000:
        raise ValueError(f"Expected 60,000 test IDs; found {len(test_ids):,}")

    clean = pd.read_parquet(CLEAN_PATH)
    if "loan_id" not in clean.columns:
        clean = clean.reset_index().rename(columns={"index": "loan_id"})
    clean["loan_id"] = clean["loan_id"].astype("int64")
    test = clean.loc[clean["loan_id"].isin(test_ids)].copy()
    test = test.set_index("loan_id").loc[test_ids].reset_index()
    if len(test) != 60_000:
        raise ValueError(f"Test ID join returned {len(test):,} rows")
    y_true = test["early_default"].astype("int8").to_numpy()

    X_static = preprocess_static(test.drop(columns=["loan_id", "early_default"]))
    context = pd.read_parquet(CONTEXT_PATH)
    if "loan_id" not in context.columns:
        context = context.reset_index().rename(columns={"index": "loan_id"})
    context["loan_id"] = context["loan_id"].astype("int64")
    test_context = context.set_index("loan_id").loc[test_ids].reset_index()
    X_behav_raw = test.merge(
        test_context[["loan_id"] + CONTEXT_COLS], on="loan_id", how="left", validate="one_to_one"
    )
    if X_behav_raw[CONTEXT_COLS].isna().any().any():
        raise ValueError("Missing context feature values for test borrowers")
    X_behav = preprocess_behavioural(
        X_behav_raw.drop(columns=["loan_id", "early_default"])
    )

    with STATIC_MODEL_PATH.open("rb") as f:
        static_pipeline = pickle.load(f)
    with BEHAV_MODEL_PATH.open("rb") as f:
        behavioural_pipeline = pickle.load(f)
    p_static = static_pipeline.predict_proba(X_static)[:, 1]
    p_behav = behavioural_pipeline.predict_proba(X_behav)[:, 1]

    static_auc = float(roc_auc_score(y_true, p_static))
    static_ap = float(average_precision_score(y_true, p_static))
    behavioural_auc = float(roc_auc_score(y_true, p_behav))
    behavioural_ap = float(average_precision_score(y_true, p_behav))
    prediction = pd.DataFrame({
        "loan_id": test_ids, "y_true": y_true,
        "p_static": p_static, "p_behav": p_behav,
    })
    percentile = prediction["p_behav"].rank(method="average", pct=True)
    prediction["behavioural_percentile"] = percentile
    prediction["risk_tier"] = percentile.map(_tier_from_percentile)
    prediction.to_parquet(OUT / "test_predictions.parquet", index=False)

    # Stratified 100/100 sample, constrained to the fixed 60k test IDs.
    sample_labels = test[["loan_id", "early_default"]].merge(
        prediction[["loan_id", "risk_tier", "behavioural_percentile"]],
        on="loan_id", validate="one_to_one",
    )
    sample = pd.concat([
        sample_labels.loc[sample_labels["early_default"] == 1].sample(n=100, random_state=42),
        sample_labels.loc[sample_labels["early_default"] == 0].sample(n=100, random_state=42),
    ], ignore_index=True)
    sample_ids = sample["loan_id"].astype("int64").tolist()

    # Preserve the prior raw top-3 explanation and compute raw behavioral SHAP top-5
    # for this dashboard sample using the same TreeExplainer method.
    with SHAP_LOCAL_PATH.open(encoding="utf-8") as f:
        prior_local = {int(row["loan_id"]): row for row in json.load(f)}
    import shap
    from shap_explainer import shap_compatible_booster
    fitted_prep = behavioural_pipeline.named_steps["prep"]
    test_id_to_row = {int(lid): i for i, lid in enumerate(test_ids)}
    sample_positions = [test_id_to_row[lid] for lid in sample_ids]
    X_shap = fitted_prep.transform(X_behav.iloc[sample_positions])
    transformed_names = list(fitted_prep.get_feature_names_out())
    explainer = shap.TreeExplainer(shap_compatible_booster(behavioural_pipeline.named_steps["clf"]))
    shap_values = np.asarray(explainer.shap_values(X_shap))
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, -1]
    shap_row_by_id = {int(lid): i for i, lid in enumerate(sample_ids)}

    sample_detail = test.loc[test["loan_id"].isin(sample_ids), ["loan_id"] + BASE_BORROWER_COLS]
    sample_detail = sample_detail.merge(
        test_context[["loan_id"] + CONTEXT_COLS], on="loan_id", how="left", validate="one_to_one"
    ).merge(sample[["loan_id", "risk_tier", "behavioural_percentile"]],
            on="loan_id", how="left", validate="one_to_one")
    raw_lists, display_lists = [], []
    for lid in sample_detail["loan_id"].astype("int64"):
        ranked = sorted(
            zip(transformed_names, shap_values[shap_row_by_id[int(lid)]].tolist()),
            key=lambda pair: abs(pair[1]), reverse=True,
        )
        top_raw = [{"feature": name, "shap_value": float(value)} for name, value in ranked[:5]]
        display = [{"feature": name, "shap_value": float(value)}
                   for name, value in ranked if not _is_date_derived(name)][:5]
        raw_lists.append(_serialize_driver_list(top_raw))
        display_lists.append(_serialize_driver_list(display))
    sample_detail["shap_top5_raw"] = raw_lists
    sample_detail["display_drivers"] = display_lists
    sample_detail["y_true"] = sample_detail["loan_id"].map(
        dict(zip(test["loan_id"], y_true))
    ).astype("int8")
    sample_detail["p_static"] = sample_detail["loan_id"].map(dict(zip(test_ids, p_static)))
    sample_detail["p_behav"] = sample_detail["loan_id"].map(dict(zip(test_ids, p_behav)))
    sample_detail.to_parquet(OUT / "sample_borrowers.parquet", index=False)

    # Arrow dataset filter + bounded record batches; never read all sequence rows.
    sequence_dataset = ds.dataset(SEQUENCE_PATH, format="parquet")
    scanner = sequence_dataset.scanner(
        columns=["loan_id", "month", "salary_delay_days", "account_balance",
                 "savings_balance", "discretionary_spend", "emi_status"],
        filter=ds.field("loan_id").isin(sample_ids),
        batch_size=65_536,
    )
    sequence_batches = [batch for batch in scanner.to_batches()]
    sequence_table = __import__("pyarrow").Table.from_batches(sequence_batches)
    sample_sequences = sequence_table.to_pandas().sort_values(["loan_id", "month"])
    if sample_sequences["loan_id"].nunique() != 200:
        raise ValueError("Filtered sequences did not return all 200 sampled borrowers")
    if sample_sequences.groupby("loan_id").size().ne(24).any():
        raise ValueError("At least one sampled borrower does not have exactly 24 sequence months")
    pq.write_table(sequence_table, OUT / "sample_sequences.parquet")

    # Convert the prior sensitivity report to lift-by-gap rows and document seeds.
    with SENSITIVITY_PATH.open(encoding="utf-8") as f:
        sensitivity = json.load(f)
    sensitivity_rows = []
    for row in sensitivity["rows"]:
        if row["run"] == "static_subsample":
            continue
        gap = round(float(row["p_shock_default"]) - float(row["p_shock_nondefault"]), 2)
        sensitivity_rows.append({
            "gap": gap,
            "p_shock_default": row["p_shock_default"],
            "p_shock_nondefault": row["p_shock_nondefault"],
            "seed": row["seed"],
            "roc_auc": row["roc_auc"],
            "pr_auc": row["pr_auc"],
            "run": row["run"],
        })
    with (OUT / "sensitivity.json").open("x", encoding="utf-8") as f:
        json.dump({
            "scope": "200k-borrower subsample, 60k test",
            "seed_note": "The 0.05, 0.11 and 0.15 gap runs used seed 43.",
            "rows": sorted(sensitivity_rows, key=lambda row: row["gap"]),
        }, f, indent=2)

    with (OUT / "reference.json").open("x", encoding="utf-8") as f:
        json.dump({
            "scope": "Full-run reference (not subsample metrics)",
            "static_roc_auc": 0.7345,
            "behavioural_roc_auc": 0.7698,
            "behavioural_roc_auc_configurable": True,
            "behavioural_alternate_reported_value": 0.7686,
            "lstm_roc_auc": 0.6379,
        }, f, indent=2)

    with (ROOT / "reports" / "xgboost_behavioural_results_null.json").open(encoding="utf-8") as f:
        null_metrics = json.load(f)["results"][0]
    with (ROOT / "reports" / "surrogate_fidelity_subsample.json").open(encoding="utf-8") as f:
        surrogate_r2 = float(json.load(f)["xgb_behavioural"])
    with (OUT / "model_metrics.json").open("x", encoding="utf-8") as f:
        json.dump({
            "scope": "200k-borrower subsample, 60k test",
            "static": {"label": "Static XGBoost", "auc_roc": static_auc, "pr_auc": static_ap},
            "behavioural_main": {"label": "Behavioral XGBoost", "auc_roc": behavioural_auc, "pr_auc": behavioural_ap},
            "null": {"label": "Null-control behavioral XGBoost", "auc_roc": null_metrics["auc_roc"], "pr_auc": null_metrics["pr_auc"]},
            "surrogate_r2": surrogate_r2,
        }, f, indent=2)

    print(f"Test rows: {len(prediction):,}")
    print(f"Static ROC-AUC: {static_auc:.4f}; PR-AUC: {static_ap:.4f}")
    print(f"Behavioural ROC-AUC: {behavioural_auc:.4f}; PR-AUC: {behavioural_ap:.4f}")
    print(f"Sample rows: {len(sample_detail):,} (100 defaulters, 100 non-defaulters)")
    print(f"Filtered sequence rows: {len(sample_sequences):,}; borrowers: {sample_sequences['loan_id'].nunique()}")
    if len(prediction) != 60_000 or round(static_auc, 4) != 0.7254 or round(behavioural_auc, 4) != 0.7625:
        raise RuntimeError("Dashboard prediction sanity check failed; do not proceed to agents/pages")


if __name__ == "__main__":
    main()
