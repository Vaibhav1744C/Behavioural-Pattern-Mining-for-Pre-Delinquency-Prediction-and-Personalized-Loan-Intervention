"""Shared cached readers and driver safeguards for dashboard agents."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DASHBOARD_DATA = ROOT / "dashboard" / "data"
SUBSAMPLE_IDS = ROOT / "data" / "processed" / "subsample_loan_ids.parquet"
CONTEXT_FILE = ROOT / "data" / "processed" / "sequence_context_features_subsample.parquet"
CLEAN_FILE = ROOT / "data" / "processed" / "lending_club_clean.parquet"
FUZZY_RULES_FILE = ROOT / "reports" / "surrogate_rules_xgb_behavioural_subsample.json"
CONTEXT_COLS = [
    "salary_stability_idx", "salary_delay_trend", "savings_slope",
    "savings_volatility", "cashflow_compression", "emi_stress_count", "seq_len",
]

DATE_TOKENS = (
    "issue_d", "earliest_cr_line", "sec_app_earliest_cr_line",
    "months_since_issue", "months_since_earliest", "months_since_",
    "mths_since_", "mo_sin_", "credit_history_age", "cr_line_age",
    "loan_vintage", "date_derived", "last_pymnt_d", "last_credit_pull_d",
)


def is_date_derived(feature: str) -> bool:
    key = str(feature).casefold()
    return any(token in key for token in DATE_TOKENS)


def clean_display_drivers(drivers) -> list[dict]:
    cleaned = []
    if drivers is None:
        return cleaned
    for driver in drivers:
        if isinstance(driver, dict) and not is_date_derived(driver.get("feature", "")):
            cleaned.append({
                "feature": str(driver["feature"]),
                "shap_value": float(driver.get("shap_value", 0.0)),
            })
    return cleaned


@lru_cache(maxsize=1)
def sample_borrowers() -> pd.DataFrame:
    return pd.read_parquet(DASHBOARD_DATA / "sample_borrowers.parquet")


@lru_cache(maxsize=1)
def test_predictions() -> pd.DataFrame:
    return pd.read_parquet(DASHBOARD_DATA / "test_predictions.parquet")


@lru_cache(maxsize=1)
def test_context() -> pd.DataFrame:
    """Context for test IDs only; the much larger sequence table is not read."""
    ids = pd.read_parquet(SUBSAMPLE_IDS)
    test_ids = ids.loc[ids["split"] == "test", "loan_id"].astype("int64")
    ctx = pd.read_parquet(CONTEXT_FILE)
    if "loan_id" not in ctx.columns:
        ctx = ctx.reset_index().rename(columns={"index": "loan_id"})
    return ctx.loc[ctx["loan_id"].isin(test_ids), ["loan_id"] + CONTEXT_COLS].copy()


def borrower_record(loan_id: int) -> dict:
    row = sample_borrowers().loc[sample_borrowers()["loan_id"].astype("int64") == int(loan_id)]
    if row.empty:
        raise KeyError(f"loan_id {loan_id} is not in the 200-borrower dashboard sample")
    return row.iloc[0].to_dict()


def prediction_record(loan_id: int) -> dict:
    row = test_predictions().loc[test_predictions()["loan_id"].astype("int64") == int(loan_id)]
    if row.empty:
        raise KeyError(f"loan_id {loan_id} is not in the fixed test prediction set")
    return row.iloc[0].to_dict()


@lru_cache(maxsize=1)
def _static_surrogate_inputs() -> pd.DataFrame:
    # Cleaned row position is the stable loan_id used by the fixed-ID split.
    return pd.read_parquet(CLEAN_FILE, columns=["sub_grade", "avg_cur_bal"])


@lru_cache(maxsize=1)
def fuzzy_surrogate_rules() -> list[dict]:
    import json
    with FUZZY_RULES_FILE.open(encoding="utf-8") as f:
        return json.load(f).get("rules", [])


def matching_fuzzy_rule(loan_id: int) -> tuple[str, dict] | None:
    """Match a fuzzy surrogate leaf using only non-date features and inputs."""
    borrower = borrower_record(loan_id)
    static = _static_surrogate_inputs().loc[int(loan_id)]
    sub_grade = str(static["sub_grade"])
    grade_map = {f"{grade}{number}": (ord(grade) - ord("A")) * 5 + number
                 for grade in "ABCDEFG" for number in range(1, 6)}
    term = pd.to_numeric(str(borrower.get("term", "")).replace("months", "").strip(), errors="coerce")
    int_rate = pd.to_numeric(str(borrower.get("int_rate", "")).replace("%", "").strip(), errors="coerce")
    values = {
        "salary_delay_trend": float(borrower["salary_delay_trend"]),
        "sub_grade": float(grade_map.get(sub_grade, float("nan"))),
        "int_rate": float(int_rate),
        "avg_cur_bal": float(static["avg_cur_bal"]),
        "term": float(term),
        "loan_amnt": float(borrower["loan_amnt"]),
    }
    for index, rule in enumerate(fuzzy_surrogate_rules(), start=1):
        conditions = rule.get("conditions", [])
        if not conditions or any(is_date_derived(c["feature"]) for c in conditions):
            continue
        matched = True
        for condition in conditions:
            feature = condition["feature"]
            if feature not in values or not pd.notna(values[feature]):
                matched = False
                break
            value, threshold, op = values[feature], float(condition["threshold"]), condition["op"]
            if (op == "<=" and value > threshold) or (op == ">" and value <= threshold):
                matched = False
                break
        if matched:
            return f"FUZZY_XGB_{index:02d}", rule
    return None
