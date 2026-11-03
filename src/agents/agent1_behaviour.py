"""Agent 1: flag high/low-tail behavioral context values by test percentile."""

from __future__ import annotations

import numpy as np

from .common import CONTEXT_COLS, borrower_record, test_context

TAILS = {
    "salary_stability_idx": "low",
    "salary_delay_trend": "high",
    "savings_slope": "low",
    "savings_volatility": "high",
    "cashflow_compression": "high",
    "emi_stress_count": "high",
    "seq_len": "low",
}


def run(loan_id: int) -> dict:
    borrower = borrower_record(loan_id)
    reference = test_context()
    features = {name: float(borrower[name]) for name in CONTEXT_COLS}
    percentile_values = {}
    anomalies = []
    for feature, direction in TAILS.items():
        values = reference[feature].astype(float).to_numpy()
        current = features[feature]
        midrank = float(np.mean(values < current) + 0.5 * np.mean(values == current))
        pct = midrank if direction == "high" else 1.0 - midrank
        percentile_values[feature] = round(pct, 4)
        if pct >= 0.90:
            anomalies.append({"feature": feature, "direction": direction, "percentile": round(pct, 4)})
    return {
        "inputs": {"loan_id": int(loan_id), "context_features": features,
                   "percentile_reference": "60,000 fixed test borrowers"},
        "rule_applied": "behavioral_anomaly_top_decile",
        "output": {"anomalies": anomalies, "feature_tail_percentiles": percentile_values},
        "simulated": True,
    }
