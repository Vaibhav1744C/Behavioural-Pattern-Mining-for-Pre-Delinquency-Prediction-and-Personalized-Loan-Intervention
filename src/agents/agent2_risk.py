"""Agent 2: return available model scores and fuzzy tier confidence margin."""

from __future__ import annotations

from .common import borrower_record, prediction_record

BOUNDARIES = (0.125, 0.375, 0.625, 0.875)


def run(loan_id: int) -> dict:
    borrower = borrower_record(loan_id)
    prediction = prediction_record(loan_id)
    percentile = float(prediction["behavioural_percentile"])
    margin = min(abs(percentile - boundary) for boundary in BOUNDARIES)
    output = {
        "p_static": float(prediction["p_static"]),
        "p_behav": float(prediction["p_behav"]),
        "tier": str(prediction["risk_tier"]),
        "behavioural_percentile": percentile,
        "confidence_margin": float(margin),
    }
    return {
        "inputs": {"loan_id": int(loan_id), "percentile": percentile,
                   "tier_boundaries": list(BOUNDARIES)},
        "rule_applied": "percentile_risk_tier_and_nearest_boundary_margin",
        "output": output,
        "simulated": True,
    }
