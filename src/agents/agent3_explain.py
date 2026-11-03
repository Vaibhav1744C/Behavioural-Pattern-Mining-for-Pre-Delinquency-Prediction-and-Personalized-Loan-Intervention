"""Agent 3: choose fuzzy-only or SHAP-plus-fuzzy explanation branch."""

from __future__ import annotations

from .common import borrower_record, clean_display_drivers, matching_fuzzy_rule

FUZZY_ONLY_MARGIN = 0.05
TIER_NAMES = ["Very Low", "Low", "Medium", "High", "Very High"]
TIER_CENTERS = [0.0, 0.25, 0.5, 0.75, 1.0]


def run(loan_id: int, risk_result: dict) -> dict:
    borrower = borrower_record(loan_id)
    risk = risk_result["output"]
    margin = float(risk["confidence_margin"])
    branch = "fuzzy-only" if margin > FUZZY_ONLY_MARGIN else "SHAP+fuzzy"
    drivers = clean_display_drivers(borrower.get("display_drivers", []))
    match = matching_fuzzy_rule(loan_id)
    if match:
        fuzzy_rule_id, rule = match
        fuzzy = {
            "risk_linguistic": rule["risk_linguistic"],
            "surrogate_risk_score": float(rule["risk_score"]),
            "rule_id": fuzzy_rule_id,
            "percentile_tier": risk["tier"],
        }
        fuzzy_rule_name = fuzzy_rule_id
    else:
        percentile = float(risk["behavioural_percentile"])
        center = TIER_CENTERS[TIER_NAMES.index(risk["tier"])]
        delta = round((percentile - center) / 0.25, 4)
        fuzzy = {
            "risk_linguistic": f"({risk['tier']}, {delta:+.4f})",
            "rule_id": "FUZZY_PERCENTILE_FALLBACK",
            "percentile_tier": risk["tier"],
        }
        fuzzy_rule_name = "fuzzy_percentile_fallback"
    output = {"branch_taken": branch, "fuzzy_risk": fuzzy}
    if branch == "SHAP+fuzzy":
        output["display_drivers"] = drivers
    return {
        "inputs": {"loan_id": int(loan_id), "confidence_margin": margin,
                   "margin_threshold": FUZZY_ONLY_MARGIN,
                   "display_drivers_available": len(drivers)},
        "rule_applied": f"{fuzzy_rule_name}; high_margin_fuzzy_only_else_display_shap_plus_fuzzy",
        "output": output,
        "simulated": True,
    }
