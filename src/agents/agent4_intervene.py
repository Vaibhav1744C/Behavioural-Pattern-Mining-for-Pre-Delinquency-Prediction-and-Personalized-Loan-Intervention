"""Agent 4: deterministic tier/driver intervention rules with stable IDs."""

from __future__ import annotations

from .common import clean_display_drivers


def run(risk_result: dict, explanation_result: dict) -> dict:
    risk = risk_result["output"]
    tier = risk["tier"]
    drivers = clean_display_drivers(
        explanation_result["output"].get("display_drivers", [])
    )
    names = {driver["feature"] for driver in drivers}

    if tier == "Very High":
        rule_id, action, priority = "VH_EMI_RM_01", "EMI restructuring + RM escalation", "Urgent"
    elif tier == "High" and "emi_stress_count" in names:
        rule_id, action, priority = "H_EMI_02", "EMI restructuring", "High"
    elif tier == "High" and "salary_delay_trend" in names:
        rule_id, action, priority = "H_SALARY_CALL_03", "Outreach call", "High"
    elif tier == "High":
        rule_id, action, priority = "H_DEFAULT_CALL_04", "Outreach call", "High"
    elif tier == "Medium":
        rule_id, action, priority = "M_COUNSELLING_05", "Counselling + closer monitoring", "Medium"
    else:
        rule_id, action, priority = "L_STANDARD_06", "Standard monitoring", "Routine"

    return {
        "inputs": {"tier": tier, "display_driver_names": sorted(names)},
        "rule_applied": rule_id,
        "output": {"rule_id": rule_id, "recommended_action": action,
                   "priority": priority},
        "simulated": True,
    }
