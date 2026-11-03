"""Run the four deterministic, simulated workflow agents in order."""

from __future__ import annotations

from .agent1_behaviour import run as run_behaviour
from .agent2_risk import run as run_risk
from .agent3_explain import run as run_explain
from .agent4_intervene import run as run_intervene


def run_pipeline(loan_id: int) -> dict:
    behaviour = run_behaviour(int(loan_id))
    risk = run_risk(int(loan_id))
    explanation = run_explain(int(loan_id), risk)
    intervention = run_intervene(risk, explanation)
    return {
        "loan_id": int(loan_id),
        "steps": [behaviour, risk, explanation, intervention],
        "simulated": True,
    }
