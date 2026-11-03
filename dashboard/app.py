"""Streamlit dashboard for the fixed-ID behavioral prediction subsample."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dashboard" / "data"
sys.path.insert(0, str(ROOT))

from src.agents.common import clean_display_drivers, is_date_derived

NAVY = "#1E2761"
TEAL = "#028090"
SCOPE = "200k-borrower subsample, 60k test"
PAGE_OPTIONS = [
    "Results", "Borrower Explorer", "Agent Workflow", "Portfolio",
    "Methodology and Limitations",
]

st.set_page_config(page_title="Behavioural Loan Risk Explorer", page_icon="◈", layout="wide")
st.markdown(
    f"""
    <style>
      .stApp {{ background: #f5f7fb; }}
      h1, h2, h3 {{ color: {NAVY}; }}
      div[data-testid="stMetric"] {{ background: white; border: 1px solid #e4e8f0;
        border-radius: 12px; padding: 12px 16px; }}
      .sim-banner {{ background: #fff4d6; color: #493900; padding: 12px 16px;
        border-left: 5px solid #e3a008; border-radius: 6px; font-weight: 650; }}
      .scope-label {{ color: #536078; font-size: 0.82rem; font-weight: 600; }}
      .attention-placeholder {{ color: #707887; background: #e8ebf0; padding: 22px;
        border-radius: 10px; text-align: center; border: 1px dashed #aeb5c1; }}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_parquet(name: str, cache_version: int) -> pd.DataFrame:
    return pd.read_parquet(DATA / name)


@st.cache_data(show_spinner=False)
def load_json(name: str, cache_version: int) -> dict:
    with (DATA / name).open(encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_surrogate_fidelity(cache_version: int) -> dict:
    with (ROOT / "reports" / "surrogate_fidelity_subsample.json").open(encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_report_json(name: str, cache_version: int) -> dict:
    with (ROOT / "reports" / name).open(encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_data(cache_version: int) -> dict:
    predictions = load_parquet("test_predictions.parquet", cache_version)
    borrowers = load_parquet("sample_borrowers.parquet", cache_version)
    sequences = load_parquet("sample_sequences.parquet", cache_version)
    sensitivity = load_json("sensitivity.json", cache_version)
    references = load_json("reference.json", cache_version)
    return {
        "predictions": predictions,
        "borrowers": borrowers,
        "sequences": sequences,
        "sensitivity": sensitivity,
        "references": references,
        "fidelity": load_surrogate_fidelity(cache_version),
        "static_results": load_report_json("baseline_subsample_results.json", cache_version),
        "behavioural_results": load_report_json("xgboost_behavioural_results_subsample.json", cache_version),
        "null_results": load_report_json("xgboost_behavioural_results_null.json", cache_version),
    }


def page_title(title: str, subtitle: str = "") -> None:
    st.title(title)
    st.markdown(f'<div class="scope-label">{SCOPE}</div>', unsafe_allow_html=True)
    if subtitle:
        st.caption(subtitle)


def tier_color(tier: str) -> str:
    return {
        "Very Low": "#9db9c0", "Low": "#62b6a8", "Medium": "#f0c36e",
        "High": "#ef8b62", "Very High": "#c74343",
    }.get(tier, TEAL)


def draw_results(data: dict) -> None:
    page_title("Results", "Subsample model comparison, threshold behavior, and shock-gap sensitivity.")
    def metric_row(label: str, result: dict) -> dict:
        metrics = result["results"][0]
        return {"Model": label, "ROC-AUC": float(metrics["auc_roc"]),
                "PR-AUC": float(metrics["pr_auc"]), "Scope": SCOPE}
    comparison = pd.DataFrame([
        metric_row("Static XGBoost", data["static_results"]),
        metric_row("Behavioral XGBoost", data["behavioural_results"]),
        metric_row("Null-control behavioral XGBoost", data["null_results"]),
    ])
    st.subheader("Model comparison")
    st.dataframe(comparison.style.format({"ROC-AUC": "{:.4f}", "PR-AUC": "{:.4f}"}),
                 use_container_width=True, hide_index=True)
    left, right = st.columns(2)
    with left:
        fig = px.bar(comparison, x="Model", y="ROC-AUC", color="Model",
                     color_discrete_sequence=[NAVY, TEAL, "#8fa3b8"],
                     title=f"ROC-AUC · {SCOPE}")
        fig.update_layout(showlegend=False, yaxis_range=[0.68, 0.79])
        st.plotly_chart(fig, use_container_width=True)
    with right:
        fig = px.bar(comparison, x="Model", y="PR-AUC", color="Model",
                     color_discrete_sequence=[NAVY, TEAL, "#8fa3b8"],
                     title=f"PR-AUC · {SCOPE}")
        fig.update_layout(showlegend=False, yaxis_range=[0.35, 0.50])
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Decision threshold on fixed test predictions")
    model_key = st.selectbox("Score used for classification", ["p_behav", "p_static"],
                             format_func=lambda key: "Behavioral XGBoost" if key == "p_behav" else "Static XGBoost")
    threshold = st.slider("Classification threshold", 0.0, 1.0, 0.5, 0.01)
    pred = data["predictions"]
    y = pred["y_true"].astype(int).to_numpy()
    yhat = (pred[model_key].to_numpy() >= threshold).astype(int)
    tp = int(((y == 1) & (yhat == 1)).sum())
    fp = int(((y == 0) & (yhat == 1)).sum())
    tn = int(((y == 0) & (yhat == 0)).sum())
    fn = int(((y == 1) & (yhat == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"Precision · {SCOPE}", f"{precision:.3f}")
    c2.metric(f"Recall · {SCOPE}", f"{recall:.3f}")
    c3.metric(f"False alarms (FP) · {SCOPE}", f"{fp:,}")
    c4.metric(f"Threshold · {SCOPE}", f"{threshold:.2f}")
    cm = pd.DataFrame([[tn, fp], [fn, tp]], index=["Actual 0", "Actual 1"],
                      columns=["Predicted 0", "Predicted 1"])
    cf = px.imshow(cm, text_auto=True, color_continuous_scale=[[0, "#e7eef3"], [1, TEAL]],
                   title=f"Confusion matrix · {SCOPE}", aspect="auto")
    st.plotly_chart(cf, use_container_width=True)

    st.subheader("Sensitivity to assumed shock-probability gap")
    sensitivity = pd.DataFrame(data["sensitivity"]["rows"])
    baseline_roc, baseline_pr = 0.7254, 0.4018
    sensitivity["ROC-AUC lift vs static"] = sensitivity["roc_auc"] - baseline_roc
    sensitivity["PR-AUC lift vs static"] = sensitivity["pr_auc"] - baseline_pr
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sensitivity["gap"], y=sensitivity["ROC-AUC lift vs static"],
                             mode="lines+markers", name="ROC-AUC lift vs static", line={"color": NAVY, "width": 3}))
    fig.add_trace(go.Scatter(x=sensitivity["gap"], y=sensitivity["PR-AUC lift vs static"],
                             mode="lines+markers", name="PR-AUC lift vs static", line={"color": TEAL, "width": 3}))
    fig.update_layout(title=f"Lift vs shock-probability gap · {SCOPE}",
                      xaxis_title="p(default shock) − p(non-default shock)",
                      yaxis_title="Metric lift vs static subsample model")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(data["sensitivity"]["seed_note"])

    refs = data["references"]
    st.markdown("### Full-run reference (separate from subsample results)")
    st.info(
        f"Full-run ROC-AUC reference — Static: {refs['static_roc_auc']:.4f}; "
        f"Behavioral: {refs['behavioural_roc_auc']:.4f}; LSTM: {refs['lstm_roc_auc']:.4f}. "
        "Behavioral reference is configurable; the other earlier script reported 0.7686."
    )


def draw_borrower_explorer(data: dict) -> None:
    page_title("Borrower Explorer", "Inspect a fixed test-sample borrower and their 24-month synthetic sequence.")
    borrowers = data["borrowers"]
    loan_ids = borrowers["loan_id"].astype(int).tolist()
    selected_id = st.selectbox("Loan ID", loan_ids, format_func=lambda value: str(value))
    row = borrowers.loc[borrowers["loan_id"].astype(int) == int(selected_id)].iloc[0]
    tier = str(row["risk_tier"])
    percentile = float(row["behavioural_percentile"])
    left, right = st.columns([1, 2])
    with left:
        gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=percentile * 100,
            number={"suffix": "th pct", "font": {"color": NAVY}},
            title={"text": f"{tier} risk tier", "font": {"color": tier_color(tier)}},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": tier_color(tier)},
                   "steps": [{"range": [0, 25], "color": "#edf2f5"},
                             {"range": [25, 50], "color": "#e4edf0"},
                             {"range": [50, 75], "color": "#d9e8e8"},
                             {"range": [75, 100], "color": "#c9dddf"}]},
        ))
        gauge.update_layout(height=270, margin={"t": 50, "b": 10, "l": 25, "r": 25})
        st.plotly_chart(gauge, use_container_width=True)
        st.metric(f"Behavioral score · {SCOPE}", f"{float(row['p_behav']):.3f}")
        st.write({key: row[key] for key in ["grade", "dti", "annual_inc", "loan_amnt", "term", "int_rate"]})
    with right:
        seq = data["sequences"].loc[data["sequences"]["loan_id"].astype(int) == int(selected_id)].sort_values("month")
        numeric_cols = [col for col in ["salary_delay_days", "account_balance", "savings_balance", "discretionary_spend"] if col in seq]
        if not seq.empty and numeric_cols:
            long = seq.melt(id_vars="month", value_vars=numeric_cols, var_name="Measure", value_name="Value")
            fig = px.line(long, x="month", y="Value", color="Measure", markers=True,
                          title="Monthly behavioral trajectories · synthetic sequence")
            fig.update_layout(colorway=[NAVY, TEAL, "#73a9ad", "#d88a55"])
            st.plotly_chart(fig, use_container_width=True)
        if "emi_status" in seq:
            status_map = {"on_time": 0, "delayed": 1, "missed": 2}
            emi = seq[["month", "emi_status"]].copy()
            emi["status_code"] = emi["emi_status"].map(status_map).fillna(0)
            fig = px.scatter(emi, x="month", y="status_code", color="emi_status",
                             title="EMI payment status · synthetic sequence",
                             color_discrete_map={"on_time": TEAL, "delayed": "#e3a008", "missed": "#c74343"})
            fig.update_yaxes(tickvals=[0, 1, 2], ticktext=["On time", "Delayed", "Missed"])
            st.plotly_chart(fig, use_container_width=True)

    drivers = clean_display_drivers(row.get("display_drivers", []))
    drivers = [d for d in drivers if not is_date_derived(d["feature"])]
    st.subheader("SHAP drivers · date-derived features hidden")
    st.dataframe(pd.DataFrame(drivers), hide_index=True, use_container_width=True)
    from src.agents.orchestrator import run_pipeline
    fuzzy = run_pipeline(int(selected_id))["steps"][2]["output"]["fuzzy_risk"]
    names = [d["feature"] for d in drivers[:2]]
    lead = ", ".join(names) if names else "no non-date driver in the displayed top list"
    st.write(f"Fuzzy surrogate risk label: **{fuzzy['risk_linguistic']}** (percentile tier: {tier}). Templated narrative: The borrower is in the {tier} behavioral risk tier; the leading eligible model drivers are {lead}.")
    st.markdown('<div class="attention-placeholder">Attention view requires the LSTM, not run on the subsample.</div>', unsafe_allow_html=True)


def draw_agent_workflow(data: dict) -> None:
    page_title("Agent Workflow", "Deterministic workflow demonstration; outputs are simulated.")
    st.markdown('<div class="sim-banner">Rule-based simulation of the planned multi-agent architecture.</div>', unsafe_allow_html=True)
    from src.agents.orchestrator import run_pipeline
    borrowers = data["borrowers"]
    selected_id = st.selectbox("Sampled test borrower", borrowers["loan_id"].astype(int).tolist(), key="agent_loan")
    if st.button("Run", type="primary"):
        result = run_pipeline(int(selected_id))
        labels = ["Agent 1 · behavior scan", "Agent 2 · risk", "Agent 3 · explanation", "Agent 4 · intervention"]
        for i, (label, step) in enumerate(zip(labels, result["steps"])):
            with st.expander(label, expanded=(i == 0)):
                st.write("**Input**")
                st.json(step["inputs"])
                st.write("**Rule applied**")
                st.code(str(step["rule_applied"]))
                if i == 2:
                    st.write(f"**Branch taken:** {step['output']['branch_taken']}")
                st.write("**Output**")
                st.json(step["output"])
                st.caption("simulated: true")


def draw_portfolio(data: dict) -> None:
    page_title("Portfolio", "Risk tiers across the fixed test set and a sample intervention queue.")
    predictions = data["predictions"]
    tier_order = ["Very Low", "Low", "Medium", "High", "Very High"]
    counts = predictions["risk_tier"].value_counts().reindex(tier_order, fill_value=0).rename_axis("Tier").reset_index(name="Borrowers")
    fig = px.bar(counts, x="Tier", y="Borrowers", color="Tier", category_orders={"Tier": tier_order},
                 color_discrete_map={tier: tier_color(tier) for tier in tier_order},
                 title=f"Behavioral risk-tier distribution · {SCOPE}")
    st.plotly_chart(fig, use_container_width=True)

    from src.agents.agent4_intervene import run as run_intervention
    sample = predictions.nlargest(50, "p_behav")
    rows = []
    for record in sample.to_dict(orient="records"):
        intervention = run_intervention(
            {"output": {"tier": str(record["risk_tier"])}},
            {"output": {"display_drivers": []}},
        )
        action = intervention["output"]
        rows.append({"loan_id": int(record["loan_id"]), "tier": record["risk_tier"],
                     "p_behav": float(record["p_behav"]), "priority": action["priority"],
                     "recommended_action": action["recommended_action"], "rule_id": action["rule_id"]})
    st.subheader(f"Intervention queue · top 50 by behavioral score · {SCOPE}")
    st.caption("Queue interventions use tier rules; individual SHAP driver evidence is available for the 200-borrower explorer sample.")
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def draw_methodology(data: dict) -> None:
    page_title("Methodology and Limitations", "How the demo data and simulated decision workflow were assembled.")
    st.subheader("Pipeline")
    flow = go.Figure(go.Sankey(
        node={"label": ["Lending Club resolved loans", "Cleaned cohort", "Fixed-ID subsample\n200k borrowers",
                        "Synthetic monthly behavior", "Static + behavioral XGBoost", "Risk explanation", "Simulated intervention"],
              "color": [NAVY, NAVY, TEAL, TEAL, NAVY, TEAL, "#e3a008"]},
        link={"source": [0, 1, 2, 3, 4, 5], "target": [1, 2, 3, 4, 5, 6],
              "value": [1, 1, 1, 1, 1, 1]},
    ))
    flow.update_layout(height=300, title="Research pipeline (LSTM unavailable for subsample)")
    st.plotly_chart(flow, use_container_width=True)
    st.markdown("""
    - Dataset: Lending Club resolved-loan cohort; dashboard evaluation uses 200,000 selected borrowers.
    - Split: 140,000 train and 60,000 test borrowers; the dashboard predictions use the fixed test IDs.
    - Synthetic behavior is generated from static borrower characteristics plus stochastic shocks.
    - Null control uses equal shock probabilities for default and non-default borrowers.
    """)
    st.caption("Source cohort: 1,879,234 resolved loans × 71 cleaned columns; early-default rate 20.19% before subsampling.")
    pred = data["predictions"]
    null = data["null_results"]
    null_roc = float(null["results"][0]["auc_roc"])
    null_pr = float(null["results"][0]["pr_auc"])
    st.write(f"Null-control subsample result: ROC-AUC {null_roc:.4f}, PR-AUC {null_pr:.4f} ({SCOPE}).")
    fidelity = data["fidelity"]
    st.warning(
        "Behavioral data is synthetic; the lift depends on the assumed shock-probability gap "
        "(about +0.037 ROC-AUC at 0.35 vs 0.12, under +0.01 at a gap of 0.11). "
        f"The surrogate R² is {fidelity['xgb_behavioural']:.3f} against the paper's about 0.807. "
        "The model includes a loan-vintage feature (issue_d) inherited from the base-paper reproduction, "
        "which is hidden from driver displays. The LSTM underperformed in the full run (0.6379) "
        "and was not rerun on the subsample."
    )
    st.caption(f"Loaded {len(pred):,} fixed test predictions. Behavioral synthetic-signal results are scenario-dependent.")


def main() -> None:
    st.sidebar.title("Loan Risk Research Demo")
    page = st.sidebar.radio("Page", PAGE_OPTIONS)
    try:
        files = [
            DATA / "test_predictions.parquet", DATA / "sample_borrowers.parquet",
            DATA / "sample_sequences.parquet", DATA / "sensitivity.json",
            DATA / "reference.json", ROOT / "reports" / "surrogate_fidelity_subsample.json",
            ROOT / "reports" / "baseline_subsample_results.json",
            ROOT / "reports" / "xgboost_behavioural_results_subsample.json",
            ROOT / "reports" / "xgboost_behavioural_results_null.json",
        ]
        cache_version = max(path.stat().st_mtime_ns for path in files)
        data = load_data(cache_version)
    except Exception as exc:
        st.error(f"Dashboard data could not be loaded: {exc}")
        st.stop()
    dispatch = {
        "Results": draw_results,
        "Borrower Explorer": draw_borrower_explorer,
        "Agent Workflow": draw_agent_workflow,
        "Portfolio": draw_portfolio,
        "Methodology and Limitations": draw_methodology,
    }
    dispatch[page](data)


if __name__ == "__main__":
    main()
