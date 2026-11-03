"""Presentation layer for the fixed-ID behavioral prediction subsample."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dashboard" / "data"
sys.path.insert(0, str(ROOT))
from src.agents.common import clean_display_drivers, is_date_derived
from dashboard.theme import (AMBER_BG, BORDER, GREY, GRID, INK, MUTED, NAVY, PAGE,
                             PLOTLY_TEMPLATE, RISK_COLORS, RISK_ORDER, TEAL, WHITE,
                             CUSTOM_CSS)

SCOPE = "200k-borrower subsample, 60k test"
REQUIRED_DATA_FILES = [
    "test_predictions.parquet",
    "sample_borrowers.parquet",
    "sample_sequences.parquet",
    "sensitivity.json",
    "reference.json",
    "model_metrics.json",
]
PAGES = ["Results", "Borrower Explorer", "Agent Workflow", "Portfolio", "Methodology and Limitations"]
NAV = {"Results": "▤", "Borrower Explorer": "⌕", "Agent Workflow": "⑂", "Portfolio": "▥", "Methodology and Limitations": "◫"}
st.set_page_config(page_title="Behavioural Loan Risk Research", layout="wide", initial_sidebar_state="expanded")
st.markdown(f"<style>{CUSTOM_CSS}</style>", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_parquet(name: str, version: int) -> pd.DataFrame:
    return pd.read_parquet(DATA / name)


@st.cache_data(show_spinner=False)
def load_json(name: str, version: int) -> dict:
    return json.loads((DATA / name).read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_all(version: int) -> dict:
    pred = load_parquet("test_predictions.parquet", version)
    borrowers = load_parquet("sample_borrowers.parquet", version)
    sequences = load_parquet("sample_sequences.parquet", version)
    metrics = load_json("model_metrics.json", version)
    return {"predictions": pred, "borrowers": borrowers, "sequences": sequences,
            "sensitivity": load_json("sensitivity.json", version),
            "references": load_json("reference.json", version), "metrics": metrics,
            "static_results": {"results": [{"auc_roc": metrics["static"]["auc_roc"], "pr_auc": metrics["static"]["pr_auc"]}]},
            "behavioural_results": {"results": [{"auc_roc": metrics["behavioural_main"]["auc_roc"], "pr_auc": metrics["behavioural_main"]["pr_auc"]}]},
            "null_results": {"results": [{"auc_roc": metrics["null"]["auc_roc"], "pr_auc": metrics["null"]["pr_auc"]}]}}


@st.cache_data(show_spinner=False)
def best_f1_threshold(y_values: tuple[int, ...], scores: tuple[float, ...]) -> tuple[float, float]:
    y = np.asarray(y_values, dtype=np.int8)
    p = np.asarray(scores, dtype=np.float64)
    order = np.argsort(-p, kind="mergesort")
    ps, ys = p[order], y[order]
    tp_cum = np.cumsum(ys)
    end = np.r_[np.flatnonzero(np.diff(ps)), len(ps) - 1]
    tp, predicted = tp_cum[end], end + 1
    positives = int(y.sum())
    f1 = 2 * tp / np.maximum(predicted + positives, 1)
    i = int(np.argmax(f1))
    return float(ps[end[i]]), float(f1[i])


def plot(fig: go.Figure, height: int | None = None) -> None:
    fig.update_layout(template=PLOTLY_TEMPLATE, autosize=True, height=height, paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font={"family": "Inter, sans-serif", "color": NAVY, "size": 12})
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False, "responsive": True})


def page_header(title: str, desc: str) -> None:
    st.markdown(f'<div class="page-head"><div><h1>{title}</h1><div class="page-desc">{desc}</div></div><div class="context-chip">Subsample run</div></div>', unsafe_allow_html=True)


def card_head(title: str, subtitle: str = "") -> None:
    sub = f'<div class="card-subtitle">{subtitle}</div>' if subtitle else ""
    st.markdown(f'<div class="card-title">{title}</div>{sub}', unsafe_allow_html=True)


def tier_color(tier: str) -> str:
    return RISK_COLORS.get(str(tier), TEAL)


def metric_df(data: dict) -> pd.DataFrame:
    items = [("Static XGBoost", data["static_results"]), ("Behavioral XGBoost", data["behavioural_results"]),
             ("Null-control behavioral XGBoost", data["null_results"])]
    return pd.DataFrame([{"Model": label, "ROC-AUC": float(value["results"][0]["auc_roc"]),
                          "PR-AUC": float(value["results"][0]["pr_auc"]), "Scope": SCOPE} for label, value in items])


def draw_results(data: dict) -> None:
    page_header("Results", "Compare held-out model performance, decision thresholds, and sensitivity to the synthetic shock gap.")
    comp = metric_df(data)
    st.caption(SCOPE)
    stat, beh, null = comp.iloc[0], comp.iloc[1], comp.iloc[2]
    k = st.columns(4)
    k[0].metric("Static ROC-AUC", f"{stat['ROC-AUC']:.4f}", help=SCOPE)
    k[1].metric("Behavioural ROC-AUC", f"{beh['ROC-AUC']:.4f}", f"{beh['ROC-AUC']-stat['ROC-AUC']:+.4f} vs static", help=SCOPE)
    k[2].metric("PR-AUC lift", f"{beh['PR-AUC']-stat['PR-AUC']:+.4f}", f"Behavioral − static", help=SCOPE)
    k[3].metric("Null-control ROC-AUC", f"{null['ROC-AUC']:.4f}", f"{null['ROC-AUC']-stat['ROC-AUC']:+.4f} vs static", help=SCOPE)
    left, right = st.columns([1.08, .92], gap="large")
    with left:
        card_head("Model comparison", SCOPE)
        fig = go.Figure()
        for label, color in [("Static XGBoost", NAVY), ("Behavioral XGBoost", TEAL), ("Null-control behavioral XGBoost", GREY)]:
            row = comp.loc[comp.Model == label].iloc[0]
            fig.add_bar(name=label, x=["ROC-AUC", "PR-AUC"], y=[row["ROC-AUC"], row["PR-AUC"]], marker_color=color,
                        text=[f"{row['ROC-AUC']:.4f}", f"{row['PR-AUC']:.4f}"], textposition="outside", cliponaxis=False)
        fig.update_layout(barmode="group", height=300, margin={"l": 20,"r":15,"t":25,"b":35}, yaxis={"range":[0,.85],"title":"Score"}, xaxis_title="")
        plot(fig, 300)
        st.dataframe(comp.style.format({"ROC-AUC":"{:.4f}","PR-AUC":"{:.4f}"}), use_container_width=True, hide_index=True)
    with right:
        pred = data["predictions"]
        default_t, f1_best = best_f1_threshold(tuple(pred.y_true.astype(int)), tuple(pred.p_behav.astype(float)))
        card_head("Decision threshold", "Live classification metrics on fixed test predictions")
        model = st.selectbox("Score used for classification", ["p_behav", "p_static"], format_func=lambda x: "Behavioral XGBoost" if x == "p_behav" else "Static XGBoost")
        threshold = st.slider("Classification threshold", 0.0, 1.0, float(default_t), 0.001, key="threshold", help=f"Best F1 threshold for behavioral predictions: {default_t:.3f} (F1 {f1_best:.3f}).")
        st.caption(f"Best-F1 marker: **{default_t:.3f}** · F1 {f1_best:.3f} ({SCOPE})")
        y = pred.y_true.astype(int).to_numpy(); yh = (pred[model].to_numpy() >= threshold).astype(int)
        tp = int(((y==1)&(yh==1)).sum()); fp = int(((y==0)&(yh==1)).sum()); tn = int(((y==0)&(yh==0)).sum()); fn = int(((y==1)&(yh==0)).sum())
        precision = tp/(tp+fp) if tp+fp else 0.; recall = tp/(tp+fn) if tp+fn else 0.; false_alarm = fp/(fp+tn) if fp+tn else 0.
        q = st.columns(3); q[0].metric("Precision",f"{precision:.3f}"); q[1].metric("Recall",f"{recall:.3f}"); q[2].metric("False alarms",f"{false_alarm:.3f}",help=f"False-positive rate; {fp:,} false positives")
        cm = np.array([[tn,fp],[fn,tp]])
        text = [[f"{cm[i,j]:,}<br>{cm[i,j]/max(cm[i].sum(),1):.1%}" for j in range(2)] for i in range(2)]
        fig=go.Figure(go.Heatmap(z=cm, x=["Predicted 0","Predicted 1"], y=["Actual 0","Actual 1"],
                                 text=text, texttemplate="%{text}", textfont={"color":NAVY,"size":14},
                                 colorscale=[[0,"#F4F7FA"],[1,"#B8DDE1"]],showscale=False,hovertemplate="%{y} · %{x}<br>Count: %{z:,}<extra></extra>"))
        fig.update_layout(title=f"Confusion matrix · {SCOPE}",height=250,margin={"l":20,"r":10,"t":50,"b":25},yaxis={"autorange":"reversed"})
        plot(fig,250)
    card_head("Sensitivity to assumed shock-probability gap", "Lift against the static subsample model")
    sens=pd.DataFrame(data["sensitivity"]["rows"]); sens["ROC-AUC lift"]=sens.roc_auc-float(stat["ROC-AUC"]); sens["PR-AUC lift"]=sens.pr_auc-float(stat["PR-AUC"])
    fig=go.Figure(); fig.add_scatter(x=sens.gap,y=sens["ROC-AUC lift"],name="ROC-AUC lift",mode="lines+markers+text",text=[f"{v:+.3f}" for v in sens["ROC-AUC lift"]],textposition="top center",line={"color":NAVY,"width":2.5})
    fig.add_scatter(x=sens.gap,y=sens["PR-AUC lift"],name="PR-AUC lift",mode="lines+markers",line={"color":TEAL,"width":2.5})
    fig.update_layout(height=300,xaxis_title="p(default shock) − p(non-default shock)",yaxis_title="Metric lift",margin={"t":24,"b":45})
    plot(fig,300); st.caption(data["sensitivity"]["seed_note"])
    refs=data["references"]
    st.markdown(f'<div class="card" style="background:#F2F3F7;border-style:dashed;box-shadow:none"><div class="card-title">Full-run reference, not subsample</div><div class="card-subtitle">Full-run ROC-AUC · Static {refs["static_roc_auc"]:.4f} &nbsp; Behavioural {refs["behavioural_roc_auc"]:.4f} &nbsp; LSTM {refs["lstm_roc_auc"]:.4f}. Behavioural reference is configurable; the other earlier script reported 0.7686.</div></div>',unsafe_allow_html=True)


def draw_borrower_explorer(data: dict) -> None:
    page_header("Borrower Explorer", "Review sampled test borrowers, their synthetic monthly trajectory, and model evidence.")
    borrowers=data["borrowers"]; ids=borrowers.loan_id.astype(int).tolist()
    selected=st.selectbox("Loan ID",ids,format_func=str); row=borrowers.loc[borrowers.loan_id.astype(int)==int(selected)].iloc[0]
    tier=str(row.risk_tier); pct=float(row.behavioural_percentile); prob=float(row.p_behav)
    top_l,top_r=st.columns([1,1.35],gap="large")
    with top_l:
        card_head("Relative risk tier (percentile rank in 60k test set)")
        fig=go.Figure(go.Indicator(mode="gauge+number",value=pct*100,number={"font":{"color":NAVY,"size":24}},
            title={"text":f"<b>{tier}</b>","font":{"color":tier_color(tier),"size":18}},
            gauge={"axis":{"range":[0,100],"showticklabels":False,"ticks":""},"bar":{"color":tier_color(tier),"thickness":.25},"bgcolor":"white","borderwidth":0,
                  "steps":[{"range":[0,25],"color":"#F1F5F6"},{"range":[25,50],"color":"#E9F0EE"},{"range":[50,75],"color":"#F7F1DF"},{"range":[75,100],"color":"#F8EAE5"}] }))
        fig.update_layout(height=245,margin={"t":38,"b":4,"l":25,"r":25}); plot(fig,245)
        st.metric(f"Behavioural predicted probability · {SCOPE}",f"{prob:.3f}")
        st.caption("Risk tier is relative within the test-set score distribution.")
    with top_r:
        card_head("Borrower facts", "Loan characteristics from the sampled test cohort")
        fields=[("Grade","grade"),("DTI","dti"),("Annual income","annual_inc"),("Loan amount","loan_amnt"),("Term","term"),("Interest rate","int_rate")]
        fact_cells=[]
        for label,key in fields:
            val=row.get(key,"—")
            val=f"{val:,.0f}" if isinstance(val,(float,int,np.number)) and label in {"Annual income","Loan amount"} else (f"{val:.2f}" if isinstance(val,(float,int,np.number)) else str(val))
            fact_cells.append(f'<div class="fact-cell"><span>{label}</span><strong>{val}</strong></div>')
        st.markdown('<div class="facts-grid">'+"".join(fact_cells)+'</div>',unsafe_allow_html=True)
        fuzzy=st.session_state.get(f"fuzzy_{selected}")
        if fuzzy is None:
            try:
                from src.agents.orchestrator import run_pipeline
                fuzzy=run_pipeline(int(selected))["steps"][2]["output"]["fuzzy_risk"]
            except (FileNotFoundError, OSError) as exc:
                st.caption(f"Fuzzy category is unavailable because an optional agent input is absent: {exc.filename or exc}.")
        st.markdown('<span title="This fuzzy label is an absolute probability category. The percentile tier ranks this borrower against the 60k test set, so the labels can differ because one is probability-based and the other is relative to this cohort."><b>Absolute risk label (fuzzy, from predicted probability) ⓘ</b></span>',unsafe_allow_html=True)
        if fuzzy is not None:
            st.markdown(f'<span class="risk-pill" style="background:{tier_color(tier)}">{fuzzy["risk_linguistic"]}</span>',unsafe_allow_html=True)
        else:
            st.caption("Unavailable in a fresh clone until the optional agent inputs are supplied.")
    seq=data["sequences"].loc[data["sequences"].loan_id.astype(int)==int(selected)].sort_values("month")
    card_head("24-month trajectories", "Synthetic borrower behavior · common month axis")
    cols=[("Salary delay","salary_delay_days"),("Account balance","account_balance"),("Savings","savings_balance"),("Discretionary spend","discretionary_spend")]
    chart_cols=st.columns(3)
    for i,(label,col) in enumerate(cols):
        with chart_cols[i%3]:
            if col in seq and not seq.empty:
                fig=go.Figure(go.Scatter(x=seq.month,y=seq[col],mode="lines",line={"color":TEAL,"width":2.4},fill="tozeroy",fillcolor="rgba(2,128,144,.08)",hovertemplate="Month %{x}<br>%{y:,.2f}<extra></extra>"))
                fig.update_layout(title=label,height=205,margin={"t":42,"b":24,"l":40,"r":10},xaxis={"title":"Month","dtick":6})
                plot(fig,205)
    with chart_cols[1]:
        if "emi_status" in seq and not seq.empty:
            codes={"on_time":0,"delayed":1,"missed":2}; status=seq.emi_status.map(codes).fillna(0)
            fig=go.Figure(go.Heatmap(z=[status.tolist()],x=seq.month,y=["EMI status"],zmin=0,zmax=2,
                colorscale=[[0,TEAL],[.5,"#F2C14E"],[1,"#D64550"]],showscale=False,
                text=[seq.emi_status.astype(str).tolist()],texttemplate="",hovertemplate="Month %{x}<br>%{text}<extra></extra>"))
            fig.update_layout(title="EMI status",height=205,margin={"t":42,"b":24,"l":40,"r":10},xaxis={"title":"Month","dtick":6})
            plot(fig,205)
    raw=clean_display_drivers(row.get("display_drivers",[])); drivers=[d for d in raw if not is_date_derived(d["feature"])]
    dl,dr=st.columns([1,1],gap="large")
    with dl:
        card_head("Behavioural SHAP drivers", "Date-derived features are excluded from display")
        if drivers:
            df=pd.DataFrame(drivers).head(5).rename(columns={"shap_value":"value"}).sort_values("value")
            colors=["#D64550" if float(v)>0 else TEAL for v in df.value]
            fig=go.Figure(go.Bar(x=df.value,y=df.feature,orientation="h",marker_color=colors,text=[f"{v:+.3f}" for v in df.value],textposition="outside",cliponaxis=False))
            fig.update_layout(height=270,margin={"t":15,"b":35,"l":15,"r":45},xaxis_title="SHAP contribution to raw model score",yaxis={"title":""})
            plot(fig,270)
            st.caption("Red increases the model score; teal decreases it.")
    with dr:
        from src.agents.orchestrator import run_pipeline
        try:
            pipeline=run_pipeline(int(selected)); fuzzy=pipeline["steps"][2]["output"]["fuzzy_risk"]
        except (FileNotFoundError, OSError) as exc:
            fuzzy=None
        names=", ".join(d["feature"] for d in drivers[:2]) if drivers else "no eligible feature in the displayed list"
        card_head("Narrative")
        fuzzy_text=f"The absolute fuzzy risk label is <b>{fuzzy['risk_linguistic']}</b>; " if fuzzy is not None else "The absolute fuzzy risk label is unavailable until optional agent inputs are supplied; "
        st.markdown(f'<div class="quote">The borrower is in the <b>{tier}</b> relative risk tier. {fuzzy_text}the leading eligible model drivers are {names}.</div>',unsafe_allow_html=True)
        st.markdown('<div class="notice-card"><b>Attention view unavailable</b><br>Attention view requires the LSTM, not run on the subsample.</div>',unsafe_allow_html=True)


def draw_agent_workflow(data: dict) -> None:
    page_header("Agent Workflow", "A deterministic demonstration of the planned borrower review sequence.")
    st.markdown('<div class="sim-banner">Rule-based simulation of the planned multi-agent architecture.</div>',unsafe_allow_html=True)
    from src.agents.orchestrator import run_pipeline
    ids=data["borrowers"].loan_id.astype(int).tolist(); selected=st.selectbox("Sampled test borrower",ids,format_func=str,key="workflow_loan")
    if st.button("Run workflow",type="primary",use_container_width=False):
        try:
            st.session_state.agent_result=run_pipeline(int(selected))
            st.session_state.agent_error=None
        except (FileNotFoundError, OSError) as exc:
            st.session_state.agent_result=None
            st.session_state.agent_error=str(exc.filename or exc)
    result=st.session_state.get("agent_result")
    if result:
        labels=["Behaviour scan","Risk assessment","Explanation","Intervention"]
        st.markdown('<div class="stepper">'+"<div class='step-node'>"+"</div><div class='step-line'></div><div class='step-node'>".join([str(i+1) for i in range(4)])+"</div></div>",unsafe_allow_html=True)
        for i,(label,step) in enumerate(zip(labels,result["steps"])):
            with st.expander(f"{i+1:02d}  {label}",expanded=(i==0)):
                c1,c2=st.columns([1,1.2])
                with c1: st.caption("INPUT"); st.json(step["inputs"])
                with c2:
                    st.caption("RULE APPLIED"); st.markdown(f'<span class="mono-chip">{step["rule_applied"]}</span>',unsafe_allow_html=True)
                    if i==2: st.caption(f"Branch taken: {step['output']['branch_taken']}")
                    st.caption("OUTPUT"); st.json(step["output"])
                st.caption("simulated: true")
    else:
        error=st.session_state.get("agent_error")
        if error:
            st.warning(f"The simulated workflow needs an optional agent input that is not included in a fresh clone: {error}")
        else:
            st.markdown('<div class="notice-card">Select a sampled test borrower and run the four-step workflow.</div>',unsafe_allow_html=True)


def draw_portfolio(data: dict) -> None:
    page_header("Portfolio", "Explore relative test-set risk bands and observed outcomes by band.")
    pred=data["predictions"]; counts=pred.risk_tier.value_counts().reindex(RISK_ORDER,fill_value=0)
    rates=pred.groupby("risk_tier",observed=False).y_true.mean().reindex(RISK_ORDER)
    st.caption("Tier sizes are relative bands by construction; the observed default-rate chart shows the realized y_true mean within each tier.")
    a,b=st.columns(2,gap="large")
    with a:
        card_head("Tier distribution",SCOPE)
        fig=go.Figure(go.Pie(labels=RISK_ORDER,values=counts.values,hole=.68,sort=False,
            marker={"colors":[RISK_COLORS[t] for t in RISK_ORDER],"line":{"color":"white","width":3}},
            textinfo="label+percent",hovertemplate="%{label}<br>%{value:,} borrowers<extra></extra>"))
        fig.update_layout(height=325,showlegend=False,margin={"t":18,"b":15,"l":10,"r":10},annotations=[{"text":f"{len(pred):,}<br><span style='font-size:11px'>test IDs</span>","showarrow":False,"font":{"size":21,"color":NAVY}}])
        plot(fig,325)
    with b:
        card_head("Observed default rate by tier", "Mean y_true among test predictions in each relative tier")
        fig=go.Figure(go.Bar(x=RISK_ORDER,y=rates.values,marker_color=[RISK_COLORS[t] for t in RISK_ORDER],
             text=[f"{v:.1%}" for v in rates.values],textposition="outside",cliponaxis=False,
             customdata=counts.values,hovertemplate="%{x}<br>Observed default rate: %{y:.3f}<br>Borrowers: %{customdata:,}<extra></extra>"))
        fig.update_layout(height=325,yaxis={"title":"Mean y_true","tickformat":".0%","range":[0,max(.25,float(rates.max())*1.2)]},xaxis={"title":"Relative tier"},margin={"t":18,"b":50,"l":55,"r":12})
        plot(fig,325)
    card_head("Intervention queue",f"Top 50 by behavioral score · {SCOPE}")
    search=st.text_input("Filter queue by loan ID, tier, action or rule ID",placeholder="Search the queue")
    from src.agents.agent4_intervene import run as intervention_run
    sample=pred.nlargest(50,"p_behav"); rows=[]
    for rank,rec in enumerate(sample.to_dict(orient="records"),start=1):
        action=intervention_run({"output":{"tier":str(rec["risk_tier"])}},{"output":{"display_drivers":[]}})["output"]
        rows.append({"Rank":rank,"loan_id":int(rec["loan_id"]),"Tier":str(rec["risk_tier"]),"Score":float(rec["p_behav"]),"Action":action["recommended_action"],"Priority":action["priority"],"Rule ID":action["rule_id"]})
    table=pd.DataFrame(rows)
    if search:
        mask=table.astype(str).apply(lambda col:col.str.contains(search,case=False,regex=False)).any(axis=1); table=table[mask]
    if table.empty: st.info("No queue entries match this filter.")
    else:
        st.dataframe(table.style.format({"Score":"{:.3f}"}).bar(subset=["Score"],color="#B8DDE1",vmin=0,vmax=1)
                     .map(lambda x:f"background-color: {tier_color(x)}; color: white",subset=["Tier"]),use_container_width=True,hide_index=True,height=460)


def draw_methodology(data: dict) -> None:
    page_header("Methodology and Limitations", "A compact view of the cohort, modeling pipeline, and limits on interpretation.")
    card_head("Research pipeline")
    labels=["Resolved Lending Club\nloans","Cleaned cohort\n1,879,234 × 71","Fixed-ID subsample\n200k borrowers","Synthetic monthly\nbehavior","Static + behavioral\nXGBoost","Explainable review\n+ simulated action"]
    cols=st.columns([1, .15, 1, .15, 1, .15, 1, .15, 1, .15, 1])
    for i,label in enumerate(labels):
        with cols[2*i]: st.markdown(f'<div class="flow-box">{label.replace(chr(10),"<br>")}</div>',unsafe_allow_html=True)
        if i<len(labels)-1:
            with cols[2*i+1]: st.markdown('<div style="text-align:center;color:#028090;font-size:1.4rem;padding-top:9px">→</div>',unsafe_allow_html=True)
    st.caption("The LSTM was not run on this subsample.")
    card_head("Dataset facts",SCOPE)
    facts=[("Source cohort","1,879,234 resolved loans"),("Cleaned columns","71"),("Subsample","200,000 borrowers"),("Training IDs","140,000"),("Test IDs","60,000"),("Early-default rate","20.19% before subsampling")]
    c=st.columns(3)
    for i,(name,value) in enumerate(facts):
        with c[i%3]: st.metric(name,value)
    null=data["null_results"]["results"][0]
    card_head("Null-control result")
    st.write(f"Null-control subsample result: ROC-AUC {null['auc_roc']:.4f}, PR-AUC {null['pr_auc']:.4f} ({SCOPE}).")
    fidelity=float(data["metrics"]["surrogate_r2"])
    limitations=[
        ("Behavioral data is synthetic.","The lift depends on the assumed shock-probability gap (about +0.037 ROC-AUC at 0.35 vs 0.12, under +0.01 at a gap of 0.11)."),
        ("The surrogate is an approximation.",f"Its R² is {fidelity:.3f} against the paper's about 0.807."),
        ("A loan-vintage feature remains in the model.","issue_d is inherited from the base-paper reproduction and hidden from driver displays."),
        ("LSTM was not rerun on the subsample.","It underperformed in the full run (0.6379 ROC-AUC)."),
    ]
    st.subheader("Limitations")
    for i,(lead,detail) in enumerate(limitations,1): st.markdown(f'<div class="limitation"><b>{i}. {lead}</b><br><span style="color:{MUTED}">{detail}</span></div>',unsafe_allow_html=True)
    st.caption("Behavioral synthetic-signal results are scenario-dependent.")


def main() -> None:
    st.sidebar.markdown('<div class="sidebar-brand"><strong>Behavioural Risk<br>Research</strong><span>Explainable pre-delinquency patterns</span></div>',unsafe_allow_html=True)
    choice=st.sidebar.radio("Navigate",PAGES,format_func=lambda p:f"{NAV[p]}   {p}",label_visibility="collapsed")
    missing = [name for name in REQUIRED_DATA_FILES if not (DATA / name).is_file()]
    if missing:
        st.error(
            "Dashboard data is incomplete. Missing file(s): "
            + ", ".join(f"dashboard/data/{name}" for name in missing)
            + ". Create these files by running dashboard/build_dashboard_data.py "
            "after the required model and subsample artifacts are available."
        )
        st.stop()
    try:
        files=[DATA/n for n in REQUIRED_DATA_FILES]
        version=max(p.stat().st_mtime_ns for p in files); data=load_all(version)
    except Exception as exc: st.error(f"Dashboard data could not be loaded: {exc}"); st.stop()
    try:
        branch=subprocess.check_output(["git","rev-parse","--abbrev-ref","HEAD"],cwd=ROOT,text=True,stderr=subprocess.DEVNULL).strip()
        commit=subprocess.check_output(["git","rev-parse","--short","HEAD"],cwd=ROOT,text=True,stderr=subprocess.DEVNULL).strip()
    except Exception: branch,commit="subsample-rebuild","local"
    st.sidebar.markdown(f'<div class="sidebar-foot">{SCOPE}<br>{branch} · {commit}</div>',unsafe_allow_html=True)
    {"Results":draw_results,"Borrower Explorer":draw_borrower_explorer,"Agent Workflow":draw_agent_workflow,
     "Portfolio":draw_portfolio,"Methodology and Limitations":draw_methodology}[choice](data)


if __name__ == "__main__": main()
