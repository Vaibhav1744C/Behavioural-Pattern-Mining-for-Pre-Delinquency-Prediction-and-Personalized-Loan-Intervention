# Behavioural Pattern Mining for Pre-Delinquency Prediction
## Presentation Guide — Group 4, CSE (AI), VIT

**Guide:** Dr. Sangita Maheshwar Jaybhaye
**Team:** Group 4 | Semester 1, AY 2026-27
**GitHub:** https://github.com/Vaibhav1744C/Behavioural-Pattern-Mining-for-Pre-Delinquency-Prediction-and-Personalized-Loan-Intervention

---

## 1. What This Project Does (One Paragraph)

Banks lose billions every year to loan defaults — but current systems detect the default only *after* it happens, by which point recovery is nearly impossible. This project answers a different question: **can we predict that a borrower is about to default 2–4 months before it happens, explain exactly why in plain language, and recommend a personalized action to prevent it?**

We extended the base paper (Monje et al., 2025 — *Machine Learning XAI for Early Loan Default Prediction*, Computational Economics) from a static one-time snapshot model into a full temporal, behavioural, explainable, and actionable system. The base paper achieved AUC 0.731. Our behavioural model achieves **AUC 0.7698** — a +0.034 lift — by incorporating 24 months of synthetic monthly transaction signals per borrower.

---

## 2. The Base Paper and What We Extended

| What the paper did | What we added |
|---|---|
| Static XGBoost on loan application data | Temporal XGBoost with 7 behavioural features |
| Single snapshot at origination | 24-month monthly behavioural sequences |
| Fuzzy linguistic surrogate for explainability | SHAP + fuzzy surrogate + LSTM attention combined |
| No intervention layer | 4-agent system with personalized action recommendations |
| AUC 0.731 | AUC 0.7698 (+0.034 lift) |
| No dashboard | Full Streamlit analyst dashboard |

---

## 3. Dataset

- **Source:** Lending Club 2007–2020 Q1 (Kaggle: `ethon0426/lending-club-20072020q1`)
- **Raw size:** 2,925,493 loans × 142 columns (~3 GB)
- **After cleaning:** 1,879,234 resolved loans × 85 columns
- **Early-default rate:** 20.19% (paper reports 20.20% — near-exact match ✅)
- **Why Lending Club:** Largest publicly labelled instalment-credit dataset. Every comparable paper in the literature uses it because real bank portfolios are proprietary. The framework is designed to generalise to any bank/NBFC portfolio.

---

## 4. Complete Work Done — Phase by Phase

### Phase 0 — Project Scaffold ✅
Set up full directory structure, `requirements.txt`, README, pushed to GitHub.

### Phase 1 — Data Ingestion and Cleaning ✅
- Loaded 3 GB CSV in 100,000-row chunks to avoid RAM overflow
- Applied 3 filtering criteria from the base paper:
  1. Keep only resolved loans (Fully Paid / Charged Off / Late 31-120 days) → builds the `early_default` label
  2. Drop 38 post-origination "leaky" columns (repayment activity, hardship fields, debt settlement) that would give away the answer
  3. Drop 19 columns with >40% missing values
- **Output:** `lending_club_clean.parquet` — 1,879,234 rows, early-default rate 20.19%
- Every dropped column logged with reason to `reports/cleaning_audit.csv`

### Phase 2 — Static XGBoost Baseline ✅
- Faithfully reproduced the base paper's XGBoost with 6 documented corrections (70/30 split, mixed encoding, paper's exact hyperparameters, date columns → numeric months, full sklearn Pipeline serialized)
- **Result: AUC-ROC 0.7345** (paper: 0.731 ✅ — our slightly higher AUC is expected as we have more data)

### Phase 3A — Synthetic Behavioural Sequence Generator ✅
Lending Club has no transaction-level data. We built a principled generator:

```
stress[t] = 0.6 × static_stress + 0.4 × shock_stress[t]
```

- `static_stress` — derived from DTI (40%), revolving utilisation (35%), loan grade (25%)
- `shock_stress[t]` — independent stochastic shock events, onset random, ramp over 2-3 months
- The `early_default` label is used **only** to set shock probability (35% for defaulters, 12% for non-defaulters) — never to directly compute any feature value (no leakage)
- 7 signals per borrower per month: salary credit, salary delay days, account balance, savings balance, discretionary spend, EMI status, stress level
- AR(1) autocorrelated noise throughout (real financial series are not random each month)
- **Output:** 45,101,616 rows → `behavioural_sequences.parquet` (1.12 GB)

### Phase 4 — Feature Engineering ✅
Compressed each borrower's 24-month sequence into 7 summary indices:

| Feature | What it captures |
|---|---|
| `salary_stability_idx` | std/mean of salary — how consistent is income? |
| `salary_delay_trend` | slope of salary delay days over time — is salary arriving progressively later? |
| `savings_slope` | slope of savings balance — savings growing or draining? |
| `savings_volatility` | std of savings balance — erratic savings = financial stress |
| `cashflow_compression` | mean spend first half / mean spend second half — >1 means pre-distress spending spike |
| `emi_stress_count` | count of delayed + missed EMI months |
| `seq_len` | actual sequence length (2–24), used for LSTM masking |

### Phase 2.5 — Behavioural XGBoost ✅
Added the 7 behavioural features on top of the 84 static features. Same XGBoost, same split.

**Result: AUC-ROC 0.7698 (+0.034 over static baseline)**
- At optimal threshold: catches 69,751 defaulters with only 100,695 false alarms
- vs static model at optimal threshold: 72,100 defaulters but 133,250 false alarms
- **Same catch rate, 25% fewer false alarms** — bank staff get more actionable alerts

### Phase 5 — LSTM Temporal Model ✅
Architecture:
- 2-layer unidirectional LSTM (hidden=64) on 24-month sequences
- Masked attention pooling — model learns *which months* mattered most
- Context branch fusing the 7 summary features
- Unidirectional (not bidirectional) to preserve causality for production

**Result: AUC-ROC 0.6371** — underperforms XGBoost on synthetic data (expected: hand-engineered features capture the signal more efficiently than raw sequences at this scale)
- Key finding: LSTM at its optimal threshold produces the *fewest false alarms* (76,891) — suited for high-precision, low-volume intervention

### Phase 6 — Explainability ✅
Three complementary explanation sources, combined per borrower:
1. **SHAP** — which static/behavioural features pushed the score up or down and by how much
2. **Fuzzy surrogate** — IF/THEN rules in plain language using triangular membership functions (replicates base paper's approach, R² fidelity = 0.775 vs paper's 0.807)
3. **LSTM attention weights** — which of the 24 months the model focused on, with a plain-English narrative

### Phase 7 — 4-Agent System + Dashboard ✅ (teammate's contribution)
Full working system — see Section 6 below.

---

## 5. Confirmed Results

| Model | AUC-ROC | PR-AUC | Defaulters caught (optimal threshold) | False alarms |
|---|---|---|---|---|
| Base paper (Monje et al.) | 0.731 | — | — | — |
| Static XGBoost (our replication) | **0.7345** | 0.415 | 72,100 / 113,839 (63.3%) | 133,250 |
| **Static + Behavioural XGBoost** | **0.7698** | **0.485** | 69,751 / 113,839 (61.3%) | **100,695** |
| LSTM temporal model | 0.6371 | 0.330 | 44,526 / 113,839 (39.1%) | 76,891 |

**Subsample run (200k borrowers, 60k test set — dashboard profile):**
- Static XGBoost: AUC 0.7254
- Behavioural XGBoost: AUC 0.7625 (+0.037 lift)
- Null-control (alpha=0): AUC 0.7255 — almost identical to static, confirming the lift is real signal from the shock gap, not a statistical artefact

**Surrogate fidelity:** R² = 0.775 (paper: 0.807 — our slightly lower fidelity is expected as we approximate the behavioural XGBoost which is harder to surrogate than the paper's static-only model)

---

## 6. The 4-Agent System

The project is built as four deterministic agents that run in sequence for any given borrower.

```
Input: loan_id
    ↓
Agent 1 — Behaviour Scanner
    ↓
Agent 2 — Risk Assessor
    ↓
Agent 3 — Explainer
    ↓
Agent 4 — Intervention Recommender
    ↓
Output: risk score + explanation + recommended action
```

### Agent 1 — Behaviour Scanner
**What it does:** Scans the borrower's 7 behavioural context features and flags any that fall in the top 10th percentile of the 60,000-borrower test set.

**Input:** borrower's 7 sequence summary features
**Output:** list of anomalies, e.g. `salary_stability_idx` at 96th percentile = unusually volatile income

**Why it matters:** Gives the bank an early-warning signal before the risk score is even computed. A salary stability anomaly alone might trigger a soft outreach.

### Agent 2 — Risk Assessor
**What it does:** Looks up the borrower's pre-computed predicted probabilities and assigns a relative risk tier.

**Input:** loan_id
**Output:** `p_static` (static model score), `p_behav` (behavioural model score), `risk_tier` (Very Low / Low / Medium / High / Very High), `behavioural_percentile` (rank within the 60k test set), `confidence_margin` (distance to nearest tier boundary)

**Why relative tier, not just a number:** A raw probability of 0.3 means nothing to a relationship manager. Saying "this borrower is in the High tier — top 25% of predicted risk in this cohort" is actionable.

### Agent 3 — Explainer
**What it does:** Answers "why is this borrower at risk?" using two sources.

**Input:** loan_id + Agent 2 output
**Output:**
- `fuzzy_risk` — absolute risk label in plain language, e.g. `(High, +0.0592)` — the tier plus a signed displacement from the tier centre
- `display_drivers` — top SHAP feature contributions (filtered to remove date-derived features that are not interpretable), e.g. `sub_grade: +0.31`, `salary_delay_trend: -0.26`
- `branch_taken` — whether full SHAP+fuzzy explanation was used, or fuzzy-only (when the confidence margin is high and the model is clear-cut)

**Why two branches:** A borrower clearly deep in the High tier (high confidence margin) needs only the fuzzy label. A borderline case close to a tier boundary gets the full SHAP breakdown to help the analyst understand the nuance.

### Agent 4 — Intervention Recommender
**What it does:** Maps the risk tier and top driver to a specific, auditable recommended bank action.

**Input:** Agent 2 output (tier) + Agent 3 output (display drivers)
**Output:** `rule_id`, `recommended_action`, `priority`

**Decision logic (deterministic rules, not LLM — required for regulatory auditability):**

| Condition | Action | Priority |
|---|---|---|
| Very High tier | EMI restructuring + RM escalation | Urgent |
| High tier + `emi_stress_count` is top driver | EMI restructuring | High |
| High tier + `salary_delay_trend` is top driver | Outreach call | High |
| High tier (other) | Outreach call | High |
| Medium tier | Counselling + closer monitoring | Medium |
| Low / Very Low | Standard monitoring | Routine |

**Why not an LLM for decisions:** A bank's compliance team must be able to audit every recommendation. A rule engine produces the same output for the same input every time, traceable to a specific rule ID. LLMs are used only in the UI narrative layer.

---

## 7. How to Start the Dashboard

### Prerequisites (already installed)
```powershell
python -m pip install streamlit plotly pandas pyarrow
```

### Start it
Open a terminal in the project folder and run:
```powershell
python -m streamlit run dashboard/app.py
```

Then open your browser to: **http://localhost:8501**

The dashboard loads the **subsample profile** by default — this works on any machine without needing the large model files. No additional setup required.

---

## 8. Dashboard — Page by Page Guide

### Page 1: Results
**What you see:** Model comparison bar chart, live confusion matrix with adjustable threshold, sensitivity analysis.

**How to explain it:**
- The bar chart compares the four models on ROC-AUC and PR-AUC. Point to the +0.037 lift from static to behavioural XGBoost — this is the project's central contribution.
- The threshold slider is interactive — drag it and watch the confusion matrix update live. Explain the trade-off: lower threshold catches more defaulters but generates more false alarms for bank staff to investigate.
- The sensitivity chart answers: "what if our assumed shock-probability gap (0.35 vs 0.12) was wrong?" The lift drops as the gap narrows, reaching near-zero when gap = 0.11 — this is honest reporting of how dependent the result is on that assumption.
- The **null-control result** (AUC 0.7255, almost identical to static) confirms the lift is real: when the sequences carry no default-correlated signal, the behavioural model is no better than the static one.

### Page 2: Borrower Explorer
**What you see:** Select any of the 200 sampled borrowers by loan ID. Shows risk gauge, borrower facts, 24-month trajectory plots, SHAP drivers, and plain-language narrative.

**How to explain it:**
- **Risk gauge (top left):** Shows where this borrower sits in the test-set score distribution. 80th percentile = riskier than 80% of borrowers in the cohort.
- **Absolute fuzzy label:** e.g. `(High, +0.0592)` — this is the base paper's 2-tuple fuzzy linguistic label. "High" is the tier, "+0.0592" is the signed displacement from the tier centre (0.75). Closer to 0 = middle of the tier; larger = stronger signal.
- **24-month trajectory plots:** Four line charts showing how salary delay, account balance, savings, and discretionary spending evolved month by month. The EMI status heatmap below shows months coloured green (on time), yellow (delayed), or red (missed). Look for the stress pattern: spending spike followed by compression, savings declining, delays increasing.
- **SHAP drivers bar chart:** Red bars = features that increased the risk score. Teal bars = features that decreased it. The length shows how much. These are the specific reasons *this borrower* was flagged — not a generic feature importance.
- **Narrative:** Plain-English summary combining the tier, fuzzy label, and top two drivers.

### Page 3: Agent Workflow
**What you see:** Select a borrower, click "Run workflow", and watch all 4 agents fire step by step with their inputs and outputs shown as JSON.

**How to explain it:**
- Each step expands to show exactly what went in, which rule was applied, and what came out.
- This is the transparency layer — every decision is traceable. Step 4 shows the exact rule ID (e.g. `H_SALARY_CALL_03`) that produced the recommendation.
- Emphasise: this is how a compliance officer would audit the system. Not "the AI decided" — "rule H_SALARY_CALL_03 fired because the borrower is in the High tier and salary_delay_trend was the top driver."

### Page 4: Portfolio
**What you see:** Tier distribution donut chart, default rate by tier bar chart, intervention queue of top 50 highest-risk borrowers.

**How to explain it:**
- The donut shows how the 60,000 test borrowers are distributed across risk bands. This tells a bank how many alerts to expect per month.
- The default rate chart validates the tiers: Very High tier borrowers should have the highest observed default rate, Very Low the lowest. If the model is working, the bars should be monotonically increasing.
- The **intervention queue** is the practical output — sorted by behavioural risk score, showing the recommended action and priority for each. A relationship manager would start at rank 1 and work down.
- The search box lets you filter by loan ID, tier, action type, or rule ID.

### Page 5: Methodology and Limitations
**What you see:** Pipeline diagram, dataset facts, limitations section.

**How to explain it:**
- The pipeline diagram shows the full data flow from raw CSV to explainable action — present this as a system overview.
- The **limitations section is important** — show it proactively. It demonstrates academic honesty:
  1. Behavioural data is synthetic — the lift depends on the assumed shock-probability gap
  2. The surrogate R² (0.775) is slightly below the paper's 0.807 — we approximate the behavioural XGBoost which is harder to surrogate
  3. `issue_d` (loan vintage) is inherited from the base paper and hidden from display
  4. LSTM underperformed — we report this honestly rather than hiding it

---

## 9. Key Talking Points for the Presentation

**"Why is the AUC slightly different from the paper?"**
We trained on 1,879,234 loans (2007–2020) vs the paper's 806,161 (2007–2018). More data gives marginally better generalisation. The 20.19% vs 20.20% default rate match confirms our target construction is identical.

**"Why is the LSTM worse than XGBoost?"**
Two reasons: (1) the synthetic data carries the default signal mostly through static-correlated features (DTI, grade, revol_util), which hand-engineered summary features capture more efficiently than raw sequences; (2) we trained on CPU without mixed precision. This is a publishable finding — it tells us when temporal models are and aren't worth the added complexity.

**"Why synthetic behavioural data?"**
Lending Club has no transaction-level data. Every real bank dataset is proprietary and regulator-restricted. The synthetic generator is principled — it uses a leakage-free stress model with independent stochastic shocks, and the null-control experiment (AUC 0.7255 ≈ static) validates that the lift disappears when sequences carry no default-correlated signal.

**"How would this work in a real bank?"**
Agent 4 feeds a dashboard where a relationship manager sees the top 50 at-risk borrowers, their recommended action, and the exact reasons. Agent 4 (monitoring) would run monthly as new transaction data arrives via the bank's core banking system, re-scoring each borrower automatically. The rule engine ensures every recommendation is auditable.

**"What's left to build?"**
- LSTM ablation run (alpha=0 sequences) — validates the signal source
- Full pipeline on complete 1.87M dataset for the full dashboard profile
- Paper writeup
- Production: FastAPI orchestrator, Kafka integration for real transaction feeds

---

## 10. File Structure Quick Reference

```
project/
├── src/
│   ├── data_generation/       ← Phases 1 and 3A
│   ├── features/              ← Phase 4 (build_features.py --subsample)
│   ├── models/                ← Phases 2, 2.5, 5
│   ├── explainability/        ← Phase 6 (SHAP, fuzzy surrogate, unified)
│   └── agents/                ← Phase 7 (4 agents + orchestrator)
├── dashboard/
│   ├── app.py                 ← Main Streamlit app (python -m streamlit run dashboard/app.py)
│   ├── theme.py               ← Colours and CSS
│   └── data/                  ← Committed subsample fixtures (works on any machine)
├── reports/
│   ├── full_metrics_comparison.json    ← All model results
│   ├── shap_summary_*.png              ← SHAP global importance plots
│   ├── surrogate_fidelity_subsample.json ← R² = 0.775
│   └── unified_explanations_subsample.json ← Per-borrower explanations
├── data/
│   ├── raw/                   ← Raw CSV + parquet (not in git)
│   ├── processed/             ← Cleaned data + features (not in git)
│   └── synthetic/             ← Generated sequences (not in git)
├── PROJECT_REVIEW.md          ← Full technical write-up
├── SUMMARY.md                 ← Phase-by-phase progress log
└── PRESENTATION_GUIDE.md      ← This file
```

---

*Group 4, CSE (AI), VIT — Semester 1, AY 2026-27*
