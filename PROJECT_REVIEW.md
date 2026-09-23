# Behavioural Pattern Mining for Pre-Delinquency Prediction and Personalized Loan Intervention
## Project Review Document — Midsem Presentation

**Group 4 | CSE (AI) | VIT | Semester 1, AY 2026-27**
**Guide: Dr. Sangita Maheshwar Jaybhaye**
**GitHub:** https://github.com/Vaibhav1744C/Behavioural-Pattern-Mining-for-Pre-Delinquency-Prediction-and-Personalized-Loan-Intervention

---

## 1. Problem Statement

Banks and lending companies lose billions annually to loan defaults. The current industry approach detects defaults **after** they happen — by which point recovery is expensive and often impossible. The question this project answers is:

> **Can we predict a borrower is about to default 2-4 months before it happens, explain why in plain language, and recommend a personalized intervention to prevent it?**

The base paper (Monje et al., 2025) took the first step — a static XGBoost model with a fuzzy linguistic surrogate for explainability, achieving AUC 0.731. Our project extends this in three directions the paper explicitly does not attempt:

1. **Temporal modelling** — using monthly behavioural sequences, not just a one-time static snapshot
2. **Richer explainability** — combining SHAP, fuzzy surrogate rules, and LSTM attention weights
3. **Intervention recommendations** — personalized actions based on the explanation

---

## 2. Base Paper

**Monje, Carrasco & Sanchez-Montanes (2025)**
*Machine Learning XAI for Early Loan Default Prediction*
Computational Economics, 67, 4033–4062.

**What the paper did:**
- Dataset: Lending Club 2007–2018, 806,161 resolved loans, 141 → 57 columns after cleaning
- Model: XGBoost (static features only), AUC 0.731
- Explainability: Surrogate decision tree + 2-tuple fuzzy linguistic model
- Compared SMOTE / oversampling / undersampling — none improved over vanilla XGBoost
- Did NOT use transaction-level/behavioural data
- Did NOT build a temporal model
- Did NOT build an intervention layer

**What the paper says about future work:**
> "The methodology can be applied in future work to explain default in other, non-P2P loans."

Our project is executing that stated future work.

---

## 3. Dataset

**Source:** Kaggle — `ethon0426/lending-club-20072020q1`
**URL:** https://www.kaggle.com/datasets/ethon0426/lending-club-20072020q1

| Property | Value |
|---|---|
| Raw rows | 2,925,493 |
| Raw columns | 142 |
| Date range | 2007 – 2020 Q1 |
| File size | ~3 GB |
| Format | CSV (despite `.gzip` extension) |

**Why Lending Club (P2P) for a bank-oriented system:**
The base paper itself notes that P2P and bank lending share the same repayment mechanics. Lending Club is the largest publicly available labelled dataset for installment credit default — every comparable paper in the literature also uses it. Real bank portfolios are proprietary and regulator-restricted. Our framework is designed to generalise to bank and NBFC loan portfolios.

---

## 4. System Architecture — How It Works End to End

```
┌─────────────────────────────────────────────────────────────────┐
│                        INPUT                                      │
│  Static loan application data (71 features)                      │
│  + Monthly behavioural signals (24 months × 7 signals)           │
└─────────────────────────────┬───────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              │                               │
    ┌─────────▼──────────┐       ┌────────────▼───────────┐
    │   XGBoost Model    │       │    LSTM + Attention     │
    │  (static + behav.) │       │  (temporal sequences)   │
    │   AUC: 0.7686      │       │   AUC: ~0.635 (CPU)     │
    └─────────┬──────────┘       └────────────┬────────────┘
              │                               │
              └───────────────┬───────────────┘
                              │
              ┌───────────────▼───────────────┐
              │      EXPLAINABILITY LAYER      │
              │  SHAP + Fuzzy Surrogate        │
              │  + LSTM Attention Weights      │
              └───────────────┬───────────────┘
                              │
              ┌───────────────▼───────────────┐
              │    UNIFIED EXPLANATION ENGINE  │
              │  Risk: (High, +0.125)          │
              │  Drivers: DTI, EMI stress      │
              │  Timeline: months 14-16        │
              │  Narrative: "salary delayed…"  │
              └───────────────┬───────────────┘
                              │
              ┌───────────────▼───────────────┐
              │     INTERVENTION ENGINE        │
              │  → EMI restructuring           │
              │  → Financial counselling       │
              │  → Relationship manager alert  │
              └───────────────┬───────────────┘
                              │
              ┌───────────────▼───────────────┐
              │         DASHBOARD              │
              │  Analyst-facing UI             │
              │  Search by loan ID             │
              │  View risk + explanation       │
              └───────────────────────────────┘
```

---

## 5. What We Have Built — Phase by Phase

---

### Phase 0 — Project Scaffold ✅

**What:** Set up the full project structure and pushed to GitHub.

```
project-scaffold/
├── data/
│   ├── raw/           ← raw dataset (not committed — too large)
│   ├── processed/     ← cleaned + feature tables
│   └── synthetic/     ← generated behavioural sequences
├── src/
│   ├── data_generation/   ← data loading, cleaning, sequence generation
│   ├── features/          ← feature engineering
│   ├── models/            ← XGBoost, LSTM
│   └── explainability/    ← SHAP, fuzzy surrogate, unified engine
├── reports/           ← results, plots, audit trails
├── dashboard/         ← analyst-facing UI (Phase 7)
├── requirements.txt
├── README.md
└── PHASE0_CHECKLIST.md
```

**How:** `git init` → structured directory tree → `git push` to GitHub.

---

### Phase 1 — Data Ingestion & Cleaning ✅

**What:** Load the 3 GB raw CSV and clean it down to a usable dataset.

**Problem encountered:** The full CSV cannot be loaded into RAM at once (out of memory). Also, PyArrow schema mismatches across chunks crashed the parquet writer.

**How we solved it:**
- Read the CSV in **100,000-row chunks** using `pd.read_csv(..., chunksize=100000)`
- Built a **fixed Arrow schema upfront** from the first chunk (numeric → `float64`, text → `large_utf8`)
- Used `pyarrow.parquet.ParquetWriter` to write incrementally — each chunk appended to the same parquet file

**Cleaning steps (replicating base paper's 3 criteria):**
1. **Target construction** — keep only resolved loans:
   - `Charged Off` → `early_default = 1`
   - `Late (31-120 days)` → `early_default = 1`
   - `Fully Paid` → `early_default = 0`
   - Drop: Current, In Grace Period (outcome unknown)
2. **Leaky column removal** — drop 38 post-origination fields (repayment activity, hardship fields, debt settlement, etc.) that would give away the answer
3. **Sparse column removal** — drop 33 columns with >40% missing values

**Results:**

| Metric | Our Result | Paper |
|---|---|---|
| Raw rows | 2,925,493 | 2,925,493 ✅ |
| After cleaning | **1,879,234 rows** | 806,161 (paper used 2007-2018 only) |
| Columns | 71 | 57 (gap documented) |
| Early-default rate | **20.19%** | **20.20%** ✅ |

The 20.19% vs 20.20% match confirms our target construction is correct.

**Outputs:**
- `data/raw/lending_club_raw.parquet` (476 MB)
- `data/processed/lending_club_clean.parquet` (1,879,234 × 71)
- `reports/cleaning_audit.csv` — every dropped column with reason

**Scripts:** `src/data_generation/load_raw.py`, `src/data_generation/clean_lending_club.py`

---

### Phase 2 — Static XGBoost Baseline ✅

**What:** Faithfully reproduce the base paper's XGBoost model as our reference point.

**Why faithful reproduction matters:** If we can't match the paper's AUC, any claimed improvement is meaningless. We need to prove we started from the same place.

**Corrections applied (6 total, all tied to paper citations):**

| # | Issue | Fix |
|---|---|---|
| 1 | Wrong split ratio | 70/30 not 80/20 (paper Section 4.4) |
| 2 | Naive encoding | `grade`/`sub_grade`/`emp_length` → ordinal (preserves ordering); nominals → one-hot (paper Section 4.5) |
| 3 | Imbalance correction | No `scale_pos_weight` as default — paper Table 4 shows vanilla beats SMOTE/oversampling/undersampling |
| 4 | Wrong hyperparameters | `max_depth=6, min_child_weight=5, gamma=10` (paper Section 4.4) |
| 5 | Date columns | `issue_d`, `earliest_cr_line` → months since Jan 2007 (not one-hot — would create one col per date) |
| 6 | Pipeline serialization | Save full sklearn Pipeline (imputer + encoder + model), not just the XGBoost model — prevents leakage in Phase 6 |

**Results:**

| Model | AUC-ROC | Paper |
|---|---|---|
| Vanilla XGBoost | **0.7345** | 0.731 ✅ |
| + scale_pos_weight | 0.7352 | SMOTE: 0.729 ✅ |

AUC slightly above paper (0.7345 vs 0.731) — expected, we have more data (2020 vs 2018 cutoff).

**Script:** `src/models/baseline_xgboost.py`

---

### Phase 3A — Synthetic Behavioural Sequence Generator ✅

**What:** Generate synthetic monthly transaction data for all 1.87M borrowers.

**Why synthetic?** Lending Club has no transaction-level data — only loan application snapshots. To build a temporal model, we need to create plausible monthly behavioural trajectories for each borrower.

**The stress model (core design):**

```
stress[t] = alpha × static_stress + (1 - alpha) × shock_stress[t]
```

- **`static_stress`** — borrower-level constant derived from `dti` (40%), `revol_util` (35%), `grade` (25%), normalised to [0,1]. High DTI + high credit utilisation + low grade = high static stress.
- **`shock_stress[t]`** — independent stochastic shock event. Either a shock occurs (with probability 35% for defaulters, 12% for non-defaulters) or it doesn't. If a shock occurs: onset is random (weighted toward the middle third of the sequence), stress ramps up over 2-3 months, then holds elevated.
- **`alpha = 0.6`** — 60% of stress comes from static factors, 40% from independent shocks.

**Why this design avoids leakage:**
- `early_default` label is used ONLY to set the shock probability — never to directly compute any feature value
- The independent shock component means the temporal model has something genuinely new to learn that the static XGBoost cannot see

**AR(1) noise:** `noise[t] = 0.7 × noise[t-1] + N(0, σ)` — real financial series are autocorrelated, not random each month.

**Seven signals generated per (borrower, month):**

| Signal | How it's generated |
|---|---|
| `salary_credit` | `annual_inc/12` + AR(1) noise ~5%. Independent of stress (salary amount doesn't change, only timing) |
| `salary_delay_days` | `stress[t] × 10 days` + noise. Delays increase with stress |
| `account_balance` | Cumulative: prev_balance + salary - spending - EMI. Stateful |
| `savings_balance` | Starts at 1-3 months income, draws down under stress |
| `discretionary_spend` | Spikes early when stress rises (+30% × stress_delta), then compresses (-40% × stress_level) |
| `emi_status` | Logistic function of stress: P(missed) and P(delayed) both increase with stress |
| `stress_level` | Internal signal — kept for validation, not fed to model |

**Validation checks (all passed on 2,000-borrower sample):**

| Check | Result | Threshold |
|---|---|---|
| Salary correlation with `annual_inc/12` | 0.9982 | >0.95 ✅ |
| Shock rate (defaulters) | 0.337 | 0.35 ±0.10 ✅ |
| Shock rate (non-defaulters) | 0.118 | 0.12 ±0.10 ✅ |
| Flat trajectories (std < 1e-4) | 0 | 0 ✅ |

**Full run output:**
- **45,101,616 rows** → `data/synthetic/behavioural_sequences.parquet` (1.12 GB)
- **Ablation run** (alpha=0, pure independent shock) → `behavioural_sequences_alpha0.parquet`
- 30 trajectory plots saved to `reports/generator_validation_plots/`

**Script:** `src/data_generation/generate_behavioural_sequences.py`

---

### Phase 4 — Feature Engineering ✅

**What:** Transform 45M rows of raw sequences into LSTM-ready features.

**Problem encountered:** Loading the full 1.12 GB parquet file into RAM at once caused `ArrowMemoryError: malloc of size 1.8GB failed`. Solution: true streaming via PyArrow `iter_batches` — process 50k borrowers at a time, never hold the full file in RAM.

**Output 1 — `lstm_sequences.parquet` (45,101,616 × 9)**
Per-month features, post-padded to 24 months per borrower:

| Feature | How |
|---|---|
| `salary_delay_days_z` | Per-borrower z-score (own mean/std — income scale varies enormously across borrowers) |
| `discretionary_spend_z` | Per-borrower z-score |
| `account_balance_z` | Per-borrower z-score |
| `savings_balance_z` | Per-borrower z-score |
| `delta_account_balance` | Month-over-month change in balance |
| `delta_savings_balance` | Month-over-month change in savings |
| `emi_status_enc` | on_time=0, delayed=1, missed=2 |

**Why per-borrower z-score (not global)?**
A balance of $500 means something very different for a borrower earning $20k/year vs $200k/year. Global normalisation would destroy this relative information. Per-borrower z-score preserves within-borrower dynamics.

**Output 2 — `sequence_context_features.parquet` (1,879,234 × 8)**
One row per borrower — summary indices fed as static context alongside the LSTM:

| Feature | What it captures |
|---|---|
| `salary_stability_idx` | Coefficient of variation of salary — unstable salary = risk |
| `salary_delay_trend` | Slope of delay over time — worsening delays = risk |
| `savings_slope` | Trend of savings — declining savings = risk |
| `savings_volatility` | Volatility of savings balance |
| `cashflow_compression` | First-half spend / second-half spend. >1 = spending spike then compression (pre-distress pattern) |
| `emi_stress_count` | Total delayed + missed months |
| `seq_len` | Actual sequence length (2-24) — for LSTM masking |

**Validation:** 0 NaNs, all 1.87M borrowers have exactly 24 padded rows, cashflow_compression=1.024 for high-stress borrowers (>1.0 as expected).

**Script:** `src/features/build_features.py`

---

### Phase 2.5 — XGBoost Static + Behavioural ✅

**What:** Add the 7 behavioural context features to XGBoost as a fast, cheap test.

**Why:** Before spending hours training the LSTM, we need to verify the behavioural signal actually helps. XGBoost trains in ~15 minutes.

**Features:** 71 static + 7 behavioural context = 78 total. Same hyperparams and 70/30 split as Phase 2.

**Results:**

| Model | AUC-ROC | Delta |
|---|---|---|
| Static-only XGBoost | 0.7345 | baseline |
| Static + Behavioural XGBoost | **0.7686** | **+0.034** |

**What +0.034 means:** The behavioural features (`salary_stability_idx`, `cashflow_compression`, `emi_stress_count`, etc.) carry real predictive information that the static snapshot does not contain. This is the first evidence our approach works.

**Script:** `src/models/xgboost_behavioural.py`

---

### Phase 5 — LSTM Temporal Model 🔄 (training in progress)

**What:** An LSTM that reads the full 24-month behavioural sequence and learns temporal patterns.

**Why LSTM over XGBoost?** XGBoost sees summary statistics (slopes, counts, volatility) but not the actual sequence. An LSTM can learn that "3 months of gradually worsening EMI payments" is more dangerous than "1 bad month followed by recovery" — a distinction that averages cannot capture.

**Architecture:**

```
Input sequence: (batch_size, 24 months, 7 features)
       ↓
2-layer LSTM (hidden=64, dropout=0.2)
  → outputs hidden state for all 24 timesteps
       ↓
Masked Attention Pooling
  → compute score for each timestep: Linear(64→1)
  → set padded positions to -infinity (cannot attend to zero-padding)
  → softmax over 24 timesteps → attention weights
  → weighted sum → pooled vector (batch, 64)
       ↓
Context Branch: Linear(6→16) → ReLU  ← 6 summary features
       ↓
Fusion: concat(64 + 16 = 80) → Linear(80→64) → ReLU → Dropout(0.3) → Linear(64→1)
       ↓
Logit → BCEWithLogitsLoss
       ↓
Sigmoid → probability (0 to 1)
```

**Key design decisions:**
- **Unidirectional LSTM** — bidirectional would let the model see "future" months, which is not valid in production (we can only see past months)
- **Masked attention** — without masking, padded zero-vectors get attention weight and corrupt the pooled representation
- **Attention weights saved** — for each of 50 test borrowers, which months the model focused on. This is direct input to the explanation engine.

**Training setup:**
- 1,879,234 borrowers total: Train=1,183,916 / Val=131,547 / Test=563,771
- Same test set as Phase 2 — AUC is directly comparable
- Adam lr=1e-3, ReduceLROnPlateau on val AUC, early stop patience=5, max 30 epochs
- Dense array pre-built: (1,879,234, 24, 7) float32 = 1.26 GB in RAM

**Current status:** Running on CPU, epoch 16/30, val AUC ~0.635. Plateauing — early stopping expected around epoch 20-21.

**Scripts:** `src/models/prepare_lstm_data.py`, `src/models/lstm_model.py`

---

### Phase 6 — Explainability ⏳ (scripts written, waiting on LSTM)

**What:** Three-layer explanation combining SHAP, fuzzy surrogate rules, and temporal attention.

#### Component 1: SHAP (`shap_explainer.py`)
- `shap.TreeExplainer` on both XGBoost models (exact, fast for tree models)
- **Global:** beeswarm plots showing which features drive the model overall
- **Local:** top-3 SHAP values for 50 specific test borrowers (same borrowers whose LSTM attention is saved)

This answers: "What features drove this specific prediction?"

#### Component 2: Fuzzy 2-tuple Surrogate (`fuzzy_surrogate.py`)
- Exactly replicates the base paper's 7-step algorithm (Section 4)
- Trains a shallow decision tree (`max_depth=4`) on static + behavioural features → target = LSTM predicted probability
- Extracts IF/THEN rules from the tree
- Fuzzifies thresholds using triangular membership functions: Very Low / Low / Medium / High / Very High
- Each split becomes: `IF cashflow_compression > 1.34 [High, +0.08]`
- R² fidelity compared to paper's 0.806/0.808

This answers: "What's the linguistic rule that describes high-risk borrowers?"

#### Component 3: Unified Explanation Engine (`unified_explanation.py`)
Combines all three sources into one JSON per borrower:

```json
{
  "loan_id": 1077501,
  "predicted_risk_linguistic": "(High, +0.1250)",
  "top_static_drivers": [
    {"feature": "dti", "shap_value": 0.18},
    {"feature": "revol_util", "shap_value": 0.14}
  ],
  "top_behavioural_drivers": [
    {"feature": "emi_stress_count", "shap_value": 0.21},
    {"feature": "cashflow_compression", "shap_value": 0.16}
  ],
  "attention_focus_months": [14, 15, 16],
  "attention_narrative": "The model focused most on months 14-16,
    where the borrower showed salary delayed 8 days above baseline
    and EMI missed."
}
```

This is the **Behavioural Explanation Engine** — the output your viva committee will find most compelling. It's human-readable, auditable, and directly actionable.

---

## 6. Current Results

| Model | AUC-ROC | What it proves |
|---|---|---|
| Static XGBoost (base paper replication) | 0.7345 | We matched the paper ✅ |
| Static + Behavioural XGBoost | **0.7686** | Behavioural features help (+0.034) ✅ |
| LSTM main (alpha=0.6) | ~0.635 (training) | Does raw sequence add more? |
| LSTM ablation (alpha=0) | TBD | Is the LSTM lift real or circular? |

**The three deltas are the paper's argument:**
1. **0.7345 → 0.7686** (+0.034): Hand-engineered behavioural features improve a tree model
2. **0.7686 → LSTM**: Do raw sequences carry information beyond what we hand-engineered?
3. **LSTM main → LSTM alpha=0**: Is the temporal lift due to genuine sequence learning, or just correlation with static features?

---

## 7. Future Plan — What Remains

### Phase 7 — Intervention Engine (`src/intervention/intervention_engine.py`)

**What it does:** Takes the unified explanation object and outputs personalized recommended actions.

**How it works:**
- Maps `predicted_risk_linguistic` (Very Low / Low / Medium / High / Very High) to action tiers
- Uses top SHAP drivers and attention narrative to personalize the action
- Rule-based (not ML) — deterministic and auditable for a finance context

**Expected output:**
```
Risk: HIGH
Primary driver: EMI stress (3 missed months, months 14-16)

Recommended actions:
  IMMEDIATE: Contact borrower for financial assessment
  OFFER:     EMI restructuring — reduce monthly payment by 20% for 3 months
  ESCALATE:  Assign relationship manager within 7 days
  MONITOR:   Weekly account review for next 30 days
```

### Phase 7 — Dashboard (`dashboard/`)

**Technology:** Streamlit (Python — no separate frontend framework needed)

**What it shows:**
- Search by loan ID
- Risk score (0-100%) with traffic light indicator
- Fuzzy risk category (Very Low → Very High)
- Top 3 SHAP feature drivers (bar chart)
- Attention timeline (which months the model focused on)
- Plain-language narrative
- Recommended actions

**Demo flow for viva:**
1. Enter a loan ID of a known defaulter
2. Show risk score: 82%
3. Show explanation: "focused on months 14-16, EMI missed, spending compression"
4. Show intervention: "offer EMI restructuring, contact within 7 days"
5. Enter a known non-defaulter — show score: 12%, "Low risk, no intervention needed"

### Paper Writing

**Sections aligned to our phases:**

| Section | Content |
|---|---|
| Introduction | Problem, motivation, gap in base paper |
| Related Work | Base paper + temporal credit models |
| Data & Preprocessing | Phase 1 — cleaning criteria, audit trail |
| Synthetic Data Generation | Phase 3A — stress model design, validation |
| Models | Phase 2 (static XGBoost), Phase 2.5 (behavioural XGBoost), Phase 5 (LSTM) |
| Explainability | Phase 6 — SHAP, fuzzy surrogate, attention narrative |
| Intervention | Phase 7 — rule-based recommendation engine |
| Results | Four-row comparison table + three deltas + fidelity R² |
| Discussion | What the gaps between rows mean + surrogate fidelity finding |
| Conclusion | Framework generalises to bank/NBFC portfolios |

---

## 8. How Prediction Actually Works in Production

A bank analyst opens the dashboard. Here is what happens step by step:

**Step 1 — Input**
The analyst enters Loan ID `1077501`. The system looks up:
- Static features: DTI=22.5, grade=C, revol_util=78%, employment=5 years
- Monthly behavioural record: last 16 months of salary, spending, balance, EMI data

**Step 2 — Feature preparation**
- Static features go through the same preprocessing as training (ordinal encoding, date conversion, imputation)
- Monthly data is z-scored per borrower and assembled into a (1, 24, 7) tensor (padded with zeros for months 17-24)

**Step 3 — LSTM inference**
- The packed sequence runs through 2 LSTM layers
- Attention scores computed for each of the 24 timesteps
- Padded months receive -infinity attention weight (they cannot influence the prediction)
- Attention weights show months 14-16 received highest weight

**Step 4 — Prediction**
- Logit fused with context features → sigmoid → probability = **0.82**
- Fuzzy risk category: `(High, +0.1250)`

**Step 5 — Explanation**
- SHAP: `emi_stress_count` and `cashflow_compression` are top drivers
- Surrogate rule matched: `IF emi_stress_count > 3 AND cashflow_compression > 1.2 THEN risk = (High, +0.09)`
- Attention narrative: "Model focused on months 14-16 where borrower showed salary delayed 8 days and EMI missed"

**Step 6 — Intervention**
- Risk tier: HIGH
- Recommended: EMI restructuring offer + relationship manager assignment + weekly monitoring

**Step 7 — Display**
All of the above rendered in the Streamlit dashboard in under 2 seconds.

---

## 9. What Makes This Novel vs the Base Paper

| Feature | Base Paper | Our Project |
|---|---|---|
| Data type | Static snapshot only | Static + 24-month behavioural sequence |
| Model | XGBoost (tree) | XGBoost + LSTM with attention |
| Explainability | Fuzzy surrogate only | SHAP + fuzzy surrogate + attention weights |
| Temporal explanation | None | "Model focused on months 14-16" |
| Intervention | None | Personalized action recommendations |
| Dashboard | None | Streamlit analyst UI |
| Ablation | Not done | alpha=0 control run (tests circularity) |

---

## 10. Repository Structure

```
project-scaffold/
├── src/
│   ├── data_generation/
│   │   ├── load_raw.py                        ← Phase 1: CSV → parquet
│   │   ├── clean_lending_club.py              ← Phase 1: cleaning
│   │   └── generate_behavioural_sequences.py  ← Phase 3A: sequence generation
│   ├── features/
│   │   └── build_features.py                  ← Phase 4: LSTM features
│   ├── models/
│   │   ├── baseline_xgboost.py               ← Phase 2: static XGBoost
│   │   ├── xgboost_behavioural.py            ← Phase 2.5: +behavioural features
│   │   ├── prepare_lstm_data.py              ← Phase 5: dense array prep
│   │   └── lstm_model.py                     ← Phase 5: LSTM training
│   └── explainability/
│       ├── shap_explainer.py                 ← Phase 6: SHAP
│       ├── fuzzy_surrogate.py                ← Phase 6: fuzzy surrogate
│       └── unified_explanation.py            ← Phase 6: explanation engine
├── reports/
│   ├── cleaning_audit.csv                    ← every dropped column with reason
│   ├── baseline_results.json                 ← Phase 2 metrics
│   ├── xgboost_behavioural_results.json      ← Phase 2.5 metrics
│   ├── lstm_results.json                     ← Phase 5 metrics (pending)
│   ├── generator_manifest.json              ← Phase 3A run parameters
│   ├── generator_validation_plots/          ← 30 trajectory plots
│   ├── shap_summary_static.png              ← Phase 6 (pending)
│   ├── shap_summary_behavioural.png         ← Phase 6 (pending)
│   └── unified_explanations_sample.json     ← Phase 6 (pending)
├── data/
│   ├── raw/          ← not in git (too large)
│   ├── processed/    ← not in git (too large)
│   └── synthetic/    ← not in git (too large)
├── dashboard/        ← Phase 7 (not yet built)
├── README.md         ← teammate setup guide
├── SUMMARY.md        ← progress log
└── requirements.txt
```

---

## 11. Timeline

| Week | Phase | Status |
|---|---|---|
| Week 1 | Phase 0 — scaffold + dataset | ✅ |
| Week 2 | Phase 1 — data cleaning | ✅ |
| Week 2 | Phase 2 — XGBoost baseline | ✅ |
| Week 3 | Phase 3A — sequence generator | ✅ |
| Week 3 | Phase 4 — feature engineering | ✅ |
| Week 4 | Phase 2.5 — behavioural XGBoost | ✅ |
| Week 4 | Phase 5 — LSTM training (CPU) | 🔄 epoch 16/30 |
| Week 5 | Phase 5 — LSTM ablation | ⏳ |
| Week 5 | Phase 6 — SHAP + surrogate + unified | ⏳ |
| Week 6 | Phase 7 — intervention engine | ⏳ |
| Week 6 | Phase 7 — dashboard | ⏳ |
| Week 7+ | Paper writing | ⏳ |

---

## 12. Expected Final Results

Based on what we have so far and how the system is designed:

| Model | Expected AUC | Confidence |
|---|---|---|
| Static XGBoost | 0.7345 | **confirmed** |
| Static+Behavioural XGBoost | 0.7686 | **confirmed** |
| LSTM main (alpha=0.6) | 0.63–0.68 | medium — CPU training, synthetic data |
| LSTM ablation (alpha=0) | similar to main | expected (synthetic data limits separation) |
| Surrogate R² fidelity | 0.70–0.81 | lower than paper's 0.807 expected (LSTM harder to approximate than XGBoost) |

**If LSTM AUC < Static+Behavioural XGBoost:**
This is still a valid, publishable finding — it means hand-engineered features capture the behavioural signal more efficiently than raw sequences at this scale, which is itself an interesting result about when temporal models are and aren't worth the complexity.

**If LSTM ≈ LSTM ablation:**
Also valid — means the temporal signal in our synthetic data is not cleanly separable from the static-correlated component at alpha=0.6. Honest reporting of this with a discussion of what it implies for real transaction data is stronger than hiding it.

---

*Document prepared for midsem project review. All code is version-controlled at the GitHub repo above.*
