# Project Progress Summary

> Keep this file updated as each phase completes. Teammates: read this first
> before diving into any code.

---

## Current Status — Phase 6 (Explainability) — 🔄 In Progress

---

### Phase 2 — Baseline XGBoost Reproduction ✅ Complete

#### Script: `src/models/baseline_xgboost.py`

Faithfully reproduces the XGBoost baseline from the base paper. All 6 corrections from code review applied:

| # | Correction | Detail |
|---|---|---|
| 1 | **70/30 split** | Paper Section 4.4 states this explicitly — not 80/20 |
| 2 | **Mixed encoding** | `grade`, `sub_grade`, `emp_length` → ordinal/label encoded; `home_ownership`, `purpose`, `verification_status`, etc. → one-hot. `int_rate`/`revol_util` stripped of `%` → float. `term` stripped of `months` → float. Date cols → months since Jan-2007 |
| 3 | **No imbalance correction as default** | Paper Table 4: plain XGBoost (AUC 0.731) beats SMOTE (0.729), oversampling (0.662), undersampling (0.663). Vanilla is primary; `scale_pos_weight` runs as labeled secondary |
| 4 | **Paper's exact hyperparameters** | `max_depth=6`, `min_child_weight=5`, `gamma=10`, `objective='binary:logistic'`, `n_estimators=300` |
| 5 | **Date columns → numeric** | `issue_d`, `earliest_cr_line`, `sec_app_earliest_cr_line` converted to months-since-Jan-2007 |
| 6 | **Full pipeline serialized** | Fitted sklearn `Pipeline` (imputer + encoder + model) saved — not raw XGBoost model alone — for leak-free Phase 6 SHAP/surrogate use |

#### Verified output numbers

| Metric | Our Result | Paper Reports |
|---|---|---|
| AUC-ROC (vanilla XGBoost) | **0.7345** | **0.731** ✅ |
| AUC-ROC (scale_pos_weight) | 0.7352 | 0.729 (SMOTE) ✅ confirms paper finding |
| F1 (vanilla, threshold=0.5) | 0.1875 | — |
| Precision (vanilla) | 0.5861 | — |
| Recall (vanilla) | 0.1116 | — |

> **On AUC slightly exceeding paper (0.7345 vs 0.731):** expected — we train on 1.3M rows (2007–2020 Q1) vs the paper's ~564k (2007–2018 subset). More data = marginally better generalisation.

> **scale_pos_weight observation:** Confirms the paper's finding — imbalance correction improves recall dramatically (0.1116 → 0.6761) but at the cost of precision (0.5861 → 0.3381). For a pre-delinquency system, this trade-off is worth documenting but vanilla remains the primary baseline per the paper.

#### Artefacts saved

| File | Description |
|---|---|
| `src/models/xgb_baseline_pipeline.pkl` | Full fitted pipeline for Phase 6 |
| `data/processed/X_test.parquet` | Held-out test features for Phase 6 |
| `data/processed/y_test.parquet` | Held-out test labels for Phase 6 |
| `reports/baseline_results.json` | Full metrics JSON |

---

## What Has Been Done

### Phase 0 — Scaffold & Setup ✅

- Repo created and pushed to GitHub:
  `https://github.com/Vaibhav1744C/Behavioural-Pattern-Mining-for-Pre-Delinquency-Prediction-and-Personalized-Loan-Intervention.git`
- Project directory structure set up (`data/`, `src/`, `notebooks/`, `reports/`, `dashboard/`)
- `requirements.txt` written with all core dependencies (pandas, numpy, scikit-learn, xgboost, torch, pytorch-lightning, shap, scikit-fuzzy, kagglehub, etc.)
- `README.md` and `PHASE0_CHECKLIST.md` in place

---

### Phase 1 — Data Ingestion & Cleaning ✅ Complete

#### Dataset
- Source: Kaggle — `ethon0426/lending-club-20072020q1`
- Two files manually downloaded and placed in `data/raw/` (NOT committed to git):
  - `Loan_status_2007-2020Q3.gzip` — main dataset (plain CSV despite the `.gzip` extension)
  - `LCDataDictionary.xlsx` — column reference dictionary (153 entries)

#### Scripts written (both in `src/data_generation/`)

| Script | Purpose | Status |
|---|---|---|
| `load_raw.py` | Reads the raw CSV in 100k-row chunks, casts to a fixed Arrow schema, saves `data/raw/lending_club_raw.parquet` | ✅ Done |
| `clean_lending_club.py` | Phase 1 cleaning — filters to resolved loans, builds `early_default` target, drops leaky + sparse columns, saves `data/processed/lending_club_clean.parquet` | ✅ Done |

#### Verified output numbers

| Metric | Our Result | Paper Reports |
|---|---|---|
| Raw rows | 2,925,493 | 2,925,493 ✅ |
| Raw columns | 142 | 141 (paper had no `Unnamed: 0` index col) ✅ |
| Rows after filtering to resolved loans | 1,879,234 | 806,161 (paper used older subset) |
| Early-default rate | **20.19%** | **20.20%** ✅ near-exact match |
| Columns after cleaning | 71 | 57 |

> **On the row count gap:** The paper used a 2007–2018 subset; our dataset runs to 2020 Q1, so we have more resolved loans. The default rate match (20.19% vs 20.20%) confirms our target construction logic is correct.

> **On the column gap (71 vs 57):** The paper never publishes its final column list. We reconstructed their 3 filtering criteria faithfully. The 14-column gap is expected and documented in `reports/cleaning_audit.csv`. Every dropped column has a logged reason. We report this gap honestly in the paper rather than force-fitting to 57.

#### What `clean_lending_club.py` does (replicating the base paper)
1. **Target construction** — keeps only loans with a known outcome:
   - `Charged Off` → `early_default = 1`
   - `Late (31-120 days)` → `early_default = 1`
   - `Fully Paid` → `early_default = 0`
   - Excludes: Current, In Grace Period, <30 days late (unresolved)
2. **Leaky column removal** — dropped 38 post-origination fields:
   - Repayment activity (`total_pymnt`, `recoveries`, `last_pymnt_d`, etc.)
   - Post-origination credit data (`last_fico_range_*`, `last_credit_pull_d`)
   - Hardship programme fields (only populated for already-distressed loans)
   - Debt settlement fields (by definition post-default)
   - Identifiers and free text (`id`, `url`, `desc`, `emp_title`, etc.)
3. **Sparse column removal** — dropped 33 columns with >40% missing values
4. **Audit trail** — every dropped column logged with reason to `reports/cleaning_audit.csv`
5. **Output** — `data/processed/lending_club_clean.parquet` — 1,879,234 rows × 71 columns

---

### Phase 3A — Synthetic Behavioural Sequence Generator 🔄 (in progress)

#### Script: `src/data_generation/generate_behavioural_sequences.py`

Generates synthetic monthly behavioural sequences per borrower to feed the temporal model (Phase 5). Lending Club has no transaction-level data — this generator creates it from the static features using a principled stress model.

#### Core design (per Claude's spec)

**Stress signal (avoids circularity and leakage):**
```
stress[t] = alpha * static_stress + (1 - alpha) * shock_stress[t]
```
- `static_stress` — borrower-level constant from `dti`, `revol_util`, `grade` (scaled 0-1)
- `shock_stress[t]` — independent shock events, onset random, ramp over 2-3 months
- `early_default` label used ONLY to set shock probability (`p_shock_default=0.35` vs `p_shock_nondefault=0.12`) — never to directly set any feature value
- `alpha=0.6` for main run; companion run at `alpha=0` (pure independent shock) saved as ablation sanity check

**Output schema** — long format, one row per (loan_id, month):

| Column | Description |
|---|---|
| `loan_id` | join key back to static data |
| `month` | 0-indexed sequence position |
| `salary_credit` | monthly salary (AR(1) noise ~5% of mean) |
| `salary_delay_days` | days late vs nominal credit date (rises with stress) |
| `account_balance` | cumulative running balance |
| `savings_balance` | declines under sustained stress |
| `discretionary_spend` | spikes early on stress onset, compresses later |
| `emi_status` | `on_time` / `delayed` / `missed` (logistic fn of stress) |
| `stress_level` | internal signal, kept for validation plots |

**Key implementation details:**
- AR(1) noise throughout (`noise[t] = 0.7 * noise[t-1] + N(0,σ)`) — real financial series are autocorrelated
- Sequence length = `min(term_months, 24)` per borrower
- Chunked incremental parquet write (same pattern as `load_raw.py`) to handle 1.87M borrowers without OOM
- Validation on 1,000-5,000 borrower sample before full run — trajectory plots saved to `reports/generator_validation_plots/`
- Manifest saved to `reports/generator_manifest.json` for reproducibility

#### Outputs
| File | Description |
|---|---|
| `data/synthetic/behavioural_sequences.parquet` | Full long-format sequences (main run, alpha=0.6) |
| `data/synthetic/behavioural_sequences_alpha0.parquet` | Ablation run (alpha=0, pure shock) |
| `reports/generator_manifest.json` | Run parameters for reproducibility |
| `reports/generator_validation_plots/` | Trajectory plots for 10-15 sample borrowers |

#### To run
```bash
pip install matplotlib pyarrow
python src/data_generation/generate_behavioural_sequences.py
```

---

### Phase 4 — Feature Engineering 🔄 (in progress)

#### Script: `src/features/build_features.py`

Transforms long-format synthetic sequences into two artefacts the LSTM needs:

**1. `data/processed/lstm_sequences.parquet`** — per-month features (LSTM time-series input)

| Feature | How derived |
|---|---|
| `salary_delay_days_z` | Per-borrower z-score (borrower's own mean/std, not global) |
| `discretionary_spend_z` | Per-borrower z-score |
| `account_balance_z` | Per-borrower z-score |
| `savings_balance_z` | Per-borrower z-score |
| `delta_account_balance` | Month-over-month Δ in account_balance |
| `delta_savings_balance` | Month-over-month Δ in savings_balance |
| `emi_status_enc` | Ordinal: on_time=0, delayed=1, missed=2 |

**2. `data/processed/sequence_context_features.parquet`** — one row per borrower (static context fed alongside LSTM)

| Feature | How derived |
|---|---|
| `salary_stability_idx` | std / mean of `salary_credit` over sequence |
| `salary_delay_trend` | Linear regression slope of `salary_delay_days` over time |
| `savings_slope` | Linear regression slope of `savings_balance` |
| `savings_volatility` | std of `savings_balance` |
| `cashflow_compression` | mean spend in first half / mean spend in second half |
| `emi_stress_count` | count of delayed + missed months |
| `seq_len` | actual sequence length (2–24) — used for LSTM masking via `pack_padded_sequence` |

**Variable-length handling:** post-pad to max_len=24 with zeros; `seq_len` stored for masking — LSTM never processes padded timesteps.

**Validation checks:** zero NaNs after normalisation, spot-check 5–10 shock borrowers, shape/dtype report for both outputs.

---

### Phase 5 — LSTM Temporal Model 🔄 (in progress)

#### Scripts
| Script | Purpose |
|---|---|
| `src/models/prepare_lstm_data.py` | One-time data prep: pivot long parquet → dense `.npy` array, extract train/test split |
| `src/models/lstm_model.py` | Full LSTM training + evaluation |

#### Architecture (per Claude spec)

```
Input: (batch, 24, 7)  ← per-month features
  └─ nn.LSTM(input_size=7, hidden_size=64, num_layers=2, dropout=0.2, batch_first=True)
       └─ Attention pooling over timesteps (masked — padded positions → -inf before softmax)
            └─ pooled vector (batch, 64)

Context: (batch, 6)  ← sequence_context_features (seq_len used for masking only, not fed in)
  └─ Linear(6→16) → ReLU
       └─ dense vector (batch, 16)

Fusion: concat(64, 16) → Linear(80→64) → ReLU → Dropout(0.3) → Linear(64→1) → logit
Loss: BCEWithLogitsLoss (no pos_weight for primary run — matches Phase 2 finding)
```

**Key design decisions:**
- Unidirectional LSTM only — bidirectional would condition on future months (not causal)
- `pack_padded_sequence` + `enforce_sorted=False` for variable-length masking
- Attention mask set to `-inf` at padded positions before softmax — packing alone is not sufficient for attention
- Attention weights saved for ~50 test borrowers → `reports/lstm_attention_samples.json` (Phase 6 input)
- No gradient checkpointing / mixed precision — model is too small to need either

**Training:**
- Adam lr=1e-3, weight_decay=1e-5
- ReduceLROnPlateau on val AUC, patience=3, factor=0.5
- Early stopping on val AUC, patience=5
- Batch size 1024, up to 30 epochs
- Internal 90/10 validation split from train loan_ids (NOT the test set)

**Data prep (one-time, `prepare_lstm_data.py`):**
- Pivots `lstm_sequences.parquet` (long format) → dense `(1,879,234, 24, 7)` float32 array
- Saved as `data/processed/lstm_sequences_dense.npy` (~1.26 GB, fits in RAM)
- Extracts train loan_ids = all_ids - test_ids → `data/processed/train_loan_ids.parquet`
- Dataset.__getitem__ indexes the in-memory array — no I/O per sample during training

**Evaluation:**
- AUC-ROC on exact same test loan_ids as Phase 2 (X_test.parquet) — directly comparable to 0.7345
- Results → `reports/lstm_results.json` (same schema as `baseline_results.json`)
- Ablation (alpha=0): separate run via `--sequences-file` flag → `reports/lstm_results_alpha0.json`

**Target comparison table (after Phase 5):**

| Model | AUC-ROC |
|---|---|
| Static XGBoost (Phase 2) | 0.7345 |
| LSTM main run (alpha=0.6) | TBD |
| LSTM ablation (alpha=0) | TBD |

---
|---|---|---|
| Phase 1 | Data ingestion (`load_raw.py`) + cleaning (`clean_lending_club.py`) | ✅ Complete |
| Phase 2 | Baseline reproduction — XGBoost on the 71 cleaned features, replicate paper's AUC/F1 | ✅ Complete — AUC 0.7345 (paper: 0.731) |
| Phase 3A (full run) | Full 1.87M-borrower generation running as background job | ✅ Complete — 45,101,616 rows |
| Phase 4 | Feature engineering — LSTM sequences + sequence-level context features | ✅ Complete — 45M rows x 9 cols |
| Phase 5 | LSTM + attention temporal model | 🔄 In Progress (scripts ready, full training run pending) |
| Phase 6 | Explainability — SHAP + fuzzy surrogate + unified engine | 🔄 In Progress |
| Phase 5 | Temporal model — LSTM + attention (`src/models/`) | ⏳ |
| Phase 6 | Explainability — SHAP + fuzzy surrogate (`src/explainability/`) | ⏳ |
| Phase 7 | Intervention layer + dashboard (`src/intervention/`, `dashboard/`) | ⏳ |

---

## Key Decisions & Notes

- Raw dataset is **not committed to git** — each teammate downloads their own copy (see README for instructions)
- The base paper (Monje et al., 2025) does not publish its final column list — our 57-column target is a reconstruction from the three criteria they state; the audit CSV documents every decision
- All numeric columns stored as `float64` (not int) to handle NaN values correctly throughout the pipeline
- Parquet format used throughout for fast I/O — never re-read the raw CSV after `load_raw.py` runs

### Why a P2P (Lending Club) dataset justifies a bank-oriented pre-delinquency engine

We describe the end goal to the guide/reviewers as a pre-delinquency detection and
intervention engine applicable to lenders generally (banks, NBFCs, fintechs), while
validating it on a P2P dataset. This is deliberate, not a mismatch, for three reasons:

1. **The underlying problem is lender-agnostic.** The base paper itself states that
   P2P and bank lending share loan usage, credit evaluation, and periodic-repayment
   mechanics — the differences are intermediation, application speed, and interest
   rate, not the repayment/default dynamics we are modelling. A borrower missing
   EMIs behaves the same way, mechanically, whether the originating lender is a
   platform or a bank.
2. **No public bank-loan dataset of comparable scale and label quality exists.**
   Real bank portfolios are proprietary and regulator-restricted. Lending Club is
   the largest public dataset with resolved, labelled outcomes (fully paid /
   charged off / late) at the scale needed to train and validate a supervised
   default model — which is why every comparable paper in the literature (Chen et
   al. 2019, Zhou et al. 2019, Li et al. 2018, Ko et al. 2022, and others surveyed
   in the base paper's Table 1) also uses P2P data, for the same reason.
3. **The base paper's own conclusion names this as future work.** Monje et al.
   (2025) state that their methodology "can be applied in future work to explain
   default in other, non-P2P loans." Our project is, in effect, executing that
   stated future work: the dataset is P2P, but the framework (behavioural feature
   engineering, temporal modelling, explainability, intervention recommendation)
   is built to generalise to EMI-based consumer lending broadly.

**Framing to use in the synopsis/PPT/viva:** "We validate our framework on the
Lending Club P2P dataset — the largest publicly available labelled dataset for
installment credit default — as a proxy for the broader problem of pre-delinquency
detection in EMI-based consumer lending. The resulting methodology is designed to
generalise to bank and NBFC loan portfolios, which the base paper explicitly
identifies as future work."

---

### Phase 2.5 — XGBoost Static + Behavioural (Missing Comparison) 🔄 (scripts ready, run pending)

#### Script: `src/models/xgboost_behavioural.py`

The fast, cheap evidence that behavioural signal helps a tree model — needed BEFORE the expensive LSTM training finishes. Answers: "Do hand-engineered behavioural features lift XGBoost?" (separate question from whether raw sequences help an LSTM).

- Same hyperparams and 70/30 split as Phase 2
- Features: 71 static cols + 7 behavioural context cols = 78 cols total
- Outputs: `src/models/xgb_behavioural_pipeline.pkl`, `reports/xgboost_behavioural_results.json`
- Run: `python src/models/xgboost_behavioural.py`

---

### Phase 6 — Explainability 🔄 (scripts ready, run pending Phase 5 + Phase 2.5 first)

#### Three components, three scripts:

**1. `src/explainability/shap_explainer.py` — SHAP on both XGBoost models**
- `shap.TreeExplainer` (fast, exact for tree models) on static-only AND static+behavioural pipelines
- Global: beeswarm plots → `reports/shap_summary_static.png`, `reports/shap_summary_behavioural.png`
  - Visual evidence: "do behavioural features actually matter to the model?"
- Local: top-3 SHAP values for same 50 test borrowers as LSTM attention samples
  - Keyed by loan_id for joining in unified_explanation.py
  - → `reports/shap_local_samples.json`

**2. `src/explainability/fuzzy_surrogate.py` — 7-step surrogate tree (base paper Section 4)**
- Input: X = static + 7 behavioural context features (NOT raw sequences — keeps rules human-readable)
- Target p = black-box predicted probability (LSTM or XGBoost behavioural)
- `DecisionTreeRegressor(max_depth=4)` → IF/THEN rules
- R² fidelity reported and compared to paper's 0.806/0.808
- Splits fuzzified using triangular MFs: VL/L/M/H/VH, centers [0.0, 0.25, 0.5, 0.75, 1.0]
- Δ = signed displacement from nearest label center (paper's Eq. 1)
- Runs for BOTH LSTM and XGBoost behavioural — two rule sets to compare
- Outputs: `reports/surrogate_rules_lstm.json`, `reports/surrogate_rules_xgb_behavioural.json`, `reports/surrogate_fidelity.json`

**3. `src/explainability/unified_explanation.py` — Behavioural Explanation Engine**
- Joins all 3 sources for same 50 borrowers:
  - Fuzzy risk category (from surrogate, mapped via predicted probability)
  - Top-3 static SHAP drivers
  - Top-3 behavioural SHAP drivers
  - Top 2-3 attention months (from lstm_attention_samples.json)
  - Templated narrative: "The model focused most on months X-Y, where the borrower showed [signal_1] and [signal_2]"
- → `reports/unified_explanations_sample.json`

**Output schema per borrower:**
```json
{
  "loan_id": ...,
  "predicted_risk_linguistic": "(High, +0.1250)",
  "top_static_drivers": [...],
  "top_behavioural_drivers": [...],
  "attention_focus_months": [14, 15, 16],
  "attention_narrative": "The model focused most on months 14-16, where the borrower showed salary delayed 8 days and EMI missed."
}
```

**This file is:**
- Best midsem/demo artifact — read 3-4 entries aloud
- Direct input to Phase 7 intervention layer

**Run order:**
```bash
# 1. Static+behavioural XGBoost (fast, ~15 min)
python src/models/xgboost_behavioural.py

# 2. LSTM full training (after prepare_lstm_data.py)
python src/models/prepare_lstm_data.py
python src/models/lstm_model.py

# 3. SHAP (needs both XGBoost pipelines)
python src/explainability/shap_explainer.py

# 4. Fuzzy surrogate (needs LSTM checkpoint + XGBoost behavioural pipeline)
python src/explainability/fuzzy_surrogate.py

# 5. Unified explanation (needs all 3 above outputs)
python src/explainability/unified_explanation.py
```

---

Monje, Carrasco & Sanchez-Montanes (2025). *Machine Learning XAI for Early Loan Default Prediction*. Computational Economics, 67, 4033–4062.