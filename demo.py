"""
demo.py
-------
Live demonstration script for project review.
Shows the full pipeline from data → prediction → explanation in ~10 seconds.

Usage:
    python demo.py
"""

import json
import pickle
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src" / "models"))

from xgboost_behavioural import preprocess_features, BEHAVIOURAL_COLS

SEP = "=" * 65


def section(title):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


def main():
    print(SEP)
    print("  BEHAVIOURAL PATTERN MINING FOR PRE-DELINQUENCY PREDICTION")
    print("  Live Demo — Group 4, VIT CSE (AI)")
    print(SEP)

    # ── Step 1: Dataset ───────────────────────────────────────────────────────
    section("STEP 1 — Dataset")
    df = pd.read_parquet(ROOT / "data/processed/lending_club_clean.parquet")
    if "loan_id" not in df.columns:
        df = df.reset_index().rename(columns={"index": "loan_id"})
    print(f"  Total loans (cleaned)  : {len(df):,}")
    print(f"  Features               : {df.shape[1]} columns")
    print(f"  Default rate           : {df['early_default'].mean():.2%}")
    print(f"  Non-defaulters         : {(df['early_default']==0).sum():,}")
    print(f"  Defaulters             : {(df['early_default']==1).sum():,}")

    # ── Step 2: Model results ─────────────────────────────────────────────────
    section("STEP 2 — Model Performance Comparison")
    r1 = json.load(open(ROOT / "reports/baseline_results.json"))
    r2 = json.load(open(ROOT / "reports/xgboost_behavioural_results.json"))
    r3 = json.load(open(ROOT / "reports/lstm_results.json"))
    auc1 = r1["results"][0]["auc_roc"]
    auc2 = r2["results"][0]["auc_roc"]
    auc3 = r3["results"][0]["auc_roc"]

    print(f"  {'Model':<40} {'AUC-ROC':>8}  {'Delta':>8}")
    print(f"  {'-'*58}")
    print(f"  {'Static XGBoost (base paper replication)':<40} {auc1:>8}  {'baseline':>8}")
    print(f"  {'Static + Behavioural XGBoost':<40} {auc2:>8}  {'+'+str(round(auc2-auc1,4)):>8}  ← our contribution")
    print(f"  {'LSTM (temporal model)':<40} {auc3:>8}  {str(round(auc3-auc1,4)):>8}")
    print(f"\n  PR-AUC (better for imbalanced data):")
    fm = json.load(open(ROOT / "reports/full_metrics_comparison.json"))
    for m in fm:
        if "t=0.5" in m["label"]:
            print(f"    {m['label']:<44} PR-AUC: {m['pr_auc']}")

    # ── Step 3: Live prediction ───────────────────────────────────────────────
    section("STEP 3 — Live Prediction on Real Borrowers")
    ctx = pd.read_parquet(ROOT / "data/processed/sequence_context_features.parquet")
    if "loan_id" not in ctx.columns:
        ctx = ctx.reset_index().rename(columns={"index": "loan_id"})

    merged = df.merge(ctx[["loan_id"] + BEHAVIOURAL_COLS], on="loan_id", how="inner")
    _, test = train_test_split(
        merged, test_size=0.30, random_state=42, stratify=merged["early_default"]
    )

    with open(ROOT / "src/models/xgb_behavioural_pipeline.pkl", "rb") as f:
        pipe = pickle.load(f)

    defaulters     = test[test["early_default"] == 1].head(3)
    nondefaulters  = test[test["early_default"] == 0].head(3)

    print(f"\n  {'Loan ID':<12} {'True Label':<14} {'Predicted Risk':>15} {'Verdict':>12}")
    print(f"  {'-'*55}")
    for _, row in pd.concat([defaulters, nondefaulters]).iterrows():
        X    = preprocess_features(row.drop(["early_default", "loan_id"]).to_frame().T)
        prob = pipe.predict_proba(X)[0][1]
        tier = "HIGH RISK" if prob > 0.5 else "LOW RISK"
        true = "DEFAULTER" if row["early_default"] == 1 else "NON-DEFAULTER"
        correct = "✓" if (prob > 0.5) == (row["early_default"] == 1) else "✗"
        print(f"  {int(row.loan_id):<12} {true:<14} {prob:>14.3f}  {tier} {correct}")

    # ── Step 4: Explanation (attention) ──────────────────────────────────────
    section("STEP 4 — Temporal Explanation (LSTM Attention)")
    attn = json.load(open(ROOT / "reports/lstm_attention_samples.json"))
    print(f"\n  Showing attention focus months for 5 sample borrowers:")
    print(f"  {'Loan ID':<12} {'True':>6} {'Pred Prob':>10} {'Peak Attention Months':>25}")
    print(f"  {'-'*55}")
    for s in attn[:5]:
        weights = s["attn_weights"]
        top3    = sorted(range(len(weights)), key=lambda i: weights[i], reverse=True)[:3]
        top3    = sorted(top3)
        print(f"  {s['loan_id']:<12} {s['true_label']:>6} {s['pred_prob']:>10.3f} "
              f"  months {top3}")

    # ── Step 5: Behavioural features ─────────────────────────────────────────
    section("STEP 5 — Behavioural Features (the 7 new signals)")
    ctx_sample = ctx.head(5)[["loan_id", "salary_stability_idx",
                               "cashflow_compression", "emi_stress_count",
                               "salary_delay_trend", "seq_len"]]
    print(f"\n  Sample borrowers with their behavioural indices:")
    print(ctx_sample.to_string(index=False))

    # ── Summary ───────────────────────────────────────────────────────────────
    section("SUMMARY")
    print(f"""
  What we built:
    1. Replicated base paper XGBoost           AUC: {auc1}
    2. Added behavioural features (+0.034)     AUC: {auc2}  ← KEY RESULT
    3. Built LSTM temporal model               AUC: {auc3}
    4. Explainability: SHAP + fuzzy surrogate + attention narrative
    5. 4-agent architecture designed (Phases 7-8)

  Key finding: 7 behavioural features improve default prediction
  by +{round(auc2-auc1,4)} AUC points over the static baseline.
  Behavioural XGBoost catches 69,751 defaulters at 40.9% precision
  — actionable signal for a bank's relationship management team.
    """)
    print(SEP)


if __name__ == "__main__":
    main()
