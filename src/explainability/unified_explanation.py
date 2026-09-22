"""
unified_explanation.py
----------------------
Phase 6: Combine SHAP + fuzzy surrogate + LSTM attention into one
explanation object per borrower.

For each of the 50 sample borrowers (loan_ids from lstm_attention_samples.json):
  - Fuzzy risk category  from surrogate_rules_lstm.json
  - Top static drivers   from shap_local_samples.json
  - Top behavioural      from shap_local_samples.json
  - Attention months     from lstm_attention_samples.json
  - Narrative sentence   from templated generator (deterministic, auditable)

Output structure per borrower:
{
  "loan_id": ...,
  "predicted_risk_linguistic": "(High, +0.1250)",
  "top_static_drivers": [{"feature": ..., "shap_value": ...}, ...],
  "top_behavioural_drivers": [...],
  "attention_focus_months": [14, 15, 16],
  "attention_narrative": "The model focused most on months 14-16, where the
                          borrower showed delayed salary credit (+8 days above
                          their baseline) and rising cash-flow compression."
}

Saved to reports/unified_explanations_sample.json
This file is:
  - The best midsem/demo artifact (read 3-4 aloud)
  - The direct input to Phase 7 intervention layer

Usage:
    python src/explainability/unified_explanation.py

Requirements:
    pip install pandas pyarrow numpy
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).resolve().parents[2]
ATTN_PATH     = ROOT / "reports" / "lstm_attention_samples.json"
SHAP_PATH     = ROOT / "reports" / "shap_local_samples.json"
RULES_PATH    = ROOT / "reports" / "surrogate_rules_lstm.json"
SEQ_PATH      = ROOT / "data"    / "synthetic" / "behavioural_sequences.parquet"
CONTEXT_PATH  = ROOT / "data"    / "processed" / "sequence_context_features.parquet"
OUTPUT_PATH   = ROOT / "reports" / "unified_explanations_sample.json"

# Per-month features to check for narrative (human-readable names)
NARRATIVE_FEATURES = {
    "salary_delay_days": "salary delay",
    "discretionary_spend": "discretionary spending",
    "account_balance": "account balance",
    "savings_balance": "savings balance",
    "emi_status": "EMI payment status",
}

TOP_N_ATTN_MONTHS = 3   # how many peak-attention months to highlight


# ── Attention narrative generator ─────────────────────────────────────────────

def get_raw_values_for_months(
    loan_id: int, months: list, seq_df: pd.DataFrame
) -> dict:
    """
    Pull raw (non-z-scored) feature values for a borrower at specific months.
    Returns dict of {month: {feature: value}}.
    """
    sub = seq_df[seq_df["loan_id"] == loan_id].sort_values("month")
    if sub.empty:
        return {}
    result = {}
    for m in months:
        row = sub[sub["month"] == m]
        if row.empty:
            continue
        row = row.iloc[0]
        result[m] = {
            "salary_delay_days":   int(pd.to_numeric(row.get("salary_delay_days", 0), errors="coerce") or 0),
            "discretionary_spend": round(float(pd.to_numeric(row.get("discretionary_spend", 0), errors="coerce") or 0), 2),
            "account_balance":     round(float(pd.to_numeric(row.get("account_balance", 0), errors="coerce") or 0), 2),
            "savings_balance":     round(float(pd.to_numeric(row.get("savings_balance", 0), errors="coerce") or 0), 2),
            "emi_status":          str(row.get("emi_status", "on_time")),
        }
    return result


def get_borrower_baseline(loan_id: int, seq_df: pd.DataFrame) -> dict:
    """Compute per-borrower mean for each numeric feature (the 'own baseline')."""
    sub = seq_df[seq_df["loan_id"] == loan_id]
    if sub.empty:
        return {}
    result = {}
    for col in ["salary_delay_days", "discretionary_spend",
                "account_balance", "savings_balance"]:
        vals = pd.to_numeric(sub[col], errors="coerce").dropna()
        result[col] = float(vals.mean()) if len(vals) > 0 else 0.0
    return result


def build_narrative(
    loan_id: int,
    attn_weights: list,
    seq_len: int,
    raw_vals: dict,
    baseline: dict,
) -> tuple[list, str]:
    """
    Identify top attention months and build a deterministic narrative sentence.
    Returns (top_months, narrative_string).
    """
    # Only look at actual (non-padded) months
    valid_weights = [(t, w) for t, w in enumerate(attn_weights) if t < seq_len]
    if not valid_weights:
        return [], "Insufficient sequence data for narrative."

    top_months = sorted(valid_weights, key=lambda x: x[1], reverse=True)
    top_months = [t for t, _ in top_months[:TOP_N_ATTN_MONTHS]]
    top_months_sorted = sorted(top_months)

    # Find which features deviated most from borrower baseline in attended months
    signals = []
    for m in top_months_sorted:
        if m not in raw_vals:
            continue
        mv = raw_vals[m]

        # Salary delay
        delay = mv.get("salary_delay_days", 0)
        b_delay = baseline.get("salary_delay_days", 0)
        if delay > b_delay + 2:
            signals.append(f"salary delayed {delay} days (+{delay - int(b_delay):.0f} days above baseline)")

        # EMI status
        emi = mv.get("emi_status", "on_time")
        if emi in ("delayed", "missed"):
            signals.append(f"EMI {emi}")

        # Account balance trend
        bal   = mv.get("account_balance", 0)
        b_bal = baseline.get("account_balance", 0)
        if b_bal != 0 and (bal - b_bal) / abs(b_bal) < -0.15:
            signals.append(f"account balance down {abs((bal - b_bal)/b_bal)*100:.0f}% from baseline")

        # Spending compression or spike
        spend   = mv.get("discretionary_spend", 0)
        b_spend = baseline.get("discretionary_spend", 0)
        if b_spend != 0:
            ratio = (spend - b_spend) / abs(b_spend)
            if ratio > 0.25:
                signals.append(f"spending spike (+{ratio*100:.0f}% above baseline)")
            elif ratio < -0.20:
                signals.append(f"spending compression ({ratio*100:.0f}% below baseline)")

        if signals:
            break   # stop after first attended month with clear signals

    month_range = (f"month {top_months_sorted[0]}"
                   if len(top_months_sorted) == 1
                   else f"months {top_months_sorted[0]}-{top_months_sorted[-1]}")

    if signals:
        signal_str = " and ".join(signals[:2])
        narrative  = (f"The model focused most on {month_range}, where the borrower "
                      f"showed {signal_str}.")
    else:
        narrative  = (f"The model focused most on {month_range} of the borrower's "
                      f"24-month behavioural sequence.")

    return top_months_sorted, narrative


# ── Fuzzy risk lookup ─────────────────────────────────────────────────────────

def lookup_fuzzy_risk(loan_id: int, rules: list, X_row: dict | None) -> str:
    """
    Find the surrogate rule that applies to this borrower and return
    the linguistic risk label. Falls back to the rule with the closest
    leaf_value if feature matching isn't feasible.
    """
    if not rules:
        return "(Unknown, +0.0000)"
    # Without full row data available, use the median rule as a fallback
    # (proper matching requires the full feature vector — done in batch mode
    # when run with X_sur; here we just report the median leaf label)
    leaf_values = sorted([r["risk_score"] for r in rules])
    median_val  = leaf_values[len(leaf_values) // 2]
    for rule in rules:
        if abs(rule["risk_score"] - median_val) < 0.01:
            return rule["risk_linguistic"]
    return rules[len(rules) // 2]["risk_linguistic"]


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 64)
    print("Phase 6 — Unified Explanation Engine")
    print("=" * 64)

    # ── Load inputs ───────────────────────────────────────────────────────────
    for path, name in [
        (ATTN_PATH,  "lstm_attention_samples.json"),
        (SHAP_PATH,  "shap_local_samples.json"),
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"{name} not found at {path}\n"
                "Run shap_explainer.py and lstm_model.py first."
            )

    print("Loading explanation inputs...")
    with open(ATTN_PATH) as f: attn_samples = json.load(f)
    with open(SHAP_PATH) as f: shap_samples = json.load(f)

    # Load fuzzy rules (optional — use if available)
    fuzzy_rules = []
    if RULES_PATH.exists():
        with open(RULES_PATH) as f:
            fuzzy_rules = json.load(f).get("rules", [])
        print(f"  Loaded {len(fuzzy_rules)} fuzzy rules from surrogate")
    else:
        print(f"  NOTE: {RULES_PATH.name} not found — fuzzy risk will be placeholder")
        print("  Run fuzzy_surrogate.py to generate it")

    # Build lookup maps
    shap_by_id = {s["loan_id"]: s for s in shap_samples}

    # Load raw sequences for narrative (stream only needed columns)
    print("Loading raw sequences for narrative generation...")
    seq_cols = ["loan_id", "month", "salary_delay_days", "discretionary_spend",
                "account_balance", "savings_balance", "emi_status"]
    target_ids = set(s["loan_id"] for s in attn_samples)

    import pyarrow.parquet as pq
    pf = pq.ParquetFile(SEQ_PATH)
    seq_pieces = []
    for batch in pf.iter_batches(batch_size=1_000_000, columns=seq_cols):
        df_b = batch.to_pandas()
        df_b = df_b[df_b["loan_id"].isin(target_ids)]
        if not df_b.empty:
            seq_pieces.append(df_b)
        if len(seq_pieces) > 0 and sum(len(p) for p in seq_pieces) >= len(target_ids) * 24:
            break
    seq_df = pd.concat(seq_pieces, ignore_index=True) if seq_pieces else pd.DataFrame()
    print(f"  Loaded {len(seq_df):,} sequence rows for {seq_df['loan_id'].nunique() if not seq_df.empty else 0} borrowers")

    # ── Build unified explanation per borrower ────────────────────────────────
    print(f"\nBuilding unified explanations for {len(attn_samples)} borrowers...")
    explanations = []

    for sample in attn_samples:
        lid      = sample["loan_id"]
        attn_w   = sample["attn_weights"]
        seq_len  = sample["seq_len"]
        pred_prob= sample["pred_prob"]
        true_lbl = sample["true_label"]
        correct  = sample["correct"]

        # SHAP
        shap_rec = shap_by_id.get(lid, {})
        top_static= shap_rec.get("top_static_shap", [])
        top_behav = shap_rec.get("top_behavioural_shap", [])

        # Fuzzy risk
        # Map predicted probability -> fuzzy label directly (most reliable without full X_sur row)
        from fuzzy_surrogate import fuzzify_value
        risk_label, risk_delta = fuzzify_value(pred_prob)
        risk_linguistic = f"({risk_label}, {risk_delta:+.4f})"

        # Narrative
        raw_vals = get_raw_values_for_months(lid, list(range(seq_len)), seq_df)
        baseline = get_borrower_baseline(lid, seq_df)
        top_months, narrative = build_narrative(lid, attn_w, seq_len, raw_vals, baseline)

        explanations.append({
            "loan_id":                  lid,
            "true_label":               true_lbl,
            "predicted_prob":           pred_prob,
            "predicted_correct":        correct,
            "predicted_risk_linguistic":risk_linguistic,
            "top_static_drivers":       top_static,
            "top_behavioural_drivers":  top_behav,
            "attention_focus_months":   top_months,
            "attention_narrative":      narrative,
        })

    # ── Save ──────────────────────────────────────────────────────────────────
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(explanations, f, indent=2)

    print(f"\nSaved {len(explanations)} unified explanations -> {OUTPUT_PATH.relative_to(ROOT)}")

    # ── Print 3 sample explanations ───────────────────────────────────────────
    print("\n" + "=" * 64)
    print("SAMPLE EXPLANATIONS (3 of 50)")
    print("=" * 64)
    for ex in explanations[:3]:
        print(f"\n  Loan {ex['loan_id']} | true={ex['true_label']} "
              f"prob={ex['predicted_prob']} correct={ex['predicted_correct']}")
        print(f"  Risk (linguistic): {ex['predicted_risk_linguistic']}")
        if ex["top_static_drivers"]:
            print(f"  Top static driver : {ex['top_static_drivers'][0]['feature']} "
                  f"(SHAP={ex['top_static_drivers'][0]['shap_value']})")
        if ex["top_behavioural_drivers"]:
            print(f"  Top behav. driver : {ex['top_behavioural_drivers'][0]['feature']} "
                  f"(SHAP={ex['top_behavioural_drivers'][0]['shap_value']})")
        print(f"  Attention months  : {ex['attention_focus_months']}")
        print(f"  Narrative         : {ex['attention_narrative']}")

    print("\n" + "=" * 64)
    print("Phase 6 Unified Explanation complete.")
    print("=" * 64)


if __name__ == "__main__":
    main()
