"""
shap_explainer.py
-----------------
Phase 6: SHAP explanations for both XGBoost models.

1. Global SHAP: beeswarm/summary plots for static-only and static+behavioural
   models — visual evidence for which features matter most.

2. Local SHAP: per-borrower SHAP values for the same ~50 test borrowers
   whose LSTM attention weights were saved in Phase 5. Keyed by loan_id
   so unified_explanation.py can join all three explanation sources.

Usage:
    python src/explainability/shap_explainer.py

Outputs:
    reports/shap_summary_static.png
    reports/shap_summary_behavioural.png
    reports/shap_local_samples.json   (50 borrowers, same loan_ids as lstm_attention_samples.json)

Requirements:
    pip install shap matplotlib pandas pyarrow scikit-learn
"""

import json
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT            = Path(__file__).resolve().parents[2]
CLEAN_PATH      = ROOT / "data" / "processed" / "lending_club_clean.parquet"
CONTEXT_PATH    = ROOT / "data" / "processed" / "sequence_context_features.parquet"
X_TEST_PATH     = ROOT / "data" / "processed" / "X_test.parquet"
STATIC_PKL      = ROOT / "src"  / "models"    / "xgb_baseline_pipeline.pkl"
BEHAVIOURAL_PKL = ROOT / "src"  / "models"    / "xgb_behavioural_pipeline.pkl"
ATTN_PATH       = ROOT / "reports"            / "lstm_attention_samples.json"
SHAP_STATIC_PNG = ROOT / "reports"            / "shap_summary_static.png"
SHAP_BEHAV_PNG  = ROOT / "reports"            / "shap_summary_behavioural.png"
SHAP_LOCAL_OUT  = ROOT / "reports"            / "shap_local_samples.json"

BEHAVIOURAL_COLS = [
    "salary_stability_idx", "salary_delay_trend", "savings_slope",
    "savings_volatility", "cashflow_compression", "emi_stress_count", "seq_len",
]

# Reuse the same preprocessing helpers from xgboost_behavioural.py
import sys
sys.path.insert(0, str(ROOT / "src" / "models"))
from xgboost_behavioural import (
    preprocess_features, ORDINAL_MAPPINGS, NOMINAL_COLS,
    DROP_COLS, DATE_COLS, PCT_COLS, TERM_COL,
)
from baseline_xgboost import preprocess_features as preprocess_static


# ── Helper: build aligned X_test for each model ───────────────────────────────

def load_static_test():
    """Reproduce the static-only test set (same as Phase 2)."""
    df = pd.read_parquet(CLEAN_PATH)
    if "loan_id" not in df.columns:
        df = df.reset_index().rename(columns={"index": "loan_id"})
    from sklearn.model_selection import train_test_split
    _, test = train_test_split(
        df, test_size=0.30, random_state=42, stratify=df["early_default"]
    )
    loan_ids = test["loan_id"].values
    y_test   = test["early_default"].astype("int8")
    X_test   = test.drop(columns=["early_default", "loan_id"])
    X_test   = preprocess_static(X_test)
    return X_test, y_test, loan_ids


def load_behavioural_test():
    """Reproduce the static+behavioural test set."""
    df  = pd.read_parquet(CLEAN_PATH)
    ctx = pd.read_parquet(CONTEXT_PATH)
    if "loan_id" not in df.columns:
        df = df.reset_index().rename(columns={"index": "loan_id"})
    if "loan_id" not in ctx.columns:
        ctx = ctx.reset_index().rename(columns={"index": "loan_id"})
    ctx   = ctx[["loan_id"] + BEHAVIOURAL_COLS]
    merged= df.merge(ctx, on="loan_id", how="inner")
    from sklearn.model_selection import train_test_split
    _, test = train_test_split(
        merged, test_size=0.30, random_state=42,
        stratify=merged["early_default"]
    )
    loan_ids = test["loan_id"].values
    y_test   = test["early_default"].astype("int8")
    X_test   = test.drop(columns=["early_default", "loan_id"])
    X_test   = preprocess_features(X_test)
    return X_test, y_test, loan_ids


# ── Global SHAP ───────────────────────────────────────────────────────────────

def run_global_shap(pipeline, X_test: pd.DataFrame, title: str, out_path: Path,
                    max_display: int = 20, sample_n: int = 2000):
    """
    Compute and plot global SHAP beeswarm for a fitted sklearn Pipeline.
    Samples up to sample_n rows for speed.
    """
    print(f"  Computing global SHAP: {title} ...")
    prep  = pipeline.named_steps["prep"]
    clf   = pipeline.named_steps["clf"]

    # Transform test data through preprocessor
    X_tr  = prep.transform(X_test)

    # Get feature names from ColumnTransformer
    try:
        feat_names = prep.get_feature_names_out()
    except Exception:
        feat_names = [f"f{i}" for i in range(X_tr.shape[1])]

    # Sample for speed
    if X_tr.shape[0] > sample_n:
        idx  = np.random.default_rng(42).choice(X_tr.shape[0], sample_n, replace=False)
        X_s  = X_tr[idx]
    else:
        X_s  = X_tr

    explainer    = shap.TreeExplainer(clf)
    shap_values  = explainer.shap_values(X_s)

    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values, X_s,
        feature_names=feat_names,
        max_display=max_display,
        show=False,
        plot_type="dot",
    )
    plt.title(title, fontsize=12, pad=12)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path.relative_to(ROOT)}")
    return explainer, feat_names


# ── Local SHAP ────────────────────────────────────────────────────────────────

def run_local_shap(
    pipeline_static,
    pipeline_behav,
    X_test_static:  pd.DataFrame,
    X_test_behav:   pd.DataFrame,
    loan_ids_static: np.ndarray,
    loan_ids_behav:  np.ndarray,
    target_loan_ids: list,
) -> list:
    """
    Compute per-borrower SHAP values for target_loan_ids.
    Returns list of dicts keyed by loan_id.
    """
    print(f"  Computing local SHAP for {len(target_loan_ids)} borrowers...")

    def get_top_shap(pipeline, X_df, loan_ids_arr, lid, top_n=3):
        mask = loan_ids_arr == lid
        if not mask.any():
            return []
        idx   = np.where(mask)[0][0]
        prep  = pipeline.named_steps["prep"]
        clf   = pipeline.named_steps["clf"]
        X_tr  = prep.transform(X_df.iloc[[idx]])
        try:
            feat_names = list(prep.get_feature_names_out())
        except Exception:
            feat_names = [f"f{i}" for i in range(X_tr.shape[1])]
        exp   = shap.TreeExplainer(clf)
        sv    = exp.shap_values(X_tr)[0]
        pairs = sorted(zip(feat_names, sv.tolist()),
                       key=lambda x: abs(x[1]), reverse=True)
        return [{"feature": f, "shap_value": round(v, 5)} for f, v in pairs[:top_n]]

    records = []
    for lid in target_loan_ids:
        top_static = get_top_shap(pipeline_static, X_test_static,
                                  loan_ids_static, lid)
        top_behav  = get_top_shap(pipeline_behav,  X_test_behav,
                                  loan_ids_behav,  lid)
        records.append({
            "loan_id":            int(lid),
            "top_static_shap":    top_static,
            "top_behavioural_shap": top_behav,
        })

    return records


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 64)
    print("Phase 6 — SHAP Explainer")
    print("=" * 64)

    # ── Load pipelines ────────────────────────────────────────────────────────
    for path, name in [(STATIC_PKL, "static"), (BEHAVIOURAL_PKL, "behavioural")]:
        if not path.exists():
            raise FileNotFoundError(
                f"{name} pipeline not found: {path}\n"
                f"Run src/models/{'baseline' if name=='static' else 'xgboost_behavioural'}"
                f"_xgboost.py first"
            )

    print("\nLoading pipelines...")
    with open(STATIC_PKL,      "rb") as f: pipeline_static = pickle.load(f)
    with open(BEHAVIOURAL_PKL, "rb") as f: pipeline_behav  = pickle.load(f)

    # ── Load test sets ────────────────────────────────────────────────────────
    print("Loading test sets...")
    X_test_s, y_s, ids_s = load_static_test()
    X_test_b, y_b, ids_b = load_behavioural_test()
    print(f"  Static test   : {X_test_s.shape}")
    print(f"  Behavioural test: {X_test_b.shape}")

    # ── Global SHAP plots ─────────────────────────────────────────────────────
    print("\n[1/2] Global SHAP...")
    run_global_shap(pipeline_static, X_test_s,
                    "SHAP — Static-only XGBoost", SHAP_STATIC_PNG)
    run_global_shap(pipeline_behav,  X_test_b,
                    "SHAP — Static+Behavioural XGBoost", SHAP_BEHAV_PNG)

    # ── Local SHAP for same 50 borrowers as LSTM attention ────────────────────
    print("\n[2/2] Local SHAP (50 borrowers)...")
    if not ATTN_PATH.exists():
        print(f"  WARNING: {ATTN_PATH.name} not found — skipping local SHAP")
        print("  Run lstm_model.py first to generate attention samples")
        local_records = []
    else:
        with open(ATTN_PATH) as f:
            attn_samples = json.load(f)
        target_ids = [s["loan_id"] for s in attn_samples]
        local_records = run_local_shap(
            pipeline_static, pipeline_behav,
            X_test_s, X_test_b,
            ids_s, ids_b,
            target_ids,
        )

    SHAP_LOCAL_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(SHAP_LOCAL_OUT, "w") as f:
        json.dump(local_records, f, indent=2)
    print(f"  Saved: {SHAP_LOCAL_OUT.relative_to(ROOT)}")

    print("\n" + "=" * 64)
    print("Phase 6 SHAP complete.")
    print(f"  Global static plot   : {SHAP_STATIC_PNG.relative_to(ROOT)}")
    print(f"  Global behavioural   : {SHAP_BEHAV_PNG.relative_to(ROOT)}")
    print(f"  Local samples (50)   : {SHAP_LOCAL_OUT.relative_to(ROOT)}")
    print("=" * 64)


if __name__ == "__main__":
    main()
