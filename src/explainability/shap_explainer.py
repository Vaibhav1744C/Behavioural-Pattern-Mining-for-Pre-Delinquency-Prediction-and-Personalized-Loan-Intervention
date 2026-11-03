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
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb

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

def load_fixed_test_ids(loan_ids_file: Path) -> set[int]:
    selected = pd.read_parquet(loan_ids_file)
    if not {"loan_id", "split"}.issubset(selected.columns):
        raise KeyError("--loan-ids-file must contain loan_id and split columns")
    if selected["loan_id"].duplicated().any():
        raise ValueError("--loan-ids-file contains duplicate loan_id values")
    return set(selected.loc[selected["split"] == "test", "loan_id"].astype("int64"))


def load_static_test(loan_ids_file: Path | None = None):
    """Reproduce the static-only test set (same as Phase 2)."""
    df = pd.read_parquet(CLEAN_PATH)
    if "loan_id" not in df.columns:
        df = df.reset_index().rename(columns={"index": "loan_id"})
    if loan_ids_file is None:
        from sklearn.model_selection import train_test_split
        _, test = train_test_split(
            df, test_size=0.30, random_state=42, stratify=df["early_default"]
        )
    else:
        ids = load_fixed_test_ids(loan_ids_file)
        test = df[df["loan_id"].isin(ids)].copy()
        if set(test["loan_id"].astype("int64")) != ids:
            raise ValueError("One or more selected test IDs are absent from clean data")
    loan_ids = test["loan_id"].values
    y_test   = test["early_default"].astype("int8")
    X_test   = test.drop(columns=["early_default", "loan_id"])
    X_test   = preprocess_static(X_test)
    return X_test, y_test, loan_ids


def load_behavioural_test(
    loan_ids_file: Path | None = None,
    context_path: Path = CONTEXT_PATH,
):
    """Reproduce the static+behavioural test set."""
    df  = pd.read_parquet(CLEAN_PATH)
    ctx = pd.read_parquet(context_path)
    if "loan_id" not in df.columns:
        df = df.reset_index().rename(columns={"index": "loan_id"})
    if "loan_id" not in ctx.columns:
        ctx = ctx.reset_index().rename(columns={"index": "loan_id"})
    ctx   = ctx[["loan_id"] + BEHAVIOURAL_COLS]
    merged= df.merge(ctx, on="loan_id", how="inner")
    if loan_ids_file is None:
        from sklearn.model_selection import train_test_split
        _, test = train_test_split(
            merged, test_size=0.30, random_state=42,
            stratify=merged["early_default"]
        )
    else:
        ids = load_fixed_test_ids(loan_ids_file)
        test = merged[merged["loan_id"].isin(ids)].copy()
        if set(test["loan_id"].astype("int64")) != ids:
            raise ValueError("One or more selected test IDs are absent from clean/context data")
    loan_ids = test["loan_id"].values
    y_test   = test["early_default"].astype("int8")
    X_test   = test.drop(columns=["early_default", "loan_id"])
    X_test   = preprocess_features(X_test)
    return X_test, y_test, loan_ids


# ── Global SHAP ───────────────────────────────────────────────────────────────

def shap_compatible_booster(clf):
    """Normalize vector-form base_score only while SHAP decodes the booster.

    XGBoost 3 serializes this scalar as a one-element vector string, while
    SHAP's TreeExplainer loader calls float() on it. The shim changes only the
    decoded metadata value, leaving the fitted booster and TreeExplainer method
    untouched.
    """
    from shap.explainers import _tree
    if not getattr(_tree, "_vector_base_score_compat", False):
        original_decode = _tree.decode_ubjson_buffer

        def decode_with_scalar_base_score(buffer):
            model = original_decode(buffer)
            params = model["learner"]["learner_model_param"]
            value = params.get("base_score")
            if isinstance(value, str) and value.startswith("["):
                params["base_score"] = float(json.loads(value)[0])
            return model

        _tree.decode_ubjson_buffer = decode_with_scalar_base_score
        _tree._vector_base_score_compat = True
    return clf.get_booster()


def verify_shap_additivity(pipeline, X_test: pd.DataFrame, n: int = 10) -> float:
    """Check SHAP sum + expected value against raw margins on n rows."""
    X_tr = pipeline.named_steps["prep"].transform(X_test.iloc[:n])
    booster = shap_compatible_booster(pipeline.named_steps["clf"])
    explainer = shap.TreeExplainer(booster)
    values = np.asarray(explainer.shap_values(X_tr))
    expected = np.asarray(explainer.expected_value).reshape(-1)
    summed = values.sum(axis=1) + float(expected[0])
    margins = booster.predict(xgb.DMatrix(X_tr), output_margin=True)
    return float(np.max(np.abs(summed - margins)))

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

    explainer    = shap.TreeExplainer(shap_compatible_booster(clf))
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
    metadata_by_id: dict | None = None,
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
            **(metadata_by_id or {}).get(int(lid), {}),
        })

    return records


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate Phase 6 SHAP explanations")
    parser.add_argument("--loan-ids-file", type=Path, default=None,
                        help="Parquet file with fixed loan_id IDs and train/test membership")
    parser.add_argument("--context-file", type=Path, default=None,
                        help="Context parquet to use for the behavioural model")
    args = parser.parse_args()
    subset = args.loan_ids_file is not None
    static_pkl = ROOT / "src" / "models" / (
        "xgb_baseline_subsample_pipeline.pkl" if subset else STATIC_PKL.name
    )
    behavioural_pkl = ROOT / "src" / "models" / (
        "xgb_behavioural_pipeline_subsample.pkl" if subset else BEHAVIOURAL_PKL.name
    )
    attention_path = ROOT / "reports" / (
        "lstm_attention_samples_subsample.json" if subset else ATTN_PATH.name
    )
    context_path = args.context_file or (
        ROOT / "data" / "processed" / "sequence_context_features_subsample.parquet"
        if subset else CONTEXT_PATH
    )
    shap_static_png = ROOT / "reports" / (
        "shap_summary_static_subsample.png" if subset else SHAP_STATIC_PNG.name
    )
    shap_behav_png = ROOT / "reports" / (
        "shap_summary_behavioural_subsample.png" if subset else SHAP_BEHAV_PNG.name
    )
    shap_local_out = ROOT / "reports" / (
        "shap_local_samples_subsample.json" if subset else SHAP_LOCAL_OUT.name
    )

    print("=" * 64)
    print("Phase 6 — SHAP Explainer")
    print("=" * 64)

    # ── Load pipelines ────────────────────────────────────────────────────────
    for path, name in [(static_pkl, "static"), (behavioural_pkl, "behavioural")]:
        if not path.exists():
            raise FileNotFoundError(
                f"{name} pipeline not found: {path}\n"
                f"Run src/models/{'baseline' if name=='static' else 'xgboost_behavioural'}"
                f"_xgboost.py first"
            )

    print("\nLoading pipelines...")
    with open(static_pkl,      "rb") as f: pipeline_static = pickle.load(f)
    with open(behavioural_pkl, "rb") as f: pipeline_behav  = pickle.load(f)

    # ── Load test sets ────────────────────────────────────────────────────────
    print("Loading test sets...")
    X_test_s, y_s, ids_s = load_static_test(args.loan_ids_file)
    X_test_b, y_b, ids_b = load_behavioural_test(args.loan_ids_file, context_path)
    print(f"  Static test   : {X_test_s.shape}")
    print(f"  Behavioural test: {X_test_b.shape}")
    static_error = verify_shap_additivity(pipeline_static, X_test_s, n=10)
    behavioural_error = verify_shap_additivity(pipeline_behav, X_test_b, n=10)
    print("  SHAP additivity max abs error (10 borrowers): "
          f"static={static_error:.3e}, behavioural={behavioural_error:.3e}")

    # ── Global SHAP plots ─────────────────────────────────────────────────────
    print("\n[1/2] Global SHAP...")
    run_global_shap(pipeline_static, X_test_s,
                    "SHAP — Static-only XGBoost", shap_static_png)
    run_global_shap(pipeline_behav,  X_test_b,
                    "SHAP — Static+Behavioural XGBoost", shap_behav_png)

    # ── Local SHAP for same 50 borrowers as LSTM attention ────────────────────
    print("\n[2/2] Local SHAP (50 borrowers)...")
    if attention_path.exists():
        with open(attention_path) as f:
            attn_samples = json.load(f)
        target_ids = [s["loan_id"] for s in attn_samples]
    else:
        # Fixed, reproducible local-explanation sample when LSTM attention is absent.
        target_ids = [int(v) for v in ids_b[:50]]
        print(f"  No attention file; using first {len(target_ids)} fixed test IDs for local SHAP")
    p_behav = pipeline_behav.predict_proba(X_test_b)[:, 1]
    metadata_by_id = {
        int(lid): {
            "true_label": int(label),
            "predicted_prob": float(prob),
            "predicted_correct": bool((prob >= 0.5) == int(label)),
        }
        for lid, label, prob in zip(ids_b, y_b, p_behav)
    }
    local_records = run_local_shap(
        pipeline_static, pipeline_behav,
        X_test_s, X_test_b,
        ids_s, ids_b,
        target_ids,
        metadata_by_id,
    )

    shap_local_out.parent.mkdir(parents=True, exist_ok=True)
    with open(shap_local_out, "w") as f:
        json.dump(local_records, f, indent=2)
    print(f"  Saved: {shap_local_out.relative_to(ROOT)}")

    print("\n" + "=" * 64)
    print("Phase 6 SHAP complete.")
    print(f"  Global static plot   : {SHAP_STATIC_PNG.relative_to(ROOT)}")
    print(f"  Global behavioural   : {SHAP_BEHAV_PNG.relative_to(ROOT)}")
    print(f"  Local samples (50)   : {SHAP_LOCAL_OUT.relative_to(ROOT)}")
    print("=" * 64)


if __name__ == "__main__":
    main()
