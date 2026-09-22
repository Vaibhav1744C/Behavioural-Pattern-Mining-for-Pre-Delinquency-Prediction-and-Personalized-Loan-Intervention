"""
fuzzy_surrogate.py
------------------
Phase 6: Fuzzy 2-tuple linguistic surrogate tree.

Follows the base paper's 7-step algorithm (Section 4, Monje et al. 2025):

  Step 1: Define the input feature set X (static + 7 behavioural context)
  Step 2: Run the black-box model (LSTM or XGBoost) to get predicted probs p
  Step 3: Train a shallow DecisionTreeRegressor on X -> p
  Step 4: Extract IF/THEN rules from the tree
  Step 5: Report R² fidelity (comparable to paper's 0.806/0.808)
  Step 6: Fuzzify continuous splits using 2-tuple linguistic model
          (Very Low / Low / Medium / High / Very High — triangular MFs,
           same as paper's Figure 3)
  Step 7: Produce final linguistic rule set (same format as paper's Table 6)

Runs for both the LSTM and the static+behavioural XGBoost — gives two rule
sets to compare. Shows whether behavioural features change which linguistic
rules dominate.

Usage:
    python src/explainability/fuzzy_surrogate.py

Outputs:
    reports/surrogate_rules_lstm.json
    reports/surrogate_rules_xgb_behavioural.json
    reports/surrogate_fidelity.json

Requirements:
    pip install scikit-learn pandas pyarrow numpy torch
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, export_text

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT            = Path(__file__).resolve().parents[2]
CLEAN_PATH      = ROOT / "data" / "processed" / "lending_club_clean.parquet"
CONTEXT_PATH    = ROOT / "data" / "processed" / "sequence_context_features.parquet"
DENSE_PATH      = ROOT / "data" / "processed" / "lstm_sequences_dense.npy"
INDEX_PATH      = ROOT / "data" / "processed" / "lstm_loan_id_index.parquet"
TRAIN_IDS_PATH  = ROOT / "data" / "processed" / "train_loan_ids.parquet"
BEHAVIOURAL_PKL = ROOT / "src"  / "models"    / "xgb_behavioural_pipeline.pkl"
LSTM_CKPT       = ROOT / "src"  / "models"    / "lstm_best.pt"
RULES_LSTM_OUT  = ROOT / "reports"            / "surrogate_rules_lstm.json"
RULES_XGB_OUT   = ROOT / "reports"            / "surrogate_rules_xgb_behavioural.json"
FIDELITY_OUT    = ROOT / "reports"            / "surrogate_fidelity.json"

BEHAVIOURAL_COLS = [
    "salary_stability_idx", "salary_delay_trend", "savings_slope",
    "savings_volatility", "cashflow_compression", "emi_stress_count", "seq_len",
]

CONTEXT_COLS_MODEL = [   # seq_len excluded from LSTM model input
    "salary_stability_idx", "salary_delay_trend", "savings_slope",
    "savings_volatility", "cashflow_compression", "emi_stress_count",
]

MAX_DEPTH    = 4     # paper uses ~9 leaves which corresponds to max_depth 3-4
RANDOM_STATE = 42


# ── 2-tuple fuzzy linguistic model ───────────────────────────────────────────
# Based on paper's Figure 3: five symmetric triangular membership functions
# on [0, 1]. Each label covers a range; Δ is the symbolic translation.
#
# Labels: VL=Very Low, L=Low, M=Medium, H=High, VH=Very High
# Centers: 0.0, 0.25, 0.5, 0.75, 1.0
# Width  : 0.25 each side (triangular)

FUZZY_LABELS  = ["Very Low", "Low", "Medium", "High", "Very High"]
FUZZY_CENTERS = [0.0, 0.25, 0.5, 0.75, 1.0]


def triangular_mf(x: float, center: float, width: float = 0.25) -> float:
    """Triangular membership function."""
    return max(0.0, 1.0 - abs(x - center) / width)


def fuzzify_value(val: float) -> tuple[str, float]:
    """
    Convert a normalised value in [0,1] to a 2-tuple (label, delta).
    delta is the signed displacement from the nearest label center,
    normalised by the label width — directly replicates the paper's Eq. (1).
    """
    val   = float(np.clip(val, 0.0, 1.0))
    memberships = [triangular_mf(val, c) for c in FUZZY_CENTERS]
    best_idx    = int(np.argmax(memberships))
    label       = FUZZY_LABELS[best_idx]
    delta       = round((val - FUZZY_CENTERS[best_idx]) / 0.25, 4)
    return label, delta


def normalise_series(s: pd.Series) -> pd.Series:
    """Min-max normalise to [0,1] for fuzzification."""
    lo, hi = s.min(), s.max()
    if hi == lo:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - lo) / (hi - lo)


# ── Rule extraction from decision tree ───────────────────────────────────────

def extract_rules(tree: DecisionTreeRegressor, feature_names: list) -> list:
    """
    Walk the fitted tree and return a list of rule dicts:
      { "conditions": [...], "leaf_value": float, "n_samples": int }
    Each condition: { "feature": str, "op": "<=" or ">", "threshold": float }
    """
    t        = tree.tree_
    n_nodes  = t.node_count
    children_left  = t.children_left
    children_right = t.children_right
    feature        = t.feature
    threshold      = t.threshold
    value          = t.value
    n_node_samples = t.n_node_samples

    rules = []

    def recurse(node_id, conditions):
        if children_left[node_id] == children_right[node_id]:
            # Leaf node
            rules.append({
                "conditions": list(conditions),
                "leaf_value": round(float(value[node_id][0][0]), 4),
                "n_samples":  int(n_node_samples[node_id]),
            })
            return
        feat = feature_names[feature[node_id]]
        thr  = round(float(threshold[node_id]), 4)
        recurse(children_left[node_id],
                conditions + [{"feature": feat, "op": "<=", "threshold": thr}])
        recurse(children_right[node_id],
                conditions + [{"feature": feat, "op": ">",  "threshold": thr}])

    recurse(0, [])
    return sorted(rules, key=lambda r: r["leaf_value"], reverse=True)


def fuzzify_rules(rules: list, col_ranges: dict) -> list:
    """
    Convert numeric thresholds in rules to fuzzy 2-tuple linguistic labels.
    col_ranges: {feature_name: (min_val, max_val)} for normalisation.
    """
    fuzzy_rules = []
    for rule in rules:
        fuzzy_conds = []
        for cond in rule["conditions"]:
            feat = cond["feature"]
            thr  = cond["threshold"]
            if feat in col_ranges:
                lo, hi = col_ranges[feat]
                norm   = (thr - lo) / (hi - lo + 1e-9)
                label, delta = fuzzify_value(norm)
                fuzzy_conds.append({
                    "feature":   feat,
                    "op":        cond["op"],
                    "threshold": thr,
                    "fuzzy":     f"({label}, {delta:+.4f})",
                })
            else:
                fuzzy_conds.append(cond)

        risk_val = rule["leaf_value"]
        risk_label, risk_delta = fuzzify_value(risk_val)
        fuzzy_rules.append({
            "conditions":  fuzzy_conds,
            "risk_score":  risk_val,
            "risk_linguistic": f"({risk_label}, {risk_delta:+.4f})",
            "n_samples":   rule["n_samples"],
        })
    return fuzzy_rules


# ── Build surrogate X (static + behavioural, no raw sequences) ────────────────

def build_surrogate_X(loan_ids_subset: list) -> pd.DataFrame:
    """
    Build X for surrogate: static features (numeric only, pre-encoded)
    + 7 behavioural context features.
    Returns DataFrame aligned to loan_ids_subset order.
    """
    import sys
    sys.path.insert(0, str(ROOT / "src" / "models"))
    from xgboost_behavioural import preprocess_features

    df  = pd.read_parquet(CLEAN_PATH)
    ctx = pd.read_parquet(CONTEXT_PATH)

    if "loan_id" not in df.columns:
        df = df.reset_index().rename(columns={"index": "loan_id"})
    if "loan_id" not in ctx.columns:
        ctx = ctx.reset_index().rename(columns={"index": "loan_id"})

    ctx   = ctx[["loan_id"] + BEHAVIOURAL_COLS]
    merged= df.merge(ctx, on="loan_id", how="inner")

    # Filter to requested ids
    id_set = set(loan_ids_subset)
    merged = merged[merged["loan_id"].isin(id_set)].copy()
    merged = merged.set_index("loan_id").loc[
        [lid for lid in loan_ids_subset if lid in merged["loan_id"].values]
    ].reset_index()

    X = merged.drop(columns=["early_default", "loan_id"])
    X = preprocess_features(X)

    # Keep only numeric columns (surrogate tree works on numeric inputs)
    X = X.select_dtypes(include=[np.number])
    return X, merged["loan_id"].values


# ── XGBoost probability inference ─────────────────────────────────────────────

def get_xgb_probs(pipeline, X: pd.DataFrame) -> np.ndarray:
    return pipeline.predict_proba(X)[:, 1]


# ── LSTM probability inference ────────────────────────────────────────────────

def get_lstm_probs(loan_ids: np.ndarray, X_ctx: pd.DataFrame) -> np.ndarray | None:
    """
    Run LSTM inference on the given loan_ids.
    Returns probability array or None if model not available.
    """
    if not LSTM_CKPT.exists():
        print("  WARNING: lstm_best.pt not found — skipping LSTM surrogate")
        return None
    if not DENSE_PATH.exists():
        print("  WARNING: lstm_sequences_dense.npy not found — skipping LSTM surrogate")
        return None

    try:
        import torch
        import sys
        sys.path.insert(0, str(ROOT / "src" / "models"))
        from lstm_model import LSTMPredictor, CONTEXT_COLS, MAX_LEN

        device = torch.device("cpu")
        model  = LSTMPredictor().to(device)
        model.load_state_dict(torch.load(LSTM_CKPT, map_location=device))
        model.eval()

        # Load dense array + index
        seq_arr  = np.load(DENSE_PATH)
        idx_df   = pd.read_parquet(INDEX_PATH)
        id_to_row= dict(zip(idx_df["loan_id"], idx_df["row_index"]))

        # Load seq_lens
        ctx_full = pd.read_parquet(CONTEXT_PATH)
        if "loan_id" not in ctx_full.columns:
            ctx_full = ctx_full.reset_index().rename(columns={"index": "loan_id"})
        lens_map = dict(zip(ctx_full["loan_id"], ctx_full["seq_len"]))

        # Normalise context
        ctx_feat = X_ctx[CONTEXT_COLS].copy()
        for col in CONTEXT_COLS:
            ctx_feat[col] = pd.to_numeric(ctx_feat[col], errors="coerce").fillna(0.0)
        ctx_mean = ctx_feat.mean()
        ctx_std  = ctx_feat.std().replace(0, 1)
        ctx_feat = ((ctx_feat - ctx_mean) / ctx_std).fillna(0.0)

        probs = []
        BATCH = 512
        ids_list = list(loan_ids)
        for start in range(0, len(ids_list), BATCH):
            batch_ids = ids_list[start:start+BATCH]
            rows   = [id_to_row.get(lid, 0) for lid in batch_ids]
            seq_t  = torch.tensor(seq_arr[rows], dtype=torch.float32)
            ctx_t  = torch.tensor(
                ctx_feat.loc[ctx_feat.index[:len(batch_ids)]].values,
                dtype=torch.float32
            )
            lens_t = torch.tensor(
                [max(1, lens_map.get(lid, MAX_LEN)) for lid in batch_ids],
                dtype=torch.int64
            )
            with torch.no_grad():
                logit, _ = model(seq_t, ctx_t, lens_t)
                probs.extend(torch.sigmoid(logit).cpu().numpy().tolist())
        return np.array(probs)

    except Exception as e:
        print(f"  WARNING: LSTM inference failed ({e}) — skipping LSTM surrogate")
        return None


# ── Train surrogate tree ──────────────────────────────────────────────────────

def train_surrogate(X: pd.DataFrame, p: np.ndarray, label: str) -> dict:
    """
    Train DecisionTreeRegressor on X -> p.
    Returns dict with tree, rules, fidelity R², and fuzzy rule set.
    """
    print(f"  Training surrogate tree for: {label}")

    # Hold-out slice for fidelity measurement (20%)
    idx_all   = np.arange(len(X))
    rng       = np.random.default_rng(RANDOM_STATE)
    idx_test  = rng.choice(idx_all, size=int(0.2 * len(idx_all)), replace=False)
    idx_train = np.setdiff1d(idx_all, idx_test)

    X_np = X.values.astype(np.float32)
    tree = DecisionTreeRegressor(max_depth=MAX_DEPTH, random_state=RANDOM_STATE)
    tree.fit(X_np[idx_train], p[idx_train])

    p_pred  = tree.predict(X_np[idx_test])
    r2      = r2_score(p[idx_test], p_pred)
    print(f"    R² fidelity: {r2:.4f}  (paper reports 0.806/0.808)")

    feature_names = list(X.columns)
    rules         = extract_rules(tree, feature_names)

    # Compute per-column ranges for fuzzification
    col_ranges = {col: (float(X[col].min()), float(X[col].max()))
                  for col in X.columns}
    fuzzy_rules = fuzzify_rules(rules, col_ranges)

    # Print top 5 rules
    print(f"    Top 5 rules by risk score:")
    for r in fuzzy_rules[:5]:
        cond_str = " AND ".join(
            f"{c['feature']} {c['op']} {c['threshold']} [{c.get('fuzzy','')}]"
            for c in r["conditions"]
        )
        print(f"      IF {cond_str}")
        print(f"      THEN risk = {r['risk_linguistic']}  (n={r['n_samples']})")

    return {
        "model":       label,
        "r2_fidelity": round(r2, 4),
        "n_rules":     len(fuzzy_rules),
        "rules":       fuzzy_rules,
        "tree_text":   export_text(tree, feature_names=feature_names),
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 64)
    print("Phase 6 — Fuzzy 2-tuple Surrogate Tree")
    print("=" * 64)

    # ── Get training loan_ids ─────────────────────────────────────────────────
    if TRAIN_IDS_PATH.exists():
        train_ids = pd.read_parquet(TRAIN_IDS_PATH)["loan_id"].tolist()
    else:
        print("  WARNING: train_loan_ids.parquet not found — using full dataset")
        df = pd.read_parquet(CLEAN_PATH)
        if "loan_id" not in df.columns:
            df = df.reset_index().rename(columns={"index": "loan_id"})
        train_ids = df["loan_id"].tolist()

    print(f"  Surrogate training on {len(train_ids):,} borrowers")

    # ── Build surrogate X ─────────────────────────────────────────────────────
    print("\nBuilding surrogate feature matrix...")
    X_sur, aligned_ids = build_surrogate_X(train_ids)
    print(f"  Surrogate X shape: {X_sur.shape}")

    fidelity_results = {}

    # ── XGBoost behavioural surrogate ─────────────────────────────────────────
    print("\n[1/2] XGBoost static+behavioural surrogate...")
    if BEHAVIOURAL_PKL.exists():
        with open(BEHAVIOURAL_PKL, "rb") as f:
            pipeline_b = pickle.load(f)

        import sys
        sys.path.insert(0, str(ROOT / "src" / "models"))
        from xgboost_behavioural import preprocess_features as pf_b
        X_for_xgb = pd.read_parquet(CLEAN_PATH)
        if "loan_id" not in X_for_xgb.columns:
            X_for_xgb = X_for_xgb.reset_index().rename(columns={"index": "loan_id"})
        ctx_df = pd.read_parquet(CONTEXT_PATH)
        if "loan_id" not in ctx_df.columns:
            ctx_df = ctx_df.reset_index().rename(columns={"index": "loan_id"})
        merged = X_for_xgb.merge(ctx_df[["loan_id"] + BEHAVIOURAL_COLS],
                                  on="loan_id", how="inner")
        id_set = set(aligned_ids.tolist())
        merged = merged[merged["loan_id"].isin(id_set)]
        X_xgb  = merged.drop(columns=["early_default", "loan_id"])
        X_xgb  = pf_b(X_xgb)

        p_xgb  = get_xgb_probs(pipeline_b, X_xgb)
        result_xgb = train_surrogate(X_sur[:len(p_xgb)], p_xgb,
                                     "XGBoost static+behavioural")
        fidelity_results["xgb_behavioural"] = result_xgb["r2_fidelity"]

        RULES_XGB_OUT.parent.mkdir(parents=True, exist_ok=True)
        with open(RULES_XGB_OUT, "w") as f:
            json.dump(result_xgb, f, indent=2)
        print(f"  Saved: {RULES_XGB_OUT.relative_to(ROOT)}")
    else:
        print(f"  SKIP: {BEHAVIOURAL_PKL.name} not found")
        print("  Run src/models/xgboost_behavioural.py first")

    # ── LSTM surrogate ────────────────────────────────────────────────────────
    print("\n[2/2] LSTM surrogate...")
    ctx_aligned = pd.read_parquet(CONTEXT_PATH)
    if "loan_id" not in ctx_aligned.columns:
        ctx_aligned = ctx_aligned.reset_index().rename(columns={"index": "loan_id"})
    ctx_aligned = ctx_aligned[ctx_aligned["loan_id"].isin(set(aligned_ids.tolist()))]

    p_lstm = get_lstm_probs(aligned_ids, ctx_aligned)
    if p_lstm is not None:
        result_lstm = train_surrogate(X_sur[:len(p_lstm)], p_lstm, "LSTM")
        fidelity_results["lstm"] = result_lstm["r2_fidelity"]

        with open(RULES_LSTM_OUT, "w") as f:
            json.dump(result_lstm, f, indent=2)
        print(f"  Saved: {RULES_LSTM_OUT.relative_to(ROOT)}")

    # ── Save fidelity summary ─────────────────────────────────────────────────
    fidelity_results["paper_r2"] = 0.807   # midpoint of 0.806/0.808
    with open(FIDELITY_OUT, "w") as f:
        json.dump(fidelity_results, f, indent=2)
    print(f"\nFidelity summary -> {FIDELITY_OUT.relative_to(ROOT)}")

    print("\n" + "=" * 64)
    print("Phase 6 Fuzzy Surrogate complete.")
    print("=" * 64)


if __name__ == "__main__":
    main()
