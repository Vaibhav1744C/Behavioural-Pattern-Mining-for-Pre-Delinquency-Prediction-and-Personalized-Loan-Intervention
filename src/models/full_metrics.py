"""
full_metrics.py
---------------
Compute PR-AUC, ROC-AUC, precision, recall, F1 and confusion matrices
for all three models at two thresholds each:
  - t=0.5  (standard)
  - t=best-F1  (threshold that maximises F1 on the test set)

This gives a metric-complete picture for the review presentation.

Usage:
    python src/models/full_metrics.py

Output:
    reports/full_metrics_comparison.json
    (printed comparison table to stdout)
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "models"))
sys.path.insert(0, str(ROOT / "src" / "features"))

from baseline_xgboost import preprocess_features as preprocess_static
from xgboost_behavioural import preprocess_features as preprocess_behav, BEHAVIOURAL_COLS
from lstm_model import (
    LSTMPredictor, CONTEXT_COLS, MAX_LEN,
    LoanSequenceDataset, split_indices, load_all_data,
)

RANDOM_STATE = 42


# ── Helpers ───────────────────────────────────────────────────────────────────

def find_best_f1_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    prec, rec, thresholds = precision_recall_curve(y_true, y_prob)
    f1s = 2 * prec * rec / (prec + rec + 1e-9)
    best_idx = int(np.argmax(f1s[:-1]))
    return round(float(thresholds[best_idx]), 3)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray,
                    threshold: float, label: str) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return {
        "label":      label,
        "threshold":  threshold,
        "roc_auc":    round(float(roc_auc_score(y_true, y_prob)), 4),
        "pr_auc":     round(float(average_precision_score(y_true, y_prob)), 4),
        "precision":  round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall":     round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1":         round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "confusion_matrix": cm.tolist(),
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("Full Metric Suite — All 3 Models")
    print("=" * 70)

    # ── Load clean data + reproduce split ─────────────────────────────────────
    df = pd.read_parquet(ROOT / "data/processed/lending_club_clean.parquet")
    if "loan_id" not in df.columns:
        df = df.reset_index().rename(columns={"index": "loan_id"})

    ctx = pd.read_parquet(ROOT / "data/processed/sequence_context_features.parquet")
    if "loan_id" not in ctx.columns:
        ctx = ctx.reset_index().rename(columns={"index": "loan_id"})

    _, test_df = train_test_split(
        df, test_size=0.30, random_state=RANDOM_STATE,
        stratify=df["early_default"]
    )
    print(f"Test set: {len(test_df):,} borrowers, "
          f"{test_df['early_default'].mean():.2%} default rate\n")

    results = []

    # ── Model 1: Static XGBoost ───────────────────────────────────────────────
    print("[1/3] Static XGBoost...")
    with open(ROOT / "src/models/xgb_baseline_pipeline.pkl", "rb") as f:
        pipe1 = pickle.load(f)
    X1 = preprocess_static(test_df.drop(columns=["early_default", "loan_id"]))
    y1 = test_df["early_default"].astype(int).values
    prob1 = pipe1.predict_proba(X1)[:, 1]
    t1 = find_best_f1_threshold(y1, prob1)
    r1a = compute_metrics(y1, prob1, 0.5, "Static XGBoost (t=0.5)")
    r1b = compute_metrics(y1, prob1, t1,  f"Static XGBoost (t={t1} best-F1)")
    results += [r1a, r1b]
    print(f"  ROC-AUC={r1a['roc_auc']}  PR-AUC={r1a['pr_auc']}  best-F1 threshold={t1}")

    # ── Model 2: Static + Behavioural XGBoost ────────────────────────────────
    print("[2/3] Static+Behavioural XGBoost...")
    with open(ROOT / "src/models/xgb_behavioural_pipeline.pkl", "rb") as f:
        pipe2 = pickle.load(f)
    merged = test_df.merge(ctx[["loan_id"] + BEHAVIOURAL_COLS], on="loan_id", how="inner")
    X2 = preprocess_behav(merged.drop(columns=["early_default", "loan_id"]))
    y2 = merged["early_default"].astype(int).values
    prob2 = pipe2.predict_proba(X2)[:, 1]
    t2 = find_best_f1_threshold(y2, prob2)
    r2a = compute_metrics(y2, prob2, 0.5, "Behav XGBoost (t=0.5)")
    r2b = compute_metrics(y2, prob2, t2,  f"Behav XGBoost (t={t2} best-F1)")
    results += [r2a, r2b]
    print(f"  ROC-AUC={r2a['roc_auc']}  PR-AUC={r2a['pr_auc']}  best-F1 threshold={t2}")

    # ── Model 3: LSTM ─────────────────────────────────────────────────────────
    print("[3/3] LSTM...")
    dense_path = ROOT / "data/processed/lstm_sequences_dense.npy"
    seq_arr, ctx_arr, labels, seq_lens, loan_ids, all_ids = load_all_data(dense_path)
    tv_mask, test_mask = split_indices(all_ids, loan_ids)
    test_idx = np.where(test_mask)[0]

    test_ds = LoanSequenceDataset(
        seq_arr[test_idx], ctx_arr[test_idx],
        labels[test_idx], seq_lens[test_idx], loan_ids[test_idx]
    )
    test_loader = DataLoader(test_ds, batch_size=2048, shuffle=False, num_workers=0)

    device = torch.device("cpu")
    model = LSTMPredictor().to(device)
    model.load_state_dict(
        torch.load(ROOT / "src/models/lstm_best.pt", map_location=device)
    )
    model.eval()

    all_probs, all_labels = [], []
    with torch.no_grad():
        for seq, ctx_t, lbl, lens, _ in test_loader:
            logit, _ = model(seq.to(device), ctx_t.to(device), lens.to(device))
            all_probs.extend(torch.sigmoid(logit).cpu().numpy().tolist())
            all_labels.extend(lbl.numpy().tolist())

    prob3 = np.array(all_probs)
    y3    = np.array(all_labels).astype(int)
    t3    = find_best_f1_threshold(y3, prob3)
    r3a   = compute_metrics(y3, prob3, 0.5, "LSTM (t=0.5)")
    r3b   = compute_metrics(y3, prob3, t3,  f"LSTM (t={t3} best-F1)")
    results += [r3a, r3b]
    print(f"  ROC-AUC={r3a['roc_auc']}  PR-AUC={r3a['pr_auc']}  best-F1 threshold={t3}")

    # ── Print comparison table ────────────────────────────────────────────────
    print()
    print("=" * 80)
    header = f"{'Model':<44} {'ROC-AUC':>7} {'PR-AUC':>7} {'Prec':>7} {'Recall':>7} {'F1':>7}"
    print(header)
    print("-" * 80)
    for m in results:
        print(f"{m['label']:<44} {m['roc_auc']:>7} {m['pr_auc']:>7} "
              f"{m['precision']:>7} {m['recall']:>7} {m['f1']:>7}")
    print("=" * 80)

    # ── Print confusion matrices ──────────────────────────────────────────────
    print("\nConfusion matrices (TP/FP/FN/TN) at best-F1 threshold:")
    for m in [r1b, r2b, r3b]:
        print(f"\n  {m['label']}")
        print(f"    TP (caught defaulters)    : {m['tp']:>7,}")
        print(f"    FP (false alarms)         : {m['fp']:>7,}")
        print(f"    FN (missed defaulters)    : {m['fn']:>7,}")
        print(f"    TN (correct non-default)  : {m['tn']:>7,}")

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = ROOT / "reports/full_metrics_comparison.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
