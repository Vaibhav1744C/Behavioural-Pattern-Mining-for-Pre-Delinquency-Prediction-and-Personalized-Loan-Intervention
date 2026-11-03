"""Audit full-run inputs and canonical held-out metrics without training."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.config import DATA_ROOT

# (relative path, approximate expected bytes, creating script)
REQUIRED = [
    ("data/raw/Loan_status_2007-2020Q3.gzip", "~1.7 GB", "external Lending Club Kaggle download"),
    ("data/processed/lending_club_clean.parquet", "~133 MB", "src/data_generation/clean_lending_club.py"),
    ("data/processed/X_test.parquet", "~39 MB", "src/models/baseline_xgboost.py"),
    ("data/processed/y_test.parquet", "~0.1 MB", "src/models/baseline_xgboost.py"),
    ("data/processed/sequence_context_features.parquet", "tens of MB", "src/features/build_features.py"),
    ("data/processed/lstm_sequences.parquet", ">1 GB", "src/features/build_features.py"),
    ("data/synthetic/behavioural_sequences.parquet", "~1.1 GB", "src/data_generation/generate_behavioural_sequences.py"),
    ("data/processed/lstm_sequences_dense.npy", "~1.26 GB", "src/models/prepare_lstm_data.py"),
    ("data/processed/lstm_loan_id_index.parquet", "~10–20 MB", "src/models/prepare_lstm_data.py"),
    ("data/processed/train_loan_ids.parquet", "~10 MB", "src/models/prepare_lstm_data.py"),
    ("src/models/xgb_baseline_pipeline.pkl", "~0.7 MB", "src/models/baseline_xgboost.py"),
    ("src/models/xgb_behavioural_pipeline.pkl", "<1 MB", "src/models/xgboost_behavioural.py"),
    ("src/models/lstm_best.pt", "<1 MB", "src/models/lstm_model.py"),
    ("reports/full_metrics_comparison.json", "~2.4 KB", "src/models/full_metrics.py"),
    ("reports/lstm_attention_samples.json", "~14 KB (about 50 borrowers)", "src/models/lstm_model.py"),
]
EXPECTED = {"static": 0.7345, "behavioural_options": (0.7686, 0.7698), "lstm": 0.6379}
TOLERANCE = 0.0005


def main() -> int:
    missing = []
    print(f"DATA_ROOT: {DATA_ROOT}")
    for rel, approx, creator in REQUIRED:
        # Data artifacts follow BPM_DATA_ROOT; model/report outputs stay in the repo.
        path = (DATA_ROOT / Path(rel).relative_to("data")) if rel.startswith("data/") else (ROOT / rel)
        if path.is_file():
            print(f"FOUND  {path} | {path.stat().st_size:,} bytes | approx {approx}")
        else:
            missing.append((path, approx, creator))
    if missing:
        print("\nMissing full-run files:")
        for path, approx, creator in missing:
            print(f"MISSING {path} | expected size {approx} | created by {creator}")

    metrics_path = ROOT / "reports" / "full_metrics_comparison.json"
    if not metrics_path.is_file():
        print("Cannot report metrics: reports/full_metrics_comparison.json is missing.")
        return 1
    results = json.loads(metrics_path.read_text(encoding="utf-8"))
    by_label = {row["label"].casefold(): row for row in results}
    def find(prefix: str) -> float | None:
        for label, row in by_label.items():
            if label.startswith(prefix):
                return float(row["roc_auc"])
        return None
    static, behavioural, lstm = find("static xgboost"), find("behav xgboost"), find("lstm")
    clean_path = DATA_ROOT / "processed" / "lending_club_clean.parquet"
    rows = None
    if clean_path.is_file():
        import pandas as pd
        from sklearn.model_selection import train_test_split
        labels = pd.read_parquet(clean_path, columns=["early_default"])["early_default"]
        _, test_labels = train_test_split(labels, test_size=.30, random_state=42, stratify=labels)
        rows = len(test_labels)
    print(f"Test row count: {rows if rows is not None else 'unavailable (clean cohort missing)'}")
    print(f"Static ROC-AUC: {static if static is not None else 'unavailable'}")
    print(f"Behavioural ROC-AUC: {behavioural if behavioural is not None else 'unavailable'}")
    print(f"LSTM ROC-AUC: {lstm if lstm is not None else 'unavailable'}")

    errors = bool(missing)
    # full_metrics_comparison.json is the canonical source for metrics; row count
    # here is the fixed Phase 2 test count, independently represented by the report.
    for name, value, targets in (
        ("static", static, (EXPECTED["static"],)),
        ("behavioural", behavioural, EXPECTED["behavioural_options"]),
        ("LSTM", lstm, (EXPECTED["lstm"],)),
    ):
        if value is None or min(abs(value - target) for target in targets) > TOLERANCE:
            print(f"Mismatch: {name} ROC-AUC outside tolerance {TOLERANCE}")
            errors = True
    if rows != 563771:
        print("Mismatch: reproduced test split must contain exactly 563,771 rows")
        errors = True
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
