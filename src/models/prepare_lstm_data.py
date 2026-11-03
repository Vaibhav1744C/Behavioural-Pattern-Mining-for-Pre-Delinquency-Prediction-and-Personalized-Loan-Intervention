"""
prepare_lstm_data.py
--------------------
Phase 5 data preparation (run once before lstm_model.py).

What it does:
  1. Pivots lstm_sequences.parquet (long format, 45M rows) into a dense
     NumPy array of shape (n_borrowers, 24, 7), saved as
     data/processed/lstm_sequences_dense.npy
     Memory: 1.87M x 24 x 7 x 4 bytes ~ 1.26 GB — fits comfortably in RAM.

  2. Saves a loan_id -> row_index mapping as
     data/processed/lstm_loan_id_index.parquet
     so lstm_model.py can align labels + context features with array rows.

  3. Extracts train loan_ids = all_loan_ids - test_loan_ids (set difference
     against X_test.parquet) and saves as
     data/processed/train_loan_ids.parquet
     Makes the 70/30 split explicit and reusable — no risk of drift between
     Phase 2 and Phase 5 splits.

  4. Optionally builds an ablation-run dense array from
     behavioural_sequences_alpha0 sequences (same pivot logic, separate file).

Usage:
    python src/models/prepare_lstm_data.py              # full run
    python src/models/prepare_lstm_data.py --sample 5000  # quick test
    python src/models/prepare_lstm_data.py --ablation   # also build alpha=0 array

Outputs:
    data/processed/lstm_sequences_dense.npy         shape (N, 24, 7) float32
    data/processed/lstm_loan_id_index.parquet        loan_id -> row_index
    data/processed/train_loan_ids.parquet            train set loan_ids
    data/processed/lstm_sequences_dense_alpha0.npy   (if --ablation)

Requirements:
    pip install pandas pyarrow numpy
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).resolve().parents[2]
LSTM_SEQ_PATH = ROOT / "data" / "processed" / "lstm_sequences.parquet"
ABL_SEQ_PATH  = ROOT / "data" / "synthetic" / "behavioural_sequences_alpha0.parquet"
CLEAN_PATH    = ROOT / "data" / "processed" / "lending_club_clean.parquet"
X_TEST_PATH   = ROOT / "data" / "processed" / "X_test.parquet"
FEATURES_PATH = ROOT / "data" / "processed" / "src" / "features" / "build_features.py"

DENSE_OUT     = ROOT / "data" / "processed" / "lstm_sequences_dense.npy"
DENSE_ABL_OUT = ROOT / "data" / "processed" / "lstm_sequences_dense_alpha0.npy"
INDEX_OUT     = ROOT / "data" / "processed" / "lstm_loan_id_index.parquet"
TRAIN_IDS_OUT = ROOT / "data" / "processed" / "train_loan_ids.parquet"

MAX_LEN    = 24
N_FEATURES = 7   # salary_delay_days_z, discretionary_spend_z, account_balance_z,
                 # savings_balance_z, delta_account_balance, delta_savings_balance,
                 # emi_status_enc

# Column order must match what lstm_model.py expects
FEATURE_COLS = [
    "salary_delay_days_z",
    "discretionary_spend_z",
    "account_balance_z",
    "savings_balance_z",
    "delta_account_balance",
    "delta_savings_balance",
    "emi_status_enc",
]


# ── Core: pivot long parquet → dense array ────────────────────────────────────

def build_dense_array(
    seq_parquet_path: Path,
    loan_id_order: list,
    sample_ids: set | None = None,
) -> np.ndarray:
    """
    Read lstm_sequences.parquet (long format) and pivot to
    shape (len(loan_id_order), MAX_LEN, N_FEATURES) float32.

    loan_id_order: the canonical ordering of borrowers (defines row indices).
    sample_ids: if set, only process those loan_ids (for --sample mode).
    """
    id_to_idx = {lid: i for i, lid in enumerate(loan_id_order)}
    n         = len(loan_id_order)
    arr       = np.zeros((n, MAX_LEN, N_FEATURES), dtype=np.float32)

    pf         = pq.ParquetFile(seq_parquet_path)
    total_rows = pf.metadata.num_rows
    processed  = 0

    print(f"  Parquet: {total_rows:,} rows → dense ({n:,}, {MAX_LEN}, {N_FEATURES})")

    cols_needed = ["loan_id", "month"] + FEATURE_COLS
    for batch in pf.iter_batches(batch_size=2_000_000, columns=cols_needed):
        df = batch.to_pandas()

        if sample_ids is not None:
            df = df[df["loan_id"].isin(sample_ids)]
        if df.empty:
            continue

        # Coerce feature columns to float32
        for col in FEATURE_COLS:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype("float32")

        df["month"] = df["month"].astype(int)

        # Filter to valid months (0..MAX_LEN-1) — padded rows are already 0
        df = df[df["month"] < MAX_LEN]

        # Vectorised write into dense array
        row_idx  = df["loan_id"].map(id_to_idx).values
        month    = df["month"].values
        vals     = df[FEATURE_COLS].values.astype(np.float32)

        # Only fill rows we know about
        valid    = ~pd.isna(df["loan_id"].map(id_to_idx)).values
        arr[row_idx[valid], month[valid], :] = vals[valid]

        processed += len(df)
        print(f"    processed {processed:>12,} rows so far", end="\r")

    print(f"\n  Done. Array shape: {arr.shape}  dtype: {arr.dtype}")
    return arr


# ── Train/test split extraction ───────────────────────────────────────────────

def extract_train_ids(all_ids: list, loan_ids_file: Path | None = None) -> pd.DataFrame:
    """
    Train loan_ids = all_loan_ids - X_test loan_ids.
    Reuses the exact same test set as Phase 2 for an apples-to-apples
    AUC comparison.
    """
    if loan_ids_file is not None:
        split_df = pd.read_parquet(loan_ids_file)
        if not {"loan_id", "split"}.issubset(split_df.columns):
            raise KeyError("--loan-ids-file must contain loan_id and split columns")
        if split_df["loan_id"].duplicated().any():
            raise ValueError("--loan-ids-file contains duplicate loan_id values")
        if set(split_df["split"].unique()) - {"train", "test"}:
            raise ValueError("split values in --loan-ids-file must be 'train' or 'test'")
        selected_ids = set(int(lid) for lid in all_ids)
        file_ids = set(split_df["loan_id"].astype("int64"))
        if selected_ids != file_ids:
            raise ValueError("LSTM feature IDs do not exactly match --loan-ids-file")
        train_df = split_df.loc[split_df["split"] == "train", ["loan_id"]].copy()
        print(f"  Fixed split: {len(train_df):,} train / {(split_df['split'] == 'test').sum():,} test")
        return train_df.reset_index(drop=True)

    if not X_TEST_PATH.exists():
        raise FileNotFoundError(
            f"X_test.parquet not found at {X_TEST_PATH}.\n"
            "Run src/models/baseline_xgboost.py (Phase 2) first."
        )
    test_df   = pd.read_parquet(X_TEST_PATH, columns=["loan_id"]) \
                if "loan_id" in pq.read_schema(X_TEST_PATH).names \
                else pd.read_parquet(X_TEST_PATH).reset_index()[["index"]].rename(columns={"index": "loan_id"})

    # X_test may not have loan_id if it was indexed by position — handle both
    try:
        test_ids = set(pd.read_parquet(X_TEST_PATH, columns=["loan_id"])["loan_id"].tolist())
    except Exception:
        # X_test was saved without loan_id column — reproduce the same 70/30
        # stratified split from the clean data using the same seed as Phase 2
        print("  INFO: loan_id not in X_test.parquet — reproducing split from clean data")
        from sklearn.model_selection import train_test_split as _tts
        clean   = pd.read_parquet(CLEAN_PATH, columns=["early_default"])
        clean   = clean.reset_index().rename(columns={"index": "loan_id"})
        _, test = _tts(
            clean, test_size=0.30, random_state=42, stratify=clean["early_default"]
        )
        test_ids = set(test["loan_id"].tolist())
        # Save train_loan_ids.parquet now so future runs skip the fallback
        all_set   = set(all_ids)
        train_ids_save = sorted(all_set - test_ids)
        pd.DataFrame({"loan_id": train_ids_save}).to_parquet(TRAIN_IDS_OUT, index=False)
        print(f"  Saved train_loan_ids.parquet for future runs")

    all_ids_set  = set(all_ids)
    train_ids    = sorted(all_ids_set - test_ids)
    print(f"  Total borrowers : {len(all_ids):,}")
    print(f"  Test  loan_ids  : {len(test_ids):,}")
    print(f"  Train loan_ids  : {len(train_ids):,}")
    return pd.DataFrame({"loan_id": train_ids})


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Phase 5 data prep")
    p.add_argument("--sample",   type=int, default=None,
                   help="Only process N borrowers (quick test)")
    p.add_argument("--loan-ids-file", type=Path, default=None,
                   help="Parquet file with the fixed loan_id set and split labels")
    p.add_argument("--ablation", action="store_true",
                   help="Also build dense array for alpha=0 ablation sequences")
    return p.parse_args()


def main():
    args = parse_args()
    if args.sample and args.loan_ids_file:
        raise ValueError("Use either --sample or --loan-ids-file, not both")
    subsample_tag = args.loan_ids_file is not None
    seq_path = ROOT / "data" / "processed" / (
        "lstm_sequences_subsample.parquet" if subsample_tag else LSTM_SEQ_PATH.name
    )
    dense_out = ROOT / "data" / "processed" / (
        "lstm_sequences_dense_subsample.npy" if subsample_tag else DENSE_OUT.name
    )
    index_out = ROOT / "data" / "processed" / (
        "lstm_loan_id_index_subsample.parquet" if subsample_tag else INDEX_OUT.name
    )
    train_ids_out = ROOT / "data" / "processed" / (
        "train_loan_ids_subsample.parquet" if subsample_tag else TRAIN_IDS_OUT.name
    )

    print("=" * 64)
    print("Phase 5 Data Prep — build dense arrays + train/test split")
    print("=" * 64)

    # ── 1. Get canonical loan_id order from lstm_sequences.parquet ────────────
    print("\n[1/4] Reading loan_id order from lstm_sequences.parquet...")
    all_ids = (
        pq.read_table(seq_path, columns=["loan_id"])
        .to_pandas()["loan_id"]
        .unique()
        .tolist()
    )
    print(f"  Total borrowers: {len(all_ids):,}")

    sample_ids = None
    if args.loan_ids_file:
        requested = pd.read_parquet(args.loan_ids_file)
        if not {"loan_id", "split"}.issubset(requested.columns):
            raise KeyError("--loan-ids-file must contain loan_id and split columns")
        requested_ids = requested["loan_id"].astype("int64").tolist()
        if len(requested_ids) != len(set(requested_ids)):
            raise ValueError("--loan-ids-file contains duplicate loan_id values")
        if set(requested["split"].unique()) - {"train", "test"}:
            raise ValueError("split values in --loan-ids-file must be 'train' or 'test'")
        sequence_ids = set(int(lid) for lid in all_ids)
        missing_ids = set(requested_ids) - sequence_ids
        if missing_ids:
            raise ValueError(f"{len(missing_ids):,} requested IDs are absent from {seq_path}")
        all_ids = requested_ids
        sample_ids = set(requested_ids)
        print(f"  Fixed borrower IDs: {len(all_ids):,} from {args.loan_ids_file}")
    elif args.sample:
        all_ids    = all_ids[:args.sample]
        sample_ids = set(all_ids)
        print(f"  Subsetting to {len(all_ids):,} borrowers (--sample)")

    # ── 2. Build dense array ──────────────────────────────────────────────────
    print("\n[2/4] Building dense array from lstm_sequences.parquet...")
    arr = build_dense_array(seq_path, all_ids, sample_ids)

    dense_out.parent.mkdir(parents=True, exist_ok=True)
    np.save(dense_out, arr)
    size_mb = dense_out.stat().st_size / (1024 ** 2)
    print(f"  Saved: {dense_out.relative_to(ROOT)}  ({size_mb:.1f} MB)")
    del arr

    # ── 3. Save loan_id -> row_index mapping ──────────────────────────────────
    print("\n[3/4] Saving loan_id index mapping...")
    idx_df = pd.DataFrame({"loan_id": all_ids, "row_index": range(len(all_ids))})
    idx_df.to_parquet(index_out, index=False)
    print(f"  Saved: {index_out.relative_to(ROOT)}  ({len(idx_df):,} rows)")

    # ── 4. Extract train/test split ───────────────────────────────────────────
    print("\n[4/4] Extracting train loan_ids (all - test)...")
    train_df = extract_train_ids(all_ids, args.loan_ids_file)
    train_df.to_parquet(train_ids_out, index=False)
    print(f"  Saved: {train_ids_out.relative_to(ROOT)}  ({len(train_df):,} rows)")

    # ── Optional: ablation dense array ───────────────────────────────────────
    if args.ablation:
        if not ABL_SEQ_PATH.exists():
            print(f"\n  WARNING: ablation parquet not found at {ABL_SEQ_PATH} — skipping")
        else:
            print("\n[+] Building ablation dense array (alpha=0)...")
            # Need to build LSTM features from ablation sequences first
            # For now, use the same lstm_sequences.parquet structure but
            # note that the ablation features need to be built separately
            # via: python src/features/build_features.py --seq-parquet data/synthetic/behavioural_sequences_alpha0.parquet
            print("  NOTE: ablation LSTM features must be built first:")
            print("  python src/features/build_features.py \\")
            print("    --seq-parquet data/synthetic/behavioural_sequences_alpha0.parquet")
            abl_lstm_path = ROOT / "data" / "processed" / "lstm_sequences_alpha0.parquet"
            if abl_lstm_path.exists():
                abl_arr = build_dense_array(abl_lstm_path, all_ids, sample_ids)
                np.save(DENSE_ABL_OUT, abl_arr)
                size_mb = DENSE_ABL_OUT.stat().st_size / (1024 ** 2)
                print(f"  Saved: {DENSE_ABL_OUT.relative_to(ROOT)}  ({size_mb:.1f} MB)")
                del abl_arr
            else:
                print(f"  {abl_lstm_path.name} not found — skipping ablation array build")

    print("\n" + "=" * 64)
    print("Phase 5 data prep complete.")
    print(f"  Dense array : {dense_out.relative_to(ROOT)}")
    print(f"  ID index    : {index_out.relative_to(ROOT)}")
    print(f"  Train IDs   : {train_ids_out.relative_to(ROOT)}")
    print("=" * 64)


if __name__ == "__main__":
    main()
