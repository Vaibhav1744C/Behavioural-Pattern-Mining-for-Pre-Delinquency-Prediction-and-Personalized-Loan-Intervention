"""
build_features.py
-----------------
Phase 4: Feature engineering from synthetic behavioural sequences.

Reads the long-format behavioural sequences parquet and the static cleaned
data, and produces two artefacts:

  1. data/processed/lstm_sequences.parquet
     Per-month features for LSTM input, post-padded to max_len=24.
     One row per (loan_id, month), padded rows have all features = 0.

  2. data/processed/sequence_context_features.parquet
     One row per borrower. Sequence-level summary indices used as static
     context alongside the LSTM hidden state.

Variable-length sequence handling
  - Sequences range from 2 to 24 months.
  - Post-padded to 24 with zeros.
  - seq_len stored explicitly — use with torch.nn.utils.rnn.pack_padded_sequence
    in Phase 5 so the LSTM never processes padded timesteps.

Per-borrower normalisation
  - Z-score uses each borrower's OWN mean/std across their sequence.
  - Income scale varies enormously across borrowers (e.g. $20k vs $200k/yr),
    so global normalisation would destroy relative within-borrower dynamics.
  - Guarded divide: std=0 -> normalised value = 0.0 (not NaN).

Usage:
    python src/features/build_features.py                   # full data
    python src/features/build_features.py --sample 2000     # sample test
    python src/features/build_features.py --seq-parquet path/to/custom.parquet

Outputs:
    data/processed/lstm_sequences.parquet
    data/processed/sequence_context_features.parquet
    reports/feature_engineering_report.json

Requirements:
    pip install pandas pyarrow numpy scipy
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.stats import linregress

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).resolve().parents[2]
CLEAN_PATH    = ROOT / "data" / "processed" / "lending_club_clean.parquet"
SEQ_PATH      = ROOT / "data" / "synthetic"  / "behavioural_sequences.parquet"
LSTM_OUT      = ROOT / "data" / "processed"  / "lstm_sequences.parquet"
CONTEXT_OUT   = ROOT / "data" / "processed"  / "sequence_context_features.parquet"
REPORT_OUT    = ROOT / "reports"             / "feature_engineering_report.json"

MAX_LEN = 24   # post-pad target length

# ── EMI encoding ──────────────────────────────────────────────────────────────
EMI_MAP = {"on_time": 0, "delayed": 1, "missed": 2}


# ── Per-borrower z-score (guarded) ────────────────────────────────────────────

def zscore_per_borrower(
    df: pd.DataFrame,
    col: str,
    out_col: str,
) -> pd.DataFrame:
    """
    Z-score `col` using each borrower's own mean/std across their sequence.
    std=0 -> 0.0 (guarded divide, no NaN).
    Result written to `out_col` in-place.
    """
    stats = df.groupby("loan_id")[col].agg(["mean", "std"]).rename(
        columns={"mean": f"_{col}_mean", "std": f"_{col}_std"}
    )
    df = df.join(stats, on="loan_id")
    std_col = f"_{col}_std"
    mean_col = f"_{col}_mean"
    df[out_col] = np.where(
        df[std_col].fillna(0) > 0,
        (df[col] - df[mean_col]) / df[std_col],
        0.0,
    )
    df = df.drop(columns=[mean_col, std_col])
    return df


# ── Sequence-level feature helpers ───────────────────────────────────────────

def safe_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Linear regression slope. Returns 0.0 if not enough points."""
    if len(x) < 2:
        return 0.0
    try:
        slope, *_ = linregress(x, y)
        return float(slope) if np.isfinite(slope) else 0.0
    except Exception:
        return 0.0


def safe_std(arr: np.ndarray) -> float:
    if len(arr) < 2:
        return 0.0
    v = float(np.std(arr))
    return v if np.isfinite(v) else 0.0


def safe_mean(arr: np.ndarray) -> float:
    if len(arr) == 0:
        return 0.0
    v = float(np.mean(arr))
    return v if np.isfinite(v) else 0.0


def cashflow_compression(spend: np.ndarray) -> float:
    """
    Ratio of mean spend in first half to mean spend in second half.
    > 1  -> spending was higher early (typical pre-distress pattern)
    < 1  -> spending compressed early (atypical)
    Returns 1.0 if second-half mean is 0 (no compression measurable).
    """
    n = len(spend)
    if n < 2:
        return 1.0
    mid = n // 2
    first_half  = safe_mean(spend[:mid])
    second_half = safe_mean(spend[mid:])
    if second_half == 0:
        return 1.0
    ratio = first_half / second_half
    return float(ratio) if np.isfinite(ratio) else 1.0


# ── Build per-month (LSTM) features ──────────────────────────────────────────

def build_lstm_features(seq_df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes long-format sequence DataFrame and returns per-month LSTM features,
    post-padded to MAX_LEN rows per borrower.

    Output columns per month:
      loan_id, month, salary_delay_days_z, discretionary_spend_z,
      account_balance_z, savings_balance_z,
      delta_account_balance, delta_savings_balance, emi_status_enc
    """
    df = seq_df.copy()

    # Numeric coercion (parquet may have stored as object in sample runs)
    for col in ["salary_delay_days", "discretionary_spend",
                "account_balance", "savings_balance", "salary_credit"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # EMI ordinal encoding
    df["emi_status_enc"] = df["emi_status"].map(EMI_MAP).fillna(0).astype("int8")

    # Per-borrower z-scores
    for raw, out in [
        ("salary_delay_days",   "salary_delay_days_z"),
        ("discretionary_spend", "discretionary_spend_z"),
        ("account_balance",     "account_balance_z"),
        ("savings_balance",     "savings_balance_z"),
    ]:
        df = zscore_per_borrower(df, raw, out)

    # Month-over-month deltas (within each borrower's sequence)
    df = df.sort_values(["loan_id", "month"])
    df["delta_account_balance"] = df.groupby("loan_id")["account_balance"].diff().fillna(0.0)
    df["delta_savings_balance"] = df.groupby("loan_id")["savings_balance"].diff().fillna(0.0)

    lstm_cols = [
        "loan_id", "month",
        "salary_delay_days_z", "discretionary_spend_z",
        "account_balance_z", "savings_balance_z",
        "delta_account_balance", "delta_savings_balance",
        "emi_status_enc",
    ]
    df = df[lstm_cols]

    # Post-pad each borrower to MAX_LEN months with zeros
    all_ids   = df["loan_id"].unique()
    pad_rows  = []
    for lid in all_ids:
        sub = df[df["loan_id"] == lid].sort_values("month")
        actual_len = len(sub)
        if actual_len < MAX_LEN:
            pad_months = range(actual_len, MAX_LEN)
            pad_block  = pd.DataFrame({
                "loan_id":               lid,
                "month":                 list(pad_months),
                "salary_delay_days_z":   0.0,
                "discretionary_spend_z": 0.0,
                "account_balance_z":     0.0,
                "savings_balance_z":     0.0,
                "delta_account_balance": 0.0,
                "delta_savings_balance": 0.0,
                "emi_status_enc":        0,
            })
            pad_rows.append(pad_block)

    if pad_rows:
        df = pd.concat([df] + pad_rows, ignore_index=True)

    df = df.sort_values(["loan_id", "month"]).reset_index(drop=True)
    return df


# ── Build sequence-context (static summary) features ─────────────────────────

def build_context_features(seq_df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per borrower. Sequence-level indices fed as static context
    alongside the LSTM.
    """
    df = seq_df.copy()
    for col in ["salary_credit", "salary_delay_days",
                "savings_balance", "discretionary_spend"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    df["emi_status_enc"] = df["emi_status"].map(EMI_MAP).fillna(0).astype(int)
    df = df.sort_values(["loan_id", "month"])

    records = []
    for lid, grp in df.groupby("loan_id"):
        grp      = grp.sort_values("month")
        months   = grp["month"].values.astype(float)
        sal_cr   = grp["salary_credit"].values
        sal_del  = grp["salary_delay_days"].values
        sav_bal  = grp["savings_balance"].values
        disc_sp  = grp["discretionary_spend"].values
        emi_enc  = grp["emi_status_enc"].values
        L        = len(grp)

        # Salary Stability Index: std/mean (coefficient of variation)
        sal_mean = safe_mean(sal_cr)
        sal_std  = safe_std(sal_cr)
        sal_stab = (sal_std / sal_mean) if sal_mean > 0 else 0.0

        # Salary Delay Trend: slope of delay days over time
        sal_delay_trend = safe_slope(months, sal_del)

        # Savings slope and volatility
        sav_slope = safe_slope(months, sav_bal)
        sav_vol   = safe_std(sav_bal)

        # Cash-Flow Compression Score
        cf_compression = cashflow_compression(disc_sp)

        # EMI stress count: delayed + missed months
        emi_stress = int(np.sum(emi_enc >= 1))

        records.append({
            "loan_id":               int(lid),
            "salary_stability_idx":  round(sal_stab, 6),
            "salary_delay_trend":    round(sal_delay_trend, 6),
            "savings_slope":         round(sav_slope, 6),
            "savings_volatility":    round(sav_vol, 2),
            "cashflow_compression":  round(cf_compression, 6),
            "emi_stress_count":      emi_stress,
            "seq_len":               L,
        })

    return pd.DataFrame(records)


# ── Validation ────────────────────────────────────────────────────────────────

def validate_outputs(
    lstm_df: pd.DataFrame,
    ctx_df: pd.DataFrame,
    seq_df: pd.DataFrame,
    n_spot: int = 10,
) -> dict:
    """
    Run validation checks and return a report dict.
    Raises AssertionError if any critical check fails.
    """
    report = {}

    # --- Check 1: no NaNs in either output -----------------------------------
    lstm_nans = lstm_df.isnull().sum().sum()
    ctx_nans  = ctx_df.isnull().sum().sum()
    print(f"  NaNs in lstm_sequences:          {lstm_nans}  (must be 0)")
    print(f"  NaNs in context_features:        {ctx_nans}   (must be 0)")
    assert lstm_nans == 0, f"NaNs found in lstm_sequences: {lstm_nans}"
    assert ctx_nans  == 0, f"NaNs found in context_features: {ctx_nans}"
    report["lstm_nans"] = int(lstm_nans)
    report["ctx_nans"]  = int(ctx_nans)

    # --- Check 2: shapes -----------------------------------------------------
    report["lstm_shape"]    = list(lstm_df.shape)
    report["ctx_shape"]     = list(ctx_df.shape)
    report["lstm_dtypes"]   = lstm_df.dtypes.astype(str).to_dict()
    report["ctx_dtypes"]    = ctx_df.dtypes.astype(str).to_dict()
    print(f"  lstm_sequences shape:            {lstm_df.shape}")
    print(f"  context_features shape:          {ctx_df.shape}")

    # --- Check 3: every borrower has exactly MAX_LEN rows in lstm_df ----------
    counts = lstm_df.groupby("loan_id")["month"].count()
    wrong  = (counts != MAX_LEN).sum()
    print(f"  Borrowers with !={MAX_LEN} months in lstm: {wrong}  (must be 0)")
    assert wrong == 0, f"{wrong} borrowers don't have exactly {MAX_LEN} padded rows"
    report["wrong_length_count"] = int(wrong)

    # --- Check 4: spot-check shock borrowers ---------------------------------
    # Borrowers with high emi_stress_count should have cashflow_compression > 1
    # (more spending in first half than second — pre-distress spike)
    if "emi_stress_count" in ctx_df.columns:
        high_stress = ctx_df[ctx_df["emi_stress_count"] >= 3]
        if len(high_stress) > 0:
            mean_cf = high_stress["cashflow_compression"].mean()
            print(f"  Mean cashflow_compression for high-stress borrowers: {mean_cf:.3f}  (expect >1.0)")
            report["high_stress_mean_cf"] = round(float(mean_cf), 4)

    # --- Spot-check 5-10 borrowers by hand -----------------------------------
    spot_ids = ctx_df.nlargest(5, "emi_stress_count")["loan_id"].tolist()
    spot_ids += ctx_df.nsmallest(5, "emi_stress_count")["loan_id"].tolist()
    spot_results = []
    for lid in spot_ids[:n_spot]:
        ctx_row = ctx_df[ctx_df["loan_id"] == lid].iloc[0]
        spot_results.append({
            "loan_id":              int(lid),
            "seq_len":              int(ctx_row["seq_len"]),
            "emi_stress_count":     int(ctx_row["emi_stress_count"]),
            "cashflow_compression": round(float(ctx_row["cashflow_compression"]), 4),
            "salary_delay_trend":   round(float(ctx_row["salary_delay_trend"]), 4),
            "savings_slope":        round(float(ctx_row["savings_slope"]), 4),
        })
    report["spot_checks"] = spot_results

    print("  All validation checks passed [OK]")
    return report


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Phase 4: Feature engineering")
    p.add_argument("--seq-parquet", type=str, default=None,
                   help="Path to behavioural sequences parquet (default: data/synthetic/behavioural_sequences.parquet)")
    p.add_argument("--sample",      type=int, default=None,
                   help="Only process N borrowers (for testing)")
    p.add_argument("--chunk-size",  type=int, default=50_000,
                   help="Borrowers per chunk for chunked parquet write")
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 64)
    print("Phase 4 — Feature Engineering")
    print("=" * 64)

    seq_path = Path(args.seq_parquet) if args.seq_parquet else SEQ_PATH

    # Derive output paths — if custom seq-parquet given, use a matching suffix
    # e.g. behavioural_sequences_alpha0.parquet → lstm_sequences_alpha0.parquet
    if args.seq_parquet:
        suffix = seq_path.stem.replace("behavioural_sequences", "").strip("_") or ""
        suffix = f"_{suffix}" if suffix else ""
        lstm_out    = ROOT / "data" / "processed" / f"lstm_sequences{suffix}.parquet"
        context_out = ROOT / "data" / "processed" / f"sequence_context_features{suffix}.parquet"
        report_out  = ROOT / "reports" / f"feature_engineering_report{suffix}.json"
    else:
        lstm_out    = LSTM_OUT
        context_out = CONTEXT_OUT
        report_out  = REPORT_OUT

    # ── Generate sample sequences if full parquet doesn't exist yet -----------
    if not seq_path.exists():
        print(f"\n  {seq_path.name} not found.")
        if args.sample:
            print(f"  Generating a {args.sample}-borrower sample on-the-fly for testing...")
            import sys
            sys.path.insert(0, str(ROOT / "src" / "data_generation"))
            from generate_behavioural_sequences import (
                generate_sequences_for_chunk,
                DEFAULT_ALPHA, DEFAULT_P_SHOCK_DEFAULT,
                DEFAULT_P_SHOCK_NONDEFAULT, DEFAULT_MAX_SEQ_LEN,
                DEFAULT_MAX_DELAY_DAYS,
            )
            static_df = pd.read_parquet(CLEAN_PATH)
            if "loan_id" not in static_df.columns:
                static_df = static_df.reset_index().rename(columns={"index": "loan_id"})
            static_df["term"] = pd.to_numeric(
                static_df["term"].astype(str)
                .str.replace("months", "", regex=False).str.strip(),
                errors="coerce"
            ).fillna(36)
            sample_static = static_df.sample(n=min(args.sample, len(static_df)), random_state=42)
            rng = __import__("numpy").random.default_rng(42)
            seq_df, _ = generate_sequences_for_chunk(
                sample_static,
                DEFAULT_ALPHA, DEFAULT_P_SHOCK_DEFAULT,
                DEFAULT_P_SHOCK_NONDEFAULT, DEFAULT_MAX_SEQ_LEN,
                DEFAULT_MAX_DELAY_DAYS, rng,
                return_shock_flags=True,
            )
            seq_path = ROOT / "data" / "synthetic" / f"behavioural_sequences_sample{args.sample}.parquet"
            seq_df["emi_status"] = seq_df["emi_status"].astype(str)
            seq_df.to_parquet(seq_path, index=False)
            print(f"  Sample sequences saved to {seq_path.relative_to(ROOT)}")
        else:
            raise FileNotFoundError(
                f"Sequences parquet not found at {seq_path}.\n"
                "Run generate_behavioural_sequences.py first, or use --sample N for a test run."
            )

    # ── Load sequences ────────────────────────────────────────────────────────
    print(f"\nLoading sequences: {seq_path.relative_to(ROOT)}")

    # Get all unique loan_ids without loading full data
    pf          = pq.ParquetFile(seq_path)
    total_rows  = pf.metadata.num_rows
    print(f"  Total rows in parquet: {total_rows:,}")

    # Read just loan_id column to get borrower list (memory efficient)
    all_ids = pq.read_table(seq_path, columns=["loan_id"]).to_pandas()["loan_id"].unique()
    n_borrowers = len(all_ids)
    print(f"  Borrowers: {n_borrowers:,}")

    if args.sample and not args.seq_parquet:
        all_ids = all_ids[:args.sample]
        n_borrowers = len(all_ids)
        print(f"  Subsetting to {n_borrowers:,} borrowers")

    # ── Chunked processing ────────────────────────────────────────────────────
    CHUNK_SIZE   = args.chunk_size
    id_chunks    = [all_ids[i:i+CHUNK_SIZE] for i in range(0, n_borrowers, CHUNK_SIZE)]
    n_chunks     = len(id_chunks)
    print(f"  Processing {n_chunks} chunks of up to {CHUNK_SIZE:,} borrowers")

    LSTM_OUT.parent.mkdir(parents=True, exist_ok=True)
    lstm_writer  = None
    ctx_records  = []
    total_lstm   = 0

    # ── True streaming: one parquet pass per borrower chunk ──────────────────
    # For each id_chunk, stream through all row groups of the parquet, keeping
    # only rows belonging to that chunk. Never hold the full file in RAM.
    BATCH_SIZE = 500_000   # rows per pyarrow batch

    print("\nBuilding features (streaming)...")

    for ci, id_chunk in enumerate(id_chunks):
        id_set       = set(id_chunk)
        pieces       = []

        for batch in pf.iter_batches(
            batch_size=BATCH_SIZE,
            columns=["loan_id", "month", "salary_credit",
                     "salary_delay_days", "account_balance",
                     "savings_balance", "discretionary_spend", "emi_status"],
        ):
            df_b = batch.to_pandas()
            df_b = df_b[df_b["loan_id"].isin(id_set)]
            if not df_b.empty:
                pieces.append(df_b)

        if not pieces:
            continue
        chunk_df = pd.concat(pieces, ignore_index=True)
        del pieces

        lstm_chunk = build_lstm_features(chunk_df)
        ctx_chunk  = build_context_features(chunk_df)
        ctx_records.append(ctx_chunk)
        del chunk_df

        table = pa.Table.from_pandas(lstm_chunk, preserve_index=False)
        if lstm_writer is None:
            if lstm_out.exists():
                lstm_out.unlink()
            lstm_writer = pq.ParquetWriter(lstm_out, table.schema, compression="snappy")
        lstm_writer.write_table(table)
        total_lstm += len(lstm_chunk)
        del lstm_chunk

        print(f"  Chunk {ci+1:>4}/{n_chunks}: {len(id_chunk):>7,} borrowers | "
              f"lstm rows so far: {total_lstm:>12,}")

    if lstm_writer:
        lstm_writer.close()

    ctx_df = pd.concat(ctx_records, ignore_index=True)
    del ctx_records

    print(f"\n  lstm rows total      : {total_lstm:,}  (padded to {MAX_LEN} months)")
    print(f"  context_features     : {ctx_df.shape}")

    # ── Validation (run on first chunk only to avoid OOM) ────────────────────
    print("\nRunning validation checks (on first chunk)...")
    first_chunk_ids = set(id_chunks[0])
    lstm_sample = pd.read_parquet(lstm_out, filters=[("loan_id", "in", list(first_chunk_ids))])
    ctx_sample  = ctx_df[ctx_df["loan_id"].isin(first_chunk_ids)]
    seq_sample  = pd.read_parquet(seq_path, filters=[("loan_id", "in", list(first_chunk_ids))])
    report = validate_outputs(lstm_sample, ctx_sample, seq_sample)

    # ── Save context (lstm already saved incrementally) ───────────────────────
    print("\nSaving context features...")
    ctx_df.to_parquet(context_out, index=False)

    report_out.parent.mkdir(parents=True, exist_ok=True)
    with open(report_out, "w") as f:
        json.dump(report, f, indent=2)

    print(f"  lstm_sequences   -> {lstm_out.relative_to(ROOT)}")
    print(f"  context_features -> {context_out.relative_to(ROOT)}")
    print(f"  report           -> {report_out.relative_to(ROOT)}")

    print("\n" + "=" * 64)
    print("Phase 4 complete.")
    print(f"  LSTM rows total   : {total_lstm:,}  (padded to {MAX_LEN} months each)")
    print(f"  Context shape     : {ctx_df.shape}  (one row per borrower)")
    print("=" * 64)


if __name__ == "__main__":
    main()
