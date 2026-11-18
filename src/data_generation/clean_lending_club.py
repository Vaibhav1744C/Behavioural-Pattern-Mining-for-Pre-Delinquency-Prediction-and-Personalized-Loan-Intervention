"""
Phase 1: Clean the raw Lending Club data, replicating the filtering described in
Monje, Carrasco & Sanchez-Montanes (2025), "Machine Learning XAI for Early Loan
Default Prediction", Computational Economics 67, 4033-4062.

The base paper reports reducing 141 raw columns to 57, but does not publish the
resulting column list. We therefore RECONSTRUCT their filtering from the three
criteria they state explicitly:

  (1) Keep only loans whose outcome is known (fully paid / charged off /
      31-120 days late). Paper reports 806,161 such loans, 20.20% early default.
  (2) Drop variables containing post-origination ("leaky") information.
  (3) Drop variables with more than 40% missing values.

Every drop is logged with a reason to reports/cleaning_audit.csv so the process
is auditable and reportable in the paper.

Usage:
    python src/data/clean_lending_club.py
"""

from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
RAW_PARQUET = ROOT / "data" / "raw" / "lending_club_raw.parquet"
OUT_PARQUET = ROOT / "data" / "processed" / "lending_club_clean.parquet"
AUDIT_CSV = ROOT / "reports" / "cleaning_audit.csv"

MISSING_THRESHOLD = 0.40  # paper: drop columns with >40% missing


# --------------------------------------------------------------------------
# Criterion 2: leaky / non-predictive columns
#
# "Leaky" = information that only exists AFTER the loan was originated. Training
# on these inflates performance meaninglessly: e.g. `recoveries` is non-zero
# essentially only for loans that already defaulted.
# --------------------------------------------------------------------------

LEAKY_PAYMENT = [
    # Repayment activity — only known after disbursement
    "out_prncp", "out_prncp_inv",
    "total_pymnt", "total_pymnt_inv",
    "total_rec_prncp", "total_rec_int", "total_rec_late_fee",
    "recoveries", "collection_recovery_fee",
    "last_pymnt_d", "last_pymnt_amnt", "next_pymnt_d",
    "pymnt_plan",
]

LEAKY_POST_ORIGINATION_CREDIT = [
    # Credit-bureau data refreshed AFTER origination
    "last_credit_pull_d",
    "last_fico_range_high", "last_fico_range_low",
]

LEAKY_HARDSHIP = [
    # Hardship programme fields — only populated for already-distressed loans
    "hardship_flag", "hardship_type", "hardship_reason", "hardship_status",
    "deferral_term", "hardship_amount", "hardship_start_date",
    "hardship_end_date", "payment_plan_start_date", "hardship_length",
    "hardship_dpd", "hardship_loan_status",
    "orig_projected_additional_accrued_interest",
    "hardship_payoff_balance_amount", "hardship_last_payment_amount",
]

LEAKY_SETTLEMENT = [
    # Debt settlement — by definition post-default
    "debt_settlement_flag", "debt_settlement_flag_date",
    "settlement_status", "settlement_date", "settlement_amount",
    "settlement_percentage", "settlement_term",
]

IDENTIFIERS_AND_FREETEXT = [
    # No predictive content, or free text outside our scope (the base paper
    # does not use text features; Kriebel & Stitz 2022 do, if you want to
    # justify revisiting this later)
    "id", "member_id", "url", "desc", "title", "emp_title", "zip_code",
]

FUNDING_MECHANICS = [
    # Determined by investor behaviour on the platform after listing, not by
    # the borrower's own profile
    "funded_amnt_inv", "total_pymnt_inv",
]

LEAKY_COLUMNS = (
    LEAKY_PAYMENT
    + LEAKY_POST_ORIGINATION_CREDIT
    + LEAKY_HARDSHIP
    + LEAKY_SETTLEMENT
    + IDENTIFIERS_AND_FREETEXT
    + FUNDING_MECHANICS
)


# --------------------------------------------------------------------------
# Criterion 1: target construction
#
# Paper's definition (Section 4.3):
#   1 = loan that at some point was >30 days in default — charged off,
#       31-120 days late, OR fully paid but over a longer term than agreed
#   0 = fully paid within the agreed term
# Loans with unresolved status (Current, In Grace Period, <30 days late) are
# EXCLUDED, because their outcome is not yet known.
# --------------------------------------------------------------------------

STATUS_DEFAULT = [
    "Charged Off",
    "Late (31-120 days)",
    "Does not meet the credit policy. Status:Charged Off",
]
STATUS_PAID = [
    "Fully Paid",
    "Does not meet the credit policy. Status:Fully Paid",
]


def build_target(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only resolved loans and construct the binary early-default target."""
    status = df["loan_status"].astype("string").str.strip()

    keep = status.isin(STATUS_DEFAULT + STATUS_PAID)
    df = df.loc[keep].copy()
    status = status.loc[keep]

    df["early_default"] = status.isin(STATUS_DEFAULT).astype("int8")

    # NOTE: the paper also counts "fully paid but over a longer term than
    # agreed" as early default. Reconstructing that requires comparing
    # last_pymnt_d against issue_d + term — but last_pymnt_d is itself a leaky
    # column. We therefore implement the majority of the definition (charged
    # off / 31-120 late) and DOCUMENT this deviation rather than silently
    # using a leaky field to build the label. Expect our default rate to come
    # in slightly BELOW the paper's 20.20% as a result.
    return df


# --------------------------------------------------------------------------
def main() -> None:
    if not RAW_PARQUET.exists():
        raise FileNotFoundError(
            f"Raw parquet not found at {RAW_PARQUET}. Run load_raw.py first."
        )

    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_CSV.parent.mkdir(parents=True, exist_ok=True)

    audit: list[dict] = []

    # ---- Pass 1: read schema + compute missing fractions via chunked scan --
    print("Scanning parquet schema and missing fractions (chunked, low-RAM)...")
    pf = pq.ParquetFile(RAW_PARQUET)
    n_cols_start = len(pf.schema_arrow.names)
    col_names = pf.schema_arrow.names
    print(f"  columns in raw file: {n_cols_start}")

    # Accumulate null counts and total rows across batches
    null_counts = {c: 0 for c in col_names}
    n_rows_start = 0
    BATCH = 200_000
    for batch in pf.iter_batches(batch_size=BATCH):
        df_batch = batch.to_pandas()
        n_rows_start += len(df_batch)
        for c in col_names:
            if c in df_batch.columns:
                null_counts[c] += int(df_batch[c].isna().sum())

    print(f"  total rows in raw file: {n_rows_start:,}")

    # ---- Determine columns to drop upfront --------------------------------

    # Criterion 2: leaky
    reason_by_group = [
        (LEAKY_PAYMENT,                "leaky: repayment activity (post-disbursement)"),
        (LEAKY_POST_ORIGINATION_CREDIT,"leaky: credit data refreshed post-origination"),
        (LEAKY_HARDSHIP,               "leaky: hardship programme (post-distress)"),
        (LEAKY_SETTLEMENT,             "leaky: debt settlement (post-default)"),
        (IDENTIFIERS_AND_FREETEXT,     "identifier or free text (not used by base paper)"),
        (FUNDING_MECHANICS,            "leaky: investor funding mechanics"),
    ]
    leaky_drop = set()
    for group, reason in reason_by_group:
        for c in group:
            if c in col_names and c not in leaky_drop:
                audit.append({"column": c, "reason": reason})
                leaky_drop.add(c)
    audit.append({"column": "loan_status",
                  "reason": "target source (dropped after target construction)"})

    # Criterion 3: >40% missing
    sparse_drop = set()
    for c, nc in null_counts.items():
        frac = nc / n_rows_start if n_rows_start else 0
        if frac > MISSING_THRESHOLD and c not in leaky_drop and c != "loan_status":
            audit.append({"column": c,
                          "reason": f"{frac:.1%} missing (>{MISSING_THRESHOLD:.0%})"})
            sparse_drop.add(c)

    all_drop = leaky_drop | sparse_drop | {"loan_status"}
    keep_cols = [c for c in col_names if c not in all_drop]
    print(f"\n  leaky cols to drop   : {len(leaky_drop)}")
    print(f"  sparse cols to drop  : {len(sparse_drop)}")
    print(f"  columns to keep      : {len(keep_cols)}")

    # ---- Pass 2: filter rows + write output in chunks ---------------------
    print("\nFiltering rows and writing cleaned parquet (chunked)...")
    writer = None
    n_kept = 0
    default_count = 0

    for batch in pf.iter_batches(batch_size=BATCH, columns=keep_cols + ["loan_status"]):
        df_batch = batch.to_pandas()

        # Criterion 1: keep only resolved loans + build target
        status = df_batch["loan_status"].astype("string").str.strip()
        keep_mask = status.isin(STATUS_DEFAULT + STATUS_PAID)
        df_batch = df_batch.loc[keep_mask].copy()
        if df_batch.empty:
            continue
        status = status.loc[keep_mask]
        df_batch["early_default"] = status.isin(STATUS_DEFAULT).astype("int8")
        df_batch = df_batch.drop(columns=["loan_status"])

        n_kept += len(df_batch)
        default_count += int(df_batch["early_default"].sum())

        # Write chunk
        table = pa.Table.from_pandas(df_batch, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(OUT_PARQUET, table.schema)
        writer.write_table(table)
        print(f"  rows kept so far: {n_kept:,}", end="\r")

    if writer:
        writer.close()

    rate = default_count / n_kept if n_kept else 0
    n_cols_end = len(keep_cols) + 1  # +1 for early_default

    # Save audit
    pd.DataFrame(audit).to_csv(AUDIT_CSV, index=False)

    print(f"\n\n{'=' * 62}")
    print(f"  rows kept        : {n_kept:,}  (of {n_rows_start:,})")
    print(f"  early-default    : {rate:.2%}  (paper: 20.20%)")
    print(f"  columns          : {n_cols_start} -> {n_cols_end}  (paper: 141 -> 57)")
    print(f"  saved            : {OUT_PARQUET.relative_to(ROOT)}")
    print(f"  audit            : {AUDIT_CSV.relative_to(ROOT)}")
    print(f"{'=' * 62}")

    if n_cols_end != 57:
        print(
            f"\n  NOTE: we have {n_cols_end} columns, the paper reports 57.\n"
            "  This is EXPECTED — the paper never publishes its column list.\n"
            "  See reports/cleaning_audit.csv for every drop decision.\n"
        )


if __name__ == "__main__":
    main()
