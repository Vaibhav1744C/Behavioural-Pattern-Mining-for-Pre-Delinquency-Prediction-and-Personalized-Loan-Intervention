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

    print("Loading raw parquet...")
    df = pd.read_parquet(RAW_PARQUET)
    n_cols_start = df.shape[1]
    print(f"  loaded: {df.shape[0]:,} rows x {n_cols_start} columns")

    # ---- Criterion 1: resolved loans only + target -----------------------
    print("\n[1/3] Filtering to resolved loans and building target...")
    n_rows_start = len(df)
    df = build_target(df)
    rate = df["early_default"].mean()
    print(f"  kept {len(df):,} of {n_rows_start:,} loans "
          f"({len(df) / n_rows_start:.1%})")
    print(f"  early-default rate: {rate:.2%}  (paper reports 20.20%)")
    print(f"  paper kept 806,161 loans — ours: {len(df):,}")

    # loan_status has served its purpose; keeping it would leak the target
    df = df.drop(columns=["loan_status"])
    audit.append({"column": "loan_status", "reason": "target source (dropped after target construction)"})

    # ---- Criterion 2: leaky columns --------------------------------------
    print("\n[2/3] Dropping leaky / non-predictive columns...")
    reason_by_group = [
        (LEAKY_PAYMENT, "leaky: repayment activity (post-disbursement)"),
        (LEAKY_POST_ORIGINATION_CREDIT, "leaky: credit data refreshed post-origination"),
        (LEAKY_HARDSHIP, "leaky: hardship programme (post-distress)"),
        (LEAKY_SETTLEMENT, "leaky: debt settlement (post-default)"),
        (IDENTIFIERS_AND_FREETEXT, "identifier or free text (not used by base paper)"),
        (FUNDING_MECHANICS, "leaky: investor funding mechanics"),
    ]
    to_drop = []
    for group, reason in reason_by_group:
        for c in group:
            if c in df.columns and c not in to_drop:
                audit.append({"column": c, "reason": reason})
                to_drop.append(c)
    df = df.drop(columns=to_drop)
    print(f"  dropped {len(to_drop)} columns")

    missing_from_list = [c for c in LEAKY_COLUMNS if c not in df.columns and c not in to_drop]
    if missing_from_list:
        print(f"  note: {len(missing_from_list)} listed columns were not in the "
              f"data (harmless): {missing_from_list[:5]}...")

    # ---- Criterion 3: >40% missing ---------------------------------------
    print(f"\n[3/3] Dropping columns with >{MISSING_THRESHOLD:.0%} missing...")
    missing_frac = df.isna().mean()
    sparse = missing_frac[missing_frac > MISSING_THRESHOLD].sort_values(ascending=False)
    for c, frac in sparse.items():
        audit.append({"column": c, "reason": f"{frac:.1%} missing (>{MISSING_THRESHOLD:.0%})"})
    df = df.drop(columns=list(sparse.index))
    print(f"  dropped {len(sparse)} columns")

    # ---- Save -------------------------------------------------------------
    df.to_parquet(OUT_PARQUET, index=False)
    pd.DataFrame(audit).to_csv(AUDIT_CSV, index=False)

    n_cols_end = df.shape[1]
    print("\n" + "=" * 62)
    print(f"  columns: {n_cols_start} -> {n_cols_end}   (paper: 141 -> 57)")
    print(f"  rows:    {len(df):,}")
    print(f"  saved:   {OUT_PARQUET.relative_to(ROOT)}")
    print(f"  audit:   {AUDIT_CSV.relative_to(ROOT)}")
    print("=" * 62)

    if n_cols_end != 57:
        print(
            f"\n  NOTE: we have {n_cols_end} columns, the paper reports 57.\n"
            "  This is EXPECTED — the paper never publishes its column list, so\n"
            "  an exact match would be coincidence. Report the gap honestly and\n"
            "  cite reports/cleaning_audit.csv as your documented criteria.\n"
            "  If the gap is large (>10), review the audit file for columns you\n"
            "  may want to reclassify."
        )


if __name__ == "__main__":
    main()
