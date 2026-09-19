"""
load_raw.py
-----------
Loads the Lending Club 2007-2020 Q1 dataset from the local raw file
and saves it as a parquet file in data/raw/ for downstream processing.

Reads in chunks to stay within RAM. Casts every chunk to a fixed Arrow
schema (all-string columns cast to large_string, numeric to double) so
the ParquetWriter never sees a schema mismatch — even for columns that
are entirely null in some chunks.

Usage:
    python src/data_generation/load_raw.py

Requirements:
    pip install pandas pyarrow openpyxl
"""

import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DIR  = os.path.join(ROOT_DIR, "data", "raw")

RAW_FILE  = os.path.join(RAW_DIR, "Loan_status_2007-2020Q3.gzip")
DICT_FILE = os.path.join(RAW_DIR, "LCDataDictionary.xlsx")
OUT_PATH  = os.path.join(RAW_DIR, "lending_club_raw.parquet")

CHUNK_SIZE = 100_000


# ── Columns that are truly numeric (everything else → string) ─────────────────
# Derived from the dataset's known schema; all numeric cols use float64 so
# NaN values are handled correctly.
NUMERIC_COLS = {
    "loan_amnt", "funded_amnt", "funded_amnt_inv", "installment",
    "annual_inc", "dti", "delinq_2yrs", "fico_range_low", "fico_range_high",
    "inq_last_6mths", "mths_since_last_delinq", "mths_since_last_record",
    "open_acc", "pub_rec", "revol_bal", "total_acc", "out_prncp",
    "out_prncp_inv", "total_pymnt", "total_pymnt_inv", "total_rec_prncp",
    "total_rec_int", "total_rec_late_fee", "recoveries",
    "collection_recovery_fee", "last_pymnt_amnt", "last_fico_range_high",
    "last_fico_range_low", "collections_12_mths_ex_med",
    "mths_since_last_major_derog", "policy_code", "annual_inc_joint",
    "dti_joint", "acc_now_delinq", "tot_coll_amt", "tot_cur_bal",
    "open_acc_6m", "open_act_il", "open_il_12m", "open_il_24m",
    "mths_since_rcnt_il", "total_bal_il", "il_util", "open_rv_12m",
    "open_rv_24m", "max_bal_bc", "all_util", "total_rev_hi_lim", "inq_fi",
    "total_cu_tl", "inq_last_12m", "acc_open_past_24mths", "avg_cur_bal",
    "bc_open_to_buy", "bc_util", "chargeoff_within_12_mths", "delinq_amnt",
    "mo_sin_old_il_acct", "mo_sin_old_rev_tl_op", "mo_sin_rcnt_rev_tl_op",
    "mo_sin_rcnt_tl", "mort_acc", "mths_since_recent_bc",
    "mths_since_recent_bc_dlq", "mths_since_recent_inq",
    "mths_since_recent_revol_delinq", "num_accts_ever_120_pd",
    "num_actv_bc_tl", "num_actv_rev_tl", "num_bc_sats", "num_bc_tl",
    "num_il_tl", "num_op_rev_tl", "num_rev_accts", "num_rev_tl_bal_gt_0",
    "num_sats", "num_tl_120dpd_2m", "num_tl_30dpd", "num_tl_90g_dpd_24m",
    "num_tl_op_past_12m", "pct_tl_nvr_dlq", "percent_bc_gt_75",
    "pub_rec_bankruptcies", "tax_liens", "tot_hi_cred_lim",
    "total_bal_ex_mort", "total_bc_limit", "total_il_high_credit_limit",
    "revol_bal_joint", "sec_app_fico_range_low", "sec_app_fico_range_high",
    "sec_app_inq_last_6mths", "sec_app_mort_acc", "sec_app_open_acc",
    "sec_app_revol_util", "sec_app_open_act_il", "sec_app_num_rev_accts",
    "sec_app_chargeoff_within_12_mths", "sec_app_collections_12_mths_ex_med",
    "deferral_term", "hardship_amount", "hardship_length", "hardship_dpd",
    "orig_projected_additional_accrued_interest",
    "hardship_payoff_balance_amount", "hardship_last_payment_amount",
    "verification_status_joint",
}


def get_columns() -> list:
    """Read just the header row to get the full column list."""
    cols = pd.read_csv(RAW_FILE, compression=None, nrows=0).columns.tolist()
    print(f"  {len(cols)} columns found: {cols[:5]} ... {cols[-3:]}")
    return cols


def build_arrow_schema(columns: list) -> pa.Schema:
    """
    Build a fixed Arrow schema: numeric cols → float64, everything else →
    large_string. large_string handles null-only chunks without type conflicts.
    """
    fields = []
    for col in columns:
        if col in NUMERIC_COLS:
            fields.append(pa.field(col, pa.float64()))
        else:
            fields.append(pa.field(col, pa.large_utf8()))
    return pa.schema(fields)


def cast_chunk(chunk: pd.DataFrame, schema: pa.Schema) -> pa.Table:
    """Convert a pandas chunk to an Arrow table that matches schema exactly."""
    arrays = []
    for field in schema:
        col = field.name
        series = chunk[col] if col in chunk.columns else pd.Series([None] * len(chunk))
        if field.type == pa.float64():
            arrays.append(pa.array(pd.to_numeric(series, errors="coerce").tolist(),
                                   type=pa.float64()))
        else:
            # Cast to string; replace float NaN representations with None
            str_series = series.where(series.notna(), other=None).astype(str)
            str_series = str_series.replace("nan", None).replace("<NA>", None)
            arrays.append(pa.array(str_series.tolist(), type=pa.large_utf8()))
    return pa.table(arrays, schema=schema)


def load_and_save_parquet() -> None:
    print("Scanning header...")
    columns = get_columns()
    schema  = build_arrow_schema(columns)

    print(f"\nReading : {RAW_FILE}")
    print(f"Writing : {OUT_PATH}")
    print(f"Chunk   : {CHUNK_SIZE:,} rows\n")

    if os.path.exists(OUT_PATH):
        os.remove(OUT_PATH)

    writer     = pq.ParquetWriter(OUT_PATH, schema, compression="snappy")
    total_rows = 0
    chunk_num  = 0

    reader = pd.read_csv(
        RAW_FILE,
        compression=None,
        chunksize=CHUNK_SIZE,
        dtype=str,           # read everything as str — Arrow handles casting
        on_bad_lines="skip",
    )

    for chunk in reader:
        chunk_num  += 1
        total_rows += len(chunk)
        table = cast_chunk(chunk, schema)
        writer.write_table(table)
        print(f"  Chunk {chunk_num:>4}: {len(chunk):>7,} rows  |  total: {total_rows:>10,}")

    writer.close()
    size_mb = os.path.getsize(OUT_PATH) / (1024 ** 2)
    print(f"\nFinished. {total_rows:,} rows → {OUT_PATH}  ({size_mb:.1f} MB)")


def load_data_dictionary() -> pd.DataFrame:
    print(f"\nReading data dictionary: {DICT_FILE}")
    dd = pd.read_excel(DICT_FILE)
    print(f"Dictionary has {len(dd)} entries")
    return dd


def verify_parquet() -> None:
    print("\nVerifying parquet output...")
    df = pd.read_parquet(OUT_PATH)
    print(f"Shape  : {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print("\nFirst 5 rows:")
    print(df.head())


if __name__ == "__main__":
    load_and_save_parquet()
    load_data_dictionary()
    verify_parquet()
