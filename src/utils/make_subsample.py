"""Select a fixed, stratified 200,000-loan subset from the canonical split.

Loan IDs in this project are row positions in lending_club_clean.parquet.
The cleaned parquet has no loan_id column, so this script reconstructs the
same IDs used by the existing model scripts with reset_index().

The original 70/30 test assignment is reproduced first (random_state=42,
stratified on early_default). Then 60,000 IDs are sampled from that original
test partition and 140,000 from its original training partition, each
stratified on early_default. No model or feature method is changed.

Run only when approved:
    python src/utils/make_subsample.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[2]
CLEAN_PATH = ROOT / "data" / "processed" / "lending_club_clean.parquet"
OUTPUT_PATH = ROOT / "data" / "processed" / "subsample_loan_ids.parquet"
X_TEST_PATH = ROOT / "data" / "processed" / "X_test.parquet"
Y_TEST_PATH = ROOT / "data" / "processed" / "y_test.parquet"


def verify_original_test_split(clean: pd.DataFrame) -> pd.DataFrame:
    """Verify the reconstructed canonical test rows against saved Phase 2 data."""
    missing_paths = [path for path in (X_TEST_PATH, Y_TEST_PATH) if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(
            "Cannot verify original split; missing: "
            + ", ".join(str(path) for path in missing_paths)
        )

    required = {"early_default", "annual_inc", "dti", "grade"}
    missing_columns = required.difference(clean.columns)
    if missing_columns:
        raise KeyError(f"Cleaned data is missing verification columns: {sorted(missing_columns)}")

    positions = pd.Series(range(len(clean)), dtype="int64")
    labels = clean["early_default"].astype("int8").reset_index(drop=True)
    _, test_positions, _, y_expected = train_test_split(
        positions,
        labels,
        test_size=0.30,
        random_state=42,
        stratify=labels,
    )
    if len(test_positions) != 563_771:
        raise RuntimeError(
            f"Reproduced test split has {len(test_positions):,} rows; expected 563,771"
        )

    y_saved = pd.read_parquet(Y_TEST_PATH)
    if "early_default" not in y_saved.columns:
        raise KeyError("y_test.parquet has no early_default column")
    y_saved_values = y_saved["early_default"].to_numpy()
    if not (y_expected.to_numpy() == y_saved_values).all():
        raise RuntimeError("Reproduced test labels do not match y_test.parquet row for row")

    x_saved = pd.read_parquet(X_TEST_PATH, columns=["annual_inc", "dti", "grade"])
    if len(x_saved) != len(test_positions):
        raise RuntimeError(
            f"X_test.parquet has {len(x_saved):,} rows; reproduced split has {len(test_positions):,}"
        )

    expected_features = clean.iloc[test_positions][["annual_inc", "dti", "grade"]].copy()
    # Match the exact ordinal encoding applied by baseline_xgboost.py before
    # saving X_test.parquet. No model is loaded or fitted during verification.
    grade_map = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7}
    expected_features["grade"] = expected_features["grade"].map(grade_map).astype("float64")

    for column in ("annual_inc", "dti", "grade"):
        expected_values = pd.to_numeric(expected_features[column], errors="coerce").to_numpy(dtype="float64")
        saved_values = pd.to_numeric(x_saved[column], errors="coerce").to_numpy(dtype="float64")
        import numpy as np

        if not np.array_equal(expected_values, saved_values, equal_nan=True):
            raise RuntimeError(
                f"Reproduced test feature {column!r} does not match X_test.parquet row for row"
            )

    print("Canonical split verification passed:")
    print("  test rows: 563,771")
    print("  early_default: exact row-order match to y_test.parquet")
    print("  annual_inc, dti, grade: exact row-order match to X_test.parquet")
    return clean.iloc[test_positions][["loan_id", "early_default"]]


def make_subsample(
    clean_path: Path = CLEAN_PATH,
    output_path: Path = OUTPUT_PATH,
    n_borrowers: int = 200_000,
    test_fraction: float = 0.30,
    random_state: int = 42,
) -> pd.DataFrame:
    """Build and save the fixed-ID subset with its original split membership."""
    if not clean_path.exists():
        raise FileNotFoundError(f"Cleaned dataset not found: {clean_path}")
    if n_borrowers <= 0:
        raise ValueError("n_borrowers must be positive")
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be strictly between 0 and 1")

    clean = pd.read_parquet(clean_path)
    if "early_default" not in clean.columns:
        raise KeyError("Cleaned dataset must contain early_default")

    # Existing project scripts create loan_id from the clean parquet row index.
    if "loan_id" not in clean.columns:
        clean = clean.reset_index().rename(columns={"index": "loan_id"})
    clean["loan_id"] = clean["loan_id"].astype("int64")
    clean["early_default"] = clean["early_default"].astype("int8")

    if clean["loan_id"].duplicated().any():
        raise ValueError("loan_id values must be unique")

    n_test = int(round(n_borrowers * test_fraction))
    n_train = n_borrowers - n_test
    if n_borrowers > len(clean):
        raise ValueError(f"Requested {n_borrowers:,} rows, only {len(clean):,} exist")

    # Verify saved Phase 2 artifacts first, then recreate original membership.
    verified_test = verify_original_test_split(clean)
    test_id_set = set(verified_test["loan_id"].tolist())
    test_mask = clean["loan_id"].isin(test_id_set)
    original_test = clean.loc[test_mask, ["loan_id", "early_default"]]
    original_train = clean.loc[~test_mask, ["loan_id", "early_default"]]

    def stratified_take(frame: pd.DataFrame, count: int, seed: int) -> pd.DataFrame:
        if count > len(frame):
            raise ValueError(f"Cannot select {count:,} rows from {len(frame):,}")
        if count == len(frame):
            return frame.copy()
        selected, _ = train_test_split(
            frame,
            train_size=count,
            random_state=seed,
            stratify=frame["early_default"],
        )
        return selected

    train_sample = stratified_take(original_train, n_train, random_state)
    test_sample = stratified_take(original_test, n_test, random_state)
    train_sample = train_sample.assign(split="train")
    test_sample = test_sample.assign(split="test")

    result = pd.concat([train_sample, test_sample], ignore_index=True)
    result = result[["loan_id", "split", "early_default"]]
    result = result.sort_values("loan_id", kind="stable").reset_index(drop=True)

    if len(result) != n_borrowers or result["loan_id"].duplicated().any():
        raise AssertionError("Subsample size or uniqueness check failed")
    if result["split"].value_counts().to_dict() != {"train": n_train, "test": n_test}:
        raise AssertionError("Requested train/test allocation was not achieved")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_path, index=False)
    print(f"Saved {len(result):,} IDs to {output_path}")
    print(result.groupby(["split", "early_default"]).size().to_string())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-parquet", type=Path, default=CLEAN_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--n-borrowers", type=int, default=200_000)
    parser.add_argument("--test-fraction", type=float, default=0.30)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
    make_subsample(
        clean_path=args.clean_parquet,
        output_path=args.output,
        n_borrowers=args.n_borrowers,
        test_fraction=args.test_fraction,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()
