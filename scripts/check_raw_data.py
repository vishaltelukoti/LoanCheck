"""Validate the fixed raw LoanCheck dataset files."""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


def main() -> None:
    """Print raw-data diagnostics and validate the labeled/scoring schema contract."""
    train = pd.read_csv(RAW_DATA_DIR / "loan_train.csv")
    scoring = pd.read_csv(RAW_DATA_DIR / "loan_scoring.csv")

    print(f"loan_train.csv: {len(train)} rows")
    print(f"columns: {train.columns.tolist()}")
    print("Loan_Status ratio:")
    print(train["Loan_Status"].value_counts(normalize=True).sort_index())
    print(f"loan_scoring.csv: {len(scoring)} rows")
    print(f"columns: {scoring.columns.tolist()}")

    assert "Loan_Status" not in scoring.columns, "loan_scoring.csv must not contain Loan_Status"
    assert set(train.columns) - {"Loan_Status"} == set(scoring.columns), (
        "loan_train.csv and loan_scoring.csv must have matching non-target columns"
    )


if __name__ == "__main__":
    main()
