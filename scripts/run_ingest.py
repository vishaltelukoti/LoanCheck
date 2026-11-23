"""Copy the raw labeled dataset into the DVC interim area after a basic contract check."""

from pathlib import Path

import pandas as pd

from loancheck.data.contracts import validate_schema

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "loan_train.csv"
INTERIM_PATH = PROJECT_ROOT / "data" / "interim" / "loan_train_clean.csv"


def main() -> None:
    """Validate the labeled raw data and copy it into the interim area."""
    raw_data = pd.read_csv(RAW_PATH)
    schema_errors = validate_schema(raw_data)
    if schema_errors is not None:
        raise ValueError(f"Raw training data failed schema validation: {schema_errors}")
    INTERIM_PATH.parent.mkdir(parents=True, exist_ok=True)
    raw_data.to_csv(INTERIM_PATH, index=False)


if __name__ == "__main__":
    main()
