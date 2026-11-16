from pathlib import Path

import pandas as pd
import pytest

RAW_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
TRAIN_PATH = RAW_DATA_DIR / "loan_train.csv"
SCORING_PATH = RAW_DATA_DIR / "loan_scoring.csv"


def test_raw_data_schema_contract() -> None:
    if not TRAIN_PATH.is_file() or not SCORING_PATH.is_file():
        pytest.skip("Raw data is not available locally; it will be managed by DVC in Task 5.")

    train = pd.read_csv(TRAIN_PATH)
    scoring = pd.read_csv(SCORING_PATH)

    assert "Loan_Status" not in scoring.columns
    assert set(train.columns) - {"Loan_Status"} == set(scoring.columns)
