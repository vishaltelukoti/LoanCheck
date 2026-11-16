"""Run LoanCheck raw-data quality checks and write a JSON report."""

import json
from pathlib import Path

import pandas as pd

from loancheck.data.contracts import validate_schema
from loancheck.data.drift import check_drift, drift_status
from loancheck.data.missing_values import fill_missing, report_missing

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
REPORT_PATH = PROJECT_ROOT / "reports" / "data_quality.json"
NUMERIC_DRIFT_COLUMNS = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount"]


def main() -> None:
    """Create a schema, missingness, and drift report for the raw datasets."""
    train = pd.read_csv(RAW_DATA_DIR / "loan_train.csv")
    scoring = pd.read_csv(RAW_DATA_DIR / "loan_scoring.csv")
    train_schema_errors = validate_schema(train)
    scoring_schema_errors = validate_schema(scoring)

    missing_report = {"train": report_missing(train), "scoring": report_missing(scoring)}
    filled_train = fill_missing(train)
    filled_scoring = fill_missing(scoring)
    drift_report = check_drift(filled_train, filled_scoring, NUMERIC_DRIFT_COLUMNS)
    drift_states = {column: drift_status(psi) for column, psi in drift_report.items()}

    schema_failed = train_schema_errors is not None or scoring_schema_errors is not None
    if schema_failed or "fail" in drift_states.values():
        overall_status = "fail"
    elif "warn" in drift_states.values():
        overall_status = "warn"
    else:
        overall_status = "pass"

    report = {
        "schema_errors": {
            "train": str(train_schema_errors) if train_schema_errors else None,
            "scoring": str(scoring_schema_errors) if scoring_schema_errors else None,
        },
        "missing_report": missing_report,
        "drift_report": drift_report,
        "overall_status": overall_status,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(REPORT_PATH.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
