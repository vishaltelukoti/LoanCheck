"""Evaluate the saved baseline pipeline against the held-out feature artifact."""

import json
from pathlib import Path

import joblib
import pandas as pd

from loancheck.models.evaluate import evaluate_model

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "test_features.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "model.pkl"
METRICS_PATH = PROJECT_ROOT / "reports" / "metrics.json"


def main() -> None:
    """Write held-out evaluation metrics to the reports directory."""
    test_data = pd.read_csv(TEST_FEATURES_PATH)
    pipeline = joblib.load(MODEL_PATH)
    metrics = evaluate_model(
        pipeline,
        test_data.drop(columns="Loan_Status"),
        test_data["Loan_Status"],
    )
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
