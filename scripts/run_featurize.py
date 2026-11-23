"""Split raw training data before building features, then persist DVC artifacts."""

import json
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from loancheck.data.missing_values import fit_missing_value_stats
from loancheck.features.build_features import build_features

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTERIM_PATH = PROJECT_ROOT / "data" / "interim" / "loan_train_clean.csv"
TRAIN_FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "train_features.csv"
TEST_FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "test_features.csv"
MISSING_VALUE_STATS_PATH = PROJECT_ROOT / "data" / "processed" / "missing_value_stats.json"
PARAMS_PATH = PROJECT_ROOT / "params.yaml"


def main() -> None:
    """Create training and test feature sets without cross-split imputation leakage."""
    params = yaml.safe_load(PARAMS_PATH.read_text(encoding="utf-8"))
    raw_data = pd.read_csv(INTERIM_PATH)
    target = raw_data["Loan_Status"]
    raw_features = raw_data.drop(columns="Loan_Status")
    # Split raw data first: fitting missing-value behavior on all rows would leak test information.
    x_train_raw, x_test_raw, y_train, y_test = train_test_split(
        raw_features,
        target,
        test_size=params["test_size"],
        random_state=params["random_seed"],
        stratify=target,
    )
    missing_value_stats = fit_missing_value_stats(x_train_raw)
    train_features = build_features(x_train_raw, stats=missing_value_stats)
    test_features = build_features(x_test_raw, stats=missing_value_stats)
    train_features["Loan_Status"] = y_train.to_numpy()
    test_features["Loan_Status"] = y_test.to_numpy()
    TRAIN_FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    train_features.to_csv(TRAIN_FEATURES_PATH, index=False)
    test_features.to_csv(TEST_FEATURES_PATH, index=False)
    MISSING_VALUE_STATS_PATH.write_text(
        json.dumps(missing_value_stats, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
