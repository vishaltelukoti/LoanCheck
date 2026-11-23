"""Feature construction for loan-approval model inputs."""

import numpy as np
import pandas as pd

from loancheck.data.contracts import get_feature_columns
from loancheck.data.missing_values import apply_missing_value_fill, fit_missing_value_stats

ENGINEERED_FEATURE_COLUMNS = ["TotalIncome", "LoanAmountToIncomeRatio"]


def build_features(df: pd.DataFrame, stats: dict[str, object] | None = None) -> pd.DataFrame:
    """Build approved raw and engineered model features.

    Args:
        df: Raw loan application dataframe.
        stats: Optional training-derived imputation statistics. When omitted, they
            are fitted from ``df`` for backward-compatible local use.

    Returns:
        Approved non-sensitive raw features plus engineered income features.

    Raises:
        ValueError: If required imputation statistics cannot be fitted or applied.
    """
    fitted_stats = fit_missing_value_stats(df) if stats is None else stats
    filled = apply_missing_value_fill(df, fitted_stats)
    features = filled.loc[:, get_feature_columns()].copy()
    total_income = filled["ApplicantIncome"] + filled["CoapplicantIncome"]
    features["TotalIncome"] = total_income
    features["LoanAmountToIncomeRatio"] = np.where(
        total_income != 0,
        filled["LoanAmount"] / total_income,
        0.0,
    )
    return features.loc[:, get_feature_columns() + ENGINEERED_FEATURE_COLUMNS]
