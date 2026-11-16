"""Feature construction for loan-approval model inputs."""

import numpy as np
import pandas as pd

from loancheck.data.contracts import get_feature_columns
from loancheck.data.missing_values import fill_missing

ENGINEERED_FEATURE_COLUMNS = ["TotalIncome", "LoanAmountToIncomeRatio"]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Fill expected gaps and return the approved raw and engineered features."""
    filled = fill_missing(df)
    features = filled.loc[:, get_feature_columns()].copy()
    total_income = filled["ApplicantIncome"] + filled["CoapplicantIncome"]
    features["TotalIncome"] = total_income
    features["LoanAmountToIncomeRatio"] = np.where(
        total_income != 0,
        filled["LoanAmount"] / total_income,
        0.0,
    )
    return features.loc[:, get_feature_columns() + ENGINEERED_FEATURE_COLUMNS]
