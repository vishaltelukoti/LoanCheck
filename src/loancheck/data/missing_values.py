"""Missing-value checks and deterministic imputation for raw loan data."""

# LoanAmount uses the median because loan amounts are skewed by outliers; the
# categorical and binary fields use their mode to retain their most common value.

from numbers import Real
from pathlib import Path
from typing import cast

import pandas as pd
import yaml

PARAMS_PATH = Path(__file__).resolve().parents[3] / "params.yaml"
MODE_COLUMNS = ("Credit_History", "Self_Employed", "Gender", "Loan_Amount_Term")
MEDIAN_COLUMN = "LoanAmount"


def report_missing(df: pd.DataFrame) -> dict[str, float]:
    """Calculate missing-value percentages for every dataframe column.

    Args:
        df: Dataframe whose missingness will be measured.

    Returns:
        Percentage missing for each column.
    """
    return {str(column): float(value) for column, value in (df.isna().mean() * 100).items()}


def _mode_value(series: pd.Series) -> object:
    """Return the first mode, failing clearly when no mode can be computed."""
    modes = series.mode(dropna=True)
    if modes.empty:
        msg = f"Cannot impute {series.name}: it has no non-missing values"
        raise ValueError(msg)
    return cast(object, modes.iloc[0])


def _validate_missingness(df: pd.DataFrame) -> None:
    """Raise when raw data exceeds the permitted missing-value threshold."""
    params = yaml.safe_load(PARAMS_PATH.read_text(encoding="utf-8"))
    missingness_limit = float(params["missingness_limit_pct"])
    missing = report_missing(df)
    over_limit = {
        column: percent for column, percent in missing.items() if percent > missingness_limit
    }
    if over_limit:
        msg = f"Missingness exceeds {missingness_limit}%: {over_limit}"
        raise ValueError(msg)


def fit_missing_value_stats(df: pd.DataFrame) -> dict[str, object]:
    """Fit imputation values from one dataset after validating missingness.

    Args:
        df: Training dataframe used to derive imputation statistics.

    Returns:
        Mode statistics for categorical fields and a median for ``LoanAmount``.

    Raises:
        ValueError: If missingness exceeds the permitted threshold or no statistic exists.
    """
    _validate_missingness(df)
    stats = {column: _mode_value(df[column]) for column in MODE_COLUMNS}
    loan_amount_median = df[MEDIAN_COLUMN].median()
    if pd.isna(loan_amount_median):
        msg = f"Cannot impute {MEDIAN_COLUMN}: it has no non-missing values"
        raise ValueError(msg)
    stats[MEDIAN_COLUMN] = float(loan_amount_median)
    return stats


def _get_stat(stats: dict[str, object], column: str) -> str | float:
    """Return a pandas-compatible fitted value for one required column."""
    value = stats[column]
    if isinstance(value, str):
        return value
    if isinstance(value, Real):
        return float(value)
    msg = f"Invalid imputation stat for {column}: {value!r}"
    raise TypeError(msg)


def apply_missing_value_fill(df: pd.DataFrame, stats: dict[str, object]) -> pd.DataFrame:
    """Apply previously fitted imputation values to a dataframe copy.

    Args:
        df: Dataframe to impute, including possible one-row API inputs.
        stats: Training-derived imputation statistics for required raw columns.

    Returns:
        A copied dataframe with missing supported fields filled.

    Raises:
        ValueError: If a required statistic is absent.
        TypeError: If a supplied statistic has an unsupported type.
    """
    expected_columns = (*MODE_COLUMNS, MEDIAN_COLUMN)
    missing_stats = [column for column in expected_columns if column not in stats]
    if missing_stats:
        msg = f"Missing imputation stats for columns: {missing_stats}"
        raise ValueError(msg)

    filled = df.copy()
    for column in MODE_COLUMNS:
        filled[column] = filled[column].fillna(_get_stat(stats, column))
    filled[MEDIAN_COLUMN] = filled[MEDIAN_COLUMN].fillna(_get_stat(stats, MEDIAN_COLUMN))
    return filled


def fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Fit and apply imputation values for callers without split control.

    Args:
        df: Dataframe from which to fit and apply missing-value statistics.

    Returns:
        A copy with supported missing values filled.

    Raises:
        ValueError: If missingness exceeds the permitted threshold.
    """
    return apply_missing_value_fill(df, fit_missing_value_stats(df))
