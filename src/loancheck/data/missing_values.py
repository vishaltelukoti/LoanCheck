"""Missing-value checks and deterministic imputation for raw loan data."""

# LoanAmount uses the median because loan amounts are skewed by outliers; the
# categorical and binary fields use their mode to retain their most common value.

from numbers import Real
from typing import cast

import pandas as pd

MAX_MISSING_PERCENT = 10.0
MODE_COLUMNS = ("Credit_History", "Self_Employed", "Gender", "Loan_Amount_Term")
MEDIAN_COLUMN = "LoanAmount"


def report_missing(df: pd.DataFrame) -> dict[str, float]:
    """Return the percentage of missing entries for every column."""
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
    missing = report_missing(df)
    over_limit = {column: percent for column, percent in missing.items() if percent > MAX_MISSING_PERCENT}
    if over_limit:
        msg = f"Missingness exceeds {MAX_MISSING_PERCENT}%: {over_limit}"
        raise ValueError(msg)


def fit_missing_value_stats(df: pd.DataFrame) -> dict[str, object]:
    """Fit imputation values from one dataset after validating raw missingness."""
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
    """Return a copy imputed with previously fitted values, including one-row inputs."""
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
    """Fit then apply imputation values for callers that do not need split control."""
    return apply_missing_value_fill(df, fit_missing_value_stats(df))
