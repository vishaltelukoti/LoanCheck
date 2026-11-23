"""Population Stability Index utilities for numeric raw-data drift checks."""

import numpy as np
import pandas as pd


def compute_psi(train: pd.Series, scoring: pd.Series, bins: int = 10) -> float:
    """Compute Population Stability Index for two numeric distributions.

    Args:
        train: Reference numeric values from training data.
        scoring: Numeric values from the scoring population.
        bins: Number of equal-width bins used for the comparison.

    Returns:
        The non-negative PSI value.

    Raises:
        ValueError: If fewer than two bins are requested or a series has no values.
    """
    if bins < 2:
        raise ValueError("bins must be at least 2")

    train_values = train.dropna().to_numpy(dtype=float)
    scoring_values = scoring.dropna().to_numpy(dtype=float)
    if train_values.size == 0 or scoring_values.size == 0:
        raise ValueError("PSI requires at least one non-missing value in each series")

    combined = np.concatenate((train_values, scoring_values))
    lower, upper = float(combined.min()), float(combined.max())
    if lower == upper:
        return 0.0

    edges = np.linspace(lower, upper, bins + 1)
    edges[0] = -np.inf
    edges[-1] = np.inf
    train_counts, _ = np.histogram(train_values, bins=edges)
    scoring_counts, _ = np.histogram(scoring_values, bins=edges)
    epsilon = 1e-6
    train_proportions = np.maximum(train_counts / train_counts.sum(), epsilon)
    scoring_proportions = np.maximum(scoring_counts / scoring_counts.sum(), epsilon)
    return float(
        np.sum(
            (scoring_proportions - train_proportions)
            * np.log(scoring_proportions / train_proportions)
        )
    )


def check_drift(
    train_df: pd.DataFrame, scoring_df: pd.DataFrame, columns: list[str]
) -> dict[str, float]:
    """Compute numeric PSI values for selected columns.

    Args:
        train_df: Reference training dataframe.
        scoring_df: Scoring dataframe to compare with the reference.
        columns: Numeric column names to evaluate.

    Returns:
        A PSI value keyed by column name.
    """
    return {column: compute_psi(train_df[column], scoring_df[column]) for column in columns}


def drift_status(psi: float, warn_threshold: float, fail_threshold: float) -> str:
    """Classify a PSI value using LoanCheck's documented thresholds.

    Args:
        psi: Population Stability Index value to classify.
        warn_threshold: PSI boundary at which status changes from pass to warn.
        fail_threshold: PSI boundary at which status changes from warn to fail.

    Returns:
        ``pass``, ``warn``, or ``fail``.
    """
    if warn_threshold >= fail_threshold:
        msg = "The PSI warning threshold must be lower than the failure threshold."
        raise ValueError(msg)
    if psi < warn_threshold:
        return "pass"
    if psi < fail_threshold:
        return "warn"
    return "fail"
