import pandas as pd
import pytest

from loancheck.data.missing_values import (
    apply_missing_value_fill,
    fill_missing,
    fit_missing_value_stats,
)


def test_fill_missing_rejects_column_above_missingness_limit() -> None:
    data = pd.DataFrame(
        {
            "Credit_History": [1.0] * 8 + [None] * 2,
            "Self_Employed": ["No"] * 10,
            "Gender": ["Male"] * 10,
            "Loan_Amount_Term": [360.0] * 10,
            "LoanAmount": [100.0] * 10,
        }
    )

    with pytest.raises(ValueError, match="Missingness exceeds"):
        fill_missing(data)


def test_fit_missing_value_stats_returns_training_values() -> None:
    data = pd.DataFrame(
        {
            "Credit_History": [1.0] * 9 + [None],
            "Self_Employed": ["No"] * 9 + [None],
            "Gender": ["Female"] * 9 + [None],
            "Loan_Amount_Term": [360.0] * 9 + [None],
            "LoanAmount": [100.0] * 9 + [None],
        }
    )

    stats = fit_missing_value_stats(data)

    assert stats == {
        "Credit_History": 1.0,
        "Self_Employed": "No",
        "Gender": "Female",
        "Loan_Amount_Term": 360.0,
        "LoanAmount": 100.0,
    }


def test_apply_missing_value_fill_uses_stats_for_single_row() -> None:
    single_row = pd.DataFrame(
        {
            "Credit_History": [None],
            "Self_Employed": [None],
            "Gender": [None],
            "Loan_Amount_Term": [None],
            "LoanAmount": [None],
        }
    )
    stats = {
        "Credit_History": 1.0,
        "Self_Employed": "No",
        "Gender": "Female",
        "Loan_Amount_Term": 360.0,
        "LoanAmount": 125.0,
    }

    filled = apply_missing_value_fill(single_row, stats)

    assert filled.loc[0].to_dict() == stats
