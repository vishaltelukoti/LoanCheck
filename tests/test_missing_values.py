import pandas as pd
import pytest

from loancheck.data.missing_values import fill_missing


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
