import pandas as pd
import pytest

from loancheck.features.build_features import build_features


def test_build_features_calculates_income_features() -> None:
    raw_data = pd.DataFrame(
        {
            "Loan_ID": ["A1"],
            "Gender": ["Female"],
            "Married": ["Yes"],
            "Dependents": ["0"],
            "Education": ["Graduate"],
            "Self_Employed": ["No"],
            "ApplicantIncome": [4_000],
            "CoapplicantIncome": [1_000.0],
            "LoanAmount": [100.0],
            "Loan_Amount_Term": [360.0],
            "Credit_History": [1.0],
            "Property_Area": ["Urban"],
            "Loan_Status": ["Y"],
        }
    )

    features = build_features(raw_data)

    assert features.loc[0, "TotalIncome"] == 5_000.0
    assert features.loc[0, "LoanAmountToIncomeRatio"] == pytest.approx(0.02)
    assert "Loan_ID" not in features.columns
    assert "Loan_Status" not in features.columns
    assert "Gender" not in features.columns


def test_build_features_uses_supplied_training_statistics() -> None:
    raw_data = pd.DataFrame(
        {
            "Loan_ID": ["A2"],
            "Gender": ["Female"],
            "Married": ["Yes"],
            "Dependents": ["0"],
            "Education": ["Graduate"],
            "Self_Employed": ["No"],
            "ApplicantIncome": [4_000],
            "CoapplicantIncome": [1_000.0],
            "LoanAmount": [None],
            "Loan_Amount_Term": [360.0],
            "Credit_History": [1.0],
            "Property_Area": ["Urban"],
        }
    )
    training_stats = {
        "Credit_History": 1.0,
        "Self_Employed": "No",
        "Gender": "Female",
        "Loan_Amount_Term": 360.0,
        "LoanAmount": 125.0,
    }

    features = build_features(raw_data, stats=training_stats)

    assert features.loc[0, "LoanAmountToIncomeRatio"] == pytest.approx(0.025)
