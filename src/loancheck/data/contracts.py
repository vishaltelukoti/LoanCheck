"""Pandera contracts for raw LoanCheck application data."""

# Fairness note: Gender is validated for data-quality purposes but excluded from
# get_feature_columns() so loan decisions do not depend on gender.

import pandas as pd
import pandera.pandas as pa

LOAN_APPLICATION_SCHEMA = pa.DataFrameSchema(
    {
        "Loan_ID": pa.Column(pa.String, unique=True),
        "Gender": pa.Column(pa.String, nullable=True, checks=pa.Check.isin(["Male", "Female"])),
        "Married": pa.Column(pa.String, nullable=True, checks=pa.Check.isin(["Yes", "No"])),
        "Dependents": pa.Column(
            pa.String, nullable=True, checks=pa.Check.isin(["0", "1", "2", "3+"])
        ),
        "Education": pa.Column(pa.String, checks=pa.Check.isin(["Graduate", "Not Graduate"])),
        "Self_Employed": pa.Column(pa.String, nullable=True, checks=pa.Check.isin(["Yes", "No"])),
        "ApplicantIncome": pa.Column(pa.Int, checks=pa.Check.greater_than_or_equal_to(0)),
        "CoapplicantIncome": pa.Column(pa.Float, checks=pa.Check.greater_than_or_equal_to(0)),
        "LoanAmount": pa.Column(pa.Float, nullable=True, checks=pa.Check.greater_than(0)),
        "Loan_Amount_Term": pa.Column(pa.Float, nullable=True, checks=pa.Check.greater_than(0)),
        "Credit_History": pa.Column(
            pa.Float, nullable=True, checks=pa.Check.isin([0.0, 1.0])
        ),
        "Property_Area": pa.Column(
            pa.String, checks=pa.Check.isin(["Urban", "Semiurban", "Rural"])
        ),
        "Loan_Status": pa.Column(pa.String, required=False, checks=pa.Check.isin(["Y", "N"])),
    },
    strict=True,
    coerce=True,
)


def get_feature_columns() -> list[str]:
    """Return the current hard-coded, non-leaking feature column list.

    Returns:
        Approved raw feature names, excluding identifiers, target, and Gender.
    """
    return [
        "Married",
        "Dependents",
        "Education",
        "Self_Employed",
        "ApplicantIncome",
        "CoapplicantIncome",
        "LoanAmount",
        "Loan_Amount_Term",
        "Credit_History",
        "Property_Area",
    ]


def validate_schema(df: pd.DataFrame) -> pa.errors.SchemaErrors | None:
    """Validate a raw dataset, returning errors instead of raising them.

    Args:
        df: Raw loan application dataframe to validate.

    Returns:
        Pandera schema errors on validation failure; otherwise ``None``.
    """
    try:
        LOAN_APPLICATION_SCHEMA.validate(df, lazy=True)
    except pa.errors.SchemaErrors as errors:
        return errors
    return None
