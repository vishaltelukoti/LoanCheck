from loancheck.data.contracts import get_feature_columns


def test_feature_columns_exclude_identifiers_and_target() -> None:
    feature_columns = get_feature_columns()

    assert "Loan_ID" not in feature_columns
    assert "Loan_Status" not in feature_columns
    assert "Gender" not in feature_columns
