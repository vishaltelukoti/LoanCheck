import pandas as pd

from loancheck.models.train import train_model


def test_train_model_returns_fitted_pipeline_with_predict() -> None:
    features = pd.DataFrame(
        {
            "Married": ["Yes", "No", "Yes", "No"],
            "ApplicantIncome": [2_000, 3_000, 5_000, 7_000],
            "LoanAmount": [80.0, 90.0, 120.0, 150.0],
        }
    )
    target = pd.Series(["N", "N", "Y", "Y"])

    pipeline = train_model(
        features,
        target,
        "logistic_regression",
        random_seed=42,
        hyperparameters={
            "logistic_regression": {"max_iter": 100},
            "random_forest": {"n_estimators": 10},
        },
    )

    assert hasattr(pipeline, "predict")
    assert len(pipeline.predict(features)) == len(features)
