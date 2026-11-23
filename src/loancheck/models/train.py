"""Scikit-learn pipeline training utilities."""

from collections.abc import Mapping
from typing import Literal

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

ModelType = Literal["logistic_regression", "decision_tree", "random_forest"]


def train_model(
    features_df: pd.DataFrame,
    target: pd.Series,
    model_type: ModelType,
    random_seed: int,
    hyperparameters: Mapping[str, Mapping[str, int]],
) -> Pipeline:
    """Fit a preprocessing-and-classification pipeline.

    Args:
        features_df: Training feature rows before preprocessing.
        target: Corresponding loan-status labels.
        model_type: Classifier family to fit.
        random_seed: Reproducibility seed sourced from ``params.yaml``.
        hyperparameters: Model-specific values sourced from ``params.yaml``.

    Returns:
        A fitted pipeline that bundles preprocessing and classification.
    """
    categorical_columns = features_df.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()
    numeric_columns = [
        column for column in features_df.columns if column not in categorical_columns
    ]
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_columns),
            ("numeric", "passthrough", numeric_columns),
        ]
    )
    classifiers = {
        "logistic_regression": LogisticRegression(
            max_iter=hyperparameters["logistic_regression"]["max_iter"],
            random_state=random_seed,
        ),
        "decision_tree": DecisionTreeClassifier(random_state=random_seed),
        "random_forest": RandomForestClassifier(
            n_estimators=hyperparameters["random_forest"]["n_estimators"],
            random_state=random_seed,
        ),
    }
    pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", classifiers[model_type])])
    pipeline.fit(features_df, target)
    return pipeline
