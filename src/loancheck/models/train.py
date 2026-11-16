"""Scikit-learn pipeline training utilities."""

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
) -> Pipeline:
    """Fit and return one pipeline containing preprocessing and classification."""
    categorical_columns = features_df.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()
    numeric_columns = [column for column in features_df.columns if column not in categorical_columns]
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_columns),
            ("numeric", "passthrough", numeric_columns),
        ]
    )
    classifiers = {
        "logistic_regression": LogisticRegression(max_iter=1_000, random_state=random_seed),
        "decision_tree": DecisionTreeClassifier(random_state=random_seed),
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=random_seed),
    }
    pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", classifiers[model_type])])
    pipeline.fit(features_df, target)
    return pipeline
