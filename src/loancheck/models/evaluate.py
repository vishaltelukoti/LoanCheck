"""Metric calculation for fitted LoanCheck classifiers."""

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline


def evaluate_model(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    """Return binary classification metrics for a fitted pipeline."""
    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]
    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, pos_label="Y", zero_division=0)),
        "recall": float(recall_score(y_test, predictions, pos_label="Y", zero_division=0)),
        "f1": float(f1_score(y_test, predictions, pos_label="Y", zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
    }
