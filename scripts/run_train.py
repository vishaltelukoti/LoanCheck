"""Train and compare LoanCheck classifiers with MLflow experiment tracking."""

import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from joblib import dump
from mlflow.exceptions import MlflowException
from mlflow.models import infer_signature
from sklearn.metrics import ConfusionMatrixDisplay

from loancheck.config import Settings
from loancheck.models.evaluate import evaluate_model
from loancheck.models.lineage import get_dvc_data_version, get_git_commit
from loancheck.models.train import train_model

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "train_features.csv"
TEST_FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "test_features.csv"
MISSING_VALUE_STATS_INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "missing_value_stats.json"
MISSING_VALUE_STATS_OUTPUT_PATH = PROJECT_ROOT / "models" / "missing_value_stats.json"
MODEL_OUTPUT_PATH = PROJECT_ROOT / "models" / "model.pkl"
PARAMS_PATH = PROJECT_ROOT / "params.yaml"
EXPERIMENT_NAME = "loancheck"
MODEL_TYPES = ("logistic_regression", "decision_tree", "random_forest")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


@dataclass(frozen=True)
class ModelResult:
    """Metrics and MLflow identity for one model-comparison run."""

    model_type: str
    run_id: str
    metrics: dict[str, float]


def _log_diagnostic_artifacts(
    pipeline: object,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_type: str,
) -> None:
    """Create and log confusion-matrix and feature-importance visualizations."""
    classifier = pipeline.named_steps["classifier"]
    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    if model_type == "logistic_regression":
        importance = abs(classifier.coef_[0])
    else:
        importance = classifier.feature_importances_

    with tempfile.TemporaryDirectory() as directory:
        artifact_dir = Path(directory)
        confusion_path = artifact_dir / "confusion_matrix.png"
        importance_path = artifact_dir / "feature_importance.png"

        figure, axis = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay.from_predictions(y_test, pipeline.predict(X_test), ax=axis)
        figure.tight_layout()
        figure.savefig(confusion_path, dpi=150)
        plt.close(figure)

        top_indices = importance.argsort()[-20:]
        figure, axis = plt.subplots(figsize=(9, 7))
        axis.barh(feature_names[top_indices], importance[top_indices])
        axis.set_title(f"{model_type} feature importance")
        figure.tight_layout()
        figure.savefig(importance_path, dpi=150)
        plt.close(figure)

        mlflow.log_artifact(str(confusion_path), artifact_path="diagnostics")
        mlflow.log_artifact(str(importance_path), artifact_path="diagnostics")


def _promote_challenger(results: list[ModelResult], settings: Settings) -> tuple[str | None, bool]:
    """Register the best qualifying run and promote it when it beats champion."""
    qualifying = [
        result
        for result in results
        if result.metrics["accuracy"] >= settings.model_quality_bar_accuracy
        and result.metrics["roc_auc"] >= settings.model_quality_bar_roc_auc
    ]
    if not qualifying:
        print("No model met the configured accuracy and ROC-AUC quality bars.")
        return None, False

    challenger = max(qualifying, key=lambda result: result.metrics["accuracy"])
    registered_model = mlflow.register_model(f"runs:/{challenger.run_id}/model", "loancheck")
    client = mlflow.tracking.MlflowClient()
    client.set_registered_model_alias("loancheck", "challenger", registered_model.version)

    try:
        champion = client.get_model_version_by_alias("loancheck", "champion")
    except MlflowException:
        client.set_registered_model_alias("loancheck", "champion", registered_model.version)
        print("No champion existed; promoted the challenger to champion.")
        return challenger.model_type, True

    champion_accuracy = client.get_run(champion.run_id).data.metrics.get("accuracy", 0.0)
    if challenger.metrics["accuracy"] > champion_accuracy:
        client.set_registered_model_alias("loancheck", "champion", registered_model.version)
        print("Challenger beat champion accuracy and was promoted.")
        return challenger.model_type, True

    print(
        "Challenger was not promoted: its accuracy "
        f"({challenger.metrics['accuracy']:.4f}) does not exceed champion accuracy "
        f"({champion_accuracy:.4f})."
    )
    return challenger.model_type, False


def _print_summary(results: list[ModelResult], challenger: str | None, champion_changed: bool) -> None:
    """Print a compact model-comparison summary for local and CI logs."""
    print("\nModel comparison")
    print("model_type           accuracy  precision  recall    f1        roc_auc")
    for result in results:
        metrics = result.metrics
        print(
            f"{result.model_type:<20} {metrics['accuracy']:.4f}    "
            f"{metrics['precision']:.4f}     {metrics['recall']:.4f}    "
            f"{metrics['f1']:.4f}    {metrics['roc_auc']:.4f}"
        )
    print(f"challenger: {challenger or 'none'}")
    print(f"champion changed: {champion_changed}")


def main() -> None:
    """Train, evaluate, log, register, and compare the three baseline model types."""
    params = yaml.safe_load(PARAMS_PATH.read_text(encoding="utf-8"))
    settings = Settings()
    os.environ["MLFLOW_TRACKING_URI"] = settings.mlflow_tracking_uri
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
    training_data = pd.read_csv(TRAIN_FEATURES_PATH)
    test_data = pd.read_csv(TEST_FEATURES_PATH)
    if not MISSING_VALUE_STATS_INPUT_PATH.exists():
        msg = f"Missing fitted imputation statistics: {MISSING_VALUE_STATS_INPUT_PATH}"
        raise FileNotFoundError(msg)
    MISSING_VALUE_STATS_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    MISSING_VALUE_STATS_OUTPUT_PATH.write_text(
        MISSING_VALUE_STATS_INPUT_PATH.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    X_train = training_data.drop(columns="Loan_Status")
    y_train = training_data["Loan_Status"]
    X_test = test_data.drop(columns="Loan_Status")
    y_test = test_data["Loan_Status"]
    dataset = mlflow.data.from_pandas(training_data, source=str(TRAIN_FEATURES_PATH))
    results: list[ModelResult] = []
    pipelines: dict[str, object] = {}

    for model_type in MODEL_TYPES:
        with mlflow.start_run(run_name=model_type) as run:
            mlflow.log_params(
                {
                    "model_type": model_type,
                    "random_seed": params["random_seed"],
                    "test_size": params["test_size"],
                }
            )
            mlflow.set_tags(
                {
                    "git_commit": get_git_commit(),
                    "dvc_data_version": get_dvc_data_version(),
                }
            )
            mlflow.log_input(dataset, context="training")
            pipeline = train_model(X_train, y_train, model_type, params["random_seed"])
            metrics = evaluate_model(pipeline, X_test, y_test)
            mlflow.log_metrics(metrics)
            _log_diagnostic_artifacts(pipeline, X_test, y_test, model_type)
            signature = infer_signature(X_train, pipeline.predict(X_train))
            mlflow.sklearn.log_model(
                pipeline,
                name="model",
                signature=signature,
                input_example=X_train.head(1),
                # Tree estimators include trusted scikit-learn C-extension state that
                # MLflow's default skops serializer refuses to load in MLflow 3.x.
                serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_PICKLE,
            )
            pipelines[model_type] = pipeline
            results.append(ModelResult(model_type, run.info.run_id, metrics))

    challenger, champion_changed = _promote_challenger(results, settings)
    if challenger is None:
        msg = "No model qualified for persistence as the DVC training output."
        raise RuntimeError(msg)
    dump(pipelines[challenger], MODEL_OUTPUT_PATH)
    _print_summary(results, challenger, champion_changed)


if __name__ == "__main__":
    main()
