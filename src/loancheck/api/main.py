"""FastAPI serving interface for the LoanCheck champion model."""

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal, Protocol, cast

import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
from fastapi import FastAPI, Request
from pydantic import BaseModel, Field

from loancheck.config import Settings
from loancheck.features.build_features import build_features

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MISSING_VALUE_STATS_PATH = PROJECT_ROOT / "models" / "missing_value_stats.json"


class ProbabilityModel(Protocol):
    """Minimum probability-prediction interface required from the champion model."""

    def predict_proba(
        self, features: pd.DataFrame
    ) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
        """Return class probabilities for supplied feature rows.

        Args:
            features: Preprocessed model feature rows.

        Returns:
            A two-column array of negative and positive class probabilities.
        """


class LoanApplication(BaseModel):
    """The ten approved, non-sensitive raw features accepted by the API.

    Attributes:
        Married: Applicant marital status, when supplied.
        Dependents: Applicant dependent count, when supplied.
        Education: Applicant education category.
        Self_Employed: Self-employment status, when supplied.
        ApplicantIncome: Applicant monthly income.
        CoapplicantIncome: Co-applicant monthly income.
        LoanAmount: Requested loan amount, when supplied.
        Loan_Amount_Term: Requested loan term, when supplied.
        Credit_History: Credit-history indicator, when supplied.
        Property_Area: Property-area category.
    """

    Married: Literal["Yes", "No"] | None = None
    Dependents: Literal["0", "1", "2", "3+"] | None = None
    Education: Literal["Graduate", "Not Graduate"]
    Self_Employed: Literal["Yes", "No"] | None = None
    ApplicantIncome: Annotated[int, Field(ge=0)]
    CoapplicantIncome: Annotated[float, Field(ge=0)]
    LoanAmount: Annotated[float, Field(gt=0)] | None = None
    Loan_Amount_Term: Annotated[float, Field(gt=0)] | None = None
    Credit_History: Literal[0, 1] | None = None
    Property_Area: Literal["Urban", "Semiurban", "Rural"]


class PredictionResponse(BaseModel):
    """A binary approval decision and positive-class probability.

    Attributes:
        approval_probability: Champion model probability for approval.
        approved: Whether the probability meets the configured decision threshold.
    """

    approval_probability: float
    approved: bool


def _as_probability_model(loaded_model: object) -> ProbabilityModel:
    """Extract sklearn's probability interface from an MLflow pyfunc model."""
    if callable(getattr(loaded_model, "predict_proba", None)):
        return cast(ProbabilityModel, loaded_model)

    model_implementation = getattr(loaded_model, "_model_impl", None)
    sklearn_model = getattr(model_implementation, "sklearn_model", None)
    if callable(getattr(sklearn_model, "predict_proba", None)):
        return cast(ProbabilityModel, sklearn_model)

    msg = "Champion model does not expose predict_proba through its MLflow pyfunc wrapper."
    raise RuntimeError(msg)


def _load_champion_model() -> ProbabilityModel:
    """Load and validate the model-registry champion once during application startup."""
    try:
        loaded_model = mlflow.pyfunc.load_model("models:/loancheck@champion")
        return _as_probability_model(loaded_model)
    except Exception as error:
        msg = "Unable to load MLflow champion model 'loancheck@champion'."
        raise RuntimeError(msg) from error


def _load_missing_value_stats() -> dict[str, object]:
    """Load the imputation values fitted on the training split."""
    try:
        with MISSING_VALUE_STATS_PATH.open(encoding="utf-8") as stats_file:
            stats = json.load(stats_file)
        if not isinstance(stats, dict) or not all(isinstance(key, str) for key in stats):
            msg = "Missing-value statistics must be a JSON object with string keys."
            raise ValueError(msg)
        return cast(dict[str, object], stats)
    except Exception as error:
        msg = f"Unable to load missing-value statistics from {MISSING_VALUE_STATS_PATH}."
        raise RuntimeError(msg) from error


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    """Configure tracking and load the champion before accepting requests."""
    settings = Settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    application.state.settings = settings
    application.state.champion_model = _load_champion_model()
    application.state.missing_value_stats = _load_missing_value_stats()
    yield


app = FastAPI(title="LoanCheck API", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    """Return a simple liveness result.

    Returns:
        An ``ok`` service-status payload.
    """
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(application: LoanApplication, request: Request) -> PredictionResponse:
    """Return the champion approval probability for one loan application.

    Args:
        application: Validated raw application fields from the request body.
        request: FastAPI request that carries startup-loaded application state.

    Returns:
        Probability and thresholded approval decision from the champion model.
    """
    raw_application = application.model_dump()
    # Gender is deliberately excluded from the public contract and model features.
    # Its persisted training-mode statistic fills this internal preprocessing column.
    raw_application["Gender"] = None
    missing_value_stats = cast(dict[str, object], request.app.state.missing_value_stats)
    features = build_features(pd.DataFrame([raw_application]), stats=missing_value_stats)
    champion_model = cast(ProbabilityModel, request.app.state.champion_model)
    settings = cast(Settings, request.app.state.settings)
    probability = float(champion_model.predict_proba(features)[0, 1])
    return PredictionResponse(
        approval_probability=probability,
        approved=probability >= settings.approval_threshold,
    )
