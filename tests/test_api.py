"""Tests for the LoanCheck FastAPI serving interface."""

from collections.abc import Generator

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from loancheck.api import main


class FakeChampionModel:
    """Deterministic replacement for the MLflow-loaded classifier in API tests."""

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
        """Return one fixed positive-class probability per requested row."""
        return np.tile(np.array([[0.2, 0.8]]), (len(features), 1))


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Generator[TestClient]:
    """Start the app with a fake champion so tests do not require MLflow infrastructure."""
    monkeypatch.setattr(main.mlflow.pyfunc, "load_model", lambda _: FakeChampionModel())
    monkeypatch.setattr(
        main,
        "_load_missing_value_stats",
        lambda: {
            "Credit_History": 1.0,
            "Self_Employed": "No",
            "Gender": "Male",
            "Loan_Amount_Term": 360.0,
            "LoanAmount": 128.0,
        },
    )
    with TestClient(main.app) as test_client:
        yield test_client


def _valid_payload() -> dict[str, object]:
    return {
        "Married": "Yes",
        "Dependents": "1",
        "Education": "Graduate",
        "Self_Employed": "No",
        "ApplicantIncome": 5000,
        "CoapplicantIncome": 1500.0,
        "LoanAmount": 128.0,
        "Loan_Amount_Term": 360.0,
        "Credit_History": 1.0,
        "Property_Area": "Urban",
    }


def test_health_returns_ok(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_returns_probability_and_approval(client: TestClient) -> None:
    response = client.post("/predict", json=_valid_payload())

    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["approval_probability"] <= 1.0
    assert body["approved"] is True


def test_predict_rejects_missing_required_field(client: TestClient) -> None:
    payload = _valid_payload()
    payload.pop("Education")

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_predict_imputes_omitted_credit_history(client: TestClient) -> None:
    payload = _valid_payload()
    payload.pop("Credit_History")

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    assert 0.0 <= response.json()["approval_probability"] <= 1.0
