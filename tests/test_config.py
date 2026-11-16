from __future__ import annotations

import pytest

from loancheck.config import Settings


def test_loads_development_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENVIRONMENT", "dev")
    settings = Settings()

    assert settings.log_level == "DEBUG"
    assert settings.model_quality_bar_accuracy == 0.70
    assert settings.mlflow_tracking_uri == "http://localhost:5000"
