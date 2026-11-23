"""Runtime configuration loaded from the selected environment YAML file."""

import os
from pathlib import Path

from pydantic_settings import BaseSettings


def _read_yaml_settings(path: Path) -> dict[str, str]:
    """Read the simple key/value YAML used for LoanCheck environment settings.

    Args:
        path: YAML configuration file to parse.

    Returns:
        String-valued settings keyed by configuration field name.

    Raises:
        ValueError: If a non-comment line is not a key/value setting.
    """
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped_line = line.strip()
        if not stripped_line or stripped_line.startswith("#"):
            continue
        key, separator, value = stripped_line.partition(":")
        if not separator:
            msg = f"Invalid setting in {path}: {line}"
            raise ValueError(msg)
        values[key.strip()] = value.strip()
    return values


class Settings(BaseSettings):
    """Settings for the environment selected by ``ENVIRONMENT`` (default: ``dev``).

    Attributes:
        environment: Selected deployment environment.
        log_level: Application logging level.
        mlflow_tracking_uri: MLflow tracking-server location.
        model_quality_bar_accuracy: Minimum accuracy required for promotion.
        model_quality_bar_roc_auc: Minimum ROC-AUC required for promotion.
        data_dir: Raw data directory configured for the environment.
    """

    environment: str
    log_level: str
    mlflow_tracking_uri: str
    model_quality_bar_accuracy: float
    model_quality_bar_roc_auc: float
    data_dir: str

    def __init__(self, environment: str | None = None) -> None:
        """Load settings from the selected YAML file and environment overrides.

        Args:
            environment: Optional environment name; otherwise uses ``ENVIRONMENT``.

        Raises:
            FileNotFoundError: If the selected configuration file does not exist.
            ValueError: If the configuration file has invalid setting syntax.
        """
        environment_name = environment or os.environ.get("ENVIRONMENT", "dev")
        project_root = Path(__file__).resolve().parents[2]
        config_path = project_root / "configs" / f"{environment_name}.yaml"
        if not config_path.is_file():
            msg = f"Configuration file does not exist: {config_path}"
            raise FileNotFoundError(msg)

        settings = _read_yaml_settings(config_path)
        settings["environment"] = environment_name
        if tracking_uri := os.environ.get("MLFLOW_TRACKING_URI"):
            settings["mlflow_tracking_uri"] = tracking_uri
        super().__init__(**settings)  # type: ignore[arg-type]
