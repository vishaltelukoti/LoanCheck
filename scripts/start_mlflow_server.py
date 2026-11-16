"""Start the local MLflow tracking server for LoanCheck development."""

import subprocess
import sys


def main() -> None:
    """Run MLflow with local SQLite metadata and filesystem artifacts."""
    subprocess.run(
        [
            sys.executable,
            "-m",
            "mlflow",
            "server",
            "--backend-store-uri",
            "sqlite:///mlflow.db",
            "--default-artifact-root",
            "./mlruns",
            "--host",
            "0.0.0.0",
            "--port",
            "5000",
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
