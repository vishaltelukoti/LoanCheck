"""Start the local MLflow tracking server for LoanCheck development."""

import os
from pathlib import Path
import subprocess
import sys


def main() -> None:
    """Run MLflow with local SQLite metadata and filesystem artifacts."""
    # Keep MLflow's SQLite database and local artifact paths rooted in the repository.
    project_root = Path(__file__).resolve().parents[1]
    os.chdir(project_root)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "mlflow",
            "server",
            "--backend-store-uri",
            "sqlite:///mlflow.db",
            "--host",
            "0.0.0.0",
            "--port",
            "5000",
            # MLflow multi-worker subprocess startup is unreliable on Windows.
            "--workers",
            "1",
            "--serve-artifacts",
            "--allowed-hosts",
            # MLflow matches Host headers as host:port unless the port wildcard is used.
            "localhost:*,127.0.0.1:*,host.docker.internal:*",
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
