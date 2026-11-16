"""Version identifiers that connect MLflow runs to code and DVC data."""

import subprocess
import warnings
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def get_git_commit() -> str:
    """Return the current Git commit, appending ``-dirty`` for local changes."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        is_dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=PROJECT_ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError) as error:
        msg = "Unable to determine Git commit: this directory is not a usable Git repository."
        raise RuntimeError(msg) from error

    if is_dirty:
        warnings.warn("Training from a working tree with uncommitted changes.", stacklevel=2)
        return f"{commit}-dirty"
    return commit


def get_dvc_data_version(stage: str = "train") -> str:
    """Return the MD5 of the first dependency tracked for a DVC stage."""
    lock_path = PROJECT_ROOT / "dvc.lock"
    if not lock_path.is_file():
        raise FileNotFoundError(f"DVC lockfile does not exist: {lock_path}")
    lock_data = yaml.safe_load(lock_path.read_text(encoding="utf-8"))
    try:
        stage_data = lock_data["stages"][stage]
        data_version = stage_data["deps"][0]["md5"]
    except (KeyError, IndexError, TypeError) as error:
        msg = f"No dependency MD5 found for DVC stage '{stage}'."
        raise ValueError(msg) from error
    if not isinstance(data_version, str):
        raise TypeError(f"DVC stage '{stage}' has a non-string dependency MD5.")
    return data_version
