"""Integration tests for the repository's actual Git and DVC state."""

import os

import pytest

from loancheck.models.lineage import get_dvc_data_version, get_git_commit

pytestmark = pytest.mark.skipif(
    os.environ.get("CI") == "true",
    reason="CI does not yet check out DVC pipeline state; enable with the pipeline CI job.",
)


def test_get_git_commit_returns_value() -> None:
    assert get_git_commit()


def test_get_dvc_data_version_returns_train_hash() -> None:
    data_version = get_dvc_data_version("train")

    assert len(data_version) == 32
    assert all(character in "0123456789abcdef" for character in data_version)
