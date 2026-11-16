import numpy as np
import pandas as pd
import pytest

from loancheck.data.drift import compute_psi


def test_compute_psi_is_zero_for_identical_distributions() -> None:
    values = pd.Series(np.arange(100))

    assert compute_psi(values, values) == pytest.approx(0.0)


def test_compute_psi_is_large_for_different_distributions() -> None:
    train = pd.Series(np.zeros(100))
    scoring = pd.Series(np.full(100, 100.0))

    assert compute_psi(train, scoring) > 1.0
