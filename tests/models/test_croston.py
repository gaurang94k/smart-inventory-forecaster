"""Tests for src/models/croston.py — CrostonForecaster.

Uses a 200-day synthetic intermittent series (~70% zeros) to match the
profile of HOUSEHOLD_1_430_CA_1, the intermittent SKU in the shortlist.
Module-scoped fixture trains the model once to keep the suite fast.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.base import NotFittedError
from src.models.croston import CrostonForecaster

# ---------------------------------------------------------------------------
# Synthetic intermittent data helper
# ---------------------------------------------------------------------------

_N_DAYS = 200
_HORIZON = 14


def _make_intermittent_history(seed: int = 0) -> pd.DataFrame:
    """200-day series with ~70% zero rate, matching the intermittent SKU profile."""
    rng = np.random.default_rng(seed=seed)
    mask = rng.random(_N_DAYS) > 0.70
    sales = np.where(mask, rng.integers(1, 6, size=_N_DAYS).astype(float), 0.0)
    dates = pd.date_range("2021-01-01", periods=_N_DAYS, freq="D")
    return pd.DataFrame({"date": dates, "sales": sales})


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def history() -> pd.DataFrame:
    return _make_intermittent_history(seed=0)


@pytest.fixture(scope="module")
def fitted_croston(history) -> CrostonForecaster:
    return CrostonForecaster().fit(history)


@pytest.fixture(scope="module")
def croston_forecast(fitted_croston) -> pd.DataFrame:
    return fitted_croston.forecast(_HORIZON)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_name():
    assert CrostonForecaster().name() == "Croston"


def test_not_fitted_raises():
    with pytest.raises(NotFittedError):
        CrostonForecaster().forecast(7)


def test_fit_returns_self(history):
    m = CrostonForecaster()
    assert m.fit(history) is m


def test_forecast_columns(croston_forecast):
    assert "date" in croston_forecast.columns
    assert "forecast" in croston_forecast.columns


def test_forecast_row_count(croston_forecast):
    assert len(croston_forecast) == _HORIZON


def test_forecast_non_negative(croston_forecast):
    assert (croston_forecast["forecast"] >= 0.0).all()


def test_forecast_dates_after_training(fitted_croston, history):
    last_train = history["date"].max()
    fc = fitted_croston.forecast(_HORIZON)
    assert (fc["date"] > last_train).all()
    assert fc["date"].iloc[0] == last_train + pd.Timedelta(days=1)
