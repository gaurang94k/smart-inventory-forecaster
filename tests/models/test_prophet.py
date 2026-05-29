"""Tests for src/models/prophet.py.

Speed note: Prophet fitting is slow. All tests that need a fitted model share
a single module-scoped fixture that fits once on a 100-day synthetic series
using a minimal config (no yearly seasonality, no holidays). Full accuracy
testing is left to the rolling-origin CV harness in src/evaluation/cv.py.
"""

import numpy as np
import pandas as pd
import pytest

from src.models.base import NotFittedError
from src.models.prophet import ProphetForecaster

# ---------------------------------------------------------------------------
# Synthetic data and fitted-model fixture
# ---------------------------------------------------------------------------

_N_DAYS = 100
_HORIZON = 7


def _make_history() -> pd.DataFrame:
    """100-day sine-wave sales series with light noise."""
    dates = pd.date_range("2020-01-01", periods=_N_DAYS, freq="D")
    rng = np.random.default_rng(seed=0)
    sales = (
        10.0
        + 5.0 * np.sin(np.arange(_N_DAYS) * 2 * np.pi / 7)
        + rng.normal(0, 0.5, _N_DAYS)
    ).clip(0)
    return pd.DataFrame({"date": dates, "sales": sales})


@pytest.fixture(scope="module")
def fitted_model() -> ProphetForecaster:
    """ProphetForecaster fitted once on the synthetic series.

    yearly_seasonality=False avoids the >2-year data requirement;
    country_holidays=None removes the holiday-lookup overhead.
    Both choices keep fit time well under 10 seconds.
    """
    model = ProphetForecaster(
        yearly_seasonality=False,
        weekly_seasonality=True,
        daily_seasonality=False,
        country_holidays=None,
    )
    model.fit(_make_history())
    return model


@pytest.fixture(scope="module")
def forecast_df(fitted_model) -> pd.DataFrame:
    return fitted_model.forecast(_HORIZON)


# ===========================================================================
# Tests
# ===========================================================================


def test_name_returns_prophet():
    """name() returns the string 'Prophet' regardless of config."""
    assert ProphetForecaster().name() == "Prophet"
    assert ProphetForecaster(yearly_seasonality=False).name() == "Prophet"


def test_not_fitted_raises_before_fit():
    """forecast() raises NotFittedError when called before fit()."""
    model = ProphetForecaster()
    with pytest.raises(NotFittedError):
        model.forecast(7)


def test_fit_returns_self():
    """fit() returns the model instance, enabling fluent chaining."""
    model = ProphetForecaster(
        yearly_seasonality=False,
        weekly_seasonality=True,
        daily_seasonality=False,
        country_holidays=None,
    )
    result = model.fit(_make_history())
    assert result is model


def test_forecast_has_correct_columns(forecast_df):
    """forecast() output contains exactly the 'date' and 'forecast' columns."""
    assert "date" in forecast_df.columns
    assert "forecast" in forecast_df.columns


def test_forecast_has_correct_row_count(forecast_df):
    """forecast() returns exactly horizon rows."""
    assert len(forecast_df) == _HORIZON


def test_forecast_values_non_negative(forecast_df):
    """All forecast values are >= 0.0 (clipping is applied)."""
    assert (forecast_df["forecast"] >= 0.0).all()


def test_forecast_dates_strictly_after_training(fitted_model, forecast_df):
    """Forecast dates begin the day after the last training date."""
    history = _make_history()
    last_train_date = history["date"].max()
    expected_first = last_train_date + pd.Timedelta(days=1)
    assert pd.Timestamp(forecast_df["date"].iloc[0]) == expected_first
    assert (forecast_df["date"] > last_train_date).all()


def test_input_not_mutated():
    """fit() does not rename or modify the caller's DataFrame."""
    history = _make_history()
    original_cols = list(history.columns)
    model = ProphetForecaster(
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
        country_holidays=None,
    )
    model.fit(history)
    assert list(history.columns) == original_cols
