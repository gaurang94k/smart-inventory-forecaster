"""Tests for src/models/sarima.py.

Speed note: SARIMA fitting is slow on long series. All tests that require a
fitted model share a single module-scoped fixture that fits once on a 100-day
synthetic series using a minimal (0,1,1)(0,0,0,7) order. This keeps the full
test run under a few seconds while still exercising the real fitting path.
"""

import numpy as np
import pandas as pd
import pytest

from src.models.base import NotFittedError
from src.models.sarima import SarimaForecaster

# ---------------------------------------------------------------------------
# Synthetic data and fitted-model fixture
# ---------------------------------------------------------------------------

_N_DAYS = 100
_HORIZON = 7


def _make_history() -> pd.DataFrame:
    """100-day sine-wave series with light noise — enough structure for SARIMA."""
    dates = pd.date_range("2020-01-01", periods=_N_DAYS, freq="D")
    rng = np.random.default_rng(seed=42)
    sales = (
        10.0
        + 5.0 * np.sin(np.arange(_N_DAYS) * 2 * np.pi / 7)
        + rng.normal(0, 0.5, _N_DAYS)
    ).clip(0)
    return pd.DataFrame({"date": dates, "sales": sales})


@pytest.fixture(scope="module")
def fitted_model() -> SarimaForecaster:
    """SarimaForecaster fitted once on the synthetic series.

    Uses (0,1,1)(0,0,0,7) — IMA(1,1) with weekly period, no seasonal terms —
    to minimise fit time while still exercising the full fit/forecast path.
    """
    model = SarimaForecaster(
        order=(0, 1, 1),
        seasonal_order=(0, 0, 0, 7),
        trend="n",
    )
    model.fit(_make_history())
    return model


@pytest.fixture(scope="module")
def forecast_df(fitted_model) -> pd.DataFrame:
    return fitted_model.forecast(_HORIZON)


# ===========================================================================
# Tests
# ===========================================================================


def test_name_default_format():
    """name() returns the canonical SARIMA(p,d,q)(P,D,Q)[s] string for defaults."""
    model = SarimaForecaster()
    assert model.name() == "SARIMA(1,1,1)(1,1,0)[7]"


def test_name_custom_order():
    """name() reflects the custom order and seasonal_order passed at construction."""
    model = SarimaForecaster(order=(2, 0, 1), seasonal_order=(0, 1, 1, 52))
    assert model.name() == "SARIMA(2,0,1)(0,1,1)[52]"


def test_not_fitted_raises_before_fit():
    """forecast() raises NotFittedError when called before fit()."""
    model = SarimaForecaster()
    with pytest.raises(NotFittedError):
        model.forecast(7)


def test_fit_returns_self():
    """fit() returns the model instance, enabling fluent chaining."""
    model = SarimaForecaster(order=(0, 1, 1), seasonal_order=(0, 0, 0, 7))
    result = model.fit(_make_history())
    assert result is model


def test_forecast_has_correct_columns(forecast_df):
    """forecast() output contains exactly 'date' and 'forecast' columns."""
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
    assert forecast_df["date"].iloc[0] == expected_first
    assert (forecast_df["date"] > last_train_date).all()
