"""Tests for src/models/hybrid.py — HybridForecaster.

Module-scoped fixture trains the model once with minimal SARIMA orders and
n_estimators=30 to keep the suite fast.  The helper _make_sku_features()
mirrors the pattern in test_lgbm.py and produces a DataFrame with all
FEATURE_COLS present.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.base import NotFittedError
from src.models.lgbm import FEATURE_COLS
from src.models.hybrid import HybridForecaster

# ---------------------------------------------------------------------------
# Data helper (mirrors test_lgbm._make_sku_features)
# ---------------------------------------------------------------------------

_N_DAYS = 200
_HORIZON = 14


def _make_sku_features(seed: int = 0) -> pd.DataFrame:
    dates = pd.date_range("2021-01-01", periods=_N_DAYS, freq="D")
    rng = np.random.default_rng(seed=seed)
    sales = (
        10.0
        + 5.0 * np.sin(np.arange(_N_DAYS) * 2 * np.pi / 7)
        + rng.normal(0, 1.0, _N_DAYS)
    ).clip(0)

    df = pd.DataFrame({"date": dates, "sales": sales})

    for lag in [1, 7, 14, 28, 365]:
        df[f"lag_{lag}"] = df["sales"].shift(lag)

    df["roll_mean_7"] = df["sales"].shift(1).rolling(7).mean()
    df["roll_mean_28"] = df["sales"].shift(1).rolling(28).mean()
    df["roll_std_7"] = df["sales"].shift(1).rolling(7).std()

    df["day_of_week"] = df["date"].dt.dayofweek
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    df["is_weekend"] = df["day_of_week"] >= 5

    df["is_event"] = 0
    df["snap_flag"] = 0

    df["sell_price"] = 2.99
    df["price_roll_mean_28"] = 2.99
    df["price_vs_mean"] = 1.0

    return df


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def history() -> pd.DataFrame:
    return _make_sku_features(seed=0)


@pytest.fixture(scope="module")
def fitted_hybrid(history) -> HybridForecaster:
    return HybridForecaster(
        sarima_order=(0, 1, 1),
        sarima_seasonal_order=(0, 0, 0, 7),
        n_estimators=30,
        min_child_samples=5,
    ).fit(history)


@pytest.fixture(scope="module")
def hybrid_forecast(fitted_hybrid) -> pd.DataFrame:
    return fitted_hybrid.forecast(_HORIZON)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_name_format():
    m = HybridForecaster(
        sarima_order=(1, 1, 1),
        sarima_seasonal_order=(1, 1, 0, 7),
    )
    assert m.name() == "Hybrid-SARIMA(1,1,1)(1,1,0)[7]+LightGBM"


def test_name_reflects_actual_orders():
    m = HybridForecaster(
        sarima_order=(0, 1, 1),
        sarima_seasonal_order=(0, 0, 0, 7),
    )
    assert m.name() == "Hybrid-SARIMA(0,1,1)(0,0,0)[7]+LightGBM"


def test_not_fitted_raises():
    with pytest.raises(NotFittedError):
        HybridForecaster().forecast(7)


def test_fit_returns_self(history):
    m = HybridForecaster(
        sarima_order=(0, 1, 1),
        sarima_seasonal_order=(0, 0, 0, 7),
        n_estimators=10,
        min_child_samples=5,
    )
    assert m.fit(history) is m


def test_forecast_columns(hybrid_forecast):
    assert "date" in hybrid_forecast.columns
    assert "forecast" in hybrid_forecast.columns


def test_forecast_row_count(hybrid_forecast):
    assert len(hybrid_forecast) == _HORIZON


def test_forecast_non_negative(hybrid_forecast):
    assert (hybrid_forecast["forecast"] >= 0.0).all()


def test_forecast_dates_after_training(fitted_hybrid, history):
    last_train = history["date"].max()
    fc = fitted_hybrid.forecast(_HORIZON)
    assert (fc["date"] > last_train).all()
    assert fc["date"].iloc[0] == last_train + pd.Timedelta(days=1)
