"""Tests for src/models/lgbm.py — LightGBMForecaster and GlobalLightGBMForecaster.

All tests that need a fitted model share module-scoped fixtures so LightGBM
is trained exactly once per class.  Fixtures use n_estimators=50 to keep fit
time well under a second.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.base import NotFittedError
from src.models.lgbm import (
    FEATURE_COLS,
    GLOBAL_ONLY_COLS,
    GlobalLightGBMForecaster,
    LightGBMForecaster,
)

# ---------------------------------------------------------------------------
# Synthetic feature-engineered data helpers
# ---------------------------------------------------------------------------

_N_DAYS = 200
_HORIZON = 7


def _make_sku_features(
    sku_id: str,
    cat_id: str = "FOODS",
    dept_id: str = "FOODS_3",
    store_id: str = "CA_1",
    state_id: str = "CA",
    seed: int = 0,
) -> pd.DataFrame:
    """200-day feature-engineered DataFrame for one SKU.

    Mirrors the column schema produced by build_features_per_sku / global,
    including all FEATURE_COLS and GLOBAL_ONLY_COLS.
    """
    dates = pd.date_range("2021-01-01", periods=_N_DAYS, freq="D")
    rng = np.random.default_rng(seed=seed)
    sales = (
        10.0
        + 5.0 * np.sin(np.arange(_N_DAYS) * 2 * np.pi / 7)
        + rng.normal(0, 1.0, _N_DAYS)
    ).clip(0)

    df = pd.DataFrame(
        {
            "id": sku_id,
            "item_id": f"{sku_id}_ITEM",
            "dept_id": dept_id,
            "cat_id": cat_id,
            "store_id": store_id,
            "state_id": state_id,
            "date": dates,
            "sales": sales,
        }
    )

    # Lag features (matching builder.py convention)
    for lag in [1, 7, 14, 28, 365]:
        df[f"lag_{lag}"] = df["sales"].shift(lag)

    # Rolling stats with shift(1) to exclude the current day
    df["roll_mean_7"] = df["sales"].shift(1).rolling(7).mean()
    df["roll_mean_28"] = df["sales"].shift(1).rolling(28).mean()
    df["roll_std_7"] = df["sales"].shift(1).rolling(7).std()

    # Calendar
    df["day_of_week"] = df["date"].dt.dayofweek
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    df["is_weekend"] = df["day_of_week"] >= 5  # bool, matches builder output

    # Event / SNAP
    df["is_event"] = 0
    df["snap_flag"] = 0

    # Price (constant for simplicity — all rows have a valid price)
    df["sell_price"] = 2.99
    df["price_roll_mean_28"] = 2.99
    df["price_vs_mean"] = 1.0

    return df


def _make_global_features() -> pd.DataFrame:
    """Two-SKU feature-engineered DataFrame for the global model."""
    sku_a = _make_sku_features(
        "FOODS_3_090_CA_1",
        cat_id="FOODS", dept_id="FOODS_3", store_id="CA_1", state_id="CA",
        seed=0,
    )
    sku_b = _make_sku_features(
        "HOUSEHOLD_1_118_TX_1",
        cat_id="HOUSEHOLD", dept_id="HOUSEHOLD_1", store_id="TX_1", state_id="TX",
        seed=1,
    )
    return pd.concat([sku_a, sku_b], ignore_index=True)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def per_sku_history() -> pd.DataFrame:
    return _make_sku_features("FOODS_3_090_CA_1", seed=0)


@pytest.fixture(scope="module")
def global_history() -> pd.DataFrame:
    return _make_global_features()


@pytest.fixture(scope="module")
def fitted_per_sku(per_sku_history) -> LightGBMForecaster:
    return LightGBMForecaster(n_estimators=50, min_child_samples=5).fit(per_sku_history)


@pytest.fixture(scope="module")
def fitted_global(global_history) -> GlobalLightGBMForecaster:
    return GlobalLightGBMForecaster(n_estimators=50, min_child_samples=5).fit(global_history)


@pytest.fixture(scope="module")
def per_sku_forecast(fitted_per_sku) -> pd.DataFrame:
    return fitted_per_sku.forecast(_HORIZON)


@pytest.fixture(scope="module")
def global_forecast(fitted_global) -> pd.DataFrame:
    return fitted_global.forecast(_HORIZON)


# ===========================================================================
# Tests
# ===========================================================================

# --- name() -----------------------------------------------------------------

def test_per_sku_name():
    assert LightGBMForecaster().name() == "LightGBM-per-SKU"


def test_global_name():
    assert GlobalLightGBMForecaster().name() == "LightGBM-global"


# --- NotFittedError before fit() -------------------------------------------

def test_per_sku_not_fitted_raises():
    with pytest.raises(NotFittedError):
        LightGBMForecaster().forecast(7)


def test_global_not_fitted_raises():
    with pytest.raises(NotFittedError):
        GlobalLightGBMForecaster().forecast(7)


# --- fit() returns self -------------------------------------------------------

def test_per_sku_fit_returns_self(per_sku_history):
    model = LightGBMForecaster(n_estimators=10, min_child_samples=5)
    assert model.fit(per_sku_history) is model


def test_global_fit_returns_self(global_history):
    model = GlobalLightGBMForecaster(n_estimators=10, min_child_samples=5)
    assert model.fit(global_history) is model


# --- forecast() output structure --------------------------------------------

def test_per_sku_forecast_columns(per_sku_forecast):
    assert "date" in per_sku_forecast.columns
    assert "forecast" in per_sku_forecast.columns


def test_per_sku_forecast_row_count(per_sku_forecast):
    assert len(per_sku_forecast) == _HORIZON


def test_global_forecast_columns(global_forecast):
    assert "date" in global_forecast.columns
    assert "forecast" in global_forecast.columns


def test_global_forecast_row_count(global_forecast):
    assert len(global_forecast) == _HORIZON


# --- Clipping ---------------------------------------------------------------

def test_per_sku_forecast_non_negative(per_sku_forecast):
    assert (per_sku_forecast["forecast"] >= 0.0).all()


def test_global_forecast_non_negative(global_forecast):
    assert (global_forecast["forecast"] >= 0.0).all()


# --- Recursive horizon length -----------------------------------------------

def test_per_sku_forecast_28_rows(fitted_per_sku):
    """Recursive loop runs for the full 28-day horizon without errors."""
    result = fitted_per_sku.forecast(28)
    assert len(result) == 28


# --- Date ordering ----------------------------------------------------------

def test_per_sku_forecast_dates_after_training(fitted_per_sku, per_sku_history):
    last_train = per_sku_history["date"].max()
    fc = fitted_per_sku.forecast(_HORIZON)
    assert (fc["date"] > last_train).all()
    assert fc["date"].iloc[0] == last_train + pd.Timedelta(days=1)


# --- Global model: sku_id routing -------------------------------------------

def test_global_forecast_explicit_sku_id(fitted_global, global_history):
    """Forecasting with an explicit sku_id returns a valid result."""
    sku = "HOUSEHOLD_1_118_TX_1"
    fc = fitted_global.forecast(_HORIZON, sku_id=sku)
    assert len(fc) == _HORIZON
    assert (fc["forecast"] >= 0.0).all()
