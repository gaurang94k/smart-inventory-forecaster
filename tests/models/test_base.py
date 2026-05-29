"""Tests for src/models/base.py — Forecaster ABC and NotFittedError."""

import pandas as pd
import pytest

from src.models.base import Forecaster, NotFittedError


# ---------------------------------------------------------------------------
# Minimal concrete subclass used across multiple tests
# ---------------------------------------------------------------------------

class _MinimalForecaster(Forecaster):
    def fit(self, history: pd.DataFrame) -> "Forecaster":
        return self

    def forecast(self, horizon: int) -> pd.DataFrame:
        return pd.DataFrame({"date": [], "forecast": []})

    def name(self) -> str:
        return "Minimal"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_forecaster_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        Forecaster()  # type: ignore[abstract]


def test_minimal_subclass_instantiates():
    model = _MinimalForecaster()
    assert isinstance(model, Forecaster)


def test_fit_returns_self_for_chaining():
    model = _MinimalForecaster()
    result = model.fit(pd.DataFrame({"date": [], "sales": []}))
    assert result is model


def test_subclass_missing_forecast_cannot_be_instantiated():
    class _NoForecast(Forecaster):
        def fit(self, history: pd.DataFrame) -> "Forecaster":
            return self

        def name(self) -> str:
            return "NoForecast"

    with pytest.raises(TypeError):
        _NoForecast()  # type: ignore[abstract]


def test_not_fitted_error_is_runtime_error():
    assert issubclass(NotFittedError, RuntimeError)


def test_not_fitted_error_can_be_raised_and_caught():
    with pytest.raises(NotFittedError, match="call fit"):
        raise NotFittedError("Must call fit() before forecast().")
