"""Croston intermittent demand forecaster wrapping statsforecast.

Implements the Forecaster ABC so the CV loop can treat Croston identically to
every other model.  Intended for the one intermittent SKU in the shortlist
(HOUSEHOLD_1_430_CA_1, ~78% zero rate) where ARIMA/LightGBM are a poor fit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import CrostonOptimized

from src.models.base import Forecaster, NotFittedError


class CrostonForecaster(Forecaster):
    """Per-SKU Croston model via statsforecast.

    Args:
        alias: Croston variant to use. Currently only "CrostonOptimized" is
            exposed; the parameter is kept for forward-compatibility.
    """

    def __init__(self, alias: str = "CrostonOptimized") -> None:
        self.alias = alias
        self._sf: StatsForecast | None = None
        self._last_date: pd.Timestamp | None = None
        self._unique_id: str = "item"
        self._fitted: bool = False

    def fit(self, history: pd.DataFrame) -> "CrostonForecaster":
        """Fit CrostonOptimized on the sales column of history.

        Args:
            history: DataFrame with at least 'date' (datetime) and 'sales'
                (numeric) columns. Sorted by date before fitting.

        Returns:
            self, for fluent chaining.
        """
        history = history.sort_values("date")

        sf_df = pd.DataFrame({
            "unique_id": self._unique_id,
            "ds": pd.to_datetime(history["date"]),
            "y": history["sales"].to_numpy(dtype=float),
        })

        self._sf = StatsForecast(models=[CrostonOptimized()], freq="D")
        self._sf.fit(sf_df)

        self._last_date = pd.Timestamp(history["date"].iloc[-1])
        self._fitted = True
        return self

    def forecast(self, horizon: int) -> pd.DataFrame:
        """Produce flat demand-rate forecasts for the next horizon calendar days.

        Croston yields a constant demand-rate estimate, so all horizon rows
        will carry the same value.

        Args:
            horizon: Number of days to forecast ahead.

        Returns:
            DataFrame with columns 'date' and 'forecast'. Values clipped at 0.

        Raises:
            NotFittedError: If called before fit().
        """
        if not self._fitted:
            raise NotFittedError(
                f"{self.name()} has not been fitted. Call fit() before forecast()."
            )

        pred = self._sf.predict(h=horizon)
        values = np.clip(pred["CrostonOptimized"].to_numpy(dtype=float), 0.0, None)

        dates = pd.date_range(
            start=self._last_date + pd.Timedelta(days=1),
            periods=horizon,
            freq="D",
        )
        return pd.DataFrame({"date": dates, "forecast": values})

    def name(self) -> str:
        return "Croston"
