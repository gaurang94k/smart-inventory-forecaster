"""SARIMA forecaster wrapping statsmodels SARIMAX.

Implements the Forecaster ABC from src.models.base so the rolling-origin CV
loop can treat it the same as every other model in the Phase 2 comparison.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.models.base import Forecaster, NotFittedError


class SarimaForecaster(Forecaster):
    """Per-SKU SARIMA model via statsmodels SARIMAX.

    Fits one SARIMAX model on a single time series (one SKU). For the global
    LightGBM model that trains across all SKUs simultaneously, see lgbm.py.

    Args:
        order: ARIMA (p, d, q) non-seasonal order. Default (1, 1, 1).
        seasonal_order: Seasonal (P, D, Q, s) order. Default (1, 1, 0, 7)
            for weekly retail data.
        trend: Deterministic trend term passed to SARIMAX. 'n' = no trend
            (default, recommended for differenced models). See statsmodels
            docs for other options ('c', 't', 'ct').
    """

    def __init__(
        self,
        order: tuple[int, int, int] = (1, 1, 1),
        seasonal_order: tuple[int, int, int, int] = (1, 1, 0, 7),
        trend: str | None = "n",
    ) -> None:
        self.order = order
        self.seasonal_order = seasonal_order
        self.trend = trend
        self._fitted = None
        self._last_date: pd.Timestamp | None = None

    def fit(self, history: pd.DataFrame) -> "SarimaForecaster":
        """Fit SARIMAX on the sales column of history.

        Args:
            history: DataFrame with at least 'date' (datetime) and 'sales'
                (numeric) columns. Rows are sorted by date before fitting.

        Returns:
            self, for fluent chaining.
        """
        history = history.sort_values("date")
        sales = history["sales"].to_numpy(dtype=float)

        model = SARIMAX(
            sales,
            order=self.order,
            seasonal_order=self.seasonal_order,
            trend=self.trend,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        # Suppress convergence and numerical warnings that are normal for
        # retail time series with structural breaks and promotion spikes.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._fitted = model.fit(disp=False)

        self._last_date = pd.Timestamp(history["date"].iloc[-1])
        return self

    def forecast(self, horizon: int) -> pd.DataFrame:
        """Produce point forecasts for the next horizon calendar days.

        Args:
            horizon: Number of days to forecast ahead (e.g. 7, 14, 28).

        Returns:
            DataFrame with columns:
                'date'     — datetime, starting the day after the last training date
                'forecast' — float, point forecast clipped at 0.0

        Raises:
            NotFittedError: If called before fit().
        """
        if self._fitted is None:
            raise NotFittedError(
                f"{self.name()} has not been fitted. Call fit() before forecast()."
            )

        fc = self._fitted.get_forecast(steps=horizon)
        values = np.clip(np.asarray(fc.predicted_mean, dtype=float), 0.0, None)

        dates = pd.date_range(
            start=self._last_date + pd.Timedelta(days=1),
            periods=horizon,
            freq="D",
        )
        return pd.DataFrame({"date": dates, "forecast": values})

    def name(self) -> str:
        """Return the model identifier used in comparison tables and logs.

        Returns:
            String in the form "SARIMA(p,d,q)(P,D,Q)[s]".

        Example:
            >>> SarimaForecaster().name()
            'SARIMA(1,1,1)(1,1,0)[7]'
        """
        p, d, q = self.order
        P, D, Q, s = self.seasonal_order
        return f"SARIMA({p},{d},{q})({P},{D},{Q})[{s}]"
