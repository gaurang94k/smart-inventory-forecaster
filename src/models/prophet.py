"""Prophet forecaster wrapping Meta's prophet library.

Implements the Forecaster ABC from src.models.base so the rolling-origin CV
loop can treat it the same as every other model in the Phase 2 comparison.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

# Suppress Prophet's and cmdstanpy's verbose init/fit output before import.
logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

from prophet import Prophet  # noqa: E402  (must come after logging config)

from src.models.base import Forecaster, NotFittedError


class ProphetForecaster(Forecaster):
    """Per-SKU Prophet model.

    Wraps Meta's Prophet (daily frequency, optional US holidays). A new
    Prophet instance is created inside each fit() call because Prophet does
    not support re-fitting on a different dataset.

    Args:
        yearly_seasonality: Include a Fourier-series yearly component.
            Default True. Set False for short series (< 2 years) to avoid
            overfitting.
        weekly_seasonality: Include a weekly Fourier component. Default True.
        daily_seasonality: Include a daily Fourier component. Default False
            (not meaningful for daily-aggregated retail data).
        country_holidays: ISO-3166 country code passed to
            add_country_holidays(). Default "US" (M5 Walmart data). Pass
            None to skip holiday effects.
    """

    def __init__(
        self,
        yearly_seasonality: bool = True,
        weekly_seasonality: bool = True,
        daily_seasonality: bool = False,
        country_holidays: str | None = "US",
    ) -> None:
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.country_holidays = country_holidays
        self._model: Prophet | None = None
        self._last_date: pd.Timestamp | None = None

    def fit(self, history: pd.DataFrame) -> "ProphetForecaster":
        """Fit a fresh Prophet model on the sales column of history.

        Args:
            history: DataFrame with at least 'date' (datetime) and 'sales'
                (numeric) columns. Rows are sorted by date before fitting.
                The input DataFrame is not mutated.

        Returns:
            self, for fluent chaining.
        """
        history = history.sort_values("date")
        # Prophet requires columns named 'ds' and 'y' — copy to avoid mutation
        prophet_df = history[["date", "sales"]].rename(
            columns={"date": "ds", "sales": "y"}
        )

        model = Prophet(
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
        )
        if self.country_holidays is not None:
            model.add_country_holidays(country_name=self.country_holidays)

        model.fit(prophet_df)
        self._model = model
        self._last_date = pd.Timestamp(history["date"].iloc[-1])
        return self

    def forecast(self, horizon: int) -> pd.DataFrame:
        """Produce point forecasts for the next horizon calendar days.

        Args:
            horizon: Number of days to forecast ahead (e.g. 7, 14, 28).

        Returns:
            DataFrame with columns:
                'date'     — datetime, the next horizon days after training end
                'forecast' — float, Prophet's yhat clipped at 0.0

        Raises:
            NotFittedError: If called before fit().
        """
        if self._model is None:
            raise NotFittedError(
                f"{self.name()} has not been fitted. Call fit() before forecast()."
            )

        future = self._model.make_future_dataframe(
            periods=horizon, include_history=False
        )
        prediction = self._model.predict(future)

        return pd.DataFrame(
            {
                "date": prediction["ds"],
                "forecast": prediction["yhat"].clip(lower=0.0),
            }
        ).reset_index(drop=True)

    def name(self) -> str:
        """Return the model identifier used in comparison tables and logs.

        Returns:
            "Prophet"
        """
        return "Prophet"
