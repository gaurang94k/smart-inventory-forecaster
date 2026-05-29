"""Hybrid SARIMA + LightGBM residuals forecaster.

SARIMA captures the linear/seasonal structure; LightGBM models the residuals
to pick up nonlinear effects (promotions, events, day-of-week × category
interactions).  Both sub-models implement the Forecaster ABC, so the CV loop
can treat the hybrid identically to any other model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.base import Forecaster, NotFittedError
from src.models.sarima import SarimaForecaster
from src.models.lgbm import LightGBMForecaster, FEATURE_COLS


class HybridForecaster(Forecaster):
    """Per-SKU hybrid: SARIMA on the level, LightGBM on the residuals.

    Args:
        sarima_order: ARIMA (p, d, q) non-seasonal order. Default (1, 1, 1).
        sarima_seasonal_order: Seasonal (P, D, Q, s) order. Default (1, 1, 0, 7).
        n_estimators: LightGBM boosting rounds. Default 300.
        learning_rate: LightGBM step-size shrinkage. Default 0.05.
        num_leaves: LightGBM max leaves per tree. Default 31.
        min_child_samples: LightGBM min samples per leaf. Default 20.
        random_state: Random seed for reproducibility. Default 42.
    """

    def __init__(
        self,
        sarima_order: tuple[int, int, int] = (1, 1, 1),
        sarima_seasonal_order: tuple[int, int, int, int] = (1, 1, 0, 7),
        n_estimators: int = 300,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        min_child_samples: int = 20,
        random_state: int = 42,
    ) -> None:
        self.sarima_order = sarima_order
        self.sarima_seasonal_order = sarima_seasonal_order
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.min_child_samples = min_child_samples
        self.random_state = random_state
        self._sarima: SarimaForecaster | None = None
        self._lgbm: LightGBMForecaster | None = None
        self._last_date: pd.Timestamp | None = None
        self._is_fitted: bool = False

    def fit(self, history: pd.DataFrame) -> "HybridForecaster":
        """Fit SARIMA on sales, then fit LightGBM on the SARIMA residuals.

        Args:
            history: DataFrame with 'date', 'sales', and all FEATURE_COLS columns.

        Returns:
            self, for fluent chaining.
        """
        history = history.sort_values("date").reset_index(drop=True)

        # Step 1: fit SARIMA on the raw sales series
        self._sarima = SarimaForecaster(
            order=self.sarima_order,
            seasonal_order=self.sarima_seasonal_order,
        )
        self._sarima.fit(history[["date", "sales"]])

        # Step 2: extract in-sample residuals directly from the fitted result
        residuals = np.asarray(self._sarima._fitted.resid, dtype=float)

        # Step 3: build a residuals dataframe — keep all FEATURE_COLS intact,
        # replace 'sales' with residuals so LightGBM learns to predict them
        resid_df = history.copy()
        resid_df["sales"] = residuals

        # Step 4: fit LightGBM on the residuals
        self._lgbm = LightGBMForecaster(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            min_child_samples=self.min_child_samples,
            random_state=self.random_state,
        )
        self._lgbm.fit(resid_df)

        self._last_date = pd.Timestamp(history["date"].iloc[-1])
        self._is_fitted = True
        return self

    def forecast(self, horizon: int) -> pd.DataFrame:
        """Combine SARIMA and LightGBM residual forecasts.

        Args:
            horizon: Number of days to forecast ahead (e.g. 7, 14, 28).

        Returns:
            DataFrame with columns 'date' and 'forecast'. Values clipped at 0.

        Raises:
            NotFittedError: If called before fit().
        """
        if not self._is_fitted:
            raise NotFittedError(
                f"{self.name()} has not been fitted. Call fit() before forecast()."
            )

        sarima_fc = self._sarima.forecast(horizon)
        lgbm_fc = self._lgbm.forecast(horizon)

        combined = np.clip(
            sarima_fc["forecast"].to_numpy() + lgbm_fc["forecast"].to_numpy(),
            0.0,
            None,
        )

        return pd.DataFrame({"date": sarima_fc["date"], "forecast": combined})

    def name(self) -> str:
        p, d, q = self.sarima_order
        P, D, Q, s = self.sarima_seasonal_order
        return f"Hybrid-SARIMA({p},{d},{q})({P},{D},{Q})[{s}]+LightGBM"
