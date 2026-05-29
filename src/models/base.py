"""Abstract base class and shared exceptions for Phase 2 forecasting models."""

from abc import ABC, abstractmethod

import pandas as pd


# RuntimeError rather than plain Exception: calling forecast() before fit() is
# a misuse of object runtime state (the model exists but isn't ready), not a
# value or type error.  RuntimeError is the idiomatic Python signal for that.
class NotFittedError(RuntimeError):
    """Raised when forecast() is called before fit() on a Forecaster subclass."""


class Forecaster(ABC):
    """Common interface for every model in the Phase 2 comparison.

    Concrete subclasses wrap a specific library (statsmodels, prophet,
    lightgbm, statsforecast) and conform their native API to this interface
    so the cross-validation loop in src/evaluation/cv.py can stay
    model-agnostic.
    """

    @abstractmethod
    def fit(self, history: pd.DataFrame) -> "Forecaster":
        """Train on a historical slice.

        Args:
            history: DataFrame with at minimum a 'date' (datetime) and
                'sales' column. Subclasses may require additional columns
                (e.g. engineered features); they must document their
                requirements.

        Returns:
            self, to allow fluent chaining like Model().fit(df).forecast(28).
        """

    @abstractmethod
    def forecast(self, horizon: int) -> pd.DataFrame:
        """Produce point forecasts for the next `horizon` days.

        Args:
            horizon: number of days to forecast ahead (typically 7, 14, or 28).

        Returns:
            DataFrame with columns 'date' (datetime, future dates) and
            'forecast' (float, predicted units). Subclasses MAY add
            'lower' / 'upper' columns for prediction intervals when supported.
        """

    @abstractmethod
    def name(self) -> str:
        """Human-readable identifier used in comparison tables and logs.

        Returns:
            A short string label for this model.

        Examples:
            'SARIMA', 'Prophet', 'LightGBM-per-SKU',
            'LightGBM-global', 'Hybrid', 'Croston'.
        """
