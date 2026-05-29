"""In-house forecast accuracy metrics for the Phase 2 model comparison.

MAPE, RMSE, and MASE are implemented from scratch — the math is simple enough
that writing them ourselves is both instructive and gives full control over edge
cases (zero actuals, short training series).

WRMSSE delegates to the utilsforecast reference implementation; see the
wrmsse() docstring and docs/phase2_architecture.md Decision 4 for the rationale.
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_array(x: np.ndarray | pd.Series) -> np.ndarray:
    """Coerce Series or ndarray to a 1-D float64 ndarray."""
    return np.asarray(x, dtype=np.float64)


def _validate_lengths(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Length mismatch: y_true has {len(y_true)} elements, "
            f"y_pred has {len(y_pred)}."
        )


# ---------------------------------------------------------------------------
# Public metrics
# ---------------------------------------------------------------------------

def mape(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    """Mean absolute percentage error, expressed as a percentage.

    Args:
        y_true: Observed actuals. Must contain no zeros (MAPE is undefined
            for zero actuals; use mase() for intermittent series instead).
        y_pred: Model forecasts. Must be the same length as y_true.

    Returns:
        MAPE as a percentage (e.g. 12.5 means 12.5%, not 0.125).

    Raises:
        ValueError: If y_true contains any zeros.
        ValueError: If y_true and y_pred have different lengths.

    Example:
        >>> mape([100, 100], [110, 90])
        10.0
    """
    y_true = _to_array(y_true)
    y_pred = _to_array(y_pred)
    _validate_lengths(y_true, y_pred)

    n_zeros = int(np.sum(y_true == 0))
    if n_zeros > 0:
        raise ValueError(
            f"MAPE undefined: {n_zeros} zero actuals. "
            "Use MASE for intermittent series."
        )

    return float(np.mean(np.abs((y_true - y_pred) / np.abs(y_true))) * 100)


def rmse(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    """Root mean squared error.

    Args:
        y_true: Observed actuals.
        y_pred: Model forecasts. Must be the same length as y_true.

    Returns:
        RMSE in the same units as y_true.

    Raises:
        ValueError: If y_true and y_pred have different lengths.

    Example:
        >>> rmse([0, 10], [0, 7])
        3.0
    """
    y_true = _to_array(y_true)
    y_pred = _to_array(y_pred)
    _validate_lengths(y_true, y_pred)

    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mase(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    y_train: np.ndarray | pd.Series,
    season_length: int = 7,
) -> float:
    """Mean absolute scaled error.

    Scales the forecast error by the in-sample seasonal-naive error, making
    MASE comparable across series of different scales. The preferred metric
    for intermittent demand (where MAPE is undefined).

    Args:
        y_true: Observed actuals for the test window.
        y_pred: Model forecasts. Must be the same length as y_true.
        y_train: Training-set actuals used to compute the seasonal-naive
            scaling denominator.
        season_length: Number of periods in one season. Default 7 (weekly
            seasonality for daily retail data).

    Returns:
        MASE (dimensionless). Values below 1.0 mean the model beats a
        seasonal-naive baseline.

    Raises:
        ValueError: If y_true and y_pred have different lengths.
        ValueError: If y_train has fewer than season_length + 1 observations
            (denominator requires at least one seasonal lag step).
        ValueError: If the seasonal-naive denominator is zero (constant
            training series — degenerate case).

    Example:
        >>> import numpy as np
        >>> y_train = np.arange(1, 22, dtype=float)  # 21 obs, season=7
        >>> mase([22.0], [23.0], y_train, season_length=7)
        0.142857...
    """
    y_true = _to_array(y_true)
    y_pred = _to_array(y_pred)
    y_train = _to_array(y_train)
    _validate_lengths(y_true, y_pred)

    if len(y_train) < season_length + 1:
        raise ValueError(
            f"y_train has {len(y_train)} observations but season_length={season_length} "
            f"requires at least {season_length + 1} to compute the scaling denominator."
        )

    denominator = float(
        np.mean(np.abs(y_train[season_length:] - y_train[:-season_length]))
    )
    if denominator == 0.0:
        raise ValueError(
            "Seasonal-naive denominator is zero: y_train is constant over the "
            "training window. MASE is undefined for a flat series."
        )

    return float(np.mean(np.abs(y_true - y_pred)) / denominator)


def wrmsse(
    forecasts_df: pd.DataFrame,
    actuals_df: pd.DataFrame,
    weights_df: pd.DataFrame,
) -> float:
    """Weighted root mean squared scaled error (M5 official competition metric).

    Delegates to the utilsforecast reference implementation from the Nixtla
    ecosystem. We do not re-implement WRMSSE in-house because the M5
    hierarchical weighting scheme is complex and any subtle bug would be
    immediately visible to reviewers familiar with the competition. See
    docs/phase2_architecture.md Decision 4 for the full rationale.

    # TODO: add utilsforecast to the project with `uv add utilsforecast`
    #       (pending explicit approval) before calling this function.

    Args:
        forecasts_df: DataFrame of point forecasts. Expected shape mirrors
            the utilsforecast input convention: rows are series × horizon,
            with columns ['unique_id', 'ds', '<model_name>'].
        actuals_df: DataFrame of ground-truth actuals in the same shape.
        weights_df: Per-series weights as defined by the M5 competition
            (dollar-value weights derived from the sell-price files).

    Returns:
        WRMSSE scalar (float). Lower is better; 1.0 equals the M5 benchmark
        naive forecast.

    Raises:
        ImportError: If utilsforecast is not installed.
        NotImplementedError: Always — this function is a stub until the
            utilsforecast dependency is added.
    """
    raise NotImplementedError(
        "Pending utilsforecast dependency — see docstring and "
        "docs/phase2_architecture.md Decision 4."
    )
