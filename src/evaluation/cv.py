"""Rolling-origin cross-validation split generator for time-series forecasting."""

from __future__ import annotations

from collections.abc import Iterator

import pandas as pd


def rolling_origin_splits(
    df: pd.DataFrame,
    date_col: str = "date",
    n_folds: int = 5,
    horizon: int = 28,
    step: int = 28,
    initial_train_end: pd.Timestamp | str | None = None,
) -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
    """Generate rolling-origin (train, test) DataFrame pairs.

    Expanding-window rolling-origin CV. Each fold's training set grows by `step`
    days; the test set is always the next `horizon` days. This is the time-series
    analogue of sklearn's TimeSeriesSplit, but yields full DataFrames sliced by
    date rather than index arrays — simpler to compose with pandas-based modeling
    code.

    Works for both:

    - Per-SKU: df contains one SKU's time series. Each fold's train/test slices
      contain the days for that single SKU.
    - Global: df contains many SKUs in long format. Each fold's slices contain
      the days for ALL SKUs in that date range.

    Args:
        df: Input dataframe. Must have a datetime column named by `date_col`.
            Rows do NOT need to be unique by date (multi-SKU is fine).
        date_col: Name of the datetime column. Default "date".
        n_folds: Number of folds. Default 5.
        horizon: Test-window length in days. Default 28.
        step: Days between fold origins. Default 28 (non-overlapping folds).
        initial_train_end: Date marking the END of the first fold's training
            set. The first test window is the `horizon` days immediately after
            this date. Accepts pd.Timestamp or an ISO date string.

            If None, defaults to the value that anchors the last fold's test
            window exactly at max(df[date_col]):

                initial_train_end = max(df[date_col])
                                    - pd.Timedelta(days=(n_folds - 1) * step + horizon)

            Verified against the M5 layout: with max_date=day 1940, n_folds=5,
            step=28, horizon=28 → initial_train_end = day 1800 → fold 1 test
            = days 1801–1828, fold 5 test = days 1913–1940.

    Yields:
        (train_df, test_df) tuples. Both are boolean-indexed slices of the
        input df (pandas creates a copy for boolean indexing).

    Raises:
        ValueError: If date_col is not a column in df.
        ValueError: If df is empty.
        ValueError: If the date_col column is not datetime dtype.
        ValueError: If initial_train_end (default or explicit) falls before
            min(df[date_col]) — not enough data for the first training fold.
        ValueError: If the last fold's test window extends beyond
            max(df[date_col]) — data too short for the requested fold layout.

    Example:
        >>> import pandas as pd
        >>> dates = pd.date_range("2020-01-01", periods=200, freq="D")
        >>> df = pd.DataFrame({"date": dates, "sales": 1.0})
        >>> folds = list(rolling_origin_splits(df, n_folds=5, horizon=28))
        >>> len(folds)
        5
    """
    if date_col not in df.columns:
        raise ValueError(
            f"date_col '{date_col}' not found in dataframe columns: "
            f"{list(df.columns)}."
        )
    if df.empty:
        raise ValueError("Input dataframe is empty.")
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        raise ValueError(
            f"Column '{date_col}' must be datetime dtype, got {df[date_col].dtype}. "
            "Parse it with pd.to_datetime() before passing to this function."
        )

    max_date: pd.Timestamp = df[date_col].max()
    min_date: pd.Timestamp = df[date_col].min()

    if initial_train_end is None:
        train_end_0 = max_date - pd.Timedelta(days=(n_folds - 1) * step + horizon)
    else:
        train_end_0 = pd.Timestamp(initial_train_end)

    if train_end_0 < min_date:
        raise ValueError(
            f"initial_train_end ({train_end_0.date()}) is before the earliest date "
            f"in the dataframe ({min_date.date()}). "
            f"Not enough data to form the first training fold."
        )

    last_test_end = train_end_0 + pd.Timedelta(days=(n_folds - 1) * step + horizon)
    if last_test_end > max_date:
        raise ValueError(
            f"Data does not support {n_folds} fold(s) with "
            f"step={step}, horizon={horizon}. "
            f"Last fold requires data through {last_test_end.date()}, "
            f"but df ends at {max_date.date()}."
        )

    horizon_delta = pd.Timedelta(days=horizon)
    step_delta = pd.Timedelta(days=step)

    for k in range(n_folds):
        train_end = train_end_0 + k * step_delta
        test_end = train_end + horizon_delta

        train_df = df[df[date_col] <= train_end]
        test_df = df[(df[date_col] > train_end) & (df[date_col] <= test_end)]

        yield train_df, test_df
