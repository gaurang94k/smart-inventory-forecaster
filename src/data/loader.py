"""Lazy loaders for the M5 raw CSV files.

All functions read from DATA_DIR, which resolves to <repo_root>/data/raw.
Nothing is loaded on import.
"""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

DATA_DIR: Path = Path(__file__).resolve().parents[2] / "data" / "raw"


def load_sales(sample_skus: int | None = None) -> pd.DataFrame:
    """Load sales_train_evaluation.csv in wide format.

    Args:
        sample_skus: If given, randomly sample this many SKU-store rows from
            the full file. Uses random_state=42 for reproducibility.
            If None, the full ~30 K-row file is returned.

    Returns:
        DataFrame where each row is one SKU-store combination and columns
        include 'id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id',
        and day columns 'd_1' … 'd_1941'.

    Raises:
        FileNotFoundError: If sales_train_evaluation.csv is not found in DATA_DIR.

    Example:
        >>> df = load_sales(sample_skus=100)
        >>> df.shape[0]
        100
    """
    path = DATA_DIR / "sales_train_evaluation.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"sales_train_evaluation.csv not found in {DATA_DIR}. "
            "Download the M5 dataset from Kaggle and place it there."
        )
    df = pd.read_csv(path)
    if sample_skus is not None:
        df = df.sample(n=sample_skus, random_state=42).reset_index(drop=True)
    logger.info("Loaded %d rows from %s", len(df), path.name)
    return df


def validate_sales(df: pd.DataFrame) -> None:
    """Validate the sales dataframe for known data-quality invariants.

    Raises ValueError with a descriptive message if any check fails.
    Does not return anything on success — call it for its side effect.
    """
    # 1. Expected ID columns present
    id_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    missing_id = [c for c in id_cols if c not in df.columns]
    if missing_id:
        raise ValueError(f"Missing expected ID column(s): {missing_id}")

    # 2. Day columns are contiguous: d_1, d_2, ..., d_N — no gaps
    day_cols = [c for c in df.columns if c.startswith("d_") and c[2:].isdigit()]
    if not day_cols:
        raise ValueError("No day columns (d_1, d_2, ...) found in dataframe.")
    day_nums = sorted(int(c[2:]) for c in day_cols)
    expected_nums = list(range(1, day_nums[-1] + 1))
    if day_nums != expected_nums:
        missing = sorted(set(expected_nums) - set(day_nums))
        raise ValueError(
            f"Day columns have gaps. Missing day indices (first 5): {missing[:5]}"
        )

    # 3. No fully-NaN d_* columns
    nan_cols = [c for c in day_cols if df[c].isna().all()]
    if nan_cols:
        raise ValueError(
            f"Found {len(nan_cols)} fully-NaN day column(s): {nan_cols[:3]}"
        )

    # 4. No negative values in any d_* column
    neg_mask = df[day_cols] < 0
    total_neg = int(neg_mask.values.sum())
    if total_neg > 0:
        stacked = neg_mask.stack()
        offending = stacked[stacked].index[:3]
        parts = [
            f"id={df.at[row_i, 'id']} col={col} value={df.at[row_i, col]}"
            for row_i, col in offending
        ]
        raise ValueError(
            f"Found {total_neg} negative sales value(s). "
            f"First offenders: {', '.join(parts)}"
        )

    # 5. No duplicate (item_id, store_id) pairs
    dupes = df.duplicated(subset=["item_id", "store_id"])
    n_dupes = int(dupes.sum())
    if n_dupes > 0:
        dupe_rows = df[dupes][["item_id", "store_id"]].head(3)
        parts = [
            f"item_id={r.item_id} store_id={r.store_id}"
            for r in dupe_rows.itertuples()
        ]
        raise ValueError(
            f"Found {n_dupes} duplicate (item_id, store_id) pair(s). "
            f"First offenders: {', '.join(parts)}"
        )


def load_calendar() -> pd.DataFrame:
    """Load calendar.csv with the 'date' column parsed as datetime.

    Returns:
        DataFrame with columns including 'd' (day id matching sales columns),
        'date' (datetime), 'wm_yr_wk', 'weekday', 'wday', 'month', 'year',
        'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2',
        'snap_CA', 'snap_TX', 'snap_WI'.

    Raises:
        FileNotFoundError: If calendar.csv is not found in DATA_DIR.

    Example:
        >>> cal = load_calendar()
        >>> cal['date'].dtype
        dtype('<M8[ns]')
    """
    path = DATA_DIR / "calendar.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"calendar.csv not found in {DATA_DIR}. "
            "Download the M5 dataset from Kaggle and place it there."
        )
    df = pd.read_csv(path, parse_dates=["date"])
    logger.info("Loaded %d rows from %s", len(df), path.name)
    return df


def load_prices() -> pd.DataFrame:
    """Load sell_prices.csv as-is.

    Returns:
        DataFrame with columns 'store_id', 'item_id', 'wm_yr_wk', 'sell_price'.

    Raises:
        FileNotFoundError: If sell_prices.csv is not found in DATA_DIR.

    Example:
        >>> prices = load_prices()
        >>> list(prices.columns)
        ['store_id', 'item_id', 'wm_yr_wk', 'sell_price']
    """
    path = DATA_DIR / "sell_prices.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"sell_prices.csv not found in {DATA_DIR}. "
            "Download the M5 dataset from Kaggle and place it there."
        )
    df = pd.read_csv(path)
    logger.info("Loaded %d rows from %s", len(df), path.name)
    return df
