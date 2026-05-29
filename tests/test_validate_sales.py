"""Tests for validate_sales() in src/data/loader.py.

All dataframes are synthetic — no M5 data required.
"""

import pandas as pd
import pytest

from src.data.loader import validate_sales


def _make_clean_df(n_skus: int = 3, n_days: int = 5) -> pd.DataFrame:
    """Build a minimal valid sales dataframe for testing."""
    rows = []
    for i in range(n_skus):
        row = {
            "id": f"ITEM_{i}_CA_1",
            "item_id": f"ITEM_{i}",
            "dept_id": f"DEPT_{i % 2}",
            "cat_id": "CAT_1",
            "store_id": "CA_1",
            "state_id": "CA",
        }
        for d in range(1, n_days + 1):
            row[f"d_{d}"] = float(i + d)
        rows.append(row)
    return pd.DataFrame(rows)


def test_happy_path_passes():
    df = _make_clean_df()
    validate_sales(df)  # must not raise


def test_missing_id_column_raises():
    df = _make_clean_df().drop(columns=["dept_id"])
    with pytest.raises(ValueError, match="Missing expected ID column"):
        validate_sales(df)


def test_gap_in_day_sequence_raises():
    df = _make_clean_df(n_days=5).drop(columns=["d_3"])
    with pytest.raises(ValueError, match="Day columns have gaps"):
        validate_sales(df)


def test_fully_nan_day_column_raises():
    df = _make_clean_df(n_days=5)
    df["d_3"] = float("nan")
    with pytest.raises(ValueError, match="fully-NaN day column"):
        validate_sales(df)


def test_negative_sales_raises():
    df = _make_clean_df(n_days=5)
    df.at[1, "d_2"] = -3.0
    with pytest.raises(ValueError, match="negative sales value"):
        validate_sales(df)


def test_negative_error_includes_offender_details():
    df = _make_clean_df(n_days=5)
    df.at[0, "d_4"] = -99.0
    with pytest.raises(ValueError, match=r"id=ITEM_0_CA_1 col=d_4 value=-99\.0"):
        validate_sales(df)


def test_duplicate_item_store_raises():
    df = _make_clean_df(n_days=5)
    extra = df.iloc[[0]].copy()
    extra["id"] = "ITEM_0_CA_1_dup"
    df = pd.concat([df, extra], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        validate_sales(df)
