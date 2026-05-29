"""Tests for src/evaluation/cv.py."""

import pandas as pd
import pytest

from src.evaluation.cv import rolling_origin_splits


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _single_sku_df(n_days: int = 200) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=n_days, freq="D")
    return pd.DataFrame({"date": dates, "sales": 1.0})


def _multi_sku_df(n_days: int = 200, skus: list[str] | None = None) -> pd.DataFrame:
    if skus is None:
        skus = ["SKU_A", "SKU_B", "SKU_C"]
    dates = pd.date_range("2020-01-01", periods=n_days, freq="D")
    return pd.concat(
        [pd.DataFrame({"date": dates, "sku": sku, "sales": 1.0}) for sku in skus],
        ignore_index=True,
    )


# ---------------------------------------------------------------------------
# Happy path — single-SKU
# ---------------------------------------------------------------------------

def test_happy_path_yields_n_folds():
    folds = list(rolling_origin_splits(_single_sku_df(), n_folds=5, horizon=28, step=28))
    assert len(folds) == 5


def test_happy_path_expanding_train():
    folds = list(rolling_origin_splits(_single_sku_df(), n_folds=5, horizon=28, step=28))
    train_sizes = [len(train) for train, _ in folds]
    assert train_sizes == sorted(train_sizes), "Training sets must grow each fold."
    assert len(set(train_sizes)) == 5, "Each fold must have a strictly larger training set."


def test_happy_path_test_size_equals_horizon():
    horizon = 28
    folds = list(rolling_origin_splits(_single_sku_df(), n_folds=5, horizon=horizon, step=28))
    for _, test in folds:
        assert len(test) == horizon


def test_happy_path_no_train_test_overlap():
    folds = list(rolling_origin_splits(_single_sku_df(), n_folds=5, horizon=28, step=28))
    for train, test in folds:
        assert test["date"].min() > train["date"].max()


# ---------------------------------------------------------------------------
# Multi-SKU
# ---------------------------------------------------------------------------

def test_multi_sku_all_skus_present_in_every_fold():
    skus = ["SKU_A", "SKU_B", "SKU_C"]
    df = _multi_sku_df(skus=skus)
    for train, test in rolling_origin_splits(df, n_folds=5, horizon=28, step=28):
        assert set(train["sku"].unique()) == set(skus)
        assert set(test["sku"].unique()) == set(skus)
        assert len(test) == 28 * len(skus)


# ---------------------------------------------------------------------------
# Custom initial_train_end
# ---------------------------------------------------------------------------

def test_custom_initial_train_end_respected():
    df = _single_sku_df()
    t0 = pd.Timestamp("2020-03-15")
    folds = list(rolling_origin_splits(df, initial_train_end=t0, n_folds=3, horizon=28, step=28))
    train0, _ = folds[0]
    assert train0["date"].max() == t0


# ---------------------------------------------------------------------------
# Fold-boundary arithmetic
# ---------------------------------------------------------------------------

def test_fold_boundary_train_ends_match_expected():
    df = _single_sku_df()
    t0 = pd.Timestamp("2020-03-15")
    folds = list(rolling_origin_splits(df, initial_train_end=t0, n_folds=3, horizon=28, step=28))
    expected_ends = [
        t0,
        t0 + pd.Timedelta(days=28),
        t0 + pd.Timedelta(days=56),
    ]
    for (train, _), expected in zip(folds, expected_ends):
        assert train["date"].max() == expected


# ---------------------------------------------------------------------------
# Pre-flight validation
# ---------------------------------------------------------------------------

def test_too_short_df_raises():
    # 50 days: default initial_train_end = max - 140 days → before min_date
    df = _single_sku_df(n_days=50)
    with pytest.raises(ValueError, match="Not enough data"):
        list(rolling_origin_splits(df, n_folds=5, horizon=28, step=28))


def test_custom_initial_train_end_too_late_raises():
    # initial_train_end that pushes last fold's test window past max_date
    df = _single_sku_df(n_days=50)
    too_late = pd.Timestamp("2020-02-15")  # last fold test ends well past 2020-02-19
    with pytest.raises(ValueError, match="does not support"):
        list(rolling_origin_splits(df, initial_train_end=too_late, n_folds=5, horizon=28, step=28))


def test_missing_date_col_raises():
    df = pd.DataFrame({"sales": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="date_col"):
        list(rolling_origin_splits(df))


def test_wrong_date_dtype_raises():
    df = pd.DataFrame({"date": ["2020-01-01", "2020-01-02", "2020-01-03"], "sales": 1.0})
    with pytest.raises(ValueError, match="datetime dtype"):
        list(rolling_origin_splits(df))


def test_empty_df_raises():
    df = pd.DataFrame(
        {"date": pd.Series(dtype="datetime64[ns]"), "sales": pd.Series(dtype=float)}
    )
    with pytest.raises(ValueError, match="empty"):
        list(rolling_origin_splits(df))
