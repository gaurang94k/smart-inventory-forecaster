"""Tests for src/evaluation/metrics.py."""

import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import mape, mase, rmse, wrmsse


# ---------------------------------------------------------------------------
# mape
# ---------------------------------------------------------------------------

def test_mape_happy_path():
    # y_true=100,100  y_pred=110,90  → errors 10% and 10% → mean 10.0
    assert mape([100, 100], [110, 90]) == pytest.approx(10.0)


def test_mape_zero_actual_raises_with_count():
    with pytest.raises(ValueError, match=r"MAPE undefined: 2 zero actuals"):
        mape([0, 100, 0], [1, 110, 2])


def test_mape_zero_actual_message_mentions_mase():
    with pytest.raises(ValueError, match="Use MASE for intermittent series"):
        mape([0], [1])


@pytest.mark.parametrize("array_type", [list, np.array, pd.Series])
def test_mape_accepts_list_ndarray_series(array_type):
    y_true = array_type([100.0, 200.0])
    y_pred = array_type([110.0, 180.0])
    result = mape(y_true, y_pred)
    assert isinstance(result, float)
    assert result == pytest.approx(10.0)


def test_mape_length_mismatch_raises():
    with pytest.raises(ValueError, match="Length mismatch"):
        mape([100, 200], [110])


# ---------------------------------------------------------------------------
# rmse
# ---------------------------------------------------------------------------

def test_rmse_happy_path():
    # errors: -3, +3 → MSE = 9 → RMSE = 3.0
    assert rmse([0, 10], [3, 7]) == pytest.approx(3.0)


def test_rmse_zero_actuals_no_error():
    # RMSE has no issue with zero actuals
    result = rmse([0, 0, 0], [1, 2, 3])
    assert result == pytest.approx(np.sqrt((1 + 4 + 9) / 3))


def test_rmse_length_mismatch_raises():
    with pytest.raises(ValueError, match="Length mismatch"):
        rmse([1, 2, 3], [1, 2])


# ---------------------------------------------------------------------------
# mase
# ---------------------------------------------------------------------------

def test_mase_happy_path():
    # y_train: 8 observations spaced 1 apart, season=7
    # seasonal-naive diffs: |y[7]-y[0]| = |8-1| = 7  → denominator = 7.0
    # test error: |10 - 11| = 1  → MASE = 1/7 ≈ 0.142857
    y_train = np.arange(1.0, 9.0)  # [1,2,3,4,5,6,7,8]
    result = mase([10.0], [11.0], y_train, season_length=7)
    assert result == pytest.approx(1.0 / 7.0)


def test_mase_length_mismatch_raises():
    y_train = np.arange(1.0, 9.0)
    with pytest.raises(ValueError, match="Length mismatch"):
        mase([10.0, 11.0], [12.0], y_train)


def test_mase_y_train_too_short_raises():
    # season_length=7 needs at least 8 observations; give only 7
    y_train = np.arange(1.0, 8.0)  # 7 elements
    with pytest.raises(ValueError, match="requires at least 8"):
        mase([10.0], [11.0], y_train, season_length=7)


def test_mase_constant_y_train_raises():
    # All-same training values → seasonal-naive diffs all zero → denominator=0
    y_train = np.ones(10)
    with pytest.raises(ValueError, match="denominator is zero"):
        mase([1.0], [2.0], y_train, season_length=7)


# ---------------------------------------------------------------------------
# wrmsse (stub / dependency guard)
# ---------------------------------------------------------------------------

def test_wrmsse_raises_not_implemented():
    with pytest.raises(NotImplementedError):
        wrmsse(pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
