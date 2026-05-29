"""Tests for src/features/builder.py."""

import numpy as np
import pandas as pd
import pytest

from src.features.builder import build_features_global, build_features_per_sku

# ---------------------------------------------------------------------------
# Synthetic data configuration
# ---------------------------------------------------------------------------

N_DAYS = 400
START_DATE = pd.Timestamp("2011-01-29")  # M5 dataset start date (a Saturday)

_SKUS = [
    {
        "id": "FOODS_3_090_CA_1",
        "item_id": "FOODS_3_090",
        "dept_id": "FOODS_3",
        "cat_id": "FOODS",
        "store_id": "CA_1",
        "state_id": "CA",
    },
    {
        "id": "HOUSEHOLD_1_118_TX_1",
        "item_id": "HOUSEHOLD_1_118",
        "dept_id": "HOUSEHOLD_1",
        "cat_id": "HOUSEHOLD",
        "store_id": "TX_1",
        "state_id": "TX",
    },
    {
        "id": "HOBBIES_1_348_WI_1",
        "item_id": "HOBBIES_1_348",
        "dept_id": "HOBBIES_1",
        "cat_id": "HOBBIES",
        "store_id": "WI_1",
        "state_id": "WI",
    },
]

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sales_long() -> pd.DataFrame:
    """3 SKUs × 400 days of synthetic long-format sales."""
    dates = pd.date_range(START_DATE, periods=N_DAYS, freq="D")
    rows = []
    for seed, sku in enumerate(_SKUS):
        rng = np.random.default_rng(seed=seed)
        sales_vals = rng.integers(0, 100, size=N_DAYS).astype(float)
        for day_idx, date in enumerate(dates):
            rows.append({**sku, "date": date, "sales": sales_vals[day_idx]})
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def calendar() -> pd.DataFrame:
    """Synthetic M5-shaped calendar over the 400-day window."""
    dates = pd.date_range(START_DATE, periods=N_DAYS, freq="D")
    rng = np.random.default_rng(seed=99)

    iso = pd.DatetimeIndex(dates).isocalendar()
    wm_yr_wk = (iso["year"].values * 100 + iso["week"].values).astype(int)

    # ~10% of days have an event to ensure both branches of is_event are tested
    event_name_1 = [
        "SomeEvent" if rng.random() < 0.10 else None for _ in range(N_DAYS)
    ]

    return pd.DataFrame(
        {
            "date": dates,
            "d": [f"d_{i + 1}" for i in range(N_DAYS)],
            "wm_yr_wk": wm_yr_wk,
            "weekday": pd.DatetimeIndex(dates).day_name(),
            "wday": pd.DatetimeIndex(dates).dayofweek + 1,
            "month": pd.DatetimeIndex(dates).month,
            "year": pd.DatetimeIndex(dates).year,
            "event_name_1": event_name_1,
            "event_type_1": None,
            "event_name_2": None,
            "event_type_2": None,
            "snap_CA": rng.integers(0, 2, size=N_DAYS).astype(int),
            "snap_TX": rng.integers(0, 2, size=N_DAYS).astype(int),
            "snap_WI": rng.integers(0, 2, size=N_DAYS).astype(int),
        }
    )


@pytest.fixture(scope="module")
def prices(calendar) -> pd.DataFrame:
    """One sell_price per (store_id, item_id, wm_yr_wk)."""
    unique_wks = calendar["wm_yr_wk"].unique()
    rng = np.random.default_rng(seed=7)
    rows = []
    for sku in _SKUS:
        for wk in unique_wks:
            rows.append(
                {
                    "store_id": sku["store_id"],
                    "item_id": sku["item_id"],
                    "wm_yr_wk": int(wk),
                    "sell_price": round(float(rng.uniform(1.0, 20.0)), 2),
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def features_per_sku(sales_long, calendar, prices) -> pd.DataFrame:
    return build_features_per_sku(sales_long, calendar, prices)


@pytest.fixture(scope="module")
def features_global(sales_long, calendar, prices) -> pd.DataFrame:
    return build_features_global(sales_long, calendar, prices)


# ---------------------------------------------------------------------------
# Helper: one SKU's rows in chronological order with a clean RangeIndex
# ---------------------------------------------------------------------------


def _sku_rows(df: pd.DataFrame, sku_id: str) -> pd.DataFrame:
    return df[df["id"] == sku_id].sort_values("date").reset_index(drop=True)


# ===========================================================================
# Tests
# ===========================================================================


def test_output_has_all_expected_columns(features_per_sku):
    """Output contains every required feature column."""
    required = [
        "lag_1", "lag_7", "lag_14", "lag_28", "lag_365",
        "roll_mean_7", "roll_mean_28", "roll_std_7",
        "day_of_week", "week_of_year", "month", "year", "is_weekend",
        "is_event",
        "snap_flag",
        "sell_price", "price_roll_mean_28", "price_vs_mean",
    ]
    for col in required:
        assert col in features_per_sku.columns, f"Missing column: {col}"


def test_row_count_preserved(sales_long, features_per_sku):
    """Output has the same number of rows as the input."""
    assert len(features_per_sku) == len(sales_long)


def test_lag_1_correct_shift(features_per_sku):
    """lag_1 at position N equals sales at position N-1 within the same SKU."""
    sku = _sku_rows(features_per_sku, "FOODS_3_090_CA_1")
    # First row has no prior day — must be NaN
    assert pd.isna(sku["lag_1"].iloc[0])
    # Row 1: lag_1 = sales from row 0
    assert sku["lag_1"].iloc[1] == pytest.approx(sku["sales"].iloc[0])
    # Row 10: lag_1 = sales from row 9
    assert sku["lag_1"].iloc[10] == pytest.approx(sku["sales"].iloc[9])


def test_lag_7_correct_shift(features_per_sku):
    """lag_7 at position N equals sales at position N-7 within the same SKU."""
    sku = _sku_rows(features_per_sku, "FOODS_3_090_CA_1")
    assert pd.isna(sku["lag_7"].iloc[0])
    # Row 7: lag_7 = sales from row 0
    assert sku["lag_7"].iloc[7] == pytest.approx(sku["sales"].iloc[0])
    # Row 50: lag_7 = sales from row 43
    assert sku["lag_7"].iloc[50] == pytest.approx(sku["sales"].iloc[43])


def test_rolling_mean_excludes_current_day(features_per_sku):
    """roll_mean_7 at position 7 equals mean(sales[0:7]) — current day excluded.

    After shift(1), position 7 in the shifted series holds sales[6]. The
    rolling(7) window at position 7 covers shifted positions 1..7, which map
    back to original sales[0]..sales[6].
    """
    sku = _sku_rows(features_per_sku, "FOODS_3_090_CA_1")
    expected = sku["sales"].iloc[0:7].mean()
    assert sku["roll_mean_7"].iloc[7] == pytest.approx(expected)


def test_nan_rows_preserved(features_per_sku):
    """Rows with NaN lag/rolling values are retained — no silent dropping."""
    sku = _sku_rows(features_per_sku, "FOODS_3_090_CA_1")
    # First row of each SKU has no prior history
    assert pd.isna(sku["lag_1"].iloc[0])
    assert pd.isna(sku["roll_mean_7"].iloc[0])
    # lag_365 requires 365 prior days; first 365 rows must be NaN, rest non-NaN
    assert sku["lag_365"].iloc[:365].isna().all()
    assert sku["lag_365"].iloc[365:].notna().all()


def test_snap_flag_ca(features_per_sku, calendar):
    """snap_flag is 1 for CA-state rows on snap_CA=1 days, 0 otherwise."""
    snap_on_dates = set(calendar.loc[calendar["snap_CA"] == 1, "date"])
    snap_off_dates = set(calendar.loc[calendar["snap_CA"] == 0, "date"])

    ca_snap_on = features_per_sku[
        (features_per_sku["state_id"] == "CA")
        & (features_per_sku["date"].isin(snap_on_dates))
    ]
    ca_snap_off = features_per_sku[
        (features_per_sku["state_id"] == "CA")
        & (features_per_sku["date"].isin(snap_off_dates))
    ]

    assert len(ca_snap_on) > 0, "Fixture has no snap_CA=1 dates — increase N_DAYS"
    assert (ca_snap_on["snap_flag"] == 1).all()
    assert (ca_snap_off["snap_flag"] == 0).all()


def test_snap_flag_uses_state_specific_column(features_per_sku, calendar):
    """A day where snap_CA=1 and snap_TX=0 gives flag=1 for CA, flag=0 for TX."""
    mixed = calendar[(calendar["snap_CA"] == 1) & (calendar["snap_TX"] == 0)]
    if mixed.empty:
        pytest.skip("No date with snap_CA=1 and snap_TX=0 in synthetic data")

    date = mixed["date"].iloc[0]
    ca_row = features_per_sku[
        (features_per_sku["state_id"] == "CA") & (features_per_sku["date"] == date)
    ].iloc[0]
    tx_row = features_per_sku[
        (features_per_sku["state_id"] == "TX") & (features_per_sku["date"] == date)
    ].iloc[0]

    assert ca_row["snap_flag"] == 1
    assert tx_row["snap_flag"] == 0


def test_price_join_works(features_per_sku):
    """sell_price is populated from the prices table; price_vs_mean follows."""
    assert features_per_sku["sell_price"].notna().any()
    # Where sell_price is present, price_vs_mean must also be computable
    valid = features_per_sku[features_per_sku["sell_price"].notna()]
    assert valid["price_vs_mean"].notna().all()


def test_global_has_id_columns(features_global):
    """build_features_global output retains all categorical ID columns."""
    for col in ["cat_id", "dept_id", "store_id", "state_id"]:
        assert col in features_global.columns, f"Missing ID column: {col}"


def test_global_per_sku_same_rows(features_per_sku, features_global):
    """build_features_global and build_features_per_sku produce the same row count."""
    assert len(features_global) == len(features_per_sku)


def test_calendar_features_correct(features_per_sku):
    """day_of_week, is_weekend, and month are derived correctly from the date.

    2011-01-29 (START_DATE) is a Saturday: day_of_week=5, is_weekend=True, month=1.
    2011-01-31 is a Monday: day_of_week=0, is_weekend=False.
    """
    sat_rows = features_per_sku[features_per_sku["date"] == pd.Timestamp("2011-01-29")]
    row_sat = sat_rows.iloc[0]
    assert row_sat["day_of_week"] == 5
    assert row_sat["is_weekend"]
    assert row_sat["month"] == 1

    mon_rows = features_per_sku[features_per_sku["date"] == pd.Timestamp("2011-01-31")]
    row_mon = mon_rows.iloc[0]
    assert row_mon["day_of_week"] == 0
    assert not row_mon["is_weekend"]


def test_is_event_correct(features_per_sku, calendar):
    """is_event is 1 where event_name_1 is not null, 0 otherwise."""
    event_dates = set(calendar.loc[calendar["event_name_1"].notna(), "date"])
    non_event_dates = set(calendar.loc[calendar["event_name_1"].isna(), "date"])

    event_rows = features_per_sku[features_per_sku["date"].isin(event_dates)]
    non_event_rows = features_per_sku[features_per_sku["date"].isin(non_event_dates)]

    assert len(event_rows) > 0, "Fixture has no event dates — increase event probability"
    assert (event_rows["is_event"] == 1).all()
    assert (non_event_rows["is_event"] == 0).all()
