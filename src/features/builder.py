"""Feature engineering for LightGBM forecasting models.

Two public entry points:
- build_features_per_sku: lag, rolling, calendar, event, SNAP, and price features
- build_features_global: same as per_sku; categorical ID columns pass through automatically
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_features_per_sku(
    sales_long: pd.DataFrame,
    calendar: pd.DataFrame,
    prices: pd.DataFrame,
) -> pd.DataFrame:
    """Add lag, rolling, calendar, event, SNAP, and price features to a long-format sales frame.

    Args:
        sales_long: Long-format sales dataframe. Required columns:
            ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'date', 'sales'].
            'date' must be datetime dtype. One row per (SKU, date).
        calendar: M5 calendar dataframe (from load_calendar()). Required columns include
            ['date', 'wm_yr_wk', 'event_name_1', 'snap_CA', 'snap_TX', 'snap_WI'].
        prices: M5 sell_prices dataframe (from load_prices()). Required columns:
            ['store_id', 'item_id', 'wm_yr_wk', 'sell_price'].

    Returns:
        DataFrame with the same rows as sales_long plus these additional columns:

        Lag features (sales shifted N days within each SKU's series):
            lag_1, lag_7, lag_14, lag_28, lag_365

        Rolling stats (shift(1) applied first — current day excluded from window):
            roll_mean_7, roll_mean_28, roll_std_7

        Calendar:
            day_of_week (0=Mon..6=Sun), week_of_year, month, year, is_weekend (bool)

        Event:
            is_event — 1 if event_name_1 is not NaN, else 0

        SNAP:
            snap_flag — 1 if the SKU's state has a SNAP day on that date

        Price:
            sell_price, price_roll_mean_28 (28-day rolling mean within item+store),
            price_vs_mean (sell_price / price_roll_mean_28)

    Note:
        NaN values from insufficient history are NOT filled or dropped. LightGBM
        handles NaN natively. All lag and rolling operations are grouped by 'id'
        to prevent mixing rows across different SKU-store combinations.
    """
    # Sort to guarantee temporal order within each SKU group before applying shifts
    df = sales_long.sort_values(["id", "date"]).copy()

    # --- Lag features ---
    # Default-argument trick (l=lag) avoids late-binding closure in the loop
    for lag in [1, 7, 14, 28, 365]:
        df[f"lag_{lag}"] = df.groupby("id")["sales"].transform(
            lambda x, l=lag: x.shift(l)
        )

    # --- Rolling stats: shift(1) first so the current day is excluded from every window ---
    df["roll_mean_7"] = df.groupby("id")["sales"].transform(
        lambda x: x.shift(1).rolling(7).mean()
    )
    df["roll_mean_28"] = df.groupby("id")["sales"].transform(
        lambda x: x.shift(1).rolling(28).mean()
    )
    df["roll_std_7"] = df.groupby("id")["sales"].transform(
        lambda x: x.shift(1).rolling(7).std()
    )

    # --- Calendar join (one row per date in the M5 calendar) ---
    cal_cols = ["date", "wm_yr_wk", "event_name_1", "snap_CA", "snap_TX", "snap_WI"]
    cal = calendar[cal_cols].drop_duplicates("date")
    df = df.merge(cal, on="date", how="left")

    # Derive calendar features directly from the date column
    df["day_of_week"] = df["date"].dt.dayofweek          # 0=Mon, 6=Sun
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    df["is_weekend"] = df["day_of_week"] >= 5

    # --- Event flag ---
    df["is_event"] = df["event_name_1"].notna().astype(int)

    # --- SNAP flag: select the column that matches each row's state ---
    df["snap_flag"] = np.select(
        [df["state_id"] == "CA", df["state_id"] == "TX", df["state_id"] == "WI"],
        [df["snap_CA"].values, df["snap_TX"].values, df["snap_WI"].values],
        default=0,
    ).astype(int)

    # --- Price join via wm_yr_wk (available after the calendar merge above) ---
    df = df.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")

    # Rolling mean of price uses min_periods=1 so a value is produced as soon as
    # the first price is available (prices are weekly, so early rows aren't NaN-starved)
    df["price_roll_mean_28"] = df.groupby("id")["sell_price"].transform(
        lambda x: x.rolling(28, min_periods=1).mean()
    )
    df["price_vs_mean"] = df["sell_price"] / df["price_roll_mean_28"]

    return df


def build_features_global(
    sales_long: pd.DataFrame,
    calendar: pd.DataFrame,
    prices: pd.DataFrame,
) -> pd.DataFrame:
    """Build features for the global LightGBM model (all SKUs trained jointly).

    Identical to build_features_per_sku. The categorical ID columns (cat_id, dept_id,
    store_id, state_id) already exist in sales_long and pass through automatically —
    no logic is duplicated here.

    Pass these columns to LightGBM via its ``categorical_feature`` parameter.
    Do NOT one-hot encode them; LightGBM handles categoricals natively.

    Args:
        sales_long: Long-format sales dataframe. Must contain all columns required by
            build_features_per_sku, including 'cat_id' and 'dept_id'.
        calendar: M5 calendar dataframe. Same requirements as build_features_per_sku.
        prices: M5 sell_prices dataframe. Same requirements as build_features_per_sku.

    Returns:
        Same output as build_features_per_sku, with cat_id, dept_id, store_id,
        and state_id guaranteed present (they originate in sales_long).
    """
    return build_features_per_sku(sales_long, calendar, prices)
