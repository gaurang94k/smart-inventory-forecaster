"""End-to-end Phase 2 pipeline.

Loads the 5 shortlist SKUs from Phase 1, runs all 6 models through
rolling-origin CV, and writes results to data/processed/phase2_cv_results.parquet.

Usage (from project root):
    uv run python -m scripts.run_phase2

Model assignments:
    All 5 SKUs:   SARIMA, Prophet, LightGBM-per-SKU, LightGBM-global
    Non-intermittent only (zero_rate < 0.5): Hybrid
    Intermittent only (HOUSEHOLD_1_430_CA_1): Croston

CV layout:
    n_folds=5, step=28, horizon=28 (expanding window)
    Metrics reported at sub-horizons 7, 14, 28.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Resolve project root so this module can be run with `python -m scripts.run_phase2`
# from the project root without installing the package.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import load_calendar, load_prices, load_sales
from src.evaluation.cv import rolling_origin_splits
from src.evaluation.metrics import mape, mase, rmse
from src.features.builder import build_features_global, build_features_per_sku
from src.models.croston import CrostonForecaster
from src.models.hybrid import HybridForecaster
from src.models.lgbm import GlobalLightGBMForecaster, LightGBMForecaster
from src.models.prophet import ProphetForecaster
from src.models.sarima import SarimaForecaster

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_FOLDS = 5
HORIZON = 28
STEP = 28
REPORT_HORIZONS = [7, 14, 28]
INTERMITTENT_THRESHOLD = 0.5   # zero_rate >= this → use Croston, skip Hybrid/MAPE

OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "phase2_cv_results.parquet"
SHORTLIST_PATH = PROJECT_ROOT / "data" / "processed" / "eda_sku_shortlist.parquet"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_sku_store(sku_store: str) -> tuple[str, str]:
    """Split 'FOODS_3_090_CA_1' → ('FOODS_3_090', 'CA_1').

    M5 store IDs always follow '{STATE_2CHAR}_{DIGIT}', so the last two
    underscore-delimited tokens form the store_id.
    """
    parts = sku_store.split("_")
    store_id = "_".join(parts[-2:])
    item_id = "_".join(parts[:-2])
    return item_id, store_id


def _build_sku_long(
    sales_raw: pd.DataFrame,
    calendar: pd.DataFrame,
    sku_store: str,
) -> pd.DataFrame:
    """Extract one SKU from the M5 wide-format sales and melt to long format.

    Returns a DataFrame with columns: id, item_id, dept_id, cat_id, store_id,
    state_id, date, sales.  The 'id' column is set to sku_store (not the M5
    '_evaluation' suffixed value) so it's consistent with our canonical
    identifiers throughout the pipeline.
    """
    item_id, store_id = _parse_sku_store(sku_store)
    mask = (sales_raw["item_id"] == item_id) & (sales_raw["store_id"] == store_id)
    sku_wide = sales_raw[mask]
    if sku_wide.empty:
        raise ValueError(
            f"SKU {sku_store!r} (item_id={item_id!r}, store_id={store_id!r}) "
            "not found in sales data."
        )

    id_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    day_cols = [c for c in sku_wide.columns if c.startswith("d_") and c[2:].isdigit()]

    long = sku_wide.melt(
        id_vars=id_cols,
        value_vars=day_cols,
        var_name="d",
        value_name="sales",
    )
    long = long.merge(calendar[["d", "date"]], on="d", how="left")
    long = long.drop(columns=["d"])
    long["date"] = pd.to_datetime(long["date"])
    long["id"] = sku_store  # canonical id — no "_evaluation" suffix
    return long.sort_values("date").reset_index(drop=True)


def _safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    try:
        return mape(y_true, y_pred)
    except ValueError:
        return None


def _safe_mase(
    y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray
) -> float | None:
    try:
        return mase(y_true, y_pred, y_train)
    except ValueError:
        return None


def _record_fold(
    results: list[dict],
    sku_store: str,
    model_name: str,
    fold_idx: int,
    fc: pd.DataFrame,
    test_sorted: pd.DataFrame,
    y_train: np.ndarray,
    compute_mape: bool,
) -> None:
    """Append one row per sub-horizon into results.

    Both forecast_dates and forecast_values are sliced to [:h] so the two
    lists are always the same length in the parquet. (Bug fix: previously
    forecast_values was not sliced, producing 28 values for every sub-horizon
    row including h=7 and h=14.)
    """
    for h in REPORT_HORIZONS:
        fc_vals = fc["forecast"].to_numpy()[:h]
        actual_vals = test_sorted["sales"].to_numpy()[:h]

        results.append({
            "sku_store": sku_store,
            "model": model_name,
            "fold": fold_idx + 1,
            "horizon": h,
            "mape": _safe_mape(actual_vals, fc_vals) if compute_mape else None,
            "rmse": rmse(actual_vals, fc_vals),
            "mase": _safe_mase(actual_vals, fc_vals, y_train),
            # Fix: slice forecast_values to [:h] to match forecast_dates[:h]
            "forecast_dates": fc["date"].tolist()[:h],
            "forecast_values": fc_vals.tolist(),  # fc_vals already [:h] from line above
        })


# ---------------------------------------------------------------------------
# Per-SKU CV
# ---------------------------------------------------------------------------

def run_per_sku_cv(
    sku_store: str,
    zero_rate: float,
    feat_df: pd.DataFrame,
    results: list[dict],
) -> None:
    """Run all applicable per-SKU models through CV for one SKU."""
    intermittent = zero_rate >= INTERMITTENT_THRESHOLD
    compute_mape = not intermittent

    # Model registry: name → factory.  Factories are lambdas so each fold
    # gets a fresh, unfitted instance.
    per_sku_models: dict[str, object] = {
        "SARIMA": lambda: SarimaForecaster(),
        "Prophet": lambda: ProphetForecaster(),
        "LightGBM-per-SKU": lambda: LightGBMForecaster(n_estimators=500),
    }
    if intermittent:
        per_sku_models["Croston"] = lambda: CrostonForecaster()
    else:
        per_sku_models["Hybrid"] = lambda: HybridForecaster()

    # Models that need only date + sales (no feature columns)
    simple_fit_models = {"SARIMA", "Prophet", "Croston"}

    folds = list(rolling_origin_splits(feat_df, n_folds=N_FOLDS, horizon=HORIZON, step=STEP))

    for model_name, factory in per_sku_models.items():
        t0 = time.time()
        for fold_idx, (train, test) in enumerate(folds):
            model = factory()

            fit_df = train[["date", "sales"]] if model_name in simple_fit_models else train
            model.fit(fit_df)
            fc = model.forecast(HORIZON)

            y_train = train.sort_values("date")["sales"].to_numpy(dtype=float)
            test_sorted = test.sort_values("date")

            _record_fold(results, sku_store, model_name, fold_idx, fc, test_sorted, y_train, compute_mape)

        elapsed = time.time() - t0
        print(f"    {model_name:<22} {N_FOLDS} folds  ({elapsed:.1f}s)")


# ---------------------------------------------------------------------------
# Global LightGBM CV
# ---------------------------------------------------------------------------

def run_global_cv(
    shortlist: pd.DataFrame,
    feat_dfs: dict[str, pd.DataFrame],
    results: list[dict],
) -> None:
    """Train GlobalLightGBMForecaster jointly on all 5 SKUs, per fold."""
    print("  LightGBM-global")

    # Use the first SKU's DF to define fold boundaries (all share the same dates)
    anchor_sku = shortlist["sku_store"].iloc[0]
    folds = list(
        rolling_origin_splits(feat_dfs[anchor_sku], n_folds=N_FOLDS, horizon=HORIZON, step=STEP)
    )

    # Pre-compute fold date boundaries from the anchor SKU
    fold_boundaries = [(train["date"].max(), test["date"].max()) for train, test in folds]

    t0 = time.time()
    for fold_idx, (train_end, test_end) in enumerate(fold_boundaries):
        # Concatenate all SKUs' training slices for this fold
        train_slices = []
        for sku_store in shortlist["sku_store"]:
            df = feat_dfs[sku_store]
            train_slices.append(df[df["date"] <= train_end])
        global_train = pd.concat(train_slices, ignore_index=True)

        model = GlobalLightGBMForecaster(n_estimators=500)
        model.fit(global_train)

        # Forecast and score per SKU
        for _, row in shortlist.iterrows():
            sku_store = row["sku_store"]
            zero_rate = row["zero_rate"]
            compute_mape = zero_rate < INTERMITTENT_THRESHOLD

            df = feat_dfs[sku_store]
            test = df[(df["date"] > train_end) & (df["date"] <= test_end)]
            test_sorted = test.sort_values("date")

            y_train = df[df["date"] <= train_end].sort_values("date")["sales"].to_numpy(dtype=float)

            fc = model.forecast(HORIZON, sku_id=sku_store)
            _record_fold(results, sku_store, "LightGBM-global", fold_idx, fc, test_sorted, y_train, compute_mape)

    elapsed = time.time() - t0
    print(f"    {'LightGBM-global':<22} {N_FOLDS} folds  ({elapsed:.1f}s)")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _print_summary(results_df: pd.DataFrame) -> None:
    """Print mean MAPE and RMSE per model across all SKUs and folds (horizon=28)."""
    h28 = results_df[results_df["horizon"] == 28].copy()
    summary = (
        h28.groupby("model")[["mape", "rmse", "mase"]]
        .mean(numeric_only=True)
        .round(3)
        .sort_values("rmse")
    )
    print("\n" + "=" * 60)
    print("Phase 2 CV Summary — mean metrics at horizon 28")
    print("(MAPE excludes intermittent SKU; MASE covers all)")
    print("=" * 60)
    print(summary.to_string())
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    total_t0 = time.time()
    print("=" * 60)
    print("Phase 2 pipeline starting")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load shortlist
    # ------------------------------------------------------------------
    print("\n[1/4] Loading shortlist and raw M5 data...")
    shortlist = pd.read_parquet(SHORTLIST_PATH)
    print(f"  Shortlist: {len(shortlist)} SKUs")
    for _, r in shortlist.iterrows():
        print(f"    {r['sku_store']:<28}  zero_rate={r['zero_rate']:.0%}  mean_daily={r['mean_daily']:.1f}")

    sales_raw = load_sales()
    calendar = load_calendar()
    prices = load_prices()
    print(f"  Raw sales: {len(sales_raw):,} rows  |  calendar: {len(calendar)} days  |  prices: {len(prices):,} rows")

    # ------------------------------------------------------------------
    # 2. Build feature-engineered DataFrames for all 5 SKUs
    # ------------------------------------------------------------------
    print("\n[2/4] Building feature-engineered DataFrames...")
    feat_dfs: dict[str, pd.DataFrame] = {}
    for _, row in shortlist.iterrows():
        sku_store = row["sku_store"]
        long_df = _build_sku_long(sales_raw, calendar, sku_store)
        feat_df = build_features_per_sku(long_df, calendar, prices)
        feat_dfs[sku_store] = feat_df
        print(f"  {sku_store:<28}  {len(feat_df)} rows, {feat_df['date'].min().date()} → {feat_df['date'].max().date()}")

    # ------------------------------------------------------------------
    # 3. Rolling-origin CV
    # ------------------------------------------------------------------
    print(f"\n[3/4] Running CV  (n_folds={N_FOLDS}, step={STEP}, horizon={HORIZON})...")
    results: list[dict] = []

    # Per-SKU models
    for _, row in shortlist.iterrows():
        sku_store = row["sku_store"]
        zero_rate = row["zero_rate"]
        print(f"\n  SKU: {sku_store}  (zero_rate={zero_rate:.0%})")
        run_per_sku_cv(sku_store, zero_rate, feat_dfs[sku_store], results)

    # Global LightGBM
    print(f"\n  SKU: all 5 (global model)")
    run_global_cv(shortlist, feat_dfs, results)

    # ------------------------------------------------------------------
    # 4. Save results
    # ------------------------------------------------------------------
    print("\n[4/4] Saving results...")
    results_df = pd.DataFrame(results)
    results_df.to_parquet(OUTPUT_PATH, index=False)
    print(f"  Saved {len(results_df)} rows → {OUTPUT_PATH}")
    print(f"  Columns: {list(results_df.columns)}")

    _print_summary(results_df)

    total_elapsed = time.time() - total_t0
    print(f"\nTotal runtime: {total_elapsed / 60:.1f} min")


if __name__ == "__main__":
    main()
