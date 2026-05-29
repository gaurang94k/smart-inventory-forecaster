from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.inventory.optimizer import InventoryParams, compute_inventory

_PROCESSED = Path(__file__).resolve().parents[2] / "data" / "processed"

# Document why LightGBM-per-SKU is chosen as best model.
# Selection criterion: lowest RMSE at horizon=28, averaged across all CV folds.
# LightGBM-per-SKU achieves RMSE=9.28 (best among continuous-SKU models).
# SARIMA achieves the best MASE (0.775) due to its explicit weekly-seasonal
# structure matching the MASE denominator, but RMSE is used here because it
# directly informs the safety-stock formula (forecast_std = mean RMSE at h=28).
# Croston is assigned to HOUSEHOLD_1_430_CA_1 on methodological grounds: it is
# the only model designed for intermittent demand. SARIMA achieves lower RMSE
# (1.518 vs 1.756) on this SKU's backtest, but would produce negative forecasts
# on future zero-heavy windows where Croston's flat demand-rate estimate is
# more appropriate.
_BEST_MODEL: dict[str, str] = {
    "FOODS_3_090_CA_1": "LightGBM-per-SKU",
    "FOODS_3_586_CA_1": "LightGBM-per-SKU",
    "HOUSEHOLD_1_118_CA_1": "LightGBM-per-SKU",
    "HOBBIES_1_348_CA_1": "LightGBM-per-SKU",
    "HOUSEHOLD_1_430_CA_1": "Croston",
}


def list_skus() -> list[str]:
    """Return the 5 SKU-store IDs in the Phase 2 shortlist."""
    shortlist = pd.read_parquet(_PROCESSED / "eda_sku_shortlist.parquet")
    return shortlist["sku_store"].tolist()


def get_forecast_summary(sku_store: str) -> dict:
    """Return mean MAPE, RMSE, MASE at horizon=28 for the best model for a SKU.

    MAPE is undefined when the test window contains zero-sales days
    (division by zero). For affected SKUs, mape is returned as null and
    mape_note explains why. RMSE and MASE are always available and are the
    primary accuracy metrics for those SKUs.

    Raises ValueError if the SKU is not in the shortlist.
    """
    cv = pd.read_parquet(_PROCESSED / "phase2_cv_results.parquet")
    known = list(_BEST_MODEL.keys())
    if sku_store not in known:
        raise ValueError(
            f"Unknown SKU '{sku_store}'. Valid options: {known}"
        )

    best_model = _BEST_MODEL[sku_store]
    h28 = cv[
        (cv["sku_store"] == sku_store)
        & (cv["model"] == best_model)
        & (cv["horizon"] == 28)
    ]

    raw_mape = h28["mape"].mean()

    # detect NaN MAPE and return an informative note instead of
    # silently returning null, which would confuse the LLM.
    if pd.isna(raw_mape):
        mape_value = None
        mape_note = (
            "MAPE is undefined for this SKU because the test windows contain "
            "zero-sales days (division by zero). Use RMSE and MASE instead."
        )
    else:
        mape_value = round(float(raw_mape), 4)
        mape_note = None

    return {
        "sku_store": sku_store,
        "best_model": best_model,
        "mape": mape_value,
        "mape_note": mape_note,
        "rmse": round(float(h28["rmse"].mean()), 4),
        "mase": round(float(h28["mase"].mean()), 4),
    }


def get_inventory_recommendation(sku_store: str) -> dict:
    """Return the Phase 3 inventory recommendation for a SKU.

    Raises ValueError if the SKU is not found in the recommendations parquet.
    """
    recs = pd.read_parquet(_PROCESSED / "phase3_inventory_recommendations.parquet")
    row = recs[recs["sku_store"] == sku_store]
    if row.empty:
        raise ValueError(
            f"No inventory recommendation found for '{sku_store}'."
        )
    return row.iloc[0].to_dict()


def run_service_level_scenario(sku_store: str, service_level: float) -> dict:
    """Re-run the inventory optimizer for a SKU at a different service level.

    Returns the new recommendation plus deltas vs the stored 95% baseline.
    Uses the ordering_cost column directly from the parquet when available
    (added in run_phase3.py) to avoid floating-point drift from
    back-calculating annual_ordering_cost / annual_orders.
    """
    recs = pd.read_parquet(_PROCESSED / "phase3_inventory_recommendations.parquet")
    row = recs[recs["sku_store"] == sku_store]
    if row.empty:
        raise ValueError(
            f"No inventory recommendation found for '{sku_store}'."
        )
    base = row.iloc[0]

    # Prefer stored ordering_cost; fall back to back-calculation for
    # parquets generated before this column was added.
    if "ordering_cost" in base.index and pd.notna(base["ordering_cost"]):
        ordering_cost = float(base["ordering_cost"])
    else:
        ordering_cost = float(base["annual_ordering_cost"] / base["annual_orders"])

    params = InventoryParams(
        sku_store=sku_store,
        mean_daily_demand=float(base["mean_daily_demand"]),
        forecast_std=float(base["forecast_std"]),
        lead_time_days=int(base["lead_time_days"]),
        service_level=service_level,
        ordering_cost=ordering_cost,
        holding_cost_per_unit=float(base["holding_cost_per_unit"]),
        unit_cost=float(base["unit_cost"]),
    )
    new = compute_inventory(params)

    return {
        "sku_store": new.sku_store,
        "service_level": service_level,
        "safety_stock": new.safety_stock,
        "reorder_point": new.reorder_point,
        "eoq": new.eoq,
        "annual_orders": new.annual_orders,
        "total_annual_cost": new.total_annual_cost,
        "stockout_risk_pct": new.stockout_risk_pct,
        "safety_stock_delta": round(new.safety_stock - float(base["safety_stock"]), 2),
        "reorder_point_delta": round(new.reorder_point - float(base["reorder_point"]), 2),
        "total_cost_delta": round(new.total_annual_cost - float(base["total_annual_cost"]), 2),
    }
