"""Phase 3 inventory optimization pipeline.

Loads Phase 2 CV results, derives forecast error statistics per SKU,
runs the inventory optimizer, prints a recommendations table, and saves
results to data/processed/phase3_inventory_recommendations.parquet.

Usage (from project root):
    uv run python -m scripts.run_phase3
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inventory.optimizer import InventoryParams, compute_inventory

# Best model per SKU (lowest MASE at horizon=28 from Phase 2 results)
BEST_MODEL: dict[str, str] = {
    "FOODS_3_090_CA_1": "LightGBM-per-SKU",
    "FOODS_3_586_CA_1": "LightGBM-per-SKU",
    "HOUSEHOLD_1_118_CA_1": "LightGBM-per-SKU",
    "HOBBIES_1_348_CA_1": "LightGBM-per-SKU",
    "HOUSEHOLD_1_430_CA_1": "Croston",
}

UNIT_COST: dict[str, float] = {
    "FOODS": 3.50,
    "HOUSEHOLD": 8.00,
    "HOBBIES": 12.00,
}

LEAD_TIME_DAYS = 7
SERVICE_LEVEL = 0.95
ORDERING_COST = 50.0        # fixed cost per order ($)
HOLDING_COST_RATE = 0.25    # 25% of unit cost per year


def load_forecast_std(cv_path: Path) -> dict[str, float]:
    """Mean RMSE at horizon=28 for the best model per SKU."""
    cv = pd.read_parquet(cv_path)
    h28 = cv[cv["horizon"] == 28]
    result: dict[str, float] = {}
    for sku, model in BEST_MODEL.items():
        mask = (h28["sku_store"] == sku) & (h28["model"] == model)
        mean_rmse = h28.loc[mask, "rmse"].mean()
        result[sku] = float(mean_rmse)
    return result


def load_mean_daily_demand(shortlist_path: Path) -> dict[str, tuple[float, str]]:
    """Returns {sku_store: (mean_daily, cat_id)}."""
    shortlist = pd.read_parquet(shortlist_path)
    return {
        row.sku_store: (row.mean_daily, row.cat_id)
        for row in shortlist.itertuples()
    }


def main() -> None:
    processed = PROJECT_ROOT / "data" / "processed"
    cv_path = processed / "phase2_cv_results.parquet"
    shortlist_path = processed / "eda_sku_shortlist.parquet"
    out_path = processed / "phase3_inventory_recommendations.parquet"

    print("Loading Phase 2 CV results...")
    forecast_std = load_forecast_std(cv_path)

    print("Loading SKU shortlist...")
    sku_info = load_mean_daily_demand(shortlist_path)

    records = []
    for sku, model in BEST_MODEL.items():
        mean_daily, cat_id = sku_info[sku]
        unit_cost = UNIT_COST[cat_id]
        holding_cost = unit_cost * HOLDING_COST_RATE

        params = InventoryParams(
            sku_store=sku,
            mean_daily_demand=mean_daily,
            forecast_std=forecast_std[sku],
            lead_time_days=LEAD_TIME_DAYS,
            service_level=SERVICE_LEVEL,
            ordering_cost=ORDERING_COST,
            holding_cost_per_unit=holding_cost,
            unit_cost=unit_cost,
        )
        rec = compute_inventory(params)
        records.append(
            {
                "sku_store": rec.sku_store,
                "best_model": model,
                "mean_daily_demand": mean_daily,
                "forecast_std": forecast_std[sku],
                "safety_stock": rec.safety_stock,
                "reorder_point": rec.reorder_point,
                "eoq": rec.eoq,
                "annual_orders": rec.annual_orders,
                "annual_holding_cost": rec.annual_holding_cost,
                "annual_ordering_cost": rec.annual_ordering_cost,
                "total_annual_cost": rec.total_annual_cost,
                "stockout_risk_pct": rec.stockout_risk_pct,
                "unit_cost": unit_cost,
                "holding_cost_per_unit": holding_cost,
                "lead_time_days": LEAD_TIME_DAYS,
                "service_level": SERVICE_LEVEL,
                # Fix : save ordering_cost directly so the dashboard
                # what-if can reconstruct exact InventoryParams without floating
                # point drift from back-calculating annual_ordering_cost / annual_orders
                "ordering_cost": ORDERING_COST,
            }
        )

    results_df = pd.DataFrame(records)

    # ── Print recommendations table ───────────────────────────────────────────
    print("\n" + "=" * 90)
    print("PHASE 3 - INVENTORY RECOMMENDATIONS  (service level 95%, lead time 7 days)")
    print("=" * 90)

    header = (
        f"{'SKU':<25} {'Mean Daily':>11} {'Safety Stk':>11} "
        f"{'Reorder Pt':>11} {'EOQ':>8} {'Annual Cost':>13}"
    )
    print(header)
    print("-" * 90)

    for row in results_df.itertuples():
        print(
            f"{row.sku_store:<25} {row.mean_daily_demand:>11.1f} "
            f"{row.safety_stock:>11.1f} {row.reorder_point:>11.1f} "
            f"{row.eoq:>8.1f} ${row.total_annual_cost:>12,.2f}"
        )

    print("=" * 90)
    print(
        "\nNotes:\n"
        "  Safety Stock  = z(service_level) x forecast_std x sqrt(lead_time)\n"
        "  Reorder Point = mean_daily x lead_time + safety_stock\n"
        "  EOQ           = sqrt(2 x annual_demand x ordering_cost / holding_cost)\n"
        "  Annual Cost   = holding cost + ordering cost (excludes purchase cost)\n"
        "  Forecast std  = mean RMSE at horizon=28 from best-model CV folds\n"
        f"  Ordering cost = ${ORDERING_COST:.2f} per order (fixed)\n"
        f"  Holding cost  = {HOLDING_COST_RATE:.0%} of unit cost per year"
    )

    # ── Save results ──────────────────────────────────────────────────────────
    results_df.to_parquet(out_path, index=False)
    print(f"\nResults saved -> {out_path.relative_to(PROJECT_ROOT)}")
    print(f"Columns: {list(results_df.columns)}")


if __name__ == "__main__":
    main()
