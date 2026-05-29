from __future__ import annotations

import math
from dataclasses import dataclass

from scipy.stats import norm

Z_SCORES = {0.90: 1.282, 0.95: 1.645, 0.99: 2.326}


@dataclass
class InventoryParams:
    sku_store: str
    mean_daily_demand: float
    forecast_std: float
    lead_time_days: int
    service_level: float
    ordering_cost: float
    holding_cost_per_unit: float
    unit_cost: float


@dataclass
class InventoryRecommendation:
    sku_store: str
    safety_stock: float
    reorder_point: float
    eoq: float
    annual_orders: float
    annual_holding_cost: float
    annual_ordering_cost: float
    total_annual_cost: float
    stockout_risk_pct: float


def compute_inventory(params: InventoryParams) -> InventoryRecommendation:
    z = norm.ppf(params.service_level)
    safety_stock = z * params.forecast_std * math.sqrt(params.lead_time_days)
    reorder_point = (params.mean_daily_demand * params.lead_time_days) + safety_stock

    annual_demand = params.mean_daily_demand * 365
    eoq = math.sqrt(
        (2 * annual_demand * params.ordering_cost) / params.holding_cost_per_unit
    )

    # EOQ depends only on demand and costs — not on service_level
    annual_orders = annual_demand / eoq
    annual_holding_cost = (eoq / 2 + safety_stock) * params.holding_cost_per_unit
    annual_ordering_cost = (annual_demand / eoq) * params.ordering_cost
    total_annual_cost = annual_holding_cost + annual_ordering_cost
    stockout_risk_pct = (1 - params.service_level) * 100

    return InventoryRecommendation(
        sku_store=params.sku_store,
        safety_stock=round(safety_stock, 2),
        reorder_point=round(reorder_point, 2),
        eoq=round(eoq, 2),
        annual_orders=round(annual_orders, 2),
        annual_holding_cost=round(annual_holding_cost, 2),
        annual_ordering_cost=round(annual_ordering_cost, 2),
        total_annual_cost=round(total_annual_cost, 2),
        stockout_risk_pct=round(stockout_risk_pct, 2),
    )
