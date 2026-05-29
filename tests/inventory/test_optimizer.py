import pytest
from src.inventory.optimizer import (
    InventoryParams,
    InventoryRecommendation,
    compute_inventory,
)


def base_params(**overrides) -> InventoryParams:
    defaults = dict(
        sku_store="TEST_SKU_CA_1",
        mean_daily_demand=20.0,
        forecast_std=5.0,
        lead_time_days=7,
        service_level=0.95,
        ordering_cost=50.0,
        holding_cost_per_unit=2.0,
        unit_cost=10.0,
    )
    defaults.update(overrides)
    return InventoryParams(**defaults)


def test_compute_inventory_returns_recommendation():
    result = compute_inventory(base_params())
    assert isinstance(result, InventoryRecommendation)


def test_safety_stock_increases_with_higher_service_level():
    low = compute_inventory(base_params(service_level=0.90))
    high = compute_inventory(base_params(service_level=0.99))
    assert high.safety_stock > low.safety_stock


def test_safety_stock_increases_with_higher_forecast_std():
    low = compute_inventory(base_params(forecast_std=3.0))
    high = compute_inventory(base_params(forecast_std=10.0))
    assert high.safety_stock > low.safety_stock


def test_reorder_point_greater_than_safety_stock():
    result = compute_inventory(base_params())
    assert result.reorder_point > result.safety_stock


def test_eoq_decreases_as_holding_cost_increases():
    low_hold = compute_inventory(base_params(holding_cost_per_unit=1.0))
    high_hold = compute_inventory(base_params(holding_cost_per_unit=5.0))
    assert low_hold.eoq > high_hold.eoq


def test_eoq_increases_as_ordering_cost_increases():
    low_order = compute_inventory(base_params(ordering_cost=20.0))
    high_order = compute_inventory(base_params(ordering_cost=200.0))
    assert high_order.eoq > low_order.eoq


def test_total_annual_cost_equals_holding_plus_ordering():
    result = compute_inventory(base_params())
    expected = round(result.annual_holding_cost + result.annual_ordering_cost, 2)
    assert result.total_annual_cost == expected


def test_stockout_risk_pct():
    result = compute_inventory(base_params(service_level=0.95))
    assert result.stockout_risk_pct == pytest.approx(5.0, abs=1e-6)


def test_stockout_risk_pct_90():
    result = compute_inventory(base_params(service_level=0.90))
    assert result.stockout_risk_pct == pytest.approx(10.0, abs=1e-6)


def test_sku_store_propagated():
    result = compute_inventory(base_params(sku_store="FOODS_3_090_CA_1"))
    assert result.sku_store == "FOODS_3_090_CA_1"
