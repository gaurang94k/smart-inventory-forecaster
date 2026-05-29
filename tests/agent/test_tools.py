import pytest
from src.agent.tools import (
    get_forecast_summary,
    get_inventory_recommendation,
    list_skus,
    run_service_level_scenario,
)

_VALID_SKU = "FOODS_3_090_CA_1"
_ALL_SKUS = [
    "FOODS_3_090_CA_1",
    "FOODS_3_586_CA_1",
    "HOUSEHOLD_1_118_CA_1",
    "HOBBIES_1_348_CA_1",
    "HOUSEHOLD_1_430_CA_1",
]


# ── list_skus ──────────────────────────────────────────────────────────────────

def test_list_skus_returns_five_strings():
    result = list_skus()
    assert isinstance(result, list)
    assert len(result) == 5
    assert all(isinstance(s, str) for s in result)


def test_list_skus_contains_expected_ids():
    result = list_skus()
    for sku in _ALL_SKUS:
        assert sku in result


# ── get_forecast_summary ───────────────────────────────────────────────────────

def test_get_forecast_summary_returns_required_keys():
    result = get_forecast_summary(_VALID_SKU)
    for key in ("sku_store", "best_model", "mape", "rmse", "mase"):
        assert key in result


def test_get_forecast_summary_metric_values_are_floats():
    result = get_forecast_summary(_VALID_SKU)
    assert isinstance(result["mape"], float)
    assert isinstance(result["rmse"], float)
    assert isinstance(result["mase"], float)


def test_get_forecast_summary_metrics_are_positive():
    result = get_forecast_summary(_VALID_SKU)
    assert result["rmse"] > 0
    assert result["mase"] > 0


def test_get_forecast_summary_raises_for_unknown_sku():
    with pytest.raises(ValueError, match="Unknown SKU"):
        get_forecast_summary("DOES_NOT_EXIST_CA_1")


def test_get_forecast_summary_best_model_is_string():
    result = get_forecast_summary(_VALID_SKU)
    assert isinstance(result["best_model"], str)
    assert len(result["best_model"]) > 0


# ── get_inventory_recommendation ──────────────────────────────────────────────

def test_get_inventory_recommendation_returns_dict_with_safety_stock():
    result = get_inventory_recommendation(_VALID_SKU)
    assert isinstance(result, dict)
    assert "safety_stock" in result


def test_get_inventory_recommendation_contains_all_key_fields():
    result = get_inventory_recommendation(_VALID_SKU)
    for key in ("sku_store", "safety_stock", "reorder_point", "eoq", "total_annual_cost"):
        assert key in result


def test_get_inventory_recommendation_raises_for_unknown_sku():
    with pytest.raises(ValueError):
        get_inventory_recommendation("FAKE_SKU_CA_1")


# ── run_service_level_scenario ────────────────────────────────────────────────

def test_run_service_level_scenario_returns_delta_keys():
    result = run_service_level_scenario(_VALID_SKU, 0.99)
    assert "safety_stock_delta" in result
    assert "total_cost_delta" in result


def test_run_service_level_scenario_higher_sl_increases_safety_stock():
    result_99 = run_service_level_scenario(_VALID_SKU, 0.99)
    result_80 = run_service_level_scenario(_VALID_SKU, 0.80)
    assert result_99["safety_stock"] > result_80["safety_stock"]


def test_run_service_level_scenario_delta_sign_is_correct():
    # 99% > 95% baseline => positive delta
    result = run_service_level_scenario(_VALID_SKU, 0.99)
    assert result["safety_stock_delta"] > 0


def test_run_service_level_scenario_service_level_stored_in_result():
    sl = 0.90
    result = run_service_level_scenario(_VALID_SKU, sl)
    assert result["service_level"] == sl
