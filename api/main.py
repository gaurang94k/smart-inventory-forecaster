"""FastAPI backend for Smart Inventory Forecaster.

Exposes forecasting, inventory optimisation, and LLM agent logic
as REST endpoints for the standalone HTML frontend.

Usage (from project root):
    uv run uvicorn api.main:app --reload --port 8000

Endpoints:
    GET  /skus                   — list all 5 SKU-store IDs
    GET  /forecast/{sku_store}   — CV metrics + fold chart data with actuals
    GET  /inventory              — baseline recommendations at 95% SL
    POST /inventory/whatif       — recalculate at a different service level
    POST /agent/query            — run the LLM planning agent
    GET  /health                 — health check
"""

from __future__ import annotations

import math
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import anthropic
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# ── Path setup ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.agent import run_agent  # noqa: E402
from src.inventory.optimizer import InventoryParams, compute_inventory  # noqa: E402

PROCESSED = PROJECT_ROOT / "data" / "processed"
FRONTEND = PROJECT_ROOT / "frontend"

# ── In-memory data store (loaded once at startup) ─────────────────────────────
_cv: pd.DataFrame | None = None
_recs: pd.DataFrame | None = None
_shortlist: pd.DataFrame | None = None
_eda: pd.DataFrame | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _cv, _recs, _shortlist, _eda
    _cv = pd.read_parquet(PROCESSED / "phase2_cv_results.parquet")
    _recs = pd.read_parquet(PROCESSED / "phase3_inventory_recommendations.parquet")
    _shortlist = pd.read_parquet(PROCESSED / "eda_sku_shortlist.parquet")
    sku_ids = [f"{s}_evaluation" for s in _shortlist["sku_store"]]
    _eda = pd.read_parquet(
        PROCESSED / "eda_sample_long_CA1.parquet",
        columns=["id", "date", "sales"],
        filters=[("id", "in", sku_ids)],
    )
    _eda["date"] = pd.to_datetime(_eda["date"])
    print(f"Loaded: {len(_cv)} CV rows | {len(_recs)} SKUs | {len(_eda)} EDA rows (filtered)")
    yield


# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Smart Inventory Forecaster API",
    description="Demand forecasting and inventory optimisation for 5 Walmart M5 SKUs.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

if FRONTEND.exists():
    app.mount("/app", StaticFiles(directory=str(FRONTEND), html=True), name="frontend")


# ── Request / Response models ──────────────────────────────────────────────────
class WhatIfRequest(BaseModel):
    service_level: float = Field(
        ..., ge=0.80, le=0.99,
        description="Target service level as a decimal (0.80–0.99)",
    )


class AgentRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)


# ── Helpers ────────────────────────────────────────────────────────────────────
def _sku_list() -> list[str]:
    return _shortlist["sku_store"].tolist()


def _require_sku(sku_store: str) -> None:
    if sku_store not in _sku_list():
        raise HTTPException(
            status_code=404,
            detail=f"SKU '{sku_store}' not found. Valid: {_sku_list()}",
        )


def _safe_float(v) -> float | None:
    """Convert v to float, returning None for NaN / inf / non-numeric values.

    Fixes: ValueError: Out of range float values are not JSON compliant
    which is triggered when pandas NaN or numpy inf values reach json.dumps().
    Croston rows have NaN MAPE (intermittent demand — MAPE undefined when
    actuals contain zeros), which is the main source of this error.
    """
    try:
        f = float(v)
        return None if not math.isfinite(f) else f
    except (TypeError, ValueError):
        return None


def _clean_record(d: dict) -> dict:
    """Recursively sanitise a dict so all float values are JSON-safe."""
    out = {}
    for k, v in d.items():
        if isinstance(v, float):
            out[k] = _safe_float(v)
        elif isinstance(v, list):
            out[k] = [_safe_float(x) if isinstance(x, float) else x for x in v]
        else:
            out[k] = v
    return out


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
def root():
    index = FRONTEND / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return {"message": "Smart Inventory Forecaster API — see /docs for endpoints."}


@app.get("/skus", summary="List all SKU-store IDs")
def get_skus() -> list[str]:
    return _sku_list()


@app.get("/forecast/{sku_store}", summary="CV metrics and chart data for one SKU")
def get_forecast(sku_store: str):
    """Return model performance table and per-fold forecast vs actual data.

    Performance metrics are averaged across all folds and sub-horizons.
    Chart data is returned for horizon=28 only (all folds, best model).
    NaN / inf values are sanitised to null so the response is always valid
    JSON — this handles Croston's null MAPE for the intermittent SKU.
    """
    _require_sku(sku_store)

    sku_cv = _cv[_cv["sku_store"] == sku_store]

    # ── Performance table ──────────────────────────────────────────────────
    perf_raw = (
        sku_cv.groupby("model")[["mape", "rmse", "mase"]]
        .mean(numeric_only=True)
        .round(3)
        .sort_values("mase")
        .reset_index()
    )

    perf = []
    for _, row in perf_raw.iterrows():
        perf.append({
            "model": str(row["model"]),
            "mape": _safe_float(row["mape"]),   # None for Croston (NaN in parquet)
            "rmse": _safe_float(row["rmse"]),
            "mase": _safe_float(row["mase"]),
        })

    best_model = perf[0]["model"] if perf else None

    # ── Fold chart data (best model, h=28) ─────────────────────────────────
    h28 = sku_cv[
        (sku_cv["model"] == best_model) & (sku_cv["horizon"] == 28)
    ].copy()

    actuals_series = (
        _eda[_eda["id"] == f"{sku_store}_evaluation"]
        .set_index("date")["sales"]
    )

    folds = []
    for _, row in h28.iterrows():
        try:
            dates = pd.to_datetime(row["forecast_dates"])
            actual_vals = actuals_series.reindex(dates).fillna(0).tolist()
            folds.append({
                "fold": int(row["fold"]),
                "dates": [d.strftime("%Y-%m-%d") for d in dates],
                "forecast": [_safe_float(v) for v in row["forecast_values"]],
                "actual":   [_safe_float(v) for v in actual_vals],
            })
        except Exception:
            pass

    return {
        "sku_store": sku_store,
        "best_model": best_model,
        "performance": perf,
        "folds": sorted(folds, key=lambda x: x["fold"]),
    }


@app.get("/inventory", summary="Baseline inventory recommendations at 95% SL")
def get_inventory():
    keep = [
        "sku_store", "best_model", "mean_daily_demand", "forecast_std",
        "safety_stock", "reorder_point", "eoq", "annual_orders",
        "annual_holding_cost", "annual_ordering_cost", "total_annual_cost",
        "stockout_risk_pct", "unit_cost", "holding_cost_per_unit",
        "lead_time_days", "service_level",
    ]
    cols = [c for c in keep if c in _recs.columns]
    return [_clean_record(r) for r in _recs[cols].to_dict(orient="records")]


@app.post("/inventory/whatif", summary="Recalculate inventory at a new service level")
def inventory_whatif(body: WhatIfRequest):
    results = []
    for _, base in _recs.iterrows():
        if "ordering_cost" in base.index and pd.notna(base["ordering_cost"]):
            ordering_cost = float(base["ordering_cost"])
        else:
            ordering_cost = float(base["annual_ordering_cost"] / base["annual_orders"])

        params = InventoryParams(
            sku_store=base["sku_store"],
            mean_daily_demand=float(base["mean_daily_demand"]),
            forecast_std=float(base["forecast_std"]),
            lead_time_days=int(base["lead_time_days"]),
            service_level=body.service_level,
            ordering_cost=ordering_cost,
            holding_cost_per_unit=float(base["holding_cost_per_unit"]),
            unit_cost=float(base["unit_cost"]),
        )
        rec = compute_inventory(params)

        results.append(_clean_record({
            "sku_store": rec.sku_store,
            "service_level": body.service_level,
            "safety_stock": rec.safety_stock,
            "reorder_point": rec.reorder_point,
            "eoq": rec.eoq,
            "total_annual_cost": rec.total_annual_cost,
            "stockout_risk_pct": rec.stockout_risk_pct,
            "safety_stock_delta":   round(rec.safety_stock       - float(base["safety_stock"]),       2),
            "reorder_point_delta":  round(rec.reorder_point      - float(base["reorder_point"]),      2),
            "cost_delta":           round(rec.total_annual_cost  - float(base["total_annual_cost"]),  2),
        }))

    return results


@app.post("/agent/query", summary="Run the LLM inventory planning agent")
def agent_query(body: AgentRequest):
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise HTTPException(
            status_code=503,
            detail="ANTHROPIC_API_KEY environment variable is not set.",
        )
    try:
        answer = run_agent(body.question)
        return {"question": body.question, "answer": answer}
    except anthropic.APIStatusError as exc:
        if exc.status_code == 529:
            raise HTTPException(
                status_code=503,
                detail="AI service is temporarily overloaded. Please try again.",
            )
        raise HTTPException(
            status_code=502,
            detail=f"Anthropic API error ({exc.status_code}): {exc.message}",
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/health", summary="Health check")
def health():
    return {
        "status": "ok",
        "skus": _sku_list(),
        "cv_rows": len(_cv),
        "recs_rows": len(_recs),
        "eda_rows": len(_eda),
        "anthropic_key_set": bool(os.environ.get("ANTHROPIC_API_KEY")),
    }
