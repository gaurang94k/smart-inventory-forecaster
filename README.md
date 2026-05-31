# smart-inventory-forecaster

[**Live demo**](https://smart-inventory-forecaster-9voq.onrender.com) - hosted on Render free tier, first load takes about 30 seconds to wake up.

Demand forecasting and inventory optimization for retail, built around a six-model comparison and wrapped in an LLM planning agent.

I started this after working on the floor of a Canadian drugstore chain. The inventory problems I saw repeat themselves were pretty predictable: empty shelves on promo weeks, slow movers sitting in the back for months, reorder decisions made by gut feel because the tools either didn't exist or weren't accessible to the people who actually needed them. This is my attempt to build something closer to what those teams should have had.

\---

## What it does

Three layers that build on each other.

The forecasting core runs six models against the same rolling-origin cross-validation pipeline: SARIMA, Prophet, LightGBM per-SKU, LightGBM global, a hybrid SARIMA+LightGBM model, and CrostonOptimized for intermittent demand. The goal isn't to pick one winner - each model makes a different bet about what structure exists in retail demand data, and running them on the same evaluation protocol shows those tradeoffs clearly.

The inventory layer takes forecast error and turns it into actual decisions. For each SKU it computes safety stock, reorder point, and economic order quantity, and outputs a recommendation with projected holding and ordering cost. Most forecasting projects stop at accuracy numbers. This one connects them to dollar figures, which is what a planner actually cares about.

The LLM agent is a Claude Haiku instance with four tools that answers inventory planning questions in plain English. It doesn't do the forecasting - it calls the right tools, reads the output, and explains the answer. Deterministic models do the math, the LLM handles the natural language part.

\---

## Results

Six-model comparison at horizon=28 days, mean across five rolling-origin CV folds:

|Model|MAPE|RMSE|MASE|
|-|-|-|-|
|LightGBM-per-SKU|40.8%|9.28|0.979|
|LightGBM-global|41.1%|9.14|1.100|
|SARIMA|49.6%|10.71|**0.775**|
|Prophet|53.2%|10.74|0.982|
|Hybrid|78.1%|14.15|0.959|
|Croston|-|1.756|1.154|

MAPE is left out for SKUs with zero-sales days since it's undefined there. Croston outputs a flat demand rate so MAPE doesn't really apply.

\---

## Things worth explaining in the results

**SARIMA has the best MASE but worse MAPE.** MASE normalizes by the seasonal naive forecast. SARIMA's (1,1,1)(1,1,0)\[7] spec is tuned directly to weekly retail seasonality, so it wins on that normalized measure even though LightGBM beats it on raw error. Which metric matters depends on what decision you're making.

**The hybrid model was the worst at 78% MAPE, and that's actually informative.** The idea was that SARIMA captures linear structure and LightGBM models the residuals. In practice SARIMA's residuals on short daily retail series are mostly noise - there's not enough leftover signal for LightGBM to learn anything useful from. Running the experiment and understanding why it failed is more useful than skipping it.

**Global LightGBM MASE is above 1.0**, meaning it does worse than a naive seasonal baseline on average. The reason is one SKU - HOUSEHOLD\_1\_430\_CA\_1 has 78% zero sales days. Training it together with regular-demand SKUs messes up the target distribution. The fix would be training within demand categories, but with 5 SKUs it wasn't worth doing.

**Croston is assigned to the intermittent SKU based on method, not just numbers.** SARIMA actually gets lower RMSE on that SKU's holdout (1.518 vs 1.756). But SARIMA has no way to handle sparse demand properly and would produce negative forecasts on zero-heavy windows. Croston's flat rate estimate is the right tool for that demand shape regardless of what one backtest shows.

\---

## Dataset

[M5 Forecasting Accuracy](https://www.kaggle.com/competitions/m5-forecasting-accuracy) - 30,490 Walmart product-store time series, five years of daily sales, with calendar events, SNAP flags, and price history. Standard benchmark for retail forecasting research.

This project uses 5 SKUs from one California store, covering Foods, Household, and Hobbies across demand volumes from 1.2 to 66.4 units/day and zero rates from 0% to 78%. The methodology scales to the full dataset but 5 SKUs is enough to surface real model differences without the compute cost.

\---

## Tech stack

|Layer|Tools|
|-|-|
|Data and features|Python 3.11, pandas, NumPy|
|Statistical models|statsmodels (SARIMA), Prophet|
|ML models|LightGBM|
|Intermittent demand|statsforecast (CrostonOptimized)|
|Inventory optimization|Custom Python module|
|LLM agent|Anthropic API, Claude Haiku 4.5, tool use|
|FastAPI backend|FastAPI, uvicorn|
|HTML frontend|Vanilla JS, Chart.js|
|Streamlit dashboard|Streamlit|
|Project tooling|uv, ruff, pytest|

\---

## Running it locally

You'll need Python 3.11, [uv](https://docs.astral.sh/uv/), and the M5 dataset CSVs in `data/raw/`.

```bash
uv sync
```

**FastAPI + HTML dashboard (recommended)**

```bash
$env:ANTHROPIC\_API\_KEY = "sk-ant-..."
uv run uvicorn api.main:app --reload --port 8000
```

Open `http://localhost:8000`. Four sections: model comparison overview, forecast explorer, inventory what-if with a service level slider, and the AI planning assistant.

**Streamlit app**

```bash
$env:ANTHROPIC\_API\_KEY = "sk-ant-..."
uv run streamlit run src/dashboard/app.py
```

**Regenerate CV results** (takes 20-40 min):

```bash
uv run python -m scripts.run\_phase2
```

**Regenerate inventory recommendations:**

```bash
uv run python -m scripts.run\_phase3
```

**Tests:**

```bash
uv run pytest tests/ -v
# 128 tests, all passing
```

\---

## Project structure

```
smart-inventory-forecaster/
├── api/
│   └── main.py              # FastAPI backend, 6 endpoints
├── frontend/
│   └── index.html           # Standalone HTML dashboard
├── src/
│   ├── data/                # M5 loaders, validation
│   ├── features/            # Lag, rolling, calendar, SNAP, price features
│   ├── models/              # SARIMA, Prophet, LightGBM, Hybrid, Croston
│   ├── inventory/           # Safety stock, ROP, EOQ optimizer
│   ├── agent/               # LLM agent, tools
│   └── dashboard/           # Streamlit app
├── scripts/
│   ├── run\_phase2.py        # CV pipeline
│   ├── run\_phase3.py        # Inventory optimization pipeline
│   └── demo\_agent.py        # Sample agent queries
├── tests/                   # 128 unit and integration tests
├── notebooks/               # EDA and model development
└── data/
    ├── raw/                 # M5 CSVs (gitignored)
    └── processed/           # Parquet outputs
```

\---

## Limitations

The safety stock formula uses RMSE at h=28 as the error standard deviation. This overestimates per-period error a bit since h=28 error accumulates over the full horizon, so safety stock ends up slightly conservative. A production system would calibrate against actual observed stockout rates.

The LLM agent is stateless between separate calls. The tool loop works correctly within one query but there's no memory across queries. Fine for a demo, limiting for anything conversational.

WRMSSE (the official M5 metric) isn't implemented. The hierarchical weight calculation is a project on its own. MAPE, RMSE, and MASE are enough for what this is doing.

\---

Built by **Gaurangkumar Makwana**. Open to feedback and questions - reach out on [LinkedIn](#).

