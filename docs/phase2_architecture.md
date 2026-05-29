# Phase 2 Architecture

> **Status:** Locked. All six decisions agreed before writing any Phase 2 code.
> Date locked: end of week 2 of the project.

This document captures the six architectural decisions that shape Phase 2 (the
forecasting-models phase). The goal of writing this down before coding is simple:
when something goes sideways three weeks from now, this file explains *why* we
made the choice we did and what the alternatives looked like. Future-me will
thank current-me.

---

## Context

Phase 2 trains six forecasting models on five M5 SKUs (from the Phase 1
shortlist), compares them under rolling-origin cross-validation, and produces a
comparison table that's defensible in technical interviews.

The models:

| # | Model                     | Scope          | Role                                  |
|---|---------------------------|----------------|---------------------------------------|
| 1 | SARIMA                    | Per-SKU        | Classical statistical baseline        |
| 2 | Prophet                   | Per-SKU        | Robust seasonal, planner-readable     |
| 3 | LightGBM, per-SKU         | Per-SKU        | Modern industry-default ML            |
| 4 | LightGBM, global          | All 5 SKUs     | Cross-series learning test            |
| 5 | Hybrid SARIMA + LightGBM  | Per-SKU        | Statistical + ML residual model       |
| 6 | Croston                   | Intermittent SKU only | Right-tool-for-the-data demo |

The shortlist of five SKUs spans all three M5 categories and mixes continuous
demand (zero-rates 0%–24%) with one intermittent contrast case (78% zero-rate).

---

## Decision 1 — Common model interface

**Choice:** Shared `Forecaster` abstract base class.

```python
# src/models/base.py
from abc import ABC, abstractmethod
import pandas as pd

class Forecaster(ABC):
    """Common interface for every model in the Phase 2 comparison."""

    @abstractmethod
    def fit(self, history: pd.DataFrame) -> "Forecaster":
        """Train on a historical slice. Returns self for chaining."""

    @abstractmethod
    def forecast(self, horizon: int) -> pd.DataFrame:
        """Produce point forecasts (and confidence intervals if supported)
        for the next `horizon` days."""

    @abstractmethod
    def name(self) -> str:
        """Human-readable identifier used in comparison tables and logs."""
```

Every model (`SarimaForecaster`, `ProphetForecaster`, `LightGBMForecaster`,
`GlobalLightGBMForecaster`, `HybridForecaster`, `CrostonForecaster`) subclasses
this and wraps its native library's API to conform.

**Alternatives considered:**

- *Native APIs, separate driver scripts per model.* Faster to first result, but
  the CV loop would have to be reimplemented per model and the comparison code
  would be brittle. Six ad-hoc scripts read as junior; one abstraction read as
  senior.
- *Python `Protocol` (structural typing) instead of ABC.* Lighter-weight,
  more flexible, but doesn't enforce inheritance — easier to drift. ABC
  catches "you forgot to implement `forecast`" at class definition time.

**Why ABC:** the portfolio signal of "I built a clean abstraction over
heterogeneous forecasting libraries" is exactly what a senior ML engineer would
do, and the wrapper layer is a day of work at most. The CV loop becomes
model-agnostic, which compounds across the rest of Phase 2.

---

## Decision 2 — Feature engineering

**Choice:** Dedicated `src/features/` module with two entry points.

```python
# src/features/builder.py
def build_features_per_sku(df: pd.DataFrame) -> pd.DataFrame: ...
def build_features_global(df: pd.DataFrame) -> pd.DataFrame: ...
```

**Feature set:**

- **Lags:** sales at t-1, t-7, t-14, t-28, t-365
- **Rolling statistics:** 7-day mean, 28-day mean, 7-day std
- **Calendar:** day-of-week, week-of-year, month, year, is_weekend
- **Events:** one-hot from `event_name_1`, plus `event_type_1`
- **SNAP flags:** binary indicators for SNAP days (state-specific)
- **Price:** `sell_price`, price-vs-rolling-mean, price-change indicator
- **Global-only:** `item_id`, `dept_id`, `cat_id`, `store_id`, `state_id` via
  LightGBM's native `categorical_feature` parameter (no one-hot explosion, no
  target-encoding leakage risk)

**Anti-leakage rule:** features are computed *inside each CV fold* on training
data only, then applied to the test fold. This prevents the classic mistake of
computing rolling means on the full series before splitting (which lets test
statistics bleed into training features).

**Why a separate `features/` module instead of nesting under `models/`:** the
hybrid model needs features for its residual stage, and the inventory layer in
Phase 3 may want some of the same features for demand-variability estimates.
Treat features as a shared concern.

**Why LightGBM native categorical handling for the global model:** one-hot
encoding 5 SKUs is fine but doesn't generalize if we ever expand the SKU count;
target encoding risks fold-leakage if implemented carelessly. LightGBM's native
support is the production-standard choice and what retail ML teams actually do.

---

## Decision 3 — Cross-validation

**Choice:** Expanding-window rolling-origin CV.

```
Fold 1: train days 1–1801   → test days 1802–1829
Fold 2: train days 1–1829   → test days 1830–1857
Fold 3: train days 1–1857   → test days 1858–1885
Fold 4: train days 1–1885   → test days 1886–1913
Fold 5: train days 1–1913   → test days 1914–1941
```

- **Window type:** expanding (training set grows each fold)
- **Number of folds:** 5
- **Step between folds:** 28 days
- **Horizons evaluated:** 7, 14, 28 days within each test window
- **Test coverage:** days 1802–1941 (≈ last 7 months of the M5 timeline)

**Alternatives considered:**

- *Sliding window* (fixed-size training set, drops oldest data each fold). Right
  choice when older data is misleading — regime shifts, business model changes,
  store remodels. M5 has none of those, so expanding wins because more history
  is better.
- *Fewer folds (3) or more (10).* 5 is the sweet spot: enough samples for stable
  averages, few enough to keep total Phase 2 runtime under a few hours.

**Horizon choice:** 28 is M5's official horizon and what the WRMSSE metric is
defined for. 7 and 14 reflect realistic reorder cycles for a drugstore — the
business case the project actually addresses.

**Croston:** identical folds and horizons. Only the reported metric switches
from MAPE to MASE because MAPE is undefined on zero-actual days, which is
exactly when intermittent demand happens.

---

## Decision 4 — Metrics

**Choice:** In-house for the textbook metrics, library for the M5-specific one.

| Metric  | Implementation     | Why                                                      |
|---------|--------------------|----------------------------------------------------------|
| MAPE    | In-house, ~10 LOC  | Trivial, instructive, no dep risk                         |
| RMSE    | In-house, ~5 LOC   | Trivial, sklearn has it but writing it shows we know it   |
| MASE    | In-house, ~15 LOC  | Standard formula, important for intermittent              |
| WRMSSE  | Reference impl     | Complex hierarchical metric; correctness > NIH on M5      |

**Why split:** writing MAPE/RMSE/MASE ourselves demonstrates we understand the
math and gives full control over edge cases (zero actuals, missing values).
WRMSSE involves M5's specific hierarchical weighting scheme — implementing it
from scratch is a known footgun, and any reviewer familiar with the M5
competition will spot a subtle bug. Reference implementations exist
(`utilsforecast` from the Nixtla ecosystem); we use one and document the choice.

**Portfolio framing:** "We implemented the textbook metrics from scratch to
demonstrate understanding; for the M5-specific WRMSSE we used the reference
implementation because methodological correctness on the competition metric
matters more than NIH."

---

## Decision 5 — Folder structure

**Choice:** Flat siblings under `src/`.

```
src/
├── data/          # loaders, validators
├── features/      # feature builders
├── models/        # base.py + one file per model
├── evaluation/    # metrics + cv
├── inventory/     # Phase 3
├── agent/         # Phase 4
└── dashboard/     # Phase 5
```

Tests mirror this structure.

**Alternative considered:** group features and evaluation under `models/`. More
"research lab" style, but produces deeper imports (`from
src.models.implementations.lgbm import ...`) and obscures the fact that
features and evaluation are reusable across non-modeling code paths.

**Why flat:** it's what scikit-learn-shaped projects do, it reads cleanly on
GitHub, and it matches what production ML codebases at retail-tech companies
use. Onboarding cost for a reviewer scanning the repo: near zero.

---

## Decision 6 — Notebook vs script split

**Choice:** Per-model notebooks for narrative, scripts for reproducibility.

```
notebooks/
  01_eda.ipynb                ← Phase 1 (done)
  02_baseline_sarima.ipynb
  03_baseline_prophet.ipynb
  04_lgbm_per_sku.ipynb
  05_lgbm_global.ipynb
  06_hybrid.ipynb
  07_croston.ipynb
  08_comparison.ipynb         ← final headline table + plots

scripts/
  run_phase2.py               ← end-to-end reproducible pipeline
```

**The two paths consume the same `src/` code.** No model logic lives in
notebooks; notebooks call functions from `src/`. Same for scripts. This
prevents the classic "the notebook says one thing, the script says another"
drift.

**Why this combination:**

- *Notebooks alone:* great for narrative, terrible for "run the whole pipeline
  reliably" — cell-ordering bugs, hidden kernel state, hard to integrate with
  the Phase 4 LLM agent.
- *Scripts alone:* fully reproducible, but loses the inline-plots-and-markdown
  surface that makes the project legible to a recruiter scrolling on GitHub.

The combination gives both: notebooks for human readers, scripts for machines
(and, later, the LLM agent calling modeling code as functions).

**Bite-size discipline:** notebooks 02–07 will be created one at a time as each
model is developed, not all upfront. SARIMA first, get it working end-to-end,
*then* move to Prophet. Each one is a complete deliverable on its own.

---

## What this enables

By the end of Phase 2, the repo will produce:

- A comparison table (per-SKU and aggregated) with MAPE, RMSE, MASE, WRMSSE for
  every (model, SKU, horizon) combination.
- Per-model notebooks each telling the story of one approach with diagnostic plots.
- A single `scripts/run_phase2.py` that recreates everything end-to-end on a
  fresh clone with `uv run python scripts/run_phase2.py`.
- A clean abstraction (`Forecaster`) and shared evaluation infrastructure
  (`evaluation/cv.py`, `evaluation/metrics.py`) that Phase 3 (inventory
  optimization) and Phase 4 (LLM agent) can call into directly.

The decisions above are the boring scaffolding that makes the interesting work
in Phase 2 (per-SKU vs global LightGBM comparison; hybrid residual analysis;
Croston validation on intermittent demand) actually possible.

---

## Revision log

- *Initial version:* end of week 2 — six decisions locked before writing any
  Phase 2 code.
