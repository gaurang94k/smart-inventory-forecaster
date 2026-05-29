from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import anthropic
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.agent.agent import run_agent
from src.inventory.optimizer import InventoryParams, compute_inventory

PROCESSED = Path(__file__).resolve().parents[2] / "data" / "processed"

SKUS = [
    "FOODS_3_090_CA_1",
    "FOODS_3_586_CA_1",
    "HOUSEHOLD_1_118_CA_1",
    "HOBBIES_1_348_CA_1",
    "HOUSEHOLD_1_430_CA_1",
]

EXAMPLE_QUESTIONS = [
    "Which SKU has the lowest forecast error at horizon 28?",
    "What safety stock does HOUSEHOLD_1_118_CA_1 need at a 99% service level?",
    "Summarize the inventory recommendations for the FOODS category.",
    "What is the annual carrying cost for HOBBIES_1_348_CA_1?",
]

st.set_page_config(page_title="Smart Inventory Forecaster", layout="wide")


@st.cache_data
def load_cv() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / "phase2_cv_results.parquet")


@st.cache_data
def load_recs() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / "phase3_inventory_recommendations.parquet")


@st.cache_data
def load_eda() -> pd.DataFrame:
    return pd.read_parquet(
        PROCESSED / "eda_sample_long_CA1.parquet",
        columns=["id", "date", "sales"],
    )


# ── Navigation ───────────────────────────────────────────────────────────────
st.sidebar.title("Smart Inventory Forecaster")
page = st.sidebar.radio(
    "Navigate",
    ["Forecast Explorer", "Inventory Recommendations", "AI Planning Assistant"],
)


# ── Page 1: Forecast Explorer ────────────────────────────────────────────────
if page == "Forecast Explorer":
    sku = st.sidebar.selectbox("Select SKU", SKUS)

    st.title("Forecast Explorer")

    cv = load_cv()

    # ── Model performance table ───────────────────────────────────────────────
    st.markdown("#### Model performance — mean across all folds and horizons")

    # no hardcoded model list — show all models present for this SKU
    sku_cv = cv[cv["sku_store"] == sku]
    metric_df = (
        sku_cv.groupby("model")[["mape", "rmse", "mase"]]
        .mean()
        .rename(columns={"mape": "MAPE", "rmse": "RMSE", "mase": "MASE"})
        .sort_values("MASE")  # /#16: sort by MASE ascending
    )

    # /#16: best model by lowest MASE, label states the metric
    best_model = metric_df["MASE"].idxmin()
    st.caption(f"SKU: **{sku}**  |  Best model (by MASE): **{best_model}**")

    # values already stored as percentages (e.g. 27.8), use {:.1f}% not {:.1%}
    st.dataframe(
        metric_df.style.format(
            {"MAPE": "{:.1f}%", "RMSE": "{:.2f}", "MASE": "{:.3f}"},
            na_rep="—",
        ),
        use_container_width=True,
    )

    # ── Forecast vs Actual chart ──────────────────────────────────────────────
    st.markdown("#### Forecast vs Actual")

    # model and fold selectors
    available_models = sorted(sku_cv["model"].unique().tolist())
    default_model_idx = available_models.index(best_model) if best_model in available_models else 0

    col_model, col_fold = st.columns(2)
    with col_model:
        chart_model = st.selectbox("Model", available_models, index=default_model_idx)
    with col_fold:
        model_cv = sku_cv[sku_cv["model"] == chart_model]
        available_folds = sorted(model_cv["fold"].unique().tolist())
        # Default to the last fold (most recent out-of-sample window)
        chart_fold = st.selectbox("Fold", available_folds, index=len(available_folds) - 1)

    chart_rows = model_cv[
        (model_cv["fold"] == chart_fold)
        & (model_cv["horizon"] == 28)
    ]

    st.markdown(
        f"*{chart_model} — fold {chart_fold}, horizon = 28 days*"
    )

    if chart_rows.empty:
        st.warning(
            f"No horizon-28 data found for {chart_model} / fold {chart_fold}. "
            "This model may not store per-date forecasts (e.g. Croston outputs a "
            "single rate, not a date-indexed series)."
        )
    else:
        try:
            r = chart_rows.iloc[0]
            forecast_dates = pd.to_datetime(r["forecast_dates"])
            forecast_values = list(r["forecast_values"])

            eda = load_eda()
            actuals_series = (
                eda[eda["id"] == f"{sku}_evaluation"].set_index("date")["sales"]
            )
            actuals = actuals_series.reindex(forecast_dates).values

            # consistent date labels — %b %d gives "Mar 06" for all ticks
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.plot(forecast_dates, actuals, label="Actual", color="#1f4e79")
            ax.plot(forecast_dates, forecast_values, label="Forecast", color="#5ba3d9")
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            fig.autofmt_xdate(rotation=45)
            ax.legend()
            ax.set_ylabel("Units sold")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        except (KeyError, TypeError) as exc:
            st.warning(
                f"Could not render chart for {chart_model}: {exc}. "
                "The model may not store date-indexed forecast values."
            )


# ── Page 2: Inventory Recommendations ───────────────────────────────────────
elif page == "Inventory Recommendations":
    sku = st.sidebar.selectbox("Select SKU (for delta metrics)", SKUS)

    st.title("Inventory Recommendations")

    recs = load_recs()

    # ── Baseline table ────────────────────────────────────────────────────────
    st.markdown("#### Baseline recommendations (service level = 95%)")
    display_cols = [
        "sku_store",
        "best_model",
        "safety_stock",
        "reorder_point",
        "eoq",
        "total_annual_cost",
    ]
    # rename snake_case columns
    baseline_display = (
        recs[display_cols]
        .set_index("sku_store")
        .rename(columns={
            "best_model": "Best Model",
            "safety_stock": "Safety Stock",
            "reorder_point": "Reorder Point",
            "eoq": "EOQ",
            "total_annual_cost": "Total Annual Cost",
        })
    )
    st.dataframe(
        baseline_display.style.format(
            {
                "Safety Stock": "{:.1f}",
                "Reorder Point": "{:.1f}",
                "EOQ": "{:.0f}",
                "Total Annual Cost": "${:,.0f}",
            }
        ),
        use_container_width=True,
    )

    st.markdown("---")

    # ── What-if section ───────────────────────────────────────────────────────
    st.markdown("#### What-if: adjust service level")

    # integer % range, divide internally
    service_level_pct = st.slider(
        "Target service level",
        min_value=80,
        max_value=99,
        value=95,
        step=1,
        format="%d%%",
    )
    service_level = service_level_pct / 100

    new_rows = []
    for _, base in recs.iterrows():
        # root cause: back-calculating ordering_cost from parquet can
        # introduce rounding drift and change EOQ. We store ordering_cost directly
        # if the column exists; otherwise fall back to the division.
        if "ordering_cost" in base.index:
            ordering_cost = float(base["ordering_cost"])
        else:
            ordering_cost = float(base["annual_ordering_cost"] / base["annual_orders"])

        params = InventoryParams(
            sku_store=base["sku_store"],
            mean_daily_demand=float(base["mean_daily_demand"]),
            forecast_std=float(base["forecast_std"]),
            lead_time_days=int(base["lead_time_days"]),
            service_level=service_level,
            ordering_cost=ordering_cost,
            holding_cost_per_unit=float(base["holding_cost_per_unit"]),
            unit_cost=float(base["unit_cost"]),
        )
        result = compute_inventory(params)
        # Issues #8/#9: proper column names + EOQ included
        new_rows.append(
            {
                "SKU": result.sku_store,
                "Safety Stock": result.safety_stock,
                "Reorder Point": result.reorder_point,
                "EOQ": result.eoq,
                "Total Annual Cost": result.total_annual_cost,
            }
        )

    new_df = pd.DataFrame(new_rows).set_index("SKU")
    st.dataframe(
        new_df.style.format(
            {
                "Safety Stock": "{:.1f}",
                "Reorder Point": "{:.1f}",
                "EOQ": "{:.0f}",
                "Total Annual Cost": "${:,.0f}",
            }
        ),
        use_container_width=True,
    )

    # ── Delta metrics ─────────────────────────────────────────────────────────
    st.markdown(f"#### Delta vs 95% baseline — {sku}")
    base_row = recs[recs["sku_store"] == sku].iloc[0]
    new_row = new_df.loc[sku]

    ss_delta = new_row["Safety Stock"] - float(base_row["safety_stock"])
    rop_delta = new_row["Reorder Point"] - float(base_row["reorder_point"])
    cost_delta = new_row["Total Annual Cost"] - float(base_row["total_annual_cost"])

    col1, col2, col3 = st.columns(3)
    col1.metric(
        "Safety Stock",
        f"{new_row['Safety Stock']:.1f} units",
        delta=f"{ss_delta:+.1f}",
    )
    col2.metric(
        "Reorder Point",
        f"{new_row['Reorder Point']:.1f} units",
        delta=f"{rop_delta:+.1f}",
    )
    # delta_color="inverse" — cost increase = red (bad), decrease = green (good)
    col3.metric(
        "Total Annual Cost",
        f"${new_row['Total Annual Cost']:,.0f}",
        delta=f"${cost_delta:+,.0f}",
        delta_color="inverse",
    )


# ── Page 3: AI Planning Assistant ───────────────────────────────────────────
elif page == "AI Planning Assistant":
    st.title("AI Planning Assistant")

    # persistent conversation history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # session state for populating input from example buttons
    if "pending_question" not in st.session_state:
        st.session_state.pending_question = ""

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        st.warning(
            "⚠️ ANTHROPIC_API_KEY environment variable is not set. "
            "Set it before launching Streamlit to enable the AI assistant."
        )

    # text_input with key so session state can pre-populate it
    question = st.text_input(
        "Ask a planning question:",
        value=st.session_state.pending_question,
        key="question_input",
        placeholder="e.g. What is the reorder point for FOODS_3_090_CA_1?",
        disabled=not bool(api_key),
    )
    # Clear pending after it has been read into the widget
    st.session_state.pending_question = ""

    submit = st.button("Submit", disabled=not bool(api_key))

    if submit and question.strip():
        # loading indicator
        with st.spinner("Thinking…"):
            try:
                response = run_agent(question.strip())
            # Issues #10/#11: structured error handling
            except anthropic.APIStatusError as exc:
                if exc.status_code == 529:
                    response = (
                        "⚠️ The AI service is currently overloaded. "
                        "Please wait a moment and try again."
                    )
                elif exc.status_code == 401:
                    response = (
                        "⚠️ Authentication failed. Check that ANTHROPIC_API_KEY is "
                        "set correctly and the key is valid."
                    )
                else:
                    response = f"⚠️ API error ({exc.status_code}): {exc.message}"
            except Exception as exc:
                response = f"⚠️ Unexpected error: {exc}"

        st.session_state.chat_history.append(
            {"question": question.strip(), "answer": response}
        )

    # Render conversation history
    for entry in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(entry["question"])
        with st.chat_message("assistant"):
            st.markdown(entry["answer"])

    # clickable example buttons that populate the input without bullets
    if not st.session_state.chat_history:  # only show when chat is empty
        st.markdown("**Example questions** — click to use:")
        for ex in EXAMPLE_QUESTIONS:
            if st.button(ex, key=f"ex_{hash(ex)}", disabled=not bool(api_key)):
                st.session_state.pending_question = ex
                st.rerun()
