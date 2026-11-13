"""Streamlit Decision-Support Dashboard for Cinema Operations Managers.
Show-level T-48h Ticket Demand Forecasting MVP.
"""
from datetime import date
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.predict import CinemaPredictor

st.set_page_config(
    page_title="Cinema Ticket Demand Forecasting",
    page_icon="🎬",
    layout="wide"
)

# Mandatory Header & Business Decision Framing
st.title("🎬 Cinema Ticket Demand Forecasting (Show-Level MVP)")
st.caption("Algions AI Labs Internship Assessment | Operational Decision-Support")

# Permanent Mandatory Disclaimer Banner
st.warning(
    "⚠️ **OPERATIONAL DISCLAIMER:** Decision-support only. Human managerial review required. "
    "This system is not an automated high-stakes decision maker. All screen and staffing allocations "
    "must be confirmed by theater operations personnel."
)


@st.cache_resource
def load_predictor():
    """Cache the predictor instance across dashboard sessions."""
    return CinemaPredictor()


try:
    predictor = load_predictor()
    model_ready = True
except Exception as e:
    st.error(f"Failed to load model artifacts: {e}. Please ensure 'python -m src.train' has been run.")
    model_ready = False

if model_ready:
    # Sidebar - Screening Parameters
    st.sidebar.header("🎯 Screening Parameters (T-48h)")
    st.sidebar.markdown("Enter details for upcoming screening scheduled in >= 48 hours.")

    # Populate known codes from history for convenience
    known_cinemas = sorted(list(predictor.cinema_median_caps.keys()))
    known_films = sorted(list(predictor.history_store.film_first_seen.keys()))

    cinema_input = st.sidebar.number_input(
        "Cinema Code",
        min_value=1,
        max_value=99999,
        value=int(known_cinemas[0]) if known_cinemas else 448,
        help="Anonymized ID of cinema complex"
    )

    film_input = st.sidebar.number_input(
        "Film Code",
        min_value=1,
        max_value=999999,
        value=int(known_films[0]) if known_films else 1576,
        help="Anonymized ID of scheduled film"
    )

    selected_date = st.sidebar.date_input(
        "Screening Date",
        value=date(2018, 10, 15),
        min_value=date(2018, 1, 1),
        max_value=date(2019, 12, 31),
        help="Date of the screening. History only up to Date - 2 days will be used."
    )

    slot_input = st.sidebar.number_input(
        "Show Time Slot",
        min_value=1,
        max_value=30,
        value=3,
        help="Screening slot index (1=matinee/first show, 2=afternoon, etc.)"
    )

    default_cap = float(predictor.cinema_median_caps.get(cinema_input, predictor.global_median_cap))
    capacity_input = st.sidebar.number_input(
        "Auditorium Capacity",
        min_value=10,
        max_value=5000,
        value=int(default_cap),
        help="Total seat capacity of the assigned screening room"
    )

    predict_btn = st.sidebar.button("Generate T-48h Forecast", type="primary")

    # Main Area
    st.subheader("📊 Forecast Summary")

    try:
        forecast = predictor.predict_single(
            cinema_code=cinema_input,
            film_code=film_input,
            date=str(selected_date),
            show_time=slot_input,
            capacity=capacity_input
        )

        # Cold-Start Warning Banner
        if forecast.is_cold_start:
            st.error(f"🚨 **{forecast.cold_start_warning}**")
        else:
            st.success("✅ **Established Screening:** Robust historical records available for this film and cinema.")

        # Metric KPIs
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Predicted Tickets Sold", f"{forecast.predicted_tickets:,.0f}")
        col2.metric("Predicted Occupancy", f"{forecast.predicted_occupancy * 100:.1f}%")
        col3.metric("80% Prediction Interval", f"[{forecast.interval_80_lower:,.0f} - {forecast.interval_80_upper:,.0f}]")
        col4.metric("Screen Capacity", f"{forecast.capacity:,.0f} seats")

        st.markdown("---")

        # Visualizations
        col_left, col_right = st.columns([1, 1])

        with col_left:
            st.markdown("### 🎟️ Demand vs Capacity Range")
            fig, ax = plt.subplots(figsize=(6, 3.5))
            
            # Bar for forecast and error bar for interval
            y_err_lower = forecast.predicted_tickets - forecast.interval_80_lower
            y_err_upper = forecast.interval_80_upper - forecast.predicted_tickets
            
            ax.bar(["Forecast"], [forecast.predicted_tickets], color="#1f77b4", width=0.4, label="Predicted Tickets")
            ax.errorbar(
                ["Forecast"], [forecast.predicted_tickets],
                yerr=[[y_err_lower], [y_err_upper]],
                fmt="none", ecolor="black", elinewidth=2, capsize=8, label="80% Interval"
            )
            ax.axhline(forecast.capacity, color="crimson", linestyle="--", linewidth=1.5, label=f"Capacity ({forecast.capacity:,.0f})")
            ax.set_ylim(0, max(forecast.capacity * 1.15, forecast.interval_80_upper * 1.15))
            ax.set_ylabel("Tickets")
            ax.grid(axis="y", linestyle=":", alpha=0.6)
            ax.legend(loc="upper right")
            st.pyplot(fig)

        with col_right:
            st.markdown("### 🔍 Key Input Features for this Slot")
            if forecast.top_contributing_features:
                feat_df = pd.DataFrame(forecast.top_contributing_features)
                st.dataframe(feat_df, use_container_width=True, hide_index=True)
            else:
                st.info("Feature contributions unavailable.")

        # Governance & Operational Guide
        with st.expander("ℹ️ Managerial Guidance & Operational Rules"):
            st.markdown("""
            - **Screen Allocation:** If upper interval reaches 90%+ capacity, consider upgrading to a larger auditorium if available.
            - **Staffing Allocation:** Plan usher and concession staffing proportional to the **upper interval bound** during opening weekends to avoid queue choke-points.
            - **Cold-Start Buffer:** For new films (`is_cold_start = True`), maintain flexible shift scheduling because variance is substantially higher ($\pm 73\%$ WAPE).
            """)

    except Exception as e:
        st.error(f"Inference error: {e}")
