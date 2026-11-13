"""Streamlit Decision-Support Dashboard for Cinema Operations Managers.
Show-level T-48h Ticket Demand Forecasting & Staffing Optimization.
"""
import sys
from datetime import date, timedelta
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# Ensure project root is in sys.path for robust imports across all runtimes
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import CinemaPredictor

# Page Setup
st.set_page_config(
    page_title="CinemaOps AI | Demand Forecasting",
    page_icon="🍿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #2563EB;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .disclaimer-box {
        background-color: #FFFBEB;
        border: 1px solid #FCD34D;
        border-radius: 8px;
        padding: 12px 16px;
        color: #92400E;
        font-size: 0.92rem;
        margin-bottom: 1.5rem;
    }
    .action-badge {
        font-weight: 600;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_predictor():
    """Load and cache the inference engine."""
    return CinemaPredictor()


# Top Navigation / Title
st.markdown('<div class="main-header">🍿 CinemaOps AI: Show Demand Forecaster</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Algions AI Labs MVP • 48-Hour Lookahead Decision-Support System</div>', unsafe_allow_html=True)

# Mandatory Operational Governance Disclaimer
st.markdown("""
<div class="disclaimer-box">
    ⚠️ <strong>OPERATIONAL DECISION-SUPPORT ONLY:</strong> Human managerial review required. 
    This system provides data-driven advisory forecasts (T-48h lead time) to assist auditorium screen and staffing allocation. 
    It is not an automated high-stakes decision maker. Managers must account for local weather, holidays, and competitor dynamics.
</div>
""", unsafe_allow_html=True)

try:
    predictor = load_predictor()
    model_ready = True
except Exception as e:
    st.error(f"❌ Could not initialize forecasting engine: {e}. Please ensure 'python -m src.train' was executed.")
    model_ready = False

if model_ready:
    # Sidebar: Preset Profiles & Inputs
    st.sidebar.header("⚙️ Screening Configuration")
    
    # Presets for quick evaluation
    preset = st.sidebar.selectbox(
        "⚡ Quick Test Presets",
        [
            "Custom Scenario",
            "Opening Night Blockbuster (High Footfall)",
            "Midweek Afternoon Matinee (Low Footfall)",
            "Cold-Start Debut Film (Zero Past History)",
            "Small Boutique Screen (100 seats)"
        ]
    )

    known_cinemas = sorted(list(predictor.cinema_median_caps.keys()))
    known_films = sorted(list(predictor.history_store.film_first_seen.keys()))

    if preset == "Opening Night Blockbuster (High Footfall)":
        def_cinema, def_film, def_date, def_slot, def_cap = 448, 1576, date(2018, 10, 15), 4, 350.0
    elif preset == "Midweek Afternoon Matinee (Low Footfall)":
        def_cinema, def_film, def_date, def_slot, def_cap = 448, 1576, date(2018, 10, 16), 1, 200.0
    elif preset == "Cold-Start Debut Film (Zero Past History)":
        def_cinema, def_film, def_date, def_slot, def_cap = 448, 99999, date(2018, 10, 15), 3, 250.0
    elif preset == "Small Boutique Screen (100 seats)":
        def_cinema, def_film, def_date, def_slot, def_cap = 100, 1576, date(2018, 10, 15), 3, 100.0
    else:
        def_cinema, def_film, def_date, def_slot, def_cap = 448, 1576, date(2018, 10, 15), 3, 300.0

    cinema_input = st.sidebar.number_input("Cinema Code", min_value=1, max_value=99999, value=int(def_cinema))
    film_input = st.sidebar.number_input("Film Code", min_value=1, max_value=999999, value=int(def_film))
    date_input = st.sidebar.date_input("Screening Date (T-48h)", value=def_date)
    slot_input = st.sidebar.slider("Show Time Slot", min_value=1, max_value=15, value=int(def_slot),
                                  help="Slot index (1=matinee/first show, 2=early afternoon, 3=evening peak, etc.)")
    capacity_input = st.sidebar.number_input("Screen Auditorium Capacity (Seats)", min_value=10, max_value=5000, value=int(def_cap))

    # Tabs
    tab_forecast, tab_roster, tab_metrics, tab_api = st.tabs([
        "🎯 Show Forecast", 
        "👥 Staffing & Screen Recommendations", 
        "📊 Model Evaluation & Metrics", 
        "🔌 API Integration"
    ])

    with tab_forecast:
        # Run inference
        forecast = predictor.predict_single(
            cinema_code=cinema_input,
            film_code=film_input,
            date=str(date_input),
            show_time=slot_input,
            capacity=capacity_input
        )

        # Cold-Start Diagnostic Banner
        if forecast.is_cold_start:
            st.error(f"🚨 **COLD-START / LOW-HISTORY ALERT:** {forecast.cold_start_warning}")
        else:
            st.success("✅ **Established Screening:** Robust historical tracking data exists for this film and theater.")

        # Metric Tiles
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Predicted Tickets Sold", f"{forecast.predicted_tickets:,.0f}", help="Expected attendance count")
        with c2:
            st.metric("Projected Occupancy", f"{forecast.predicted_occupancy * 100:.1f}%", help="Tickets sold / Capacity")
        with c3:
            st.metric("80% Prediction Interval", f"[{forecast.interval_80_lower:,.0f} - {forecast.interval_80_upper:,.0f}]", help="Empirically calibrated from validation residuals")
        with c4:
            st.metric("Auditorium Capacity", f"{forecast.capacity:,.0f} seats", help="Current room capacity ceiling")

        st.markdown("---")

        col_left, col_right = st.columns([3, 2])

        with col_left:
            st.markdown("#### 🎟️ Demand Distribution vs Auditorium Ceiling")
            fig, ax = plt.subplots(figsize=(7, 3.8))
            
            y_err_lower = forecast.predicted_tickets - forecast.interval_80_lower
            y_err_upper = forecast.interval_80_upper - forecast.predicted_tickets

            bar_color = "#3B82F6" if not forecast.is_cold_start else "#F59E0B"
            ax.bar(["Predicted Demand"], [forecast.predicted_tickets], color=bar_color, width=0.35, label="Expected Tickets")
            ax.errorbar(
                ["Predicted Demand"], [forecast.predicted_tickets],
                yerr=[[y_err_lower], [y_err_upper]],
                fmt="none", ecolor="#1E293B", elinewidth=2.5, capsize=8, label="80% Confidence Interval"
            )
            ax.axhline(forecast.capacity, color="#EF4444", linestyle="--", linewidth=2, label=f"Max Capacity ({forecast.capacity:,.0f})")
            
            # Fill headroom/shortage
            y_top = max(forecast.capacity * 1.15, forecast.interval_80_upper * 1.15)
            ax.set_ylim(0, y_top)
            ax.set_ylabel("Ticket Count", fontsize=10)
            ax.grid(axis="y", linestyle=":", alpha=0.6)
            ax.legend(loc="upper right", frameon=True)
            st.pyplot(fig)

        with col_right:
            st.markdown("#### 🔍 Key Historical Driving Features")
            if forecast.top_contributing_features:
                feat_df = pd.DataFrame(forecast.top_contributing_features)
                feat_df.columns = ["Model Feature", "Historical Value"]
                st.dataframe(feat_df, use_container_width=True, hide_index=True)
            else:
                st.info("Local feature breakdowns unavailable.")

            st.caption(f"**Prediction Horizon:** T-48h (Screening: {date_input} | Cutoff: {date_input - timedelta(days=2)})")

    with tab_roster:
        st.markdown("### 📋 Operational Roster & Room Allocation Guidance")
        st.markdown("Actionable operational staffing rules calibrated to avoid concession bottlenecks and staffing shortages:")

        upper_bound = forecast.interval_80_upper
        occupancy = forecast.predicted_occupancy

        # Screen Allocation Recommendation
        if occupancy >= 0.85 or upper_bound >= forecast.capacity:
            screen_action = "🚨 **UPGRADE SCREEN:** High risk of sold-out demand truncation. Consider swapping to a larger hall if available."
            action_color = "red"
        elif occupancy <= 0.20 and forecast.capacity > 200:
            screen_action = "ℹ️ **DOWNGRADE SCREEN:** Low projected occupancy. Downsizing to a smaller screen frees larger halls for high-demand titles."
            action_color = "orange"
        else:
            screen_action = "✅ **OPTIMAL ROOM ALLOCATION:** Capacity aligns well with projected attendance window."
            action_color = "green"

        st.info(screen_action)

        # Front-of-House Staffing Heuristics
        st.markdown("#### 👥 Front-of-House Staffing Allocations (Based on Upper 80% Bound: **{:,.0f} attendees**)".format(upper_bound))
        
        # Heuristics:
        # Ushers: 1 per 120 upper attendees (min 1)
        # Concession Cashiers: 1 per 70 upper attendees (min 1)
        # Box Office / Greeters: 1 per 150 upper attendees (min 1)
        ushers = max(1, int(np.ceil(upper_bound / 120.0)))
        concession = max(1, int(np.ceil(upper_bound / 70.0)))
        box_office = max(1, int(np.ceil(upper_bound / 150.0)))

        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("Box Office / Scanners", f"{box_office} staff")
        sc2.metric("Concession Cashiers", f"{concession} staff")
        sc3.metric("Floor Ushers", f"{ushers} staff")
        sc4.metric("Total Shift Staff", f"{box_office + concession + ushers} staff")

        st.caption("Staffing planned against the upper confidence bound to mitigate the asymmetric cost of under-staffing (1.5x penalty).")

    with tab_metrics:
        st.markdown("### 📊 Real Computed Model Evaluation (Held-Out Test Set)")
        st.markdown("Evaluated chronologically on **22,764 test shows** (2018-09-30 to 2018-11-04):")

        metrics_data = predictor.bundle.get("metrics", {}).get("models", {})
        if metrics_data:
            eval_table = pd.DataFrame([
                {"Model": "Hierarchical Baseline", **metrics_data.get("baseline", {})},
                {"Model": "Ridge (Occupancy)", **metrics_data.get("ridge", {})},
                {"Model": "LightGBM (Tickets L1)", **metrics_data.get("lightgbm", {})}
            ])
            st.dataframe(eval_table, use_container_width=True, hide_index=True)

        col_res1, col_res2 = st.columns(2)
        with col_res1:
            st.markdown("#### Residual Diagnostics on Test Set")
            res_plot_path = Path("reports/residual_plot.png")
            if res_plot_path.exists():
                st.image(str(res_plot_path), caption="LightGBM Residuals vs Prediction with Validation Q10/Q90 Quantiles")
            else:
                st.info("Residual plot not found.")

        with col_res2:
            st.markdown("#### Top 5 Features by Gain")
            feat_imp_path = Path("reports/feature_importance.csv")
            if feat_imp_path.exists():
                imp_df = pd.read_csv(feat_imp_path).head(5)
                st.dataframe(imp_df, use_container_width=True, hide_index=True)

    with tab_api:
        st.markdown("### 🔌 FastAPI Microservice Integration")
        st.markdown("""
        The backend runs as a high-performance REST API.
        - **Local API Base:** `http://localhost:8000`
        - **Interactive Docs:** `http://localhost:8000/docs`
        """)

        st.markdown("#### Example `curl` Request:")
        st.code(f"""curl -X POST "http://localhost:8000/predict" \\
     -H "Content-Type: application/json" \\
     -d '{{"cinema_code": {cinema_input}, "film_code": {film_input}, "date": "{date_input}", "show_time": {slot_input}, "capacity": {capacity_input}}}'""", language="bash")

        st.markdown("#### Live JSON Response Preview:")
        st.json({
            "predicted_tickets": forecast.predicted_tickets,
            "predicted_occupancy": forecast.predicted_occupancy,
            "capacity": forecast.capacity,
            "interval_80_lower": forecast.interval_80_lower,
            "interval_80_upper": forecast.interval_80_upper,
            "is_cold_start": forecast.is_cold_start,
            "cold_start_warning": forecast.cold_start_warning,
            "top_contributing_features": forecast.top_contributing_features,
            "limitation_note": forecast.limitation_note
        })
