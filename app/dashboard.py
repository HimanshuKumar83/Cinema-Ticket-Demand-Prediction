"""
CinemaOps AI - Enterprise Decision-Support Dashboard
Comprehensive 7-page operations console for show-level ticket demand forecasting,
5-model benchmarking, SHAP explainability, and capacity/staffing optimization.
"""

import sys
import json
from datetime import date, timedelta
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import CinemaPredictor

st.set_page_config(
    page_title="CinemaOps AI | Decision-Support System",
    page_icon="🍿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 800; color: #0F172A; margin-bottom: 0.2rem; }
    .sub-header { font-size: 1.05rem; color: #475569; margin-bottom: 1.2rem; }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 12px;
        padding: 16px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .disclaimer-box {
        background-color: #FEF3C7;
        border-left: 5px solid #F59E0B;
        border-radius: 8px;
        padding: 12px 18px;
        color: #92400E;
        font-size: 0.92rem;
        margin-bottom: 1.5rem;
    }
    .kpi-title { font-size: 0.82rem; text-transform: uppercase; font-weight: 700; color: #64748B; letter-spacing: 0.5px; }
    .kpi-value { font-size: 1.85rem; font-weight: 800; color: #0F172A; }
    .kpi-sub { font-size: 0.82rem; color: #64748B; margin-top: 4px; }
    .badge-upgrade { background-color: #FEE2E2; color: #DC2626; font-weight: 700; padding: 4px 8px; border-radius: 6px; }
    .badge-maintain { background-color: #DCFCE7; color: #16A34A; font-weight: 700; padding: 4px 8px; border-radius: 6px; }
    .badge-downgrade { background-color: #FEF3C7; color: #D97706; font-weight: 700; padding: 4px 8px; border-radius: 6px; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_predictor():
    return CinemaPredictor()


# Top Navigation / Title
st.markdown('<div class="main-header">🍿 CinemaOps AI: Show Demand Forecaster</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Algions AI Labs MVP • T-48h Advance Scheduling & Staffing Decision-Support System</div>', unsafe_allow_html=True)

# Mandatory Operational Governance Disclaimer
st.markdown("""
<div class="disclaimer-box">
    ⚠️ <strong>OPERATIONAL DECISION-SUPPORT ONLY:</strong> Human managerial review required. 
    This system provides data-driven advisory forecasts (T-48h lead time) to assist auditorium screen and staffing allocation. 
    It is strictly a decision-support advisory tool and never an automated decision maker. Managers must account for local weather, holidays, and live operational constraints.
</div>
""", unsafe_allow_html=True)

try:
    predictor = load_predictor()
    model_ready = True
except Exception as e:
    st.error(f"❌ Could not initialize forecasting engine: {e}. Please run 'python -m src.train' first.")
    model_ready = False

if model_ready:
    # Top 7 Navigation Tabs
    tab_overview, tab_forecast, tab_benchmark, tab_coldstart, tab_shap, tab_advisory, tab_governance = st.tabs([
        "📊 1. Overview & KPIs",
        "🎯 2. T-48 Demand Forecast",
        "🏆 3. 5-Model Benchmark",
        "❄️ 4. Cold-Start Analytics",
        "🧠 5. Explainability (SHAP)",
        "📋 6. Capacity & Staffing Advisory",
        "🛡️ 7. Monitoring & Governance",
    ])

    # -------------------------------------------------------------
    # TAB 1: OVERVIEW & KPIS
    # -------------------------------------------------------------
    with tab_overview:
        st.markdown("### 🚀 System Operational Overview")
        st.markdown("Executive summary of cinema forecasting pipeline performance across 30,000 scheduled shows:")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown('<div class="metric-card"><div class="kpi-title">Forecast Horizon</div><div class="kpi-value">T-48h</div><div class="kpi-sub">Strictly zero future lookahead</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="metric-card"><div class="kpi-title">Best Model MAE</div><div class="kpi-value">11.27</div><div class="kpi-sub">LightGBM (Held-out Test)</div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown('<div class="metric-card"><div class="kpi-title">Test WAPE</div><div class="kpi-value">11.22%</div><div class="kpi-sub">Weighted percentage error</div></div>', unsafe_allow_html=True)
        with c4:
            st.markdown('<div class="metric-card"><div class="kpi-title">80% Interval Coverage</div><div class="kpi-value">76.64%</div><div class="kpi-sub">Capacity-Normalized Conformal</div></div>', unsafe_allow_html=True)

        st.markdown("---")
        col_flow, col_meta = st.columns([3, 2])
        with col_flow:
            st.markdown("#### 🔄 Zero-Leakage Forecasting Workflow")
            st.info("""
            1. **T-168h to T-72h:** Advance bookings open online; initial customer traction observed.
            2. **T-48h Cutoff (Prediction Timestamp):** Advance bookings ($B_{48}$), 24h booking velocity ($V_{24}$), and historical averages are locked.
            3. **Inference Execution:** LightGBM generates attendance expectation and 80% conformal interval $[L_{80}, U_{80}]$.
            4. **Operational Advisory:** System alerts managers if room upgrade/downgrade is needed and recommends usher/concession staffing rosters.
            5. **T-0h Show Showtime:** Final show attendance verified; metrics logged for drift and calibration monitoring.
            """)

        with col_meta:
            st.markdown("#### 🏢 Active Portfolio Scale")
            meta_path = Path("data/simulated/simulation_meta.json")
            if meta_path.exists():
                with open(meta_path, "r") as f:
                    meta_data = json.load(f)
                st.dataframe(pd.DataFrame([
                    {"Parameter": "Total Shows Analyzed", "Value": f"{meta_data.get('total_shows', 30000):,}"},
                    {"Parameter": "Granular Booking Events", "Value": f"{meta_data.get('total_booking_events', 2065328):,}"},
                    {"Parameter": "Cinema Complexes", "Value": f"{meta_data.get('cinemas_count', 30)} (Tier-1 & Tier-2)"},
                    {"Parameter": "Active Movie Titles", "Value": f"{meta_data.get('movies_count', 100)}"},
                    {"Parameter": "Mean Screen Capacity", "Value": f"{meta_data.get('mean_capacity', 250):.1f} seats"},
                    {"Parameter": "Mean Portfolio Occupancy", "Value": f"{meta_data.get('mean_occupancy', 0.62)*100:.1f}%"},
                ]), use_container_width=True, hide_index=True)
            else:
                st.info("Simulation metadata not found.")

    # -------------------------------------------------------------
    # TAB 2: T-48 DEMAND FORECAST
    # -------------------------------------------------------------
    with tab_forecast:
        st.sidebar.header("⚙️ Show Configuration (T-48h)")
        preset = st.sidebar.selectbox(
            "⚡ Quick Operational Presets",
            [
                "Custom Parameters",
                "Friday Night Blockbuster (High Velocity)",
                "Midweek Matinee (Low Attendance)",
                "Cold-Start Debut Film (New Movie)",
                "Boutique Theater Hall (100 seats)",
            ]
        )

        if preset == "Friday Night Blockbuster (High Velocity)":
            p_cin, p_mov, p_cap, p_slot, p_price, p_adv, p_v24, p_vprev, p_wknd, p_block = "CIN_01", "MOV_001", 360, "Night", 320.0, 180.0, 75.0, 45.0, 1, 1
        elif preset == "Midweek Matinee (Low Attendance)":
            p_cin, p_mov, p_cap, p_slot, p_price, p_adv, p_v24, p_vprev, p_wknd, p_block = "CIN_05", "MOV_012", 200, "Matinee", 200.0, 25.0, 8.0, 6.0, 0, 0
        elif preset == "Cold-Start Debut Film (New Movie)":
            p_cin, p_mov, p_cap, p_slot, p_price, p_adv, p_v24, p_vprev, p_wknd, p_block = "CIN_02", "MOV_999", 250, "Evening", 250.0, 30.0, 12.0, 10.0, 1, 0
        elif preset == "Boutique Theater Hall (100 seats)":
            p_cin, p_mov, p_cap, p_slot, p_price, p_adv, p_v24, p_vprev, p_wknd, p_block = "CIN_10", "MOV_005", 100, "Evening", 280.0, 40.0, 18.0, 12.0, 1, 0
        else:
            p_cin, p_mov, p_cap, p_slot, p_price, p_adv, p_v24, p_vprev, p_wknd, p_block = "CIN_01", "MOV_001", 300, "Evening", 250.0, 110.0, 45.0, 30.0, 1, 1

        c_cin = st.sidebar.text_input("Cinema ID", value=p_cin)
        c_mov = st.sidebar.text_input("Movie ID", value=p_mov)
        c_cap = st.sidebar.number_input("Screen Capacity (Seats)", min_value=50, max_value=500, value=int(p_cap))
        c_slot = st.sidebar.selectbox("Show Slot", ["Morning", "Matinee", "Evening", "Night"], index=["Morning", "Matinee", "Evening", "Night"].index(p_slot))
        c_price = st.sidebar.number_input("Average Ticket Price ($)", min_value=50.0, max_value=1000.0, value=float(p_price))
        c_adv = st.sidebar.number_input("Advance Bookings at T-48h", min_value=0.0, max_value=float(c_cap), value=float(p_adv))
        c_v24 = st.sidebar.number_input("24h Booking Velocity [T-72h, T-48h]", min_value=0.0, max_value=float(c_cap), value=float(p_v24))
        c_vprev = st.sidebar.number_input("Previous 24h Velocity [T-96h, T-72h]", min_value=0.0, max_value=float(c_cap), value=float(p_vprev))
        c_wknd = st.sidebar.checkbox("Is Weekend (Fri-Sun)", value=bool(p_wknd))
        c_block = st.sidebar.checkbox("Is Blockbuster Title", value=bool(p_block))

        forecast = predictor.predict_single(
            cinema_id=c_cin,
            movie_id=c_mov,
            screen_capacity=c_cap,
            show_slot=c_slot,
            ticket_price=c_price,
            advance_bookings_t48=c_adv,
            booking_velocity_24h=c_v24,
            booking_velocity_previous_24h=c_vprev,
            is_weekend=int(c_wknd),
            is_blockbuster=int(c_block),
        )

        if forecast.is_cold_start:
            st.warning(f"🚨 **COLD-START NOTICE:** {forecast.cold_start_warning}")
        else:
            st.success("✅ **Established Tracking Data:** Historical shows exist for both this movie and cinema.")

        fc1, fc2, fc3, fc4 = st.columns(4)
        fc1.metric("Predicted Tickets Sold", f"{forecast.predicted_tickets:.0f}", help="Expected total attendance")
        fc2.metric("Projected Occupancy", f"{forecast.predicted_occupancy * 100:.1f}%", help="Predicted / Capacity")
        fc3.metric("80% Prediction Interval", f"[{forecast.interval_80_lower:.0f} - {forecast.interval_80_upper:.0f}]", help="Residual quantile bounds")
        fc4.metric("Screen Capacity", f"{forecast.screen_capacity:.0f} seats", help="Current room capacity")

        st.markdown("---")
        g_left, g_right = st.columns([3, 2])

        with g_left:
            st.markdown("#### 🎟️ Demand Distribution vs Screen Ceiling")
            fig, ax = plt.subplots(figsize=(7, 3.8))
            y_lower_err = forecast.predicted_tickets - forecast.interval_80_lower
            y_upper_err = forecast.interval_80_upper - forecast.predicted_tickets
            ax.bar(["Predicted Demand"], [forecast.predicted_tickets], color="#2563EB", width=0.35, label="Predicted Tickets")
            ax.errorbar(["Predicted Demand"], [forecast.predicted_tickets],
                         yerr=[[y_lower_err], [y_upper_err]], fmt="none", ecolor="#0F172A", elinewidth=2.5, capsize=8, label="80% Prediction Interval")
            ax.axhline(forecast.screen_capacity, color="#EF4444", linestyle="--", linewidth=2, label=f"Capacity Ceiling ({forecast.screen_capacity:.0f})")
            ax.set_ylim(0, max(forecast.screen_capacity * 1.15, forecast.interval_80_upper * 1.15))
            ax.set_ylabel("Tickets Sold", fontsize=10)
            ax.grid(axis="y", linestyle=":", alpha=0.6)
            ax.legend(loc="upper right")
            st.pyplot(fig)

        with g_right:
            st.markdown("#### 🧠 Local Driver Breakdown (SHAP)")
            st.markdown("**Positive Factors Increasing Demand:**")
            if forecast.positive_drivers:
                for d in forecast.positive_drivers:
                    st.write(f"➕ **{d['feature']}** ({d['feature_value']}): +{d['shap_impact']:.1f} tickets")
            else:
                st.caption("None significant.")

            st.markdown("**Negative Factors Dampening Demand:**")
            if forecast.negative_drivers:
                for d in forecast.negative_drivers:
                    st.write(f"➖ **{d['feature']}** ({d['feature_value']}): {d['shap_impact']:.1f} tickets")
            else:
                st.caption("None significant.")

    # -------------------------------------------------------------
    # TAB 3: 5-MODEL BENCHMARK
    # -------------------------------------------------------------
    with tab_benchmark:
        st.markdown("### 🏆 Rigorous 5-Model Evaluation Benchmark")
        st.markdown("All models evaluated on the exact same held-out chronological test split (4,500 shows, 2025-01-12 to 2025-03-19):")

        bench_path = Path("artifacts/model_benchmark.json")
        if bench_path.exists():
            with open(bench_path, "r") as f:
                bench_data = json.load(f)
            df_b = pd.DataFrame(bench_data)
            st.dataframe(df_b, use_container_width=True, hide_index=True)

            st.markdown("#### 📊 Model Error Comparison (MAE)")
            fig_b, ax_b = plt.subplots(figsize=(8, 3.5))
            colors = ["#16A34A" if m in ["Random Forest", "LightGBM"] else "#3B82F6" for m in df_b["Model"]]
            ax_b.bar(df_b["Model"], df_b["MAE"], color=colors, width=0.45)
            ax_b.set_ylabel("Mean Absolute Error (Tickets)", fontsize=10)
            ax_b.grid(axis="y", linestyle=":", alpha=0.6)
            for i, v in enumerate(df_b["MAE"]):
                ax_b.text(i, v + 0.8, f"{v:.2f}", ha="center", fontweight="bold", fontsize=9)
            st.pyplot(fig_b)
        else:
            st.info("Benchmark data not found. Run 'python -m src.train'.")

    # -------------------------------------------------------------
    # TAB 4: COLD-START ANALYTICS
    # -------------------------------------------------------------
    with tab_coldstart:
        st.markdown("### ❄️ Cold-Start & Operational Segment Slicing")
        st.markdown("Evaluation of model resilience when handling unobserved movies, low screening histories, or boutique screen sizes:")

        metrics_path = Path("artifacts/metrics.json")
        if metrics_path.exists():
            with open(metrics_path, "r") as f:
                full_m = json.load(f)
            slices = full_m.get("operational_slices", {})

            col_s1, col_s2 = st.columns(2)
            with col_s1:
                st.markdown("#### Cold-Start & History Segments")
                if "cold_start" in slices:
                    st.dataframe(pd.DataFrame(slices["cold_start"]).T, use_container_width=True)
                else:
                    st.caption("Cold start slice data unavailable.")

            with col_s2:
                st.markdown("#### Screen Capacity Buckets")
                if "capacity_buckets" in slices:
                    st.dataframe(pd.DataFrame(slices["capacity_buckets"]).T, use_container_width=True)
                else:
                    st.caption("Capacity bucket data unavailable.")
        else:
            st.info("Slice data not found.")

    # -------------------------------------------------------------
    # TAB 5: SHAP EXPLAINABILITY
    # -------------------------------------------------------------
    with tab_shap:
        st.markdown("### 🧠 TreeSHAP Global Feature Importance")
        st.markdown("Empirical feature importance calculated using TreeSHAP on test observations:")

        shap_path = Path("artifacts/shap_summary.json")
        if shap_path.exists():
            with open(shap_path, "r") as f:
                shap_list = json.load(f)
            df_shap = pd.DataFrame(shap_list).head(10)

            c_sh1, c_sh2 = st.columns([3, 2])
            with c_sh1:
                fig_sh, ax_sh = plt.subplots(figsize=(7, 4.2))
                ax_sh.barh(df_shap["feature"][::-1], df_shap["importance"][::-1], color="#2563EB", height=0.55)
                ax_sh.set_xlabel("Mean |SHAP Value| (Impact on Ticket Forecast)", fontsize=10)
                ax_sh.grid(axis="x", linestyle=":", alpha=0.6)
                st.pyplot(fig_sh)

            with c_sh2:
                st.markdown("#### Top Predictive Drivers")
                st.dataframe(df_shap, use_container_width=True, hide_index=True)
                st.caption("Advance bookings at T-48 and 24h velocity dominate show-level prediction accuracy.")

    # -------------------------------------------------------------
    # TAB 6: CAPACITY & STAFFING ADVISORY
    # -------------------------------------------------------------
    with tab_advisory:
        st.markdown("### 📋 Screen Allocation & Front-of-House Staffing Advisory")
        adv = forecast.operational_advisory

        # Decision Pill
        decision = adv["screen_decision"]
        if decision == "UPGRADE_SCREEN":
            st.markdown(f'<span class="badge-upgrade">🚨 RECOMMENDATION: {adv["recommended_action"]}</span>', unsafe_allow_html=True)
        elif decision == "DOWNGRADE_SCREEN":
            st.markdown(f'<span class="badge-downgrade">⚠️ RECOMMENDATION: {adv["recommended_action"]}</span>', unsafe_allow_html=True)
        else:
            st.markdown(f'<span class="badge-maintain">✅ RECOMMENDATION: {adv["recommended_action"]}</span>', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("#### 👥 Recommended Front-of-House Staffing Allocation")
        st.caption("Calculated to prevent box-office queues and concession stockouts:")

        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("Box Office / Scanners", f"{adv.get('box_office_staff', 1)} staff")
        sc2.metric("Concession Cashiers", f"{adv.get('concession_cashiers', 2)} staff")
        sc3.metric("Floor Ushers", f"{adv.get('ushers_count', 1)} staff")
        sc4.metric("Cleaning Crew", f"{adv.get('cleaning_crew', 2)} staff")

        st.metric("Total Floor Staff", f"{adv.get('total_floor_staff', 6)} staff members")
        st.metric("Estimated Concession Units (Popcorn/Soda)", f"{adv.get('concession_inventory_units', 100)} units", help="Projected 0.70 concession units per attendee")

    # -------------------------------------------------------------
    # TAB 7: MONITORING & GOVERNANCE
    # -------------------------------------------------------------
    with tab_governance:
        st.markdown("### 🛡️ Operational Monitoring & Model Governance")
        st.markdown("""
        **Operational Safety Guardrails:**
        - **Hard Capacity Bound:** Predictions are strictly constrained within $[0, \\text{Screen Capacity}]$.
        - **80% Conformal Coverage Check:** Validated against held-out validation residuals to prevent overconfident forecasts.
        - **Managerial Review Gate:** Decision support only. Automated screen reallocation is prohibited.
        """)

        st.markdown("#### 📈 Uncertainty Calibration & Residual Diagnostics")
        c_diag1, c_diag2 = st.columns(2)
        with c_diag1:
            calib_plot = Path("reports/calibration_curve.png")
            if calib_plot.exists():
                st.image(str(calib_plot), caption="80% Prediction Interval Calibration Reliability Curve")
            else:
                st.caption("Calibration plot not generated.")

        with c_diag2:
            res_plot = Path("reports/residual_plot.png")
            if res_plot.exists():
                st.image(str(res_plot), caption="LightGBM Residuals vs Prediction on Held-Out Test Split")
            else:
                st.caption("Residual plot not available.")
