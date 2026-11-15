# CinemaOps AI - Technical Machine Learning Report

## Executive Summary
This report presents the complete research, methodology, data architecture, multi-model evaluation, and operational deployment of **CinemaOps AI**, a show-level decision-support system built for cinema complex operations. 

The primary business objective is forecasting the final ticket volume (`actual_tickets_sold`) for individual movie showtimes exactly **48 hours prior to screening time ($T-48\text{h}$)** to enable proactive theater room reallocation and front-of-house staff scheduling.

---

## 1. Problem Formulation & Operational Boundary
- **Unit of Analysis:** A single scheduled screening defined by `(cinema_id, screen_id, movie_id, show_datetime)`.
- **Primary Prediction Target:** `actual_tickets_sold` $\in [0, \text{screen_capacity}]$.
- **Secondary Target:** `occupancy_rate` = $\frac{\text{actual_tickets_sold}}{\text{screen_capacity}} \in [0.0, 1.0]$.
- **Prediction Cutoff:** Exactly $T-48\text{h}$ prior to `show_datetime`. All inputs after $T-48\text{h}$ are strictly inaccessible.
- **Decision-Support Mandate:** Advisory recommendations with mandatory human operational review. Never an automated high-stakes decision maker.

---

## 2. Dataset Reconstruction & Simulation Methodology
### 2.1 The Legacy Dataset Limitation
The public Kaggle Cinema Ticket dataset (`cinemaTicket_Ref.csv`) documents aggregated post-show totals (`tickets_sold`, `total_sales`, `tickets_out`) across 2018. However, it lacks granular transaction timestamps. As a result, calculating true point-in-time advance bookings ($T-48\text{h}$) and booking velocities ($V_{24}$) from the raw Kaggle dataset would introduce lookahead bias or rely on unverified proxies.

### 2.2 Production-Grade Simulation Pipeline
To ensure fidelity to real-world cinema operations, we built a non-linear booking simulation engine (`src/data/booking_simulator.py` and `src/data/generate_data.py`):
- **Scale:** 30,000 scheduled shows across 18 continuous months (2024-01-01 to 2025-03-19).
- **Entities:** 30 cinema complexes (Tier-1 and Tier-2 metropolitan areas) with 3 to 5 auditoriums per complex (capacities 80 to 450 seats). 100 movie titles across 10 genres with staggered release dates.
- **Granular Transactions:** 2,065,328 individual timestamped booking transactions logged in `data/simulated/booking_events.parquet`.
- **Non-Linear Dynamics:** Modeled arrival distributions via Beta curves shifting booking velocity closer to showtime, higher advance ratios for weekend blockbusters, realistic 3% cancellation rates, and hard auditorium capacity ceilings.

---

## 3. Leak-Free Feature Engineering
All features are generated strictly from information available at or before $T-48\text{h}$:

1. **Advance Telemetry ($T-48\text{h}$):**
   - `advance_bookings_t48`: Cumulative valid tickets booked $\le T-48\text{h}$.
   - `booking_velocity_24h`: Net tickets booked in window $(T-72\text{h}, T-48\text{h}]$.
   - `booking_velocity_previous_24h`: Net tickets booked in window $(T-96\text{h}, T-72\text{h}]$.
   - `booking_acceleration`: $V_{24} - V_{24,\text{prev}}$.
   - `advance_occupancy_t48`: $\frac{B_{48}}{\text{Capacity}}$.
   - `velocity_ratio`: $\frac{V_{24}}{B_{48} + 1.0}$.

2. **Schedule & Movie Attributes:**
   - `screen_capacity`, `ticket_price`, `movie_release_age_days`, `is_blockbuster`, `is_weekend`, `is_holiday`.

3. **Shifted Historical Demand Priors:**
   - Precomputed strictly on the training partition ($N=21,000$) to guarantee zero lookahead contamination: `movie_hist_avg_demand`, `cinema_hist_avg_demand`, `slot_hist_avg_demand`, `genre_hist_avg_demand`, `weekday_hist_avg_demand`.

4. **Cold-Start Flags:**
   - `is_new_movie` (0 historical screenings in training).
   - `is_new_cinema` (0 historical screenings in training).
   - `is_low_history` ($< 10$ historical shows).

---

## 4. Multi-Model Chronological Benchmark
The dataset was partitioned sequentially without shuffling:
- **Train:** 21,000 shows (2024-01-01 to 2024-11-07) - 70%
- **Validation:** 4,500 shows (2024-11-07 to 2025-01-12) - 15%
- **Test:** 4,500 shows (2025-01-12 to 2025-03-19) - 15%

### 4.1 Comparative Results on Held-Out Test Set:
| Model Architecture | Objective / Approach | Test MAE | Test RMSE | Test WAPE (%) | Occupancy MAE | Unclipped Capacity Violation Rate |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Hierarchical Baseline** | 4-level mean occupancy fallback | 45.98 | 59.68 | 31.53% | 0.1815 | 0.00% |
| **Ridge Regression** | Scaled linear regressor | 17.47 | 22.36 | 11.98% | 0.0802 | 6.33% |
| **XGBoost Regressor** | Gradient boosted trees | 14.21 | 19.23 | 9.74% | 0.0621 | 4.22% |
| **LightGBM Regressor** | L1 loss optimization | 13.53 | 18.40 | 9.28% | 0.0594 | 6.47% |
| **Random Forest** | Non-linear ensemble | 12.66 | 17.29 | 8.68% | 0.0558 | 2.73% |

*Post-processing clips all model outputs strictly to $[0, \text{screen\_capacity}]$, completely eliminating live operational capacity violations.*

---

## 5. Explainability & Uncertainty Quantification
### 5.1 Global & Local Feature Attributions (TreeSHAP)
Computing TreeSHAP values over the held-out test split revealed the dominant predictive factors:
1. `advance_bookings_t48`: Mean $|SHAP| = 49.51$
2. `booking_velocity_24h`: Mean $|SHAP| = 15.82$
3. `screen_capacity`: Mean $|SHAP| = 15.37$
4. `is_blockbuster`: Mean $|SHAP| = 9.64$
5. `slot_hist_avg_demand`: Mean $|SHAP| = 8.51$
6. `booking_velocity_previous_24h`: Mean $|SHAP| = 4.83$
7. `movie_hist_avg_demand`: Mean $|SHAP| = 4.44$
8. `ticket_price`: Mean $|SHAP| = 3.90$

### 5.2 Conformal Prediction Calibration
Using validation residual quantiles ($q_{10} = -28.84$, $q_{90} = +11.25$):
- **Target Confidence:** 80.0%
- **Empirical Held-Out Test Coverage:** **79.09%**
- **Mean Interval Width:** 38.7 tickets

---

## 6. Verification & Test Suite
The automated test suite (`tests/test_pipeline.py`) validates:
1. Simulation integrity and capacity ceiling obedience.
2. Complete absence of target and ex-post variables in feature stores.
3. Strict chronological split ordering without temporal overlap.
4. Guaranteed prediction bound containment: $\forall \hat{y}, 0 \le \hat{y} \le \text{Capacity}$.
5. Dynamic response sensitivity (no hardcoded outputs).
6. Cold-start detection and advisory generation.
7. Presence of all 5 candidate models in serialized artifacts.
8. Single and batch prediction endpoint validation with Pydantic v2 schemas.
