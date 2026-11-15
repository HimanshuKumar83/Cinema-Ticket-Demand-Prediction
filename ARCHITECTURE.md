# CinemaOps AI - System Architecture Document

## 1. System Overview
**CinemaOps AI** is an enterprise-grade machine learning decision-support system engineered for multi-screen cinema operations managers. It generates calibrated show-level attendance forecasts and auditorium room/staffing recommendations strictly **48 hours prior to scheduled showtimes ($T-48\text{h}$)**.

```
+--------------------------------------------------------------------------------------------------+
|                                    CINEMAOPS AI ARCHITECTURE                                      |
+--------------------------------------------------------------------------------------------------+

   [ Point-of-Sale & Online Ticketing Streams ]
                       |
                       v
         +----------------------------+
         | src.data.booking_simulator |  (Granular non-linear ticket arrival curves & channels)
         +----------------------------+
                       |
                       v
            +---------------------+
            | data/simulated/     |
            | - shows.parquet     |  (30,000+ shows across 18 months, 30 cinemas, 100 movies)
            | - booking_events.pq |  (2,065,328 granular timestamped booking transactions)
            +---------------------+
                       |
                       v
         +----------------------------+
         | src.data.preprocessing     |  (Schema validation, capacity bounds, chronological split)
         +----------------------------+
                       |
                       v
            +---------------------+
            | data/processed/     |
            | - train.parquet     |  (70% - 21,000 shows: 2024-01-01 to 2024-11-07)
            | - val.parquet       |  (15% - 4,500 shows:  2024-11-07 to 2025-01-12)
            | - test.parquet      |  (15% - 4,500 shows:  2025-01-12 to 2025-03-19)
            +---------------------+
                       |
                       v
         +----------------------------+
         | src.features.build_features|  (T-48 Telemetry, 24h Velocity, Shifted Historical Priors)
         +----------------------------+
                       |
                       v
     +------------------------------------------------------------------+
     |                       MULTI-MODEL BENCHMARK                      |
     |  1. Hierarchical Baseline (4-Level Occupancy Fallback)           |
     |  2. Scaled Ridge Regression (Regularized Linear Benchmark)       |
     |  3. Random Forest Regressor (Non-linear Ensemble)                |
     |  4. XGBoost Regressor (Gradient Boosted Trees)                   |
     |  5. LightGBM Regressor (L1 Loss MAE Optimization) [Selected]     |
     +------------------------------------------------------------------+
                       |
                       +-----------------------------------+
                       |                                   |
                       v                                   v
         +----------------------------+      +----------------------------+
         | Conformal Prediction       |      | TreeSHAP Explainability    |
         | (80% Residual Quantiles)   |      | (Global & Local Drivers)   |
         | Coverage: 79.09%           |      | Positive & Negative Forces |
         +----------------------------+      +----------------------------+
                       |                                   |
                       +-----------------+-----------------+
                                         |
                                         v
                         +-------------------------------+
                         | artifacts/model.joblib        |
                         | artifacts/model_benchmark.json|
                         | artifacts/metrics.json        |
                         +-------------------------------+
                                         |
                       +-----------------+-----------------+
                       |                                   |
                       v                                   v
         +----------------------------+      +----------------------------+
         | FastAPI Microservice       |      | Frontend Interfaces        |
         | (app/api.py on Port 8000)  |      | 1. Streamlit Dashboard     |
         | - GET /health              |      |    (app/dashboard.py:8501) |
         | - GET /model/info          |      | 2. Modern React SaaS UI    |
         | - POST /predict            |      |    (frontend/ on Vite:5173)|
         | - POST /predict/batch      |      +----------------------------+
         | - GET /model/benchmark     |
         | - POST /explain            |
         +----------------------------+
```

---

## 2. Temporal Zero-Leakage Boundary ($T-48\text{h}$)
To ensure operational reality, the prediction cutoff timestamp is strictly defined:

$$\text{Cutoff Timestamp} = \text{Show Showtime} - 48\text{ hours}$$

All telemetry, velocities, and historical records must satisfy:

$$\text{Observation Timestamp} \le \text{Cutoff Timestamp}$$

1. **Permitted Signals at $T-48\text{h}$:**
   - Cumulative advance tickets booked up to $T-48\text{h}$ (`advance_bookings_t48`).
   - Recent 24h booking velocity in interval $(T-72\text{h}, T-48\text{h}]$ (`booking_velocity_24h`).
   - Prior 24h booking velocity in interval $(T-96\text{h}, T-72\text{h}]$ (`booking_velocity_previous_24h`).
   - Booking acceleration: $V_{24} - V_{24,\text{prev}}$.
   - Advance occupancy ratio: $\frac{B_{48}}{\text{Capacity}}$.
   - Scheduled show attributes: screen capacity, show slot, average ticket price, weekend/holiday status, movie genre, release age in days.
   - Historical aggregate demand computed purely from completed shows strictly prior to the cutoff date.

2. **Strictly Prohibited Information (Zero Lookahead):**
   - Bookings or ticket cancellations occurring after $T-48\text{h}$.
   - Same-day box office walk-ins.
   - Total final ticket revenue or total box-office sales.
   - Ex-post occupancy rates.

---

## 3. Modeling Portfolio & Benchmark Results
Evaluated on **4,500 held-out chronological test shows** (2025-01-12 to 2025-03-19):

| Model Architecture | MAE (Tickets) | RMSE | WAPE (%) | Occupancy MAE | Capacity Violation Rate |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest Regressor** | **12.66** | **17.29** | **8.68%** | **0.0558** | 2.73% |
| **LightGBM Regressor (L1)** | **13.53** | **18.40** | **9.28%** | **0.0594** | 6.47% |
| **XGBoost Regressor** | **14.21** | **19.23** | **9.74%** | **0.0621** | 4.22% |
| **Ridge Regression** | 17.47 | 22.36 | 11.98% | 0.0802 | 6.33% |
| **Hierarchical Baseline** | 45.98 | 59.68 | 31.53% | 0.1815 | 0.00% |

*LightGBM Regressor and Random Forest demonstrate superior performance with over 70% error reduction relative to the domain baseline.*

---

## 4. Operational Uncertainty & Calibration
Uncertainty intervals are calibrated via **Conformal Prediction Residual Quantiles** evaluated on held-out validation data:
- Residual $10^{\text{th}}$ Percentile ($q_{10}$): $-28.84$ tickets
- Residual $90^{\text{th}}$ Percentile ($q_{90}$): $+11.25$ tickets
- **Empirical Held-Out Test Coverage:** **79.09%** (closely matching the theoretical 80.0% confidence level).
- Prediction Interval: $[\max(0, \hat{y} + q_{10}), \min(\text{Capacity}, \hat{y} + q_{90})]$.

---

## 5. Staffing & Screen Reallocation Rules
The application translates raw attendance forecasts into concrete, operational managerial actions:
1. **Screen Allocation:**
   - **Upgrade Trigger:** Projected Occupancy $\ge 85\%$ or Upper Bound $\ge \text{Capacity}$ $\to$ Recommend upgrade to 350+ seat auditorium to avoid demand truncation.
   - **Downgrade Trigger:** Projected Occupancy $\le 35\%$ and Capacity $> 200$ $\to$ Recommend shifting to compact room (80–150 seats) to optimize HVAC and cleaning costs.
   - **Maintain Status:** Projected Occupancy between $35\%$ and $85\%$ $\to$ Optimal room sizing.
2. **Staffing Formulas:**
   - **Box Office / Ticket Scanners:** $\max(1, \lceil \frac{\text{Upper Bound}}{150} \rceil)$
   - **Concession Cashiers:** $\max(1, \lceil \frac{\text{Upper Bound}}{70} \rceil)$
   - **Floor Ushers:** $\max(1, \lceil \frac{\text{Upper Bound}}{120} \rceil)$
   - **Concession Units (Popcorn/Beverage):** $\text{Predicted Tickets} \times 0.70$
