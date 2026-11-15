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
Evaluated on **4,500 held-out chronological test shows** (2025-01-13 to 2025-03-20):

| Model Architecture | MAE (Tickets) | RMSE | WAPE (%) | Occupancy MAE | Asymmetric Cost (1.5x) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **LightGBM Regressor (L1)** ★ | **11.27** | **15.09** | **11.22%** | **0.0481** | **12.86** |
| **Random Forest Regressor** | **12.15** | **16.04** | **12.10%** | **0.0519** | 13.80 |
| **XGBoost Regressor** | **12.19** | **16.30** | **12.14%** | **0.0520** | 13.30 |
| **Ridge Regression** | 16.68 | 20.49 | 16.61% | 0.0821 | 18.01 |
| **Hierarchical Baseline** | 59.12 | 76.40 | 58.86% | 0.2295 | 60.98 |

*LightGBM Regressor demonstrates superior performance with an 80.9% error reduction relative to the hierarchical domain baseline.*

---

## 4. Operational Uncertainty & Calibration
Uncertainty intervals are calibrated via **Capacity-Normalized Conformal Prediction** evaluated on held-out validation data:
- Normalized Quantiles: $q_{10,\text{norm}} = -0.0763$, $q_{90,\text{norm}} = +0.0572$
- **Empirical Held-Out Test Coverage:** **76.64%** (Warm-Start: 83.88%, Cold-Start: 67.69%).
- Subgroup Interval Widths: 16.0 tickets on $\le 150$-seat rooms, 28.5 tickets on $151-300$-seat rooms, 50.7 tickets on $>300$-seat rooms.
- Prediction Interval: $[\max(0, \hat{y} + q_{10,\text{norm}} \cdot \text{Capacity}), \min(\text{Capacity}, \hat{y} + q_{90,\text{norm}} \cdot \text{Capacity})]$.
- Reliability Curve: `reports/calibration_curve.png`.

---

## 5. Staffing & Screen Reallocation Rules
The application translates raw attendance forecasts into concrete, operational managerial actions:
1. **Screen Allocation:**
   - **Upgrade Trigger:** Projected Occupancy $\ge 85\%$ or Upper Bound $\ge \text{Capacity}$ $\to$ Recommend upgrade to 350+ seat auditorium to avoid demand truncation.
   - **Downgrade Trigger:** Projected Occupancy $\le 35\%$ and Capacity $> 180$ $\to$ Recommend shifting to compact room (80–150 seats) to optimize HVAC and cleaning costs.
   - **Maintain Status:** Projected Occupancy between $35\%$ and $85\%$ $\to$ Optimal room sizing.
2. **Staffing Formulas (Centralized in `src/operational/advisory.py`):**
   - **Box Office / Ticket Scanners:** $\max(1, \lceil \frac{\text{Upper Bound}}{150} \rceil)$
   - **Concession Cashiers:** $\max(1, \lceil \frac{\text{Upper Bound}}{70} \rceil)$
   - **Floor Ushers:** $\max(1, \lceil \frac{\text{Upper Bound}}{120} \rceil)$
   - **Cleaning Crew:** $\max(1, \lceil \frac{\text{Capacity}}{100} \rceil)$
   - **Concession Units (Popcorn/Beverage):** $\text{round}(\text{Predicted Tickets} \times 0.70)$

