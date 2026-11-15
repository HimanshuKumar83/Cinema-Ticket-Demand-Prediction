# CinemaOps AI – Show-Level Cinema Ticket Demand Forecasting & Staffing Decision Support System

**Algions AI Labs Private Limited — Senior Machine Learning Engineer Assessment**  
**Stack:** Python 3.10+, PowerShell (Windows), LightGBM, XGBoost, Scikit-Learn, SHAP, FastAPI, Streamlit, React 18 (Vite), Pytest, Vitest

---

## 📌 Executive Summary
**CinemaOps AI** is an enterprise-grade machine learning decision-support system engineered to predict final show-level attendance volume (`actual_tickets_sold`) and auditorium occupancy for movie screenings strictly **48 hours prior to scheduled showtimes ($T-48\text{h}$)**.

The system assists cinema complex managers in two primary operational decisions:
1. **Auditorium Screen Allocation:** Reallocating shows to larger auditorium halls ($300+$ seats) to avoid sold-out demand truncation or downsizing under-performing titles to compact boutique screens ($80-150$ seats) to cut HVAC and operational overhead.
2. **Front-of-House Staffing Scheduling:** Scheduling ticket scanners, ushers, cleaning crews, and concession inventory against calibrated **80% prediction intervals** to mitigate asymmetric under-staffing costs.

> **Operational Decision-Support Mandate:** Advisory decision support only. All automated recommendations require human managerial review prior to room reallocation or roster publishing.

---

## 🎯 Framing & Operational Boundary
- **Prediction Unit:** Single scheduled screening defined by `(cinema_id, screen_id, movie_id, show_datetime)`.
- **Primary Target:** `actual_tickets_sold` $\in [0, \text{screen_capacity}]$.
- **Secondary Target:** `occupancy_rate` = $\frac{\text{actual_tickets_sold}}{\text{screen_capacity}} \in [0.0, 1.0]$.
- **Prediction Horizon:** Strictly $T-48\text{h}$ prior to showtime (`show_datetime - 48 hours`). All signals after $T-48\text{h}$ are strictly excluded.
- **Zero Lookahead Leakage:** Features use strictly historical data up to $T-48\text{h}$ (advance bookings, 24h booking velocity, and strictly shifted historical averages). Target-derived columns (`total_sales`, `final_occupancy`, `tickets_out`) are completely excluded.

---

## 📊 Held-Out Test Set 5-Model Benchmark
All 5 candidate models were evaluated on the exact same held-out chronological test partition (**4,500 shows**, 2025-01-13 to 2025-03-20):

| Model Architecture | Objective / Approach | Test MAE | Test RMSE | Test WAPE (%) | Occupancy MAE | Asymmetric Cost (1.5x) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **LightGBM Regressor (L1)** ★ | L1 loss MAE optimization | **11.27** | **15.09** | **11.22%** | **0.0481** | **12.86** |
| **Random Forest Regressor** | Non-linear ensemble (100 trees) | 12.15 | 16.04 | 12.10% | 0.0519 | 13.80 |
| **XGBoost Regressor** | Gradient boosted trees | 12.19 | 16.30 | 12.14% | 0.0520 | 13.30 |
| **Ridge Regression** | Scaled regularized linear benchmark | 16.68 | 20.49 | 16.61% | 0.0821 | 18.01 |
| **Hierarchical Baseline** | 4-level mean occupancy fallback | 59.12 | 76.40 | 58.86% | 0.2295 | 60.98 |

*Post-processing guarantees: All raw predictions are strictly clamped within $[0, \text{screen\_capacity}]$, ensuring a zero operational capacity violation rate in production.*

---

## ❄️ Empirical Cold-Start vs. Warm-Start Evaluation
Following extended lifecycle simulation with 37 train/test overlapping movies, the held-out chronological test partition contains both warm-start and cold-start titles:

| Evaluation Slice | Definition | Test Shows ($N$) | Test MAE | Test RMSE | Test WAPE (%) | Occupancy MAE | 80% Coverage |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Warm-Start Shows** | Movie observed in training split | **2,488 (55.3%)** | **9.47** | **12.64** | **10.84%** | **0.0398** | **83.88%** |
| **Cold-Start Shows** | New movie debut (zero train shows) | **2,012 (44.7%)** | **13.50** | **17.65** | **11.58%** | **0.0584** | **67.69%** |
| **Adequate History** | Screening count $\ge 10$ shows | 2,373 (52.7%) | 9.48 | 12.66 | 10.81% | 0.0397 | 83.82% |
| **Low History** | Screening count $< 10$ shows | 2,127 (47.3%) | 13.27 | 17.40 | 11.58% | 0.0575 | 68.64% |

---

## ⚖️ Capacity-Normalized Conformal Prediction (80% Interval)
- **Method:** Capacity-Normalized Conformal Calibration ($e_{\text{norm}} = \frac{y - \hat{y}}{\text{capacity}}$):
  - Normalized Quantiles: $q_{10,\text{norm}} = -0.0763$, $q_{90,\text{norm}} = +0.0572$
  - Unnormalized Residuals: $q_{10} = -17.61$, $q_{90} = +14.48$
- **Target Coverage:** 80.0%
- **Empirical Test Set Coverage:** **76.64%** (Warm: 83.88%, Cold: 67.69%)
- **Subgroup Calibration by Room Size:**
  - *Small Screens ($\le 150$ seats):* Mean interval width: **16.0 tickets** (vs. 31.5 for homoscedastic)
  - *Medium Screens ($151-300$ seats):* Mean interval width: **28.5 tickets** (76.24% coverage)
  - *Large Screens ($> 300$ seats):* Mean interval width: **50.7 tickets** (83.74% coverage vs. 63.28% for homoscedastic)
- **Reliability Diagram:** Saved to `reports/calibration_curve.png`.

---

## 🧠 TreeSHAP Feature Attributions
Global feature importance computed via TreeSHAP on test observations:
1. `advance_bookings_t48`: Mean $|SHAP| = 61.22$ (Dominant demand anchor)
2. `is_blockbuster`: Mean $|SHAP| = 15.36$ (Title tier premium)
3. `booking_velocity_24h`: Mean $|SHAP| = 12.99$ (Recent momentum)
4. `screen_capacity`: Mean $|SHAP| = 8.18$ (Room ceiling constraint)
5. `slot_hist_avg_demand`: Mean $|SHAP| = 5.74$ (Time slot bias)
6. `movie_release_age_days`: Mean $|SHAP| = 5.46$ (Theatrical decay curve)
7. `booking_velocity_previous_24h`: Mean $|SHAP| = 4.84$ (Acceleration baseline)
8. `velocity_ratio`: Mean $|SHAP| = 3.21$ (Surge velocity multiplier)

---

## 🛠️ Data Architecture & Simulation Disclosure
*Disclosure:* Public datasets (e.g. Kaggle Cinema Ticket) only record post-show daily aggregated totals without transaction timestamps, making true point-in-time advance bookings and velocity calculation impossible without lookahead artifacts.

To reflect authentic operations, this project includes a production-grade simulation pipeline (`src/data/booking_simulator.py` and `src/data/generate_data.py`):
- **30,000 scheduled shows** over 18 months across 30 cinema complexes (Tier-1 and Tier-2 cities).
- **2,065,328 granular timestamped booking transactions** (`data/simulated/booking_events.parquet`).
- Non-linear arrival curves (Beta distributions), weekend velocity surges, channel distributions (online vs box office), realistic 3% cancellation rates, and hard screen capacity saturation.

---

## 🚀 Quickstart & Local Execution Guide (Windows PowerShell)

### Option 1: One-Click Interactive Launcher
From the project root:
```powershell
.\run_local.ps1
```
*(Provides an interactive menu to launch the React App, Streamlit Dashboard, FastAPI Server, or run test suites).*

---

### Option 2: Step-by-Step Manual Execution

#### 1. Setup Virtual Environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

#### 2. Generate Dataset & Clean Chronological Splits
```powershell
python -m src.data.generate_data
python -m src.data.preprocessing
```

#### 3. Run Multi-Model Training & Benchmark
```powershell
python -m src.train
```

#### 4. Run Automated Test Suites
```powershell
# Run backend pytest suite (8 comprehensive tests)
pytest -v

# Run React frontend vitest suite (4 UI tests)
npm --prefix frontend run test
```

#### 5. Launch User Interfaces & Microservices
- **FastAPI Microservice (Backend):**
  ```powershell
  uvicorn app.api:app --reload --port 8000
  ```
  - Interactive Swagger Docs: `http://localhost:8000/docs`
  - OpenAPI Schema: `http://localhost:8000/openapi.json`
- **Streamlit Decision-Support Dashboard:**
  ```powershell
  streamlit run app/dashboard.py
  ```
  - Accessible at `http://localhost:8501`
- **Modern React SaaS Frontend (Vite):**
  ```powershell
  npm --prefix frontend run dev
  ```
  - Accessible at `http://localhost:5173`

---

## 🔌 FastAPI Microservice Endpoints
| HTTP Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Container health probe & model readiness status |
| `GET` | `/model/info` | Operational model specifications, features, & quantiles |
| `POST` | `/predict` | Single show attendance forecast, intervals, & staffing advisory |
| `POST` | `/predict/batch` | Multi-show roster forecast (up to 100 shows) |
| `GET` | `/metrics` | Held-out test metrics, conformal stats, & slice evaluations |
| `GET` | `/model/benchmark` | 5-model comparative benchmark table |
| `POST` | `/explain` | Local TreeSHAP attribution waterfall for individual shows |

---

## 📁 Repository Structure
```
Cinema_Ticket/
│
├── .gitignore                      # Enforces exclusion of .venv, data, cache, and secrets
├── config.yaml                     # Project configuration & random seeds
├── requirements.txt                # Pinned production Python dependencies
├── run_local.ps1                   # Interactive PowerShell launcher
├── Dockerfile                      # Production container definition
├── docker-compose.yml              # Microservice orchestration
│
├── PROJECT_AUDIT.md                # Initial project inspection & architectural audit
├── ARCHITECTURE.md                 # System architecture, data flow & T-48 boundary
├── TECHNICAL_REPORT.md             # Comprehensive ML research & evaluation report
├── DATA_CARD.md                    # Data lineage, features, & schema documentation
├── DEPLOYMENT.md                   # Cloud & container deployment guide
│
├── src/
│   ├── data/
│   │   ├── booking_simulator.py    # Non-linear arrival curves & ticket simulation
│   │   ├── generate_data.py        # 30,000 shows & 2M+ booking transactions generator
│   │   ├── preprocessing.py        # Schema checks, capacity bounds & chronological splits
│   │   ├── audit.py                # Legacy Kaggle audit & leakage analysis
│   │   └── clean.py                # Legacy Kaggle cleaner & deduplicator
│   ├── features/
│   │   └── build_features.py       # T-48 signals, 24h velocities, shifted historical priors
│   ├── models/
│   │   └── baseline.py             # 4-level hierarchical occupancy fallback baseline
│   ├── explainability/
│   │   └── shap_analysis.py        # TreeSHAP global importance & local driver waterfall
│   ├── evaluation/
│   │   └── metrics.py              # MAE, RMSE, WAPE, Occupancy MAE, capacity checks
│   ├── train.py                    # 5-model training, benchmarking & artifact serialization
│   └── predict.py                  # Production inference, conformal intervals & advisory
│
├── app/
│   ├── api.py                      # FastAPI REST microservice with CORS
│   └── dashboard.py                # Streamlit 7-tab enterprise operations console
│
├── frontend/                       # Modern React 18 SaaS UI (Vite)
│   ├── src/
│   │   ├── components/             # Reusable UI components (Chart, Form, Advisor, Modal)
│   │   ├── test/                   # Vitest frontend unit tests
│   │   ├── api.js                  # Axios/Fetch API client connecting to FastAPI
│   │   └── App.jsx                 # Main application console
│   └── dist/                       # Production pre-built static bundle
│
├── tests/
│   └── test_pipeline.py            # Comprehensive 8-test pytest suite
│
├── artifacts/                      # Serialized models, priors, quantiles & benchmarks
│   ├── model.joblib                # Trained LightGBM + all candidate models
│   ├── model_benchmark.json        # 5-model test comparison metrics
│   ├── metrics.json                # Complete evaluation metrics & operational slices
│   ├── feature_list.json           # Explicit list of 25 modeling features
│   ├── priors.json                 # Precomputed training priors (zero leakage)
│   └── conformal_quantiles.json    # 80% residual quantiles & test coverage
│
└── reports/
    ├── shap_summary.png            # Global TreeSHAP feature attribution plot
    ├── residual_plot.png           # LightGBM test residual diagnostic plot
    └── leakage_table.csv           # Legacy data leakage audit report
```

---

## 🔒 Security & Operational Governance
- **Zero Credentials Committed:** No API keys, passwords, or cloud tokens are stored or committed. Harmless placeholders only in `.env.example`.
- **Git Hygiene:** Clean commit history directly tracked on `main`.
- **Safety Boundary:** Strict decision-support advisory system with human managerial sign-off required for all theater room reallocations and front-of-house shift rosters.
