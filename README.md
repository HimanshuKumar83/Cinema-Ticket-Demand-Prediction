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
All 5 candidate models were evaluated on the exact same held-out chronological test partition (**4,500 shows**, 2025-01-12 to 2025-03-19):

| Model Architecture | Objective / Approach | Test MAE | Test RMSE | Test WAPE (%) | Occupancy MAE | Unclipped Violation Rate |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest Regressor** | Non-linear ensemble (100 trees) | **12.66** | **17.29** | **8.68%** | **0.0558** | 2.73% |
| **LightGBM Regressor (L1)** | L1 loss MAE optimization | **13.53** | **18.40** | **9.28%** | **0.0594** | 6.47% |
| **XGBoost Regressor** | Gradient boosted trees | 14.21 | 19.23 | 9.74% | 0.0621 | 4.22% |
| **Ridge Regression** | Scaled regularized linear benchmark | 17.47 | 22.36 | 11.98% | 0.0802 | 6.33% |
| **Hierarchical Baseline** | 4-level mean occupancy fallback | 45.98 | 59.68 | 31.53% | 0.1815 | 0.00% |

*Post-processing guarantees: All raw predictions are strictly clamped within $[0, \text{screen\_capacity}]$, ensuring a zero operational capacity violation rate in production.*

---

## ⚖️ Uncertainty Calibration & Conformal Prediction
- **Calibration Method:** Non-parametric conformal prediction using validation residual quantiles:
  - $10^{\text{th}}$ Percentile ($q_{10}$): $-28.84$ tickets
  - $90^{\text{th}}$ Percentile ($q_{90}$): $+11.25$ tickets
- **Target Coverage:** 80.0%
- **Empirical Test Set Coverage:** **79.09%** (closely matching theoretical calibration)
- **Mean Interval Width:** 38.7 tickets

---

## 🧠 TreeSHAP Feature Attributions
Global feature importance computed via TreeSHAP on test observations:
1. `advance_bookings_t48`: Mean $|SHAP| = 49.51$ (Dominant signal)
2. `booking_velocity_24h`: Mean $|SHAP| = 15.82$ (Recent demand acceleration)
3. `screen_capacity`: Mean $|SHAP| = 15.37$ (Room ceiling constraint)
4. `is_blockbuster`: Mean $|SHAP| = 9.64$ (Title tier)
5. `slot_hist_avg_demand`: Mean $|SHAP| = 8.51$ (Time slot bias)
6. `booking_velocity_previous_24h`: Mean $|SHAP| = 4.83$ (Momentum baseline)
7. `movie_hist_avg_demand`: Mean $|SHAP| = 4.44$ (Title track record)
8. `ticket_price`: Mean $|SHAP| = 3.90$ (Price elasticity)

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
