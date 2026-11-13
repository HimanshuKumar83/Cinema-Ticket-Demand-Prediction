# Cinema Ticket Demand Forecasting (Show-Level MVP)

**Algions AI Labs — Machine Learning Engineer Assessment**  
**Candidate:** Senior ML Engineer  
**Stack:** Python 3.10+, LightGBM, Scikit-Learn, FastAPI, React 18 (Vite), Streamlit, Pytest  

---

## 📌 Executive Summary

This project delivers an end-to-end Machine Learning system that predicts **movie show attendance (`tickets_sold`)** and **occupancy rate** strictly **48 hours in advance of showtime**.

It solves two critical operational decisions for cinema complex managers:
1. **Auditorium Screen Allocation:** Deciding whether to upgrade a film to a larger screening room to avoid sold-out demand truncation or downsize to a smaller boutique screen.
2. **Staff Scheduling:** Accurately scheduling floor ushers, ticket scanners, and concession cashiers based on the **upper 80% prediction interval bound** to prevent long queues.

**Core Benchmark Result:** On a chronologically held-out test set of **22,764 screenings**, the primary candidate model (**LightGBM**) achieved **39.14% WAPE (a 37.7% MAE reduction over the historical baseline)**.

---

## 🎯 Framing & Operational Rules

- **Prediction Unit:** Single show-slot: `(cinema_code, film_code, date, show_time)`
- **Target Variable:** `tickets_sold` (show attendance); secondary metric is `occupancy = tickets_sold / capacity`
- **Prediction Horizon:** Strict **T - 48h** lookahead. Only historical data from dates **<= show_date - 2 days** and predetermined schedule attributes (auditorium capacity, calendar slot) are used.
- **Decision Boundary:** **Decision-support only.** Human managerial review is required; this is not an autonomous high-stakes decision maker.
- **Zero Leakage:** Target-derived columns (`occu_perc`), ex-post financials (`total_sales`, `ticket_price`), and post-show metrics (`tickets_out`, `ticket_use`) are strictly excluded from input features.

---

## 📊 Held-Out Test Set Results (2018-09-30 to 2018-11-04, N=22,764)

Every metric below was directly computed on the held-out test set:

| Model Architecture | MAE (Tickets) | RMSE (Tickets) | WAPE (%) | Occupancy MAE | Asymmetric Cost (cu=1.5, co=1.0) | Clipped at Capacity (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Hierarchical Baseline** | 77.63 | 154.01 | 62.83% | 0.1148 | 89.07 | 0.00% |
| **Ridge Regression (Occupancy)** | 67.50 | 137.55 | 54.63% | 0.1081 | 91.84 | 0.00% |
| **LightGBM (Tickets L1) ★** | **48.36** | **108.96** | **39.14%** | **0.0832** | **65.66** | **0.54%** |

### Key Highlights:
- **Baseline Outperformed:** LightGBM reduces MAE by **37.7%** (48.36 vs 77.63 tickets) and cuts WAPE from 62.83% to 39.14%.
- **Asymmetric Operational Cost:** Penalizing under-forecasting (1.5x penalty: risking long concession queues) vs over-forecasting (1.0x penalty: idle staff) shows LightGBM reduces operational cost by **26.3%** (65.66 vs 89.07).
- **Uncertainty Calibration:** Validation residual quantiles yield Q10 = -41.21 tickets, Q90 = +36.94 tickets. Empirical test coverage is **69.28%**.
- **Cold-Start Slicing:** Established films achieve **38.01% WAPE**, while new films debut at **73.33% WAPE**, triggering an automated warning flag in the UI for managers.

---

## 🚀 Quickstart Guide (Windows PowerShell)

### One-Click Interactive Launcher
From the project root directory:
```powershell
.\run_local.ps1
```
*(Provides an interactive menu to start the React App, Streamlit Dashboard, FastAPI server, or run the test suites).*

---

### Step-by-Step Manual Commands

#### 1. Setup & Dependencies
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

#### 2. Download Dataset
Downloads the Kaggle dataset via Kaggle CLI and stores the file's SHA-256 in `config.yaml`:
```powershell
python scripts/download_data.py
```
*(Manual fallback: Download `cinemaTicket_Ref.csv` from Kaggle, place it into `data/raw/`, and re-run).*

#### 3. Audit & Clean Data
Audits schema, duplicate keys, and leakage, then generates clean splits:
```powershell
python -m src.data.audit
python -m src.data.clean
```

#### 4. Train Models
Runs the full training pipeline, logs metrics, and saves the model bundle to `artifacts/`:
```powershell
python -m src.train
```

#### 5. Run Automated Tests
```powershell
# Backend Tests (Leakage independence, capacity bounds, split order, API validation)
pytest -v

# Frontend Tests (Component rendering, staffing logic, modal triggers)
npm --prefix frontend run test
```

#### 6. Launch Applications
- **Modern React Frontend:** `npm --prefix frontend run dev` -> Open `http://localhost:5173`
- **FastAPI Microservice (Serves React UI):** `uvicorn app.api:app --reload --port 8000` -> Open `http://localhost:8000`
- **Streamlit Operations Dashboard:** `streamlit run app/dashboard.py` -> Open `http://localhost:8501`

---

## ☁️ Step-by-Step Deployment Guide

### Option 1: Streamlit Community Cloud (Free 1-Click Hosting)
1. Go to share.streamlit.io and sign in with GitHub.
2. Click **New app**.
3. Select this repository: `HimanshuKumar83/Cinema-Ticket-Demand-Prediction`.
4. Set **Main file path**: `app/dashboard.py`.
5. Click **Deploy!**. The app will be live on a public URL in ~2 minutes.

*(The model bundle `artifacts/model.joblib` and lookup table `artifacts/history.parquet` are included in the repository, so the app runs immediately without re-training).*

---

### Option 2: Docker & Docker Compose
A production `Dockerfile` and `docker-compose.yml` are provided:
```bash
# Build and run containers in background
docker compose up -d --build
```
- Dashboard: `http://localhost:8501`
- FastAPI & Swagger Docs: `http://localhost:8000/docs`

To stop:
```bash
docker compose down
```

---

## 🎯 Acceptance Tests Mapping

| Assessment Requirement | Demonstration Command / Target | Verification Evidence |
| :--- | :--- | :--- |
| **1. Reproduce split & training** | `python -m src.train` | Generates identical splits in `artifacts/split_info.json` and trains pipeline end-to-end. |
| **2. Baseline vs candidate metrics** | `python -c "import json; print(json.load(open('artifacts/metrics.json'))['models'])"` | Saved in `artifacts/metrics.json` and detailed in `reports/technical_report.md`. |
| **3. Real prediction via API/App** | `python -m src.predict` or `http://localhost:8000/docs` | Predictor loads `model.joblib` and `history.parquet`; returns live forecasts and intervals. |
| **4. Explain one prediction & limitations** | Check `tests/test_pipeline.py` or inspect API response | Returns local input feature drivers, cold-start diagnostic, and limitation notes. |
| **5. Show no future leakage or secrets** | `pytest -k test_leakage_future_independence` | Formally asserts features for date D are invariant to future perturbations; zero credentials committed. |

---

## 🔒 Security & Data Compliance
- **Zero Credentials Committed:** API keys and credentials are not stored, printed, logged, or tracked.
- **Dataset Attribution:** Kaggle Cinema Tickets dataset by Möbius (CC BY-NC-SA 4.0). The raw CSV is not redistributed; it is downloaded locally and verified via SHA-256.
- **Self-Contained Artifacts:** Model weights and lookup tables are committed in `artifacts/` to enable immediate evaluation and deployment.
