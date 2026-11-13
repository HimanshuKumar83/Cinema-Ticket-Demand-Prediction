# 🎬 Cinema Ticket Demand Forecasting (Show-Level MVP)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![LightGBM](https://img.shields.io/badge/Model-LightGBM%20L1-green.svg)](https://lightgbm.readthedocs.io/)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/Frontend-React%2018%20%2B%20Vite-61DAFB.svg)](https://react.dev/)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Tests](https://img.shields.io/badge/Tests-Pytest%20%2B%20Vitest%20(10%2F10%20Passing)-brightgreen.svg)]()
[![License](https://img.shields.io/badge/Dataset%20License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

> **Algions AI Labs — Machine Learning Engineer Assessment**  
> **Candidate Track:** Senior ML Engineer  
> **Key Achievement:** Deployed an end-to-end, leakage-free 48-hour demand forecasting system. LightGBM achieved **39.14% WAPE (a 37.7% MAE reduction over historical baseline)** across 22,764 held-out test screenings.

---

## 📌 Executive Summary & Business Framing

Cinema complex operations managers must decide **auditorium room allocation** (swapping to larger halls for surging demand or downsizing for sparse attendance) and **front-of-house staffing levels** (ushers, ticket scanners, concession cashiers) **48 hours in advance of showtime**.

| Dimension | Specification |
| :--- | :--- |
| **Business Decision** | Screen sizing and staff roster planning for upcoming movie screenings |
| **Prediction Unit** | Single show-slot: `(cinema_code, film_code, date, show_time)` |
| **Target Variable** | `tickets_sold` (show attendance volume); also reports `occupancy = tickets_sold / capacity` |
| **Prediction Horizon** | Strict **T - 48h** lead time. Only historical data from dates **<= show_date - 2 days** and predetermined timetable metadata (capacity, calendar slot) are accessible |
| **Decision Boundary** | **Strictly decision-support.** Advisory output requiring human managerial confirmation; not an automated high-stakes decision maker |

---

## 🌟 Interactive Frontends & API

This project provides **two complementary frontends** and a high-performance REST backend:

1. **Modern React Web Dashboard (`frontend/`)**:
   - Modern slate & indigo UI built with **React 18 + Vite + Lucide Icons**.
   - Scenario presets (*Blockbuster Night*, *Afternoon Matinee*, *Cold-Start Debut Film*, *Boutique Screen*).
   - Dynamic SVG Demand vs Capacity chart with **80% prediction interval error whiskers**.
   - Automated **Auditorium Reallocation Alert** & **Shift Staffing Roster Calculator** (ushers, box office, concessions).
   - In-app Model Benchmarks modal displaying real test-set metrics.
2. **Streamlit Operations Dashboard (`app/dashboard.py`)**:
   - Interactive Python-native interface with scenario controls, metric tiles, and human review governance disclaimers.
3. **FastAPI Microservice (`app/api.py`)**:
   - Real-time prediction endpoint (`POST /predict`), health check (`GET /health`), and model inspection (`GET /model-info`). Serves the prebuilt React UI directly at `http://localhost:8000/`.

---

## ☁️ Step-by-Step Deployment Guide

### Option 1: Streamlit Community Cloud (100% Free & Recommended for Live Demo)
Streamlit Cloud offers free hosting directly connected to your GitHub repository:
1. Fork or push this repository to your GitHub account (`https://github.com/HimanshuKumar83/Cinema-Ticket-Demand-Prediction`).
2. Visit **[share.streamlit.io](https://share.streamlit.io/)** and sign in with GitHub.
3. Click **"New app"**:
   - **Repository:** `HimanshuKumar83/Cinema-Ticket-Demand-Prediction`
   - **Branch:** `main`
   - **Main file path:** `app/dashboard.py`
4. Click **"Deploy!"**. Within ~2 minutes, your dashboard will be live on a public URL (e.g. `https://cinema-demand-forecaster.streamlit.app`).

---

### Option 2: Hugging Face Spaces (Free CPU Cloud Hosting)
1. Log in to [Hugging Face](https://huggingface.co/) and navigate to **Spaces** -> **"Create new Space"**.
2. Name your space `cinema-ticket-demand-forecasting` and select **Streamlit** (or **Docker**).
3. Connect your GitHub repository or push directly via Git:
   ```bash
   git remote add space https://huggingface.co/spaces/<your-username>/cinema-ticket-demand-forecasting
   git push space main
   ```
4. The space installs `requirements.txt` automatically and serves the app online.

---

### Option 3: Docker & Docker Compose (Containerized Production)
The repository includes a production-ready `Dockerfile` and `docker-compose.yml`:
```bash
# Build and launch both FastAPI and Streamlit in background
docker compose up -d --build
```
- **Streamlit App:** `http://localhost:8501`
- **FastAPI Backend & Built React UI:** `http://localhost:8000`
- **Interactive Swagger Docs:** `http://localhost:8000/docs`

To stop:
```bash
docker compose down
```

---

## 🚀 Quickstart & Windows Execution Guide (PowerShell)

### One-Click Interactive Runner
In your Windows PowerShell terminal:
```powershell
.\run_local.ps1
```
*(Provides an interactive menu to launch the React App, Streamlit Dashboard, FastAPI, or run test suites).*

---

### Manual Step-by-Step Execution

#### 1. Environment Setup
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

#### 2. Dataset Download & Verification
Downloads the Kaggle dataset via Kaggle CLI (or verifies existing `data/raw/cinemaTicket_Ref.csv`) and records its SHA-256 in `config.yaml`:
```powershell
python scripts/download_data.py
```
*(Manual fallback: Download zip from [Kaggle Cinema Ticket](https://www.kaggle.com/datasets/arashnic/cinema-ticket), unzip `cinemaTicket_Ref.csv` into `data/raw/`, and re-run).*

#### 3. Dataset Audit & Leakage Check
Audits schema, composite key duplicates, leakage vectors, and evaluates GO/NO-GO criteria:
```powershell
python -m src.data.audit
```

#### 4. Data Cleaning & Leakage Table
Deduplicates keys, imputes missing capacity, sanitizes overcapacity, and outputs `reports/leakage_table.csv`:
```powershell
python -m src.data.clean
```

#### 5. Reproduce Training Pipeline
Trains the Hierarchical Baseline, Ridge Regression, and LightGBM models on chronological splits:
```powershell
python -m src.train
```

#### 6. Run Test Suites
```powershell
# Python Backend Tests (Leakage independence, capacity bounds, split order, API validation)
pytest -v

# React Frontend Tests (Component rendering, staffing logic, modal triggers)
npm --prefix frontend run test
```

#### 7. Launch Frontends
- **React Frontend (Vite):** `npm --prefix frontend run dev` -> [http://localhost:5173](http://localhost:5173)
- **FastAPI + Built UI:** `uvicorn app.api:app --reload --port 8000` -> [http://localhost:8000](http://localhost:8000)
- **Streamlit Dashboard:** `streamlit run app/dashboard.py` -> [http://localhost:8501](http://localhost:8501)

---

## 📊 Held-Out Test Set Results (2018-09-30 to 2018-11-04, N=22,764)

Every number below was directly computed on the held-out test split:

| Model Architecture | MAE (Tickets) | RMSE (Tickets) | WAPE (%) | Occupancy MAE | Asymmetric Cost (cu=1.5, co=1.0) | Clipped at Capacity (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Hierarchical Baseline** | 77.63 | 154.01 | 62.83% | 0.1148 | 89.07 | 0.00% |
| **Ridge Regression (Occupancy)** | 67.50 | 137.55 | 54.63% | 0.1081 | 91.84 | 0.00% |
| **LightGBM (Tickets L1) ★** | **48.36** | **108.96** | **39.14%** | **0.0832** | **65.66** | **0.54%** |

### Key Observations:
- **Baseline Outperformed:** LightGBM delivers a **37.7% reduction in MAE** (48.36 vs 77.63 tickets) and brings WAPE down from 62.83% to 39.14%.
- **Asymmetric Operational Cost:** Penalizing under-forecasting (1.5x penalty: risking long concession lines and walkouts) versus over-forecasting (1.0x penalty: idle usher shifts) shows LightGBM reduces operational penalty by **26.3%** (65.66 vs 89.07).
- **Uncertainty Calibration:** Validation residual quantiles yield Q10 = -41.21 tickets, Q90 = +36.94 tickets. Empirical coverage on held-out test shows is **69.28%** (reflecting late-2018 blockbuster surge variance).
- **Cold-Start Slicing:** Established films achieve **38.01% WAPE**, whereas brand-new films debut at **73.33% WAPE**, justifying the automated UI cold-start alert advising managers to maintain flexible staffing buffers.

---

## 🎯 Acceptance Tests Mapping

| Required Acceptance Test | Verification Command / Target | Verification Evidence |
| :--- | :--- | :--- |
| **1. Reproduce split & training** | `python -m src.train` | Generates identical splits in `artifacts/split_info.json` and trains pipeline end-to-end. |
| **2. Baseline vs candidate metrics** | `python -c "import json; print(json.load(open('artifacts/metrics.json'))['models'])"` | Saved in `artifacts/metrics.json` and detailed in `reports/technical_report.md`. |
| **3. Real prediction via API/App** | `python -m src.predict` or `http://localhost:8000/docs` | Predictor loads `model.joblib` and `history.parquet`; returns live forecasts and intervals. |
| **4. Explain one prediction & limitations** | Check `tests/test_pipeline.py` or inspect API response | Returns local input feature drivers, cold-start diagnostic, and limitation notes. |
| **5. Show no future leakage or secrets** | `pytest -k test_leakage_future_independence` | Formally asserts features for date D are invariant to future perturbations; zero credentials committed. |

---

## 🔒 Security, Privacy & Integrity
- **Zero API Tokens or Credentials Committed:** Git tracking excludes `.venv/`, `data/`, `.env`, and cache folders.
- **Dataset Attribution:** Kaggle Cinema Tickets dataset by Möbius ([CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)). The raw CSV is not redistributed; it is downloaded locally via Kaggle CLI and verified with SHA-256.
- **All Deliverables Tracked:** `artifacts/model.joblib` and `artifacts/history.parquet` are tracked so any cloned repository or cloud container can perform inference immediately.
