# Cinema Ticket Demand Forecasting (Show-Level MVP)

> **Algions AI Labs Internship Assessment**  
> **Candidate Track:** Senior Machine Learning Engineer  
> **Stack:** Python 3.10+, PowerShell (Windows), LightGBM, Scikit-Learn, FastAPI, Streamlit, Pytest  

---

## 📌 Framing & Operational Problem Definition
- **Business Decision:** Assist cinema complex operations managers in allocating auditorium screens (upgrading/downgrading theater room size) and scheduling usher, box-office, and concession staff for upcoming shows.
- **Prediction Unit:** Single show-slot defined by `(cinema_code, film_code, date, show_time)`.
- **Target Variable:** `tickets_sold` (show attendance volume); secondary reporting metric is `occupancy` = $\frac{\text{tickets\_sold}}{\text{capacity}}$.
- **Prediction Horizon:** Strictly $T - 48\text{h}$ prior to showtime (`show_date - 2 days`). Only historical data $\le \text{show\_date} - 2$ plus advance schedule attributes (capacity, calendar, slot) are permitted.
- **Decision-Support Boundary:** Strictly decision-support with mandatory human operational review. Never an automated high-stakes decision maker.

---

## 🔒 Security & Privacy Rules
- **No Credentials Committed:** Zero API keys or tokens are stored, printed, logged, or tracked in git.
- **`.gitignore` Enforced:** Excludes `.venv/`, `data/`, `.env`, `__pycache__/`, `*.pyc`, `.ipynb_checkpoints/`. Historical lookup (`artifacts/*.parquet`) is permitted; raw datasets are strictly omitted.
- **`.env.example` Provided:** Contains harmless placeholder schema only.

---

## 🚀 Quickstart & Windows Execution Guide (PowerShell)

### 1. Environment Setup & Dependency Installation
```powershell
# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install pinned dependencies
pip install -r requirements.txt
```

### 2. Dataset Download & Verification
Downloads the Kaggle dataset via Kaggle CLI (or verifies existing `data/raw/cinemaTicket_Ref.csv`) and stores SHA-256 in `config.yaml`:
```powershell
python scripts/download_data.py
```
*(Manual fallback: Download zip from [Kaggle Cinema Ticket](https://www.kaggle.com/datasets/arashnic/cinema-ticket), unzip `cinemaTicket_Ref.csv` into `data/raw/`, and re-run).*

### 3. Dataset Audit & Leakage Check
Audits schema, composite key duplicates, occupancy/ticket price leakage, and evaluates GO/NO-GO criteria:
```powershell
python -m src.data.audit
```

### 4. Data Cleaning & Leakage Table Generation
Deduplicates composite keys, imputes missing capacity, and saves `reports/leakage_table.csv`:
```powershell
python -m src.data.clean
```

### 5. Reproduce Chronological Training & Evaluation
Runs end-to-end training across Baseline, Ridge, and LightGBM models, logging metrics and calibration:
```powershell
python -m src.train
```

### 6. Run Automated Test Suite (Pytest)
Executes 6 independent tests verifying future-independence, split integrity, bounds, and API validation:
```powershell
pytest -v
```

### 7. Run Modern React Frontend (Vite)
Starts the modern React dashboard on port 5173 with hot-module reload:
```powershell
npm --prefix frontend run dev
```
- Interactive React App: [http://localhost:5173](http://localhost:5173)
- Run React Unit Tests: `npm --prefix frontend run test`

### 8. Run FastAPI Microservice (Serves API & Built React UI)
Starts the real-time inference server on port 8000 (loads saved `artifacts/model.joblib` and `artifacts/history.parquet`):
```powershell
uvicorn app.api:app --reload --port 8000
```
- Built React Web UI: [http://localhost:8000](http://localhost:8000)
- Interactive Swagger Docs: [http://localhost:8000/docs](http://localhost:8000/docs)
- Health Check: `curl http://127.0.0.1:8000/health`

### 9. Run Streamlit Decision Dashboard
Alternative Python-native dashboard:
```powershell
streamlit run app/dashboard.py
```

---

## 📊 Held-Out Test Set Results (2018-09-30 to 2018-11-04, N=22,764)

Every number below was directly computed on the held-out test split:

| Model Architecture | MAE (Tickets) | RMSE (Tickets) | WAPE (%) | Occupancy MAE | Asymmetric Cost ($c_u=1.5, c_o=1.0$) | Clipped at Capacity (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Hierarchical Baseline** | 77.63 | 154.01 | 62.83% | 0.1148 | 89.07 | 0.00% |
| **Ridge (Occupancy)** | 67.50 | 137.55 | 54.63% | 0.1081 | 91.84 | 0.00% |
| **LightGBM (Tickets L1)** | **48.36** | **108.96** | **39.14%** | **0.0832** | **65.66** | **0.54%** |

- **Uncertainty Calibration (80% Interval):** Validation residual quantiles $Q_{10} = -41.21$, $Q_{90} = +36.94$. Empirical test coverage: **69.28%**.
- **Cold-Start Performance (LightGBM):** Existing films achieve 38.01% WAPE; unseen new films jump to 73.33% WAPE, triggering automated manager alerts.

---

## 🎯 Acceptance Tests Mapping

| Acceptance Criteria | Demonstration Command / File | Verification Evidence |
| :--- | :--- | :--- |
| **1. Reproduce split & training** | `python -m src.train` | Generates identical splits in `artifacts/split_info.json` and trains pipeline end-to-end. |
| **2. Baseline vs candidate metrics on test set** | `python -c "import json; print(json.load(open('artifacts/metrics.json'))['models'])"` | Metrics saved in `artifacts/metrics.json` and detailed in `reports/technical_report.md`. |
| **3. Real prediction through API/App** | `python -m src.predict` or `uvicorn app.api:app --reload` | Production predictor loads `model.joblib` & `history.parquet`; returns predictions and intervals. |
| **4. Explain one prediction & limitations** | Check `tests/test_pipeline.py` or `python -m src.predict` | Returns local feature contributions, cold-start diagnostic, and explicit limitation disclaimer. |
| **5. Show no future leakage or secrets** | `pytest tests/test_pipeline.py -k test_leakage_future_independence` | Formally asserts features for date $D$ are invariant to future perturbations; zero credentials committed. |

---

## 📋 Assessment Checklist

- [x] Lean, correct, reproducible MVP code without clever abstractions.
- [x] Clear business decision framing at top of README and technical report.
- [x] Zero credential leaks, `.gitignore` configured, placeholder `.env.example` provided.
- [x] Dataset download script with SHA-256 hash calculation and verification in `config.yaml`.
- [x] Dataset audit script with GO verdict and leakage table saved to `reports/`.
- [x] Chronological non-shuffled split (~70% train / ~15% val / ~15% test) saved to `artifacts/split_info.json`.
- [x] Strict $T-48\text{h}$ lookahead horizon enforced across all rolling features ($\le \text{show\_date} - 2$).
- [x] Baseline hierarchical occupancy model implemented and compared against Ridge and LightGBM.
- [x] Real computed test-set metrics (MAE, RMSE, WAPE, Occupancy MAE, Asymmetric Cost, Clipped %).
- [x] Capacity-aware slicing and cold-start subgroup evaluations.
- [x] 80% uncertainty interval calibrated from validation residuals with reported test coverage.
- [x] Error analysis with top-10 worst test predictions and saved residual plot `reports/residual_plot.png`.
- [x] Global feature importance and local single-prediction explanation.
- [x] FastAPI service (`app/api.py`) with `/predict`, `/health`, and `/model-info`.
- [x] Streamlit dashboard (`app/dashboard.py`) with mandatory human review disclaimer and cold-start alerts.
- [x] Pytest suite with 6 passing tests using zero-dependency synthetic fixtures.
- [x] Full `DATA_CARD.md` and `reports/technical_report.md` documenting limitations and failure modes.

