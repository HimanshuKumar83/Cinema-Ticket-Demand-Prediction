# CinemaOps AI: Comprehensive Project Audit & Redesign Plan

**Author:** Senior Machine Learning & Systems Engineer  
**Organization:** Algions AI Labs Internship Assessment  
**Document:** Initial Architecture Audit & Technical Gap Analysis  
**Date:** November 2026  
**Status:** COMPLETE AUDIT  

---

## 1. Executive Summary

A comprehensive architectural, technical, and algorithmic audit of the existing **CinemaOps AI** repository was conducted. While the initial MVP established solid project scaffolding (FastAPI backend, basic chronological splitting, LightGBM regressor, and React/Streamlit frontends), several structural and dataset-level limitations prevent it from fulfilling the official **Algions AI Labs** production assessment requirements for genuine $T-48\text{h}$ show-level forecasting.

Most critically, the legacy Kaggle 2018 dataset records only aggregate post-show attendance figures without individual booking event timestamps. Consequently, dynamic lead-time signals—such as `advance_bookings_t48`, `booking_velocity_24h`, and `booking_velocity_previous_24h`—cannot be derived from it. 

To satisfy the official assignment rubric with 100% technical rigor and zero data leakage, we are implementing a **reproducible, transaction-grounded cinema booking simulation engine** (generating Show Metadata and Booking Events tables), training 5 distinct model architectures (Baseline, Ridge, Random Forest, XGBoost, LightGBM), integrating formal SHAP explainability and conformal prediction intervals, expanding the FastAPI contract, and upgrading the enterprise SaaS dashboard.

---

## 2. Existing Repository Architecture & Technology Stack

| Layer | Current Implementation | Technologies | Status / Health |
| :--- | :--- | :--- | :--- |
| **Data Storage** | Single static CSV (`cinemaTicket_Ref.csv`) | Raw CSV, Parquet (`artifacts/history.parquet`) | **Inadequate:** Lacks transaction timestamps |
| **Feature Store** | Date-shifted daily rolling means | Pandas, NumPy | **Basic:** Only computes macro daily lags |
| **Models** | Hierarchical Baseline, Ridge (Occupancy), LightGBM | Scikit-Learn 1.5, LightGBM 4.5 | **Partial:** Missing Random Forest & XGBoost |
| **Evaluation** | Held-out chronological test set metrics | Custom metrics script | **Solid:** Needs multi-model benchmark reporting |
| **Explainability** | Heuristic feature gain | LightGBM feature gain | **Incomplete:** Lacks genuine SHAP tree explainer |
| **Backend API** | FastAPI microservice | FastAPI, Pydantic v2, Uvicorn | **Partial:** Needs `/predict/batch`, `/metrics`, `/model/benchmark`, `/explain` |
| **Frontend UI** | Streamlit & React 18 (Vite) | Streamlit 1.39, React 18, Lucide | **Good base:** Needs enterprise SaaS page restructuring |
| **Test Suite** | 6 unit tests with synthetic fixtures | Pytest 8.3, Vitest | **Passing:** Needs expanded coverage for batch, SHAP, and simulator |

---

## 3. Detailed Technical Gap Analysis

### Gap 1: Inability of Legacy Data to Support T-48 Booking Dynamics
- **Issue:** The Kaggle 2018 dataset records only final show attendance (`tickets_sold`) and monetary totals. It does not record transaction timestamps or advance booking curves.
- **Requirement:** Algions AI Labs specifically mandates show-level prediction generated at $T-48\text{h}$ utilizing:
  1. `advance_bookings_t48`: Cumulative confirmed tickets booked prior to 48 hours before showtime.
  2. `booking_velocity_24h`: Rate of ticket bookings in the 24 hours immediately preceding $T-48\text{h}$ ($[T-72\text{h}, T-48\text{h}]$).
  3. `booking_velocity_previous_24h`: Rate of ticket bookings in $[T-96\text{h}, T-72\text{h}]$.
  4. Separate **Show Metadata Table** and **Booking Events Table** with individual transaction timestamps, booking channels (online vs box-office), and booking statuses (confirmed vs cancelled).
- **Remedy:** Implement `src/data/generate_data.py` and `src/data/booking_simulator.py` to produce a high-fidelity synthetic dataset (~30,000 shows, 30 cinemas, 100 movies, 18 months) with fully reproducible random seeds and realistic non-linear demand curves.

### Gap 2: Model Portfolio Coverage
- **Issue:** Currently, only Ridge and LightGBM are evaluated against the Baseline.
- **Requirement:** The assignment mandates:
  1. **Model 1: Capacity-adjusted Hierarchical Baseline** (4 fallback levels: Movie+Cinema -> Movie -> Cinema+Slot -> Global).
  2. **Model 2: Ridge Regression** (regularized linear baseline).
  3. **Model 3: Random Forest Regressor** (bagging non-linear trees).
  4. **Model 4: XGBoost Regressor** (gradient boosted trees).
  5. **Model 5: LightGBM Regressor** (histogram-based gradient booster).
- **Remedy:** Develop a unified model training and evaluation pipeline in `src/models/` implementing all 5 models under identical chronological splits.

### Gap 3: Explainability Method
- **Issue:** Previous local explanations relied on heuristic tree gain ranking rather than game-theoretic feature attribution.
- **Requirement:** Official rubric specifies genuine **SHAP (SHapley Additive exPlanations)** with global summary plots, local single-prediction force/waterfall values, and explicit positive and negative demand drivers.
- **Remedy:** Build `src/explainability/shap_analysis.py` wrapping TreeExplainer and LinearExplainer to serve both the API `/explain` endpoint and the UI dashboard.

### Gap 4: REST API Endpoint Completeness
- **Issue:** The existing FastAPI server exposed only `/predict`, `/health`, and `/model-info`.
- **Requirement:** The required endpoint contract includes:
  - `GET /health` (readiness & liveness)
  - `GET /model/info` (model metadata & features)
  - `POST /predict` (single show forecast with Pydantic validation)
  - `POST /predict/batch` (multi-show schedule batch inference)
  - `GET /metrics` (latest test set metrics: MAE, RMSE, WAPE, Occupancy MAE)
  - `GET /model/benchmark` (multi-model comparison across all 5 models)
  - `POST /explain` (local SHAP breakdown of top positive and negative drivers)
- **Remedy:** Rebuild `app/api.py` to implement all 7 endpoints with complete Pydantic v2 schemas and error guards.

### Gap 5: Enterprise SaaS Frontend Expansion
- **Issue:** While functional, the UI currently groups several views together.
- **Requirement:** Present an enterprise-grade decision-support suite covering:
  - Overview Dashboard (macro demand trends, KPI metrics)
  - T-48 Demand Forecasting (live interactive single-show prediction)
  - Model Benchmarks (5-model side-by-side leaderboard)
  - Cold-Start Analytics (new film & new cinema fallback tracking)
  - Model Explainability (SHAP global importance and local driver bars)
  - Capacity & Staffing Advisory (actionable screen swap and shift roster calculators)
  - Model Monitoring (actual vs predicted scatter and residual tracking)

---

## 4. Components Reusability Assessment

| Component | Status | Strategy |
| :--- | :--- | :--- |
| `app/dashboard.py` | **Preserve & Upgrade** | Expand into full multi-tab Enterprise SaaS platform with all 7 views. |
| `frontend/` (React) | **Preserve & Upgrade** | Update API client, forms, and results displays to reflect new dynamic booking velocity fields. |
| `src/data/audit.py` | **Preserve & Adapt** | Retain audit methodology; apply to generated transaction tables. |
| `src/evaluation/metrics.py` | **Preserve & Expand** | Retain WAPE, MAE, RMSE, Asymmetric Cost; add capacity violation rate and coverage. |
| `artifacts/` | **Retain structure** | Save new model artifacts, conformal residual quantiles, and SHAP explainers. |
| `run_local.ps1` & Docker | **Preserve** | Maintain Windows PowerShell launcher and containerization workflows. |

---

## 5. Phased Redesign & Implementation Roadmap

```
Phase 1: Architecture Audit & Strategy (Complete - PROJECT_AUDIT.md)
   │
Phase 2: Dataset Reconstruction & Simulation (src/data/generate_data.py, booking_simulator.py)
   │     - 30k shows, 30 cinemas, 100 movies, 18-month timeline
   │     - Show Metadata & Booking Events tables with transaction timestamps
   │
Phase 3: Leakage-Free T-48 Feature Engineering (src/features/build_features.py)
   │     - advance_bookings_t48, booking_velocity_24h, booking_velocity_previous_24h
   │     - Strictly shifted completed-show historical averages
   │
Phase 4: Multi-Model Portfolio Development (src/models/)
   │     - Baseline (4-level fallback), Ridge, Random Forest, XGBoost, LightGBM
   │
Phase 5: Chronological & Cold-Start Evaluation (src/models/evaluate.py)
   │     - 70% Train / 15% Val / 15% Test
   │     - Separate slicing for New Movie, New Cinema, Low History
   │
Phase 6: Uncertainty & Explainability (src/explainability/shap_analysis.py)
   │     - Split conformal 80% prediction intervals
   │     - SHAP global and local demand drivers
   │
Phase 7: FastAPI Backend Redesign (app/api.py)
   │     - 7 production endpoints with Pydantic validation
   │
Phase 8: Frontend SaaS Redesign (app/dashboard.py & frontend/)
   │     - 7 modular dashboard tabs with actionable managerial guidance
   │
Phase 9: Comprehensive Test Suite & Documentation
   │     - Pytest suite covering all 14 assignment requirements
   │     - TECHNICAL_REPORT.md, ARCHITECTURE.md, updated README.md
   │
Phase 10: Git Verification & Remote Push (main branch)
```

