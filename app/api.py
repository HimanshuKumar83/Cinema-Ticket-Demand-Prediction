"""
CinemaOps AI - Production FastAPI Decision-Support Microservice
Exposes RESTful endpoints for real-time show forecasting, batch predictions,
SHAP explainability, benchmark metrics, and operational staffing advisory.
"""

import sys
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import CinemaPredictor, ForecastResult

# Global predictor instance
predictor: Optional[CinemaPredictor] = None


def get_predictor() -> CinemaPredictor:
    """Lazy loader for CinemaPredictor singleton."""
    global predictor
    if predictor is None:
        predictor = CinemaPredictor()
    return predictor


@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    try:
        predictor = get_predictor()
    except Exception as e:
        print(f"Warning: Predictor failed to load on startup: {e}")
    yield


app = FastAPI(
    title="CinemaOps AI - Demand Forecasting & Staffing Microservice",
    description="Show-level T-48h demand forecasting & decision-support system for cinema operations.",
    version="2.0.0",
    lifespan=lifespan,
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------------------------------------------
# PYDANTIC SCHEMAS
# -------------------------------------------------------------
class PredictRequest(BaseModel):
    # Modern simulated schema parameters
    cinema_id: Optional[str] = Field("CIN_01", description="Cinema identifier (e.g. CIN_01 to CIN_30)")
    movie_id: Optional[str] = Field("MOV_001", description="Movie identifier (e.g. MOV_001 to MOV_100)")
    screen_capacity: Optional[float] = Field(250.0, description="Screen seating capacity (80 - 450)", gt=0)
    show_slot: Optional[str] = Field("Evening", description="Show slot: Morning, Matinee, Evening, Night")
    ticket_price: Optional[float] = Field(250.0, description="Average ticket price", gt=0)
    advance_bookings_t48: Optional[float] = Field(None, description="Net tickets booked <= T-48h", ge=0)
    booking_velocity_24h: Optional[float] = Field(None, description="Bookings in (T-72h, T-48h]", ge=0)
    booking_velocity_previous_24h: Optional[float] = Field(None, description="Bookings in (T-96h, T-72h]", ge=0)
    is_weekend: Optional[int] = Field(0, description="1 if Fri/Sat/Sun, 0 otherwise")
    is_holiday: Optional[int] = Field(0, description="1 if public holiday, 0 otherwise")
    is_blockbuster: Optional[int] = Field(0, description="1 if blockbuster release, 0 otherwise")
    movie_release_age_days: Optional[float] = Field(7.0, description="Days since movie release", ge=0)
    genre: Optional[str] = Field("Action", description="Movie genre")

    # Legacy backward-compatibility parameters
    cinema_code: Optional[Any] = Field(None, description="Legacy cinema code")
    film_code: Optional[Any] = Field(None, description="Legacy film code")
    date: Optional[str] = Field(None, description="Screening date (YYYY-MM-DD)")
    show_time: Optional[Any] = Field(None, description="Legacy slot number (1-4)")
    capacity: Optional[float] = Field(None, description="Legacy capacity field", gt=0)


class BatchPredictRequest(BaseModel):
    shows: List[PredictRequest] = Field(..., description="List of shows to predict (up to 100)")


class DriverImpact(BaseModel):
    feature: str
    feature_value: Any
    shap_impact: float
    is_positive: bool


class OperationalAdvisory(BaseModel):
    screen_decision: str
    recommended_action: str
    suggested_capacity_range: str
    box_office_staff: int
    ushers_count: int
    concession_cashiers: Optional[int] = None
    cleaning_crew: int
    total_floor_staff: Optional[int] = None
    concession_inventory_units: int
    managerial_review_required: bool = True
    operational_disclaimer: Optional[str] = "Rule-based operational decision support only. Not ML output. Managerial sign-off required."
    staffing_details: Optional[Dict[str, Any]] = None
    screen_details: Optional[Dict[str, Any]] = None


class PredictResponse(BaseModel):
    predicted_tickets: float
    predicted_occupancy: float
    screen_capacity: float
    capacity: Optional[float] = None
    interval_80_lower: float
    lower_bound: Optional[float] = None
    interval_80_upper: float
    upper_bound: Optional[float] = None
    is_cold_start: bool
    cold_start_warning: Optional[str]
    positive_drivers: List[DriverImpact]
    negative_drivers: List[DriverImpact]
    operational_advisory: OperationalAdvisory
    limitation_note: str


class ModelInfoResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    model_type: str
    target: str
    prediction_horizon: str
    features_count: int
    features_list: List[str]
    best_model_test_mae: float
    best_model_test_wape: float
    conformal_coverage_pct: float
    residual_q10: float
    residual_q90: float


# -------------------------------------------------------------
# ENDPOINTS
# -------------------------------------------------------------
@app.get("/health", tags=["Monitoring"])
def health_check():
    """Health check endpoint for container probes."""
    try:
        pred = get_predictor()
        ready = pred is not None and pred.model is not None
    except Exception:
        ready = False

    return {
        "status": "healthy" if ready else "unhealthy",
        "service": "CinemaOps AI Demand Forecasting Microservice",
        "version": "2.0.0",
        "model_loaded": ready,
        "prediction_horizon": "T-48h",
    }


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Monitoring"])
@app.get("/model-info", response_model=ModelInfoResponse, tags=["Monitoring"])
def model_info():
    """Retrieve operational model specifications, features, and evaluation benchmarks."""
    pred = get_predictor()
    if pred is None:
        raise HTTPException(status_code=503, detail="Predictor service unavailable.")

    metrics = pred.bundle.get("metrics", {})
    return ModelInfoResponse(
        model_type="LightGBM Regressor (L1 Objective) + 80% Conformal Calibration",
        target="actual_tickets_sold",
        prediction_horizon="Strictly T-48h prior to showtime (Zero Lookahead)",
        features_count=len(pred.feature_cols),
        features_list=pred.feature_cols,
        best_model_test_mae=float(metrics.get("best_model_test_mae", 11.27)),
        best_model_test_wape=float(metrics.get("best_model_test_wape", 11.22)),
        conformal_coverage_pct=float(metrics.get("conformal", {}).get("empirical_test_coverage_pct", 76.64)),
        residual_q10=pred.q10,
        residual_q90=pred.q90,
    )


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
def predict(req: PredictRequest):
    """Generate show-level demand forecast, prediction intervals, and operational advisory."""
    try:
        pred = get_predictor()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Predictor initialization failed: {e}")

    try:
        res = pred.predict_single(
            cinema_id=req.cinema_id,
            movie_id=req.movie_id,
            screen_capacity=req.screen_capacity,
            show_slot=req.show_slot,
            ticket_price=req.ticket_price,
            advance_bookings_t48=req.advance_bookings_t48,
            booking_velocity_24h=req.booking_velocity_24h,
            booking_velocity_previous_24h=req.booking_velocity_previous_24h,
            is_weekend=req.is_weekend,
            is_holiday=req.is_holiday,
            is_blockbuster=req.is_blockbuster,
            movie_release_age_days=req.movie_release_age_days,
            genre=req.genre,
            cinema_code=req.cinema_code,
            film_code=req.film_code,
            date=req.date,
            show_time=req.show_time,
            capacity=req.capacity,
        )

        return PredictResponse(
            predicted_tickets=res.predicted_tickets,
            predicted_occupancy=res.predicted_occupancy,
            screen_capacity=res.screen_capacity,
            capacity=res.capacity or res.screen_capacity,
            interval_80_lower=res.interval_80_lower,
            lower_bound=res.lower_bound or res.interval_80_lower,
            interval_80_upper=res.interval_80_upper,
            upper_bound=res.upper_bound or res.interval_80_upper,
            is_cold_start=res.is_cold_start,
            cold_start_warning=res.cold_start_warning,
            positive_drivers=[DriverImpact(**d) for d in res.positive_drivers],
            negative_drivers=[DriverImpact(**d) for d in res.negative_drivers],
            operational_advisory=OperationalAdvisory(**res.operational_advisory),
            limitation_note=res.limitation_note,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")


@app.post("/predict/batch", tags=["Inference"])
def predict_batch(req: BatchPredictRequest):
    """Batch inference endpoint for multi-show scheduling rosters (up to 100 shows)."""
    if len(req.shows) > 100:
        raise HTTPException(status_code=400, detail="Batch limit exceeded. Max 100 shows per request.")

    pred = get_predictor()
    results = []
    for item in req.shows:
        try:
            res = pred.predict_single(
                cinema_id=item.cinema_id,
                movie_id=item.movie_id,
                screen_capacity=item.screen_capacity,
                show_slot=item.show_slot,
                ticket_price=item.ticket_price,
                advance_bookings_t48=item.advance_bookings_t48,
                booking_velocity_24h=item.booking_velocity_24h,
                booking_velocity_previous_24h=item.booking_velocity_previous_24h,
                is_weekend=item.is_weekend,
                is_holiday=item.is_holiday,
                is_blockbuster=item.is_blockbuster,
                movie_release_age_days=item.movie_release_age_days,
                genre=item.genre,
                cinema_code=item.cinema_code,
                film_code=item.film_code,
                date=item.date,
                show_time=item.show_time,
                capacity=item.capacity,
            )
            results.append(res.to_dict())
        except Exception as e:
            results.append({"error": str(e)})

    return {"total_shows": len(results), "forecasts": results}


@app.get("/metrics", tags=["Evaluation"])
def get_metrics():
    """Retrieve test set metrics, conformal calibration, and operational slice performance."""
    metrics_path = PROJECT_ROOT / "artifacts" / "metrics.json"
    if not metrics_path.exists():
        raise HTTPException(status_code=404, detail="Metrics artifact not found.")

    with open(metrics_path, "r") as f:
        data = json.load(f)
    return data


@app.get("/model/benchmark", tags=["Evaluation"])
def get_benchmark():
    """Retrieve 5-model held-out chronological test benchmark comparison table."""
    bench_path = PROJECT_ROOT / "artifacts" / "model_benchmark.json"
    if not bench_path.exists():
        raise HTTPException(status_code=404, detail="Benchmark artifact not found.")

    with open(bench_path, "r") as f:
        data = json.load(f)
    return {"benchmark": data}


@app.post("/explain", tags=["Explainability"])
def explain_show(req: PredictRequest):
    """Retrieve detailed TreeSHAP attribution waterfall for an individual show."""
    pred = get_predictor()
    if pred.explainer is None:
        raise HTTPException(status_code=503, detail="SHAP explainer is not initialized.")

    try:
        # Run predict_single to obtain feature vector
        res = pred.predict_single(
            cinema_id=req.cinema_id,
            movie_id=req.movie_id,
            screen_capacity=req.screen_capacity,
            show_slot=req.show_slot,
            ticket_price=req.ticket_price,
            advance_bookings_t48=req.advance_bookings_t48,
            booking_velocity_24h=req.booking_velocity_24h,
            booking_velocity_previous_24h=req.booking_velocity_previous_24h,
            is_weekend=req.is_weekend,
            is_holiday=req.is_holiday,
            is_blockbuster=req.is_blockbuster,
            movie_release_age_days=req.movie_release_age_days,
            genre=req.genre,
        )
        return {
            "predicted_tickets": res.predicted_tickets,
            "predicted_occupancy": res.predicted_occupancy,
            "positive_drivers": res.positive_drivers,
            "negative_drivers": res.negative_drivers,
            "operational_advisory": res.operational_advisory,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Explanation error: {e}")


# Mount built React single-page frontend if dist exists
frontend_dist_dir = PROJECT_ROOT / "frontend" / "dist"
if frontend_dist_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dist_dir), html=True), name="frontend")
