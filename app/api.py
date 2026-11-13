"""FastAPI inference service for Cinema Ticket Demand Forecasting.
Loads trained model and history store; exposes /predict, /health, and /model-info.
"""
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional
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
    title="Cinema Ticket Demand Forecasting API",
    description="Show-level T-48h demand forecasting MVP for cinema operations managers.",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for React and local development frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    cinema_code: int = Field(..., description="Anonymized cinema complex identifier", json_schema_extra={"example": 448})
    film_code: int = Field(..., description="Anonymized movie identifier", json_schema_extra={"example": 1576})
    date: str = Field(..., description="Screening date (YYYY-MM-DD)", json_schema_extra={"example": "2018-10-15"})
    show_time: int = Field(..., description="Daily screening slot index (e.g. 1, 2, 3...)", ge=1, le=100, json_schema_extra={"example": 3})
    capacity: Optional[float] = Field(None, description="Auditorium seat capacity (positive float)", gt=0, json_schema_extra={"example": 300.0})


class FeatureContribution(BaseModel):
    feature: str
    value: Any


class PredictResponse(BaseModel):
    predicted_tickets: float
    predicted_occupancy: float
    capacity: float
    interval_80_lower: float
    interval_80_upper: float
    is_cold_start: bool
    cold_start_warning: Optional[str]
    top_contributing_features: list[FeatureContribution]
    limitation_note: str


class ModelInfoResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    
    model_type: str
    target: str
    prediction_horizon: str
    features_count: int
    validation_metrics: dict
    test_metrics: dict
    uncertainty_quantiles: dict


@app.get("/health", tags=["Monitoring"])
def health_check():
    """Health status and model readiness probe."""
    try:
        pred = get_predictor()
        is_ready = pred is not None and pred.model is not None
    except Exception:
        is_ready = False
        
    return {
        "status": "healthy" if is_ready else "unhealthy",
        "model_loaded": is_ready,
        "horizon": "T-48h"
    }


@app.get("/model-info", response_model=ModelInfoResponse, tags=["Monitoring"])
def model_info():
    """Metadata regarding the loaded production model."""
    pred = get_predictor()
    if pred is None:
        raise HTTPException(status_code=503, detail="Predictor is not loaded.")
    
    metrics = pred.bundle.get("metrics", {})
    return ModelInfoResponse(
        model_type="LightGBM Regressor (L1 Objective)",
        target="tickets_sold",
        prediction_horizon="48 hours prior (strictly data <= show_date - 2)",
        features_count=len(pred.feature_cols),
        validation_metrics={},
        test_metrics=metrics.get("models", {}).get("lightgbm", {}),
        uncertainty_quantiles={
            "q10_residual": pred.q10,
            "q90_residual": pred.q90,
            "test_coverage_pct": metrics.get("uncertainty_80pct", {}).get("test_coverage_pct", 0.0)
        }
    )


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
def predict(req: PredictRequest):
    """Generate real-time show-level ticket demand and occupancy forecast."""
    try:
        pred = get_predictor()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Predictor initialization failed: {e}")

    try:
        res: ForecastResult = pred.predict_single(
            cinema_code=req.cinema_code,
            film_code=req.film_code,
            date=req.date,
            show_time=req.show_time,
            capacity=req.capacity
        )
        return PredictResponse(
            predicted_tickets=res.predicted_tickets,
            predicted_occupancy=res.predicted_occupancy,
            capacity=res.capacity,
            interval_80_lower=res.interval_80_lower,
            interval_80_upper=res.interval_80_upper,
            is_cold_start=res.is_cold_start,
            cold_start_warning=res.cold_start_warning,
            top_contributing_features=[
                FeatureContribution(feature=f["feature"], value=f["value"])
                for f in res.top_contributing_features
            ],
            limitation_note=res.limitation_note
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")


# Mount built React single-page frontend if dist exists
frontend_dist_dir = Path(__file__).resolve().parent.parent / "frontend" / "dist"
if frontend_dist_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dist_dir), html=True), name="frontend")

