"""Inference module for Cinema Ticket Demand Forecasting.
Loads trained model and history store; produces ticket predictions, occupancy,
uncertainty intervals, cold-start warnings, and local feature contributions.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
import joblib
import numpy as np
import pandas as pd

from src.features.build_features import (
    HistoryFeatureStore,
    compute_calendar_features,
    extract_features_for_dataset,
)

ARTIFACTS_DIR = Path("artifacts")
MODEL_PATH = ARTIFACTS_DIR / "model.joblib"
HISTORY_PATH = ARTIFACTS_DIR / "history.parquet"


@dataclass
class ForecastResult:
    predicted_tickets: float
    predicted_occupancy: float
    capacity: float
    interval_80_lower: float
    interval_80_upper: float
    is_cold_start: bool
    cold_start_warning: Optional[str]
    top_contributing_features: list[dict[str, Any]]
    limitation_note: str


class CinemaPredictor:
    """Production predictor loading model bundle and history store."""

    def __init__(self, model_path: Path = MODEL_PATH, history_path: Path = HISTORY_PATH):
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}. Run 'python -m src.train' first.")
        if not history_path.exists():
            raise FileNotFoundError(f"History file not found: {history_path}. Run 'python -m src.train' first.")

        print(f"Loading model bundle from {model_path}...")
        self.bundle = joblib.load(model_path)
        self.model = self.bundle["model"]
        self.feature_cols = self.bundle["feature_cols"]
        self.cat_features = self.bundle["cat_features"]
        self.q10 = self.bundle["residual_q10"]
        self.q90 = self.bundle["residual_q90"]

        print(f"Loading history store from {history_path}...")
        self.history_store = HistoryFeatureStore.from_parquet(history_path)

        # Baseline cinema capacity lookup for fallback
        self.cinema_median_caps = (
            self.history_store.history_df[self.history_store.history_df["capacity"] > 0]
            .groupby("cinema_code")["capacity"]
            .median()
            .to_dict()
        )
        self.global_median_cap = float(
            self.history_store.history_df[self.history_store.history_df["capacity"] > 0]["capacity"].median()
        )

    def predict_single(
        self,
        cinema_code: int,
        film_code: int,
        date: str,
        show_time: int,
        capacity: Optional[float] = None
    ) -> ForecastResult:
        """Generate forecast for a single screening slot."""
        # Clean capacity
        capacity_imputed = 0
        if capacity is None or capacity <= 0 or np.isnan(capacity):
            capacity = float(self.cinema_median_caps.get(cinema_code, self.global_median_cap))
            capacity_imputed = 1

        df_input = pd.DataFrame([{
            "cinema_code": int(cinema_code),
            "film_code": int(film_code),
            "date": pd.to_datetime(date),
            "show_time": int(show_time),
            "capacity": float(capacity),
            "capacity_imputed": capacity_imputed
        }])

        # Extract features enforcing strictly past-only T-48h horizon
        feats = extract_features_for_dataset(df_input, self.history_store)
        
        # Prepare categorical types for LightGBM
        X_infer = feats[self.feature_cols].copy()
        for col in self.cat_features:
            if col in X_infer.columns:
                X_infer[col] = X_infer[col].astype("category")

        # Predict
        raw_pred = float(self.model.predict(X_infer)[0])
        pred_tickets = float(np.clip(raw_pred, 0.0, capacity))
        pred_occupancy = float(pred_tickets / max(1.0, capacity))

        # Uncertainty intervals
        lower = float(np.clip(pred_tickets + self.q10, 0.0, capacity))
        upper = float(np.clip(pred_tickets + self.q90, 0.0, capacity))

        # Cold-start diagnostic
        is_new_film = bool(feats["is_new_film"].iloc[0] == 1)
        film_count = int(feats["film_history_count"].iloc[0])
        is_new_cinema = bool(feats["is_new_cinema"].iloc[0] == 1)
        cinema_count = int(feats["cinema_history_count"].iloc[0])

        is_cold = is_new_film or (film_count < 10) or is_new_cinema
        warning_msg = None
        if is_new_film:
            warning_msg = (
                f"COLD-START WARNING: Film {film_code} has zero prior screening history in the past 48+ hours. "
                "Forecast relies on cinema-level and slot-level historical priors."
            )
        elif film_count < 10:
            warning_msg = (
                f"LOW-HISTORY NOTICE: Film {film_code} has only {film_count} prior screening(s). "
                "Prediction variance is higher than normal."
            )
        elif is_new_cinema:
            warning_msg = (
                f"COLD-START WARNING: Cinema {cinema_code} has no previous records. "
                "Forecast uses global benchmark occupancy."
            )

        # Top local contributing features
        # Approximation using tree feature importance and input feature values
        top_features = self._explain_local_prediction(X_infer.iloc[0])

        limitation = (
            "Decision-support only: human manager review required. Forecast computed at T-48h horizon. "
            "Model cannot anticipate sudden unannounced local marketing campaigns, severe weather, or viral spikes."
        )

        return ForecastResult(
            predicted_tickets=round(pred_tickets, 1),
            predicted_occupancy=round(pred_occupancy, 4),
            capacity=float(capacity),
            interval_80_lower=round(lower, 1),
            interval_80_upper=round(upper, 1),
            is_cold_start=is_cold,
            cold_start_warning=warning_msg,
            top_contributing_features=top_features,
            limitation_note=limitation
        )

    def _explain_local_prediction(self, sample_series: pd.Series) -> list[dict[str, Any]]:
        """Identify key contributing input features for the prediction."""
        # Top known dominant features from LightGBM global importance
        key_inspect_cols = [
            "film_cinema_mean_tickets_14d",
            "cinema_slot_mean_tickets_14d",
            "cinema_mean_tickets_7d_filled",
            "film_mean_tickets_7d_filled",
            "capacity",
            "weekday",
            "show_time",
            "is_opening_week"
        ]
        contributions = []
        for col in key_inspect_cols:
            if col in sample_series:
                val = sample_series[col]
                # Format readable description
                contributions.append({
                    "feature": col,
                    "value": round(float(val), 2) if isinstance(val, (int, float, np.floating, np.integer)) else str(val)
                })
        return contributions[:5]


if __name__ == "__main__":
    predictor = CinemaPredictor()
    # Test an existing show
    res_known = predictor.predict_single(cinema_code=448, film_code=1576, date="2018-10-15", show_time=3, capacity=300)
    print("\n--- Inference on Known Show ---")
    print(res_known)

    # Test a cold-start film
    res_cold = predictor.predict_single(cinema_code=448, film_code=99999, date="2018-10-15", show_time=3, capacity=300)
    print("\n--- Inference on Cold-Start Film ---")
    print(res_cold)

