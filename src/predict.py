"""
CinemaOps AI - Production Prediction Engine
Provides show-level ticket demand forecasts, 80% conformal prediction intervals,
cold-start diagnostics, SHAP local driver breakdowns, and operational staffing advisory.
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import joblib
import numpy as np
import pandas as pd

from src.features.build_features import compute_features, HistoryPriors
from src.explainability.shap_analysis import CinemaExplainer
from src.operational.advisory import calculate_staffing_and_screen_advisory


ARTIFACTS_DIR = Path("artifacts")
MODEL_PATH = ARTIFACTS_DIR / "model.joblib"


@dataclass
class OperationalRecommendation:
    screen_decision: str
    recommended_action: str
    suggested_capacity_range: str
    box_office_staff: int
    ushers_count: int
    cleaning_crew: int
    concession_inventory_units: int
    managerial_review_required: bool = True


@dataclass
class ForecastResult:
    predicted_tickets: float
    predicted_occupancy: float
    screen_capacity: float
    interval_80_lower: float
    interval_80_upper: float
    is_cold_start: bool
    cold_start_warning: Optional[str]
    positive_drivers: List[Dict[str, Any]]
    negative_drivers: List[Dict[str, Any]]
    operational_advisory: Dict[str, Any]
    limitation_note: str
    capacity: Optional[float] = None
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None

    def __post_init__(self):
        if self.capacity is None:
            self.capacity = self.screen_capacity
        if self.lower_bound is None:
            self.lower_bound = self.interval_80_lower
        if self.upper_bound is None:
            self.upper_bound = self.interval_80_upper

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CinemaPredictor:
    """Production predictor loading model bundle and priors."""

    def __init__(self, model_path: Path = MODEL_PATH):
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}. Run 'python -m src.train' first.")

        print(f"Loading model bundle from {model_path}...")
        self.bundle = joblib.load(model_path)
        self.model = self.bundle["model"]
        self.feature_cols = self.bundle["feature_cols"]
        self.q10 = float(self.bundle.get("residual_q10", -25.0))
        self.q90 = float(self.bundle.get("residual_q90", 12.0))
        self.q10_norm = float(self.bundle.get("residual_q10_norm", -0.115))
        self.q90_norm = float(self.bundle.get("residual_q90_norm", 0.055))
        self.use_normalized_conformal = "residual_q10_norm" in self.bundle
        self.priors = HistoryPriors(self.bundle.get("priors", {}))
        
        # Initialize SHAP explainer
        try:
            self.explainer = CinemaExplainer(model_path)
        except Exception as e:
            print(f"Warning: SHAP explainer initialization deferred: {e}")
            self.explainer = None

    def predict_single(
        self,
        cinema_id: Optional[str] = None,
        movie_id: Optional[str] = None,
        screen_capacity: Optional[float] = None,
        show_slot: Optional[str] = "Evening",
        ticket_price: Optional[float] = 250.0,
        advance_bookings_t48: Optional[float] = None,
        booking_velocity_24h: Optional[float] = None,
        booking_velocity_previous_24h: Optional[float] = None,
        is_weekend: Optional[int] = 0,
        is_holiday: Optional[int] = 0,
        is_blockbuster: Optional[int] = 0,
        movie_release_age_days: Optional[float] = 7.0,
        genre: Optional[str] = "Action",
        # Legacy parameter aliases for backward compatibility
        cinema_code: Optional[Any] = None,
        film_code: Optional[Any] = None,
        date: Optional[str] = None,
        show_time: Optional[Any] = None,
        capacity: Optional[float] = None,
    ) -> ForecastResult:
        """
        Generate forecast and operational advisory for a single movie show slot.
        """
        # Resolve aliases
        c_id = str(cinema_id or cinema_code or "CIN_01")
        m_id = str(movie_id or film_code or "MOV_001")
        cap = float(screen_capacity or capacity or 250.0)

        # Slot resolution
        slot_map = {1: "Morning", 2: "Matinee", 3: "Evening", 4: "Night"}
        if show_time in slot_map:
            slot_str = slot_map[show_time]
        elif isinstance(show_time, (int, float)):
            slot_str = "Evening" if show_time >= 3 else "Matinee"
        else:
            slot_str = str(show_slot or "Evening")

        # Weekend resolution
        if date is not None and is_weekend == 0:
            try:
                dt = pd.to_datetime(date)
                is_weekend = int(dt.weekday() >= 4)
            except Exception:
                pass

        # Realistic default advance bookings if omitted
        if advance_bookings_t48 is None:
            advance_bookings_t48 = round(cap * (0.35 if is_blockbuster else 0.20), 1)

        if booking_velocity_24h is None:
            booking_velocity_24h = round(advance_bookings_t48 * 0.40, 1)

        if booking_velocity_previous_24h is None:
            booking_velocity_previous_24h = round(booking_velocity_24h * 0.75, 1)

        # Construct single-row DataFrame
        input_data = pd.DataFrame([{
            "cinema_id": c_id,
            "movie_id": m_id,
            "screen_capacity": cap,
            "show_slot": slot_str,
            "ticket_price": float(ticket_price or 250.0),
            "advance_bookings_t48": float(advance_bookings_t48),
            "booking_velocity_24h": float(booking_velocity_24h),
            "booking_velocity_previous_24h": float(booking_velocity_previous_24h),
            "is_weekend": int(is_weekend),
            "is_holiday": int(is_holiday),
            "is_blockbuster": int(is_blockbuster),
            "movie_release_age_days": float(movie_release_age_days or 7.0),
            "genre": str(genre or "Action"),
        }])

        # Extract features
        feats, _ = compute_features(input_data, priors=self.priors)
        X_vec = feats[self.feature_cols].values

        # Predict
        raw_pred = float(self.model.predict(X_vec)[0])
        pred_tickets = float(np.clip(raw_pred, 0.0, cap))
        pred_occupancy = float(round(pred_tickets / max(1.0, cap), 4))

        # 80% Conformal Prediction Intervals (Capacity-Aware Normalized Calibration)
        if self.use_normalized_conformal:
            lower = float(np.clip(round(pred_tickets + self.q10_norm * cap, 1), 0.0, cap))
            upper = float(np.clip(round(pred_tickets + self.q90_norm * cap, 1), 0.0, cap))
        else:
            lower = float(np.clip(round(pred_tickets + self.q10, 1), 0.0, cap))
            upper = float(np.clip(round(pred_tickets + self.q90, 1), 0.0, cap))

        # Cold start diagnosis
        is_new_movie = bool(feats["is_new_movie"].iloc[0] == 1)
        is_new_cinema = bool(feats["is_new_cinema"].iloc[0] == 1)
        is_low_history = bool(feats["is_low_history"].iloc[0] == 1)

        is_cold = is_new_movie or is_new_cinema or is_low_history
        cold_msg = None
        if is_new_movie:
            cold_msg = f"COLD-START NOTICE: Movie '{m_id}' has zero prior screenings in training. Prediction relies on genre/cinema priors."
        elif is_low_history:
            cold_msg = f"LOW-HISTORY NOTICE: Limited historical data (< 10 shows). Prediction uncertainty interval is wider."
        elif is_new_cinema:
            cold_msg = f"COLD-START NOTICE: Cinema '{c_id}' is unobserved. Prediction relies on tier priors."

        # SHAP local drivers
        pos_drivers, neg_drivers = [], []
        if self.explainer is not None:
            try:
                explanation = self.explainer.explain_instance(X_vec[0], top_k=4)
                pos_drivers = explanation["positive_drivers"]
                neg_drivers = explanation["negative_drivers"]
            except Exception as e:
                print(f"SHAP explanation error: {e}")

        # Centralized Operational Capacity and Staffing Advisory
        advisory_obj = calculate_staffing_and_screen_advisory(
            predicted_tickets=pred_tickets,
            predicted_occupancy=pred_occupancy,
            screen_capacity=cap,
            interval_80_upper=upper,
        )
        advisory_dict = advisory_obj.to_dict()

        limitation = (
            "Decision-support system only: All operational screen adjustments and staffing rosters "
            "must be validated by an authorized cinema manager. Computed at T-48h horizon."
        )

        return ForecastResult(
            predicted_tickets=round(pred_tickets, 1),
            predicted_occupancy=pred_occupancy,
            screen_capacity=cap,
            capacity=cap,
            interval_80_lower=lower,
            lower_bound=lower,
            interval_80_upper=upper,
            upper_bound=upper,
            is_cold_start=is_cold,
            cold_start_warning=cold_msg,
            positive_drivers=pos_drivers,
            negative_drivers=neg_drivers,
            operational_advisory=advisory_dict,
            limitation_note=limitation,
        )

    def _generate_operational_advisory(
        self, predicted_tickets: float, occupancy: float, capacity: float, interval_80_upper: Optional[float] = None
    ) -> OperationalRecommendation:
        """Rule-based decision support logic for screen allocation and staffing (delegates to centralized engine)."""
        upper = interval_80_upper if interval_80_upper is not None else min(capacity, predicted_tickets + 15.0)
        adv = calculate_staffing_and_screen_advisory(
            predicted_tickets=predicted_tickets,
            predicted_occupancy=occupancy,
            screen_capacity=capacity,
            interval_80_upper=upper,
        )
        return OperationalRecommendation(
            screen_decision=adv.screen_decision,
            recommended_action=adv.recommended_action,
            suggested_capacity_range=adv.suggested_capacity_range,
            box_office_staff=adv.box_office_staff,
            ushers_count=adv.ushers_count,
            cleaning_crew=adv.cleaning_crew,
            concession_inventory_units=adv.concession_inventory_units,
            managerial_review_required=True,
        )


if __name__ == "__main__":
    predictor = CinemaPredictor()
    res = predictor.predict_single(
        cinema_id="CIN_01",
        movie_id="MOV_001",
        screen_capacity=300,
        show_slot="Evening",
        is_weekend=1,
        is_blockbuster=1,
        advance_bookings_t48=120,
        booking_velocity_24h=45,
    )
    print("\n--- Test Forecast Result ---")
    print(res.to_dict())
