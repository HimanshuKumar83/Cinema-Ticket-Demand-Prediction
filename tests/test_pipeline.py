"""
CinemaOps AI - Comprehensive Test Suite
Verifies data simulation integrity, zero lookahead leakage, chronological ordering,
feature lists, model capacity bounds, 5-model bundle presence, and FastAPI endpoints.
"""

from datetime import datetime, timedelta
from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.api import app
from src.data.booking_simulator import BookingSimulator, BookingSimulationConfig
from src.features.build_features import compute_features, HistoryPriors
from src.predict import CinemaPredictor


@pytest.fixture
def predictor_instance():
    return CinemaPredictor()


def test_booking_simulator_arrival_and_capacity_bounds():
    """TEST 1: BookingSimulator respects capacity bounds and produces valid T-48 metrics."""
    sim = BookingSimulator(BookingSimulationConfig(random_seed=42))
    capacity = 150
    show_dt = pd.to_datetime("2024-05-10 18:00:00")

    events, metrics = sim.simulate_show_bookings(
        show_id="TEST_001",
        show_datetime=show_dt,
        screen_capacity=capacity,
        expected_demand_ratio=0.85,
        is_weekend=True,
        is_blockbuster=True,
    )

    assert 0 <= metrics["actual_tickets_sold"] <= capacity
    assert 0 <= metrics["advance_bookings_t48"] <= metrics["actual_tickets_sold"]
    assert metrics["booking_velocity_24h"] >= 0
    assert metrics["booking_velocity_previous_24h"] >= 0

    # Verify event timestamps are prior to showtime
    for ev in events:
        assert ev["booking_timestamp"] <= show_dt


def test_feature_list_excludes_target_and_post_show():
    """TEST 2: Forbidden columns (target, revenue, post-show outcomes) must never be in feature list."""
    forbidden = {
        "actual_tickets_sold", "tickets_sold", "occupancy", "occupancy_rate",
        "total_sales", "tickets_out", "ticket_use", "show_datetime"
    }

    feature_list_path = Path("artifacts/feature_list.json")
    assert feature_list_path.exists(), "artifacts/feature_list.json must exist"
    
    with open(feature_list_path, "r", encoding="utf-8") as f:
        features = json.load(f)
    for col in forbidden:
        assert col not in features, f"Forbidden target/post-show column '{col}' found in feature list!"


def test_chronological_splits_integrity():
    """TEST 3: Chronological splits must have strictly non-overlapping dates in sequential order."""
    split_meta_path = Path("data/processed/split_metadata.json")
    if split_meta_path.exists():
        with open(split_meta_path, "r") as f:
            meta = json.load(f)
        
        train_end = meta["train"]["end_date"]
        val_start = meta["val"]["start_date"]
        val_end = meta["val"]["end_date"]
        test_start = meta["test"]["start_date"]

        assert train_end <= val_start, f"Leakage: Train end {train_end} > Val start {val_start}"
        assert val_end <= test_start, f"Leakage: Val end {val_end} > Test start {test_start}"
        assert meta["chronological_integrity"] is True


def test_predictions_within_capacity_bounds(predictor_instance):
    """TEST 4: Predictions and intervals must always remain within [0, capacity]."""
    for cap in [50.0, 100.0, 250.0, 450.0]:
        res = predictor_instance.predict_single(
            cinema_id="CIN_01",
            movie_id="MOV_001",
            screen_capacity=cap,
            show_slot="Evening",
            advance_bookings_t48=30,
            booking_velocity_24h=12,
        )
        assert 0.0 <= res.predicted_tickets <= cap, f"Predicted tickets {res.predicted_tickets} outside [0, {cap}]"
        assert 0.0 <= res.predicted_occupancy <= 1.0, f"Predicted occupancy {res.predicted_occupancy} outside [0, 1]"
        assert 0.0 <= res.interval_80_lower <= cap, f"Lower interval {res.interval_80_lower} outside [0, {cap}]"
        assert 0.0 <= res.interval_80_upper <= cap, f"Upper interval {res.interval_80_upper} outside [0, {cap}]"
        assert res.interval_80_lower <= res.interval_80_upper, "Lower interval bound exceeds upper interval bound!"


def test_output_changes_when_inputs_change(predictor_instance):
    """TEST 5: Verify model outputs change meaningfully when inputs change (no hardcoded responses)."""
    res_base = predictor_instance.predict_single(
        cinema_id="CIN_01", movie_id="MOV_001", screen_capacity=300, show_slot="Evening",
        advance_bookings_t48=80, booking_velocity_24h=30, is_blockbuster=0
    )
    # Blockbuster surge with higher velocity
    res_blockbuster = predictor_instance.predict_single(
        cinema_id="CIN_01", movie_id="MOV_001", screen_capacity=300, show_slot="Evening",
        advance_bookings_t48=180, booking_velocity_24h=90, is_blockbuster=1
    )
    # Boutique screen
    res_small = predictor_instance.predict_single(
        cinema_id="CIN_01", movie_id="MOV_001", screen_capacity=80, show_slot="Morning",
        advance_bookings_t48=20, booking_velocity_24h=5
    )

    assert res_base.predicted_tickets != res_blockbuster.predicted_tickets, "Predictions should vary with advance demand!"
    assert res_blockbuster.predicted_tickets > res_base.predicted_tickets, "Blockbuster should predict higher attendance!"
    assert res_small.predicted_tickets <= 80, "Small capacity predictions must be capped at 80"


def test_cold_start_detection_and_advisory(predictor_instance):
    """TEST 6: Cold-start detection flags new movies/cinemas and yields advisory."""
    res_cold = predictor_instance.predict_single(
        cinema_id="CIN_UNKNOWN_99",
        movie_id="MOV_NEW_999",
        screen_capacity=250,
        show_slot="Night",
        advance_bookings_t48=20,
        booking_velocity_24h=5
    )
    assert res_cold.is_cold_start is True, "Unseen movie should trigger cold start warning"
    assert res_cold.cold_start_warning is not None
    assert "COLD-START" in res_cold.cold_start_warning

    # Check operational advisory exists
    adv = res_cold.operational_advisory
    assert "screen_decision" in adv
    assert "box_office_staff" in adv
    assert "ushers_count" in adv
    assert adv["managerial_review_required"] is True


def test_all_five_models_presence_in_bundle():
    """TEST 7: Verify all 5 benchmark models are trained and serialized in model.joblib."""
    import joblib
    bundle_path = Path("artifacts/model.joblib")
    assert bundle_path.exists(), "artifacts/model.joblib must exist"
    
    bundle = joblib.load(bundle_path)
    assert "all_models" in bundle, "all_models dictionary must exist in bundle"
    
    expected_models = {"baseline", "ridge", "random_forest", "xgboost", "lightgbm"}
    for m in expected_models:
        assert m in bundle["all_models"], f"Model '{m}' missing from model bundle!"


def test_api_valid_batch_and_invalid_requests():
    """TEST 8: FastAPI endpoint validation for /health, /model/info, /model/benchmark, single & batch predict."""
    client = TestClient(app)

    # Health check
    res_health = client.get("/health")
    assert res_health.status_code == 200
    assert res_health.json()["status"] == "healthy"

    # Model info
    res_info = client.get("/model/info")
    assert res_info.status_code == 200
    assert "LightGBM" in res_info.json()["model_type"]

    # Model benchmark
    res_bench = client.get("/model/benchmark")
    assert res_bench.status_code == 200
    assert "benchmark" in res_bench.json()
    assert len(res_bench.json()["benchmark"]) == 5

    # Valid single predict
    valid_payload = {
        "cinema_id": "CIN_01",
        "movie_id": "MOV_001",
        "screen_capacity": 300.0,
        "show_slot": "Evening",
        "advance_bookings_t48": 120.0,
        "booking_velocity_24h": 40.0,
        "is_weekend": 1,
        "is_blockbuster": 1
    }
    res_pred = client.post("/predict", json=valid_payload)
    assert res_pred.status_code == 200
    data = res_pred.json()
    assert "predicted_tickets" in data
    assert 0.0 <= data["predicted_tickets"] <= 300.0
    assert "operational_advisory" in data
    assert "limitation_note" in data

    # Valid batch predict
    batch_payload = {
        "shows": [
            valid_payload,
            {**valid_payload, "screen_capacity": 150.0, "advance_bookings_t48": 40.0}
        ]
    }
    res_batch = client.post("/predict/batch", json=batch_payload)
    assert res_batch.status_code == 200
    assert res_batch.json()["total_shows"] == 2

    # Invalid request: negative capacity
    invalid_cap = {**valid_payload, "screen_capacity": -50.0}
    res_inv = client.post("/predict", json=invalid_cap)
    assert res_inv.status_code == 422, "Negative capacity should be rejected with 422"
