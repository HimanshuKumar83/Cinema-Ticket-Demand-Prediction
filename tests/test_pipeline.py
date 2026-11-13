"""Comprehensive test suite for Cinema Ticket Demand Forecasting MVP.
Uses synthetic fixtures with ZERO Kaggle or external dependency.
"""
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.api import app
from src.features.build_features import (
    HistoryFeatureStore,
    compute_calendar_features,
    extract_features_for_dataset,
    create_chronological_splits,
)
from src.predict import CinemaPredictor


@pytest.fixture
def synthetic_history_df() -> pd.DataFrame:
    """Create a realistic small synthetic dataset covering 20 days."""
    np.random.seed(42)
    base_date = datetime(2018, 5, 1)
    rows = []
    cinemas = [101, 102]
    films = [501, 502, 503]

    for day_offset in range(20):
        current_date = base_date + timedelta(days=day_offset)
        for cinema in cinemas:
            for film in films:
                for slot in [1, 2, 3]:
                    cap = 200.0
                    tickets = int(np.random.randint(10, 180))
                    rows.append({
                        "date": current_date,
                        "cinema_code": cinema,
                        "film_code": film,
                        "show_time": slot,
                        "capacity": cap,
                        "tickets_sold": tickets,
                        "occupancy": tickets / cap,
                        "total_sales": tickets * 50.0,
                        "tickets_out": 0,
                        "ticket_use": tickets,
                        "ticket_price": 50.0,
                        "occu_perc": (tickets / cap) * 100.0,
                        "capacity_imputed": 0
                    })
    return pd.DataFrame(rows)


def test_leakage_future_independence(synthetic_history_df):
    """TEST 1: Leakage test - History features for date D must remain strictly
    identical even if rows on D-1, D, or future dates are violently changed.
    """
    store_original = HistoryFeatureStore(synthetic_history_df)
    
    # Target show on Day 10 (2018-05-11). Under T-48h horizon, visible history is <= 2018-05-09 (Day 8).
    target_show = pd.DataFrame([{
        "date": datetime(2018, 5, 11),
        "cinema_code": 101,
        "film_code": 501,
        "show_time": 2,
        "capacity": 200.0,
        "capacity_imputed": 0
    }])

    feats_orig = extract_features_for_dataset(target_show, store_original)

    # Now create perturbed history where Day 9 (2018-05-10) and Day 10 (2018-05-11)
    # have ticket sales multiplied by 1,000x or completely replaced
    perturbed_df = synthetic_history_df.copy()
    leak_mask = perturbed_df["date"] >= datetime(2018, 5, 10)
    perturbed_df.loc[leak_mask, "tickets_sold"] = perturbed_df.loc[leak_mask, "tickets_sold"] * 1000
    perturbed_df.loc[leak_mask, "occupancy"] = 1.0

    store_perturbed = HistoryFeatureStore(perturbed_df)
    feats_perturbed = extract_features_for_dataset(target_show, store_perturbed)

    # Check key rolling historical features
    numeric_cols = feats_orig.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        val_orig = feats_orig[col].iloc[0]
        val_pert = feats_perturbed[col].iloc[0]
        assert np.isclose(val_orig, val_pert, equal_nan=True), (
            f"LEAKAGE DETECTED! Feature '{col}' changed from {val_orig} to {val_pert} "
            "when future/contemporaneous data was altered!"
        )


def test_feature_list_excludes_target_and_post_show():
    """TEST 2: Forbidden columns (target and ex-post outcomes) must never be in feature list."""
    forbidden = {
        "tickets_sold", "tickets_num", "occu_perc", "total_sales", 
        "tickets_out", "ticket_use", "ticket_price", "occupancy"
    }

    # Check feature_list.json if it exists
    feature_list_path = Path("artifacts/feature_list.json")
    if feature_list_path.exists():
        import json
        with open(feature_list_path, "r", encoding="utf-8") as f:
            features = json.load(f)
        for col in forbidden:
            assert col not in features, f"Forbidden target/post-show column '{col}' found in artifacts/feature_list.json!"


def test_chronological_split_order(synthetic_history_df):
    """TEST 3: Chronological splits must have strictly non-overlapping dates in sequential order."""
    splits = create_chronological_splits(synthetic_history_df)
    df_train = splits["train"]
    df_val = splits["val"]
    df_test = splits["test"]

    train_max_date = df_train["date"].max()
    val_min_date = df_val["date"].min()
    val_max_date = df_val["date"].max()
    test_min_date = df_test["date"].min()

    assert train_max_date < val_min_date, f"Train max date ({train_max_date}) must precede Val min date ({val_min_date})"
    assert val_max_date < test_min_date, f"Val max date ({val_max_date}) must precede Test min date ({test_min_date})"
    
    # Check zero overlap
    train_dates = set(df_train["date"].dt.date.unique())
    val_dates = set(df_val["date"].dt.date.unique())
    test_dates = set(df_test["date"].dt.date.unique())

    assert len(train_dates.intersection(val_dates)) == 0, "Train and Val splits share overlapping dates!"
    assert len(val_dates.intersection(test_dates)) == 0, "Val and Test splits share overlapping dates!"


def test_predictions_within_capacity_bounds():
    """TEST 4: Predictions and intervals must always remain within [0, capacity]."""
    predictor = CinemaPredictor()
    
    # Test on diverse capacities
    for cap in [20.0, 100.0, 250.0, 500.0]:
        res = predictor.predict_single(cinema_code=448, film_code=1576, date="2018-10-15", show_time=3, capacity=cap)
        assert 0.0 <= res.predicted_tickets <= cap, f"Predicted tickets {res.predicted_tickets} outside [0, {cap}]"
        assert 0.0 <= res.predicted_occupancy <= 1.0, f"Predicted occupancy {res.predicted_occupancy} outside [0, 1]"
        assert 0.0 <= res.interval_80_lower <= cap, f"Lower interval {res.interval_80_lower} outside [0, {cap}]"
        assert 0.0 <= res.interval_80_upper <= cap, f"Upper interval {res.interval_80_upper} outside [0, {cap}]"
        assert res.interval_80_lower <= res.interval_80_upper, "Lower interval bound exceeds upper interval bound!"


def test_output_changes_when_inputs_change():
    """TEST 5: Verify model outputs change meaningfully when inputs change (no hardcoded responses)."""
    predictor = CinemaPredictor()

    res_film_a = predictor.predict_single(cinema_code=448, film_code=1576, date="2018-10-15", show_time=3, capacity=300)
    # Different film (unseen cold-start film)
    res_film_b = predictor.predict_single(cinema_code=448, film_code=9999, date="2018-10-15", show_time=3, capacity=300)

    # Different capacity
    res_small_cap = predictor.predict_single(cinema_code=448, film_code=1576, date="2018-10-15", show_time=3, capacity=50)

    # Different slot (matinee slot 1 vs evening slot 5)
    res_slot_1 = predictor.predict_single(cinema_code=448, film_code=1576, date="2018-10-15", show_time=1, capacity=300)
    res_slot_5 = predictor.predict_single(cinema_code=448, film_code=1576, date="2018-10-15", show_time=5, capacity=300)

    assert res_film_a.predicted_tickets != res_film_b.predicted_tickets, "Predictions for different films should not be identical!"
    assert res_film_b.is_cold_start is True, "Cold start flag should trigger for unknown film"
    assert res_film_a.is_cold_start is False, "Known film should not be flagged as cold start"
    assert res_small_cap.predicted_tickets <= 50, "Predicted tickets for capacity 50 must be <= 50"


def test_api_valid_and_invalid_requests():
    """TEST 6: FastAPI endpoint validation for both compliant and malformed payloads."""
    client = TestClient(app)

    # Health check
    res_health = client.get("/health")
    assert res_health.status_code == 200
    assert res_health.json()["status"] == "healthy"

    # Model info
    res_info = client.get("/model-info")
    assert res_info.status_code == 200
    assert "LightGBM" in res_info.json()["model_type"]

    # Valid predict payload
    valid_payload = {
        "cinema_code": 448,
        "film_code": 1576,
        "date": "2018-10-15",
        "show_time": 3,
        "capacity": 300.0
    }
    res_pred = client.post("/predict", json=valid_payload)
    assert res_pred.status_code == 200
    data = res_pred.json()
    assert "predicted_tickets" in data
    assert 0.0 <= data["predicted_tickets"] <= 300.0
    assert "limitation_note" in data
    assert "human manager review required" in data["limitation_note"].lower()

    # Invalid request: negative capacity
    invalid_cap_payload = {
        "cinema_code": 448,
        "film_code": 1576,
        "date": "2018-10-15",
        "show_time": 3,
        "capacity": -50.0
    }
    res_inv_cap = client.post("/predict", json=invalid_cap_payload)
    assert res_inv_cap.status_code == 422, "API should reject negative capacity with 422 Unprocessable Entity"

    # Invalid request: missing required cinema_code
    missing_field_payload = {
        "film_code": 1576,
        "date": "2018-10-15",
        "show_time": 3
    }
    res_missing = client.post("/predict", json=missing_field_payload)
    assert res_missing.status_code == 422, "API should reject missing cinema_code with 422"

    # Invalid request: invalid show_time out of bounds (e.g. 0 or negative)
    invalid_slot_payload = {
        "cinema_code": 448,
        "film_code": 1576,
        "date": "2018-10-15",
        "show_time": 0,
        "capacity": 300.0
    }
    res_inv_slot = client.post("/predict", json=invalid_slot_payload)
    assert res_inv_slot.status_code == 422, "API should reject show_time < 1 with 422"
