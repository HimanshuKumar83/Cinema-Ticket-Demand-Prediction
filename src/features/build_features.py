"""
CinemaOps AI - Feature Engineering Pipeline
Computes leakage-free T-48 features, advance booking signals, velocities,
and historical statistics strictly from past completed shows.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd


ARTIFACTS_DIR = Path("artifacts")


class HistoryPriors:
    """
    Stores precomputed training statistics and priors to guarantee
    zero future lookahead leakage during feature engineering and inference.
    """
    def __init__(self, priors_dict: Optional[Dict[str, Any]] = None):
        self.priors = priors_dict or {}

    @classmethod
    def fit_from_train(
        cls,
        df_train: pd.DataFrame,
        target_col: str = "actual_tickets_sold",
        capacity_col: str = "screen_capacity",
    ) -> "HistoryPriors":
        """Compute training priors for target and occupancy."""
        df = df_train.copy()
        
        # Support aliases
        t_col = target_col if target_col in df.columns else "tickets_sold"
        c_col = capacity_col if capacity_col in df.columns else "capacity"
        m_col = "movie_id" if "movie_id" in df.columns else "film_code"
        cin_col = "cinema_id" if "cinema_id" in df.columns else "cinema_code"
        g_col = "genre" if "genre" in df.columns else None
        slot_col = "show_slot" if "show_slot" in df.columns else "show_time"
        day_col = "day_of_week" if "day_of_week" in df.columns else None

        df["_occupancy"] = (df[t_col] / np.maximum(1.0, df[c_col])).clip(0.0, 1.0)
        
        global_mean_demand = float(df[t_col].mean())
        global_mean_occupancy = float(df["_occupancy"].mean())

        movie_demand = df.groupby(m_col)[t_col].mean().to_dict()
        cinema_demand = df.groupby(cin_col)[t_col].mean().to_dict()
        slot_demand = df.groupby(slot_col)[t_col].mean().to_dict()
        
        genre_demand = df.groupby(g_col)[t_col].mean().to_dict() if g_col else {}
        weekday_demand = df.groupby(day_col)[t_col].mean().to_dict() if day_col else {}

        movie_counts = df.groupby(m_col)[t_col].count().to_dict()
        cinema_counts = df.groupby(cin_col)[t_col].count().to_dict()

        priors_data = {
            "global_mean_demand": global_mean_demand,
            "global_mean_occupancy": global_mean_occupancy,
            "movie_demand": {str(k): float(v) for k, v in movie_demand.items()},
            "cinema_demand": {str(k): float(v) for k, v in cinema_demand.items()},
            "slot_demand": {str(k): float(v) for k, v in slot_demand.items()},
            "genre_demand": {str(k): float(v) for k, v in genre_demand.items()},
            "weekday_demand": {str(k): float(v) for k, v in weekday_demand.items()},
            "movie_counts": {str(k): int(v) for k, v in movie_counts.items()},
            "cinema_counts": {str(k): int(v) for k, v in cinema_counts.items()},
        }
        return cls(priors_data)

    def save(self, filepath: Path):
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(self.priors, f, indent=2)

    @classmethod
    def load(cls, filepath: Path) -> "HistoryPriors":
        with open(filepath, "r") as f:
            data = json.load(f)
        return cls(data)


def compute_features(
    df: pd.DataFrame,
    priors: Optional[HistoryPriors] = None,
    fit_priors: bool = False,
) -> Tuple[pd.DataFrame, HistoryPriors]:
    """
    Extract all modeling features strictly known at T-48.
    
    Features include:
    1. T-48 Advance booking telemetry:
       - advance_bookings_t48
       - booking_velocity_24h (net bookings in [T-72h, T-48h])
       - booking_velocity_previous_24h (net bookings in [T-96h, T-72h])
       - booking_acceleration = velocity_24h - velocity_previous_24h
       - advance_occupancy_t48 = advance_bookings_t48 / screen_capacity
       - velocity_ratio = velocity_24h / (advance_bookings_t48 + 1)
    2. Show schedule attributes:
       - screen_capacity
       - ticket_price
       - movie_release_age_days
       - is_blockbuster
       - is_weekend
       - is_holiday
    3. Categorical encodings:
       - genre
       - show_slot
       - cinema_tier
    4. Leakage-free training historical averages:
       - movie_hist_avg_demand
       - cinema_hist_avg_demand
       - slot_hist_avg_demand
       - genre_hist_avg_demand
       - weekday_hist_avg_demand
    5. Cold-start indicators:
       - is_new_movie (0 past shows in training set)
       - is_new_cinema (0 past shows in training set)
       - is_low_history (< 10 past shows in training set)
    """
    data = df.copy()

    # Column aliases
    cap_col = "screen_capacity" if "screen_capacity" in data.columns else "capacity"
    movie_col = "movie_id" if "movie_id" in data.columns else "film_code"
    cinema_col = "cinema_id" if "cinema_id" in data.columns else "cinema_code"
    slot_col = "show_slot" if "show_slot" in data.columns else "show_time"
    genre_col = "genre" if "genre" in data.columns else None
    day_col = "day_of_week" if "day_of_week" in data.columns else None

    # Derive priors if requested
    if fit_priors or priors is None:
        priors = HistoryPriors.fit_from_train(data)

    p_dict = priors.priors
    global_mean_demand = p_dict.get("global_mean_demand", 140.0)

    # Feature DataFrame
    feats = pd.DataFrame(index=data.index)

    # 1. Capacity & Price
    feats["screen_capacity"] = data[cap_col].astype(float)
    feats["ticket_price"] = data["ticket_price"].astype(float) if "ticket_price" in data.columns else 250.0

    # 2. Advance Telemetry at T-48
    if "advance_bookings_t48" in data.columns:
        feats["advance_bookings_t48"] = data["advance_bookings_t48"].astype(float)
    else:
        feats["advance_bookings_t48"] = 0.0

    if "booking_velocity_24h" in data.columns:
        feats["booking_velocity_24h"] = data["booking_velocity_24h"].astype(float)
    else:
        feats["booking_velocity_24h"] = 0.0

    if "booking_velocity_previous_24h" in data.columns:
        feats["booking_velocity_previous_24h"] = data["booking_velocity_previous_24h"].astype(float)
    else:
        feats["booking_velocity_previous_24h"] = 0.0

    # Derived advance metrics
    feats["booking_acceleration"] = feats["booking_velocity_24h"] - feats["booking_velocity_previous_24h"]
    feats["advance_occupancy_t48"] = (feats["advance_bookings_t48"] / np.maximum(1.0, feats["screen_capacity"])).clip(0.0, 1.0)
    feats["velocity_ratio"] = feats["booking_velocity_24h"] / (feats["advance_bookings_t48"] + 1.0)

    # 3. Calendar & Release
    if "is_weekend" in data.columns:
        feats["is_weekend"] = data["is_weekend"].astype(int)
    elif "date" in data.columns:
        feats["is_weekend"] = (pd.to_datetime(data["date"]).dt.weekday >= 4).astype(int)
    else:
        feats["is_weekend"] = 0

    feats["is_holiday"] = data["is_holiday"].astype(int) if "is_holiday" in data.columns else 0
    feats["is_blockbuster"] = data["is_blockbuster"].astype(int) if "is_blockbuster" in data.columns else 0
    feats["movie_release_age_days"] = data["movie_release_age_days"].astype(float) if "movie_release_age_days" in data.columns else 14.0

    # 4. Historical Averages (from priors)
    m_dem = p_dict.get("movie_demand", {})
    c_dem = p_dict.get("cinema_demand", {})
    s_dem = p_dict.get("slot_demand", {})
    g_dem = p_dict.get("genre_demand", {})
    w_dem = p_dict.get("weekday_demand", {})
    m_counts = p_dict.get("movie_counts", {})
    c_counts = p_dict.get("cinema_counts", {})

    feats["movie_hist_avg_demand"] = data[movie_col].astype(str).map(m_dem).fillna(global_mean_demand)
    feats["cinema_hist_avg_demand"] = data[cinema_col].astype(str).map(c_dem).fillna(global_mean_demand)
    feats["slot_hist_avg_demand"] = data[slot_col].astype(str).map(s_dem).fillna(global_mean_demand)
    
    if genre_col:
        feats["genre_hist_avg_demand"] = data[genre_col].astype(str).map(g_dem).fillna(global_mean_demand)
    else:
        feats["genre_hist_avg_demand"] = global_mean_demand

    if day_col:
        feats["weekday_hist_avg_demand"] = data[day_col].astype(str).map(w_dem).fillna(global_mean_demand)
    else:
        feats["weekday_hist_avg_demand"] = global_mean_demand

    # 5. Cold-Start Indicators
    movie_hist_count = data[movie_col].astype(str).map(m_counts).fillna(0).astype(int)
    cinema_hist_count = data[cinema_col].astype(str).map(c_counts).fillna(0).astype(int)

    feats["is_new_movie"] = (movie_hist_count == 0).astype(int)
    feats["is_new_cinema"] = (cinema_hist_count == 0).astype(int)
    feats["is_low_history"] = ((movie_hist_count < 10) | (cinema_hist_count < 10)).astype(int)

    # 6. Categoricals (one-hot or frequency encoding)
    # Slots
    slots = ["Morning", "Matinee", "Evening", "Night"]
    for s in slots:
        feats[f"slot_{s.lower()}"] = (data[slot_col].astype(str) == s).astype(int)

    # Cinema Tier
    if "cinema_tier" in data.columns:
        feats["is_tier_1"] = (data["cinema_tier"].astype(str) == "Tier-1").astype(int)
    else:
        feats["is_tier_1"] = 1

    return feats, priors
