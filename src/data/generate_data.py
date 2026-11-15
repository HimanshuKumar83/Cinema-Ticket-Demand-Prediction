"""
CinemaOps AI - Synthetic Data Generator
Generates realistic show metadata and granular booking event logs over 18 months (30,000+ shows)
for leak-free, reproducible T-48 demand forecasting.
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from src.data.booking_simulator import BookingSimulator, BookingSimulationConfig


# Seeded deterministic lists for realistic data generation
CINEMA_TIERS = {
    "Tier-1": {
        "cities": ["Mumbai", "Delhi NCR", "Bengaluru", "Hyderabad", "Chennai"],
        "base_traffic_multiplier": 1.30,
        "base_price": 280,
    },
    "Tier-2": {
        "cities": ["Pune", "Ahmedabad", "Kolkata", "Jaipur", "Lucknow", "Chandigarh", "Kochi"],
        "base_traffic_multiplier": 0.95,
        "base_price": 200,
    },
}

GENRES = [
    "Action", "Comedy", "Drama", "Horror", "Sci-Fi", 
    "Romance", "Animation", "Thriller", "Adventure", "Crime"
]

GENRE_DEMAND_BIAS = {
    "Action": 1.25,
    "Sci-Fi": 1.20,
    "Animation": 1.15,
    "Comedy": 1.05,
    "Thriller": 1.00,
    "Horror": 0.95,
    "Adventure": 1.10,
    "Drama": 0.85,
    "Romance": 0.80,
    "Crime": 0.90,
}

SHOW_SLOTS = [
    {"slot": "Morning", "time": "10:00", "multiplier": 0.65},
    {"slot": "Matinee", "time": "13:30", "multiplier": 0.85},
    {"slot": "Evening", "time": "17:30", "multiplier": 1.40},
    {"slot": "Night",   "time": "21:00", "multiplier": 1.25},
]

# Major Indian / Global public holidays (2024 - 2025)
FIXED_HOLIDAYS = {
    "2024-01-01", "2024-01-26", "2024-03-08", "2024-03-25", "2024-04-11",
    "2024-05-01", "2024-06-17", "2024-08-15", "2024-10-02", "2024-10-12",
    "2024-10-31", "2024-11-01", "2024-12-25",
    "2025-01-01", "2025-01-26", "2025-03-14", "2025-03-31", "2025-04-14",
    "2025-05-01", "2025-06-07",
}


def build_cinema_catalog(num_cinemas: int = 30, seed: int = 42) -> List[Dict]:
    """Generate 30 distinct cinemas across Tier-1 and Tier-2 regions."""
    rng = np.random.default_rng(seed)
    cinemas = []
    
    tier1_cities = CINEMA_TIERS["Tier-1"]["cities"]
    tier2_cities = CINEMA_TIERS["Tier-2"]["cities"]
    
    for i in range(1, num_cinemas + 1):
        is_tier1 = (i <= 14)  # 14 Tier-1, 16 Tier-2
        tier = "Tier-1" if is_tier1 else "Tier-2"
        city = rng.choice(tier1_cities if is_tier1 else tier2_cities)
        
        # Cinema has 3 to 5 screens with varied capacities
        num_screens = int(rng.choice([3, 4, 5]))
        screens = []
        screen_templates = [
            {"screen_id": 1, "name": "Screen 1 (IMAX)", "capacity": int(rng.choice([360, 400, 450]))},
            {"screen_id": 2, "name": "Screen 2 (Large)", "capacity": int(rng.choice([250, 280, 320]))},
            {"screen_id": 3, "name": "Screen 3 (Standard)", "capacity": int(rng.choice([160, 180, 200]))},
            {"screen_id": 4, "name": "Screen 4 (Club)", "capacity": int(rng.choice([120, 140, 150]))},
            {"screen_id": 5, "name": "Screen 5 (VIP)", "capacity": int(rng.choice([80, 90, 100]))},
        ]
        
        cinemas.append({
            "cinema_id": f"CIN_{i:02d}",
            "cinema_name": f"CinemaOps {city} PLEX-{i:02d}",
            "city": city,
            "tier": tier,
            "traffic_multiplier": CINEMA_TIERS[tier]["base_traffic_multiplier"] * rng.uniform(0.92, 1.08),
            "base_price": CINEMA_TIERS[tier]["base_price"],
            "screens": screen_templates[:num_screens],
        })
    return cinemas


def build_movie_catalog(num_movies: int = 100, start_date: str = "2024-01-01", end_date: str = "2025-06-30", seed: int = 42) -> List[Dict]:
    """Generate 100 movies with staggered release dates across the 18 months."""
    rng = np.random.default_rng(seed + 10)
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    total_days = (end_dt - start_dt).days

    titles_prefix = [
        "Avatar: Echoes of", "The Dark", "Guardians of", "Quantum", "Mission",
        "Chronicles of", "The Secret", "Love in", "Rise of", "Shadow over",
        "Cyber", "Lost City of", "Echoes from", "Fast &", "The Great",
        "Starlight", "Underworld", "Midnight", "Captain", "Infinite",
    ]
    titles_suffix = [
        "Tomorrow", "Destiny", "Eternity", "the Galaxy", "Horizon",
        "Thunder", "Reckoning", "Infinity", "Memories", "Firestorm",
        "Silence", "Ascension", "Empire", "Odyssey", "Nemesis",
        "Rebellion", "Legacy", "Pandora", "Protocol", "Valhalla"
    ]

    movies = []
    # 20% blockbusters
    is_blockbuster_flags = [True] * 20 + [False] * 80
    rng.shuffle(is_blockbuster_flags)

    for i in range(1, num_movies + 1):
        prefix = titles_prefix[(i - 1) % len(titles_prefix)]
        suffix = titles_suffix[((i - 1) * 3) % len(titles_suffix)]
        title = f"{prefix} {suffix} (Vol. {(i % 4) + 1})"
        
        # Release date spread over the period (some before start_date so day 1 has active movies)
        offset_days = int(rng.uniform(-30, total_days - 14))
        release_date = (start_dt + timedelta(days=offset_days)).strftime("%Y-%m-%d")
        
        genre = rng.choice(GENRES)
        is_blockbuster = is_blockbuster_flags[i - 1]
        
        base_demand = rng.uniform(0.65, 0.90) if is_blockbuster else rng.uniform(0.30, 0.60)
        
        movies.append({
            "movie_id": f"MOV_{i:03d}",
            "movie_title": title,
            "genre": genre,
            "release_date": release_date,
            "is_blockbuster": is_blockbuster,
            "base_demand_rate": round(base_demand, 3),
            "run_weeks": int(rng.choice([4, 6, 8]) if is_blockbuster else rng.choice([2, 3, 4])),
        })
    return movies


def generate_full_dataset(
    num_cinemas: int = 30,
    num_movies: int = 100,
    target_shows: int = 30000,
    start_date: str = "2024-01-01",
    end_date: str = "2025-06-30",
    seed: int = 42,
    output_dir: str = "data/simulated",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate realistic show metadata and timestamped booking events table.
    """
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    
    print(f"Generating synthetic cinema catalog ({num_cinemas} cinemas)...")
    cinemas = build_cinema_catalog(num_cinemas, seed=seed)
    
    print(f"Generating synthetic movie catalog ({num_movies} movies)...")
    movies = build_movie_catalog(num_movies, start_date=start_date, end_date=end_date, seed=seed)
    movies_by_id = {m["movie_id"]: m for m in movies}
    
    simulator = BookingSimulator(BookingSimulationConfig(random_seed=seed))
    
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    date_range = pd.date_range(start_dt, end_dt, freq="D")
    total_days = len(date_range)
    
    shows_per_day = int(np.ceil(target_shows / total_days))
    print(f"Simulating across {total_days} days (~{shows_per_day} shows/day) targeting {target_shows} shows...")

    all_shows = []
    all_booking_events = []
    show_counter = 0

    for current_day in date_range:
        day_str = current_day.strftime("%Y-%m-%d")
        day_name = current_day.day_name()
        is_weekend = current_day.dayofweek >= 4  # Fri, Sat, Sun
        is_holiday = day_str in FIXED_HOLIDAYS
        
        # Day multiplier
        day_multiplier = 1.45 if is_weekend else 0.85
        if is_holiday:
            day_multiplier *= 1.30
        
        # Find movies currently in theaters on this day
        active_movies = []
        for m in movies:
            rel_dt = pd.to_datetime(m["release_date"])
            age_days = (current_day - rel_dt).days
            if 0 <= age_days <= (m["run_weeks"] * 7):
                active_movies.append((m, age_days))
                
        if not active_movies:
            # Fallback to recent movies
            active_movies = [(m, 14) for m in movies[:10]]
            
        # Select cinemas scheduled for this day
        # To get ~55 shows/day: sample ~15-20 cinema-screen-slot combinations
        daily_cinemas = rng.choice(cinemas, size=min(18, len(cinemas)), replace=False)
        
        for cin in daily_cinemas:
            # Pick 1-3 screens in this cinema today
            selected_screens = rng.choice(cin["screens"], size=min(len(cin["screens"]), rng.choice([2, 3])), replace=False)
            
            for scr in selected_screens:
                # Pick 1-2 slots
                selected_slots = rng.choice(SHOW_SLOTS, size=rng.choice([1, 2]), replace=False)
                
                for slot_info in selected_slots:
                    show_counter += 1
                    show_id = f"SHOW_{show_counter:05d}"
                    
                    # Choose a movie for this screen/slot
                    chosen_movie, age_days = active_movies[rng.integers(0, len(active_movies))]
                    
                    show_time_str = slot_info["time"]
                    show_datetime = pd.to_datetime(f"{day_str} {show_time_str}")
                    
                    # Compute demand expectation
                    genre_bias = GENRE_DEMAND_BIAS.get(chosen_movie["genre"], 1.0)
                    
                    # Exponential decay over release age (weeks)
                    age_decay = np.exp(-0.045 * (age_days / 7.0))
                    
                    expected_ratio = (
                        chosen_movie["base_demand_rate"]
                        * cin["traffic_multiplier"]
                        * slot_info["multiplier"]
                        * day_multiplier
                        * genre_bias
                        * age_decay
                    )
                    
                    # Price calculation
                    screen_type_mult = 1.4 if scr["capacity"] > 300 else (1.0 if scr["capacity"] > 150 else 0.85)
                    ticket_price = round(cin["base_price"] * screen_type_mult * (1.15 if is_weekend else 1.0), 2)
                    
                    # Simulate bookings
                    events, metrics = simulator.simulate_show_bookings(
                        show_id=show_id,
                        show_datetime=show_datetime,
                        screen_capacity=scr["capacity"],
                        expected_demand_ratio=expected_ratio,
                        is_weekend=is_weekend,
                        is_blockbuster=chosen_movie["is_blockbuster"],
                    )
                    
                    # We store events only for a subset of shows if needed to save memory,
                    # but storing all events gives full auditability!
                    # For performance, append events
                    all_booking_events.extend(events)
                    
                    occupancy = round(metrics["actual_tickets_sold"] / scr["capacity"], 4)
                    
                    show_record = {
                        "show_id": show_id,
                        "show_datetime": show_datetime,
                        "show_date": day_str,
                        "show_time": show_time_str,
                        "show_slot": slot_info["slot"],
                        "cinema_id": cin["cinema_id"],
                        "cinema_name": cin["cinema_name"],
                        "city": cin["city"],
                        "cinema_tier": cin["tier"],
                        "screen_id": f"SCR_{scr['screen_id']}",
                        "screen_name": scr["name"],
                        "screen_capacity": scr["capacity"],
                        "movie_id": chosen_movie["movie_id"],
                        "movie_title": chosen_movie["movie_title"],
                        "genre": chosen_movie["genre"],
                        "is_blockbuster": int(chosen_movie["is_blockbuster"]),
                        "movie_release_date": chosen_movie["release_date"],
                        "movie_release_age_days": max(0, age_days),
                        "day_of_week": day_name,
                        "is_weekend": int(is_weekend),
                        "is_holiday": int(is_holiday),
                        "ticket_price": ticket_price,
                        "advance_bookings_t48": metrics["advance_bookings_t48"],
                        "booking_velocity_24h": metrics["booking_velocity_24h"],
                        "booking_velocity_previous_24h": metrics["booking_velocity_previous_24h"],
                        "actual_tickets_sold": metrics["actual_tickets_sold"],
                        "occupancy_rate": occupancy,
                    }
                    all_shows.append(show_record)
                    
                    if show_counter >= target_shows:
                        break
                if show_counter >= target_shows:
                    break
            if show_counter >= target_shows:
                break
        if show_counter >= target_shows:
            break

    df_shows = pd.DataFrame(all_shows)
    df_shows.sort_values(by="show_datetime", inplace=True)
    df_shows.reset_index(drop=True, inplace=True)
    
    print(f"Generated {len(df_shows)} shows from {df_shows['show_date'].min()} to {df_shows['show_date'].max()}.")
    
    df_events = pd.DataFrame(all_booking_events)
    print(f"Generated {len(df_events)} granular booking event transactions.")
    
    # Save outputs to disk
    shows_parquet_path = os.path.join(output_dir, "shows.parquet")
    shows_csv_path = os.path.join(output_dir, "shows.csv")
    events_parquet_path = os.path.join(output_dir, "booking_events.parquet")
    
    print(f"Saving shows metadata to {shows_parquet_path} and {shows_csv_path}...")
    df_shows.to_parquet(shows_parquet_path, index=False)
    df_shows.to_csv(shows_csv_path, index=False)
    
    print(f"Saving booking events to {events_parquet_path}...")
    df_events.to_parquet(events_parquet_path, index=False)
    
    # Save simulation summary
    meta = {
        "dataset_name": "CinemaOps_Simulated_Benchmark",
        "description": "Production-grade synthetic cinema booking dataset with exact T-48 booking velocity and point-in-time transactions.",
        "generated_at": datetime.utcnow().isoformat(),
        "random_seed": seed,
        "total_shows": len(df_shows),
        "total_booking_events": len(df_events),
        "date_range": [str(df_shows["show_date"].min()), str(df_shows["show_date"].max())],
        "cinemas_count": df_shows["cinema_id"].nunique(),
        "movies_count": df_shows["movie_id"].nunique(),
        "mean_occupancy": float(df_shows["occupancy_rate"].mean()),
        "mean_capacity": float(df_shows["screen_capacity"].mean()),
        "mean_advance_t48": float(df_shows["advance_bookings_t48"].mean()),
        "mean_actual_sold": float(df_shows["actual_tickets_sold"].mean()),
    }
    with open(os.path.join(output_dir, "simulation_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
        
    print("Dataset generation complete!")
    return df_shows, df_events


if __name__ == "__main__":
    generate_full_dataset()

