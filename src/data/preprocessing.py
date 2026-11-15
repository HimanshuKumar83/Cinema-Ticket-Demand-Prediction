"""
CinemaOps AI - Preprocessing and Split Pipeline
Validates schemas, enforces capacity bounds, ensures data integrity,
and generates leak-free chronological train/validation/test splits with explicit warm vs. cold-start tracking.
"""

import os
import json
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, List


REQUIRED_COLUMNS = [
    "show_id",
    "show_datetime",
    "show_date",
    "show_time",
    "show_slot",
    "cinema_id",
    "screen_id",
    "screen_capacity",
    "movie_id",
    "genre",
    "is_blockbuster",
    "movie_release_age_days",
    "day_of_week",
    "is_weekend",
    "is_holiday",
    "ticket_price",
    "advance_bookings_t48",
    "booking_velocity_24h",
    "booking_velocity_previous_24h",
    "actual_tickets_sold",
]


def validate_dataset_schema(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Verify all required columns exist and check for null values."""
    errors = []
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")

    null_counts = df[REQUIRED_COLUMNS].isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0].to_dict()
    if cols_with_nulls:
        errors.append(f"Columns with null values found: {cols_with_nulls}")

    return len(errors) == 0, errors


def validate_business_constraints(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Enforce fundamental operational constraints:
    1. 0 <= actual_tickets_sold <= screen_capacity
    2. 0 <= advance_bookings_t48 <= actual_tickets_sold
    3. booking_velocity_24h >= 0
    4. booking_velocity_previous_24h >= 0
    5. No duplicate show_ids
    """
    errors = []
    
    # Duplicate shows
    duplicates = df["show_id"].duplicated().sum()
    if duplicates > 0:
        errors.append(f"Found {duplicates} duplicate show_ids.")

    # Capacity overflow
    capacity_violations = (df["actual_tickets_sold"] > df["screen_capacity"]).sum()
    if capacity_violations > 0:
        errors.append(f"Found {capacity_violations} records where tickets sold exceeds screen capacity.")

    # Negative tickets sold
    negative_sold = (df["actual_tickets_sold"] < 0).sum()
    if negative_sold > 0:
        errors.append(f"Found {negative_sold} records with negative tickets sold.")

    # Advance bookings exceeding actual sold
    advance_violations = (df["advance_bookings_t48"] > df["actual_tickets_sold"]).sum()
    if advance_violations > 0:
        errors.append(f"Found {advance_violations} records where advance_bookings_t48 exceeds actual_tickets_sold.")

    # Negative velocities
    neg_vel = ((df["booking_velocity_24h"] < 0) | (df["booking_velocity_previous_24h"] < 0)).sum()
    if neg_vel > 0:
        errors.append(f"Found {neg_vel} records with negative booking velocities.")

    return len(errors) == 0, errors


def create_chronological_splits(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    output_dir: str = "data/processed",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Split the dataset chronologically:
    - 70% Train (oldest)
    - 15% Validation (middle)
    - 15% Test (most recent)
    
    Guarantees strict temporal separation with ZERO lookahead leakage,
    while tagging warm-start vs. cold-start shows on the held-out test set.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Ensure sorted by show_datetime
    df_sorted = df.sort_values(by="show_datetime").reset_index(drop=True).copy()
    
    n = len(df_sorted)
    train_idx = int(n * train_ratio)
    val_idx = int(n * (train_ratio + val_ratio))
    
    df_train = df_sorted.iloc[:train_idx].copy()
    df_val = df_sorted.iloc[train_idx:val_idx].copy()
    df_test = df_sorted.iloc[val_idx:].copy()
    
    # Verify split temporal disjointness
    train_end = df_train["show_datetime"].max()
    val_start = df_val["show_datetime"].min()
    val_end = df_val["show_datetime"].max()
    test_start = df_test["show_datetime"].min()
    
    assert train_end <= val_start, f"Leakage: Train end {train_end} > Val start {val_start}"
    assert val_end <= test_start, f"Leakage: Val end {val_end} > Test start {test_start}"

    # Track warm vs cold start on test partition
    train_movies = set(df_train["movie_id"].unique())
    test_movies = set(df_test["movie_id"].unique())
    overlap_movies = train_movies.intersection(test_movies)

    df_test["is_cold_start_show"] = (~df_test["movie_id"].isin(train_movies)).astype(int)
    warm_count = int((df_test["is_cold_start_show"] == 0).sum())
    cold_count = int((df_test["is_cold_start_show"] == 1).sum())
    warm_pct = round(float(warm_count / len(df_test) * 100.0), 2)
    cold_pct = round(float(cold_count / len(df_test) * 100.0), 2)

    # Assert non-empty warm and cold groups
    assert warm_count > 0, "Warm-start test partition must be non-empty!"
    assert cold_count > 0, "Cold-start test partition must be non-empty!"
    
    # Save parquet splits
    train_path = os.path.join(output_dir, "train.parquet")
    val_path = os.path.join(output_dir, "val.parquet")
    test_path = os.path.join(output_dir, "test.parquet")
    
    df_train.to_parquet(train_path, index=False)
    df_val.to_parquet(val_path, index=False)
    df_test.to_parquet(test_path, index=False)
    
    split_info = {
        "train": {
            "count": len(df_train),
            "start_date": str(df_train["show_date"].min()),
            "end_date": str(df_train["show_date"].max()),
            "mean_tickets_sold": float(df_train["actual_tickets_sold"].mean()),
            "mean_occupancy": float(df_train["occupancy_rate"].mean()),
            "unique_movies": len(train_movies),
        },
        "val": {
            "count": len(df_val),
            "start_date": str(df_val["show_date"].min()),
            "end_date": str(df_val["show_date"].max()),
            "mean_tickets_sold": float(df_val["actual_tickets_sold"].mean()),
            "mean_occupancy": float(df_val["occupancy_rate"].mean()),
            "unique_movies": int(df_val["movie_id"].nunique()),
        },
        "test": {
            "count": len(df_test),
            "start_date": str(df_test["show_date"].min()),
            "end_date": str(df_test["show_date"].max()),
            "mean_tickets_sold": float(df_test["actual_tickets_sold"].mean()),
            "mean_occupancy": float(df_test["occupancy_rate"].mean()),
            "unique_movies": len(test_movies),
            "movie_overlap_with_train": len(overlap_movies),
            "warm_start_shows": warm_count,
            "cold_start_shows": cold_count,
            "warm_start_pct": warm_pct,
            "cold_start_pct": cold_pct,
        },
        "chronological_integrity": True,
    }
    
    with open(os.path.join(output_dir, "split_metadata.json"), "w") as f:
        json.dump(split_info, f, indent=2)
        
    print(f"Chronological split complete: {len(df_train)} train, {len(df_val)} val, {len(df_test)} test.")
    print(f"Test Split Composition: {warm_count} warm-start ({warm_pct}%), {cold_count} cold-start ({cold_pct}%), {len(overlap_movies)} overlapping movies.")
    return df_train, df_val, df_test, split_info


def preprocess_and_split(input_file: str = "data/simulated/shows.parquet", output_dir: str = "data/processed") -> None:
    """End-to-end preprocessing pipeline."""
    print(f"Reading raw dataset from {input_file}...")
    df = pd.read_parquet(input_file)
    
    print("Validating schema...")
    schema_ok, schema_errs = validate_dataset_schema(df)
    if not schema_ok:
        raise ValueError(f"Schema validation failed: {schema_errs}")
        
    print("Validating operational business constraints...")
    constraints_ok, constraint_errs = validate_business_constraints(df)
    if not constraints_ok:
        raise ValueError(f"Business constraint validation failed: {constraint_errs}")
        
    print("All validations PASSED! Creating chronological train/val/test splits...")
    create_chronological_splits(df, output_dir=output_dir)


if __name__ == "__main__":
    preprocess_and_split()
