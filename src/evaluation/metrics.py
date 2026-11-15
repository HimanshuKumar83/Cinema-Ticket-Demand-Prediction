"""Evaluation metrics and slicing analysis for Cinema Ticket Demand Forecasting."""
import numpy as np
import pandas as pd


def compute_metrics(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    capacities: np.ndarray,
    under_penalty: float = 1.5,
    over_penalty: float = 1.0
) -> dict:
    """Compute standard and capacity-aware regression metrics.
    
    Args:
        y_true: Ground truth ticket counts.
        y_pred: Predicted ticket counts (clipped to capacity).
        capacities: Screening auditorium capacities.
        under_penalty: Cost multiplier for under-predicting demand (stockout/staff shortage).
        over_penalty: Cost multiplier for over-predicting demand (idle staff cost).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    capacities = np.maximum(1.0, np.asarray(capacities, dtype=float))

    errors = y_true - y_pred
    abs_errors = np.abs(errors)

    mae = float(np.mean(abs_errors))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    
    sum_true = float(np.sum(y_true))
    wape = float((np.sum(abs_errors) / sum_true) * 100.0) if sum_true > 0 else 0.0

    # Occupancy MAE
    true_occu = y_true / capacities
    pred_occu = y_pred / capacities
    occu_mae = float(np.mean(np.abs(true_occu - pred_occu)))

    # Asymmetric operational cost: Under-forecasting vs Over-forecasting
    under_errors = np.maximum(0.0, errors)
    over_errors = np.maximum(0.0, -errors)
    asymmetric_cost = float(np.mean(under_penalty * under_errors + over_penalty * over_errors))

    # Share of shows where prediction was at or above capacity
    clipped_share = float(np.mean(y_pred >= (capacities - 1e-4)) * 100.0)

    return {
        "MAE": round(mae, 3),
        "RMSE": round(rmse, 3),
        "WAPE_pct": round(wape, 2),
        "Occupancy_MAE": round(occu_mae, 4),
        "Asymmetric_Cost": round(asymmetric_cost, 3),
        "Clipped_Capacity_Pct": round(clipped_share, 2),
        "N_samples": int(len(y_true))
    }


def evaluate_slices(
    df: pd.DataFrame, 
    y_pred: np.ndarray, 
    target_col: str = "tickets_sold",
    capacity_col: str = "capacity"
) -> dict:
    """Evaluate performance across operational slices:
    - Capacity buckets (Small <= 150, Medium 151-300, Large > 300)
    - Sold-out status (Sold-out vs Not sold-out)
    - Cold-start status (New film vs Established film; Low history vs High history)
    """
    df_eval = df.copy()
    df_eval["y_true"] = df_eval[target_col].to_numpy(dtype=float)
    df_eval["y_pred"] = y_pred
    df_eval["cap"] = df_eval[capacity_col].to_numpy(dtype=float)

    # 1. Capacity buckets
    cap_bins = [0, 150, 300, np.inf]
    cap_labels = ["Small (<=150)", "Medium (151-300)", "Large (>300)"]
    df_eval["cap_bucket"] = pd.cut(df_eval["cap"], bins=cap_bins, labels=cap_labels)

    slice_results = {"capacity_buckets": {}, "sold_out": {}, "cold_start": {}}

    for label in cap_labels:
        sub = df_eval[df_eval["cap_bucket"] == label]
        if len(sub) > 0:
            slice_results["capacity_buckets"][label] = compute_metrics(
                sub["y_true"].values, sub["y_pred"].values, sub["cap"].values
            )

    # 2. Sold-out vs Not sold-out
    df_eval["is_sold_out"] = df_eval["y_true"] >= df_eval["cap"]
    for status, label in [(True, "Sold-Out (100% capacity)"), (False, "Normal (<100% capacity)")]:
        sub = df_eval[df_eval["is_sold_out"] == status]
        if len(sub) > 0:
            slice_results["sold_out"][label] = compute_metrics(
                sub["y_true"].values, sub["y_pred"].values, sub["cap"].values
            )

    # 3. Cold-start slices (if present in dataframe)
    if "is_warm_start" in df_eval.columns:
        for status, label in [(1, "Warm-Start Shows (Movie in Train)"), (0, "Cold-Start Shows (New Movie)")]:
            sub = df_eval[df_eval["is_warm_start"] == status]
            if len(sub) > 0:
                slice_results["cold_start"][label] = compute_metrics(
                    sub["y_true"].values, sub["y_pred"].values, sub["cap"].values
                )

    if "is_new_film" in df_eval.columns:
        for status, label in [(1, "New Film (Zero Past Shows)"), (0, "Existing Film (Has History)")]:
            sub = df_eval[df_eval["is_new_film"] == status]
            if len(sub) > 0:
                slice_results["cold_start"][label] = compute_metrics(
                    sub["y_true"].values, sub["y_pred"].values, sub["cap"].values
                )

    if "is_low_history" in df_eval.columns:
        for status, label in [(1, "Low History (<10 shows)"), (0, "Adequate History (>=10 shows)")]:
            sub = df_eval[df_eval["is_low_history"] == status]
            if len(sub) > 0:
                slice_results["cold_start"][label] = compute_metrics(
                    sub["y_true"].values, sub["y_pred"].values, sub["cap"].values
                )

    return slice_results

