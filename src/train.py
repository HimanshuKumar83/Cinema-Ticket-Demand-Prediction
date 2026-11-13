"""End-to-end training and evaluation pipeline for Cinema Ticket Demand Forecasting.
Reproducible, leakage-free, and cross-platform.
"""
import json
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, TargetEncoder

from src.data.clean import clean_dataset, generate_leakage_table
from src.evaluation.metrics import compute_metrics, evaluate_slices
from src.features.build_features import (
    HistoryFeatureStore,
    create_chronological_splits,
    extract_features_for_dataset,
)
from src.models.baseline import HierarchicalOccupancyBaseline

ARTIFACTS_DIR = Path("artifacts")
REPORTS_DIR = Path("reports")
PROCESSED_DIR = Path("data/processed")
RANDOM_SEED = 42


def get_feature_columns(df_features: pd.DataFrame) -> list[str]:
    """Return explicit list of predictor feature columns, strictly excluding targets and leaks."""
    forbidden = {
        "tickets_sold", "occupancy", "total_sales", "tickets_out", 
        "ticket_use", "occu_perc", "ticket_price", "date"
    }
    feature_cols = [c for c in df_features.columns if c not in forbidden]
    return feature_cols


def train_and_evaluate_all():
    """Execute complete end-to-end training, model comparison, and artifact serialization."""
    print("=" * 80)
    print("STEP 2 & 3: CLEANING, FEATURES, TRAINING & EVALUATION")
    print("=" * 80)

    # 1. Clean data and generate leakage table
    clean_parquet = PROCESSED_DIR / "cleaned_cinema_shows.parquet"
    if not clean_parquet.exists():
        print("Cleaning dataset...")
        generate_leakage_table()
        df_clean, _ = clean_dataset()
    else:
        print(f"Loading cleaned dataset from {clean_parquet}...")
        df_clean = pd.read_parquet(clean_parquet)

    # 2. Chronological Splits (~70% train / ~15% val / ~15% test)
    print("\nCreating chronological splits...")
    splits = create_chronological_splits(df_clean)
    df_train = splits["train"]
    df_val = splits["val"]
    df_test = splits["test"]

    # 3. Initialize History Feature Store and save lookup
    print("\nInitializing HistoryFeatureStore (T-48h horizon)...")
    history_store = HistoryFeatureStore(df_clean)
    history_store.save_lookup(ARTIFACTS_DIR / "history.parquet")

    # 4. Extract features
    print("Extracting features for Train set...")
    X_train_raw = extract_features_for_dataset(df_train, history_store)
    print("Extracting features for Val set...")
    X_val_raw = extract_features_for_dataset(df_val, history_store)
    print("Extracting features for Test set...")
    X_test_raw = extract_features_for_dataset(df_test, history_store)

    feature_cols = get_feature_columns(X_train_raw)
    print(f"\nTotal predictor features: {len(feature_cols)}")
    
    # Save feature list
    with open(ARTIFACTS_DIR / "feature_list.json", "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, indent=2)

    y_train = df_train["tickets_sold"].to_numpy(dtype=float)
    y_val = df_val["tickets_sold"].to_numpy(dtype=float)
    y_test = df_test["tickets_sold"].to_numpy(dtype=float)

    cap_train = df_train["capacity"].to_numpy(dtype=float)
    cap_val = df_val["capacity"].to_numpy(dtype=float)
    cap_test = df_test["capacity"].to_numpy(dtype=float)

    # 5. Baseline: Hierarchical Occupancy Model
    print("\n--- Training Hierarchical Occupancy Baseline ---")
    baseline = HierarchicalOccupancyBaseline()
    baseline.fit(df_train)

    val_preds_base = baseline.predict(df_val)
    test_preds_base = baseline.predict(df_test)

    val_metrics_base = compute_metrics(y_val, val_preds_base, cap_val)
    test_metrics_base = compute_metrics(y_test, test_preds_base, cap_test)
    print("Baseline Test Metrics:", test_metrics_base)

    # 6. Candidate 1: Ridge Regression (predicting occupancy)
    print("\n--- Training Ridge Regression (on Occupancy) ---")
    y_train_occu = y_train / np.maximum(1.0, cap_train)
    
    cat_id_cols = ["cinema_code", "film_code"]
    num_cols = [c for c in feature_cols if c not in cat_id_cols]

    preprocessor_ridge = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), num_cols),
            ("cat", TargetEncoder(cv=5, smooth="auto", random_state=RANDOM_SEED), cat_id_cols)
        ]
    )

    ridge_pipeline = Pipeline([
        ("preprocessor", preprocessor_ridge),
        ("regressor", Ridge(alpha=100.0, random_state=RANDOM_SEED))
    ])

    ridge_pipeline.fit(X_train_raw[feature_cols], y_train_occu)

    # Predict occupancy, multiply by capacity, clip to [0, capacity]
    val_preds_ridge_occu = ridge_pipeline.predict(X_val_raw[feature_cols])
    val_preds_ridge = np.clip(val_preds_ridge_occu * cap_val, 0.0, cap_val)

    test_preds_ridge_occu = ridge_pipeline.predict(X_test_raw[feature_cols])
    test_preds_ridge = np.clip(test_preds_ridge_occu * cap_test, 0.0, cap_test)

    val_metrics_ridge = compute_metrics(y_val, val_preds_ridge, cap_val)
    test_metrics_ridge = compute_metrics(y_test, test_preds_ridge, cap_test)
    print("Ridge Test Metrics:", test_metrics_ridge)

    # 7. Candidate 2: LightGBM Regressor (predicting tickets directly with L1 loss)
    print("\n--- Training LightGBM Regressor ---")
    cat_features_lgb = ["cinema_code", "film_code", "weekday", "month", "quarter", "show_time", "is_weekend"]
    
    # Format categoricals
    X_train_lgb = X_train_raw[feature_cols].copy()
    X_val_lgb = X_val_raw[feature_cols].copy()
    X_test_lgb = X_test_raw[feature_cols].copy()

    for col in cat_features_lgb:
        if col in X_train_lgb.columns:
            X_train_lgb[col] = X_train_lgb[col].astype("category")
            X_val_lgb[col] = X_val_lgb[col].astype("category")
            X_test_lgb[col] = X_test_lgb[col].astype("category")

    lgb_model = LGBMRegressor(
        objective="regression_l1",  # Optimizes MAE directly
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=25,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbose=-1
    )

    lgb_model.fit(
        X_train_lgb, 
        y_train,
        eval_set=[(X_val_lgb, y_val)],
        callbacks=[]
    )

    val_preds_lgb_raw = lgb_model.predict(X_val_lgb)
    val_preds_lgb = np.clip(val_preds_lgb_raw, 0.0, cap_val)

    test_preds_lgb_raw = lgb_model.predict(X_test_lgb)
    test_preds_lgb = np.clip(test_preds_lgb_raw, 0.0, cap_test)

    val_metrics_lgb = compute_metrics(y_val, val_preds_lgb, cap_val)
    test_metrics_lgb = compute_metrics(y_test, test_preds_lgb, cap_test)
    print("LightGBM Test Metrics:", test_metrics_lgb)

    # 8. Model Comparison Table
    print("\n" + "=" * 80)
    print("TEST-SET METRIC COMPARISON TABLE (Held-out 2018-09-30 to 2018-11-04)")
    print("=" * 80)
    comparison_df = pd.DataFrame([
        {"Model": "Hierarchical Baseline", **test_metrics_base},
        {"Model": "Ridge (Occupancy)", **test_metrics_ridge},
        {"Model": "LightGBM (Tickets L1)", **test_metrics_lgb}
    ])
    print(comparison_df.to_string(index=False))

    # Log experiments
    experiments_records = [
        {"timestamp": pd.Timestamp.now().isoformat(), "model": "Hierarchical Baseline", "split": "val", **val_metrics_base},
        {"timestamp": pd.Timestamp.now().isoformat(), "model": "Hierarchical Baseline", "split": "test", **test_metrics_base},
        {"timestamp": pd.Timestamp.now().isoformat(), "model": "Ridge (Occupancy)", "split": "val", **val_metrics_ridge},
        {"timestamp": pd.Timestamp.now().isoformat(), "model": "Ridge (Occupancy)", "split": "test", **test_metrics_ridge},
        {"timestamp": pd.Timestamp.now().isoformat(), "model": "LightGBM (Tickets L1)", "split": "val", **val_metrics_lgb},
        {"timestamp": pd.Timestamp.now().isoformat(), "model": "LightGBM (Tickets L1)", "split": "test", **test_metrics_lgb},
    ]
    pd.DataFrame(experiments_records).to_csv(REPORTS_DIR / "experiments.csv", index=False)

    # 9. Uncertainty Calibration (80% Prediction Interval from Validation Residuals)
    val_residuals = y_val - val_preds_lgb
    q10 = float(np.percentile(val_residuals, 10.0))
    q90 = float(np.percentile(val_residuals, 90.0))

    # Test coverage
    test_lower = np.maximum(0.0, test_preds_lgb + q10)
    test_upper = np.minimum(cap_test, test_preds_lgb + q90)
    test_coverage_pct = float(np.mean((y_test >= test_lower) & (y_test <= test_upper)) * 100.0)
    print(f"\nUncertainty Calibration: Val Residual Q10={q10:.2f}, Q90={q90:.2f}")
    print(f"Empirical 80% Prediction Interval Test Coverage: {test_coverage_pct:.2f}%")

    # 10. Operational Slicing Analysis on Best Model (LightGBM)
    slice_eval = evaluate_slices(df_test.assign(**X_test_raw), test_preds_lgb)
    print("\n--- Capacity Bucket Slices ---")
    print(pd.DataFrame(slice_eval["capacity_buckets"]).T.to_string())

    print("\n--- Cold-Start & History Slices ---")
    print(pd.DataFrame(slice_eval["cold_start"]).T.to_string())

    # 11. Error Analysis: Top 10 worst predictions on test set
    test_errors = np.abs(y_test - test_preds_lgb)
    top10_idx = np.argsort(test_errors)[::-1][:10]

    worst_records = []
    for idx in top10_idx:
        row = df_test.iloc[idx]
        actual = float(y_test[idx])
        pred = float(test_preds_lgb[idx])
        err = float(test_errors[idx])
        cap = float(cap_test[idx])
        
        # Operational reason heuristic
        if actual >= cap:
            reason = "Sudden sell-out / censored blockbuster surge"
        elif pred > actual and actual <= 10:
            reason = "Rapid demand dropoff after opening release"
        elif actual > pred * 2:
            reason = "Unexpected local event or holiday attendance spike"
        else:
            reason = "High theater capacity variance"

        worst_records.append({
            "date": str(row["date"])[:10],
            "cinema_code": int(row["cinema_code"]),
            "film_code": int(row["film_code"]),
            "show_time": int(row["show_time"]),
            "capacity": cap,
            "actual_tickets": actual,
            "predicted_tickets": round(pred, 1),
            "abs_error": round(err, 1),
            "likely_operational_reason": reason
        })

    worst_df = pd.DataFrame(worst_records)
    worst_df.to_csv(REPORTS_DIR / "top10_worst_predictions.csv", index=False)
    print("\n--- Top 10 Worst Test Predictions ---")
    print(worst_df[["date", "film_code", "cinema_code", "actual_tickets", "predicted_tickets", "abs_error", "likely_operational_reason"]].to_string(index=False))

    # 12. Residual Plot
    plt.figure(figsize=(10, 5))
    plt.scatter(test_preds_lgb, y_test - test_preds_lgb, alpha=0.15, s=8, color="#1f77b4")
    plt.axhline(0, color="red", linestyle="--", linewidth=1.5)
    plt.axhline(q10, color="orange", linestyle=":", label=f"Val Residual Q10 ({q10:.1f})")
    plt.axhline(q90, color="orange", linestyle=":", label=f"Val Residual Q90 ({q90:.1f})")
    plt.title("Residual Plot on Held-Out Test Set (LightGBM)")
    plt.xlabel("Predicted Tickets")
    plt.ylabel("Residual (Actual - Predicted)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "residual_plot.png", dpi=200)
    plt.close()
    print(f"Residual plot saved to {REPORTS_DIR / 'residual_plot.png'}")

    # 13. Global Feature Importance (Gain)
    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "importance_gain": lgb_model.booster_.feature_importance(importance_type="gain")
    }).sort_values(by="importance_gain", ascending=False).reset_index(drop=True)
    importance_df.to_csv(REPORTS_DIR / "feature_importance.csv", index=False)
    print("\n--- Top 10 Features by Importance Gain ---")
    print(importance_df.head(10).to_string(index=False))

    # 14. Save Complete Artifacts
    full_metrics = {
        "models": {
            "baseline": test_metrics_base,
            "ridge": test_metrics_ridge,
            "lightgbm": test_metrics_lgb
        },
        "uncertainty_80pct": {
            "q10_residual": q10,
            "q90_residual": q90,
            "test_coverage_pct": test_coverage_pct
        },
        "operational_slices": slice_eval,
        "selected_best_model": "lightgbm"
    }
    with open(ARTIFACTS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(full_metrics, f, indent=2)

    # Save trained model bundle for inference
    model_bundle = {
        "model": lgb_model,
        "ridge_model": ridge_pipeline,
        "baseline_model": baseline,
        "feature_cols": feature_cols,
        "cat_features": cat_features_lgb,
        "residual_q10": q10,
        "residual_q90": q90,
        "metrics": full_metrics
    }
    joblib.dump(model_bundle, ARTIFACTS_DIR / "model.joblib")
    print(f"\nModel bundle saved to {ARTIFACTS_DIR / 'model.joblib'}")
    print("=" * 80)
    print("STEP 2 & 3 COMPLETED SUCCESSFULLY.")
    print("=" * 80)

    return model_bundle, full_metrics


if __name__ == "__main__":
    train_and_evaluate_all()

