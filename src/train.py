"""
CinemaOps AI - Multi-Model Training and Benchmark Suite
Trains and validates 5 candidate architectures under strict chronological splitting:
1. Hierarchical Occupancy Baseline (4-level fallback)
2. Regularized Linear Model (Ridge Regression)
3. Random Forest Regressor
4. XGBoost Regressor
5. LightGBM Regressor (L1 Loss)

Evaluates MAE, RMSE, WAPE, Occupancy MAE, Capacity Violations, Cold-Start slices,
computes 80% conformal prediction intervals, and computes SHAP feature attributions.
"""

import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import shap
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb

from src.models.baseline import HierarchicalOccupancyBaseline
from src.features.build_features import compute_features, HistoryPriors
from src.evaluation.metrics import compute_metrics, evaluate_slices
from src.data.preprocessing import preprocess_and_split


ARTIFACTS_DIR = Path("artifacts")
REPORTS_DIR = Path("reports")
PROCESSED_DIR = Path("data/processed")
RANDOM_SEED = 42


def get_feature_columns(df_features: pd.DataFrame) -> list:
    """Return explicit list of predictor feature columns, strictly excluding target and post-show leaks."""
    forbidden = {
        "actual_tickets_sold", "tickets_sold", "occupancy", "occupancy_rate",
        "total_sales", "tickets_out", "ticket_use", "show_datetime", "show_date",
        "show_time", "show_id", "cinema_id", "movie_id", "screen_id"
    }
    return [c for c in df_features.columns if c not in forbidden]


def train_and_benchmark_all():
    """Execute complete end-to-end multi-model training and evaluation benchmark."""
    print("=" * 80)
    print("CINEMAOPS AI: MULTI-MODEL TRAINING & BENCHMARK PIPELINE")
    print("=" * 80)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load or prepare processed splits
    train_path = PROCESSED_DIR / "train.parquet"
    val_path = PROCESSED_DIR / "val.parquet"
    test_path = PROCESSED_DIR / "test.parquet"

    if not (train_path.exists() and val_path.exists() and test_path.exists()):
        print("Processed splits not found. Running preprocessing pipeline...")
        preprocess_and_split()

    print(f"Loading datasets from {PROCESSED_DIR}...")
    df_train = pd.read_parquet(train_path)
    df_val = pd.read_parquet(val_path)
    df_test = pd.read_parquet(test_path)

    print(f"Train set: {len(df_train)} shows ({df_train['show_date'].min()} to {df_train['show_date'].max()})")
    print(f"Val set:   {len(df_val)} shows ({df_val['show_date'].min()} to {df_val['show_date'].max()})")
    print(f"Test set:  {len(df_test)} shows ({df_test['show_date'].min()} to {df_test['show_date'].max()})")

    # 2. Compute priors strictly on train split
    print("\nComputing historical priors on training split...")
    priors = HistoryPriors.fit_from_train(df_train)
    priors.save(ARTIFACTS_DIR / "priors.json")

    # 3. Extract leak-free features
    print("Extracting features for Train, Val, and Test...")
    X_train_df, _ = compute_features(df_train, priors=priors)
    X_val_df, _ = compute_features(df_val, priors=priors)
    X_test_df, _ = compute_features(df_test, priors=priors)

    feature_cols = get_feature_columns(X_train_df)
    print(f"Total modeling features ({len(feature_cols)}): {feature_cols}")

    with open(ARTIFACTS_DIR / "feature_list.json", "w") as f:
        json.dump(feature_cols, f, indent=2)

    X_train = X_train_df[feature_cols].values
    X_val = X_val_df[feature_cols].values
    X_test = X_test_df[feature_cols].values

    y_train = df_train["actual_tickets_sold"].to_numpy(dtype=float)
    y_val = df_val["actual_tickets_sold"].to_numpy(dtype=float)
    y_test = df_test["actual_tickets_sold"].to_numpy(dtype=float)

    cap_train = df_train["screen_capacity"].to_numpy(dtype=float)
    cap_val = df_val["screen_capacity"].to_numpy(dtype=float)
    cap_test = df_test["screen_capacity"].to_numpy(dtype=float)

    benchmark_records = []
    models_dict = {}

    # -------------------------------------------------------------
    # MODEL 1: Hierarchical Occupancy Baseline (4-Level Fallback)
    # -------------------------------------------------------------
    print("\n[1/5] Training Model 1: Hierarchical Occupancy Baseline...")
    baseline_model = HierarchicalOccupancyBaseline()
    baseline_model.fit(df_train)
    models_dict["baseline"] = baseline_model

    raw_preds_base_test = baseline_model.predict(df_test)
    preds_base_test = np.clip(raw_preds_base_test, 0.0, cap_test)
    raw_preds_base_val = baseline_model.predict(df_val)
    preds_base_val = np.clip(raw_preds_base_val, 0.0, cap_val)

    m_base = compute_metrics(y_test, preds_base_test, cap_test)
    m_base["Violation_Rate_Pct"] = float(np.mean(raw_preds_base_test > cap_test) * 100.0)
    benchmark_records.append({"Model": "Hierarchical Baseline", **m_base})
    print(f"  Baseline Test -> MAE: {m_base['MAE']:.2f}, RMSE: {m_base['RMSE']:.2f}, WAPE: {m_base['WAPE_pct']:.2f}%")

    # -------------------------------------------------------------
    # MODEL 2: Ridge Regression (Scaled Linear Benchmark)
    # -------------------------------------------------------------
    print("\n[2/5] Training Model 2: Ridge Regression...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    ridge_model = Ridge(alpha=50.0, random_state=RANDOM_SEED)
    ridge_model.fit(X_train_scaled, y_train)
    models_dict["ridge"] = ridge_model
    models_dict["scaler"] = scaler

    raw_preds_ridge_test = ridge_model.predict(X_test_scaled)
    preds_ridge_test = np.clip(raw_preds_ridge_test, 0.0, cap_test)
    raw_preds_ridge_val = ridge_model.predict(X_val_scaled)
    preds_ridge_val = np.clip(raw_preds_ridge_val, 0.0, cap_val)

    m_ridge = compute_metrics(y_test, preds_ridge_test, cap_test)
    m_ridge["Violation_Rate_Pct"] = float(np.mean(raw_preds_ridge_test > cap_test) * 100.0)
    benchmark_records.append({"Model": "Ridge Regression", **m_ridge})
    print(f"  Ridge Test      -> MAE: {m_ridge['MAE']:.2f}, RMSE: {m_ridge['RMSE']:.2f}, WAPE: {m_ridge['WAPE_pct']:.2f}%")

    # -------------------------------------------------------------
    # MODEL 3: Random Forest Regressor
    # -------------------------------------------------------------
    print("\n[3/5] Training Model 3: Random Forest Regressor...")
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=14,
        min_samples_split=10,
        min_samples_leaf=4,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    rf_model.fit(X_train, y_train)
    models_dict["random_forest"] = rf_model

    raw_preds_rf_test = rf_model.predict(X_test)
    preds_rf_test = np.clip(raw_preds_rf_test, 0.0, cap_test)
    raw_preds_rf_val = rf_model.predict(X_val)
    preds_rf_val = np.clip(raw_preds_rf_val, 0.0, cap_val)

    m_rf = compute_metrics(y_test, preds_rf_test, cap_test)
    m_rf["Violation_Rate_Pct"] = float(np.mean(raw_preds_rf_test > cap_test) * 100.0)
    benchmark_records.append({"Model": "Random Forest", **m_rf})
    print(f"  Random Forest   -> MAE: {m_rf['MAE']:.2f}, RMSE: {m_rf['RMSE']:.2f}, WAPE: {m_rf['WAPE_pct']:.2f}%")

    # -------------------------------------------------------------
    # MODEL 4: XGBoost Regressor
    # -------------------------------------------------------------
    print("\n[4/5] Training Model 4: XGBoost Regressor...")
    xgb_model = xgb.XGBRegressor(
        n_estimators=250,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    models_dict["xgboost"] = xgb_model

    raw_preds_xgb_test = xgb_model.predict(X_test)
    preds_xgb_test = np.clip(raw_preds_xgb_test, 0.0, cap_test)
    raw_preds_xgb_val = xgb_model.predict(X_val)
    preds_xgb_val = np.clip(raw_preds_xgb_val, 0.0, cap_val)

    m_xgb = compute_metrics(y_test, preds_xgb_test, cap_test)
    m_xgb["Violation_Rate_Pct"] = float(np.mean(raw_preds_xgb_test > cap_test) * 100.0)
    benchmark_records.append({"Model": "XGBoost", **m_xgb})
    print(f"  XGBoost         -> MAE: {m_xgb['MAE']:.2f}, RMSE: {m_xgb['RMSE']:.2f}, WAPE: {m_xgb['WAPE_pct']:.2f}%")

    # -------------------------------------------------------------
    # MODEL 5: LightGBM Regressor (L1 Loss)
    # -------------------------------------------------------------
    print("\n[5/5] Training Model 5: LightGBM Regressor (L1 Optimization)...")
    lgb_model = lgb.LGBMRegressor(
        objective="regression_l1",
        n_estimators=300,
        learning_rate=0.04,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbose=-1,
    )
    lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    models_dict["lightgbm"] = lgb_model

    raw_preds_lgb_test = lgb_model.predict(X_test)
    preds_lgb_test = np.clip(raw_preds_lgb_test, 0.0, cap_test)
    raw_preds_lgb_val = lgb_model.predict(X_val)
    preds_lgb_val = np.clip(raw_preds_lgb_val, 0.0, cap_val)

    m_lgb = compute_metrics(y_test, preds_lgb_test, cap_test)
    m_lgb["Violation_Rate_Pct"] = float(np.mean(raw_preds_lgb_test > cap_test) * 100.0)
    benchmark_records.append({"Model": "LightGBM", **m_lgb})
    print(f"  LightGBM        -> MAE: {m_lgb['MAE']:.2f}, RMSE: {m_lgb['RMSE']:.2f}, WAPE: {m_lgb['WAPE_pct']:.2f}%")

    # -------------------------------------------------------------
    # MODEL BENCHMARK TABLE COMPARISON
    # -------------------------------------------------------------
    df_benchmark = pd.DataFrame(benchmark_records).sort_values(by="MAE").reset_index(drop=True)
    print("\n" + "=" * 80)
    print("5-MODEL HELD-OUT CHRONOLOGICAL TEST BENCHMARK RESULTS")
    print("=" * 80)
    print(df_benchmark.to_string(index=False))

    benchmark_json = df_benchmark.to_dict(orient="records")
    with open(ARTIFACTS_DIR / "model_benchmark.json", "w") as f:
        json.dump(benchmark_json, f, indent=2)

    # -------------------------------------------------------------
    # UNCERTAINTY CALIBRATION (Conformal Prediction 80% Interval)
    # -------------------------------------------------------------
    print("\nComputing 80% Conformal Prediction Intervals from Validation Residuals...")
    val_residuals = y_val - preds_lgb_val
    q10 = float(np.percentile(val_residuals, 10.0))
    q90 = float(np.percentile(val_residuals, 90.0))

    # Test coverage
    test_lower = np.maximum(0.0, preds_lgb_test + q10)
    test_upper = np.minimum(cap_test, preds_lgb_test + q90)
    empirical_coverage = float(np.mean((y_test >= test_lower) & (y_test <= test_upper)) * 100.0)
    mean_interval_width = float(np.mean(test_upper - test_lower))

    conformal_summary = {
        "target_coverage_pct": 80.0,
        "residual_q10": round(q10, 3),
        "residual_q90": round(q90, 3),
        "empirical_test_coverage_pct": round(empirical_coverage, 2),
        "mean_interval_width": round(mean_interval_width, 2),
    }
    print(f"  Residual Q10: {q10:.2f}, Q90: {q90:.2f}")
    print(f"  Empirical Test Coverage: {empirical_coverage:.2f}% (Target: 80%)")
    print(f"  Mean 80% Interval Width: {mean_interval_width:.1f} tickets")

    with open(ARTIFACTS_DIR / "conformal_quantiles.json", "w") as f:
        json.dump(conformal_summary, f, indent=2)

    # -------------------------------------------------------------
    # COLD-START & SLICING EVALUATION
    # -------------------------------------------------------------
    print("\nEvaluating Operational Slices & Cold-Start Segments on LightGBM...")
    df_test_eval = df_test.copy()
    df_test_eval["is_new_film"] = X_test_df["is_new_movie"]
    df_test_eval["is_low_history"] = X_test_df["is_low_history"]
    df_test_eval["capacity"] = cap_test
    df_test_eval["tickets_sold"] = y_test

    slice_metrics = evaluate_slices(df_test_eval, preds_lgb_test, target_col="tickets_sold", capacity_col="capacity")

    # -------------------------------------------------------------
    # SHAP EXPLAINABILITY (Global & Local)
    # -------------------------------------------------------------
    print("\nComputing SHAP Attributions for LightGBM...")
    explainer = shap.TreeExplainer(lgb_model)
    # Sample 800 test instances for fast and accurate attributions
    sample_size = min(800, len(X_test))
    shap_sample = X_test[:sample_size]
    shap_values = explainer.shap_values(shap_sample)

    # Global feature importance = mean(|SHAP value|)
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    shap_importance = sorted(
        [{"feature": f, "importance": round(float(imp), 4)} for f, imp in zip(feature_cols, mean_abs_shap)],
        key=lambda x: x["importance"],
        reverse=True,
    )

    with open(ARTIFACTS_DIR / "shap_summary.json", "w") as f:
        json.dump(shap_importance, f, indent=2)

    with open(ARTIFACTS_DIR / "feature_importance.json", "w") as f:
        json.dump(shap_importance, f, indent=2)

    print("Top 8 Features by Global SHAP Importance:")
    for item in shap_importance[:8]:
        print(f"  - {item['feature']}: {item['importance']:.4f}")

    # Generate SHAP Summary Plot
    try:
        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            shap_values,
            shap_sample,
            feature_names=feature_cols,
            show=False,
            max_display=12
        )
        plt.tight_layout()
        plt.savefig(REPORTS_DIR / "shap_summary.png", dpi=200, bbox_inches="tight")
        plt.close()
        print(f"SHAP summary plot saved to {REPORTS_DIR / 'shap_summary.png'}")
    except Exception as e:
        print(f"Warning: SHAP plot rendering skipped: {e}")

    # -------------------------------------------------------------
    # SAVE MODEL BUNDLE & FULL METRICS ARTIFACTS
    # -------------------------------------------------------------
    full_metrics = {
        "benchmark": benchmark_json,
        "conformal": conformal_summary,
        "operational_slices": slice_metrics,
        "best_model": "lightgbm",
        "best_model_test_mae": m_lgb["MAE"],
        "best_model_test_wape": m_lgb["WAPE_pct"],
        "best_model_test_rmse": m_lgb["RMSE"],
    }

    with open(ARTIFACTS_DIR / "metrics.json", "w") as f:
        json.dump(full_metrics, f, indent=2)

    bundle = {
        "model": lgb_model,
        "all_models": {
            "baseline": baseline_model,
            "ridge": ridge_model,
            "random_forest": rf_model,
            "xgboost": xgb_model,
            "lightgbm": lgb_model,
        },
        "scaler": scaler,
        "feature_cols": feature_cols,
        "residual_q10": q10,
        "residual_q90": q90,
        "priors": priors.priors,
        "metrics": full_metrics,
        "shap_importance": shap_importance,
    }

    joblib.dump(bundle, ARTIFACTS_DIR / "model.joblib")
    print(f"\nSerialized model bundle to {ARTIFACTS_DIR / 'model.joblib'}")
    print("=" * 80)
    print("MULTI-MODEL BENCHMARK COMPLETED SUCCESSFULLY!")
    print("=" * 80)

    return bundle, full_metrics


if __name__ == "__main__":
    train_and_benchmark_all()
