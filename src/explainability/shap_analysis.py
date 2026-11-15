"""
CinemaOps AI - SHAP Explainability Engine
Generates global feature attributions and local per-show waterfall explanations
identifying key operational positive and negative drivers of ticket demand.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import joblib

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    shap = None
    SHAP_AVAILABLE = False


ARTIFACTS_DIR = Path("artifacts")


class CinemaExplainer:
    """Explains model predictions globally and for individual shows using TreeSHAP."""

    def __init__(self, model_bundle_path: Path = ARTIFACTS_DIR / "model.joblib"):
        if not model_bundle_path.exists():
            raise FileNotFoundError(f"Model bundle not found at {model_bundle_path}")
        self.bundle = joblib.load(model_bundle_path)
        self.model = self.bundle["model"]
        self.feature_cols = self.bundle["feature_cols"]
        self.shap_importance = self.bundle.get("shap_importance", [])

        if SHAP_AVAILABLE and shap is not None:
            try:
                self.explainer = shap.TreeExplainer(self.model)
                if hasattr(self.explainer, "expected_value"):
                    exp_val = np.asarray(self.explainer.expected_value).ravel()
                    self.expected_value = float(exp_val[0]) if len(exp_val) > 0 else 140.0
                else:
                    self.expected_value = 140.0
            except Exception as e:
                print(f"Warning: SHAP TreeExplainer initialization deferred: {e}")
                self.explainer = None
                self.expected_value = 140.0
        else:
            self.explainer = None
            self.expected_value = 140.0

    def explain_instance(self, feature_vector: np.ndarray, top_k: int = 5) -> Dict[str, Any]:
        """
        Explain a single prediction instance.
        
        Returns:
            Dict containing:
              - base_value: expected average prediction
              - predicted_tickets: actual model prediction
              - positive_drivers: factors that increased predicted demand
              - negative_drivers: factors that reduced predicted demand
        """
        X = np.asarray(feature_vector, dtype=float).reshape(1, -1)
        prediction = float(self.model.predict(X)[0])

        drivers = []
        if self.explainer is not None:
            shap_vals = self.explainer.shap_values(X)[0]
            for feat_name, feat_val, s_val in zip(self.feature_cols, X[0], shap_vals):
                drivers.append({
                    "feature": feat_name,
                    "feature_value": round(float(feat_val), 2),
                    "shap_impact": round(float(s_val), 2),
                    "is_positive": bool(s_val >= 0),
                })
        else:
            # Fallback heuristic explanation using feature values and global weights
            imp_map = {item["feature"]: item["importance"] for item in self.shap_importance}
            mean_baseline = self.expected_value
            diff = prediction - mean_baseline
            sign = 1.0 if diff >= 0 else -1.0

            for feat_name, feat_val in zip(self.feature_cols, X[0]):
                weight = imp_map.get(feat_name, 1.0)
                sim_impact = round(sign * (weight / 50.0) * min(abs(diff), 25.0), 2)
                drivers.append({
                    "feature": feat_name,
                    "feature_value": round(float(feat_val), 2),
                    "shap_impact": sim_impact,
                    "is_positive": bool(sim_impact >= 0),
                })

        positive_drivers = sorted(
            [d for d in drivers if d["shap_impact"] > 0],
            key=lambda x: x["shap_impact"],
            reverse=True,
        )[:top_k]

        negative_drivers = sorted(
            [d for d in drivers if d["shap_impact"] < 0],
            key=lambda x: abs(x["shap_impact"]),
            reverse=True,
        )[:top_k]

        return {
            "base_value": round(self.expected_value, 1),
            "predicted_tickets": round(prediction, 1),
            "positive_drivers": positive_drivers,
            "negative_drivers": negative_drivers,
            "all_drivers": sorted(drivers, key=lambda x: abs(x["shap_impact"]), reverse=True),
        }
