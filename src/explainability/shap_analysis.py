"""
CinemaOps AI - SHAP Explainability Engine
Generates global feature attributions and local per-show waterfall explanations
identifying key operational positive and negative drivers of ticket demand.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import shap
import joblib


ARTIFACTS_DIR = Path("artifacts")


class CinemaExplainer:
    """Explains model predictions globally and for individual shows using TreeSHAP."""

    def __init__(self, model_bundle_path: Path = ARTIFACTS_DIR / "model.joblib"):
        if not model_bundle_path.exists():
            raise FileNotFoundError(f"Model bundle not found at {model_bundle_path}")
        self.bundle = joblib.load(model_bundle_path)
        self.model = self.bundle["model"]
        self.feature_cols = self.bundle["feature_cols"]
        self.explainer = shap.TreeExplainer(self.model)
        if hasattr(self.explainer, "expected_value"):
            exp_val = np.asarray(self.explainer.expected_value).ravel()
            self.expected_value = float(exp_val[0]) if len(exp_val) > 0 else 140.0
        else:
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
        shap_vals = self.explainer.shap_values(X)[0]
        prediction = float(self.model.predict(X)[0])

        drivers = []
        for feat_name, feat_val, s_val in zip(self.feature_cols, X[0], shap_vals):
            drivers.append({
                "feature": feat_name,
                "feature_value": round(float(feat_val), 2),
                "shap_impact": round(float(s_val), 2),
                "is_positive": bool(s_val >= 0),
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
