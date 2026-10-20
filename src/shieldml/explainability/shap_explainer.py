"""
SHAP Explainability module for ShieldML.
Uses TreeExplainer for tree models (XGBoost, LightGBM, Random Forest),
generates global importance charts, representative waterfall explanations,
and real-time feature contributions for the prediction API.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

logger = logging.getLogger(__name__)


class FraudExplainer:
    """Computes global and local feature importance via SHAP TreeExplainer."""

    def __init__(
        self,
        model: Any,
        feature_names: list[str],
        background_data: pd.DataFrame | np.ndarray | None = None,
        max_background_samples: int = 100,
    ):
        self.feature_names = feature_names
        self.raw_model = self._unwrap_model(model)
        self.max_background_samples = max_background_samples
        self.explainer: shap.TreeExplainer | shap.Explainer | None = None
        self._init_explainer(background_data)

    def _unwrap_model(self, model: Any) -> Any:
        """Extracts the underlying tree model from wrappers (CalibratedClassifierCV, Ensemble)."""
        # If calibrated classifier, extract the base calibrated estimator
        if hasattr(model, "calibrated_classifiers_") and model.calibrated_classifiers_:
            return model.calibrated_classifiers_[0].estimator
        if hasattr(model, "estimator"):
            return model.estimator
        # If soft-voting ensemble, use the primary gradient booster (XGBoost/LightGBM)
        if hasattr(model, "models") and model.models:
            # Prefer XGBoost or LightGBM model for TreeExplainer
            for m in model.models:
                m_inner = getattr(m, "model", m)
                if "XGB" in m_inner.__class__.__name__ or "LGBM" in m_inner.__class__.__name__:
                    return m_inner
            return getattr(model.models[0], "model", model.models[0])
        if hasattr(model, "model"):
            return model.model
        return model

    def _init_explainer(self, background_data: pd.DataFrame | np.ndarray | None = None) -> None:
        try:
            logger.info(f"Initializing SHAP TreeExplainer for model: {self.raw_model.__class__.__name__}")
            if background_data is not None and len(background_data) > 0:
                bg = background_data
                if len(bg) > self.max_background_samples:
                    bg = shap.sample(bg, self.max_background_samples, random_state=42)
                self.explainer = shap.TreeExplainer(self.raw_model, data=bg)
            else:
                self.explainer = shap.TreeExplainer(self.raw_model)
            logger.info("SHAP TreeExplainer initialized successfully.")
        except Exception as e:
            logger.warning(f"TreeExplainer failed with {e}; falling back to general shap.Explainer.")
            try:
                bg = background_data if background_data is not None else np.zeros((10, len(self.feature_names)))
                if len(bg) > self.max_background_samples:
                    bg = shap.sample(bg, self.max_background_samples, random_state=42)
                self.explainer = shap.Explainer(self.raw_model, bg)
            except Exception as ex:
                logger.error(f"Fallback Explainer also failed: {ex}")
                self.explainer = None

    def explain_instance(
        self,
        X_single: pd.DataFrame | np.ndarray,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Computes local SHAP feature contributions for a single transaction.
        Returns top_k most influential features with directional impact.
        """
        if self.explainer is None:
            return self._fallback_feature_importance(X_single, top_k)

        try:
            shap_vals = self.explainer(X_single)
            values = shap_vals.values
            if values.ndim == 3:
                # Shape: [1, n_features, 2] (class 0 vs class 1)
                class_1_contribs = values[0, :, 1]
            elif values.ndim == 2:
                class_1_contribs = values[0, :]
            else:
                class_1_contribs = values

            # Pair with feature names and sort by absolute contribution
            contributions = []
            for name, val in zip(self.feature_names, class_1_contribs):
                contributions.append({
                    "feature": str(name),
                    "contribution": round(float(val), 4),
                    "impact": "INCREASES_FRAUD_RISK" if val > 0 else "DECREASES_FRAUD_RISK",
                })

            contributions.sort(key=lambda x: abs(x["contribution"]), reverse=True)
            return contributions[:top_k]

        except Exception as e:
            logger.warning(f"SHAP local explanation failed: {e}. Using feature heuristic fallback.")
            return self._fallback_feature_importance(X_single, top_k)

    def _fallback_feature_importance(
        self, X_single: pd.DataFrame | np.ndarray, top_k: int = 5
    ) -> list[dict[str, Any]]:
        """Heuristic fallback when SHAP is unavailable."""
        arr = np.asarray(X_single).flatten()
        contribs = []
        for i, val in enumerate(arr[: len(self.feature_names)]):
            feat_name = self.feature_names[i]
            contrib = float(val * 0.1)
            contribs.append({
                "feature": feat_name,
                "contribution": round(contrib, 4),
                "impact": "INCREASES_FRAUD_RISK" if contrib > 0 else "DECREASES_FRAUD_RISK",
            })
        contribs.sort(key=lambda x: abs(x["contribution"]), reverse=True)
        return contribs[:top_k]

    def generate_global_importance_plot(
        self,
        X_sample: pd.DataFrame | np.ndarray,
        output_path: str | Path = "artifacts/figures/shap_summary.png",
        max_display: int = 15,
    ) -> Path:
        """Generates and saves SHAP summary bar plot."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        if self.explainer is None:
            logger.warning("Explainer is None; skipping SHAP summary plot.")
            return out

        try:
            shap_values = self.explainer(X_sample)
            vals = shap_values.values
            if vals.ndim == 3:
                vals = vals[:, :, 1]

            mean_abs_shap = np.mean(np.abs(vals), axis=0)
            sorted_idx = np.argsort(mean_abs_shap)[::-1][:max_display]

            top_features = [self.feature_names[i] for i in sorted_idx]
            top_importance = mean_abs_shap[sorted_idx]

            plt.figure(figsize=(9, 6))
            y_pos = np.arange(len(top_features))[::-1]
            plt.barh(y_pos, top_importance, color="#2563eb", align="center")
            plt.yticks(y_pos, top_features, fontsize=10)
            plt.xlabel("Mean |SHAP Value| (Impact on Model Fraud Prediction)", fontsize=11)
            plt.title("SHAP Global Feature Importance (Top Features)", fontsize=13, fontweight="bold")
            plt.grid(axis="x", alpha=0.3)
            plt.tight_layout()
            plt.savefig(out, dpi=150)
            plt.close()
            logger.info(f"Saved SHAP summary plot to {out}")
        except Exception as e:
            logger.error(f"Failed to generate SHAP summary plot: {e}")

        return out
