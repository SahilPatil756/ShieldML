"""
Soft-voting ensemble classifier for ShieldML.
Combines Random Forest, XGBoost, and LightGBM using weighted probability averaging.
"""

from __future__ import annotations

import logging
from typing import Any
import numpy as np

logger = logging.getLogger(__name__)


class SoftVotingEnsemble:
    """Combines predictions from multiple calibrated tree models."""

    def __init__(
        self,
        models: list[Any],
        weights: list[float] | None = None,
        model_names: list[str] | None = None,
    ):
        if not models:
            raise ValueError("Must provide at least one model for the ensemble.")

        self.models = models
        self.model_names = model_names or [
            getattr(m, "model_name", m.__class__.__name__) for m in models
        ]

        if weights is None:
            # Equal weighting by default
            self.weights = [1.0 / len(models)] * len(models)
        else:
            if len(weights) != len(models):
                raise ValueError("Length of weights must match length of models.")
            total_weight = sum(weights)
            self.weights = [w / total_weight for w in weights]

        self.model_name = "SoftVotingEnsemble"

    def predict_proba(self, X: Any) -> np.ndarray:
        """Computes weighted probability average across all constituent models."""
        weighted_probs = np.zeros((len(X), 2), dtype=float)

        for model, weight in zip(self.models, self.weights):
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X)
            else:
                raw_preds = model.predict(X)
                probs = np.column_stack([1 - raw_preds, raw_preds])

            weighted_probs += weight * probs

        return weighted_probs

    def predict(self, X: Any, threshold: float = 0.5) -> np.ndarray:
        """Predicts binary classification label using optimized threshold."""
        fraud_probs = self.predict_proba(X)[:, 1]
        return (fraud_probs >= threshold).astype(int)

    def get_component_predictions(self, X: Any) -> dict[str, np.ndarray]:
        """Returns individual model probability predictions for comparison."""
        results = {}
        for name, model in zip(self.model_names, self.models):
            results[name] = model.predict_proba(X)[:, 1]
        results[self.model_name] = self.predict_proba(X)[:, 1]
        return results
