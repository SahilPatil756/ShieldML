"""
Baseline model for ShieldML: Logistic Regression with balanced class weighting.
Establishes benchmark reference metrics.
"""

from __future__ import annotations

import logging
from typing import Any
import numpy as np
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)


class BaselineModel:
    """Logistic Regression baseline classifier."""

    def __init__(
        self,
        C: float = 1.0,
        penalty: str = "l2",
        solver: str = "lbfgs",
        max_iter: int = 1000,
        class_weight: str | None = "balanced",
        random_state: int = 42,
    ):
        self.model_name = "LogisticRegressionBaseline"
        self.model = LogisticRegression(
            C=C,
            penalty=penalty,
            solver=solver,
            max_iter=max_iter,
            class_weight=class_weight,
            random_state=random_state,
        )

    def fit(self, X: Any, y: Any) -> BaselineModel:
        logger.info(f"Training {self.model_name}...")
        self.model.fit(X, y)
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        return self.model.predict_proba(X)

    def predict(self, X: Any, threshold: float = 0.5) -> np.ndarray:
        probs = self.predict_proba(X)[:, 1]
        return (probs >= threshold).astype(int)
