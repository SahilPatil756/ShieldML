"""
Random Forest model for ShieldML with optional GridSearchCV hyperparameter tuning.
"""

from __future__ import annotations

import logging
from typing import Any
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

logger = logging.getLogger(__name__)


class RandomForestModel:
    """Random Forest classifier with balanced class weighting and tuning support."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int | None = 12,
        min_samples_split: int = 5,
        min_samples_leaf: int = 2,
        class_weight: str | None = "balanced",
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        self.model_name = "RandomForestClassifier"
        self.params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "class_weight": class_weight,
            "random_state": random_state,
            "n_jobs": n_jobs,
        }
        self.model = RandomForestClassifier(**self.params)

    def fit(self, X: Any, y: Any) -> RandomForestModel:
        logger.info(f"Training {self.model_name}...")
        self.model.fit(X, y)
        return self

    def tune_grid_search(
        self,
        X: Any,
        y: Any,
        param_grid: dict[str, list[Any]] | None = None,
        cv: int = 3,
        scoring: str = "average_precision",
    ) -> dict[str, Any]:
        """Tuning using GridSearchCV optimizing for PR-AUC (Average Precision)."""
        logger.info("Tuning Random Forest hyperparameters with GridSearchCV...")
        grid = param_grid or {
            "n_estimators": [50, 100],
            "max_depth": [8, 12],
            "min_samples_split": [2, 5],
        }
        base_rf = RandomForestClassifier(
            class_weight="balanced",
            random_state=self.params["random_state"],
            n_jobs=-1,
        )
        search = GridSearchCV(
            estimator=base_rf,
            param_grid=grid,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            verbose=0,
        )
        search.fit(X, y)
        logger.info(f"GridSearchCV best parameters: {search.best_params_}, score={search.best_score_:.4f}")
        self.model = search.best_estimator_
        self.params.update(search.best_params_)
        return search.best_params_

    def predict_proba(self, X: Any) -> np.ndarray:
        return self.model.predict_proba(X)

    def predict(self, X: Any, threshold: float = 0.5) -> np.ndarray:
        probs = self.predict_proba(X)[:, 1]
        return (probs >= threshold).astype(int)
