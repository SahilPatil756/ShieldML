"""
LightGBM classifier for ShieldML with Optuna hyperparameter optimization.
"""

from __future__ import annotations

import logging
from typing import Any
import numpy as np
import optuna
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


class LightGBMModel:
    """LightGBM classifier with class imbalance weighting and Optuna tuning."""

    def __init__(
        self,
        n_estimators: int = 150,
        num_leaves: int = 31,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        scale_pos_weight: float = 10.0,
        random_state: int = 42,
        n_jobs: int = -1,
        verbose: int = -1,
    ):
        self.model_name = "LightGBMClassifier"
        self.params = {
            "n_estimators": n_estimators,
            "num_leaves": num_leaves,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "scale_pos_weight": scale_pos_weight,
            "random_state": random_state,
            "n_jobs": n_jobs,
            "verbose": verbose,
        }
        self.model = LGBMClassifier(**self.params)

    def fit(self, X: Any, y: Any) -> LightGBMModel:
        logger.info(f"Training {self.model_name}...")
        self.model.fit(X, y)
        return self

    def tune_optuna(
        self,
        X_train: Any,
        y_train: Any,
        X_val: Any,
        y_val: Any,
        n_trials: int = 10,
        timeout: int = 120,
    ) -> dict[str, Any]:
        """Tunes hyperparameters with Optuna targeting validation PR-AUC."""
        logger.info(f"Running Optuna tuning for LightGBM ({n_trials} trials)...")

        def objective(trial: optuna.Trial) -> float:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 80, 200, step=20),
                "num_leaves": trial.suggest_int("num_leaves", 15, 63),
                "max_depth": trial.suggest_int("max_depth", 4, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "scale_pos_weight": trial.suggest_float("scale_pos_weight", 2.0, 15.0),
                "random_state": 42,
                "n_jobs": -1,
                "verbose": -1,
            }
            clf = LGBMClassifier(**params)
            clf.fit(X_train, y_train)
            val_probs = clf.predict_proba(X_val)[:, 1]
            return float(average_precision_score(y_val, val_probs))

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, timeout=timeout)

        best_params = study.best_params
        best_params["random_state"] = 42
        best_params["n_jobs"] = -1
        best_params["verbose"] = -1
        logger.info(f"LightGBM best Optuna params: {best_params}, best PR-AUC={study.best_value:.4f}")

        self.params.update(best_params)
        self.model = LGBMClassifier(**self.params)
        self.model.fit(X_train, y_train)
        return best_params

    def predict_proba(self, X: Any) -> np.ndarray:
        return self.model.predict_proba(X)

    def predict(self, X: Any, threshold: float = 0.5) -> np.ndarray:
        probs = self.predict_proba(X)[:, 1]
        return (probs >= threshold).astype(int)
