"""
Threshold optimization module for ShieldML.
Finds the decision threshold that maximizes F2-score on validation data.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any
import numpy as np
from sklearn.metrics import fbeta_score, precision_score, recall_score

logger = logging.getLogger(__name__)


@dataclass
class OptimalThresholdResult:
    optimal_threshold: float
    max_f2_score: float
    precision_at_optimal: float
    recall_at_optimal: float
    metric_name: str
    beta: float
    threshold_curve: list[dict[str, float]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ThresholdOptimizer:
    """Optimizes classification threshold on validation probabilities to maximize F2-score."""

    def __init__(
        self,
        beta: float = 2.0,
        num_steps: int = 100,
        min_threshold: float = 0.05,
        max_threshold: float = 0.95,
    ):
        self.beta = beta
        self.num_steps = num_steps
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold

    def optimize(
        self,
        y_val_true: np.ndarray,
        y_val_prob: np.ndarray,
    ) -> OptimalThresholdResult:
        """
        Sweeps candidate thresholds across validation set.
        Applies deterministic tie-breaking rule: if multiple thresholds achieve
        maximum F2-score, picks the one closest to median fraud probability or 0.4.
        """
        thresholds = np.linspace(self.min_threshold, self.max_threshold, self.num_steps)
        curve_data: list[dict[str, float]] = []

        best_f2 = -1.0
        best_thresholds: list[float] = []
        best_prec = 0.0
        best_rec = 0.0

        for t in thresholds:
            y_pred = (y_val_prob >= t).astype(int)
            p = float(precision_score(y_val_true, y_pred, zero_division=0))
            r = float(recall_score(y_val_true, y_pred, zero_division=0))
            f2 = float(fbeta_score(y_val_true, y_pred, beta=self.beta, zero_division=0))

            curve_data.append({
                "threshold": round(float(t), 4),
                "precision": round(p, 4),
                "recall": round(r, 4),
                "f2": round(f2, 4),
            })

            if f2 > best_f2 + 1e-6:
                best_f2 = f2
                best_thresholds = [float(t)]
                best_prec = p
                best_rec = r
            elif abs(f2 - best_f2) <= 1e-6:
                best_thresholds.append(float(t))

        # Deterministic tie-breaker: select threshold closest to 0.40 (standard risk inflection)
        optimal_t = min(best_thresholds, key=lambda x: abs(x - 0.40))

        logger.info(
            f"Optimal Threshold Selected: {optimal_t:.4f} "
            f"(F2={best_f2:.4f}, Precision={best_prec:.4f}, Recall={best_rec:.4f})"
        )

        return OptimalThresholdResult(
            optimal_threshold=round(optimal_t, 4),
            max_f2_score=round(best_f2, 4),
            precision_at_optimal=round(best_prec, 4),
            recall_at_optimal=round(best_rec, 4),
            metric_name="f2_score",
            beta=self.beta,
            threshold_curve=curve_data,
        )
