"""
Probability calibration module for ShieldML.
Uses CalibratedClassifierCV with Platt scaling (sigmoid) or isotonic regression
to align predicted fraud probabilities with observed empirical frequencies.
"""

from __future__ import annotations

import logging
from typing import Any
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss

logger = logging.getLogger(__name__)


class ModelCalibrator:
    """Calibrates classifier probabilities on validation data."""

    def __init__(self, method: str = "sigmoid"):
        self.method = method
        self.calibrator: CalibratedClassifierCV | None = None
        self.brier_before: float | None = None
        self.brier_after: float | None = None

    def calibrate(
        self,
        base_estimator: Any,
        X_val: Any,
        y_val: Any,
    ) -> CalibratedClassifierCV:
        """
        Fits calibration curve on validation data (cv='prefit' prevents data leakage).
        Never touches test data.
        """
        raw_model = getattr(base_estimator, "model", base_estimator)

        # Compute Brier score before calibration
        if hasattr(raw_model, "predict_proba"):
            raw_probs = raw_model.predict_proba(X_val)[:, 1]
            self.brier_before = float(brier_score_loss(y_val, raw_probs))

        logger.info(f"Calibrating probabilities using method='{self.method}' (prefit)...")
        self.calibrator = CalibratedClassifierCV(
            estimator=raw_model,
            method=self.method,
            cv="prefit",
        )
        self.calibrator.fit(X_val, y_val)

        # Compute Brier score after calibration
        cal_probs = self.calibrator.predict_proba(X_val)[:, 1]
        self.brier_after = float(brier_score_loss(y_val, cal_probs))
        logger.info(
            f"Probability Calibration: Brier score {self.brier_before:.5f} -> {self.brier_after:.5f} "
            f"({'improved' if self.brier_after < (self.brier_before or 1) else 'comparable'})"
        )

        return self.calibrator
