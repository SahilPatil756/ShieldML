"""
Unit tests for threshold optimization and metrics calculation.
"""

from __future__ import annotations

import numpy as np
import pytest
from shieldml.evaluation.metrics import calculate_metrics
from shieldml.evaluation.threshold import ThresholdOptimizer


def test_metrics_calculation():
    y_true = np.array([0, 0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.3, 0.8, 0.9])

    metrics = calculate_metrics(y_true, y_prob, threshold=0.5)

    assert metrics.roc_auc == 1.0
    assert metrics.true_positives == 2
    assert metrics.true_negatives == 3
    assert metrics.false_positives == 0
    assert metrics.false_negatives == 0
    assert metrics.recall == 1.0
    assert metrics.precision == 1.0
    assert metrics.f2 == 1.0


def test_threshold_optimizer():
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_prob = np.array([0.05, 0.10, 0.20, 0.35, 0.45, 0.60, 0.75, 0.90])

    optimizer = ThresholdOptimizer(beta=2.0, num_steps=20)
    res = optimizer.optimize(y_true, y_prob)

    assert 0.05 <= res.optimal_threshold <= 0.95
    assert res.max_f2_score > 0.8
    assert res.metric_name == "f2_score"
    assert len(res.threshold_curve) == 20
