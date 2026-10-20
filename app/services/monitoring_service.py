"""
Monitoring service for ShieldML.
Maintains bounded, thread-safe in-memory metrics on latency, volume,
and predicted fraud rates.
"""

from __future__ import annotations

import collections
import threading
import time
from typing import Any
import numpy as np


class MonitoringService:
    """Thread-safe sliding-window monitoring service."""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self._lock = threading.Lock()
        self.total_predictions: int = 0
        self.successful_predictions: int = 0
        self.failed_predictions: int = 0
        self._latencies: collections.deque[float] = collections.deque(maxlen=window_size)
        self._decisions: collections.deque[int] = collections.deque(maxlen=window_size)

    def record_prediction(self, latency_ms: float, is_fraud: bool) -> None:
        with self._lock:
            self.total_predictions += 1
            self.successful_predictions += 1
            self._latencies.append(latency_ms)
            self._decisions.append(1 if is_fraud else 0)

    def record_failure(self) -> None:
        with self._lock:
            self.total_predictions += 1
            self.failed_predictions += 1

    def get_metrics(self) -> dict[str, Any]:
        with self._lock:
            latencies = list(self._latencies)
            decisions = list(self._decisions)

            avg_lat = round(float(np.mean(latencies)), 2) if latencies else 0.0
            p95_lat = round(float(np.percentile(latencies, 95)), 2) if latencies else 0.0
            fraud_rate = (
                round(float(sum(decisions) / max(1, len(decisions))), 4)
                if decisions
                else 0.0
            )

            return {
                "total_predictions": self.total_predictions,
                "successful_predictions": self.successful_predictions,
                "failed_predictions": self.failed_predictions,
                "average_latency_ms": avg_lat,
                "p95_latency_ms": p95_lat,
                "recent_predicted_fraud_rate": fraud_rate,
                "window_size": len(decisions),
                "disclaimer": (
                    "Recent fraud rate reflects real-time model inferences across the sliding window, "
                    "NOT confirmed chargeback or investigation ground-truth."
                ),
            }


monitoring_service = MonitoringService()
