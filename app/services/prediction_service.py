"""
Prediction service for ShieldML.
Orchestrates pre-processing, calibrated fraud probability prediction,
threshold-based binary decision, and SHAP feature attribution.
"""

from __future__ import annotations

import logging
import time
from typing import Any
from app.api.schemas import PredictionResponse, TopFeature, TransactionRequest
from app.services.model_loader import model_loader
from app.services.monitoring_service import monitoring_service
from app.utils.errors import ModelNotReadyException

logger = logging.getLogger(__name__)


class PredictionService:
    def predict(self, request: TransactionRequest) -> PredictionResponse:
        start_time = time.perf_counter()

        if not model_loader.is_ready or model_loader.model is None or model_loader.pipeline is None:
            monitoring_service.record_failure()
            raise ModelNotReadyException(
                "ShieldML champion model is not loaded. Run 'python scripts/train.py' to generate artifacts."
            )

        try:
            # 1. Transform single transaction via fitted pipeline
            txn_dict = request.model_dump()
            explain_requested = txn_dict.pop("explain", True)
            X_scaled = model_loader.pipeline.transform_single(txn_dict)

            # 2. Predict calibrated fraud probability
            if hasattr(model_loader.model, "predict_proba"):
                probs = model_loader.model.predict_proba(X_scaled)
                fraud_prob = float(probs[0, 1])
            else:
                raw_pred = model_loader.model.predict(X_scaled)
                fraud_prob = float(raw_pred[0])

            # 3. Classify with optimal threshold
            threshold = float(model_loader.metadata.get("optimal_threshold", 0.40))
            is_fraud = fraud_prob >= threshold
            prediction_label = "FRAUD" if is_fraud else "LEGITIMATE"

            # 4. Compute local SHAP feature explanations if requested
            top_features: list[TopFeature] = []
            if explain_requested and model_loader.explainer is not None:
                contribs = model_loader.explainer.explain_instance(X_scaled, top_k=5)
                top_features = [
                    TopFeature(
                        feature=c["feature"],
                        contribution=c["contribution"],
                        impact=c["impact"],
                    )
                    for c in contribs
                ]

            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)

            # Record monitoring metrics
            monitoring_service.record_prediction(latency_ms=elapsed_ms, is_fraud=is_fraud)

            model_name = str(model_loader.metadata.get("model_name", "ShieldMLChampion"))
            model_ver = str(model_loader.metadata.get("model_version", "1.0.0"))

            return PredictionResponse(
                transaction_id=request.transaction_id,
                fraud_probability=round(fraud_prob, 4),
                threshold=round(threshold, 4),
                prediction=prediction_label,
                model_name=model_name,
                model_version=model_ver,
                top_features=top_features,
                latency_ms=elapsed_ms,
            )

        except Exception as e:
            monitoring_service.record_failure()
            logger.error(f"Inference error for txn {request.transaction_id}: {e}", exc_info=True)
            raise


prediction_service = PredictionService()
