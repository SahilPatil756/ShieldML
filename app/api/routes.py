"""
FastAPI route definitions for ShieldML.
"""

from __future__ import annotations

from fastapi import APIRouter, status
from app.api.schemas import (
    MetricsResponse,
    ModelInfoResponse,
    PredictionResponse,
    TransactionRequest,
)
from app.core.config import settings
from app.services.model_loader import model_loader
from app.services.monitoring_service import monitoring_service
from app.services.prediction_service import prediction_service

router = APIRouter()


@router.get("/", status_code=status.HTTP_200_OK)
def root():
    return {
        "service": settings.app_name,
        "version": settings.app_version,
        "status": "OPERATIONAL",
        "model_loaded": model_loader.is_ready,
    }


@router.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    return {
        "status": "HEALTHY",
        "model_ready": model_loader.is_ready,
        "model_name": model_loader.metadata.get("model_name", "None"),
    }


@router.get("/model-info", response_model=ModelInfoResponse)
def get_model_info():
    return ModelInfoResponse(
        model_name=model_loader.metadata.get("model_name", "ShieldMLChampion"),
        model_version=model_loader.metadata.get("model_version", "1.0.0"),
        optimal_threshold=float(model_loader.metadata.get("optimal_threshold", 0.40)),
        optimization_metric=model_loader.metadata.get("optimization_metric", "f2_score"),
        features=model_loader.pipeline.feature_names if model_loader.pipeline else [],
        status="READY" if model_loader.is_ready else "UNINITIALIZED",
        metrics=model_loader.metadata.get("test_metrics", {}),
    )


@router.post("/predict", response_model=PredictionResponse)
def predict_fraud(request: TransactionRequest):
    return prediction_service.predict(request)


@router.get("/metrics", response_model=MetricsResponse)
def get_monitoring_metrics():
    return MetricsResponse(**monitoring_service.get_metrics())
