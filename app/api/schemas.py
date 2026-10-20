"""
Pydantic schemas for ShieldML FastAPI prediction service.
"""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, ConfigDict, Field


class TransactionRequest(BaseModel):
    model_config = ConfigDict(
        extra="allow",
        json_schema_extra={
            "example": {
                "transaction_id": "TXN_884920",
                "amount": 2500.0,
                "time": 86400.0,
                "V4": 4.12,
                "V10": -3.2,
                "V12": -4.05,
                "V14": -5.8,
                "V17": -2.9,
                "explain": True,
            }
        },
    )

    transaction_id: str = Field(..., description="Unique transaction reference ID")
    amount: float = Field(..., ge=0.0, description="Transaction amount in currency units")
    time: float = Field(..., ge=0.0, description="Elapsed seconds or unix timestamp")

    # PCA features V1 to V28
    V1: float = 0.0
    V2: float = 0.0
    V3: float = 0.0
    V4: float = 0.0
    V5: float = 0.0
    V6: float = 0.0
    V7: float = 0.0
    V8: float = 0.0
    V9: float = 0.0
    V10: float = 0.0
    V11: float = 0.0
    V12: float = 0.0
    V13: float = 0.0
    V14: float = 0.0
    V15: float = 0.0
    V16: float = 0.0
    V17: float = 0.0
    V18: float = 0.0
    V19: float = 0.0
    V20: float = 0.0
    V21: float = 0.0
    V22: float = 0.0
    V23: float = 0.0
    V24: float = 0.0
    V25: float = 0.0
    V26: float = 0.0
    V27: float = 0.0
    V28: float = 0.0

    # Optional explainability toggle (disabling SHAP yields sub-10ms latency)
    explain: bool = Field(True, description="Whether to include local SHAP feature explanations")


class TopFeature(BaseModel):
    feature: str
    contribution: float
    impact: str


class PredictionResponse(BaseModel):
    transaction_id: str
    fraud_probability: float
    threshold: float
    prediction: str  # "FRAUD" | "LEGITIMATE"
    model_name: str
    model_version: str
    top_features: list[TopFeature]
    latency_ms: float


class ModelInfoResponse(BaseModel):
    model_name: str
    model_version: str
    optimal_threshold: float
    optimization_metric: str
    features: list[str]
    status: str
    metrics: dict[str, Any]


class MetricsResponse(BaseModel):
    total_predictions: int
    successful_predictions: int
    failed_predictions: int
    average_latency_ms: float
    p95_latency_ms: float
    recent_predicted_fraud_rate: float
    window_size: int
    disclaimer: str
