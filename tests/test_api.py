"""
API integration and unit tests for ShieldML FastAPI endpoints.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.services.model_loader import model_loader


@pytest.fixture(scope="module")
def client(sample_dataset, fitted_pipeline):
    # Setup mock or light champion model in loader for test environment
    from shieldml.models.random_forest import RandomForestModel
    X = fitted_pipeline.transform(sample_dataset)
    y = sample_dataset["Class"]
    rf = RandomForestModel(n_estimators=5, max_depth=3).fit(X, y)

    model_loader.model = rf
    model_loader.pipeline = fitted_pipeline
    model_loader.metadata = {
        "model_name": "TestChampionModel",
        "model_version": "1.0.0",
        "optimal_threshold": 0.40,
        "optimization_metric": "f2_score",
        "test_metrics": {"roc_auc": 0.95, "f2": 0.88},
    }
    from shieldml.explainability.shap_explainer import FraudExplainer
    model_loader.explainer = FraudExplainer(
        model=rf,
        feature_names=fitted_pipeline.feature_names,
    )
    model_loader.is_ready = True

    return TestClient(app)


def test_root_endpoint(client: TestClient):
    resp = client.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "OPERATIONAL"
    assert data["model_loaded"] is True


def test_health_endpoint(client: TestClient):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "HEALTHY"
    assert data["model_ready"] is True


def test_model_info_endpoint(client: TestClient):
    resp = client.get("/model-info")
    assert resp.status_code == 200
    data = resp.json()
    assert data["model_name"] == "TestChampionModel"
    assert data["optimal_threshold"] == 0.40
    assert len(data["features"]) > 0


def test_predict_endpoint(client: TestClient, sample_transaction_payload: dict):
    resp = client.post("/predict", json=sample_transaction_payload)
    assert resp.status_code == 200
    data = resp.json()

    assert data["transaction_id"] == sample_transaction_payload["transaction_id"]
    assert 0.0 <= data["fraud_probability"] <= 1.0
    assert data["prediction"] in {"FRAUD", "LEGITIMATE"}
    assert data["threshold"] == 0.40
    assert data["latency_ms"] > 0
    assert len(data["top_features"]) > 0


def test_predict_invalid_payload(client: TestClient):
    # Negative amount should be rejected by Pydantic validation
    invalid_payload = {
        "transaction_id": "INVALID_001",
        "amount": -50.0,
        "time": 100.0,
    }
    resp = client.post("/predict", json=invalid_payload)
    assert resp.status_code == 422  # Unprocessable Entity


def test_metrics_endpoint(client: TestClient, sample_transaction_payload: dict):
    # Execute a prediction to record metrics
    client.post("/predict", json=sample_transaction_payload)

    resp = client.get("/metrics")
    assert resp.status_code == 200
    data = resp.json()

    assert data["total_predictions"] > 0
    assert data["successful_predictions"] > 0
    assert data["average_latency_ms"] >= 0.0
    assert "disclaimer" in data
