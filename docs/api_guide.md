# ShieldML API Reference Guide

The ShieldML prediction service exposes high-throughput, low-latency REST endpoints for real-time transaction scoring.

## Base URL
```
http://localhost:8000 (or http://localhost:8080)
```

## Endpoints

### 1. Root & Status
- **Method**: `GET /`
- **Description**: Returns basic application and model status.
- **Example Response**:
```json
{
  "service": "ShieldML Real-Time Fraud Detection API",
  "version": "1.0.0",
  "status": "OPERATIONAL",
  "model_loaded": true
}
```

### 2. Health Check
- **Method**: `GET /health`
- **Description**: Kubernetes / AWS ALB readiness and liveness probe.
- **Example Response**:
```json
{
  "status": "HEALTHY",
  "model_ready": true,
  "model_name": "ShieldMLChampion"
}
```

### 3. Model Information & Metadata
- **Method**: `GET /model-info`
- **Description**: Returns champion model architecture, optimal threshold, feature list, and test benchmark metrics.

### 4. Real-Time Fraud Prediction
- **Method**: `POST /predict`
- **Headers**: `Content-Type: application/json`
- **Request Body**:
```json
{
  "transaction_id": "TXN_991823",
  "amount": 1450.00,
  "time": 86400.0,
  "V4": 3.85,
  "V10": -2.90,
  "V12": -3.80,
  "V14": -5.20,
  "V17": -2.60,
  "explain": true
}
```
- **Response**:
```json
{
  "transaction_id": "TXN_991823",
  "fraud_probability": 0.9842,
  "threshold": 0.4045,
  "prediction": "FRAUD",
  "model_name": "ShieldMLChampion",
  "model_version": "1.0.0",
  "top_features": [
    {
      "feature": "V14",
      "contribution": 8.3506,
      "impact": "INCREASES_FRAUD_RISK"
    },
    {
      "feature": "V12",
      "contribution": 5.1498,
      "impact": "INCREASES_FRAUD_RISK"
    },
    {
      "feature": "V4",
      "contribution": 4.7902,
      "impact": "INCREASES_FRAUD_RISK"
    }
  ],
  "latency_ms": 11.45
}
```

### 5. In-Memory Monitoring Metrics
- **Method**: `GET /metrics`
- **Description**: Returns sliding-window counters for request throughput, average/p95 latency, and recent predicted fraud rate.
