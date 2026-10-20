# ShieldML Production Deployment Guide

## 1. Containerization
ShieldML is distributed as a multi-stage, hardened Docker container running Python 3.10/3.11 with unprivileged user permissions.

### Build and Run with Docker
```bash
# Build the Docker image
docker build -t shieldml:latest .

# Run the container
docker run -d -p 8000:8000 \
  -e ENVIRONMENT=production \
  -e API_PORT=8000 \
  -e MODEL_ARTIFACT_PATH=artifacts/models/champion_model.joblib \
  --name shieldml-api shieldml:latest
```

## 2. Docker Compose Stack
The provided `docker-compose.yml` orchestrates:
1. **shieldml-api**: The FastAPI high-performance inference engine.
2. **mlflow-tracking**: The experiment tracking and model registry server.
3. **prometheus**: Metric scraping from `/metrics`.

```bash
docker-compose up -d
```

## 3. Kubernetes Production Architecture
- **Deployment**: 3+ replicas with Horizontal Pod Autoscaler (HPA) targeting 70% CPU utilization.
- **Resource Limits**:
  - Request: `500m CPU`, `1Gi Memory`
  - Limit: `2000m CPU`, `2Gi Memory`
- **Probes**:
  - Liveness Probe: `GET /health` (delay: 10s, period: 10s)
  - Readiness Probe: `GET /health` (delay: 5s, period: 5s)

## 4. Retraining Cadence & Model Drift
1. **Trigger Conditions**:
   - Weekly scheduled pipeline execution.
   - Population Stability Index (PSI) $> 0.20$ on top SHAP features (`V14`, `V12`, `V4`).
   - Ground-truth chargeback drift $> 20\%$ variance from training prior.
2. **Shadow Deployment / Blue-Green**:
   - New models are deployed in "shadow mode" for 72 hours alongside the active champion.
   - Predictions are logged asynchronously; F2 and latency are compared before promoting to champion.
