"""
Application configuration for ShieldML FastAPI prediction service.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
import yaml
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "ShieldML Real-Time Fraud Detection API"
    app_version: str = "1.0.0"
    environment: str = os.getenv("ENVIRONMENT", "production")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("PORT", os.getenv("API_PORT", "8000")))

    # Model artifact paths
    model_artifact_path: str = os.getenv(
        "MODEL_ARTIFACT_PATH", "artifacts/models/champion_model.joblib"
    )
    pipeline_artifact_path: str = os.getenv(
        "PIPELINE_ARTIFACT_PATH", "artifacts/models/feature_pipeline.joblib"
    )
    metadata_path: str = os.getenv(
        "METADATA_PATH", "artifacts/models/champion_metadata.json"
    )

    # Explainability & Performance
    enable_shap_by_default: bool = os.getenv("ENABLE_SHAP", "true").lower() == "true"
    max_request_size_bytes: int = int(os.getenv("MAX_REQUEST_SIZE_BYTES", "1048576"))
    cors_origins: str = os.getenv("CORS_ORIGINS", "*")
    latency_target_ms: float = 50.0

    class Config:
        env_file = ".env"
        extra = "allow"


def load_yaml_config(config_path: str = "configs/config.yaml") -> dict[str, Any]:
    p = Path(config_path)
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


settings = Settings()
yaml_config = load_yaml_config()
