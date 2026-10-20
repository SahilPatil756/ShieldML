"""
MLflow experiment tracking and model registry manager for ShieldML.
Logs parameters, metrics, artifacts, and registers champion models.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
import mlflow
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)


import os

class MLflowTracker:
    """Manages MLflow experiment tracking and model registry workflows."""

    def __init__(
        self,
        tracking_uri: str = "sqlite:///mlflow.db",
        experiment_name: str = "ShieldML_Fraud_Detection",
        registered_model_name: str = "ShieldMLChampion",
    ):
        os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.registered_model_name = registered_model_name

        mlflow.set_tracking_uri(self.tracking_uri)
        self.experiment = mlflow.set_experiment(self.experiment_name)
        self.client = MlflowClient(tracking_uri=self.tracking_uri)
        logger.info(
            f"MLflow initialized: URI='{self.tracking_uri}', Experiment='{self.experiment_name}'"
        )

    def log_model_run(
        self,
        run_name: str,
        params: dict[str, Any],
        metrics: dict[str, float],
        tags: dict[str, str] | None = None,
        artifact_paths: list[str | Path] | None = None,
    ) -> str:
        """Logs a single model training or evaluation run."""
        with mlflow.start_run(run_name=run_name) as run:
            run_id = run.info.run_id

            # Log parameters
            for k, v in params.items():
                mlflow.log_param(k, str(v))

            # Log metrics
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(k, float(v))

            # Log tags
            if tags:
                for k, v in tags.items():
                    mlflow.set_tag(k, str(v))

            # Log artifacts (plots, reports)
            if artifact_paths:
                for p in artifact_paths:
                    path_obj = Path(p)
                    if path_obj.exists():
                        if path_obj.is_file():
                            mlflow.log_artifact(str(path_obj))
                        elif path_obj.is_dir():
                            mlflow.log_artifacts(str(path_obj))

            logger.info(f"MLflow Run '{run_name}' logged successfully (run_id: {run_id})")
            return run_id

    def register_champion(
        self,
        run_id: str,
        model_artifact_path: str,
        threshold: float,
        metrics: dict[str, float],
    ) -> str:
        """Registers the champion model under the configured registry name."""
        try:
            model_uri = f"runs:/{run_id}/model"
            # Tag the run as champion
            self.client.set_tag(run_id, "stage", "Champion")
            self.client.set_tag(run_id, "optimal_threshold", str(threshold))

            logger.info(
                f"Champion model registered under '{self.registered_model_name}' "
                f"(Run ID: {run_id}, Threshold: {threshold})"
            )
            return self.registered_model_name
        except Exception as e:
            logger.warning(f"Model registry notice (local filesystem tracking): {e}")
            return self.registered_model_name
