"""
Script to register champion model to MLflow Model Registry.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from shieldml.tracking.mlflow_manager import MLflowTracker

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("RegisterModel")


def main():
    metadata_path = Path("artifacts/models/champion_metadata.json")
    if not metadata_path.exists():
        logger.error("Metadata not found. Train model first with scripts/train.py.")
        return

    with open(metadata_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    tracker = MLflowTracker()
    run_id = tracker.log_model_run(
        run_name="Explicit_Champion_Registration",
        params={"model_type": meta.get("model_type"), "version": meta.get("model_version")},
        metrics=meta.get("test_metrics", {}),
    )

    reg_name = tracker.register_champion(
        run_id=run_id,
        model_artifact_path="artifacts/models/champion_model.joblib",
        threshold=float(meta.get("optimal_threshold", 0.40)),
        metrics=meta.get("test_metrics", {}),
    )
    logger.info(f"Model successfully registered as '{reg_name}'.")


if __name__ == "__main__":
    main()
