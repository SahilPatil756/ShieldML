"""
Model loader singleton for ShieldML.
Loads model artifacts, feature pipelines, and metadata at server startup.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any
import joblib

from shieldml.explainability.shap_explainer import FraudExplainer
from shieldml.features.transformer import FeaturePipeline

logger = logging.getLogger(__name__)


class ModelLoader:
    def __init__(self):
        self.model: Any = None
        self.pipeline: FeaturePipeline | None = None
        self.metadata: dict[str, Any] = {}
        self.explainer: FraudExplainer | None = None
        self.is_ready: bool = False

    def load(
        self,
        model_path: str = "artifacts/models/champion_model.joblib",
        pipeline_path: str = "artifacts/models/feature_pipeline.joblib",
        metadata_path: str = "artifacts/models/champion_metadata.json",
    ) -> bool:
        m_path = Path(model_path)
        p_path = Path(pipeline_path)
        meta_path = Path(metadata_path)

        if not (m_path.exists() and p_path.exists() and meta_path.exists()):
            logger.warning(
                f"Model artifacts not found ({model_path}, {pipeline_path}, {metadata_path}). "
                "API starting in uninitialized state."
            )
            self.is_ready = False
            return False

        try:
            logger.info(f"Loading champion model from {m_path}...")
            self.model = joblib.load(m_path)
            logger.info(f"Loading feature pipeline from {p_path}...")
            self.pipeline = joblib.load(p_path)

            logger.info(f"Loading metadata from {meta_path}...")
            with open(meta_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)

            # Initialize SHAP explainer
            logger.info("Initializing SHAP explainer for prediction serving...")
            self.explainer = FraudExplainer(
                model=self.model,
                feature_names=self.pipeline.feature_names,
            )

            self.is_ready = True
            logger.info("Champion model and pipeline successfully loaded into memory.")
            return True
        except Exception as e:
            logger.error(f"Failed to load model artifacts: {e}", exc_info=True)
            self.is_ready = False
            return False


model_loader = ModelLoader()
