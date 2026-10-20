"""
Standalone model evaluation script for ShieldML.
Evaluates champion model on arbitrary test datasets.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import joblib
import pandas as pd

from shieldml.data.ingestion import DataIngestion
from shieldml.evaluation.metrics import calculate_metrics
from shieldml.features.transformer import FeaturePipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("EvaluateScript")


def evaluate(data_path: str = "data/raw/creditcard.csv"):
    model_path = Path("artifacts/models/champion_model.joblib")
    pipeline_path = Path("artifacts/models/feature_pipeline.joblib")
    metadata_path = Path("artifacts/models/champion_metadata.json")

    if not (model_path.exists() and pipeline_path.exists() and metadata_path.exists()):
        logger.error("Required model artifacts missing. Run scripts/train.py first.")
        return

    model = joblib.load(model_path)
    pipeline = FeaturePipeline.load(str(pipeline_path))
    with open(metadata_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    threshold = float(meta.get("optimal_threshold", 0.40))
    ingestion = DataIngestion(raw_data_path=data_path)
    df, _ = ingestion.load_data()

    X_scaled = pipeline.transform(df)
    y_true = df["Class"].to_numpy()
    probs = model.predict_proba(X_scaled)[:, 1]

    metrics = calculate_metrics(y_true, probs, threshold=threshold, model_name=meta.get("model_name", "Champion"))
    print("\n" + "=" * 60)
    print(" SHIELDML EVALUATION METRICS")
    print("=" * 60)
    print(f"ROC-AUC:          {metrics.roc_auc:.4f}")
    print(f"PR-AUC:           {metrics.pr_auc:.4f}")
    print(f"F2-Score:         {metrics.f2:.4f}")
    print(f"Precision:        {metrics.precision:.4f}")
    print(f"Recall:           {metrics.recall:.4f}")
    print(f"False Negatives:  {metrics.false_negatives}")
    print(f"False Positives:  {metrics.false_positives}")
    print(f"Brier Score:      {metrics.brier_score:.5f}")
    print("=" * 60)

    Path("artifacts/reports/evaluation_run.json").write_text(
        json.dumps(metrics.to_dict(), indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    evaluate()
