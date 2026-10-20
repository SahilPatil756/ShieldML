"""
Local SHAP explanation generator for ShieldML.
Produces waterfall plots for three representative transactions:
1. Clear Legitimate transaction
2. Borderline / Ambiguous transaction
3. High-Confidence Fraud transaction
"""

from __future__ import annotations

import logging
from pathlib import Path
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from shieldml.data.ingestion import DataIngestion
from shieldml.features.transformer import FeaturePipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("ExplainScript")


def main():
    logger.info("Generating local SHAP waterfall explanations...")
    figures_dir = Path("artifacts/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    model_path = Path("artifacts/models/champion_model.joblib")
    pipeline_path = Path("artifacts/models/feature_pipeline.joblib")

    if not (model_path.exists() and pipeline_path.exists()):
        logger.error("Champion model or pipeline artifacts not found. Run scripts/train.py first.")
        return

    model = joblib.load(model_path)
    pipeline = FeaturePipeline.load(str(pipeline_path))

    # Unwrap underlying tree model if calibrated
    raw_model = model
    if hasattr(model, "calibrated_classifiers_") and model.calibrated_classifiers_:
        raw_model = model.calibrated_classifiers_[0].estimator
    elif hasattr(model, "estimator"):
        raw_model = model.estimator

    # Ingest data to select 3 representative cases
    ingestion = DataIngestion()
    df, _ = ingestion.load_data()
    X_scaled = pipeline.transform(df)
    probs = model.predict_proba(X_scaled)[:, 1]

    # 1. Clear Legitimate (low probability < 0.05)
    legit_idx = int(np.argmin(probs))
    # 2. High Fraud (high probability > 0.85)
    fraud_idx = int(np.argmax(probs))
    # 3. Borderline (closest to 0.40 threshold)
    borderline_idx = int(np.argmin(np.abs(probs - 0.40)))

    samples = [
        ("legitimate", legit_idx, "Clear Legitimate Transaction"),
        ("borderline", borderline_idx, "Borderline Ambiguous Transaction"),
        ("fraud", fraud_idx, "High-Confidence Fraudulent Transaction"),
    ]

    try:
        # Select appropriate explainer based on model family
        if "Logistic" in raw_model.__class__.__name__:
            explainer = shap.LinearExplainer(raw_model, X_scaled.iloc[:100])
        else:
            explainer = shap.TreeExplainer(raw_model)

        for label, idx, title in samples:
            x_row = X_scaled.iloc[[idx]]
            prob = probs[idx]
            shap_obj = explainer(x_row)

            # Extract 1D explanation values for class 1
            if shap_obj.values.ndim == 3:
                vals = shap_obj.values[0, :, 1]
                base_val = shap_obj.base_values[0, 1] if hasattr(shap_obj.base_values, "__len__") else shap_obj.base_values
            else:
                vals = shap_obj.values[0, :]
                base_val = shap_obj.base_values[0]

            plt.figure(figsize=(10, 6))
            # Sort by top impact features
            sort_order = np.argsort(np.abs(vals))[-10:]
            top_feats = [pipeline.feature_names[i] for i in sort_order]
            top_vals = vals[sort_order]
            colors = ["#ef4444" if v > 0 else "#3b82f6" for v in top_vals]

            plt.barh(top_feats, top_vals, color=colors)
            plt.axvline(0, color="black", lw=1)
            plt.xlabel("SHAP Contribution to Fraud Risk (Log-odds margin)", fontsize=11)
            plt.title(f"SHAP Waterfall: {title}\nPredicted Fraud Probability: {prob:.2%}", fontsize=12, fontweight="bold")
            plt.grid(axis="x", alpha=0.3)
            plt.tight_layout()

            out_file = figures_dir / f"waterfall_{label}.png"
            plt.savefig(out_file, dpi=150)
            plt.close()
            logger.info(f"Saved {label} explanation waterfall to {out_file}")

    except Exception as e:
        logger.error(f"Error computing waterfall explanations: {e}")


if __name__ == "__main__":
    main()
