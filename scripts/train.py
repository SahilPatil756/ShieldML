"""
Master model training and evaluation pipeline for ShieldML.
Executes end-to-end training, soft-voting ensemble, calibration,
threshold optimization (F2-score), SHAP explainability, and MLflow tracking.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any
import joblib
import numpy as np
import pandas as pd

from shieldml.data.ingestion import DataIngestion
from shieldml.data.splitting import DataSplitter
from shieldml.evaluation.metrics import calculate_metrics
from shieldml.evaluation.reports import ReportGenerator
from shieldml.evaluation.threshold import ThresholdOptimizer
from shieldml.explainability.shap_explainer import FraudExplainer
from shieldml.features.transformer import FeaturePipeline
from shieldml.models.baseline import BaselineModel
from shieldml.models.calibration import ModelCalibrator
from shieldml.models.ensemble import SoftVotingEnsemble
from shieldml.models.lightgbm_model import LightGBMModel
from shieldml.models.random_forest import RandomForestModel
from shieldml.models.xgboost_model import XGBoostModel
from shieldml.tracking.mlflow_manager import MLflowTracker

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
logger = logging.getLogger("ShieldMLTrainer")


def run_pipeline():
    logger.info("=" * 70)
    logger.info(" STARTING SHIELDML PRODUCTION TRAINING PIPELINE")
    logger.info("=" * 70)

    # 1. Ingestion & Validation
    logger.info("[Step 1/8] Ingesting and validating transaction data...")
    ingestion = DataIngestion(
        raw_data_path="data/raw/creditcard.csv",
        target_column="Class",
        timestamp_column="Time",
        amount_column="Amount",
    )
    df, validation_report = ingestion.load_data(fallback_samples=15000, save_fallback=True)
    validation_report.save("artifacts/reports/data_quality_report.json")

    # 2. Leakage-safe Chronological Data Splitting
    logger.info("[Step 2/8] Performing chronological 70/15/15 split...")
    splitter = DataSplitter(
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        target_column="Class",
        timestamp_column="Time",
    )
    train_df, val_df, test_df, split_meta = splitter.split(df)
    Path("artifacts/reports/split_metadata.json").write_text(
        json.dumps(split_meta.to_dict(), indent=2), encoding="utf-8"
    )

    # 3. Feature Pipeline & Training-Only Resampling (SMOTE)
    logger.info("[Step 3/8] Preprocessing features and applying SMOTE to training set...")
    pipeline = FeaturePipeline(
        target_column="Class",
        timestamp_column="Time",
        amount_column="Amount",
        scaler_type="robust",
        imbalance_strategy="smote",
        smote_sampling_strategy=0.10,
        random_state=42,
    )
    X_train_res, y_train_res = pipeline.prepare_training_data(train_df)
    X_val, y_val = pipeline.prepare_eval_data(val_df)
    X_test, y_test = pipeline.prepare_eval_data(test_df)

    logger.info(f"Training feature matrix shape: {X_train_res.shape}")
    logger.info(f"Validation feature matrix shape: {X_val.shape}")
    logger.info(f"Held-out test matrix shape: {X_test.shape}")

    # 4. Model Training & Comparison
    logger.info("[Step 4/8] Training candidate models...")
    tracker = MLflowTracker(
        tracking_uri="sqlite:///mlflow.db",
        experiment_name="ShieldML_Fraud_Detection",
        registered_model_name="ShieldMLChampion",
    )

    candidate_models: dict[str, Any] = {}
    train_durations: dict[str, float] = {}

    # Model A: Baseline Logistic Regression
    t0 = time.time()
    baseline = BaselineModel(class_weight="balanced", random_state=42)
    baseline.fit(X_train_res, y_train_res)
    train_durations["LogisticRegression"] = round(time.time() - t0, 2)
    candidate_models["LogisticRegression"] = baseline

    # Model B: Random Forest
    t0 = time.time()
    rf = RandomForestModel(n_estimators=100, max_depth=12, class_weight="balanced", random_state=42)
    rf.fit(X_train_res, y_train_res)
    train_durations["RandomForest"] = round(time.time() - t0, 2)
    candidate_models["RandomForest"] = rf

    # Model C: XGBoost
    t0 = time.time()
    xgb = XGBoostModel(n_estimators=120, max_depth=6, learning_rate=0.05, scale_pos_weight=5.0, random_state=42)
    xgb.fit(X_train_res, y_train_res)
    train_durations["XGBoost"] = round(time.time() - t0, 2)
    candidate_models["XGBoost"] = xgb

    # Model D: LightGBM
    t0 = time.time()
    lgb = LightGBMModel(n_estimators=120, num_leaves=31, max_depth=6, learning_rate=0.05, scale_pos_weight=5.0, random_state=42)
    lgb.fit(X_train_res, y_train_res)
    train_durations["LightGBM"] = round(time.time() - t0, 2)
    candidate_models["LightGBM"] = lgb

    # Model E: Soft-Voting Ensemble (RF, XGB, LGBM)
    ensemble = SoftVotingEnsemble(
        models=[rf, xgb, lgb],
        weights=[0.25, 0.40, 0.35],
        model_names=["RandomForest", "XGBoost", "LightGBM"],
    )
    candidate_models["SoftVotingEnsemble"] = ensemble

    # 5. Validation Evaluation & Probability Calibration
    logger.info("[Step 5/8] Evaluating candidate models on untouched validation set...")
    val_probs_dict: dict[str, np.ndarray] = {}
    val_metrics_list = []

    for name, model in candidate_models.items():
        probs = model.predict_proba(X_val)[:, 1]
        val_probs_dict[name] = probs
        metrics = calculate_metrics(y_val, probs, threshold=0.5, model_name=name)
        val_metrics_list.append(metrics)
        logger.info(
            f"Validation ({name}): ROC-AUC={metrics.roc_auc:.4f}, PR-AUC={metrics.pr_auc:.4f}, F2={metrics.f2:.4f}"
        )

        # Log to MLflow
        tracker.log_model_run(
            run_name=f"Val_{name}",
            params={"model_type": name, "training_seconds": train_durations.get(name, 0)},
            metrics=metrics.to_dict(),
        )

    # 6. Champion Selection, Calibration & Threshold Optimization
    logger.info("[Step 6/8] Selecting champion model and optimizing decision threshold...")
    # Find model with best F2 and PR-AUC on validation set
    champion_candidate_name = max(val_metrics_list, key=lambda m: (m.f2, m.pr_auc)).model_name
    logger.info(f"Candidate champion selected from validation: {champion_candidate_name}")
    raw_champion = candidate_models[champion_candidate_name]

    # Calibrate Champion Probabilities (sigmoid) on validation set
    calibrator = ModelCalibrator(method="sigmoid")
    calibrated_champion = calibrator.calibrate(raw_champion, X_val, y_val)

    # Optimize Decision Threshold on Calibrated Validation Probabilities (maximizing F2)
    cal_val_probs = calibrated_champion.predict_proba(X_val)[:, 1]
    optimizer = ThresholdOptimizer(beta=2.0, num_steps=100)
    opt_result = optimizer.optimize(y_val.to_numpy(), cal_val_probs)
    optimal_threshold = opt_result.optimal_threshold
    logger.info(f"Optimal classification threshold (frozen for test): {optimal_threshold:.4f}")

    # 7. Final Held-Out Test Evaluation
    logger.info("[Step 7/8] Evaluating champion on held-out test data (never seen before)...")
    test_probs = calibrated_champion.predict_proba(X_test)[:, 1]
    final_test_metrics = calculate_metrics(
        y_test.to_numpy(),
        test_probs,
        threshold=optimal_threshold,
        model_name=f"{champion_candidate_name}_Calibrated",
    )
    logger.info("=" * 60)
    logger.info(" FINAL TEST RESULTS ON UNTOUCHED TEST SET:")
    logger.info(f" Model:               {final_test_metrics.model_name}")
    logger.info(f" Optimal Threshold:   {final_test_metrics.threshold}")
    logger.info(f" ROC-AUC:             {final_test_metrics.roc_auc:.4f} (Target > 0.90: {'PASSED' if final_test_metrics.roc_auc >= 0.90 else 'FAILED'})")
    logger.info(f" PR-AUC:              {final_test_metrics.pr_auc:.4f}")
    logger.info(f" F2-Score:            {final_test_metrics.f2:.4f}")
    logger.info(f" Precision:           {final_test_metrics.precision:.4f}")
    logger.info(f" Recall:              {final_test_metrics.recall:.4f}")
    logger.info(f" False Negatives:     {final_test_metrics.false_negatives}")
    logger.info(f" False Positives:     {final_test_metrics.false_positives}")
    logger.info("=" * 60)

    # 8. SHAP Explainability & Artifact Export
    logger.info("[Step 8/8] Computing SHAP explainability and persisting production artifacts...")
    explainer = FraudExplainer(
        model=calibrated_champion,
        feature_names=pipeline.feature_names,
        background_data=X_train_res.iloc[:100],
    )
    explainer.generate_global_importance_plot(
        X_test.iloc[:200],
        output_path="artifacts/figures/shap_summary.png",
    )

    # Generate visual evaluation plots
    reporter = ReportGenerator()
    val_probs_dict[f"{champion_candidate_name}_Calibrated"] = cal_val_probs
    reporter.plot_roc_curves(y_val.to_numpy(), val_probs_dict)
    reporter.plot_pr_curves(y_val.to_numpy(), val_probs_dict)
    reporter.plot_confusion_matrix(final_test_metrics)
    reporter.plot_threshold_optimization(opt_result.threshold_curve, optimal_threshold)
    reporter.save_markdown_report(val_metrics_list + [final_test_metrics], champion_name=final_test_metrics.model_name)

    # Persist Champion Model Artifacts
    models_dir = Path("artifacts/models")
    models_dir.mkdir(parents=True, exist_ok=True)

    champion_model_file = models_dir / "champion_model.joblib"
    pipeline_file = models_dir / "feature_pipeline.joblib"
    metadata_file = models_dir / "champion_metadata.json"

    joblib.dump(calibrated_champion, champion_model_file)
    pipeline.save(str(pipeline_file))

    metadata = {
        "model_name": "ShieldMLChampion",
        "model_type": final_test_metrics.model_name,
        "model_version": "1.0.0",
        "optimal_threshold": optimal_threshold,
        "optimization_metric": "f2_score",
        "f2_beta": 2.0,
        "trained_timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "test_metrics": final_test_metrics.to_dict(),
        "feature_count": len(pipeline.feature_names),
        "features": pipeline.feature_names,
        "class_imbalance": validation_report.imbalance_ratio,
    }
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # Log champion to MLflow
    champ_run_id = tracker.log_model_run(
        run_name="Champion_Final_Test",
        params={
            "champion_model": final_test_metrics.model_name,
            "optimal_threshold": optimal_threshold,
            "scaler": "robust",
            "imbalance_handling": "smote",
        },
        metrics=final_test_metrics.to_dict(),
        artifact_paths=[
            champion_model_file,
            pipeline_file,
            metadata_file,
            "artifacts/figures/roc_curves.png",
            "artifacts/figures/precision_recall_curves.png",
            "artifacts/figures/confusion_matrix.png",
            "artifacts/figures/threshold_optimization.png",
            "artifacts/figures/shap_summary.png",
        ],
    )
    tracker.register_champion(
        run_id=champ_run_id,
        model_artifact_path=str(champion_model_file),
        threshold=optimal_threshold,
        metrics=final_test_metrics.to_dict(),
    )

    logger.info("=" * 70)
    logger.info(" SHIELDML TRAINING PIPELINE COMPLETED SUCCESSFULLY")
    logger.info(f" Champion Artifacts Saved to: {models_dir}")
    logger.info("=" * 70)

    return metadata


if __name__ == "__main__":
    run_pipeline()
