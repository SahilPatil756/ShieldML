"""
Unit tests for feature engineering and pipeline transformations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from shieldml.features.engineering import FeatureEngineer
from shieldml.features.transformer import FeaturePipeline


def test_feature_engineering_time_and_amount():
    df = pd.DataFrame({
        "Time": [0.0, 3600.0, 7200.0, 86400.0],
        "Amount": [0.0, 2.50, 100.0, 500.0],
    })

    engineer = FeatureEngineer(cyclical_time=True, log_amount=True)
    res = engineer.fit_transform(df)

    assert "log_amount" in res.columns
    assert "hour_of_day" in res.columns
    assert "sin_hour" in res.columns
    assert "cos_hour" in res.columns
    assert "is_micro_transaction" in res.columns
    assert "is_round_amount" in res.columns

    # Verify log1p(0) == 0
    assert np.isclose(res.loc[0, "log_amount"], 0.0)
    # Verify micro transaction flag for $2.50 (< $5.0)
    assert res.loc[1, "is_micro_transaction"] == 1
    # Verify round amount for $100
    assert res.loc[2, "is_round_amount"] == 1


def test_pipeline_prevents_target_leakage(sample_dataset: pd.DataFrame):
    pipeline = FeaturePipeline(imbalance_strategy="none")
    X_train, y_train = pipeline.prepare_training_data(sample_dataset)

    assert "Class" not in X_train.columns
    assert len(X_train) == len(y_train)
    assert pipeline.is_fitted is True


def test_pipeline_transform_single(fitted_pipeline: FeaturePipeline, sample_transaction_payload: dict):
    X_single = fitted_pipeline.transform_single(sample_transaction_payload)

    assert len(X_single) == 1
    assert list(X_single.columns) == fitted_pipeline.feature_names
    assert not X_single.isnull().any().any()
