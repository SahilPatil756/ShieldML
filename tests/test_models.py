"""
Unit tests for model training, prediction, and ensemble.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from shieldml.models.baseline import BaselineModel
from shieldml.models.ensemble import SoftVotingEnsemble
from shieldml.models.lightgbm_model import LightGBMModel
from shieldml.models.random_forest import RandomForestModel
from shieldml.models.xgboost_model import XGBoostModel


@pytest.fixture(scope="module")
def prepared_data(sample_dataset: pd.DataFrame):
    from shieldml.features.transformer import FeaturePipeline
    pipeline = FeaturePipeline(imbalance_strategy="none")
    X, y = pipeline.prepare_training_data(sample_dataset)
    return X, y


def test_baseline_model(prepared_data):
    X, y = prepared_data
    model = BaselineModel()
    model.fit(X, y)
    probs = model.predict_proba(X)

    assert probs.shape == (len(X), 2)
    assert np.all((probs >= 0.0) & (probs <= 1.0))
    preds = model.predict(X, threshold=0.5)
    assert len(preds) == len(X)


def test_random_forest_model(prepared_data):
    X, y = prepared_data
    rf = RandomForestModel(n_estimators=10, max_depth=4)
    rf.fit(X, y)
    probs = rf.predict_proba(X)

    assert probs.shape == (len(X), 2)
    assert np.all((probs >= 0.0) & (probs <= 1.0))


def test_xgboost_model(prepared_data):
    X, y = prepared_data
    xgb = XGBoostModel(n_estimators=10, max_depth=3)
    xgb.fit(X, y)
    probs = xgb.predict_proba(X)

    assert probs.shape == (len(X), 2)
    assert np.all((probs >= 0.0) & (probs <= 1.0))


def test_lightgbm_model(prepared_data):
    X, y = prepared_data
    lgb = LightGBMModel(n_estimators=10, max_depth=3)
    lgb.fit(X, y)
    probs = lgb.predict_proba(X)

    assert probs.shape == (len(X), 2)
    assert np.all((probs >= 0.0) & (probs <= 1.0))


def test_soft_voting_ensemble(prepared_data):
    X, y = prepared_data
    rf = RandomForestModel(n_estimators=5, max_depth=3).fit(X, y)
    xgb = XGBoostModel(n_estimators=5, max_depth=3).fit(X, y)
    lgb = LightGBMModel(n_estimators=5, max_depth=3).fit(X, y)

    ensemble = SoftVotingEnsemble(models=[rf, xgb, lgb], weights=[0.3, 0.4, 0.3])
    probs = ensemble.predict_proba(X)

    assert probs.shape == (len(X), 2)
    # Probabilities must sum to 1.0 for each row
    assert np.allclose(probs.sum(axis=1), 1.0)
