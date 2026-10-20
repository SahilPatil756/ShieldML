"""
Unit tests for SHAP feature explainability.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from shieldml.explainability.shap_explainer import FraudExplainer
from shieldml.models.random_forest import RandomForestModel


def test_shap_explainer(sample_dataset: pd.DataFrame, fitted_pipeline):
    X = fitted_pipeline.transform(sample_dataset)
    y = sample_dataset["Class"]

    rf = RandomForestModel(n_estimators=5, max_depth=3).fit(X, y)
    explainer = FraudExplainer(
        model=rf,
        feature_names=fitted_pipeline.feature_names,
        background_data=X.iloc[:20],
    )

    sample_row = X.iloc[[0]]
    explanations = explainer.explain_instance(sample_row, top_k=3)

    assert len(explanations) == 3
    for item in explanations:
        assert "feature" in item
        assert "contribution" in item
        assert "impact" in item
        assert item["impact"] in {"INCREASES_FRAUD_RISK", "DECREASES_FRAUD_RISK"}
