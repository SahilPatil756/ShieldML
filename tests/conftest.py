"""
Pytest configuration and synthetic fixtures for ShieldML test suite.
"""

from __future__ import annotations

import pytest
import numpy as np
import pandas as pd
from shieldml.data.synthetic import generate_creditcard_dataset
from shieldml.features.transformer import FeaturePipeline


@pytest.fixture(scope="session")
def sample_dataset() -> pd.DataFrame:
    """Fast, deterministic synthetic dataset for smoke tests."""
    return generate_creditcard_dataset(
        n_samples=500,
        fraud_rate=0.04,
        random_state=42,
    )


@pytest.fixture(scope="session")
def fitted_pipeline(sample_dataset: pd.DataFrame) -> FeaturePipeline:
    pipeline = FeaturePipeline(
        target_column="Class",
        timestamp_column="Time",
        amount_column="Amount",
        imbalance_strategy="none",  # fast for tests
    )
    pipeline.fit(sample_dataset)
    return pipeline


@pytest.fixture
def sample_transaction_payload() -> dict:
    return {
        "transaction_id": "TEST_TXN_001",
        "amount": 99.95,
        "time": 3600.0,
        "V1": -0.5,
        "V4": 1.2,
        "V10": -0.8,
        "V12": -1.1,
        "V14": -2.4,
        "V17": -1.5,
        "explain": True,
    }
