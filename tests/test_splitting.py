"""
Unit tests for data splitting and leakage prevention.
"""

from __future__ import annotations

import pandas as pd
import pytest
from shieldml.data.splitting import DataSplitter


def test_temporal_split_order(sample_dataset: pd.DataFrame):
    splitter = DataSplitter(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, timestamp_column="Time")
    train_df, val_df, test_df, meta = splitter.split(sample_dataset)

    assert meta.split_method == "temporal"
    assert len(train_df) + len(val_df) + len(test_df) == len(sample_dataset)

    # Verify strict temporal ordering: train <= val <= test
    max_train_time = train_df["Time"].max()
    min_val_time = val_df["Time"].min()
    max_val_time = val_df["Time"].max()
    min_test_time = test_df["Time"].min()

    assert max_train_time <= min_val_time, "Temporal leakage: train contains timestamps after val!"
    assert max_val_time <= min_test_time, "Temporal leakage: val contains timestamps after test!"


def test_stratified_split_preserves_classes(sample_dataset: pd.DataFrame):
    splitter = DataSplitter(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)
    train_df, val_df, test_df, meta = splitter.split(sample_dataset, force_stratified=True)

    assert meta.split_method == "stratified"
    assert (train_df["Class"] == 1).sum() > 0
    assert (val_df["Class"] == 1).sum() > 0
    assert (test_df["Class"] == 1).sum() > 0


def test_invalid_ratios_raise_error():
    with pytest.raises(ValueError, match="must sum to 1.0"):
        DataSplitter(train_ratio=0.8, val_ratio=0.3, test_ratio=0.1)
