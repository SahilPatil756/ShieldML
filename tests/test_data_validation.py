"""
Unit tests for data validation module.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from shieldml.data.validation import DataValidator


def test_valid_dataset(sample_dataset: pd.DataFrame):
    validator = DataValidator(target_column="Class", amount_column="Amount")
    report = validator.validate(sample_dataset)

    assert report.is_valid is True
    assert len(report.errors) == 0
    assert report.total_rows == len(sample_dataset)
    assert "0" in report.target_distribution
    assert "1" in report.target_distribution


def test_missing_target_column():
    df = pd.DataFrame({"Amount": [10.0, 20.0], "Time": [1, 2]})
    validator = DataValidator(target_column="Class")
    report = validator.validate(df)

    assert report.is_valid is False
    assert any("Target column 'Class' is missing" in e for e in report.errors)


def test_empty_dataset():
    df = pd.DataFrame()
    validator = DataValidator()
    report = validator.validate(df)

    assert report.is_valid is False
    assert any("empty" in e for e in report.errors)


def test_infinite_values_detection(sample_dataset: pd.DataFrame):
    df = sample_dataset.copy()
    df.loc[0, "V1"] = np.inf

    validator = DataValidator()
    report = validator.validate(df)

    assert report.is_valid is False
    assert "V1" in report.infinite_values


def test_negative_amounts_warning(sample_dataset: pd.DataFrame):
    df = sample_dataset.copy()
    df.loc[0, "Amount"] = -50.0

    validator = DataValidator(allow_negative_amounts=False)
    report = validator.validate(df)

    assert report.negative_amounts == 1
    assert any("negative transaction amounts" in w for w in report.warnings)
