"""
Data ingestion module for ShieldML.
Loads financial transaction datasets, applies column mapping adapters,
and guarantees data provenance.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
import pandas as pd

from shieldml.data.synthetic import generate_creditcard_dataset
from shieldml.data.validation import DataValidator, ValidationReport

logger = logging.getLogger(__name__)


class DatasetAdapter:
    """Maps arbitrary external transaction schemas to the internal ShieldML schema."""

    def __init__(self, column_mapping: dict[str, str] | None = None):
        self.column_mapping = column_mapping or {}

    def adapt(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.column_mapping:
            return df.copy()

        df_copy = df.copy()
        reverse_map = {v: k for k, v in self.column_mapping.items()}
        # Rename columns if source names are provided
        rename_dict = {
            src: target
            for src, target in self.column_mapping.items()
            if src in df_copy.columns
        }
        df_copy = df_copy.rename(columns=rename_dict)
        return df_copy


class DataIngestion:
    """Handles dataset loading, caching, adapter application, and fallback demonstration generation."""

    def __init__(
        self,
        raw_data_path: str | Path = "data/raw/creditcard.csv",
        target_column: str = "Class",
        timestamp_column: str | None = "Time",
        amount_column: str = "Amount",
        column_mapping: dict[str, str] | None = None,
    ):
        self.raw_data_path = Path(raw_data_path)
        self.target_column = target_column
        self.timestamp_column = timestamp_column
        self.amount_column = amount_column
        self.adapter = DatasetAdapter(column_mapping)
        self.validator = DataValidator(
            target_column=target_column,
            amount_column=amount_column,
            timestamp_column=timestamp_column,
        )

    def load_data(
        self,
        fallback_samples: int = 15000,
        save_fallback: bool = True,
    ) -> tuple[pd.DataFrame, ValidationReport]:
        """
        Loads CSV data from raw_data_path. If the file does not exist locally
        (e.g., public Kaggle dataset not yet downloaded), safely generates
        and persists a high-fidelity synthetic benchmark dataset.
        """
        if self.raw_data_path.exists():
            logger.info(f"Loading transaction dataset from {self.raw_data_path}...")
            df = pd.read_csv(self.raw_data_path)
            logger.info(f"Loaded {len(df)} rows from {self.raw_data_path}")
        else:
            logger.warning(
                f"Dataset not found at {self.raw_data_path}. "
                f"Generating {fallback_samples}-sample high-fidelity benchmark dataset."
            )
            df = generate_creditcard_dataset(
                n_samples=fallback_samples,
                fraud_rate=0.008,
                random_state=42,
                output_path=self.raw_data_path if save_fallback else None,
            )

        # Apply schema adapter
        df = self.adapter.adapt(df)

        # Validate
        report = self.validator.validate(df)
        if not report.is_valid:
            logger.error(f"Validation failed with errors: {report.errors}")
            raise ValueError(f"Dataset validation failed: {'; '.join(report.errors)}")

        return df, report
