"""
Leakage-safe data splitting for ShieldML.
Supports chronological (temporal) splitting and stratified split fallbacks.
Guarantees training data precedes validation and test data in time.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


@dataclass
class SplitMetadata:
    split_method: str
    train_size: int
    val_size: int
    test_size: int
    train_fraud_count: int
    val_fraud_count: int
    test_fraud_count: int
    train_fraud_rate: float
    val_fraud_rate: float
    test_fraud_rate: float
    time_boundaries: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DataSplitter:
    """Partitions financial transaction data into train (70%), validation (15%), and test (15%)."""

    def __init__(
        self,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        target_column: str = "Class",
        timestamp_column: str | None = "Time",
        random_state: int = 42,
    ):
        total = train_ratio + val_ratio + test_ratio
        if not (0.99 <= total <= 1.01):
            raise ValueError(f"Split ratios must sum to 1.0, got: {total}")

        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.target_column = target_column
        self.timestamp_column = timestamp_column
        self.random_state = random_state

    def split(
        self, df: pd.DataFrame, force_stratified: bool = False
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, SplitMetadata]:
        has_timestamp = (
            self.timestamp_column is not None
            and self.timestamp_column in df.columns
            and not force_stratified
        )

        if has_timestamp:
            return self._temporal_split(df)
        else:
            return self._stratified_split(df)

    def _temporal_split(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, SplitMetadata]:
        logger.info(f"Performing temporal chronological split on '{self.timestamp_column}'...")
        # Sort chronologically to prevent temporal leakage
        df_sorted = df.sort_values(self.timestamp_column).reset_index(drop=True)

        n = len(df_sorted)
        train_end = int(n * self.train_ratio)
        val_end = int(n * (self.train_ratio + self.val_ratio))

        train_df = df_sorted.iloc[:train_end].copy().reset_index(drop=True)
        val_df = df_sorted.iloc[train_end:val_end].copy().reset_index(drop=True)
        test_df = df_sorted.iloc[val_end:].copy().reset_index(drop=True)

        time_boundaries = {
            "train_time_min": float(train_df[self.timestamp_column].min()),
            "train_time_max": float(train_df[self.timestamp_column].max()),
            "val_time_min": float(val_df[self.timestamp_column].min()),
            "val_time_max": float(val_df[self.timestamp_column].max()),
            "test_time_min": float(test_df[self.timestamp_column].min()),
            "test_time_max": float(test_df[self.timestamp_column].max()),
        }

        metadata = self._compute_metadata("temporal", train_df, val_df, test_df, time_boundaries)
        logger.info(
            f"Temporal split complete: Train={len(train_df)} ({metadata.train_fraud_count} fraud), "
            f"Val={len(val_df)} ({metadata.val_fraud_count} fraud), "
            f"Test={len(test_df)} ({metadata.test_fraud_count} fraud)"
        )
        return train_df, val_df, test_df, metadata

    def _stratified_split(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, SplitMetadata]:
        logger.info("Performing reproducible stratified split on target...")
        y = df[self.target_column]

        # First split off train
        train_df, rem_df = train_test_split(
            df,
            train_size=self.train_ratio,
            stratify=y,
            random_state=self.random_state,
        )

        # Split remainder equally between val and test
        val_prop = self.val_ratio / (self.val_ratio + self.test_ratio)
        val_df, test_df = train_test_split(
            rem_df,
            train_size=val_prop,
            stratify=rem_df[self.target_column],
            random_state=self.random_state,
        )

        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

        metadata = self._compute_metadata("stratified", train_df, val_df, test_df, {})
        logger.info(
            f"Stratified split complete: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}"
        )
        return train_df, val_df, test_df, metadata

    def _compute_metadata(
        self,
        method: str,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        time_boundaries: dict[str, Any],
    ) -> SplitMetadata:
        n_train_f = int((train_df[self.target_column] == 1).sum())
        n_val_f = int((val_df[self.target_column] == 1).sum())
        n_test_f = int((test_df[self.target_column] == 1).sum())

        return SplitMetadata(
            split_method=method,
            train_size=len(train_df),
            val_size=len(val_df),
            test_size=len(test_df),
            train_fraud_count=n_train_f,
            val_fraud_count=n_val_f,
            test_fraud_count=n_test_f,
            train_fraud_rate=round(n_train_f / max(1, len(train_df)), 5),
            val_fraud_rate=round(n_val_f / max(1, len(val_df)), 5),
            test_fraud_rate=round(n_test_f / max(1, len(test_df)), 5),
            time_boundaries=time_boundaries,
        )
