"""
Feature preprocessing pipeline and transformer for ShieldML.
Handles feature scaling, imbalanced resampling (SMOTE on training only),
and single-transaction inference serialization.
"""

from __future__ import annotations

import logging
from typing import Any
import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import RobustScaler, StandardScaler

from shieldml.features.engineering import FeatureEngineer

logger = logging.getLogger(__name__)


class FeaturePipeline:
    """End-to-end feature pipeline ensuring leak-free scaling and SMOTE resampling."""

    def __init__(
        self,
        target_column: str = "Class",
        timestamp_column: str | None = "Time",
        amount_column: str = "Amount",
        scaler_type: str = "robust",  # "robust" handles financial transaction outliers well
        imbalance_strategy: str = "smote",  # "smote", "undersample", "none"
        smote_sampling_strategy: float = 0.10,  # Resample minority to 10% of majority in training set
        random_state: int = 42,
    ):
        self.target_column = target_column
        self.timestamp_column = timestamp_column
        self.amount_column = amount_column
        self.scaler_type = scaler_type
        self.imbalance_strategy = imbalance_strategy
        self.smote_sampling_strategy = smote_sampling_strategy
        self.random_state = random_state

        self.engineer = FeatureEngineer(
            timestamp_column=timestamp_column,
            amount_column=amount_column,
        )
        self.scaler = RobustScaler() if scaler_type == "robust" else StandardScaler()
        self.feature_names: list[str] = []
        self.is_fitted: bool = False

    def _extract_feature_matrix(self, df: pd.DataFrame, drop_target: bool = True) -> pd.DataFrame:
        """Extracts candidate features and drops raw identifiers or targets."""
        df_feat = self.engineer.transform(df)

        drop_cols = []
        if drop_target and self.target_column in df_feat.columns:
            drop_cols.append(self.target_column)

        # Drop non-feature identifiers if present
        for col in ["transaction_id", "customer_id", "merchant_id", "device_id"]:
            if col in df_feat.columns:
                drop_cols.append(col)

        # Keep Time, Amount, V1-V28, and engineered features
        X = df_feat.drop(columns=drop_cols, errors="ignore")
        # Ensure only numeric columns
        X = X.select_dtypes(include=[np.number])
        return X

    def fit(self, train_df: pd.DataFrame) -> FeaturePipeline:
        """Fits feature engineer and scaler on training data ONLY."""
        logger.info("Fitting FeaturePipeline on training data...")
        self.engineer.fit(train_df)
        X_train = self._extract_feature_matrix(train_df, drop_target=True)
        self.feature_names = list(X_train.columns)

        self.scaler.fit(X_train)
        self.is_fitted = True
        logger.info(f"FeaturePipeline fitted with {len(self.feature_names)} features: {self.feature_names[:8]}...")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transforms a DataFrame using the fitted scaler."""
        if not self.is_fitted:
            raise ValueError("FeaturePipeline must be fitted before calling transform.")

        X = self._extract_feature_matrix(df, drop_target=True)

        # Ensure all expected features are present; fill missing with 0.0
        for feat in self.feature_names:
            if feat not in X.columns:
                X[feat] = 0.0

        # Reorder strictly to match fitted feature names
        X = X[self.feature_names]

        # Scale features
        X_scaled = pd.DataFrame(
            self.scaler.transform(X),
            columns=self.feature_names,
            index=X.index,
        )
        return X_scaled

    def prepare_training_data(
        self, train_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Fits on training data, extracts features, and applies imbalance resampling (SMOTE).
        Resampling is NEVER applied to validation or test data.
        """
        self.fit(train_df)
        X_scaled = self.transform(train_df)
        y = train_df[self.target_column].copy()

        n_minority = int((y == 1).sum())
        n_majority = int((y == 0).sum())
        logger.info(f"Pre-resampling class distribution: 0={n_majority}, 1={n_minority}")

        if self.imbalance_strategy == "smote":
            # SMOTE oversamples minority class to target ratio in training data
            target_ratio = min(0.5, max(self.smote_sampling_strategy, n_minority / max(1, n_majority)))
            logger.info(f"Applying SMOTE with sampling_strategy={target_ratio:.3f} to training data only...")
            smote = SMOTE(
                sampling_strategy=target_ratio,
                random_state=self.random_state,
                k_neighbors=min(5, max(1, n_minority - 1)),
            )
            X_resampled, y_resampled = smote.fit_resample(X_scaled, y)
            X_out = pd.DataFrame(X_resampled, columns=self.feature_names)
            y_out = pd.Series(y_resampled, name=self.target_column)
            logger.info(
                f"Post-SMOTE class distribution: 0={(y_out == 0).sum()}, 1={(y_out == 1).sum()}"
            )
            return X_out, y_out

        elif self.imbalance_strategy == "undersample":
            target_ratio = min(0.5, self.smote_sampling_strategy * 2)
            rus = RandomUnderSampler(
                sampling_strategy=target_ratio,
                random_state=self.random_state,
            )
            X_resampled, y_resampled = rus.fit_resample(X_scaled, y)
            X_out = pd.DataFrame(X_resampled, columns=self.feature_names)
            y_out = pd.Series(y_resampled, name=self.target_column)
            return X_out, y_out

        return X_scaled, y

    def prepare_eval_data(
        self, eval_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Transforms validation or test data without any synthetic resampling."""
        X_scaled = self.transform(eval_df)
        y = eval_df[self.target_column].copy()
        return X_scaled, y

    def transform_single(self, transaction: dict[str, Any]) -> pd.DataFrame:
        """Transforms a single transaction dictionary for real-time API prediction."""
        df_single = pd.DataFrame([transaction])
        return self.transform(df_single)

    def save(self, filepath: str) -> None:
        joblib.dump(self, filepath)
        logger.info(f"Saved FeaturePipeline to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> FeaturePipeline:
        return joblib.load(filepath)
