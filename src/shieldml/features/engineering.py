"""
Feature engineering module for ShieldML.
Computes time features, cyclical hour encodings, amount transformations,
and behavioral velocity indicators without target leakage.
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Transforms raw transaction columns into rich ML features."""

    def __init__(
        self,
        timestamp_column: str | None = "Time",
        amount_column: str = "Amount",
        customer_id_column: str | None = None,
        merchant_id_column: str | None = None,
        cyclical_time: bool = True,
        log_amount: bool = True,
    ):
        self.timestamp_column = timestamp_column
        self.amount_column = amount_column
        self.customer_id_column = customer_id_column
        self.merchant_id_column = merchant_id_column
        self.cyclical_time = cyclical_time
        self.log_amount = log_amount

        # Historical customer and merchant state fit only on training
        self.customer_amount_stats: dict[str, tuple[float, float]] = {}
        self.merchant_amount_stats: dict[str, tuple[float, float]] = {}

    def fit(self, df: pd.DataFrame) -> FeatureEngineer:
        """Computes training baselines for customer and merchant historical statistics."""
        if self.customer_id_column and self.customer_id_column in df.columns:
            stats = df.groupby(self.customer_id_column)[self.amount_column].agg(["mean", "std"])
            self.customer_amount_stats = {
                k: (v["mean"], v["std"] if not np.isnan(v["std"]) else 0.0)
                for k, v in stats.iterrows()
            }

        if self.merchant_id_column and self.merchant_id_column in df.columns:
            stats = df.groupby(self.merchant_id_column)[self.amount_column].agg(["mean", "std"])
            self.merchant_amount_stats = {
                k: (v["mean"], v["std"] if not np.isnan(v["std"]) else 0.0)
                for k, v in stats.iterrows()
            }

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transforms data with leak-safe feature representations."""
        res = df.copy()

        # 1. Amount transformations
        if self.amount_column in res.columns:
            # log1p transformation for heavy-tailed amount distribution
            if self.log_amount:
                # Ensure non-negative before log1p
                non_neg_amt = np.maximum(0, res[self.amount_column])
                res["log_amount"] = np.log1p(non_neg_amt)

            # Card-testing and cashout flags
            res["is_micro_transaction"] = (res[self.amount_column] < 5.0).astype(int)
            res["is_round_amount"] = (res[self.amount_column] % 10.0 == 0).astype(int)

        # 2. Time transformations
        if self.timestamp_column and self.timestamp_column in res.columns:
            # Handle seconds from start (Kaggle format: 0 to 172800)
            # 172800s = 48 hours = 2 days
            times = res[self.timestamp_column]
            hour_of_day = (times / 3600.0) % 24.0
            res["hour_of_day"] = hour_of_day

            if self.cyclical_time:
                # Cyclical 24-hour periodic encoding
                res["sin_hour"] = np.sin(2 * np.pi * hour_of_day / 24.0)
                res["cos_hour"] = np.cos(2 * np.pi * hour_of_day / 24.0)

            # High risk nocturnal transaction window (e.g., 1 AM to 5 AM)
            res["is_night_transaction"] = ((hour_of_day >= 1.0) & (hour_of_day <= 5.0)).astype(int)

        # 3. Behavioral statistics if identifiers present
        if self.customer_id_column and self.customer_id_column in res.columns:
            devs = []
            for _, row in res.iterrows():
                cid = row.get(self.customer_id_column)
                amt = row.get(self.amount_column, 0.0)
                if cid in self.customer_amount_stats:
                    c_mean, c_std = self.customer_amount_stats[cid]
                    dev = (amt - c_mean) / (c_std + 1e-5)
                else:
                    dev = 0.0
                devs.append(dev)
            res["customer_amount_zscore"] = devs

        return res

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)
