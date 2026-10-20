"""
Synthetic dataset generator for financial fraud detection.
Produces datasets matching Kaggle Credit Card Fraud schema (Time, V1-V28, Amount, Class)
and optional behavioral schema (customer_id, merchant_id, device_id).
"""

from __future__ import annotations

import logging
from pathlib import Path
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def generate_creditcard_dataset(
    n_samples: int = 15000,
    fraud_rate: float = 0.008,  # 0.8% fraud rate (highly imbalanced)
    random_state: int = 42,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    Generate synthetic dataset mimicking Kaggle Credit Card Fraud dataset:
    Time (0 to 172800 seconds, i.e., 48 hours),
    V1-V28 (PCA transformed numerical features),
    Amount (transaction amount in USD),
    Class (0 = legitimate, 1 = fraudulent).
    """
    rng = np.random.default_rng(random_state)
    n_fraud = max(20, int(n_samples * fraud_rate))
    n_legit = n_samples - n_fraud

    logger.info(
        f"Generating credit card dataset: {n_samples} rows ({n_legit} legit, {n_fraud} fraud, {fraud_rate*100:.2f}%)"
    )

    # 1. Time: elapsed seconds over 48 hours (0 to 172800)
    legit_time = np.sort(rng.uniform(0, 172800, size=n_legit))
    fraud_time = np.sort(rng.uniform(0, 172800, size=n_fraud))
    time_series = np.concatenate([legit_time, fraud_time])

    # 2. Amount: Log-normal distribution
    legit_amount = rng.lognormal(mean=3.2, sigma=1.2, size=n_legit)
    fraud_mode = rng.choice([0, 1], size=n_fraud, p=[0.35, 0.65])
    fraud_amount = np.where(
        fraud_mode == 0,
        rng.uniform(0.5, 9.99, size=n_fraud),
        rng.lognormal(mean=5.8, sigma=1.0, size=n_fraud),
    )
    amounts = np.round(np.concatenate([legit_amount, fraud_amount]), 2)

    # 3. Features V1-V28: Normal distributions with distinct shifts for key fraud-sensitive features
    legit_v = rng.standard_normal(size=(n_legit, 28))
    fraud_v = rng.standard_normal(size=(n_fraud, 28))

    # Shift specific PCA dimensions for fraud signature (V14, V4, V12, V10, V17, V11)
    fraud_v[:, 13] -= rng.normal(4.5, 1.2, size=n_fraud)   # V14
    fraud_v[:, 3] += rng.normal(3.8, 1.0, size=n_fraud)    # V4
    fraud_v[:, 11] -= rng.normal(3.5, 1.1, size=n_fraud)   # V12
    fraud_v[:, 9] -= rng.normal(3.0, 1.0, size=n_fraud)    # V10
    fraud_v[:, 16] -= rng.normal(2.8, 1.2, size=n_fraud)   # V17
    fraud_v[:, 10] += rng.normal(2.5, 1.0, size=n_fraud)   # V11

    v_features = np.vstack([legit_v, fraud_v])
    classes = np.concatenate([np.zeros(n_legit, dtype=int), np.ones(n_fraud, dtype=int)])

    columns = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]
    data_matrix = np.column_stack([time_series, v_features, amounts, classes])
    df = pd.DataFrame(data_matrix, columns=columns)

    # Sort chronologically by Time
    df = df.sort_values("Time").reset_index(drop=True)
    df["Class"] = df["Class"].astype(int)

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_p, index=False)
        logger.info(f"Saved dataset to {out_p} ({len(df)} rows)")

    return df


def generate_behavioral_dataset(
    n_samples: int = 5000,
    n_customers: int = 200,
    n_merchants: int = 50,
    fraud_rate: float = 0.01,
    random_state: int = 42,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    Generate dataset with behavioral identifiers:
    transaction_id, customer_id, merchant_id, device_id, timestamp, amount, is_fraud.
    """
    rng = np.random.default_rng(random_state)
    start_ts = 1720000000
    timestamps = np.sort(rng.integers(start_ts, start_ts + 86400 * 7, size=n_samples))

    customer_ids = [f"CUST_{rng.integers(1000, 1000 + n_customers):04d}" for _ in range(n_samples)]
    merchant_ids = [f"MERCH_{rng.integers(100, 100 + n_merchants):03d}" for _ in range(n_samples)]
    device_ids = [f"DEV_{rng.integers(500, 500 + n_customers * 2):04d}" for _ in range(n_samples)]
    amounts = np.round(rng.lognormal(mean=3.5, sigma=1.0, size=n_samples), 2)
    transaction_ids = [f"TXN_{i+1:06d}" for i in range(n_samples)]

    is_fraud = (rng.uniform(0, 1, size=n_samples) < fraud_rate).astype(int)

    df = pd.DataFrame(
        {
            "transaction_id": transaction_ids,
            "timestamp": timestamps,
            "customer_id": customer_ids,
            "merchant_id": merchant_ids,
            "device_id": device_ids,
            "amount": amounts,
            "is_fraud": is_fraud,
        }
    )

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_p, index=False)

    return df
