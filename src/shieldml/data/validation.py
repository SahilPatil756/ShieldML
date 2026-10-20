"""
Data validation module for ShieldML.
Performs schema validation, data quality checks, null/inf detection,
and class imbalance reporting.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ValidationReport:
    is_valid: bool
    total_rows: int
    total_columns: int
    columns: list[str]
    missing_values: dict[str, int]
    duplicate_rows: int
    infinite_values: dict[str, int]
    negative_amounts: int
    target_distribution: dict[str, int]
    imbalance_ratio: float
    warnings: list[str]
    errors: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, filepath: str | Path) -> None:
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Saved validation report to {p}")


class DataValidator:
    """Validates financial transaction datasets against fraud detection requirements."""

    def __init__(
        self,
        target_column: str = "Class",
        amount_column: str = "Amount",
        timestamp_column: str | None = "Time",
        allow_negative_amounts: bool = False,
    ):
        self.target_column = target_column
        self.amount_column = amount_column
        self.timestamp_column = timestamp_column
        self.allow_negative_amounts = allow_negative_amounts

    def validate(self, df: pd.DataFrame) -> ValidationReport:
        errors: list[str] = []
        warnings: list[str] = []

        total_rows = len(df)
        total_columns = len(df.columns)
        columns = list(df.columns)

        if total_rows == 0:
            errors.append("Dataset is empty (0 rows).")

        # 1. Target column presence and values
        target_dist: dict[str, int] = {}
        imbalance_ratio = 0.0

        if self.target_column not in df.columns:
            errors.append(f"Target column '{self.target_column}' is missing from dataset.")
        else:
            unique_targets = set(df[self.target_column].dropna().unique())
            if not unique_targets.issubset({0, 1, 0.0, 1.0}):
                errors.append(
                    f"Target column '{self.target_column}' must contain only binary 0 and 1, found: {unique_targets}"
                )

            vc = df[self.target_column].value_counts().to_dict()
            target_dist = {str(k): int(v) for k, v in vc.items()}
            n_legit = target_dist.get("0", 0)
            n_fraud = target_dist.get("1", 0)

            if n_fraud == 0:
                errors.append("Dataset contains 0 positive (fraud) instances.")
            elif n_legit == 0:
                errors.append("Dataset contains 0 negative (legitimate) instances.")
            else:
                imbalance_ratio = float(n_legit / max(1, n_fraud))

            if imbalance_ratio > 50:
                warnings.append(
                    f"Severe class imbalance detected: 1 fraud per {imbalance_ratio:.1f} legitimate transactions."
                )

        # 2. Missing values
        missing_dict = df.isnull().sum().to_dict()
        missing_dict = {k: int(v) for k, v in missing_dict.items() if v > 0}
        total_missing = sum(missing_dict.values())
        if total_missing > 0:
            warnings.append(f"Dataset contains {total_missing} missing values across {len(missing_dict)} columns.")

        # 3. Duplicate rows
        duplicate_count = int(df.duplicated().sum())
        if duplicate_count > 0:
            warnings.append(f"Found {duplicate_count} duplicate rows ({duplicate_count/max(1, total_rows)*100:.2f}%).")

        # 4. Infinite values in numeric columns
        inf_dict: dict[str, int] = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            inf_count = int(np.isinf(df[col]).sum())
            if inf_count > 0:
                inf_dict[col] = inf_count
        if inf_dict:
            errors.append(f"Found infinite values in columns: {list(inf_dict.keys())}")

        # 5. Amount validation
        neg_amounts = 0
        if self.amount_column in df.columns:
            if not pd.api.types.is_numeric_dtype(df[self.amount_column]):
                errors.append(f"Amount column '{self.amount_column}' is not numeric.")
            else:
                neg_amounts = int((df[self.amount_column] < 0).sum())
                if neg_amounts > 0 and not self.allow_negative_amounts:
                    warnings.append(f"Found {neg_amounts} negative transaction amounts.")

        # 6. Timestamp validation
        if self.timestamp_column and self.timestamp_column in df.columns:
            if not pd.api.types.is_numeric_dtype(df[self.timestamp_column]):
                try:
                    pd.to_datetime(df[self.timestamp_column])
                except Exception:
                    warnings.append(
                        f"Timestamp column '{self.timestamp_column}' could not be parsed as datetime or numeric."
                    )

        is_valid = len(errors) == 0

        report = ValidationReport(
            is_valid=is_valid,
            total_rows=total_rows,
            total_columns=total_columns,
            columns=columns,
            missing_values=missing_dict,
            duplicate_rows=duplicate_count,
            infinite_values=inf_dict,
            negative_amounts=neg_amounts,
            target_distribution=target_dist,
            imbalance_ratio=round(imbalance_ratio, 2),
            warnings=warnings,
            errors=errors,
        )

        return report
