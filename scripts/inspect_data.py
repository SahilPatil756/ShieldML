"""
Data inspection script for ShieldML.
Inspects raw dataset schema, nulls, duplicates, and class imbalance.
"""

from __future__ import annotations

import logging
from shieldml.data.ingestion import DataIngestion

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("InspectData")


def main():
    logger.info("Starting ShieldML dataset inspection...")
    ingestion = DataIngestion(
        raw_data_path="data/raw/creditcard.csv",
        target_column="Class",
        timestamp_column="Time",
        amount_column="Amount",
    )
    df, report = ingestion.load_data(fallback_samples=15000, save_fallback=True)

    report.save("artifacts/reports/data_quality_report.json")

    print("\n" + "=" * 60)
    print(" SHIELDML DATASET INSPECTION SUMMARY")
    print("=" * 60)
    print(f"Total Rows:            {report.total_rows:,}")
    print(f"Total Columns:         {report.total_columns}")
    print(f"Duplicate Rows:        {report.duplicate_rows:,}")
    print(f"Class Distribution:    {report.target_distribution}")
    print(f"Imbalance Ratio:       1 fraud : {report.imbalance_ratio:.1f} legitimate")
    print(f"Missing Values:        {sum(report.missing_values.values())}")
    print(f"Negative Amounts:      {report.negative_amounts}")
    print(f"Schema Valid:          {report.is_valid}")
    if report.warnings:
        print("\nWarnings:")
        for w in report.warnings:
            print(f" - {w}")
    print("=" * 60)


if __name__ == "__main__":
    main()
