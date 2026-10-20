"""
Exploratory Data Analysis script for ShieldML.
Generates distributions, fraud correlation figures, amount ranges,
and writes comprehensive statistical reports.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from shieldml.data.ingestion import DataIngestion

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("RunEDA")


def main():
    logger.info("Executing Exploratory Data Analysis (EDA)...")
    figures_dir = Path("artifacts/figures")
    reports_dir = Path("artifacts/reports")
    figures_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    ingestion = DataIngestion()
    df, report = ingestion.load_data()

    # 1. Class distribution plot
    plt.figure(figsize=(6, 4))
    counts = df["Class"].value_counts()
    bars = plt.bar(["Legitimate (0)", "Fraud (1)"], counts.values, color=["#3b82f6", "#ef4444"])
    plt.yscale("log")
    plt.ylabel("Count (Log Scale)", fontsize=11)
    plt.title(f"Class Imbalance ({counts.get(1, 0)} fraud / {counts.get(0, 0)} legit)", fontweight="bold")
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, h * 1.1, f"{int(h):,}", ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    plt.savefig(figures_dir / "eda_class_distribution.png", dpi=150)
    plt.close()

    # 2. Transaction Amount by Class
    plt.figure(figsize=(8, 5))
    df_legit = df[df["Class"] == 0]["Amount"]
    df_fraud = df[df["Class"] == 1]["Amount"]

    plt.hist(np.log1p(df_legit), bins=40, alpha=0.6, label="Legitimate", color="#3b82f6", density=True)
    plt.hist(np.log1p(df_fraud), bins=40, alpha=0.6, label="Fraud", color="#ef4444", density=True)
    plt.xlabel("log1p(Amount)", fontsize=11)
    plt.ylabel("Density", fontsize=11)
    plt.title("Transaction Amount Distribution by Class (Log1p)", fontweight="bold")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / "eda_amount_distribution.png", dpi=150)
    plt.close()

    # 3. Top correlations with fraud class
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()["Class"].drop("Class").sort_values()

    plt.figure(figsize=(9, 6))
    top_corrs = pd.concat([corr.head(5), corr.tail(5)])
    colors = ["#ef4444" if c > 0 else "#3b82f6" for c in top_corrs.values]
    plt.barh(top_corrs.index, top_corrs.values, color=colors)
    plt.xlabel("Correlation with Fraud (Class)", fontsize=11)
    plt.title("Top Negative & Positive Feature Correlations with Fraud", fontweight="bold")
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / "eda_feature_correlations.png", dpi=150)
    plt.close()

    # 4. Save statistical EDA summary report
    eda_summary = {
        "dataset_rows": len(df),
        "dataset_columns": len(df.columns),
        "legitimate_count": int(counts.get(0, 0)),
        "fraud_count": int(counts.get(1, 0)),
        "fraud_percentage": round(float(counts.get(1, 0) / len(df) * 100), 3),
        "amount_statistics": {
            "overall_mean": float(df["Amount"].mean()),
            "overall_median": float(df["Amount"].median()),
            "overall_max": float(df["Amount"].max()),
            "legit_median": float(df_legit.median()),
            "fraud_median": float(df_fraud.median()),
        },
        "top_positive_correlated_features": [
            {"feature": k, "correlation": round(float(v), 4)}
            for k, v in corr.tail(5).items()
        ],
        "top_negative_correlated_features": [
            {"feature": k, "correlation": round(float(v), 4)}
            for k, v in corr.head(5).items()
        ],
    }

    with open(reports_dir / "eda_report.json", "w", encoding="utf-8") as f:
        json.dump(eda_summary, f, indent=2)

    logger.info("EDA completed successfully. Figures and report generated.")


if __name__ == "__main__":
    main()
