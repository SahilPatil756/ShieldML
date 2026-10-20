"""
Report and visualization generation for ShieldML.
Produces ROC curves, PR curves, Confusion Matrix heatmaps,
Threshold vs Metric charts, and Markdown comparison tables.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any
import matplotlib
matplotlib.use("Agg")  # Headless backend for server/CLI
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_curve

from shieldml.evaluation.metrics import ModelMetrics

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generates visual charts and markdown evaluation summaries."""

    def __init__(self, figures_dir: str = "artifacts/figures", reports_dir: str = "artifacts/reports"):
        self.figures_dir = Path(figures_dir)
        self.reports_dir = Path(reports_dir)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def plot_roc_curves(
        self,
        y_true: np.ndarray,
        model_probs: dict[str, np.ndarray],
        filename: str = "roc_curves.png",
    ) -> Path:
        plt.figure(figsize=(8, 6))
        for name, probs in model_probs.items():
            fpr, tpr, _ = roc_curve(y_true, probs)
            plt.plot(fpr, tpr, lw=2, label=name)

        plt.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random Guess (AUC = 0.50)")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate (FPR)", fontsize=11)
        plt.ylabel("True Positive Rate / Recall (TPR)", fontsize=11)
        plt.title("ROC Curves — ShieldML Candidate Models", fontsize=13, fontweight="bold")
        plt.legend(loc="lower right", frameon=True)
        plt.grid(alpha=0.3)
        plt.tight_layout()

        out_path = self.figures_dir / filename
        plt.savefig(out_path, dpi=150)
        plt.close()
        logger.info(f"Saved ROC curve plot to {out_path}")
        return out_path

    def plot_pr_curves(
        self,
        y_true: np.ndarray,
        model_probs: dict[str, np.ndarray],
        filename: str = "precision_recall_curves.png",
    ) -> Path:
        plt.figure(figsize=(8, 6))
        for name, probs in model_probs.items():
            precision, recall, _ = precision_recall_curve(y_true, probs)
            plt.plot(recall, precision, lw=2, label=name)

        baseline_fraud_rate = float(np.mean(y_true))
        plt.axhline(y=baseline_fraud_rate, color="r", linestyle="--", label=f"Baseline ({baseline_fraud_rate:.2%})")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("Recall", fontsize=11)
        plt.ylabel("Precision", fontsize=11)
        plt.title("Precision-Recall Curves — Severe Imbalance", fontsize=13, fontweight="bold")
        plt.legend(loc="upper right", frameon=True)
        plt.grid(alpha=0.3)
        plt.tight_layout()

        out_path = self.figures_dir / filename
        plt.savefig(out_path, dpi=150)
        plt.close()
        logger.info(f"Saved PR curve plot to {out_path}")
        return out_path

    def plot_confusion_matrix(
        self,
        metrics: ModelMetrics,
        filename: str = "confusion_matrix.png",
    ) -> Path:
        cm = np.array([
            [metrics.true_negatives, metrics.false_positives],
            [metrics.false_negatives, metrics.true_positives],
        ])

        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=["Predicted Legit (0)", "Predicted Fraud (1)"],
            yticklabels=["Actual Legit (0)", "Actual Fraud (1)"],
            annot_kws={"size": 14, "weight": "bold"},
        )
        plt.title(f"Confusion Matrix — {metrics.model_name}\nThreshold = {metrics.threshold}", fontsize=12)
        plt.tight_layout()

        out_path = self.figures_dir / filename
        plt.savefig(out_path, dpi=150)
        plt.close()
        logger.info(f"Saved confusion matrix plot to {out_path}")
        return out_path

    def plot_threshold_optimization(
        self,
        curve_data: list[dict[str, float]],
        optimal_threshold: float,
        filename: str = "threshold_optimization.png",
    ) -> Path:
        thresholds = [d["threshold"] for d in curve_data]
        precisions = [d["precision"] for d in curve_data]
        recalls = [d["recall"] for d in curve_data]
        f2s = [d["f2"] for d in curve_data]

        plt.figure(figsize=(9, 5))
        plt.plot(thresholds, precisions, label="Precision", color="#3b82f6", lw=2)
        plt.plot(thresholds, recalls, label="Recall", color="#10b981", lw=2)
        plt.plot(thresholds, f2s, label="F2-Score (Objective)", color="#f59e0b", lw=2.5)

        plt.axvline(
            x=optimal_threshold,
            color="#ef4444",
            linestyle="--",
            lw=2,
            label=f"Optimal Threshold ({optimal_threshold:.4f})",
        )

        plt.xlabel("Decision Threshold", fontsize=11)
        plt.ylabel("Score", fontsize=11)
        plt.title("Threshold vs. Precision, Recall & F2-Score Optimization", fontsize=13, fontweight="bold")
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), frameon=True)
        plt.grid(alpha=0.3)
        plt.tight_layout()

        out_path = self.figures_dir / filename
        plt.savefig(out_path, dpi=150)
        plt.close()
        logger.info(f"Saved threshold optimization plot to {out_path}")
        return out_path

    def save_markdown_report(
        self,
        model_metrics_list: list[ModelMetrics],
        champion_name: str,
        filename: str = "model_comparison_report.md",
    ) -> Path:
        """Writes scannable Markdown summary table of all models."""
        lines = [
            "# ShieldML Model Comparison & Benchmark Report",
            "",
            f"**Selected Champion Model**: `{champion_name}`",
            "",
            "## Model Comparison Table",
            "",
            "| Model | Threshold | ROC-AUC | PR-AUC | F2-Score | Precision | Recall | Brier Score | FPR | FNR |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]

        for m in model_metrics_list:
            is_champ = "**" if m.model_name == champion_name else ""
            lines.append(
                f"| {is_champ}{m.model_name}{is_champ} | {m.threshold} | {m.roc_auc:.4f} | {m.pr_auc:.4f} | "
                f"{m.f2:.4f} | {m.precision:.4f} | {m.recall:.4f} | {m.brier_score:.5f} | "
                f"{m.false_positive_rate:.4f} | {m.false_negative_rate:.4f} |"
            )

        lines.extend([
            "",
            "## Key Engineering Observations",
            "- **Evaluation Metric (F2-Score)**: Chosen because a false negative (missed fraud charge) is vastly more costly to financial institutions than a false alert (temporary card lock or SMS verification).",
            "- **Probability Calibration**: Calibrates tree output margins to true empirical probabilities, reducing the Brier score.",
            "- **Ensemble Resilience**: The soft-voting ensemble averages variance across tree families (Random Forest, XGBoost, LightGBM).",
            "",
        ])

        out_path = self.reports_dir / filename
        out_path.write_text("\n".join(lines), encoding="utf-8")
        logger.info(f"Saved markdown report to {out_path}")
        return out_path
