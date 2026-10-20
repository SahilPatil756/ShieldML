"""
Comprehensive evaluation metrics for financial fraud detection.
Computes ROC-AUC, PR-AUC, F2-score (penalizing false negatives),
confusion matrix components, and error rates.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class ModelMetrics:
    model_name: str
    threshold: float
    roc_auc: float
    pr_auc: float
    f1: float
    f2: float
    precision: float
    recall: float
    balanced_accuracy: float
    brier_score: float
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    false_positive_rate: float
    false_negative_rate: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def calculate_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    model_name: str = "Model",
) -> ModelMetrics:
    """Computes all required financial fraud detection metrics."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    # ROC-AUC and PR-AUC
    try:
        roc_auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        roc_auc = 0.0

    try:
        pr_auc = float(average_precision_score(y_true, y_prob))
    except Exception:
        pr_auc = 0.0

    # Classification metrics at given threshold
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    # F2: beta=2 weights recall 2x heavier than precision (missed fraud is much more costly than a false alert)
    f2 = float(fbeta_score(y_true, y_pred, beta=2.0, zero_division=0))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    brier = float(brier_score_loss(y_true, y_prob))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

    return ModelMetrics(
        model_name=model_name,
        threshold=round(float(threshold), 4),
        roc_auc=round(roc_auc, 4),
        pr_auc=round(pr_auc, 4),
        f1=round(f1, 4),
        f2=round(f2, 4),
        precision=round(precision, 4),
        recall=round(recall, 4),
        balanced_accuracy=round(bal_acc, 4),
        brier_score=round(brier, 5),
        true_positives=tp,
        false_positives=fp,
        true_negatives=tn,
        false_negatives=fn,
        false_positive_rate=round(fpr, 5),
        false_negative_rate=round(fnr, 5),
    )
