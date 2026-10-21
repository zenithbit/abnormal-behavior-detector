from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, precision_recall_curve,
    precision_recall_fscore_support, roc_auc_score,
)


def best_f1_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Threshold maximising F1; pick it on validation, then apply it unchanged to test."""
    if len(np.unique(y_true)) < 2:
        return 0.5
    p, r, t = precision_recall_curve(y_true, scores)
    f1 = 2 * p[:-1] * r[:-1] / np.clip(p[:-1] + r[:-1], 1e-9, None)
    return float(t[int(np.argmax(f1))])


def anomaly_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    pred = (scores >= threshold).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y_true, pred, average="binary", zero_division=0)
    both = len(np.unique(y_true)) == 2
    return {
        "threshold": threshold,
        "precision": float(p), "recall": float(r), "f1": float(f1),
        "roc_auc": float(roc_auc_score(y_true, scores)) if both else float("nan"),
        "pr_auc": float(average_precision_score(y_true, scores)) if both else float("nan"),
        "n": int(len(y_true)), "n_abnormal": int(y_true.sum()),
    }


def behavior_metrics(y_true: np.ndarray, y_pred: np.ndarray, classes: list[str]) -> dict:
    """On abnormal videos with a known class only (y_true >= 0)."""
    keep = y_true >= 0
    y_true, y_pred = y_true[keep], y_pred[keep]
    if len(y_true) == 0:
        return {"n": 0}
    labels = list(range(len(classes)))
    p, r, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=labels, average=None, zero_division=0)
    present = support > 0  # macro over classes that actually occur in this split
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(p[present].mean()),
        "macro_recall": float(r[present].mean()),
        "macro_f1": float(f1[present].mean()),
        "per_class": {c: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f1[i]), "support": int(support[i])}
                      for i, c in enumerate(classes)},
        "confusion_matrix": {"labels": classes, "matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist()},
    }


def end_to_end_metrics(anom_true: np.ndarray, beh_true: np.ndarray, anom_score: np.ndarray,
                       beh_pred: np.ndarray, threshold: float, classes: list[str]) -> dict:
    """Cascade as deployed: Normal if score < threshold, else the predicted behavior. 1 + C classes."""
    true = np.where(anom_true == 1, beh_true + 1, 0)
    pred = np.where(anom_score >= threshold, beh_pred + 1, 0)
    keep = ~((anom_true == 1) & (beh_true < 0))  # drop abnormal videos without a known class
    true, pred = true[keep], pred[keep]
    names = ["Normal", *classes]
    labels = list(range(len(names)))
    _, _, f1, support = precision_recall_fscore_support(true, pred, labels=labels, average=None, zero_division=0)
    return {
        "accuracy": float(accuracy_score(true, pred)),
        "macro_f1": float(f1[support > 0].mean()),
        "confusion_matrix": {"labels": names, "matrix": confusion_matrix(true, pred, labels=labels).tolist()},
    }
