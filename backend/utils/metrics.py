"""
Metrics computation for NeuralScribe v2.
Per-class precision/recall/F1, confusion matrix, top confusion pairs.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict


def compute_per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
) -> Dict[int, Dict[str, float]]:
    """Compute precision, recall, F1 per class."""
    metrics = {}
    for c in range(num_classes):
        tp = int(np.sum((y_pred == c) & (y_true == c)))
        fp = int(np.sum((y_pred == c) & (y_true != c)))
        fn = int(np.sum((y_pred != c) & (y_true == c)))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        support = int(np.sum(y_true == c))
        metrics[c] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support,
        }
    return metrics


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    """Return NxN confusion matrix."""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def top_confusion_pairs(
    confusion_matrix: np.ndarray,
    k: int = 10,
    id_to_display: Optional[Dict[int, str]] = None,
) -> List[Dict[str, Any]]:
    """
    Return the top-k most confused class pairs (off-diagonal).
    Each entry: {true_id, pred_id, true_label, pred_label, count}
    """
    cm = confusion_matrix.copy()
    np.fill_diagonal(cm, 0)
    # Flatten and get top-k indices
    flat_indices = np.argsort(cm.ravel())[::-1][:k]
    pairs = []
    for idx in flat_indices:
        true_id = int(idx // cm.shape[1])
        pred_id = int(idx % cm.shape[1])
        count = int(cm[true_id, pred_id])
        if count == 0:
            break
        true_label = id_to_display.get(true_id, str(true_id)) if id_to_display else str(true_id)
        pred_label = id_to_display.get(pred_id, str(pred_id)) if id_to_display else str(pred_id)
        pairs.append({
            "true_id": true_id,
            "pred_id": pred_id,
            "true_label": true_label,
            "pred_label": pred_label,
            "count": count,
        })
    return pairs


def compute_overall_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) == 0:
        return 0.0
    return float(np.mean(y_true == y_pred))


def compute_category_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_to_category: Dict[int, str],
) -> Dict[str, Dict[str, float]]:
    """Accuracy grouped by category (digit, uppercase, greek, etc.)."""
    category_correct = defaultdict(int)
    category_total = defaultdict(int)
    for t, p in zip(y_true, y_pred):
        cat = class_to_category.get(int(t), "unknown")
        category_total[cat] += 1
        if t == p:
            category_correct[cat] += 1
    return {
        cat: {
            "accuracy": round(category_correct[cat] / total, 4) if total > 0 else 0.0,
            "total": total,
            "correct": category_correct[cat],
        }
        for cat, total in category_total.items()
    }