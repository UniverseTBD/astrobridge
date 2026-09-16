from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report as sklearn_classification_report,
    confusion_matrix as sklearn_confusion_matrix,
    f1_score,
)


def accuracy(y_true: Sequence, y_pred: Sequence) -> float:
    return float(accuracy_score(y_true, y_pred)) if len(y_true) else 0.0


def macro_f1(y_true: Sequence, y_pred: Sequence, labels: Optional[List[str]] = None) -> float:
    return float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)) if len(y_true) else 0.0


def weighted_f1(y_true: Sequence, y_pred: Sequence, labels: Optional[List[str]] = None) -> float:
    return float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)) if len(y_true) else 0.0


def ordinal_mae(y_true: Sequence, y_pred: Sequence, labels: List[str]) -> float:
    label_to_rank = {lbl: i for i, lbl in enumerate(labels)}
    ranks_true, ranks_pred = [], []
    for yt, yp in zip(y_true, y_pred):
        if yt in label_to_rank and yp in label_to_rank:
            ranks_true.append(label_to_rank[yt])
            ranks_pred.append(label_to_rank[yp])
    if not ranks_true:
        return 0.0
    return float(np.mean(np.abs(np.array(ranks_true) - np.array(ranks_pred))))


def confusion_matrix_dict(y_true: Sequence, y_pred: Sequence, labels: List[str]) -> Dict[str, Dict[str, int]]:
    cm = sklearn_confusion_matrix(y_true, y_pred, labels=labels)
    result = {}
    for i, true_lbl in enumerate(labels):
        result[true_lbl] = {}
        for j, pred_lbl in enumerate(labels):
            result[true_lbl][pred_lbl] = int(cm[i, j])
    return result


def classification_report(
    y_true: Sequence,
    y_pred: Sequence,
    labels: List[str],
    *,
    ordinal: bool = False,
) -> Dict[str, Any]:
    total_samples = len(y_true)
    format_errors = sum(1 for p in y_pred if p not in labels)
    valid_mask = [p in labels for p in y_pred]

    y_true_valid = [yt for yt, v in zip(y_true, valid_mask) if v]
    y_pred_valid = [yp for yp, v in zip(y_pred, valid_mask) if v]

    acc = accuracy(y_true_valid, y_pred_valid)
    global_acc = accuracy(y_true, y_pred)
    mf1 = macro_f1(y_true_valid, y_pred_valid, labels=labels)
    wf1 = weighted_f1(y_true_valid, y_pred_valid, labels=labels)
    cm = confusion_matrix_dict(y_true, y_pred, labels=labels)

    report: Dict[str, Any] = {
        "total_samples": total_samples,
        "valid_samples": len(y_true_valid),
        "format_errors": format_errors,
        "format_error_rate": format_errors / total_samples if total_samples else 0.0,
        "accuracy": acc,
        "global_accuracy": global_acc,
        "macro_f1": mf1,
        "weighted_f1": wf1,
        "confusion_matrix": cm,
    }

    if ordinal:
        report["mean_absolute_error"] = ordinal_mae(y_true_valid, y_pred_valid, labels=labels)

    try:
        sk_report = sklearn_classification_report(
            y_true_valid,
            y_pred_valid,
            labels=labels,
            output_dict=True,
            zero_division=0,
        )
        report["per_class"] = {lbl: sk_report[lbl] for lbl in labels if lbl in sk_report}
    except Exception:
        report["per_class"] = {}

    return report
