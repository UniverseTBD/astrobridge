from __future__ import annotations

from .classification import (
    accuracy,
    macro_f1,
    weighted_f1,
    ordinal_mae,
    confusion_matrix_dict,
    classification_report,
)
from .multilabel import (
    sample_precision_recall_f1,
    micro_precision_recall_f1,
    snr_weighted_metrics,
    micro_snr_weighted_metrics,
    per_label_report,
    macro_label_f1,
    exact_match_rate,
    multilabel_hamming_loss,
    multilabel_report,
)
from .reporter import compute_caption_metrics, generate_report_markdown

compute_metrics = compute_caption_metrics

__all__ = [
    "accuracy",
    "macro_f1",
    "weighted_f1",
    "ordinal_mae",
    "confusion_matrix_dict",
    "classification_report",
    "sample_precision_recall_f1",
    "micro_precision_recall_f1",
    "snr_weighted_metrics",
    "micro_snr_weighted_metrics",
    "per_label_report",
    "macro_label_f1",
    "exact_match_rate",
    "multilabel_hamming_loss",
    "multilabel_report",
    "compute_caption_metrics",
    "compute_metrics",
    "generate_report_markdown",
]
