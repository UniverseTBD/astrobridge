from __future__ import annotations

from .classification import accuracy, confusion_matrix_dict
from .multilabel import mean_jaccard_index, mean_recall, perfect_match_rate
from .reporter import compute_caption_metrics

compute_metrics = compute_caption_metrics

__all__ = [
    "accuracy",
    "confusion_matrix_dict",
    "mean_jaccard_index",
    "mean_recall",
    "perfect_match_rate",
    "compute_caption_metrics",
    "compute_metrics",
]
