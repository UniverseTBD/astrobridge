"""Accuracy (overall + per-class) and precision/recall/F1, hand-rolled from a confusion matrix
rather than pulling in `scikit-learn` — the label sets here are small (3 SN types, 10 galaxy
morphology classes), so a new heavy dependency isn't worth it for a handful of formulas, and
keeps `eval/` genuinely self-contained the way the rest of this folder is designed to be.

Per-class metrics matter more here than they would on a balanced task: both the transient
taxonomy (SN Ia/II/Ibc, real counts confirmed 180/71/15 on the YSE eval set) and, to a lesser
extent, Galaxy10 are imbalanced. Overall accuracy alone rewards a model that just always predicts
the majority class — reporting per-class precision/recall/F1 (and macro-F1, which weights every
class equally regardless of its support) makes that failure mode visible instead of hidden behind
a deceptively high top-line number.
"""
from __future__ import annotations

from collections import Counter


def classification_report(y_true: list[str], y_pred: list[str | None], labels: list[str]) -> dict:
    """`y_pred` entries may be `None` (the model's answer didn't match any known label — see
    `caption_to_label.predict_label`) — counted as wrong for every metric, never silently
    dropped, since an abstention/unparseable answer is a real failure mode to measure, not noise
    to discard.

    Returns:
        {
          "n": int,
          "accuracy": float,
          "macro_f1": float,
          "per_class": {label: {"precision", "recall", "f1", "support"}},
        }
    """
    if len(y_true) != len(y_pred):
        raise ValueError(f"y_true has {len(y_true)} entries but y_pred has {len(y_pred)} — must match.")
    if not y_true:
        raise ValueError("y_true is empty — nothing to score.")

    n = len(y_true)
    n_correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    accuracy = n_correct / n

    support = Counter(y_true)
    per_class: dict[str, dict] = {}
    f1_scores = []
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(support.get(label, 0)),
        }
        f1_scores.append(f1)

    macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

    unknown_true = sorted(set(y_true) - set(labels))
    if unknown_true:
        raise ValueError(
            f"y_true contains label(s) not in `labels`: {unknown_true} — the label vocabulary "
            "passed in must cover every true label actually present, or per-class metrics for "
            "those labels would silently be skipped."
        )

    return {"n": n, "accuracy": accuracy, "macro_f1": macro_f1, "per_class": per_class}
