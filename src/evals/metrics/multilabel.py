from __future__ import annotations

from typing import List, Set


def mean_jaccard_index(
    gt_sets: List[Set[str]],
    pred_sets: List[Set[str]],
) -> float:
    if not gt_sets:
        return 0.0

    scores = []
    for gt, pred in zip(gt_sets, pred_sets):
        if not gt and not pred:
            scores.append(1.0)
        else:
            intersection = len(gt & pred)
            union = len(gt | pred)
            scores.append(intersection / union if union > 0 else 0.0)

    return float(sum(scores) / len(scores))


def mean_recall(
    gt_sets: List[Set[str]],
    pred_sets: List[Set[str]],
) -> float:
    if not gt_sets:
        return 0.0

    recalls = []
    for gt, pred in zip(gt_sets, pred_sets):
        if not gt:
            recalls.append(1.0 if not pred else 0.0)
        else:
            recalls.append(len(gt & pred) / len(gt))

    return float(sum(recalls) / len(recalls))


def perfect_match_rate(
    gt_sets: List[Set[str]],
    pred_sets: List[Set[str]],
) -> float:
    if not gt_sets:
        return 0.0

    matches = sum(1 for gt, pred in zip(gt_sets, pred_sets) if gt == pred)
    return float(matches / len(gt_sets))
