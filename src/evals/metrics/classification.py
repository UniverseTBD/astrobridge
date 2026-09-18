from __future__ import annotations

from typing import Dict, List, Sequence


def accuracy(y_true: Sequence, y_pred: Sequence) -> float:
    if not len(y_true):
        return 0.0
    correct = sum(
        1
        for yt, yp in zip(y_true, y_pred)
        if yp is not None and str(yt).strip().lower() == str(yp).strip().lower()
    )
    return float(correct / len(y_true))


def confusion_matrix_dict(
    y_true: Sequence, y_pred: Sequence, labels: List[str]
) -> Dict[str, Dict[str, int]]:
    result = {
        true_lbl: {pred_lbl: 0 for pred_lbl in labels + ["unclassified"]}
        for true_lbl in labels
    }
    label_map = {lbl.lower(): lbl for lbl in labels}

    for yt, yp in zip(y_true, y_pred):
        yt_key = label_map.get(str(yt).strip().lower())
        if yt_key is None or yt_key not in result:
            continue

        yp_key = label_map.get(str(yp).strip().lower()) if yp is not None else None
        if yp_key is not None:
            result[yt_key][yp_key] += 1
        else:
            result[yt_key]["unclassified"] += 1

    return result
