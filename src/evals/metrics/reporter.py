from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from .classification import accuracy, confusion_matrix_dict
from .multilabel import mean_jaccard_index, mean_recall, perfect_match_rate


def compute_caption_metrics(results_dir: str | Path, task: Any = None) -> Dict[str, Any]:
    results_path = Path(results_dir)
    preds_file = results_path if results_path.is_file() else results_path / "predictions.jsonl"
    if not preds_file.exists():
        raise FileNotFoundError(f"Predictions file not found at: {preds_file}")

    records: List[Dict[str, Any]] = []
    with open(preds_file, "r") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    if not records:
        return {"task_name": "unknown", "total_samples": 0, "task_metrics": {}, "caption_diagnostics": {}}

    task_name = getattr(task, "name", records[0].get("task", "unknown"))
    is_multilabel = "emission_lines" in task_name.lower()

    caption_lengths = [len(r.get("caption", {}).get("text", "")) for r in records if "caption" in r]
    caption_words = [len(r.get("caption", {}).get("text", "").split()) for r in records if "caption" in r]
    fallbacks = [1 if r.get("frontier_evaluation", {}).get("forced_fallback", False) else 0 for r in records]
    parse_success = [
        1 if r.get("frontier_evaluation", {}).get("parsed_successfully", r.get("frontier_evaluation", {}).get("prediction") is not None) else 0
        for r in records
    ]

    diagnostics = {}
    if caption_lengths:
        diagnostics = {
            "avg_caption_char_len": float(sum(caption_lengths) / len(caption_lengths)),
            "avg_caption_word_count": float(sum(caption_words) / len(caption_words)),
            "frontier_parse_success_rate": float(sum(parse_success) / len(parse_success)) if parse_success else 0.0,
            "frontier_fallback_rate": float(sum(fallbacks) / len(fallbacks)) if fallbacks else 0.0,
        }

    if is_multilabel:
        gt_sets, pred_sets, regimes = [], [], []
        for r in records:
            gt_val = r.get("ground_truth", {})
            if isinstance(gt_val, dict):
                gt_sets.append(set(k for k in gt_val.keys() if k != "regime"))
            elif isinstance(gt_val, (list, set)):
                gt_sets.append(set(gt_val))
            else:
                gt_sets.append(set())

            p_val = r.get("frontier_evaluation", {}).get("prediction")
            pred_sets.append(set(p_val) if isinstance(p_val, (list, set)) else set())
            regimes.append(r.get("regime"))

        task_metrics = {
            "mean_jaccard": mean_jaccard_index(gt_sets, pred_sets),
        }

        if any(reg is not None for reg in regimes):
            regime_groups: Dict[str, tuple[list, list]] = {}
            for gt, pred, reg in zip(gt_sets, pred_sets, regimes):
                if reg is not None:
                    reg_key = str(reg)
                    if reg_key not in regime_groups:
                        regime_groups[reg_key] = ([], [])
                    regime_groups[reg_key][0].append(gt)
                    regime_groups[reg_key][1].append(pred)

            regime_metrics = {}
            for reg, (gts, preds) in sorted(regime_groups.items()):
                stats = {
                    "mean_jaccard": mean_jaccard_index(gts, preds),
                    "samples": len(gts),
                }
                reg_clean = str(reg).strip().lower()
                if reg_clean == "pure_negative":
                    stats["perfect_match_rate"] = perfect_match_rate(gts, preds)
                elif reg_clean == "high_snr_positive":
                    stats["recall"] = mean_recall(gts, preds)
                regime_metrics[reg] = stats

            task_metrics["regime_jaccard"] = regime_metrics
    else:
        y_true, y_pred = [], []
        for r in records:
            y_true.append(str(r.get("ground_truth", "")))
            p_val = r.get("frontier_evaluation", {}).get("prediction")
            y_pred.append(str(p_val) if p_val is not None else None)

        task_categories = getattr(task, "categories", None)
        if task_categories:
            labels = list(task_categories)
        elif getattr(task, "ordered_labels", None):
            labels = list(getattr(task, "letters", task.ordered_labels))
        else:
            labels = sorted(list(dict.fromkeys(y_true)))

        acc = accuracy(y_true, y_pred)
        correct_count = sum(
            1
            for yt, yp in zip(y_true, y_pred)
            if yp is not None and str(yt).strip().lower() == str(yp).strip().lower()
        )
        cm = confusion_matrix_dict(y_true, y_pred, labels=labels)

        task_metrics = {
            "accuracy": acc,
            "correct_samples": correct_count,
            "confusion_matrix": cm,
        }

    metrics_bundle = {
        "task_name": task_name,
        "total_samples": len(records),
        "task_metrics": task_metrics,
        "caption_diagnostics": diagnostics,
    }

    out_dir = preds_file.parent
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics_bundle, f, indent=4)

    return metrics_bundle
