from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np

from .classification import (
    accuracy,
    macro_f1,
    weighted_f1,
    ordinal_mae,
    confusion_matrix_dict,
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
)


def _format_table(headers: List[str], rows: List[List[str]], aligns: Optional[List[str]] = None) -> str:
    if not aligns:
        aligns = [":---"] + [":---:"] * (len(headers) - 1)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(aligns) + " |",
    ]
    for r in rows:
        lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines)


def generate_report_markdown(
    task_name: str,
    total_samples: int,
    task_metrics: Dict[str, Any],
    diagnostics: Dict[str, Any],
    records: List[Dict[str, Any]],
    source_file: str = "predictions.jsonl",
) -> str:
    is_multilabel = "emission_lines" in task_name.lower()
    lines = [
        f"# Evaluation Report: {task_name}",
        "",
        f"**Source file:** `{source_file}`  ",
        f"**Total Samples:** {total_samples}  ",
        "",
        "## 1. Executive Summary",
        "",
    ]

    sec_idx = 2
    if is_multilabel:
        headers = ["Metric", "Value"]
        rows = [
            ["**Sample-Mean Precision**", f"{task_metrics.get('sample_precision', 0.0):.4f}"],
            ["**Sample-Mean Recall**", f"{task_metrics.get('sample_recall', 0.0):.4f}"],
            ["**Sample-Mean F1**", f"{task_metrics.get('sample_f1', 0.0):.4f}"],
            ["**Sample-Mean Jaccard (IoU)**", f"{task_metrics.get('sample_jaccard', 0.0):.4f}"],
            ["**Dataset-Level (Micro) Precision**", f"{task_metrics.get('dataset_precision', 0.0):.4f}"],
            ["**Dataset-Level (Micro) Recall**", f"{task_metrics.get('dataset_recall', 0.0):.4f}"],
            ["**Dataset-Level (Micro) F1**", f"{task_metrics.get('dataset_f1', 0.0):.4f}"],
            ["**Dataset-Level (Micro) Jaccard**", f"{task_metrics.get('dataset_jaccard', 0.0):.4f}"],
            ["**Dataset-Level Macro F1**", f"{task_metrics.get('dataset_macro_f1', 0.0):.4f}"],
            ["**Exact Match Rate**", f"{task_metrics.get('exact_match_rate', 0.0) * 100:.2f}%"],
            ["**Hamming Loss**", f"{task_metrics.get('hamming_loss', 0.0):.4f}"],
            ["**SNR-Weighted Recall (Sample)**", f"{task_metrics.get('snr_weighted_recall', 0.0):.4f}"],
            ["**SNR-Weighted F1 (Sample)**", f"{task_metrics.get('snr_weighted_f1', 0.0):.4f}"],
            ["**SNR-Weighted F1 (Dataset Micro)**", f"{task_metrics.get('dataset_micro_snr_weighted_f1', 0.0):.4f}"],
        ]
        lines.append(_format_table(headers, rows))

        regime_breakdown = task_metrics.get("regime_breakdown", {})
        if regime_breakdown:
            lines.extend(["", f"## {sec_idx}. Emission Line Regime Breakdown", ""])
            sec_idx += 1
            rb_headers = ["Regime", "Samples", "Clean Neg", "Hallucination", "Recall", "Precision", "F1", "Jaccard", "Exact Match"]
            rb_rows = []
            for reg, stats in sorted(regime_breakdown.items()):
                clean_str = f"{stats.get('clean_negative_rate', 0.0)*100:.1f}%" if "clean_negative_rate" in stats else "N/A"
                halluc_str = f"{stats.get('hallucination_rate', 0.0)*100:.1f}%" if "hallucination_rate" in stats else "N/A"
                rb_rows.append([
                    f"**{reg}**",
                    str(stats.get("num_samples", 0)),
                    clean_str,
                    halluc_str,
                    f"{stats.get('recall', 0.0):.4f}",
                    f"{stats.get('precision', 0.0):.4f}",
                    f"{stats.get('f1', 0.0):.4f}",
                    f"{stats.get('jaccard', 0.0):.4f}",
                    f"{stats.get('exact_match_rate', 0.0)*100:.1f}%",
                ])
            lines.append(_format_table(rb_headers, rb_rows))

        per_line = task_metrics.get("per_line", {})
        if per_line:
            lines.extend(["", f"## {sec_idx}. Per-Line Detection Statistics", ""])
            sec_idx += 1
            p_headers = ["Line", "Support", "TP", "FP", "FN", "Precision", "Recall", "F1", "Mean SNR"]
            p_rows = []
            for line_name, s in per_line.items():
                mean_snr_str = f"{s.get('mean_snr', 0.0):.1f}" if s.get("mean_snr") is not None else "-"
                p_rows.append([
                    f"**{line_name}**",
                    str(s.get("support", 0)),
                    str(s.get("tp", 0)),
                    str(s.get("fp", 0)),
                    str(s.get("fn", 0)),
                    f"{s.get('precision', 0.0):.4f}",
                    f"{s.get('recall', 0.0):.4f}",
                    f"{s.get('f1', 0.0):.4f}",
                    mean_snr_str,
                ])
            lines.append(_format_table(p_headers, p_rows))
    else:
        headers = ["Metric", "Value"]
        rows = [
            ["**Accuracy**", f"{task_metrics.get('accuracy', 0.0):.4f}"],
            ["**Macro F1**", f"{task_metrics.get('macro_f1', 0.0):.4f}"],
            ["**Weighted F1**", f"{task_metrics.get('weighted_f1', 0.0):.4f}"],
        ]
        if "ordinal_mae" in task_metrics:
            rows.append(["**Ordinal MAE**", f"{task_metrics.get('ordinal_mae', 0.0):.4f}"])
        lines.append(_format_table(headers, rows))

        cm = task_metrics.get("confusion_matrix")
        if isinstance(cm, dict) and cm:
            lines.extend(["", f"## {sec_idx}. Confusion Matrix", ""])
            sec_idx += 1
            labels = list(cm.keys())
            cm_headers = ["True \\ Pred"] + labels
            cm_rows = []
            for row_lbl in labels:
                cm_rows.append([f"**{row_lbl}**"] + [str(cm[row_lbl].get(col, 0)) for col in labels])
            lines.append(_format_table(cm_headers, cm_rows))

    if diagnostics:
        lines.extend([
            "",
            f"## {sec_idx}. Caption Quality Diagnostics",
            "",
            _format_table(
                ["Diagnostic Indicator", "Value"],
                [
                    ["**Avg Caption Length (Chars)**", f"{diagnostics.get('avg_caption_char_len', 0.0):.1f} ± {diagnostics.get('std_caption_char_len', 0.0):.1f}"],
                    ["**Avg Caption Word Count**", f"{diagnostics.get('avg_caption_word_count', 0.0):.1f} ± {diagnostics.get('std_caption_word_count', 0.0):.1f}"],
                    ["**Frontier Parse Success Rate**", f"{diagnostics.get('frontier_parse_success_rate', 0.0) * 100:.1f}%"],
                    ["**Frontier Fallback Retry Rate**", f"{diagnostics.get('frontier_fallback_rate', 0.0) * 100:.1f}%"],
                ]
            ),
        ])
        sec_idx += 1

    successes = [r for r in records if r.get("frontier_evaluation", {}).get("is_correct", False)]
    failures = [r for r in records if not r.get("frontier_evaluation", {}).get("is_correct", False)]
    if successes or failures:
        lines.extend(["", f"## {sec_idx}. Qualitative Spotlights", ""])
        if successes:
            lines.append("### Top Success Examples")
            for i, s in enumerate(successes[:2]):
                cap = s.get("caption", {}).get("text", "").strip()
                raw = s.get("frontier_evaluation", {}).get("raw_response", "").strip()
                sid = s.get("sample_id", f"Sample {i+1}")
                gt_val = s.get("ground_truth")
                pred_val = s.get("frontier_evaluation", {}).get("prediction")
                lines.extend([
                    f"#### Success Case #{i+1} (ID: `{sid}`)",
                    f"- **Ground Truth**: `{gt_val}` | **Predicted**: `{pred_val}`",
                    f"- **Generated Caption**:\n> {cap}" if cap else "",
                    f"- **Frontier Reasoning**:\n```\n{raw}\n```" if raw else "",
                    "",
                ])
        if failures:
            lines.append("### Top Failure Examples")
            for i, f_item in enumerate(failures[:2]):
                cap = f_item.get("caption", {}).get("text", "").strip()
                raw = f_item.get("frontier_evaluation", {}).get("raw_response", "").strip()
                sid = f_item.get("sample_id", f"Sample {i+1}")
                gt_val = f_item.get("ground_truth")
                pred_val = f_item.get("frontier_evaluation", {}).get("prediction")
                lines.extend([
                    f"#### Failure Case #{i+1} (ID: `{sid}`)",
                    f"- **Ground Truth**: `{gt_val}` | **Predicted**: `{pred_val}`",
                    f"- **Generated Caption**:\n> {cap}" if cap else "",
                    f"- **Frontier Reasoning**:\n```\n{raw}\n```" if raw else "",
                    "",
                ])

    return "\n".join(lines) + "\n"


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
        return {"total_samples": 0, "task_metrics": {}, "caption_diagnostics": {}}

    task_name = getattr(task, "name", records[0].get("task", "unknown"))
    is_multilabel = "emission_lines" in task_name.lower()

    caption_lengths = [len(r.get("caption", {}).get("text", "")) for r in records if "caption" in r]
    caption_words = [len(r.get("caption", {}).get("text", "").split()) for r in records if "caption" in r]
    fallbacks = [1 if r.get("frontier_evaluation", {}).get("forced_fallback", False) else 0 for r in records]
    parse_success = [
        0 if r.get("frontier_evaluation", {}).get("prediction") in (None, "UNKNOWN") else 1
        for r in records
    ]

    diagnostics = {}
    if caption_lengths:
        diagnostics = {
            "avg_caption_char_len": float(np.mean(caption_lengths)),
            "std_caption_char_len": float(np.std(caption_lengths)),
            "avg_caption_word_count": float(np.mean(caption_words)),
            "std_caption_word_count": float(np.std(caption_words)),
            "frontier_parse_success_rate": float(np.mean(parse_success)) if parse_success else 1.0,
            "frontier_fallback_rate": float(np.mean(fallbacks)) if fallbacks else 0.0,
        }

    task_metrics: Dict[str, Any] = {}

    if is_multilabel:
        gt_sets, pred_sets, gt_dicts = [], [], []
        for r in records:
            gt_val = r.get("ground_truth", {})
            if isinstance(gt_val, dict):
                clean_gt = {k: float(v) for k, v in gt_val.items() if k != "regime"}
                gt_sets.append(set(clean_gt.keys()))
                gt_dicts.append(clean_gt)
            elif isinstance(gt_val, (list, set)):
                gt_sets.append(set(gt_val))
                gt_dicts.append({str(k): 1.0 for k in gt_val})
            else:
                gt_sets.append(set())
                gt_dicts.append({})

            p_val = r.get("frontier_evaluation", {}).get("prediction", [])
            pred_sets.append(set(p_val) if isinstance(p_val, (list, set)) else set())

        prf = sample_precision_recall_f1(gt_sets, pred_sets)
        micro_prf = micro_precision_recall_f1(gt_sets, pred_sets)
        snr_m = snr_weighted_metrics(gt_dicts, pred_sets)
        micro_snr = micro_snr_weighted_metrics(gt_dicts, pred_sets)
        exact = exact_match_rate(gt_sets, pred_sets)

        canonical_lines = getattr(task, "canonical_lines", None)
        if not canonical_lines:
            try:
                from ..tasks.emission_lines import CANONICAL_EMISSION_LINES
                canonical_lines = CANONICAL_EMISSION_LINES
            except Exception:
                combined = set().union(*gt_sets, *pred_sets)
                canonical_lines = sorted(list(combined)) if combined else ["HALPHA", "HBETA"]

        line_m = per_label_report(gt_sets, pred_sets, labels=canonical_lines, gt_dicts=gt_dicts)
        macro_f1_val = macro_label_f1(gt_sets, pred_sets, labels=canonical_lines)
        h_loss = multilabel_hamming_loss(gt_sets, pred_sets, labels=canonical_lines)

        task_metrics = {
            "sample_precision": prf.get("mean_precision", 0.0),
            "sample_recall": prf.get("mean_recall", 0.0),
            "sample_f1": prf.get("mean_f1", 0.0),
            "sample_jaccard": prf.get("mean_jaccard", 0.0),
            "snr_weighted_recall": snr_m.get("mean_snr_weighted_recall", 0.0),
            "snr_weighted_f1": snr_m.get("mean_snr_weighted_f1", 0.0),
            "dataset_precision": micro_prf.get("micro_precision", 0.0),
            "dataset_recall": micro_prf.get("micro_recall", 0.0),
            "dataset_f1": micro_prf.get("micro_f1", 0.0),
            "dataset_jaccard": micro_prf.get("micro_jaccard", 0.0),
            "dataset_micro_snr_weighted_recall": micro_snr.get("micro_snr_weighted_recall", 0.0),
            "dataset_micro_snr_weighted_f1": micro_snr.get("micro_snr_weighted_f1", 0.0),
            "dataset_macro_f1": macro_f1_val,
            "exact_match_rate": exact,
            "hamming_loss": h_loss,
            "sample_level": {**prf, **snr_m},
            "dataset_micro_level": {**micro_prf, **micro_snr},
            "per_line": line_m,
        }

        has_regimes = any("regime" in r for r in records)
        if has_regimes:
            regime_records: Dict[str, List[Dict[str, Any]]] = {}
            for r in records:
                reg = r.get("regime")
                if reg:
                    regime_records.setdefault(str(reg), []).append(r)

            regime_breakdown = {}
            for reg, r_list in regime_records.items():
                r_gt_sets = [set(r.get("ground_truth", {}).keys()) if isinstance(r.get("ground_truth"), dict) else set(r.get("ground_truth", [])) for r in r_list]
                r_pred_sets = [set(r.get("frontier_evaluation", {}).get("prediction", [])) if isinstance(r.get("frontier_evaluation", {}).get("prediction"), (list, set)) else set() for r in r_list]
                r_prf = sample_precision_recall_f1(r_gt_sets, r_pred_sets)
                reg_data = {
                    "num_samples": len(r_list),
                    "exact_match_rate": exact_match_rate(r_gt_sets, r_pred_sets),
                    "precision": r_prf.get("mean_precision", 0.0),
                    "recall": r_prf.get("mean_recall", 0.0),
                    "f1": r_prf.get("mean_f1", 0.0),
                    "jaccard": r_prf.get("mean_jaccard", 0.0),
                }
                if reg == "pure_negative":
                    clean_neg = sum(1 for p in r_pred_sets if len(p) == 0)
                    reg_data["clean_negative_rate"] = clean_neg / len(r_list) if r_list else 0.0
                    reg_data["hallucination_rate"] = (len(r_list) - clean_neg) / len(r_list) if r_list else 0.0
                regime_breakdown[reg] = reg_data

            task_metrics["regime_breakdown"] = regime_breakdown
    else:
        y_true, y_pred = [], []
        for r in records:
            gt = r.get("ground_truth")
            pred = r.get("frontier_evaluation", {}).get("prediction")
            y_true.append(str(gt) if gt is not None else "UNKNOWN")
            y_pred.append(str(pred) if pred is not None else "UNKNOWN")

        acc = accuracy(y_true, y_pred)
        mf1 = macro_f1(y_true, y_pred)
        wf1 = weighted_f1(y_true, y_pred)
        labels = sorted(list(set(y_true + y_pred) - {"UNKNOWN"}))
        cm = confusion_matrix_dict(y_true, y_pred, labels=labels)

        ord_mae = None
        if labels and all(len(lbl) == 1 and lbl.isupper() for lbl in labels):
            try:
                ord_mae = ordinal_mae(y_true, y_pred, labels=sorted(labels))
            except Exception:
                ord_mae = None

        task_metrics = {
            "accuracy": acc,
            "macro_f1": mf1,
            "weighted_f1": wf1,
            "confusion_matrix": cm,
        }
        if ord_mae is not None:
            task_metrics["ordinal_mae"] = ord_mae

    metrics_bundle = {
        "task_name": task_name,
        "total_samples": len(records),
        "task_metrics": task_metrics,
        "caption_diagnostics": diagnostics,
    }

    out_dir = preds_file.parent
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics_bundle, f, indent=4)

    report_md = generate_report_markdown(
        task_name=task_name,
        total_samples=len(records),
        task_metrics=task_metrics,
        diagnostics=diagnostics,
        records=records,
        source_file=preds_file.name,
    )
    with open(out_dir / "report.md", "w") as f:
        f.write(report_md)

    return metrics_bundle
