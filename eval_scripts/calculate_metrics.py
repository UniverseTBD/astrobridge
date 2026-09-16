from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import List

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from evals.metrics.reporter import compute_caption_metrics


def _resolve_predictions_files(inputs: List[str]) -> List[Path]:
    resolved: List[Path] = []
    for raw in inputs:
        matches = [Path(p) for p in glob.glob(raw)] or [Path(raw)]
        for p in matches:
            if p.is_dir():
                cand = p / "predictions.jsonl"
                if cand.is_file():
                    resolved.append(cand)
            elif p.is_file():
                resolved.append(p)
    return sorted(list(set(resolved)))


def _print_metrics_summary(bundle: dict) -> None:
    task_name = bundle.get("task_name", "unknown")
    total = bundle.get("total_samples", 0)
    metrics = bundle.get("task_metrics", {})
    diag = bundle.get("caption_diagnostics", {})

    print(f"\n{'=' * 60}")
    print(f"Task: {task_name} | Total Samples: {total}")
    print(f"{'=' * 60}")

    if "emission_lines" in task_name.lower():
        print(f"  Sample-Mean Precision:  {metrics.get('sample_precision', 0.0):.4f}")
        print(f"  Sample-Mean Recall:     {metrics.get('sample_recall', 0.0):.4f}")
        print(f"  Sample-Mean F1:         {metrics.get('sample_f1', 0.0):.4f}")
        print(f"  Dataset (Micro) F1:     {metrics.get('dataset_f1', 0.0):.4f}")
        print(f"  Dataset Macro F1:       {metrics.get('dataset_macro_f1', 0.0):.4f}")
        print(f"  Exact Match Rate:       {metrics.get('exact_match_rate', 0.0) * 100:.2f}%")
        print(f"  SNR-Weighted F1:        {metrics.get('snr_weighted_f1', 0.0):.4f}")
    else:
        print(f"  Accuracy:               {metrics.get('accuracy', 0.0):.4f}")
        print(f"  Macro F1:               {metrics.get('macro_f1', 0.0):.4f}")
        print(f"  Weighted F1:            {metrics.get('weighted_f1', 0.0):.4f}")
        if "ordinal_mae" in metrics:
            print(f"  Ordinal MAE:            {metrics.get('ordinal_mae', 0.0):.4f}")

    if diag:
        print("\nDiagnostics:")
        print(f"  Parse Success Rate:     {diag.get('frontier_parse_success_rate', 0.0) * 100:.1f}%")
        print(f"  Fallback Rate:          {diag.get('frontier_fallback_rate', 0.0) * 100:.1f}%")
        print(f"  Avg Caption Words:      {diag.get('avg_caption_word_count', 0.0):.1f}")
    print(f"{'=' * 60}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Calculate metrics and generate report from predictions.jsonl.")
    parser.add_argument("inputs", nargs="+", help="Path(s) to predictions.jsonl or result directories.")
    args = parser.parse_args()

    files = _resolve_predictions_files(args.inputs)
    if not files:
        print("No valid predictions.jsonl files found.")
        sys.exit(1)

    for file_path in files:
        print(f"Processing: {file_path}")
        bundle = compute_caption_metrics(file_path)
        _print_metrics_summary(bundle)
        print(f"Updated {file_path.parent / 'metrics.json'} and {file_path.parent / 'report.md'}")


if __name__ == "__main__":
    main()
