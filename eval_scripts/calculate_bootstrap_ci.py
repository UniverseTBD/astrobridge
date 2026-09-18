#!/usr/bin/env python
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))


def _resolve_predictions_files(inputs: List[str]) -> List[Path]:
    resolved: List[Path] = []
    for raw in inputs:
        matches = [Path(p) for p in glob.glob(raw)] or [Path(raw)]
        for p in matches:
            if p.is_dir():
                cand = p / "predictions.jsonl"
                if cand.is_file():
                    resolved.append(cand)
                else:
                    resolved.extend(p.rglob("predictions.jsonl"))
            elif p.is_file():
                resolved.append(p)
    return sorted(list(set(resolved)))


def bootstrap_ci(
    scores: List[float],
    n_bootstraps: int = 10000,
    seed: int = 42,
) -> Tuple[float, Tuple[float, float]]:
    arr = np.asarray(scores, dtype=float)
    if len(arr) == 0:
        return 0.0, (0.0, 0.0)
    if len(arr) == 1:
        val = float(arr[0])
        return val, (val, val)

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(arr), size=(n_bootstraps, len(arr)))
    boot_means = arr[indices].mean(axis=1)

    obs_mean = float(np.mean(arr))
    ci_lower = float(np.percentile(boot_means, 2.5))
    ci_upper = float(np.percentile(boot_means, 97.5))
    return obs_mean, (ci_lower, ci_upper)


def _evaluate_file(file_path: Path, n_bootstraps: int = 10000, seed: int = 42) -> None:
    records: List[Dict[str, Any]] = []
    with open(file_path, "r") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    if not records:
        print(f"\nNo records found in {file_path}")
        return

    model_name = file_path.parent.parent.name
    task_name = records[0].get("task") or file_path.parent.name
    total = len(records)
    is_multilabel = "emission_lines" in task_name.lower()

    header_info = f"Model: {model_name} | Task: {task_name} | Samples: {total}"
    print(f"\n{'=' * 75}")
    print(header_info)
    print(f"File: {file_path}")
    print(f"{'=' * 75}")

    if is_multilabel:
        sample_jaccards: List[float] = []
        regime_groups: Dict[str, Dict[str, List[Any]]] = {}

        for r in records:
            gt_val = r.get("ground_truth", {})
            if isinstance(gt_val, dict):
                gt = set(k for k in gt_val.keys() if k != "regime")
            elif isinstance(gt_val, (list, set)):
                gt = set(gt_val)
            else:
                gt = set()

            p_val = r.get("frontier_evaluation", {}).get("prediction")
            pred = set(p_val) if isinstance(p_val, (list, set)) else set()

            if not gt and not pred:
                jacc = 1.0
            else:
                union = len(gt | pred)
                jacc = len(gt & pred) / union if union > 0 else 0.0

            sample_jaccards.append(jacc)

            reg = r.get("regime")
            if reg is not None:
                reg_key = str(reg)
                if reg_key not in regime_groups:
                    regime_groups[reg_key] = {"gts": [], "preds": [], "jaccards": []}
                regime_groups[reg_key]["gts"].append(gt)
                regime_groups[reg_key]["preds"].append(pred)
                regime_groups[reg_key]["jaccards"].append(jacc)

        mean_jacc, (ci_low, ci_high) = bootstrap_ci(sample_jaccards, n_bootstraps=n_bootstraps, seed=seed)
        print(f"  Overall Jaccard (95% CI):   {mean_jacc:.4f} [{ci_low:.4f}, {ci_high:.4f}]")

        if regime_groups:
            print(f"\n  By Regime (95% CI):")
            print(f"    {'Regime':<26} {'Metric':<16} {'Observed':<10} {'95% CI':<20} {'Samples':<8}")
            print(f"    {'-' * 26} {'-' * 16} {'-' * 10} {'-' * 20} {'-' * 8}")

            for reg, data in sorted(regime_groups.items()):
                reg_clean = reg.strip().lower()
                n_samples = len(data["gts"])

                if reg_clean == "pure_negative":
                    perf_scores = [1.0 if g == p else 0.0 for g, p in zip(data["gts"], data["preds"])]
                    m_perf, (l_perf, h_perf) = bootstrap_ci(perf_scores, n_bootstraps=n_bootstraps, seed=seed)
                    m_jacc, (l_jacc, h_jacc) = bootstrap_ci(data["jaccards"], n_bootstraps=n_bootstraps, seed=seed)

                    perf_str = f"{m_perf * 100:.2f}%"
                    perf_ci = f"[{l_perf * 100:.2f}%, {h_perf * 100:.2f}%]"
                    jacc_str = f"{m_jacc:.4f}"
                    jacc_ci = f"[{l_jacc:.4f}, {h_jacc:.4f}]"

                    print(f"    {reg:<26} {'Perfect Match':<16} {perf_str:<10} {perf_ci:<20} {n_samples:<8}")
                    print(f"    {'':<26} {'Jaccard':<16} {jacc_str:<10} {jacc_ci:<20}")

                elif reg_clean == "high_snr_positive":
                    rec_scores = [
                        (len(g & p) / len(g)) if g else (1.0 if not p else 0.0)
                        for g, p in zip(data["gts"], data["preds"])
                    ]
                    m_rec, (l_rec, h_rec) = bootstrap_ci(rec_scores, n_bootstraps=n_bootstraps, seed=seed)
                    m_jacc, (l_jacc, h_jacc) = bootstrap_ci(data["jaccards"], n_bootstraps=n_bootstraps, seed=seed)

                    rec_str = f"{m_rec * 100:.2f}%"
                    rec_ci = f"[{l_rec * 100:.2f}%, {h_rec * 100:.2f}%]"
                    jacc_str = f"{m_jacc:.4f}"
                    jacc_ci = f"[{l_jacc:.4f}, {h_jacc:.4f}]"

                    print(f"    {reg:<26} {'Recall':<16} {rec_str:<10} {rec_ci:<20} {n_samples:<8}")
                    print(f"    {'':<26} {'Jaccard':<16} {jacc_str:<10} {jacc_ci:<20}")

                else:
                    m_jacc, (l_jacc, h_jacc) = bootstrap_ci(data["jaccards"], n_bootstraps=n_bootstraps, seed=seed)
                    jacc_str = f"{m_jacc:.4f}"
                    jacc_ci = f"[{l_jacc:.4f}, {h_jacc:.4f}]"
                    print(f"    {reg:<26} {'Jaccard':<16} {jacc_str:<10} {jacc_ci:<20} {n_samples:<8}")
    else:
        scores: List[float] = []
        for r in records:
            gt = str(r.get("ground_truth", "")).strip().lower()
            p_val = r.get("frontier_evaluation", {}).get("prediction")
            pred = str(p_val).strip().lower() if p_val is not None else None
            scores.append(1.0 if (pred is not None and pred == gt) else 0.0)

        acc, (ci_low, ci_high) = bootstrap_ci(scores, n_bootstraps=n_bootstraps, seed=seed)
        correct_count = int(round(sum(scores)))
        print(f"  Accuracy (95% CI):   {acc * 100:.2f}% [{ci_low * 100:.2f}%, {ci_high * 100:.2f}%] ({correct_count}/{total})")

    print(f"{'=' * 75}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calculate 95% Bootstrap Confidence Intervals for model evaluation metrics without storing results."
    )
    parser.add_argument("inputs", nargs="+", help="Path(s) to predictions.jsonl or result directories.")
    parser.add_argument("--n-bootstraps", type=int, default=10000, help="Number of bootstrap resamples (default: 10000).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for bootstrapping (default: 42).")
    args = parser.parse_args()

    files = _resolve_predictions_files(args.inputs)
    if not files:
        print("No valid predictions.jsonl files found.")
        sys.exit(1)

    for file_path in files:
        _evaluate_file(file_path, n_bootstraps=args.n_bootstraps, seed=args.seed)


if __name__ == "__main__":
    main()
