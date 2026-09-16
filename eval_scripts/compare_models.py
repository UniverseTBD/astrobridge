from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple
import numpy as np
from tqdm import tqdm

from evals.metrics.classification import accuracy, macro_f1


def _sample_f1(y_true: List[Any], y_pred: List[Any]) -> float:
    scores = []
    for t, p in zip(y_true, y_pred):
        set_t = set(t) if isinstance(t, (list, set, tuple)) else (set(t.keys()) if isinstance(t, dict) else {t})
        set_p = set(p) if isinstance(p, (list, set, tuple)) else {p}
        if not set_t and not set_p:
            scores.append(1.0)
            continue
        tp = len(set_t & set_p)
        prec = tp / len(set_p) if set_p else 0.0
        rec = tp / len(set_t) if set_t else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        scores.append(f1)
    return float(np.mean(scores)) if scores else 0.0


METRIC_FUNCTIONS: Dict[str, Callable[[List[Any], List[Any]], float]] = {
    "macro-f1": macro_f1,
    "accuracy": accuracy,
    "sample-f1": _sample_f1,
}


def _load_predictions(path_str: str) -> Dict[str, Dict[str, Any]]:
    p = Path(path_str)
    if p.is_dir():
        p = p / "predictions.jsonl"
    if not p.exists():
        raise FileNotFoundError(f"Predictions file not found at: {p}")

    records = {}
    with open(p, "r") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            sid = str(item.get("sample_id", item.get("object_id", "")))
            if sid:
                records[sid] = item
    return records


def paired_bootstrap_test(
    baseline_path: str,
    treatment_path: str,
    metric_name: str = "macro-f1",
    n_bootstraps: int = 10000,
    seed: int = 42,
) -> Dict[str, Any]:
    base_dict = _load_predictions(baseline_path)
    treat_dict = _load_predictions(treatment_path)

    common_ids = sorted(list(set(base_dict.keys()) & set(treat_dict.keys())))
    if not common_ids:
        raise ValueError("No common sample_ids found between baseline and treatment.")

    y_true: List[Any] = []
    y_base: List[Any] = []
    y_treat: List[Any] = []

    for sid in common_ids:
        b_item = base_dict[sid]
        t_item = treat_dict[sid]

        gt = b_item.get("ground_truth")
        b_pred = b_item.get("frontier_evaluation", {}).get("prediction", b_item.get("prediction", "UNKNOWN"))
        t_pred = t_item.get("frontier_evaluation", {}).get("prediction", t_item.get("prediction", "UNKNOWN"))

        y_true.append(gt)
        y_base.append(b_pred)
        y_treat.append(t_pred)

    metric_fn = METRIC_FUNCTIONS[metric_name]
    score_base = metric_fn(y_true, y_base)
    score_treat = metric_fn(y_true, y_treat)
    delta_obs = score_treat - score_base

    rng = np.random.RandomState(seed)
    n = len(common_ids)
    deltas = np.zeros(n_bootstraps)

    for i in tqdm(range(n_bootstraps), desc="Bootstrapping", leave=False):
        indices = rng.choice(n, size=n, replace=True)
        boot_true = [y_true[idx] for idx in indices]
        boot_base = [y_base[idx] for idx in indices]
        boot_treat = [y_treat[idx] for idx in indices]

        b_s = metric_fn(boot_true, boot_base)
        t_s = metric_fn(boot_true, boot_treat)
        deltas[i] = t_s - b_s

    p_value = float(np.mean(deltas <= 0))
    ci_lower = float(np.percentile(deltas, 2.5))
    ci_upper = float(np.percentile(deltas, 97.5))

    return {
        "num_samples": n,
        "metric": metric_name,
        "score_baseline": score_base,
        "score_treatment": score_treat,
        "delta_observed": delta_obs,
        "ci_95": (ci_lower, ci_upper),
        "p_value": p_value,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired bootstrap test to compare two models.")
    parser.add_argument("--baseline", type=str, required=True, help="Baseline predictions.jsonl or run dir.")
    parser.add_argument("--treatment", type=str, required=True, help="Treatment predictions.jsonl or run dir.")
    parser.add_argument("--metric", type=str, choices=list(METRIC_FUNCTIONS.keys()), default="macro-f1")
    parser.add_argument("--n-bootstraps", type=int, default=10000)
    args = parser.parse_args()

    results = paired_bootstrap_test(
        baseline_path=args.baseline,
        treatment_path=args.treatment,
        metric_name=args.metric,
        n_bootstraps=args.n_bootstraps,
    )

    print("\n--- Paired Bootstrap Comparison ---")
    print(f"Samples Evaluated: {results['num_samples']}")
    print(f"Metric:            {results['metric']}")
    print(f"Baseline:          {results['score_baseline']:.4f}")
    print(f"Treatment:         {results['score_treatment']:.4f}")
    print(f"Observed Delta:    {results['delta_observed']:+.4f}")
    print(f"95% CI for Delta:  [{results['ci_95'][0]:+.4f}, {results['ci_95'][1]:+.4f}]")
    print(f"p-value (H0 <= 0): {results['p_value']:.4f}")
    if results["p_value"] < 0.05:
        print("Result: Treatment is STATISTICALLY SIGNIFICANTLY BETTER (p < 0.05).")
    else:
        print("Result: Difference is NOT statistically significant (p >= 0.05).")


if __name__ == "__main__":
    main()
