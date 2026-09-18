from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple
import numpy as np
from tqdm import tqdm

from evals.metrics.classification import accuracy
from evals.metrics.multilabel import mean_jaccard_index


def _sample_jaccard(y_true: List[Any], y_pred: List[Any]) -> float:
    gt_sets = [
        set(t.keys()) if isinstance(t, dict) else (set(t) if isinstance(t, (list, set, tuple)) else {t})
        for t in y_true
    ]
    pred_sets = [
        set(p) if isinstance(p, (list, set, tuple)) else ({p} if p is not None else set())
        for p in y_pred
    ]
    return mean_jaccard_index(gt_sets, pred_sets)


METRIC_FUNCTIONS: Dict[str, Callable[[List[Any], List[Any]], float]] = {
    "accuracy": accuracy,
    "jaccard": _sample_jaccard,
}


def _load_predictions(path_str: str | Path) -> Dict[str, Dict[str, Any]]:
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


def _run_bootstrap_on_samples(
    base_dict: Dict[str, Dict[str, Any]],
    treat_dict: Dict[str, Dict[str, Any]],
    sample_ids: List[str],
    metric_name: str,
    n_bootstraps: int = 10000,
    seed: int = 42,
) -> Dict[str, Any]:
    y_true, y_base, y_treat = [], [], []
    for sid in sample_ids:
        b_item = base_dict[sid]
        t_item = treat_dict[sid]
        y_true.append(b_item.get("ground_truth"))
        y_base.append(b_item.get("frontier_evaluation", {}).get("prediction", b_item.get("prediction")))
        y_treat.append(t_item.get("frontier_evaluation", {}).get("prediction", t_item.get("prediction")))

    metric_fn = METRIC_FUNCTIONS[metric_name]
    score_base = metric_fn(y_true, y_base)
    score_treat = metric_fn(y_true, y_treat)
    delta_obs = score_treat - score_base

    rng = np.random.RandomState(seed)
    n = len(sample_ids)
    b_scores = np.zeros(n_bootstraps)
    t_scores = np.zeros(n_bootstraps)
    deltas = np.zeros(n_bootstraps)

    for i in range(n_bootstraps):
        indices = rng.choice(n, size=n, replace=True)
        boot_true = [y_true[idx] for idx in indices]
        b_s = metric_fn(boot_true, [y_base[idx] for idx in indices])
        t_s = metric_fn(boot_true, [y_treat[idx] for idx in indices])
        b_scores[i] = b_s
        t_scores[i] = t_s
        deltas[i] = t_s - b_s

    p_val = float(np.mean(deltas <= 0))
    return {
        "num_samples": n,
        "metric": metric_name,
        "score_baseline": score_base,
        "ci_baseline": (float(np.percentile(b_scores, 2.5)), float(np.percentile(b_scores, 97.5))),
        "score_treatment": score_treat,
        "ci_treatment": (float(np.percentile(t_scores, 2.5)), float(np.percentile(t_scores, 97.5))),
        "delta_observed": delta_obs,
        "ci_95": (float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))),
        "p_value": p_val,
    }


def paired_bootstrap_test(
    baseline_path: str | Path,
    treatment_path: str | Path,
    metric_name: str = "accuracy",
    n_bootstraps: int = 10000,
    seed: int = 42,
) -> Dict[str, Any]:
    base_dict = _load_predictions(baseline_path)
    treat_dict = _load_predictions(treatment_path)
    common_ids = sorted(list(set(base_dict.keys()) & set(treat_dict.keys())))
    if not common_ids:
        raise ValueError("No common sample_ids found between baseline and treatment.")

    return _run_bootstrap_on_samples(
        base_dict=base_dict,
        treat_dict=treat_dict,
        sample_ids=common_ids,
        metric_name=metric_name,
        n_bootstraps=n_bootstraps,
        seed=seed,
    )


def paired_bootstrap_by_regime(
    baseline_path: str | Path,
    treatment_path: str | Path,
    metric_name: str = "jaccard",
    n_bootstraps: int = 10000,
    seed: int = 42,
) -> Dict[str, Dict[str, Any]]:
    base_dict = _load_predictions(baseline_path)
    treat_dict = _load_predictions(treatment_path)
    common_ids = sorted(list(set(base_dict.keys()) & set(treat_dict.keys())))

    regime_ids: Dict[str, List[str]] = {}
    for sid in common_ids:
        reg = base_dict[sid].get("regime")
        if reg:
            regime_ids.setdefault(str(reg), []).append(sid)

    results = {}
    for reg, sids in sorted(regime_ids.items()):
        if sids:
            results[reg] = _run_bootstrap_on_samples(
                base_dict=base_dict,
                treat_dict=treat_dict,
                sample_ids=sids,
                metric_name=metric_name,
                n_bootstraps=n_bootstraps,
                seed=seed,
            )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired bootstrap test to compare two models.")
    parser.add_argument("--baseline", type=str, required=True, help="Baseline predictions.jsonl or run dir.")
    parser.add_argument("--treatment", type=str, required=True, help="Treatment predictions.jsonl or run dir.")
    parser.add_argument("--metric", type=str, choices=list(METRIC_FUNCTIONS.keys()), default="accuracy")
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
    print(f"Baseline Score:    {results['score_baseline']:.4f} (95% CI: [{results['ci_baseline'][0]:.4f}, {results['ci_baseline'][1]:.4f}])")
    print(f"Treatment Score:   {results['score_treatment']:.4f} (95% CI: [{results['ci_treatment'][0]:.4f}, {results['ci_treatment'][1]:.4f}])")
    print(f"Observed Delta:    {results['delta_observed']:+.4f}")
    print(f"95% CI for Delta:  [{results['ci_95'][0]:+.4f}, {results['ci_95'][1]:+.4f}]")
    print(f"p-value (H0 <= 0): {results['p_value']:.4f}")
    if results["p_value"] < 0.05:
        print("Result: Treatment is STATISTICALLY SIGNIFICANTLY BETTER (p < 0.05).")
    else:
        print("Result: Difference is NOT statistically significant (p >= 0.05).")


if __name__ == "__main__":
    main()
