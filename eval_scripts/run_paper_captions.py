#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import dotenv
import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from evals.config import DEFAULT_HF_DATA_REPO
from evals.hub import (
    check_and_download_existing,
    upload_caption_file,
    upload_folder_to_hf,
    write_run_metadata,
)
from eval_scripts.compare_models import paired_bootstrap_by_regime, paired_bootstrap_test
from eval_scripts.run_caption_eval import run_caption_evaluation


CANONICAL_TASKS = ["source", "subclass", "distance", "emission_lines"]


def load_suite_config(suite_arg: str) -> Dict[str, Any]:
    suite_path = Path(suite_arg)
    if not suite_path.is_file():
        candidate = _REPO_ROOT / "eval_configs" / "suites" / f"{suite_arg}.yaml"
        if candidate.is_file():
            suite_path = candidate
        else:
            raise FileNotFoundError(f"Suite configuration not found at '{suite_arg}' or '{candidate}'.")

    with open(suite_path, "r") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict) or "models" not in config:
        raise ValueError(f"Suite config at {suite_path} must be a mapping with a 'models' list.")

    return config


def resolve_models_to_run(
    all_models: List[Dict[str, Any]],
    requested_ids: Optional[List[str]] = None,
    responder_type: Optional[str] = None,
) -> List[Dict[str, Any]]:
    selected = list(all_models)
    for m in selected:
        if "responder_type" not in m and "config" in m:
            cfg_path = (_REPO_ROOT / m["config"]).resolve()
            if cfg_path.is_file():
                try:
                    with open(cfg_path, "r") as f:
                        c_dict = yaml.safe_load(f)
                        if isinstance(c_dict, dict) and "responder_type" in c_dict:
                            m["responder_type"] = c_dict["responder_type"]
                except Exception:
                    pass

    if responder_type:
        rt_clean = responder_type.strip().lower()
        selected = [m for m in selected if str(m.get("responder_type", "")).strip().lower() == rt_clean]
        if not selected:
            raise ValueError(f"No models found matching responder_type '{responder_type}'.")

    if requested_ids:
        valid_ids = {m["id"]: m for m in selected if "id" in m}
        filtered = []
        for mid in requested_ids:
            clean = mid.strip()
            if clean in valid_ids:
                filtered.append(valid_ids[clean])
            elif clean:
                raise ValueError(f"Unknown model ID '{clean}'. Valid: {list(valid_ids.keys())}")
        selected = filtered

    return selected


def execute_local_captioning(
    model_config_path: str | Path,
    output_path: str | Path,
    benchmark: str = "all",
    limit: Optional[int] = None,
    batch_size: int = 32,
    device: Optional[str] = None,
    devices: Optional[str] = None,
    num_gpus: Optional[int] = None,
    model_id: Optional[str] = None,
    max_tokens: Optional[int] = None,
    fallback_max_tokens: Optional[int] = None,
    repetition_penalty: Optional[float] = None,
) -> None:
    script_path = _REPO_ROOT / "eval_scripts" / "generate_captions.py"
    cmd = [
        sys.executable,
        str(script_path),
        "--responder", str(model_config_path),
        "--output", str(output_path),
        "--benchmark", benchmark,
        "--batch-size", str(batch_size),
    ]
    if model_id:
        cmd.extend(["--model-id", str(model_id)])
    if limit:
        cmd.extend(["--limit", str(limit)])
    if max_tokens:
        cmd.extend(["--max-tokens", str(max_tokens)])
    if fallback_max_tokens:
        cmd.extend(["--fallback-max-tokens", str(fallback_max_tokens)])
    if repetition_penalty:
        cmd.extend(["--repetition-penalty", str(repetition_penalty)])
    if device:
        cmd.extend(["--device", device])
    if devices:
        cmd.extend(["--devices", devices])
    if num_gpus:
        cmd.extend(["--num-gpus", str(num_gpus)])

    print(f"\n[LOCAL CAPTIONING] Running: {' '.join(shlex.quote(c) for c in cmd)}")
    subprocess.run(cmd, check=True)


def execute_modal_captioning(
    model_config_path: str | Path,
    output_path: str | Path,
    benchmark: str = "all",
    limit: Optional[int] = None,
    batch_size: int = 32,
    modal_gpu: Optional[str] = None,
    model_id: Optional[str] = None,
    max_tokens: Optional[int] = None,
    fallback_max_tokens: Optional[int] = None,
    repetition_penalty: Optional[float] = None,
) -> None:
    modal_script = _REPO_ROOT / "eval_scripts" / "run_modal.py"
    rel_config_path = Path(model_config_path).resolve().relative_to(_REPO_ROOT)
    output_filename = Path(output_path).name
    remote_output = f"eval_results/cached_captions/{output_filename}"

    inner_args = [
        "--responder", str(rel_config_path),
        "--output", remote_output,
        "--benchmark", benchmark,
        "--batch-size", str(batch_size),
    ]
    if model_id:
        inner_args.extend(["--model-id", str(model_id)])
    if limit:
        inner_args.extend(["--limit", str(limit)])
    if max_tokens:
        inner_args.extend(["--max-tokens", str(max_tokens)])
    if fallback_max_tokens:
        inner_args.extend(["--fallback-max-tokens", str(fallback_max_tokens)])
    if repetition_penalty:
        inner_args.extend(["--repetition-penalty", str(repetition_penalty)])

    cmd = [
        "modal", "run", str(modal_script),
        "--script", "eval_scripts/generate_captions.py",
        "--args", " ".join(inner_args),
        "--remote-output", remote_output,
        "--output", str(output_path),
    ]
    env = os.environ.copy()
    if modal_gpu:
        env["MODAL_GPU"] = modal_gpu

    print(f"\n[MODAL CAPTIONING] Running: {' '.join(shlex.quote(c) for c in cmd)}")
    subprocess.run(cmd, env=env, check=True)


def run_phase1_captions(
    suite_config: Dict[str, Any],
    models_to_run: List[Dict[str, Any]],
    captions_dir: Path,
    benchmark: str,
    args: argparse.Namespace,
) -> List[str]:
    captions_dir.mkdir(parents=True, exist_ok=True)
    hf_repo = args.hf_repo or suite_config.get("hf_repo", DEFAULT_HF_DATA_REPO)
    hf_subdir = args.hf_subdir or suite_config.get("hf_subdir", "evals/captions")

    executed = []
    for model in models_to_run:
        mid = model["id"]
        cfg_rel = model.get("config") or f"eval_configs/responders/{model.get('responder_type')}.yaml"
        cfg_path = (_REPO_ROOT / cfg_rel).resolve()
        output_filename = model.get("output_filename") or f"{mid}.jsonl"
        local_output = captions_dir / output_filename
        hf_path = f"{hf_subdir.rstrip('/')}/{output_filename}"

        print(f"\n>>> [PHASE 1] Processing captions for [{mid}]")
        skip = False
        if args.skip_existing and not args.force:
            if local_output.is_file() and local_output.stat().st_size > 0:
                print(f"[SKIP] Found existing captions: {local_output}")
                skip = True
            elif not args.no_pull_hf:
                skip = check_and_download_existing(hf_repo=hf_repo, hf_path=hf_path, local_path=local_output)

        if not skip:
            bs = args.batch_size if args.batch_size is not None else model.get("batch_size", 32)
            mid_override = model.get("hf_model_id") or model.get("astrobridge_id") or model.get("model_id")
            if args.modal:
                execute_modal_captioning(
                    model_config_path=cfg_path,
                    output_path=local_output,
                    benchmark=benchmark,
                    limit=args.limit,
                    batch_size=bs,
                    modal_gpu=args.modal_gpu or model.get("modal_gpu"),
                    model_id=mid_override,
                    max_tokens=model.get("max_tokens"),
                    fallback_max_tokens=model.get("fallback_max_tokens"),
                    repetition_penalty=model.get("repetition_penalty"),
                )
            else:
                execute_local_captioning(
                    model_config_path=cfg_path,
                    output_path=local_output,
                    benchmark=benchmark,
                    limit=args.limit,
                    batch_size=bs,
                    device=args.device,
                    devices=args.devices,
                    num_gpus=args.num_gpus,
                    model_id=mid_override,
                    max_tokens=model.get("max_tokens"),
                    fallback_max_tokens=model.get("fallback_max_tokens"),
                    repetition_penalty=model.get("repetition_penalty"),
                )

        executed.append(mid)
        if args.push_to_hf:
            upload_caption_file(
                local_path=local_output,
                hf_repo=hf_repo,
                hf_path=hf_path,
                commit_message=f"Add captions for {mid} ({suite_config.get('suite_name', 'paper')})",
            )

    return executed


def resolve_captions_file(
    model: Dict[str, Any],
    captions_dir: Path,
    hf_repo: str,
    hf_subdir: str,
    no_pull_hf: bool,
) -> Path:
    mid = model["id"]
    target_name = model.get("output_filename") or f"{mid}.jsonl"
    local_target = captions_dir / target_name

    if local_target.is_file() and local_target.stat().st_size > 0:
        return local_target

    custom = model.get("caption_file") or model.get("captions_file")
    if custom:
        p = Path(custom)
        if p.is_file() and p.stat().st_size > 0:
            return p

    if not no_pull_hf:
        hf_path = f"{hf_subdir.rstrip('/')}/{target_name}"
        if check_and_download_existing(hf_repo=hf_repo, hf_path=hf_path, local_path=local_target):
            return local_target

    raise FileNotFoundError(
        f"Captions for responder '{mid}' not found at '{local_target}' or on Hugging Face ({hf_repo}:{hf_subdir}/{target_name})."
    )


def resolve_frontier_config(
    frontier_arg: Optional[str],
    suite_frontier: Optional[str],
    gemini_model: Optional[str],
    workers: Optional[int],
    temperature: Optional[float],
) -> Dict[str, Any]:
    raw = frontier_arg or suite_frontier or "gemini"
    f_path = Path(raw)
    if not f_path.is_file():
        candidate = _REPO_ROOT / "eval_configs" / "frontier" / f"{raw}.yaml"
        if candidate.is_file():
            f_path = candidate
        else:
            raise FileNotFoundError(f"Frontier config not found: {raw}")

    with open(f_path, "r") as f:
        cfg = yaml.safe_load(f)

    if gemini_model:
        cfg["gemini_model"] = gemini_model
    if workers is not None:
        cfg["gemini_num_workers"] = workers
    if temperature is not None:
        cfg["gemini_temperature"] = temperature
    return cfg


def resolve_task_configs(tasks_arg: Optional[str], suite_tasks: Optional[List[str]]) -> Dict[str, Dict[str, Any]]:
    if tasks_arg:
        names = [t.strip() for t in tasks_arg.split(",") if t.strip()]
    elif suite_tasks:
        names = list(suite_tasks)
    else:
        names = list(CANONICAL_TASKS)

    task_cfgs = {}
    for name in names:
        p = Path(name)
        if not p.is_file():
            cand = _REPO_ROOT / "eval_configs" / "tasks" / f"{name}.yaml"
            p = cand if cand.is_file() else None

        if p and p.is_file():
            with open(p, "r") as f:
                task_cfgs[name] = yaml.safe_load(f)
        else:
            task_cfgs[name] = {"name": name}
    return task_cfgs


def evaluate_model_task(
    task_name: str,
    task_cfg: Dict[str, Any],
    frontier_cfg: Dict[str, Any],
    captions_file: Path,
    output_dir: Path,
    limit: Optional[int],
    skip_existing: bool,
    force: bool,
) -> Dict[str, Any]:
    metrics_path = output_dir / "metrics.json"
    preds_path = output_dir / "predictions.jsonl"

    if skip_existing and not force and metrics_path.is_file() and preds_path.is_file():
        try:
            with open(metrics_path, "r") as f:
                return json.load(f)
        except Exception:
            pass

    return run_caption_evaluation(
        task_config=task_cfg,
        frontier_config=frontier_cfg,
        captions_file=captions_file,
        output_dir=output_dir,
        limit=limit,
    )


def format_leaderboard(
    results: Dict[str, Dict[str, Dict[str, Any]]],
    task_names: List[str],
    baseline_id: Optional[str] = None,
) -> str:
    headers = ["Model ID"]
    for t in task_names:
        headers.append("Lines Jaccard" if "emission_lines" in t else f"{t.capitalize()} Acc")

    col_widths = [max(len(h), 14) for h in headers]
    for mid, t_dict in results.items():
        col_widths[0] = max(col_widths[0], len(mid) + 2)

    header_str = " | ".join(f"{h:<{col_widths[i]}}" for i, h in enumerate(headers))
    sep_str = "-+-".join("-" * col_widths[i] for i in range(len(headers)))
    lines = [f"\n{'=' * len(header_str)}", "AstroBridge Phase 2 Evaluation Leaderboard", f"{'=' * len(header_str)}", header_str, sep_str]

    for mid, t_dict in results.items():
        row = [f"{mid:<{col_widths[0]}}"]
        for i, t in enumerate(task_names, start=1):
            bundle = t_dict.get(t, {})
            metrics = bundle.get("task_metrics", {})
            if "emission_lines" in t:
                val = metrics.get("mean_jaccard")
                val_str = f"{val:.4f}" if val is not None else "N/A"
            else:
                val = metrics.get("accuracy")
                val_str = f"{val * 100:.2f}%" if val is not None else "N/A"

            if baseline_id and mid != baseline_id and val is not None:
                base_bundle = results.get(baseline_id, {}).get(t, {})
                base_m = base_bundle.get("task_metrics", {})
                b_val = base_m.get("mean_jaccard") if "emission_lines" in t else base_m.get("accuracy")
                if b_val is not None:
                    delta = val - b_val
                    delta_str = f" ({delta:+.4f})" if "emission_lines" in t else f" ({delta * 100:+.2f}%)"
                    val_str += delta_str

            row.append(f"{val_str:<{col_widths[i]}}")
        lines.append(" | ".join(row))

    lines.append(f"{'=' * len(header_str)}\n")
    return "\n".join(lines)


def format_regime_breakdown(results: Dict[str, Dict[str, Dict[str, Any]]]) -> str:
    lines = [
        f"\n{'=' * 75}",
        "Emission Lines Regime Breakdown",
        f"{'=' * 75}",
        f"{'Model ID':<22} {'Regime':<20} {'Jaccard':<10} {'Samples':<8} {'Note':<15}",
        f"{'-' * 22} {'-' * 20} {'-' * 10} {'-' * 8} {'-' * 15}",
    ]
    has_any = False
    for mid, t_dict in results.items():
        bundle = t_dict.get("emission_lines", {})
        reg_data = bundle.get("task_metrics", {}).get("regime_jaccard", {})
        for reg, d in sorted(reg_data.items()):
            has_any = True
            jacc = d.get("mean_jaccard", 0.0)
            n = d.get("samples", 0)
            note = ""
            if "perfect_match_rate" in d:
                note = f"perfect: {d['perfect_match_rate'] * 100:.1f}%"
            elif "recall" in d:
                note = f"recall: {d['recall'] * 100:.1f}%"
            lines.append(f"{mid:<22} {reg:<20} {jacc:<10.4f} {n:<8} {note:<15}")
    lines.append(f"{'=' * 75}\n")
    return "\n".join(lines) if has_any else ""


def export_summary_csv(
    results: Dict[str, Dict[str, Dict[str, Any]]],
    task_names: List[str],
    output_path: Path,
) -> None:
    headers = ["model_id"]
    for t in task_names:
        headers.append(f"{t}_jaccard" if "emission_lines" in t else f"{t}_accuracy")

    regimes = set()
    for t_dict in results.values():
        for reg in t_dict.get("emission_lines", {}).get("task_metrics", {}).get("regime_jaccard", {}).keys():
            regimes.add(reg)
    sorted_regimes = sorted(regimes)
    for reg in sorted_regimes:
        headers.append(f"lines_{reg}_jaccard")

    has_pure_neg = any(
        "perfect_match_rate" in t_dict.get("emission_lines", {}).get("task_metrics", {}).get("regime_jaccard", {}).get("pure_negative", {})
        for t_dict in results.values()
    )
    has_high_snr = any(
        "recall" in t_dict.get("emission_lines", {}).get("task_metrics", {}).get("regime_jaccard", {}).get("high_snr_positive", {})
        for t_dict in results.values()
    )

    if has_pure_neg:
        headers.append("lines_pure_negative_perfect_match")
    if has_high_snr:
        headers.append("lines_high_snr_positive_recall")

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for mid, t_dict in results.items():
            row = [mid]
            for t in task_names:
                m = t_dict.get(t, {}).get("task_metrics", {})
                val = m.get("mean_jaccard") if "emission_lines" in t else m.get("accuracy")
                row.append(f"{val:.4f}" if val is not None else "")
            for reg in sorted_regimes:
                val = t_dict.get("emission_lines", {}).get("task_metrics", {}).get("regime_jaccard", {}).get(reg, {}).get("mean_jaccard")
                row.append(f"{val:.4f}" if val is not None else "")
            if has_pure_neg:
                pm = t_dict.get("emission_lines", {}).get("task_metrics", {}).get("regime_jaccard", {}).get("pure_negative", {}).get("perfect_match_rate")
                row.append(f"{pm:.4f}" if pm is not None else "")
            if has_high_snr:
                rec = t_dict.get("emission_lines", {}).get("task_metrics", {}).get("regime_jaccard", {}).get("high_snr_positive", {}).get("recall")
                row.append(f"{rec:.4f}" if rec is not None else "")
            writer.writerow(row)


def run_model_comparisons(
    model_ids: List[str],
    task_names: List[str],
    eval_dir: Path,
    baseline_id: str,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    comp_summary = {"baseline_model": baseline_id, "comparisons": {}}
    table_rows = []

    for mid in model_ids:
        if mid == baseline_id:
            continue
        comp_summary["comparisons"][mid] = {}

        for t in task_names:
            base_file = eval_dir / baseline_id / t / "predictions.jsonl"
            treat_file = eval_dir / mid / t / "predictions.jsonl"
            if not base_file.is_file() or not treat_file.is_file():
                continue

            metric = "jaccard" if "emission_lines" in t else "accuracy"
            overall = paired_bootstrap_test(base_file, treat_file, metric_name=metric)
            comp_summary["comparisons"][mid][t] = overall
            table_rows.append({
                "model_id": mid,
                "baseline_model": baseline_id,
                "task": t,
                "regime": "all",
                "metric": metric,
                **overall,
            })

            if "emission_lines" in t:
                by_reg = paired_bootstrap_by_regime(base_file, treat_file, metric_name="jaccard")
                comp_summary["comparisons"][mid]["emission_lines_regimes"] = by_reg
                for reg, reg_res in by_reg.items():
                    table_rows.append({
                        "model_id": mid,
                        "baseline_model": baseline_id,
                        "task": t,
                        "regime": reg,
                        "metric": "jaccard",
                        **reg_res,
                    })

    return comp_summary, table_rows


def format_comparisons(table_rows: List[Dict[str, Any]], baseline_id: str) -> str:
    if not table_rows:
        return ""

    lines = [
        f"\n{'=' * 110}",
        f"Statistical Significance Comparison vs Baseline: [{baseline_id}]",
        f"{'=' * 110}",
        f"{'Model / Task':<32} {'Metric':<10} {'Baseline (95% CI)':<22} {'Model (95% CI)':<22} {'Delta (95% CI)':<22} {'p-value':<8}",
        f"{'-' * 32} {'-' * 10} {'-' * 22} {'-' * 22} {'-' * 22} {'-' * 8}",
    ]
    for r in table_rows:
        tag = f"{r['model_id']}:{r['task']}" if r["regime"] == "all" else f"  └ regime:{r['regime']}"
        b_str = f"{r['score_baseline']:.4f} [{r['ci_baseline'][0]:.3f}, {r['ci_baseline'][1]:.3f}]"
        m_str = f"{r['score_treatment']:.4f} [{r['ci_treatment'][0]:.3f}, {r['ci_treatment'][1]:.3f}]"
        d_str = f"{r['delta_observed']:+.4f} [{r['ci_95'][0]:+.3f}, {r['ci_95'][1]:+.3f}]"
        p_str = f"{r['p_value']:.4f}"
        lines.append(f"{tag:<32} {r['metric']:<10} {b_str:<22} {m_str:<22} {d_str:<22} {p_str:<8}")

    lines.append(f"{'=' * 110}\n")
    return "\n".join(lines)


def export_comparisons_csv(table_rows: List[Dict[str, Any]], output_path: Path) -> None:
    fieldnames = [
        "model_id", "baseline_model", "task", "regime", "metric", "num_samples",
        "score_baseline", "ci_baseline_lower", "ci_baseline_upper",
        "score_model", "ci_model_lower", "ci_model_upper",
        "delta_observed", "ci_delta_lower", "ci_delta_upper",
        "p_value", "is_significant",
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in table_rows:
            writer.writerow({
                "model_id": r["model_id"],
                "baseline_model": r["baseline_model"],
                "task": r["task"],
                "regime": r["regime"],
                "metric": r["metric"],
                "num_samples": r["num_samples"],
                "score_baseline": f"{r['score_baseline']:.4f}",
                "ci_baseline_lower": f"{r['ci_baseline'][0]:.4f}",
                "ci_baseline_upper": f"{r['ci_baseline'][1]:.4f}",
                "score_model": f"{r['score_treatment']:.4f}",
                "ci_model_lower": f"{r['ci_treatment'][0]:.4f}",
                "ci_model_upper": f"{r['ci_treatment'][1]:.4f}",
                "delta_observed": f"{r['delta_observed']:+.4f}",
                "ci_delta_lower": f"{r['ci_95'][0]:+.4f}",
                "ci_delta_upper": f"{r['ci_95'][1]:+.4f}",
                "p_value": f"{r['p_value']:.4f}",
                "is_significant": r["p_value"] < 0.05,
            })


def run_phase2_evals(
    suite_config: Dict[str, Any],
    models_to_run: List[Dict[str, Any]],
    captions_dir: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    hf_repo = args.hf_repo or suite_config.get("hf_repo", DEFAULT_HF_DATA_REPO)
    hf_subdir = args.hf_subdir or suite_config.get("hf_subdir", "evals/captions")
    hf_eval_subdir = args.hf_eval_subdir or suite_config.get("hf_eval_subdir", f"evals/frontier_evals/{suite_config.get('suite_name', 'suite')}")

    task_cfgs = resolve_task_configs(args.tasks or args.task, suite_config.get("tasks"))
    frontier_cfg = resolve_frontier_config(
        args.frontier,
        suite_config.get("frontier"),
        args.gemini_model,
        args.gemini_workers,
        args.gemini_temperature,
    )

    results: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for model in models_to_run:
        mid = model["id"]
        cap_file = resolve_captions_file(model, captions_dir, hf_repo, hf_subdir, args.no_pull_hf)
        results[mid] = {}

        for task_name, t_cfg in task_cfgs.items():
            run_dir = output_dir / mid / task_name
            print(f">>> [PHASE 2] Evaluating [{mid}] on task [{task_name}]")
            bundle = evaluate_model_task(
                task_name=task_name,
                task_cfg=t_cfg,
                frontier_cfg=frontier_cfg,
                captions_file=cap_file,
                output_dir=run_dir,
                limit=args.limit,
                skip_existing=args.skip_existing,
                force=args.force,
            )
            results[mid][task_name] = bundle

    with open(output_dir / "summary_metrics.json", "w") as f:
        json.dump(results, f, indent=4)

    export_summary_csv(results, list(task_cfgs.keys()), output_dir / "summary_table.csv")

    baseline_id = args.compare_to or suite_config.get("baseline_model")
    table_rows = []
    if baseline_id and baseline_id in results:
        comp_summary, table_rows = run_model_comparisons(
            model_ids=list(results.keys()),
            task_names=list(task_cfgs.keys()),
            eval_dir=output_dir,
            baseline_id=baseline_id,
        )
        with open(output_dir / "comparison_summary.json", "w") as f:
            json.dump(comp_summary, f, indent=4)
        export_comparisons_csv(table_rows, output_dir / "comparison_table.csv")

    print(format_leaderboard(results, list(task_cfgs.keys()), baseline_id=baseline_id))
    reg_text = format_regime_breakdown(results)
    if reg_text:
        print(reg_text)
    if table_rows:
        print(format_comparisons(table_rows, baseline_id=baseline_id))

    write_run_metadata(
        output_dir=output_dir,
        suite_config=suite_config,
        models_run=list(results.keys()),
    )

    if args.push_to_hf:
        upload_folder_to_hf(
            local_dir=output_dir,
            hf_repo=hf_repo,
            hf_path=hf_eval_subdir,
            commit_message=f"Upload Phase 2 evaluations for {suite_config.get('suite_name', 'suite')}",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="AstroBridge Suite Runner: Phase 1 (Captions), Phase 2 (Evals), or Both.")
    parser.add_argument("--phase", type=str, default="1", choices=["1", "2", "all", "captions", "eval", "both"], help="Execution phase: 1 (captions), 2 (evals), or all.")
    parser.add_argument("--suite", type=str, default="paper_controls", help="Suite YAML path or name.")
    parser.add_argument("--model", type=str, default=None, help="Specific model ID to run.")
    parser.add_argument("--models", type=str, default=None, help="Comma-separated model IDs to run.")
    parser.add_argument("--responder-type", type=str, default=None, help="Filter by responder type.")
    parser.add_argument("--benchmark", type=str, default=None, help="Benchmark dataset (Phase 1).")
    parser.add_argument("--limit", type=int, default=None, help="Sample limit.")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size for Phase 1 captioning.")
    parser.add_argument("--modal", action="store_true", help="Run Phase 1 on Modal GPUs.")
    parser.add_argument("--modal-gpu", type=str, default=None, help="Override Modal GPU type.")
    parser.add_argument("--device", type=str, default=None, help="Local device (cuda:0, mps, cpu).")
    parser.add_argument("--devices", type=str, default=None, help="Local devices (e.g. '0,1').")
    parser.add_argument("--num-gpus", type=int, default=None, help="Local GPU count.")

    # Phase 2 options
    parser.add_argument("--task", type=str, default=None, help="Specific task for Phase 2.")
    parser.add_argument("--tasks", type=str, default=None, help="Comma-separated tasks for Phase 2.")
    parser.add_argument("--frontier", type=str, default=None, help="Frontier model config (default from suite or gemini).")
    parser.add_argument("--gemini-model", type=str, default=None, help="Override Gemini model name.")
    parser.add_argument("--gemini-workers", type=int, default=None, help="Frontier concurrency workers.")
    parser.add_argument("--gemini-temperature", type=float, default=None, help="Frontier temperature.")
    parser.add_argument("--compare-to", type=str, default=None, help="Baseline model ID for statistical comparison.")

    parser.add_argument(
        "--captions-dir",
        type=str,
        default="eval_results/cached_captions",
        help="Directory containing caption files.",
    )
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip already existing files locally or on HF.")
    parser.add_argument("--no-pull-hf", action="store_true", help="Do not download from HF.")
    parser.add_argument("--force", action="store_true", help="Force re-execution.")
    parser.add_argument("--push-to-hf", action="store_true", help="Upload results to Hugging Face.")
    parser.add_argument("--hf-repo", type=str, default=None, help="Override Hugging Face repo.")
    parser.add_argument("--hf-subdir", type=str, default=None, help="Override HF captions subdir.")
    parser.add_argument("--hf-eval-subdir", type=str, default=None, help="Override HF eval subdir.")
    args = parser.parse_args()

    dotenv.load_dotenv()
    suite_config = load_suite_config(args.suite)
    benchmark = args.benchmark or suite_config.get("benchmark", "all")

    requested_ids = [m.strip() for m in args.models.split(",")] if args.models else ([args.model.strip()] if args.model else None)
    models_to_run = resolve_models_to_run(suite_config.get("models", []), requested_ids=requested_ids, responder_type=args.responder_type)

    captions_dir = Path(args.captions_dir).resolve()
    suite_name = suite_config.get("suite_name", "suite")
    phase2_out_dir = Path(args.output_dir).resolve() if args.output_dir else (_REPO_ROOT / "eval_results" / "paper_evals" / suite_name)

    phase = args.phase.lower()
    run_p1 = phase in ("1", "captions", "all", "both")
    run_p2 = phase in ("2", "eval", "all", "both")

    print(f"==================================================")
    print(f"AstroBridge Suite Runner: {suite_name}")
    print(f"Phase: {phase} (Phase 1: {run_p1}, Phase 2: {run_p2})")
    print(f"Models ({len(models_to_run)}): {[m['id'] for m in models_to_run]}")
    print(f"Push to Hugging Face: {args.push_to_hf}")
    print(f"==================================================")

    if run_p1:
        run_phase1_captions(suite_config, models_to_run, captions_dir, benchmark, args)

    if run_p2:
        run_phase2_evals(suite_config, models_to_run, captions_dir, phase2_out_dir, args)

    print("\nSuite execution completed successfully!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[ABORTED] Suite execution interrupted by user.")
        sys.exit(130)
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
