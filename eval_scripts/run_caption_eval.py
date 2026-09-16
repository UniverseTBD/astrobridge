#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
import dotenv
import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from evals.data import load_benchmark_dataset
from evals.frontier import get_frontier_model
from evals.metrics import compute_caption_metrics
from evals.responders import SpectrumSample, get_responder
from evals.tasks import get_task


def run_caption_evaluation(
    task_config: dict,
    frontier_config: dict,
    responder_config: Optional[dict],
    captions_file: Optional[str],
    output_dir: str,
    limit: Optional[int] = None,
    batch_size: int = 32,
    device: Optional[str] = None,
    devices: Optional[list[str] | str] = None,
    num_gpus: Optional[int] = None,
) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    task = get_task(task_config["name"], **task_config.get("kwargs", {}))
    frontier = get_frontier_model(frontier_config)

    benchmark_name = task_config.get("benchmark", getattr(task, "benchmark_name", task.name))
    df = load_benchmark_dataset(benchmark_name)
    if limit is not None:
        df = df.head(limit)

    gt_by_id: Dict[str, Any] = {}
    rows_by_id: Dict[str, Any] = {}
    for _, row in df.iterrows():
        sid = str(row.get("sample_id", row.get("object_id", "")))
        gt_by_id[sid] = task.extract_ground_truth(row)
        rows_by_id[sid] = row

    captions_by_id: Dict[str, Dict[str, Any]] = {}
    if captions_file and os.path.exists(captions_file):
        with open(captions_file, "r") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    sid = str(item.get("sample_id", item.get("object_id", "")))
                    if sid in gt_by_id:
                        captions_by_id[sid] = item
    elif responder_config:
        captions_path = os.path.join(output_dir, "captions.jsonl")
        from eval_scripts.generate_captions import run_caption_generation
        run_caption_generation(
            responder_config=responder_config,
            output_path=captions_path,
            benchmark=benchmark_name,
            limit=limit,
            batch_size=batch_size,
            device=device,
            devices=devices,
            num_gpus=num_gpus,
            df=df,
        )
        with open(captions_path, "r") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    sid = str(item.get("sample_id", item.get("object_id", "")))
                    if sid in gt_by_id:
                        captions_by_id[sid] = item
    else:
        raise ValueError("Must provide either responder_config or captions_file.")

    eval_ids = [sid for sid in gt_by_id if sid in captions_by_id]
    frontier_prompts = [
        task.build_frontier_prompt(captions_by_id[sid]["caption"], item=rows_by_id[sid])
        for sid in eval_ids
    ]
    parse_fns = [task.get_parse_fn(item=rows_by_id[sid]) for sid in eval_ids]

    frontier_responses = frontier.predict_all(
        prompts=frontier_prompts,
        parse_fn=parse_fns,
        fallback_tag=task.fallback_tag(),
    )

    predictions_path = os.path.join(output_dir, "predictions.jsonl")
    with open(predictions_path, "w") as pred_f:
        for sid, prompt, f_resp in zip(eval_ids, frontier_prompts, frontier_responses):
            gt = gt_by_id[sid]
            pred = f_resp.parsed
            row = rows_by_id[sid]

            if "emission_lines" in task.name:
                cand_list = row.get("candidate_query_lines", None)
                if cand_list is not None and len(cand_list) > 0 and isinstance(pred, (list, set)):
                    cand_set = {str(k) for k in cand_list}
                    pred = [l for l in pred if l in cand_set]
                gt_set = set(gt.keys()) if isinstance(gt, dict) else set(gt)
                pred_set = set(pred) if isinstance(pred, (list, set)) else set()
                is_correct = (gt_set == pred_set) if pred is not None else False
            else:
                is_correct = (str(pred).strip().lower() == str(gt).strip().lower()) if pred is not None else False

            record = {
                "sample_id": sid,
                "survey": str(row.get("survey", "unknown")),
                "task": task.name,
                "ground_truth": gt,
                "caption": {
                    "text": captions_by_id[sid]["caption"],
                    "responder_type": captions_by_id[sid].get("responder_type", "unknown"),
                    "model_id": captions_by_id[sid].get("model_id", "unknown"),
                },
                "frontier_evaluation": {
                    "frontier_model": frontier_config.get("gemini_model", frontier_config.get("frontier_type")),
                    "prompt": prompt,
                    "raw_response": f_resp.raw_text,
                    "prediction": pred,
                    "is_correct": is_correct,
                    "forced_fallback": f_resp.forced_fallback,
                },
            }
            if "regime" in row:
                record["regime"] = str(row["regime"])
            if "candidate_query_lines" in row:
                record["candidate_query_lines"] = [str(k) for k in row["candidate_query_lines"]]

            pred_f.write(json.dumps(record) + "\n")

    run_config_path = os.path.join(output_dir, "run_config.json")
    with open(run_config_path, "w") as f:
        json.dump(
            {
                "task": task.get_config(),
                "frontier": frontier.get_config(),
                "responder": responder_config or {"source": captions_file},
                "total_samples": len(eval_ids),
                "timestamp": datetime.now().isoformat(),
            },
            f,
            indent=4,
        )

    metrics = compute_caption_metrics(output_dir, task)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Run caption-based spectra evaluation.")
    parser.add_argument("--task", type=str, required=True, help="Path to task YAML config.")
    parser.add_argument("--frontier", type=str, required=True, help="Path to frontier model YAML config.")
    parser.add_argument("--responder", type=str, default=None, help="Path to responder YAML config.")
    parser.add_argument("--captions", type=str, default=None, help="Path to pre-generated captions.jsonl.")
    parser.add_argument("--limit", type=int, default=None, help="Sample limit.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument("--gemini-model", type=str, default=None, help="Override Gemini model.")
    parser.add_argument("--output-dir", type=str, default=None, help="Custom output directory.")
    parser.add_argument("--suffix-tag", type=str, default=None, help="Optional suffix for the output directory name.")
    parser.add_argument("--device", type=str, default=None, help="Device to use for caption generation (cuda, mps, cpu).")
    parser.add_argument("--devices", type=str, default=None, help="Comma-separated devices for caption generation (e.g. '0,1', 'all').")
    parser.add_argument("--num-gpus", type=int, default=None, help="Number of GPUs to use for caption generation.")
    args = parser.parse_args()

    dotenv.load_dotenv()

    with open(args.task, "r") as f:
        task_config = yaml.safe_load(f)
    with open(args.frontier, "r") as f:
        frontier_config = yaml.safe_load(f)
    if args.gemini_model:
        frontier_config["gemini_model"] = args.gemini_model

    responder_config = None
    if args.responder:
        with open(args.responder, "r") as f:
            responder_config = yaml.safe_load(f)

    resp_tag = responder_config.get("responder_type", "responder") if responder_config else "cached"
    task_name = task_config.get("name", "task")

    if args.output_dir:
        output_dir = args.output_dir
    else:
        tag_suffix = f"_{args.suffix_tag}" if args.suffix_tag else ""
        timestamp_dir = datetime.now().strftime(f"%Y%m%d_%H%M%S_{resp_tag}_{task_name}{tag_suffix}")
        output_dir = os.path.join(os.getcwd(), "eval_results", "caption_eval", timestamp_dir)

    print(f"Starting caption evaluation: task={task_name}, output={output_dir}")
    metrics = run_caption_evaluation(
        task_config=task_config,
        frontier_config=frontier_config,
        responder_config=responder_config,
        captions_file=args.captions,
        output_dir=output_dir,
        limit=args.limit,
        batch_size=args.batch_size,
        device=args.device,
        devices=args.devices,
        num_gpus=args.num_gpus,
    )
    print(f"Evaluation complete. Results written to {output_dir}")


if __name__ == "__main__":
    main()
