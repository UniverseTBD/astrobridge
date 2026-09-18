#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
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
from evals.tasks import get_task


def load_captions(captions_file: str | Path, valid_ids: set[str]) -> Dict[str, Dict[str, Any]]:
    captions_by_id = {}
    path = Path(captions_file)
    if not path.is_file():
        raise FileNotFoundError(f"Captions file not found: {captions_file}")

    with open(path, "r") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                sid = str(item.get("sample_id", item.get("object_id", "")))
                if sid in valid_ids:
                    captions_by_id[sid] = item
    return captions_by_id


def evaluate_sample(
    task: Any,
    sid: str,
    row: Any,
    gt: Any,
    prompt: str,
    f_resp: Any,
    caption_item: Dict[str, Any],
    frontier_name: str,
) -> Dict[str, Any]:
    pred = f_resp.parsed
    if "emission_lines" in task.name:
        cand_list = row.get("candidate_query_lines")
        if cand_list is not None and len(cand_list) > 0 and isinstance(pred, (list, set)):
            cand_set = {str(k) for k in cand_list}
            pred = [l for l in pred if l in cand_set]
        gt_set = set(gt.keys()) if isinstance(gt, dict) else set(gt)
        pred_set = set(pred) if isinstance(pred, (list, set)) else set()
        is_correct = bool(gt_set == pred_set and pred is not None)
    else:
        is_correct = bool(pred is not None and str(pred).strip().lower() == str(gt).strip().lower())

    record = {
        "sample_id": sid,
        "survey": str(row.get("survey", "unknown")),
        "task": task.name,
        "ground_truth": gt,
        "caption": {
            "text": caption_item.get("caption", ""),
            "responder_type": caption_item.get("responder_type", "unknown"),
            "model_id": caption_item.get("model_id", "unknown"),
        },
        "frontier_evaluation": {
            "frontier_model": frontier_name,
            "prompt": prompt,
            "raw_response": f_resp.raw_text,
            "parsed_successfully": pred is not None,
            "prediction": pred,
            "is_correct": is_correct,
            "forced_fallback": f_resp.forced_fallback,
        },
    }
    if "regime" in row and row["regime"] is not None:
        record["regime"] = str(row["regime"])
    if "candidate_query_lines" in row and row["candidate_query_lines"] is not None and len(row["candidate_query_lines"]) > 0:
        record["candidate_query_lines"] = [str(k) for k in row["candidate_query_lines"]]
    return record


def run_caption_evaluation(
    task_config: dict,
    frontier_config: dict,
    captions_file: str | Path,
    output_dir: str | Path,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    task = get_task(task_config["name"], **task_config.get("kwargs", {}))
    frontier = get_frontier_model(frontier_config)
    frontier_name = frontier_config.get("gemini_model", frontier_config.get("frontier_type", "frontier"))

    benchmark_name = task_config.get("benchmark", getattr(task, "benchmark_name", task.name))
    df = load_benchmark_dataset(benchmark_name)
    if limit is not None:
        df = df.head(limit)

    gt_by_id = {}
    rows_by_id = {}
    for _, row in df.iterrows():
        sid = str(row.get("sample_id", row.get("object_id", "")))
        gt_by_id[sid] = task.extract_ground_truth(row)
        rows_by_id[sid] = row

    captions_by_id = load_captions(captions_file, set(gt_by_id.keys()))
    eval_ids = [sid for sid in gt_by_id if sid in captions_by_id]

    if not eval_ids:
        print(f"[WARNING] No matching samples found between benchmark '{benchmark_name}' and '{captions_file}'")
        return {"task_name": task.name, "total_samples": 0, "task_metrics": {}, "caption_diagnostics": {}}

    prompts = [
        task.build_frontier_prompt(captions_by_id[sid]["caption"], item=rows_by_id[sid])
        for sid in eval_ids
    ]
    parse_fns = [task.get_parse_fn(item=rows_by_id[sid]) for sid in eval_ids]

    responses = frontier.predict_all(
        prompts=prompts,
        parse_fn=parse_fns,
        fallback_tag=task.fallback_tag(),
    )

    predictions_path = out_dir / "predictions.jsonl"
    with open(predictions_path, "w") as pred_f:
        for sid, prompt, f_resp in zip(eval_ids, prompts, responses):
            rec = evaluate_sample(
                task=task,
                sid=sid,
                row=rows_by_id[sid],
                gt=gt_by_id[sid],
                prompt=prompt,
                f_resp=f_resp,
                caption_item=captions_by_id[sid],
                frontier_name=frontier_name,
            )
            pred_f.write(json.dumps(rec) + "\n")

    run_config_path = out_dir / "run_config.json"
    with open(run_config_path, "w") as f:
        json.dump(
            {
                "task": task.get_config(),
                "frontier": frontier.get_config(),
                "captions_file": str(captions_file),
                "total_samples": len(eval_ids),
                "timestamp": datetime.now().isoformat(),
            },
            f,
            indent=4,
        )

    return compute_caption_metrics(out_dir, task)


def main():
    parser = argparse.ArgumentParser(description="Run caption-based spectra evaluation (Phase 2).")
    parser.add_argument("--task", type=str, required=True, help="Path to task YAML config.")
    parser.add_argument("--frontier", type=str, required=True, help="Path to frontier model YAML config.")
    parser.add_argument("--captions", type=str, required=True, help="Path to pre-generated captions.jsonl.")
    parser.add_argument("--limit", type=int, default=None, help="Sample limit.")
    parser.add_argument("--gemini-model", type=str, default=None, help="Override Gemini model name.")
    parser.add_argument("--output-dir", type=str, default=None, help="Custom output directory.")
    parser.add_argument("--suffix-tag", type=str, default=None, help="Optional suffix for the output directory name.")
    args = parser.parse_args()

    dotenv.load_dotenv()

    with open(args.task, "r") as f:
        task_config = yaml.safe_load(f)
    with open(args.frontier, "r") as f:
        frontier_config = yaml.safe_load(f)
    if args.gemini_model:
        frontier_config["gemini_model"] = args.gemini_model

    task_name = task_config.get("name", "task")
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        tag_suffix = f"_{args.suffix_tag}" if args.suffix_tag else ""
        timestamp_dir = datetime.now().strftime(f"%Y%m%d_%H%M%S_eval_{task_name}{tag_suffix}")
        output_dir = Path.cwd() / "eval_results" / "caption_eval" / timestamp_dir

    print(f"Starting caption evaluation: task={task_name}, output={output_dir}")
    metrics = run_caption_evaluation(
        task_config=task_config,
        frontier_config=frontier_config,
        captions_file=args.captions,
        output_dir=output_dir,
        limit=args.limit,
    )
    print(f"Evaluation complete. Results written to {output_dir}")


if __name__ == "__main__":
    main()
