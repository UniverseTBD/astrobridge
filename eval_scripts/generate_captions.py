from __future__ import annotations

import argparse
import json
import os
import queue
import sys
from pathlib import Path
from typing import Any, Optional
import dotenv
import yaml
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from evals.data import load_all_benchmark_spectra, load_benchmark_dataset
from evals.responders import SpectrumSample, get_responder


def resolve_devices(
    device: Optional[str] = None,
    devices: Optional[list[str] | str] = None,
    num_gpus: Optional[int] = None,
) -> list[str]:
    """Resolve device arguments into a normalized list of device strings.

    Precedence:
    1. `devices` (comma-separated string or list)
    2. `num_gpus` (integer -> [cuda:0, ..., cuda:N-1])
    3. `device` (single device string)
    4. Auto-detect: all available CUDA devices if cuda is available and device_count > 1;
       cuda:0 if device_count == 1; mps if available; else cpu.
    """
    import torch

    if devices is not None:
        if isinstance(devices, str):
            raw_list = [d.strip() for d in devices.split(",") if d.strip()]
        else:
            raw_list = [str(d).strip() for d in devices if str(d).strip()]

        if len(raw_list) == 1 and raw_list[0].lower() == "all":
            count = torch.cuda.device_count() if torch.cuda.is_available() else 0
            if count > 0:
                return [f"cuda:{i}" for i in range(count)]
            return ["cpu"]

        resolved = []
        for d in raw_list:
            if d.isdigit():
                resolved.append(f"cuda:{d}")
            elif d.startswith("cuda:") or d in ("cuda", "mps", "cpu"):
                resolved.append(d)
            else:
                resolved.append(d)
        return resolved

    if num_gpus is not None:
        if num_gpus <= 0:
            raise ValueError(f"num_gpus must be positive, got {num_gpus}")
        return [f"cuda:{i}" for i in range(num_gpus)]

    if device is not None:
        return [device]

    # Auto-detection
    if torch.cuda.is_available():
        count = torch.cuda.device_count()
        if count > 1:
            return [f"cuda:{i}" for i in range(count)]
        return ["cuda:0"]
    elif torch.backends.mps.is_available():
        return ["mps"]
    else:
        return ["cpu"]


def _run_caption_generation_single_device(
    responder_config: dict[str, Any],
    output_path: Path,
    rows: list[dict[str, Any]],
    batch_size: int = 32,
    caption_prompt: Optional[str] = None,
    device: str = "cpu",
) -> int:
    responder = get_responder(responder_config, device)
    total_generated = 0

    with open(output_path, "w") as f:
        for i in tqdm(range(0, len(rows), batch_size), desc="Generating Captions"):
            batch_rows = rows[i : i + batch_size]
            samples = [SpectrumSample.from_row(r) for r in batch_rows]
            captions = responder.generate_captions(samples, prompt_override=caption_prompt)
            for c in captions:
                record = {
                    "sample_id": c.sample_id,
                    "survey": c.survey,
                    "caption": c.caption,
                    "responder_type": c.responder_type,
                    "model_id": c.model_id,
                    "caption_prompt": c.caption_prompt,
                }
                f.write(json.dumps(record) + "\n")
                total_generated += 1
            f.flush()

    return total_generated


def _caption_worker_loop(
    rank: int,
    device_str: str,
    responder_config: dict[str, Any],
    caption_prompt: Optional[str],
    rows: list[dict[str, Any]],
    task_queue: Any,
    result_queue: Any,
) -> None:
    try:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        import torch

        if device_str.startswith("cuda"):
            device_idx = int(device_str.split(":")[1]) if ":" in device_str else rank
            torch.cuda.set_device(device_idx)

        from evals.responders import SpectrumSample, get_responder

        responder = get_responder(responder_config, device_str)

        while True:
            task = task_queue.get()
            if task is None:
                break
            batch_idx, start_idx, end_idx = task
            batch_rows = rows[start_idx:end_idx]
            samples = [SpectrumSample.from_row(r) for r in batch_rows]
            captions = responder.generate_captions(samples, prompt_override=caption_prompt)
            records = [
                {
                    "sample_id": c.sample_id,
                    "survey": c.survey,
                    "caption": c.caption,
                    "responder_type": c.responder_type,
                    "model_id": c.model_id,
                    "caption_prompt": c.caption_prompt,
                }
                for c in captions
            ]
            result_queue.put(("SUCCESS", batch_idx, records))
    except Exception as exc:
        import traceback

        result_queue.put(
            (
                "ERROR",
                rank,
                f"Worker {rank} ({device_str}) encountered error: {exc}\n{traceback.format_exc()}",
            )
        )


def _run_caption_generation_multi_gpu(
    devices: list[str],
    responder_config: dict[str, Any],
    output_path: Path,
    rows: list[dict[str, Any]],
    batch_size: int = 32,
    caption_prompt: Optional[str] = None,
) -> int:
    import torch.multiprocessing as mp

    ctx = mp.get_context("spawn")
    total_samples = len(rows)
    if total_samples == 0:
        output_path.write_text("")
        return 0

    num_batches = (total_samples + batch_size - 1) // batch_size
    task_queue = ctx.Queue()
    result_queue = ctx.Queue()

    for b in range(num_batches):
        start_idx = b * batch_size
        end_idx = min(total_samples, (b + 1) * batch_size)
        task_queue.put((b, start_idx, end_idx))

    for _ in devices:
        task_queue.put(None)

    workers = []
    for rank, device_str in enumerate(devices):
        p = ctx.Process(
            target=_caption_worker_loop,
            args=(rank, device_str, responder_config, caption_prompt, rows, task_queue, result_queue),
            daemon=True,
        )
        p.start()
        workers.append(p)

    completed_batches: dict[int, list[dict[str, Any]]] = {}
    next_write_batch = 0
    total_generated = 0

    try:
        with open(output_path, "w") as f, tqdm(
            total=total_samples, desc=f"Generating Captions ({len(devices)} Devices)"
        ) as pbar:
            while total_generated < total_samples:
                # Check for dead workers
                for p in workers:
                    if not p.is_alive() and p.exitcode not in (0, None):
                        raise RuntimeError(
                            f"Worker process {p.pid} terminated unexpectedly with exit code {p.exitcode}"
                        )

                try:
                    msg = result_queue.get(timeout=0.5)
                except queue.Empty:
                    continue

                status = msg[0]
                if status == "ERROR":
                    _, worker_rank, err_detail = msg
                    raise RuntimeError(f"Multi-device caption worker error:\n{err_detail}")

                _, batch_idx, records = msg
                completed_batches[batch_idx] = records
                pbar.update(len(records))

                # Flush in sequential order
                while next_write_batch in completed_batches:
                    recs = completed_batches.pop(next_write_batch)
                    for rec in recs:
                        f.write(json.dumps(rec) + "\n")
                        total_generated += 1
                    f.flush()
                    next_write_batch += 1
    finally:
        for p in workers:
            if p.is_alive():
                p.join(timeout=1.0)
                if p.is_alive():
                    p.terminate()

    return total_generated


def run_caption_generation(
    responder_config: dict[str, Any],
    output_path: str | Path,
    benchmark: str = "all",
    limit: Optional[int] = None,
    batch_size: int = 32,
    caption_prompt: Optional[str] = None,
    device: Optional[str] = None,
    devices: Optional[list[str] | str] = None,
    num_gpus: Optional[int] = None,
    df: Optional[Any] = None,
) -> int:
    resolved_devices = resolve_devices(device=device, devices=devices, num_gpus=num_gpus)

    if df is not None:
        df_test = df
    elif benchmark == "all":
        df_test = load_all_benchmark_spectra()
    else:
        df_test = load_benchmark_dataset(benchmark)

    if limit is not None:
        df_test = df_test.head(limit)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = df_test.to_dict(orient="records")

    if len(resolved_devices) == 1:
        return _run_caption_generation_single_device(
            responder_config=responder_config,
            output_path=output_path,
            rows=rows,
            batch_size=batch_size,
            caption_prompt=caption_prompt,
            device=resolved_devices[0],
        )
    else:
        return _run_caption_generation_multi_gpu(
            devices=resolved_devices,
            responder_config=responder_config,
            output_path=output_path,
            rows=rows,
            batch_size=batch_size,
            caption_prompt=caption_prompt,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and cache captions for benchmark spectra.")
    parser.add_argument("--responder", type=str, required=True, help="Path to responder YAML config.")
    parser.add_argument("--output", type=str, required=True, help="Output path for captions.jsonl.")
    parser.add_argument(
        "--benchmark",
        type=str,
        default="all",
        choices=["all", "distance", "source", "subclass", "emission_lines"],
        help="Benchmark dataset to generate captions for (default: all).",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit number of spectra.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument("--caption-prompt", type=str, default=None, help="Optional caption prompt override.")
    parser.add_argument("--device", type=str, default=None, help="Single device to use (e.g. cuda, cuda:0, mps, cpu).")
    parser.add_argument(
        "--devices",
        type=str,
        default=None,
        help="Comma-separated devices to use (e.g. '0,1', 'cuda:0,cuda:1', or 'all').",
    )
    parser.add_argument("--num-gpus", type=int, default=None, help="Number of GPUs to use (cuda:0..cuda:N-1).")
    parser.add_argument(
        "--model-id",
        type=str,
        default=None,
        help="Override model checkpoint (hf_model_id for HF responders, or astrobridge_id for AstroBridge).",
    )
    parser.add_argument("--max-tokens", type=int, default=None, help="Override max_tokens.")
    parser.add_argument("--fallback-max-tokens", type=int, default=None, help="Override fallback_max_tokens.")
    parser.add_argument("--repetition-penalty", type=float, default=None, help="Override repetition_penalty.")
    parser.add_argument("--push-to-hf", action="store_true", help="Upload resulting captions.jsonl to Hugging Face Hub.")
    parser.add_argument("--hf-repo", type=str, default=None, help="Hugging Face repo ID (default: UniverseTBD/AstroBridge-Data).")
    parser.add_argument("--hf-path", type=str, default=None, help="Target path in HF repo (default: evals/captions/<filename>).")
    args = parser.parse_args()

    dotenv.load_dotenv()

    with open(args.responder, "r") as f:
        responder_config = yaml.safe_load(f)

    if args.model_id:
        if responder_config.get("responder_type") == "astrobridge":
            responder_config["astrobridge_id"] = args.model_id
        else:
            responder_config["hf_model_id"] = args.model_id

    if args.caption_prompt:
        responder_config["caption_prompt"] = args.caption_prompt
    if args.max_tokens is not None:
        responder_config["max_tokens"] = args.max_tokens
    if args.fallback_max_tokens is not None:
        responder_config["fallback_max_tokens"] = args.fallback_max_tokens
    if args.repetition_penalty is not None:
        responder_config["repetition_penalty"] = args.repetition_penalty

    total = run_caption_generation(
        responder_config=responder_config,
        output_path=args.output,
        benchmark=args.benchmark,
        limit=args.limit,
        batch_size=args.batch_size,
        caption_prompt=args.caption_prompt,
        device=args.device,
        devices=args.devices,
        num_gpus=args.num_gpus,
    )
    print(f"Generated {total} captions to {args.output}")

    if args.push_to_hf:
        from evals.hub import upload_caption_file

        upload_caption_file(
            local_path=args.output,
            hf_repo=args.hf_repo,
            hf_path=args.hf_path,
        )


if __name__ == "__main__":
    main()
