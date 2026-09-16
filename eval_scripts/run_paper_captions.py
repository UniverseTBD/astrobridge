#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
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
    write_run_metadata,
)


def load_suite_config(suite_arg: str) -> Dict[str, Any]:
    """Resolve and load suite YAML configuration."""
    suite_path = Path(suite_arg)
    if not suite_path.is_file():
        candidate = _REPO_ROOT / "eval_configs" / "suites" / f"{suite_arg}.yaml"
        if candidate.is_file():
            suite_path = candidate
        else:
            candidate_raw = _REPO_ROOT / "eval_configs" / "suites" / suite_arg
            if candidate_raw.is_file():
                suite_path = candidate_raw
            else:
                raise FileNotFoundError(
                    f"Suite configuration not found at '{suite_arg}' or '{candidate}'."
                )

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
    """Filter models from the suite based on requested model IDs and/or responder type."""
    selected = list(all_models)

    # Ensure responder_type is populated on each model
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
        selected = [
            m for m in selected if str(m.get("responder_type", "")).strip().lower() == rt_clean
        ]
        if not selected:
            valid_rts = sorted(
                {str(m.get("responder_type")) for m in all_models if m.get("responder_type")}
            )
            raise ValueError(
                f"No models found matching responder_type '{responder_type}'. Available in suite: {valid_rts}"
            )

    if requested_ids:
        valid_ids = {m["id"]: m for m in selected if "id" in m}
        filtered = []
        for mid in requested_ids:
            mid_clean = mid.strip()
            if not mid_clean:
                continue
            if mid_clean not in valid_ids:
                all_valid_ids = sorted({m["id"] for m in all_models if "id" in m})
                raise ValueError(
                    f"Unknown model ID '{mid_clean}'. Valid IDs in suite: {all_valid_ids}"
                )
            filtered.append(valid_ids[mid_clean])
        selected = filtered

    return selected


def execute_local_model(
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
    """Execute caption generation locally via subprocess for clean memory management."""
    script_path = _REPO_ROOT / "eval_scripts" / "generate_captions.py"
    cmd = [
        sys.executable,
        str(script_path),
        "--responder",
        str(model_config_path),
        "--output",
        str(output_path),
        "--benchmark",
        benchmark,
        "--batch-size",
        str(batch_size),
    ]

    if model_id is not None:
        cmd.extend(["--model-id", str(model_id)])
    if limit is not None:
        cmd.extend(["--limit", str(limit)])
    if max_tokens is not None:
        cmd.extend(["--max-tokens", str(max_tokens)])
    if fallback_max_tokens is not None:
        cmd.extend(["--fallback-max-tokens", str(fallback_max_tokens)])
    if repetition_penalty is not None:
        cmd.extend(["--repetition-penalty", str(repetition_penalty)])
    if device is not None:
        cmd.extend(["--device", device])
    if devices is not None:
        cmd.extend(["--devices", devices])
    if num_gpus is not None:
        cmd.extend(["--num-gpus", str(num_gpus)])

    print(f"\n[LOCAL EXECUTION] Running: {' '.join(shlex.quote(c) for c in cmd)}")
    subprocess.run(cmd, check=True)


def execute_modal_model(
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
    """Dispatch caption generation to Modal remote container and download results."""
    modal_script = _REPO_ROOT / "eval_scripts" / "run_modal.py"
    rel_config_path = Path(model_config_path).resolve().relative_to(_REPO_ROOT)
    output_filename = Path(output_path).name
    remote_output = f"eval_results/cached_captions/{output_filename}"

    inner_args = [
        "--responder",
        str(rel_config_path),
        "--output",
        remote_output,
        "--benchmark",
        benchmark,
        "--batch-size",
        str(batch_size),
    ]
    if model_id is not None:
        inner_args.extend(["--model-id", str(model_id)])
    if limit is not None:
        inner_args.extend(["--limit", str(limit)])
    if max_tokens is not None:
        inner_args.extend(["--max-tokens", str(max_tokens)])
    if fallback_max_tokens is not None:
        inner_args.extend(["--fallback-max-tokens", str(fallback_max_tokens)])
    if repetition_penalty is not None:
        inner_args.extend(["--repetition-penalty", str(repetition_penalty)])

    cmd = [
        "modal",
        "run",
        str(modal_script),
        "--script",
        "eval_scripts/generate_captions.py",
        "--args",
        " ".join(inner_args),
        "--remote-output",
        remote_output,
        "--output",
        str(output_path),
    ]

    env = os.environ.copy()
    if modal_gpu:
        env["MODAL_GPU"] = modal_gpu
        print(f"[MODAL EXECUTION] Setting MODAL_GPU={modal_gpu}")

    print(f"[MODAL EXECUTION] Running: {' '.join(shlex.quote(c) for c in cmd)}")
    subprocess.run(cmd, env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run evaluation caption generation across paper models and baseline controls."
    )
    parser.add_argument(
        "--suite",
        type=str,
        default="paper_controls",
        help="Path or name of suite YAML (default: paper_controls -> eval_configs/suites/paper_controls.yaml).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Specific model ID to run (e.g. astrobridge, hf_vision, hf_text).",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated model IDs to run (e.g. 'astrobridge,hf_vision').",
    )
    parser.add_argument(
        "--modal",
        action="store_true",
        help="Execute on Modal remote GPUs instead of local hardware.",
    )
    parser.add_argument(
        "--modal-gpu",
        type=str,
        default=None,
        help="Override Modal GPU type (e.g. A10G, A100-80GB, L4, T4).",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default=None,
        help="Benchmark dataset to run (overrides suite default, e.g. all, source, subclass).",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit number of spectra per model.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size per worker (overrides per-model batch_size from suite YAML).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="eval_results/cached_captions",
        help="Local directory to store output captions.jsonl files.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip generation if captions already exist locally, or download from HF Hub if available.",
    )
    parser.add_argument(
        "--no-pull-hf",
        action="store_true",
        help="When used with --skip-existing, only check local disk and do not check/download from HF Hub.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-generation even if output captions exist locally or on HF.",
    )
    parser.add_argument(
        "--push-to-hf",
        action="store_true",
        help="Upload generated caption files and run metadata to Hugging Face dataset repo.",
    )
    parser.add_argument(
        "--hf-repo",
        type=str,
        default=None,
        help="Override Hugging Face dataset repo (default from suite or UniverseTBD/AstroBridge-Data).",
    )
    parser.add_argument(
        "--hf-subdir",
        type=str,
        default=None,
        help="Override Hugging Face subdirectory (default from suite or evals/captions).",
    )
    parser.add_argument(
        "--responder-type",
        type=str,
        default=None,
        help="Filter models by responder type (e.g. hf_vision, hf_text, astrobridge).",
    )
    parser.add_argument("--device", type=str, default=None, help="Local device (e.g. cuda:0, mps, cpu).")
    parser.add_argument(
        "--devices",
        type=str,
        default=None,
        help="Local comma-separated devices (e.g. '0,1' or 'all').",
    )
    parser.add_argument("--num-gpus", type=int, default=None, help="Local number of GPUs.")
    args = parser.parse_args()

    dotenv.load_dotenv()

    suite_config = load_suite_config(args.suite)
    benchmark = args.benchmark or suite_config.get("benchmark", "all")
    hf_repo = args.hf_repo or suite_config.get("hf_repo", DEFAULT_HF_DATA_REPO)
    hf_subdir = args.hf_subdir or suite_config.get("hf_subdir", "evals/captions")

    requested_ids: Optional[List[str]] = None
    if args.models:
        requested_ids = [m.strip() for m in args.models.split(",") if m.strip()]
    elif args.model:
        requested_ids = [args.model.strip()]

    models_to_run = resolve_models_to_run(
        suite_config.get("models", []),
        requested_ids=requested_ids,
        responder_type=args.responder_type,
    )

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"==================================================")
    print(f"AstroBridge Paper Suite Runner: {suite_config.get('suite_name', 'unnamed')}")
    print(f"Models to execute: {[m['id'] for m in models_to_run]}")
    print(f"Benchmark: {benchmark}")
    print(f"Backend: {'Modal Cloud' if args.modal else 'Local'}")
    print(f"Output Directory: {output_dir}")
    print(f"Push to Hugging Face: {args.push_to_hf} ({hf_repo}:{hf_subdir})")
    print(f"==================================================\n")

    executed_models: List[str] = []

    for model in models_to_run:
        mid = model["id"]
        mname = model.get("name", mid)
        cfg_rel_path = model.get("config")
        if not cfg_rel_path:
            rt = model.get("responder_type")
            if not rt:
                raise ValueError(
                    f"Model '{mid}' must define either 'responder_type' or 'config'."
                )
            cfg_rel_path = f"eval_configs/responders/{rt}.yaml"
        cfg_abs_path = (_REPO_ROOT / cfg_rel_path).resolve()
        if not cfg_abs_path.is_file():
            raise FileNotFoundError(f"Model config file not found: {cfg_abs_path}")

        output_filename = model.get("output_filename") or f"{mid}.jsonl"
        local_output_path = output_dir / output_filename
        hf_file_path = f"{hf_subdir.rstrip('/')}/{output_filename}"

        print(f"\n>>> Processing [{mid}] ({mname})")

        skip = False
        if args.skip_existing and not args.force:
            if local_output_path.is_file() and local_output_path.stat().st_size > 0:
                print(f"[SKIP] Found existing local captions at {local_output_path}.")
                skip = True
            elif not args.no_pull_hf:
                print(f"Checking Hugging Face for existing captions ({hf_repo}:{hf_file_path})...")
                downloaded = check_and_download_existing(
                    hf_repo=hf_repo,
                    hf_path=hf_file_path,
                    local_path=local_output_path,
                )
                if downloaded:
                    print(f"[CACHE] Downloaded captions from Hugging Face to {local_output_path}.")
                    skip = True

        if not skip:
            model_batch_size = (
                args.batch_size if args.batch_size is not None else model.get("batch_size", 32)
            )
            model_id_override = (
                model.get("hf_model_id")
                or model.get("astrobridge_id")
                or model.get("model_id")
            )
            model_max_tokens = model.get("max_tokens")
            model_fallback_max_tokens = model.get("fallback_max_tokens")
            model_repetition_penalty = model.get("repetition_penalty")

            if args.modal:
                modal_gpu = args.modal_gpu or model.get("modal_gpu")
                execute_modal_model(
                    model_config_path=cfg_abs_path,
                    output_path=local_output_path,
                    benchmark=benchmark,
                    limit=args.limit,
                    batch_size=model_batch_size,
                    modal_gpu=modal_gpu,
                    model_id=model_id_override,
                    max_tokens=model_max_tokens,
                    fallback_max_tokens=model_fallback_max_tokens,
                    repetition_penalty=model_repetition_penalty,
                )
            else:
                execute_local_model(
                    model_config_path=cfg_abs_path,
                    output_path=local_output_path,
                    benchmark=benchmark,
                    limit=args.limit,
                    batch_size=model_batch_size,
                    device=args.device,
                    devices=args.devices,
                    num_gpus=args.num_gpus,
                    model_id=model_id_override,
                    max_tokens=model_max_tokens,
                    fallback_max_tokens=model_fallback_max_tokens,
                    repetition_penalty=model_repetition_penalty,
                )

        if not local_output_path.is_file():
            raise RuntimeError(f"Expected output file not found after execution: {local_output_path}")

        executed_models.append(mid)

        if args.push_to_hf:
            upload_caption_file(
                local_path=local_output_path,
                hf_repo=hf_repo,
                hf_path=hf_file_path,
                commit_message=f"Add evaluation captions for {mid} ({suite_config.get('suite_name', 'paper')})",
            )

    metadata_path = write_run_metadata(
        output_dir=output_dir,
        suite_config=suite_config,
        models_run=executed_models,
    )
    print(f"\nRun metadata written to {metadata_path}")

    if args.push_to_hf:
        hf_meta_path = f"{hf_subdir.rstrip('/')}/run_metadata.json"
        upload_caption_file(
            local_path=metadata_path,
            hf_repo=hf_repo,
            hf_path=hf_meta_path,
            commit_message=f"Update run metadata for suite {suite_config.get('suite_name', 'paper')}",
        )

    print("\nAll suite models completed successfully!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[ABORTED] Suite run interrupted by user.")
        sys.exit(130)

