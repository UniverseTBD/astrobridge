#!/usr/bin/env python
"""Run one isolated, batched, resumable v8 generative benchmark route."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from captioner.utils.config import load_config, remaining_argv
from eval.benchmark_generation import run_generative_route
from eval.benchmark_protocol import ATCAT_REVISION, GENERATIVE_ROUTES


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--route", required=True, choices=GENERATIVE_ROUTES)
    parser.add_argument("--artifact-root", default="outputs/benchmark_v8")
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--model-path", required=True, help="Existing local snapshot; downloads are not allowed in route jobs.")
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(remaining_argv())

    cfg = load_config("base", "data", "modalities", "model", "stage2")
    cfg.modalities.lightcurve.encoder.revision = ATCAT_REVISION
    result = run_generative_route(
        route=args.route,
        artifact_root=Path(args.artifact_root),
        output_jsonl=Path(args.output_jsonl),
        cfg=cfg,
        model_path=args.model_path,
        batch_size=args.batch_size,
        device=args.device,
        limit=args.limit,
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
