#!/usr/bin/env python
"""Cache frozen `(64, 384)` v8 Q-former tensors without loading Qwen or LoRA."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from captioner.data.transients_dataset import load_transients_table
from captioner.utils.config import load_config, remaining_argv
from eval.benchmark_artifacts import load_yse_manifest
from eval.benchmark_protocol import (
    ATCAT_REVISION,
    DEVELOPMENT_DATASET_ID,
    DEVELOPMENT_REVISION,
    benchmark_protocol_fingerprint,
)
from eval.benchmark_representations import extract_and_save_representations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", default="outputs/benchmark_v8")
    parser.add_argument("--dataset-cache-dir", default=None)
    parser.add_argument("--hf-cache-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(remaining_argv())
    root = Path(args.artifact_root)
    output_dir = root / "representations"
    cfg = load_config("base", "data", "modalities", "model", "stage2")
    cfg.modalities.lightcurve.encoder.revision = ATCAT_REVISION
    manifest = load_yse_manifest(
        root, expected_protocol_fingerprint=benchmark_protocol_fingerprint(cfg),
    )
    yse = pd.read_parquet(root / manifest["rows_path"])
    development = load_transients_table(
        DEVELOPMENT_DATASET_ID,
        revision=DEVELOPMENT_REVISION,
        cache_dir=Path(args.dataset_cache_dir) if args.dataset_cache_dir else None,
    )
    if len(development) != 987:
        raise ValueError(f"development dataset must contain 987 unique objects, got {len(development)}")

    result = extract_and_save_representations(
        development,
        yse,
        cfg,
        output_dir=output_dir,
        cache_dir=Path(args.hf_cache_dir) if args.hf_cache_dir else None,
        batch_size=args.batch_size,
        device=args.device,
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
