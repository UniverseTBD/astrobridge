#!/usr/bin/env python
"""Cross-validate, retrain, and evaluate one frozen-representation head."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from captioner.utils.config import load_config, remaining_argv
from eval.benchmark_heads import train_and_evaluate_head
from eval.benchmark_protocol import ATCAT_REVISION, benchmark_protocol_fingerprint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", required=True, choices=("linear", "transformer"))
    parser.add_argument("--artifact-root", default="outputs/benchmark_v8")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260917)
    args = parser.parse_args(remaining_argv())
    root = Path(args.artifact_root)
    cfg = load_config("base", "data", "modalities", "model", "stage2")
    cfg.modalities.lightcurve.encoder.revision = ATCAT_REVISION
    report = train_and_evaluate_head(
        args.kind,
        root / "representations" / "development_qformer.pt",
        root / "representations" / "yse_qformer.pt",
        root / "heads" / args.kind,
        max_epochs=args.max_epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
        expected_protocol_fingerprint=benchmark_protocol_fingerprint(cfg),
    )
    print(json.dumps({"kind": args.kind, "yse_metrics": report["yse"]["metrics"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
