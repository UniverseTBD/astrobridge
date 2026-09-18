#!/usr/bin/env python
"""Prepare the immutable YSE manifest, tables, charts, and input checksums once."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from captioner.utils.config import load_config, remaining_argv
from eval.datasets.lightcurve_yse import load_lightcurve_table
from eval.benchmark_artifacts import load_yse_manifest, manifest_path, prepare_yse_artifacts
from eval.benchmark_protocol import ATCAT_REVISION, YSE_REVISION, benchmark_protocol_fingerprint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", default="outputs/benchmark_v8")
    parser.add_argument("--dataset-cache-dir", default=None)
    args = parser.parse_args(remaining_argv())
    root = Path(args.artifact_root)
    cfg = load_config("base", "data", "modalities", "model", "stage2")
    cfg.modalities.lightcurve.encoder.revision = ATCAT_REVISION
    if manifest_path(root).exists():
        manifest = load_yse_manifest(
            root, expected_protocol_fingerprint=benchmark_protocol_fingerprint(cfg),
        )
        print(json.dumps({"status": "reused", "objects": len(manifest["objects"])}), flush=True)
        return

    table = load_lightcurve_table(
        revision=YSE_REVISION,
        cache_dir=Path(args.dataset_cache_dir) if args.dataset_cache_dir else None,
    )
    manifest = prepare_yse_artifacts(table, root, cfg)
    print(json.dumps({"status": "created", "objects": len(manifest["objects"])}), flush=True)


if __name__ == "__main__":
    main()
