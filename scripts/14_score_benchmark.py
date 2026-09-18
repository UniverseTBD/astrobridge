#!/usr/bin/env python
"""Score every completed generative benchmark route."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from captioner.utils.config import load_config, remaining_argv
from eval.benchmark_artifacts import load_yse_manifest, manifest_records_by_id
from eval.benchmark_generation import validate_generation_records
from eval.benchmark_protocol import (
    ATCAT_REVISION,
    GENERATIVE_ROUTES,
    JsonlLedger,
    atomic_write_json,
    benchmark_protocol_fingerprint,
    benchmark_metrics,
    require_successful_generation_records,
    stratified_bootstrap_intervals,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", default="outputs/benchmark_v8")
    args = parser.parse_args(remaining_argv())
    root = Path(args.artifact_root)
    cfg = load_config("base", "data", "modalities", "model", "stage2")
    cfg.modalities.lightcurve.encoder.revision = ATCAT_REVISION
    protocol_fingerprint = benchmark_protocol_fingerprint(cfg)
    manifest = load_yse_manifest(root, expected_protocol_fingerprint=protocol_fingerprint)
    manifest_by_id = manifest_records_by_id(manifest)
    ordered_ids = [str(record["object_id"]) for record in manifest["objects"]]
    reports = {}
    for route in GENERATIVE_ROUTES:
        path = root / "generations" / f"{route}.jsonl"
        latest = JsonlLedger(path).latest()
        extras = sorted(set(latest) - set(ordered_ids))
        if extras:
            raise ValueError(f"route={route} contains {len(extras)} objects outside the manifest")
        missing = [object_id for object_id in ordered_ids if object_id not in latest]
        if missing:
            raise ValueError(f"route={route} is missing {len(missing)} objects; first={missing[:5]}")
        records = [latest[object_id] for object_id in ordered_ids]
        validate_generation_records(latest, route, manifest_by_id, root, protocol_fingerprint)
        require_successful_generation_records(records, route)
        truth = [str(record["true_label"]) for record in records]
        predictions = [record.get("parsed_label") for record in records]
        reports[route] = benchmark_metrics(truth, predictions)
        reports[route]["bootstrap_95_intervals"] = stratified_bootstrap_intervals(truth, predictions)
        reports[route]["protocol_fingerprint"] = protocol_fingerprint
        runtimes = [float(record.get("runtime_seconds", 0.0)) for record in records]
        reports[route]["runtime"] = {
            "sum_amortized_seconds": sum(runtimes),
            "mean_amortized_seconds_per_object": sum(runtimes) / len(runtimes),
            "failed_records": 0,
        }
    metrics_dir = root / "metrics"
    atomic_write_json(metrics_dir / "generative_metrics.json", reports)
    print(json.dumps({"routes_scored": len(reports)}), flush=True)


if __name__ == "__main__":
    main()
