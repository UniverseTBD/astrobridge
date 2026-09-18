"""One-time, reusable input artifacts for the AstroBridge v8 benchmark."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pandas as pd

from eval.datasets.lightcurve_yse import build_raw_inputs_lightcurve, render_lightcurve_plot
from eval.benchmark_protocol import (
    ATCAT_REVISION,
    PROTOCOL_SCHEMA_VERSION,
    YSE_DATASET_ID,
    YSE_REVISION,
    atomic_write_json,
    benchmark_protocol_contract,
    benchmark_protocol_fingerprint,
    fingerprint_contract,
    numerical_table,
    sha256_bytes,
    sha256_file,
)


MANIFEST_NAME = "yse_manifest.json"
ROWS_NAME = "yse_rows.parquet"
PROTOCOL_NAME = "protocol.json"


def manifest_path(root: Path) -> Path:
    return root / "manifests" / MANIFEST_NAME


def load_yse_manifest(root: Path, *, expected_protocol_fingerprint: str | None = None) -> dict:
    path = manifest_path(root)
    payload = json.loads(path.read_text(encoding="utf-8"))
    protocol_path = root / "manifests" / PROTOCOL_NAME
    if not protocol_path.exists():
        raise ValueError("prepared YSE manifest is missing its frozen protocol.json")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    contract = protocol.get("contract")
    if protocol.get("schema_version") != PROTOCOL_SCHEMA_VERSION or not isinstance(contract, dict):
        raise ValueError("prepared protocol.json has an incompatible schema")
    calculated = fingerprint_contract(contract)
    recorded = protocol.get("protocol_fingerprint")
    if recorded != calculated:
        raise ValueError("protocol.json fingerprint does not match its canonical contract")
    if payload.get("protocol_fingerprint") != recorded:
        raise ValueError("YSE manifest and protocol.json fingerprints differ")
    if expected_protocol_fingerprint is not None and recorded != expected_protocol_fingerprint:
        raise ValueError(
            f"prepared protocol fingerprint {recorded!r} differs from current {expected_protocol_fingerprint!r}"
        )
    expected = {
        "dataset_id": YSE_DATASET_ID,
        "dataset_revision": YSE_REVISION,
        "atcat_execution_revision": ATCAT_REVISION,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"prepared manifest has {key}={payload.get(key)!r}, expected {value!r}")
    if len(payload.get("objects", [])) != 266:
        raise ValueError("prepared YSE manifest must contain exactly 266 objects")
    return payload


def _v8_input_checksum(row, cfg) -> str:
    digest = hashlib.sha256()
    raw = build_raw_inputs_lightcurve(row, cfg)["lightcurve"]
    for name in sorted(raw):
        array = raw[name].detach().cpu().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(tuple(array.shape)).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def prepare_yse_artifacts(table: pd.DataFrame, root: Path, cfg) -> dict:
    """Materialize the manifest, local source rows, tables, and charts exactly once.

    A completed compatible manifest is immutable and returned without rendering or hashing any
    inputs again.  An absent manifest means preparation is incomplete; existing partial files may
    be overwritten with deterministic content before the manifest is atomically committed.
    """
    protocol_fingerprint = benchmark_protocol_fingerprint(cfg)
    if manifest_path(root).exists():
        return load_yse_manifest(root, expected_protocol_fingerprint=protocol_fingerprint)

    if len(table) != 266:
        raise ValueError(f"YSE benchmark requires 266 objects, got {len(table)}")
    if table["object_id"].astype(str).duplicated().any():
        raise ValueError("YSE object_id values must be unique")
    counts = table["class_label"].value_counts().to_dict()
    expected_counts = {"SN Ia": 180, "SN II": 71, "SN Ibc": 15}
    if counts != expected_counts:
        raise ValueError(f"unexpected YSE class counts: {counts}; expected {expected_counts}")

    manifest_dir = root / "manifests"
    table_dir = root / "inputs" / "tables"
    chart_dir = root / "inputs" / "charts"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    chart_dir.mkdir(parents=True, exist_ok=True)

    rows_path = manifest_dir / ROWS_NAME
    rows_tmp = rows_path.with_suffix(".parquet.tmp")
    table.to_parquet(rows_tmp, index=False)
    os.replace(rows_tmp, rows_path)

    records = []
    for index, (_, row) in enumerate(table.reset_index(drop=True).iterrows()):
        stem = f"{index:04d}"
        table_path = table_dir / f"{stem}.txt"
        chart_path = chart_dir / f"{stem}.png"

        table_text = numerical_table(row) + "\n"
        table_path.write_text(table_text, encoding="utf-8", newline="\n")
        image = render_lightcurve_plot(row, x_axis_label="TIME_DAYS since first retained observation")
        image.save(chart_path, format="PNG")
        image.close()

        records.append({
            "index": index,
            "object_id": str(row["object_id"]),
            "true_label": str(row["class_label"]),
            "table_path": table_path.relative_to(root).as_posix(),
            "table_sha256": sha256_bytes(table_text.encode("utf-8")),
            "chart_path": chart_path.relative_to(root).as_posix(),
            "chart_sha256": sha256_file(chart_path),
            "v8_input_sha256": _v8_input_checksum(row, cfg),
        })

    payload = {
        "schema_version": 1,
        "protocol_fingerprint": protocol_fingerprint,
        "dataset_id": YSE_DATASET_ID,
        "dataset_revision": YSE_REVISION,
        "atcat_execution_revision": ATCAT_REVISION,
        "atcat_provenance_note": (
            "Execution-compatible with v8; the published v8 files do not prove that this was "
            "the ATCAT revision used during original training."
        ),
        "rows_path": rows_path.relative_to(root).as_posix(),
        "rows_sha256": sha256_file(rows_path),
        "class_counts": expected_counts,
        "objects": records,
    }
    atomic_write_json(root / "manifests" / PROTOCOL_NAME, {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "protocol_fingerprint": protocol_fingerprint,
        "contract": benchmark_protocol_contract(cfg),
        "checkpoint_compatibility": {
            "training_code_sha": "392fcf007fd11d12d04ffd13868d0c19dc25d190",
            "recorded_configuration_hash": "33ed49969a8fce7d",
            "kasa_config": None,
            "kasa_note": "Published adapter_config.json sets kasa_config to null; its ignored-key warning is harmless.",
        },
    })
    atomic_write_json(manifest_path(root), payload)
    return payload


def manifest_records_by_id(manifest: dict) -> dict[str, dict]:
    records = {str(record["object_id"]): record for record in manifest["objects"]}
    if len(records) != len(manifest["objects"]):
        raise ValueError("manifest contains duplicate object_id values")
    return records
