"""Adapter for ibrahimhabibeg/spectra-captioning's strategies and JSONL output."""
from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from captioner.data.spectra_dataset import _canonical_object_id

UPSTREAM_REVISION = "22c38c0c91aeaa1bcabe3c5e2fb06d82ae629a8f"
STRATEGIES = ("quotes_only_v3", "spectra_image_v3", "combined_v2", "combined_ground_truth_v1")
METADATA_FILES = ("extracted_types.csv", "extracted_emission_lines.csv")


def _plain(value: Any) -> Any:
    """Normalize Parquet's nested NumPy arrays for upstream quote handling and hashing."""
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if value is None or value is pd.NA or (isinstance(value, float) and not math.isfinite(value)):
        return None
    return value


def _digest(value: Any) -> str:
    payload = json.dumps(_plain(value), sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def prepare_groups(
    frame: pd.DataFrame,
    *,
    object_ids: list[str] | None = None,
    limit: int | None = None,
) -> list[tuple[str, pd.DataFrame]]:
    """Select whole objects, preserving all of each object's literature mentions."""
    if limit is not None and limit < 0:
        raise ValueError("limit must be nonnegative (0 selects all objects)")
    required = {"wiki_entity_id", "survey", "spectrum"}
    if missing := required.difference(frame.columns):
        raise ValueError(f"Spectra input is missing columns: {sorted(missing)}")
    frame = frame.copy()
    if frame["wiki_entity_id"].isna().any():
        raise ValueError("Every spectrum needs a wiki_entity_id for caption-to-training joins")
    frame["wiki_entity_id"] = frame["wiki_entity_id"].map(_canonical_object_id)
    if frame["wiki_entity_id"].eq("").any():
        raise ValueError("Every spectrum needs a nonempty wiki_entity_id")
    frame["survey"] = frame["survey"].map(lambda value: str(value).strip().lower())
    if not frame["survey"].isin(["sdss", "desi"]).all():
        raise ValueError("Spectra surveys must be sdss or desi")
    for name in ("evidence_quotes", "evidence_quotes.quote"):
        if name in frame:
            frame[name] = frame[name].map(_plain)
    if object_ids:
        requested = {_canonical_object_id(value) for value in object_ids}
        if missing := requested.difference(frame["wiki_entity_id"]):
            raise ValueError(f"Unknown wiki_entity_id values: {sorted(missing)}")
        frame = frame[frame["wiki_entity_id"].isin(requested)]
    groups = [(str(key), group) for key, group in frame.groupby("wiki_entity_id", sort=True)]
    if limit:
        groups = groups[:limit]
    if not groups:
        raise ValueError("No spectra objects selected")
    return groups


def _metadata_dir(config: dict) -> Path | None:
    if config["captioning"]["strategy"] != "combined_ground_truth_v1":
        return None
    directory = config["captioning"].get("metadata_dir")
    if not directory:
        raise ValueError("combined_ground_truth_v1 requires --metadata-dir with extracted metadata CSVs")
    directory = Path(directory)
    for filename in METADATA_FILES:
        if not (directory / filename).is_file():
            raise FileNotFoundError(directory / filename)
    return directory


def _build_strategy(config: dict, client: Any):
    # Import only captioning modules: the upstream CLI also imports unrelated
    # survey-query and extraction tools, which this integration does not call.
    for module in ("quotes_only", "spectra_image", "combined", "combined_ground_truth"):
        importlib.import_module(f"spectra_captioning.strategies.{module}")
    from spectra_captioning.strategies.base import get_strategy

    name = config["captioning"]["strategy"]
    strategy_class = get_strategy(name)
    directory = _metadata_dir(config)
    if directory is None:
        return strategy_class(gemini_client=client)

    # At the pinned upstream revision, these CSVs are located relative to its
    # source checkout. Redirect the constructor's lookup to our explicit input
    # directory; do not write into an installed package or silently omit metadata.
    module = importlib.import_module("spectra_captioning.strategies.combined_ground_truth")
    original_dir = module._DATA_DIR
    try:
        module._DATA_DIR = directory
        return strategy_class(gemini_client=client)
    finally:
        module._DATA_DIR = original_dir


def _run_hash(config: dict) -> str:
    captioning = config["captioning"]
    metadata_dir = _metadata_dir(config)
    metadata = {
        name: hashlib.sha256((metadata_dir / name).read_bytes()).hexdigest()
        for name in METADATA_FILES
    } if metadata_dir is not None else {}
    return _digest({
        "upstream_revision": UPSTREAM_REVISION,
        "strategy": captioning["strategy"],
        "model": captioning["model"],
        "thinking_level": captioning.get("thinking_level", "low"),
        "thinking_summaries": captioning.get("thinking_summaries", "auto"),
        "crossmatch": config["crossmatch"],
        "metadata": metadata,
    })


def _read_completed(path: Path, run_hash: str) -> dict[str, dict]:
    completed: dict[str, dict] = {}
    with path.open(encoding="utf-8") as source:
        for number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                key = record["object_key"]
                caption = record["output"]["caption"]
                provenance = record["provenance"]
                if not isinstance(key, str) or not key or not isinstance(caption, str) or not caption.strip():
                    raise ValueError("missing object_key or caption")
                if key in completed:
                    raise ValueError(f"duplicate object_key {key!r}")
                if provenance.get("astrobridge_run_sha256") != run_hash:
                    raise ValueError("generation settings differ; use a separate output file")
                completed[key] = record
            except (ValueError, KeyError, TypeError, AttributeError) as exc:
                raise ValueError(f"Cannot resume {path}:{number}: {exc}") from exc
    return completed


def generate_spectra_captions(
    groups: list[tuple[str, pd.DataFrame]],
    output_path: str | Path,
    config: dict,
    *,
    resume: bool = False,
    client: Any = None,
) -> dict[str, int]:
    """Run upstream strategies, preserving their output format and recording input hashes.

    Each completed result is flushed immediately. Failed objects stop the run;
    --resume reuses completed results only when inputs and generation settings match.
    A supplied client supports offline integration testing with the real strategies.
    """
    path = Path(output_path)
    if path.exists() and not resume:
        raise FileExistsError(f"{path} already exists; use --resume or choose another --output")
    if not groups:
        raise ValueError("No spectra objects selected")
    captioning = config["captioning"]
    if captioning["strategy"] not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {captioning['strategy']}")
    run_hash = _run_hash(config)
    completed = _read_completed(path, run_hash) if path.exists() else {}
    input_hashes = {key: _digest(group.to_dict(orient="records")) for key, group in groups}
    for key, _ in groups:
        if key in completed and completed[key]["provenance"].get("astrobridge_input_sha256") != input_hashes[key]:
            raise ValueError(f"Input changed for {key}; use a separate output file")
    pending = [(key, group) for key, group in groups if key not in completed]
    stats = {"selected": len(groups), "skipped": len(groups) - len(pending), "generated": 0, "insufficient": 0}
    if not pending:
        return stats

    from spectra_captioning.models.gemini import GeminiClient
    from spectra_captioning.output import build_output_record

    owns_client = client is None
    if owns_client:
        api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("Set GEMINI_API_KEY before generating spectra captions")
        client = GeminiClient(
            api_key=api_key,
            model=captioning["model"],
            thinking_level=captioning.get("thinking_level", "low"),
            thinking_summaries=captioning.get("thinking_summaries", "auto"),
        )
    try:
        strategy = _build_strategy(config, client)
        upstream_config = {**config, "captioning": {**captioning, "output_dir": str(path.parent)}}
        path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if resume else "x"
        with path.open(mode, encoding="utf-8") as target:
            # A valid last JSON record may lack a trailing newline (e.g. an externally
            # copied file). Separate it from the first resumed record.
            if mode == "a" and path.stat().st_size:
                with path.open("rb") as existing:
                    existing.seek(-1, os.SEEK_END)
                    if existing.read(1) != b"\n":
                        target.write("\n")
            for key, group in pending:
                result = strategy.generate_caption(key, group, "merged", upstream_config)
                if not result.caption.strip():
                    raise RuntimeError(f"Empty caption for {key}; rerun with --resume to retry")
                record = build_output_record(
                    object_key=key, group_df=group, result=result,
                    strategy_name=strategy.strategy_name, model=captioning["model"],
                    dataset="merged", config=upstream_config,
                )
                record["provenance"].update({
                    "spectra_captioning_revision": UPSTREAM_REVISION,
                    "astrobridge_run_sha256": run_hash,
                    "astrobridge_input_sha256": input_hashes[key],
                })
                target.write(json.dumps(record, ensure_ascii=False) + "\n")
                target.flush()
                stats["generated"] += 1
                stats["insufficient"] += int(record["output"]["is_insufficient"])
                print(f"[{stats['generated']}/{len(pending)}] {key}", flush=True)
    finally:
        if owns_client:
            client.client.close()
    return stats
