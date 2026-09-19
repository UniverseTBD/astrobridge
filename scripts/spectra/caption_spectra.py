#!/usr/bin/env python
# /// script
# requires-python = ">=3.13"
# dependencies = [
#   "spectra-captioning @ git+https://github.com/ibrahimhabibeg/spectra-captioning.git@22c38c0c91aeaa1bcabe3c5e2fb06d82ae629a8f",
#   "omegaconf>=2.3",
# ]
# ///
"""Generate training captions using the pinned upstream spectra-captioning project."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

# uv runs this script in its own Python 3.13 environment. Only the lightweight
# data helpers are imported from AstroBridge; the training package is not installed.
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from captioner.data.spectra_captioning import STRATEGIES, generate_spectra_captions, prepare_groups
from captioner.data.spectra_dataset import _read_parquet, load_spectra_table
from captioner.utils.config import load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=_REPO_ROOT / "configs/spectra_captioning.yaml")
    parser.add_argument("--input", type=Path, help="Local merged spectra Parquet; defaults to sources.spectra in configs/data.yaml")
    parser.add_argument("--output", type=Path, help="Output JSONL, also accepted by make captions SPECTRA_CAPTIONS=...")
    parser.add_argument("--strategy", choices=STRATEGIES)
    parser.add_argument("--model", help="Gemini model ID")
    parser.add_argument("--limit", type=int, help="Maximum objects; 0 selects all (config default: 10)")
    parser.add_argument("--object-id", action="append", help="wiki_entity_id to include; repeat for multiple objects")
    parser.add_argument("--metadata-dir", type=Path, help="Directory containing extracted_types.csv and extracted_emission_lines.csv for combined_ground_truth_v1")
    parser.add_argument("--save-plots", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Skip completed objects after checking settings and input hashes")
    parser.add_argument("--dry-run", action="store_true", help="Validate and list selected objects without contacting Gemini or writing captions")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    from dotenv import load_dotenv
    from omegaconf import OmegaConf

    args = build_parser().parse_args(argv)
    load_dotenv(_REPO_ROOT / ".env")
    cfg = OmegaConf.merge(load_config("base", "data", argv=[]), OmegaConf.load(args.config))
    for key in ("model", "strategy", "limit", "metadata_dir"):
        value = getattr(args, key)
        if value is not None:
            cfg.captioning[key] = str(value) if isinstance(value, Path) else value
    if args.save_plots:
        cfg.captioning.save_plots = True
    if cfg.captioning.limit is not None and cfg.captioning.limit < 0:
        raise ValueError("--limit must be nonnegative (0 selects all objects)")
    if cfg.captioning.strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy {cfg.captioning.strategy!r}; choose from {STRATEGIES}")
    output_path = args.output or Path(cfg.captioning.output_file)
    if args.input:
        frame = _read_parquet(str(args.input))
    else:
        frame = load_spectra_table(
            cfg.sources.spectra.hf_path,
            revision=cfg.sources.spectra.get("revision"),
            files=list(cfg.sources.spectra.get("files") or []) or None,
        )
    groups = prepare_groups(frame, object_ids=args.object_id, limit=cfg.captioning.limit)
    if args.dry_run:
        print(json.dumps({
            "selected": len(groups), "object_ids": [key for key, _ in groups],
            "strategy": cfg.captioning.strategy, "model": cfg.captioning.model,
            "output": str(output_path),
        }, indent=2))
        return 0
    stats = generate_spectra_captions(
        groups, output_path, OmegaConf.to_container(cfg, resolve=True), resume=args.resume,
    )
    print(json.dumps({**stats, "output": str(output_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
