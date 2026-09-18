#!/usr/bin/env python
"""SN-typing classification eval against `BuildNg/astrobridge-yse-test-dataset-v2` — a real,
held-out benchmark (zero object_id overlap with training, see `eval/datasets/lightcurve_yse.py`).

Two tracks (`--track`): `lightcurve_only` (works today, no blockers) and `lightcurve_plus_image`
(tests whether adding the host image improves SN-type classification — see `eval/datasets/
lightcurve_yse.py`'s `build_raw_inputs_with_image` for the one real caveat: host-image band
order/identity hasn't been separately verified against this dataset yet). No base-model
comparison — see `eval/backend.py`'s module docstring for why a plain out-of-the-box
vision-language model has no meaningful way to consume a raw lightcurve at all.

Usage:
    uv run python eval/runners/run_lightcurve_eval.py --track lightcurve_only
    uv run python eval/runners/run_lightcurve_eval.py --track lightcurve_plus_image --backend modal
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from captioner.utils.config import load_config, remaining_argv
from captioner.utils.logging import get_logger
from eval.backend import get_backend
from eval.datasets.lightcurve_yse import (
    SN_LABELS,
    build_raw_inputs_lightcurve,
    build_raw_inputs_with_image,
    load_host_image_table,
    load_lightcurve_table,
)
from eval.metrics.caption_to_label import SN_TYPE_SYNONYMS, predict_label
from eval.metrics.classification import classification_report

logger = get_logger(__name__)

DEFAULT_QUESTION = (
    "Classify this transient's supernova type based on its light curve. "
    "Choose exactly one: SN Ia, SN II, SN Ibc."
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="UniverseTBD/astrobridge-model-v5")
    parser.add_argument("--backend", choices=["local", "modal"], default="local")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--track", choices=["lightcurve_only", "lightcurve_plus_image"], default="lightcurve_only")
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None, help="evaluate only the first N objects (quick smoke test)")
    parser.add_argument("--out", default=None, help="defaults to outputs/eval/classification/yse_<track>.json")
    args = parser.parse_args(remaining_argv())

    cfg = load_config("base", "data", "modalities", "model", "stage2")

    lc_table = load_lightcurve_table()
    if args.track == "lightcurve_plus_image":
        image_table = load_host_image_table()
        lc_table = lc_table.merge(image_table, on=["object_id", "class_label"], how="inner", suffixes=("", "_img"))
        logger.info(f"{len(lc_table)} objects have both lightcurve and host-image data.")
    if args.limit is not None:
        lc_table = lc_table.head(args.limit)

    modality_names = ["image", "lightcurve"] if args.track == "lightcurve_plus_image" else ["lightcurve"]
    backend = get_backend(
        args.backend, side="equipped", cfg=cfg, repo_id=args.repo_id, device=args.device, modality_names=modality_names,
    )

    y_true: list[str] = []
    y_pred: list[str | None] = []
    for _, row in tqdm(lc_table.iterrows(), total=len(lc_table), desc=f"lightcurve eval [{args.track}]"):
        raw_inputs = (
            build_raw_inputs_with_image(row, row, cfg) if args.track == "lightcurve_plus_image"
            else build_raw_inputs_lightcurve(row, cfg)
        )
        caption = backend.generate(raw_inputs, args.question, args.max_new_tokens)
        y_true.append(row["class_label"])
        y_pred.append(predict_label(caption, SN_LABELS, SN_TYPE_SYNONYMS))

    report = {
        "dataset": "BuildNg/astrobridge-yse-test-dataset-v2",
        "track": args.track,
        "repo_id": args.repo_id,
        "equipped": classification_report(y_true, y_pred, SN_LABELS),
    }

    out_path = Path(args.out) if args.out else Path(f"outputs/eval/classification/yse_{args.track}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    logger.info(json.dumps(report, indent=2))
    logger.info(f"Wrote report to {out_path}")


if __name__ == "__main__":
    main()
