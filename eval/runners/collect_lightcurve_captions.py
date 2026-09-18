#!/usr/bin/env python
"""Plain-captioning collection over the YSE lightcurve test set, one side at a time — the
lightcurve counterpart to `collect_image_captions.py`. No classification framing, no digit/logprob
scoring: each model captions in its own natural voice, at whatever length it naturally produces.

`--side equipped` (default): the real trained `atcat_*` lightcurve arrays via
`build_raw_inputs_lightcurve`, the trained chat-template prompt with its DEFAULT system and
instruction (`question=None`/`system=None` fall back to `configs/model.yaml`'s
`system_variants[0]`/`instruction_variants[0]`).

`--side base`: `render_lightcurve_plot`'s rendered flux-vs-time PNG (a raw lightcurve array isn't
something an out-of-the-box vision-language model can consume directly — see
`eval.datasets.lightcurve_yse`'s module docstring), a plain "describe this plot" instruction, no
classification framing.

Deliberately writes each side to its OWN file — pass `--out` explicitly, or rely on the
side-specific default (`yse_captions.json` / `yse_captions_base.json`). No sampling by default:
the full 266-object YSE test set, for real statistical power once scored.

Usage:
    uv run python -m eval.runners.collect_lightcurve_captions --backend modal
    uv run python -m eval.runners.collect_lightcurve_captions --side base --backend modal
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from captioner.utils.config import load_config, remaining_argv
from captioner.utils.logging import get_logger
from eval.backend import free_local_backend, get_backend
from eval.datasets.lightcurve_yse import (
    balanced_sample,
    build_raw_inputs_lightcurve,
    load_lightcurve_table,
    render_lightcurve_plot,
    stratified_sample,
)

logger = get_logger(__name__)

DEFAULT_MAX_NEW_TOKENS = 300

# Base has no equivalent to equipped's trained instruction_variants default — plain, no
# classification framing, matching "the model captions in its own natural voice" for equipped.
BASE_CAPTION_QUESTION = "Describe this light curve plot in detail."

_OUT_BY_SIDE = {
    "equipped": "outputs/eval/raw_generations/yse_captions.json",
    "base": "outputs/eval/raw_generations/yse_captions_base.json",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", choices=["equipped", "base"], default="equipped")
    parser.add_argument("--repo-id", default="UniverseTBD/astrobridge-model-v7")
    parser.add_argument("--backend", choices=["local", "modal"], default="local")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n", type=int, default=None, help="proportional sample size; omit for the full 266-object set")
    parser.add_argument("--per-class", type=int, default=None, help="exactly this many per class instead (overrides --n)")
    parser.add_argument("--min-per-class", type=int, default=3, help="--n mode only")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument(
        "--batch-size", type=int, default=16,
        help="objects per forward pass. Decode is memory-bandwidth-bound, so B sequences cost "
             "barely more wall-clock than 1 while emitting B tokens per weight read; marginal "
             "VRAM is ~40MB/sequence against ~18GB of fixed weights. 1 restores the old "
             "one-at-a-time path.",
    )
    parser.add_argument(
        "--base-enable-thinking", action="store_true", default=False,
        help="--side base only. Off by default — Qwen's own chat template opens a reasoning "
             "block by default, confirmed live elsewhere in this eval bench to truncate the "
             "actual answer before it's ever reached at a bounded token budget.",
    )
    parser.add_argument("--out", default=None, help="defaults to yse_captions.json (equipped) or yse_captions_base.json (base)")
    parser.add_argument(
        "--resume", action="store_true",
        help="skip objects already present in --out and keep them. The output file is rewritten "
             "after every batch, so a crashed or hung run resumes instead of redoing everything.",
    )
    args = parser.parse_args(remaining_argv())

    cfg = load_config("base", "data", "modalities", "model", "stage2")

    table = load_lightcurve_table()
    if args.per_class is not None:
        table = balanced_sample(table, args.per_class, args.seed)
    elif args.n is not None:
        table = stratified_sample(table, args.n, args.seed, args.min_per_class)
    logger.info(
        f"Captioning {len(table)} objects (side={args.side!r}): {table['class_label'].value_counts().to_dict()}"
    )

    if args.side == "equipped":
        backend = get_backend(
            args.backend, side="equipped", cfg=cfg, repo_id=args.repo_id, device=args.device, modality_names=["lightcurve"],
        )
    else:
        backend = get_backend(
            args.backend, side="base", cfg=cfg, device=args.device, enable_thinking=args.base_enable_thinking,
        )

    out_path = Path(args.out) if args.out else Path(_OUT_BY_SIDE[args.side])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = [row for _, row in table.iterrows()]
    question = None if args.side == "equipped" else BASE_CAPTION_QUESTION

    # Resume: keep whatever a previous run already captioned and caption only the rest. A Modal
    # container that dies mid-run leaves the client hanging on .remote() with no error, so a long
    # run losing everything is a real failure mode, not a hypothetical.
    objects = []
    if args.resume and out_path.exists():
        objects = json.loads(out_path.read_text()).get("objects", [])
        done = {o["object_id"] for o in objects}
        rows = [row for row in rows if row["object_id"] not in done]
        logger.info(f"Resuming: {len(objects)} already captioned, {len(rows)} to go.")

    def _write() -> None:
        out_path.write_text(json.dumps({
            "dataset": "BuildNg/astrobridge-yse-test-dataset-v2",
            "side": args.side,
            "repo_id": args.repo_id if args.side == "equipped" else None,
            "question": question,
            "max_new_tokens": args.max_new_tokens,
            "batch_size": args.batch_size,
            "sampling": {
                "mode": "balanced" if args.per_class is not None else ("proportional" if args.n is not None else "full"),
                "per_class": args.per_class, "n": args.n, "min_per_class": args.min_per_class, "seed": args.seed,
                "actual_counts": table["class_label"].value_counts().to_dict(),
            },
            "objects": objects,
        }, indent=2))

    with tqdm(total=len(rows), desc=f"collect [{args.side} captions]") as bar:
        for start in range(0, len(rows), args.batch_size):
            chunk = rows[start:start + args.batch_size]
            if args.side == "equipped":
                raw_inputs_list = [build_raw_inputs_lightcurve(row, cfg) for row in chunk]
            else:
                raw_inputs_list = [{"image": render_lightcurve_plot(row)} for row in chunk]
            captions = backend.generate_batch(raw_inputs_list, question, args.max_new_tokens)
            if len(captions) != len(chunk):
                raise RuntimeError(f"batched backend returned {len(captions)} captions for {len(chunk)} objects.")
            for row, caption in zip(chunk, captions):
                objects.append({
                    "object_id": row["object_id"],
                    "label_name": row["class_label"],
                    "caption": caption,
                })
            _write()
            bar.update(len(chunk))
    if args.backend == "local":
        free_local_backend(backend)

    _write()
    logger.info(f"Wrote {len(objects)} captions to {out_path}")


if __name__ == "__main__":
    main()
