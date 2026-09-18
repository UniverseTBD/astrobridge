#!/usr/bin/env python
"""Step 1/2 of the lightcurve eval, same two-step shape as the image track
(`collect_image_labels.py` / `score_image_eval.py`): run both models over a stratified YSE sample
and save their raw answers — the compute-intensive step, meant to run once per sample. Scoring
(`score_lightcurve_eval.py`) is pure post-processing over this file's output: no GPU, no Modal, no
re-running the model.

**Both models, not equipped-only**: the base model gets `render_lightcurve_plot`'s rendered
flux-vs-time PNG (`{"image": <PIL.Image>}`, the image track's base-side contract), the equipped
model gets the raw `atcat_*` arrays via `build_raw_inputs_lightcurve` (or
`build_raw_inputs_with_image` under `--track lightcurve_plus_image`) — genuinely different inputs
per side, same as the image track's `image_rgb` vs `image_bands`.

**`logprob_argmax` (default, `--answer-format`)**: neither model is ever asked to emit a label at
all. Both generate free reasoning against `SN_REASONING_PROMPT`, then `eval/backend.py`'s
`.classify(...)` scores `SN_CANDIDATES`'s three exact label strings as teacher-forced continuations
of that reasoning and returns their log-probs — the predicted label is just the argmax, always one
of the three by construction. This replaces the earlier `digit_code`/`verbose_class` prompting,
both of which showed real compliance failures (a bare digit collapsing to one symbol regardless of
the object; free text needing fuzzy parsing that can fail outright — see
`eval.datasets.lightcurve_yse`'s module docstring for the confirmed-live evidence).

`--answer-format digit_code` is kept selectable so the two can be compared on the same objects — if
`logprob_argmax` also collapses to one class, that is evidence of a genuine content-blind prior
rather than a decoding artifact.

**Every random decision is seeded and saved**: the sample itself (`--seed`, `--n`/`--per-class`,
`--min-per-class`) is recorded in the output JSON, and greedy decoding (`do_sample=False`, used
throughout `.generate`/`.classify`) makes generation itself deterministic given the same
weights/hardware.

Usage:
    uv run python -m eval.runners.collect_lightcurve_labels --per-class 15 --seed 0
    uv run python -m eval.runners.collect_lightcurve_labels --per-class 15 --seed 0 --backend modal
    uv run python -m eval.runners.collect_lightcurve_labels --answer-format digit_code --per-class 15
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
    SN_CANDIDATES,
    SN_CLASS_CODE_PROMPT,
    SN_CLASS_CODES,
    SN_REASONING_PROMPT,
    balanced_sample,
    build_raw_inputs_lightcurve,
    build_raw_inputs_with_image,
    load_host_image_table,
    load_lightcurve_table,
    render_lightcurve_plot,
    stratified_sample,
)

logger = get_logger(__name__)

# Base needs materially more room than equipped for the digit_code path's short answer budget —
# see eval.datasets.lightcurve_yse's module docstring / earlier live evidence. logprob_argmax's
# reasoning step uses --max-reasoning-tokens for both sides instead (scoring itself is 1 forward
# pass per candidate, no generation budget needed).
DEFAULT_DIGITCODE_BASE_MAX_NEW_TOKENS = 40
DEFAULT_DIGITCODE_EQUIPPED_MAX_NEW_TOKENS = 8
DEFAULT_MAX_REASONING_TOKENS = 150


def _collect_logprob_argmax(lc_table, args, cfg, modality_names) -> list[dict]:
    base_backend = get_backend(args.backend, side="base", cfg=cfg, device=args.device)
    base_results: dict[str, dict] = {}
    for _, row in tqdm(lc_table.iterrows(), total=len(lc_table), desc="collect [base]"):
        image = render_lightcurve_plot(row)
        base_results[row["object_id"]] = base_backend.classify(
            {"image": image}, SN_REASONING_PROMPT, SN_CANDIDATES, args.max_reasoning_tokens,
        )
    if args.backend == "local":
        free_local_backend(base_backend)

    equipped_backend = get_backend(
        args.backend, side="equipped", cfg=cfg, repo_id=args.repo_id, device=args.device, modality_names=modality_names,
    )
    equipped_results: dict[str, dict] = {}
    for _, row in tqdm(lc_table.iterrows(), total=len(lc_table), desc="collect [equipped]"):
        raw_inputs = (
            build_raw_inputs_with_image(row, row, cfg) if args.track == "lightcurve_plus_image"
            else build_raw_inputs_lightcurve(row, cfg)
        )
        equipped_results[row["object_id"]] = equipped_backend.classify(
            raw_inputs, SN_REASONING_PROMPT, SN_CANDIDATES, args.max_reasoning_tokens,
        )
    if args.backend == "local":
        free_local_backend(equipped_backend)

    objects = []
    for _, row in lc_table.iterrows():
        oid = row["object_id"]
        for side, result in (("base", base_results[oid]), ("equipped", equipped_results[oid])):
            result["predicted_label"] = max(result["logprobs"], key=result["logprobs"].get).strip()
        objects.append({
            "object_id": oid,
            "label_name": row["class_label"],
            "base": base_results[oid],
            "equipped": equipped_results[oid],
        })
    return objects


def _collect_digit_code(lc_table, args, cfg, modality_names) -> list[dict]:
    base_backend = get_backend(
        args.backend, side="base", cfg=cfg, device=args.device, enable_thinking=False,
    )
    base_answers: dict[str, str] = {}
    for _, row in tqdm(lc_table.iterrows(), total=len(lc_table), desc="collect [base]"):
        image = render_lightcurve_plot(row)
        base_answers[row["object_id"]] = base_backend.generate(
            {"image": image}, SN_CLASS_CODE_PROMPT, args.base_max_new_tokens,
        )
    if args.backend == "local":
        free_local_backend(base_backend)

    equipped_backend = get_backend(
        args.backend, side="equipped", cfg=cfg, repo_id=args.repo_id, device=args.device, modality_names=modality_names,
    )
    equipped_answers: dict[str, str] = {}
    for _, row in tqdm(lc_table.iterrows(), total=len(lc_table), desc="collect [equipped]"):
        raw_inputs = (
            build_raw_inputs_with_image(row, row, cfg) if args.track == "lightcurve_plus_image"
            else build_raw_inputs_lightcurve(row, cfg)
        )
        equipped_answers[row["object_id"]] = equipped_backend.generate(
            raw_inputs, SN_CLASS_CODE_PROMPT, args.equipped_max_new_tokens,
        )
    if args.backend == "local":
        free_local_backend(equipped_backend)

    objects = []
    for _, row in lc_table.iterrows():
        oid = row["object_id"]
        objects.append({
            "object_id": oid,
            "label_name": row["class_label"],
            "base_answer": base_answers[oid],
            "equipped_answer": equipped_answers[oid],
        })
    return objects


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="UniverseTBD/astrobridge-model-v5")
    parser.add_argument("--backend", choices=["local", "modal"], default="local")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--track", choices=["lightcurve_only", "lightcurve_plus_image"], default="lightcurve_only")
    parser.add_argument("--n", type=int, default=None, help="total sample size across all 3 classes, drawn population-PROPORTIONALLY (SN Ia will dominate); omit to use the whole 266-object eval set. Ignored if --per-class is given.")
    parser.add_argument("--per-class", type=int, default=None, help="draw exactly this many objects from EACH class instead (equal buckets, no imbalance). Capped at SN Ibc's 15 in the YSE test set; overrides --n and --min-per-class.")
    parser.add_argument("--min-per-class", type=int, default=3, help="proportional (--n) mode only: floor per class. SN Ibc only has 15 objects total in this eval set, so keep this low")
    parser.add_argument("--seed", type=int, default=0, help="the ONE seed that determines the whole sample (irrelevant if --n is omitted)")
    parser.add_argument(
        "--answer-format", choices=["logprob_argmax", "digit_code"], default="logprob_argmax",
        help="logprob_argmax (default): score SN_CANDIDATES as a teacher-forced continuation of "
             "free reasoning and take the argmax — always a valid label, no parsing. digit_code: "
             "the older path where the model must emit a bare digit, kept selectable for "
             "comparison — see eval.datasets.lightcurve_yse's module docstring for why it's no "
             "longer the default.",
    )
    parser.add_argument("--base-max-new-tokens", type=int, default=DEFAULT_DIGITCODE_BASE_MAX_NEW_TOKENS, help="digit_code only")
    parser.add_argument("--equipped-max-new-tokens", type=int, default=DEFAULT_DIGITCODE_EQUIPPED_MAX_NEW_TOKENS, help="digit_code only")
    parser.add_argument("--max-reasoning-tokens", type=int, default=DEFAULT_MAX_REASONING_TOKENS, help="logprob_argmax only")
    parser.add_argument("--out", default=None, help="defaults to outputs/eval/raw_generations/yse_<track>_<format>_seed<seed>_<desc>.json")
    args = parser.parse_args(remaining_argv())

    cfg = load_config("base", "data", "modalities", "model", "stage2")

    lc_table = load_lightcurve_table()
    if args.track == "lightcurve_plus_image":
        image_table = load_host_image_table()
        lc_table = lc_table.merge(image_table, on=["object_id", "class_label"], how="inner", suffixes=("", "_img"))
        logger.info(f"{len(lc_table)} objects have both lightcurve and host-image data.")

    if args.per_class is not None:
        lc_table = balanced_sample(lc_table, args.per_class, args.seed)
    elif args.n is not None:
        lc_table = stratified_sample(lc_table, args.n, args.seed, args.min_per_class)
    logger.info(
        f"Evaluating {len(lc_table)} objects (track={args.track!r}, answer_format={args.answer_format!r}): "
        f"{lc_table['class_label'].value_counts().to_dict()}"
    )

    modality_names = ["image", "lightcurve"] if args.track == "lightcurve_plus_image" else ["lightcurve"]
    if args.answer_format == "logprob_argmax":
        objects = _collect_logprob_argmax(lc_table, args, cfg, modality_names)
    else:
        objects = _collect_digit_code(lc_table, args, cfg, modality_names)

    output = {
        "dataset": "BuildNg/astrobridge-yse-test-dataset-v2",
        "track": args.track,
        "repo_id": args.repo_id,
        "answer_format": args.answer_format,
        "reasoning_prompt": SN_REASONING_PROMPT if args.answer_format == "logprob_argmax" else None,
        "candidates": SN_CANDIDATES if args.answer_format == "logprob_argmax" else None,
        "class_codes": SN_CLASS_CODES if args.answer_format == "digit_code" else None,
        "sampling": {
            "mode": "balanced" if args.per_class is not None else "proportional",
            "per_class": args.per_class,
            "n": args.n,
            "min_per_class": args.min_per_class,
            "seed": args.seed,
            "actual_counts": lc_table["class_label"].value_counts().to_dict(),
        },
        "objects": objects,
    }

    if args.out:
        out_path = Path(args.out)
    else:
        desc = f"bal{args.per_class}" if args.per_class is not None else (f"n{args.n}" if args.n is not None else "nall")
        out_path = Path(f"outputs/eval/raw_generations/yse_{args.track}_{args.answer_format}_seed{args.seed}_{desc}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))
    logger.info(f"Wrote {len(objects)} raw generations to {out_path}")


if __name__ == "__main__":
    main()
