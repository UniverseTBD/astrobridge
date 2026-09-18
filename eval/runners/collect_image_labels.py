#!/usr/bin/env python
"""Step 1/2 of the image eval: run both models over a stratified Galaxy10 sample and save their
raw answers — the compute-intensive step, meant to run once per sample. Step 2
(`score_image_eval.py`) is pure post-processing over this file's output: no GPU, no Modal, no
re-running the model — so scoring logic can be iterated on for free without ever paying for
inference again.

**`logprob_argmax` (default, `--answer-format`)**: neither model is ever asked to emit a label or
a digit at all. Both generate free reasoning against `REASONING_PROMPT`, then `eval/backend.py`'s
`.classify(...)` scores `CANDIDATES`'s ten exact label strings as teacher-forced continuations of
that reasoning and returns their log-probs — the predicted label is just the argmax, always one of
the ten by construction. This replaces the earlier `digit_code` prompting, which showed a real
compliance failure: a live n=150 run found the equipped model never once emitted 4 of the 10 digit
codes (0/3/8/9), regardless of image content — a hard gap in the digit tokens themselves, not
scattered wrong guesses (see `eval.datasets.image_galaxy10`'s module docstring). `--answer-format
digit_code` (optionally `--shuffle-codes`) is kept selectable for comparison on the same objects.

**Every random decision is seeded and saved**: the sample itself (`--seed`, `--n`,
`--min-per-class`) is recorded in the output JSON, and greedy decoding (`do_sample=False`
throughout `.generate`/`.classify`) makes generation itself deterministic given the same
weights/hardware.

Usage:
    uv run python -m eval.runners.collect_image_labels --n 150 --seed 0
    uv run python -m eval.runners.collect_image_labels --n 150 --seed 0 --backend modal
    uv run python -m eval.runners.collect_image_labels --answer-format digit_code --n 150
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from captioner.utils.config import load_config, remaining_argv
from captioner.utils.logging import get_logger
from eval.backend import free_local_backend, get_backend
from eval.datasets.image_galaxy10 import (
    CANDIDATES,
    CLASS_CODE_PROMPT,
    CLASS_CODES,
    REASONING_PROMPT,
    SHUFFLED_CLASS_CODE_PROMPT,
    SHUFFLED_CLASS_CODES,
    build_raw_inputs,
    decode_rgb_image,
    load_galaxy10_aion_bands,
    load_galaxy10_rgb_only,
    stratified_sample,
)

logger = get_logger(__name__)

# digit_code only — base needs real room to reach an answer, equipped needs much less.
DEFAULT_DIGITCODE_BASE_MAX_NEW_TOKENS = 40
DEFAULT_DIGITCODE_EQUIPPED_MAX_NEW_TOKENS = 8
DEFAULT_MAX_REASONING_TOKENS = 150


def _align_bands_to(sampled_rgb, bands_table):
    sampled_ids = set(sampled_rgb["Galaxy10_DECals_index"])
    sampled_bands = bands_table[bands_table["Galaxy10_DECals_index"].isin(sampled_ids)]
    # Align to the exact same row order as sampled_rgb so per-object results line up 1:1.
    return sampled_bands.set_index("Galaxy10_DECals_index").loc[sampled_rgb["Galaxy10_DECals_index"]].reset_index()


def _collect_logprob_argmax(sampled_rgb, sampled_bands, args, cfg) -> list[dict]:
    base_backend = get_backend(args.backend, side="base", cfg=cfg, device=args.device)
    base_results: dict[int, dict] = {}
    for _, row in tqdm(sampled_rgb.iterrows(), total=len(sampled_rgb), desc="collect [base]"):
        image = decode_rgb_image(row["image_rgb"])
        base_results[row["Galaxy10_DECals_index"]] = base_backend.classify(
            {"image": image}, REASONING_PROMPT, CANDIDATES, args.max_reasoning_tokens,
        )
    if args.backend == "local":
        free_local_backend(base_backend)

    equipped_backend = get_backend(
        args.backend, side="equipped", cfg=cfg, repo_id=args.repo_id, device=args.device, modality_names=["image"],
    )
    equipped_results: dict[int, dict] = {}
    for _, row in tqdm(sampled_bands.iterrows(), total=len(sampled_bands), desc="collect [equipped]"):
        raw_inputs = build_raw_inputs(row)
        equipped_results[row["Galaxy10_DECals_index"]] = equipped_backend.classify(
            raw_inputs, REASONING_PROMPT, CANDIDATES, args.max_reasoning_tokens,
        )
    if args.backend == "local":
        free_local_backend(equipped_backend)

    objects = []
    for _, row in sampled_rgb.iterrows():
        idx = row["Galaxy10_DECals_index"]
        for result in (base_results[idx], equipped_results[idx]):
            result["predicted_label"] = max(result["logprobs"], key=result["logprobs"].get).strip()
        objects.append({
            "Galaxy10_DECals_index": int(idx),
            "ra": float(row["ra"]),
            "dec": float(row["dec"]),
            "label_name": row["label_name"],
            "base": base_results[idx],
            "equipped": equipped_results[idx],
        })
    return objects


def _collect_digit_code(sampled_rgb, sampled_bands, args, cfg, question, base_max_new_tokens, equipped_max_new_tokens) -> list[dict]:
    base_backend = get_backend(
        args.backend, side="base", cfg=cfg, device=args.device, enable_thinking=args.base_enable_thinking,
    )
    base_answers: dict[int, str] = {}
    for _, row in tqdm(sampled_rgb.iterrows(), total=len(sampled_rgb), desc="collect [base]"):
        image = decode_rgb_image(row["image_rgb"])
        base_answers[row["Galaxy10_DECals_index"]] = base_backend.generate({"image": image}, question, base_max_new_tokens)
    if args.backend == "local":
        free_local_backend(base_backend)

    equipped_backend = get_backend(
        args.backend, side="equipped", cfg=cfg, repo_id=args.repo_id, device=args.device, modality_names=["image"],
    )
    equipped_answers: dict[int, str] = {}
    for _, row in tqdm(sampled_bands.iterrows(), total=len(sampled_bands), desc="collect [equipped]"):
        raw_inputs = build_raw_inputs(row)
        equipped_answers[row["Galaxy10_DECals_index"]] = equipped_backend.generate(raw_inputs, question, equipped_max_new_tokens)
    if args.backend == "local":
        free_local_backend(equipped_backend)

    objects = []
    for _, row in sampled_rgb.iterrows():
        idx = row["Galaxy10_DECals_index"]
        objects.append({
            "Galaxy10_DECals_index": int(idx),
            "ra": float(row["ra"]),
            "dec": float(row["dec"]),
            "label_name": row["label_name"],
            "base_answer": base_answers[idx],
            "equipped_answer": equipped_answers[idx],
        })
    return objects


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="UniverseTBD/astrobridge-model-v5")
    parser.add_argument("--backend", choices=["local", "modal"], default="local")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n", type=int, default=150, help="total sample size across all 10 classes")
    parser.add_argument("--min-per-class", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0, help="the ONE seed that determines the whole sample")
    parser.add_argument(
        "--answer-format", choices=["logprob_argmax", "digit_code"], default="logprob_argmax",
        help="logprob_argmax (default): score CANDIDATES as a teacher-forced continuation of free "
             "reasoning and take the argmax — always a valid label, no parsing. digit_code: the "
             "older path where the model must emit a bare digit, kept selectable for comparison — "
             "see eval.datasets.image_galaxy10's module docstring for why it's no longer the "
             "default.",
    )
    parser.add_argument("--question", default=None, help="digit_code only — defaults to CLASS_CODE_PROMPT, or SHUFFLED_CLASS_CODE_PROMPT if --shuffle-codes is set")
    parser.add_argument(
        "--shuffle-codes", action="store_true", default=False,
        help="digit_code only. Use SHUFFLED_CLASS_CODES (a full derangement of the default "
             "digit->class mapping) instead of CLASS_CODES — see eval.datasets.image_galaxy10.py's "
             "SHUFFLED_CLASS_CODES docstring for why this diagnostic exists.",
    )
    parser.add_argument("--base-max-new-tokens", type=int, default=DEFAULT_DIGITCODE_BASE_MAX_NEW_TOKENS, help="digit_code only")
    parser.add_argument("--equipped-max-new-tokens", type=int, default=DEFAULT_DIGITCODE_EQUIPPED_MAX_NEW_TOKENS, help="digit_code only")
    parser.add_argument("--max-reasoning-tokens", type=int, default=DEFAULT_MAX_REASONING_TOKENS, help="logprob_argmax only")
    parser.add_argument(
        "--base-enable-thinking", action="store_true", default=False,
        help="digit_code only — Qwen/Qwen3.5-9B's own chat template opens a reasoning block by "
             "default; confirmed live that leaving it on makes the base model truncate before ever "
             "reaching a digit code. logprob_argmax always uses enable_thinking=False for its "
             "reasoning step regardless of this flag — see score_completions_qwen_native's "
             "docstring for why an open <think> block breaks that path's contract.",
    )
    parser.add_argument("--out", default=None, help="defaults to outputs/eval/raw_generations/galaxy10_<format>_seed<seed>_n<n>.json")
    args = parser.parse_args(remaining_argv())

    cfg = load_config("base", "data", "modalities", "model", "stage2")

    rgb_table = load_galaxy10_rgb_only()
    sampled_rgb = stratified_sample(rgb_table, args.n, args.seed, args.min_per_class)
    logger.info(f"Sampled {len(sampled_rgb)} objects (seed={args.seed}, answer_format={args.answer_format!r}) covering all 10 classes.")

    bands_table = load_galaxy10_aion_bands()
    sampled_bands = _align_bands_to(sampled_rgb, bands_table)

    if args.answer_format == "logprob_argmax":
        objects = _collect_logprob_argmax(sampled_rgb, sampled_bands, args, cfg)
        class_codes = None
    else:
        class_codes = SHUFFLED_CLASS_CODES if args.shuffle_codes else CLASS_CODES
        default_question = SHUFFLED_CLASS_CODE_PROMPT if args.shuffle_codes else CLASS_CODE_PROMPT
        question = args.question if args.question is not None else default_question
        objects = _collect_digit_code(
            sampled_rgb, sampled_bands, args, cfg, question, args.base_max_new_tokens, args.equipped_max_new_tokens,
        )

    output = {
        "dataset": "astronolan/galaxy10-aion",
        "repo_id": args.repo_id,
        "answer_format": args.answer_format,
        "reasoning_prompt": REASONING_PROMPT if args.answer_format == "logprob_argmax" else None,
        "candidates": CANDIDATES if args.answer_format == "logprob_argmax" else None,
        "class_codes": class_codes,
        "shuffle_codes": args.shuffle_codes if args.answer_format == "digit_code" else None,
        "sampling": {"n": args.n, "min_per_class": args.min_per_class, "seed": args.seed},
        "objects": objects,
    }

    if args.out:
        out_path = Path(args.out)
    else:
        suffix = "_shuffled" if (args.answer_format == "digit_code" and args.shuffle_codes) else ""
        out_path = Path(f"outputs/eval/raw_generations/galaxy10_{args.answer_format}_seed{args.seed}_n{args.n}{suffix}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))
    logger.info(f"Wrote {len(objects)} raw generations to {out_path}")


if __name__ == "__main__":
    main()
