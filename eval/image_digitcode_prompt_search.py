#!/usr/bin/env python
"""Compute-backed prompt search for the image digit-code task, equipped side only. NOT part of
the formal eval bench (no sampling seed contract, no report file) — a diagnostic tool for one
specific, confirmed-live problem: under the new chat-template prompting (`configs/model.yaml`'s
`prompt:` block), the equipped model emits 0/30 parseable digits on `CLASS_CODE_PROMPT`
(`outputs/eval/raw_generations/galaxy10_digit_code_seed0_n30.json` — it goes straight into its
trained free-form captioning voice regardless of the instruction, and the tight 8-token budget
then truncates it before anything resembling a digit ever appears).

Runs several `(system, instruction, max_new_tokens)` variants against a small real sample of
Galaxy10 objects and reports each variant's digit-parse rate — an empirical answer, not a guess,
to the question `eval/prompt_playground.py`'s docstring already posed: is this a prompt-wording
problem (fixable here) or a genuine caption-prior-always-wins problem (needs decode-time
constraint or training-side fix, not more prompt engineering)?

Usage:
    uv run python -m eval.image_digitcode_prompt_search --n 20 --backend modal
"""
from __future__ import annotations

import argparse
import json

from tqdm import tqdm

from captioner.utils.config import load_config
from eval.backend import free_local_backend, get_backend
from eval.datasets.image_galaxy10 import CLASS_CODE_LEGEND, CLASS_CODES, build_raw_inputs, load_galaxy10_aion_bands, stratified_sample
from eval.metrics.caption_to_label import predict_label_from_code

# Each variant is (name, system_override_or_None, instruction, max_new_tokens). `system=None`
# means "use configs/model.yaml's default system_variants[0]" — same as every other equipped call
# in this eval bench, so a variant that leaves it None isolates the instruction wording as the
# only thing being tested.
VARIANTS: list[tuple[str, str | None, str, int]] = [
    (
        "baseline (current CLASS_CODE_PROMPT)",
        None,
        f"Galaxy morphology classifier. Output ONLY the digit code, nothing else.\n{CLASS_CODE_LEGEND}\nImage class code:",
        8,
    ),
    (
        "forceful system override",
        "You are a digit-code classifier. You must respond with exactly one digit and nothing "
        "else — no words, no description, no punctuation, no explanation of any kind.",
        f"Galaxy morphology classifier. Output ONLY the digit code, nothing else.\n{CLASS_CODE_LEGEND}\nImage class code:",
        8,
    ),
    (
        "explicit anti-caption instruction",
        None,
        f"{CLASS_CODE_LEGEND}\nRespond with exactly one digit (0-9) from the list above. Do NOT "
        "describe the image. Do NOT write a caption. Output nothing but the single digit.\nAnswer:",
        8,
    ),
    (
        "forceful system + explicit anti-caption instruction",
        "You are a digit-code classifier. You must respond with exactly one digit and nothing "
        "else — no words, no description, no punctuation, no explanation of any kind.",
        f"{CLASS_CODE_LEGEND}\nRespond with exactly one digit (0-9) from the list above. Do NOT "
        "describe the image. Do NOT write a caption. Output nothing but the single digit.\nAnswer:",
        8,
    ),
    (
        "wider budget, same as baseline",
        None,
        f"Galaxy morphology classifier. Output ONLY the digit code, nothing else.\n{CLASS_CODE_LEGEND}\nImage class code:",
        32,
    ),
    (
        "mid-sentence completion style",
        None,
        f"{CLASS_CODE_LEGEND}\nThe single digit code for the galaxy shown is",
        8,
    ),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="UniverseTBD/astrobridge-model-v5")
    parser.add_argument("--backend", choices=["local", "modal"], default="local")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n", type=int, default=20, help="sample size across all 10 classes")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = load_config("base", "data", "modalities", "model", "stage2")

    bands_table = load_galaxy10_aion_bands()
    sample = stratified_sample(bands_table, args.n, args.seed, min_per_class=1, label_col="label_name")
    print(f"Sampled {len(sample)} objects (seed={args.seed}).")

    backend = get_backend(
        args.backend, side="equipped", cfg=cfg, repo_id=args.repo_id, device=args.device, modality_names=["image"],
    )

    results = []
    for name, system, instruction, max_new_tokens in VARIANTS:
        n_parsed = 0
        raw_answers = []
        for _, row in tqdm(sample.iterrows(), total=len(sample), desc=name):
            raw_inputs = build_raw_inputs(row)
            answer = backend.generate(raw_inputs, instruction, max_new_tokens, system=system)
            raw_answers.append(answer)
            if predict_label_from_code(answer, CLASS_CODES) is not None:
                n_parsed += 1
        parse_rate = n_parsed / len(sample)
        results.append({
            "name": name, "system": system, "instruction": instruction, "max_new_tokens": max_new_tokens,
            "parse_rate": parse_rate, "n_parsed": n_parsed, "n": len(sample),
            "sample_answers": raw_answers[:5],
        })
        print(f"\n=== {name} (max_new_tokens={max_new_tokens}) ===")
        print(f"parse_rate={parse_rate:.2f} ({n_parsed}/{len(sample)})")
        for a in raw_answers[:5]:
            print(f"  {a!r}")

    if args.backend == "local":
        free_local_backend(backend)

    results.sort(key=lambda r: r["parse_rate"], reverse=True)
    print("\n=== Ranked by parse rate ===")
    for r in results:
        print(f"{r['parse_rate']:.2f}  {r['name']}")

    out_path = "outputs/eval/image_digitcode_prompt_search.json"
    import os

    os.makedirs("outputs/eval", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote full results to {out_path}")


if __name__ == "__main__":
    main()
