"""Shared prompt-text formatting — used by both data/dataset.py (training/eval, reading from
cache) and inference.py (live, uncached single-object inference), so the chat-template wording an
object was trained against is guaranteed identical to what inference constructs at serve time.

The training example is a ChatML turn with the Q-Former vectors spliced into the user turn:

    {wrapper_pre with {system} filled}   <-- text BEFORE the 48 vectors
    [48 Q-Former vectors]
    {wrapper_post with {instruction} filled}   <-- text AFTER the vectors, up to the assistant header
    {caption}{caption_suffix}   <-- the loss target

`build_wrapper_text` returns the two text halves; `pick_prompt_variants` samples one
(system, instruction) pair. See configs/model.yaml's `prompt:` block and RETRAIN_SKETCH.md.
"""
from __future__ import annotations

import numpy as np

# Human-readable phrase per modality, so prompts read naturally as a modality is added. A modality
# with no entry falls back to "a <name>", which is right for most names but wrong for "image" (needs
# "an") and clumsy for compound names like "lightcurve".
DISPLAY_NAMES = {
    "image": "an image",
    "spectra": "a spectrum",
    "lightcurve": "a light curve",
}


def _display(name: str) -> str:
    return DISPLAY_NAMES.get(name, f"a {name}")


def human_readable_subset(subset: frozenset[str]) -> str:
    names = sorted(subset)
    if not names:
        # Only reachable via _sample_target_subset's defensive empty fallback (a caption-less
        # example the degenerate-batch guard then skips). Return a noun so a "{modalities}"
        # instruction still reads as a sentence rather than "Based on  alone,".
        return "the observation"
    return " and ".join(_display(n) for n in names)


def pick_prompt_variants(
    prompt_cfg, subset: frozenset[str], rng: np.random.Generator,
) -> tuple[str, str]:
    """Sample one `(system, instruction)` pair from `prompt_cfg.system_variants` /
    `prompt_cfg.instruction_variants`, independently and uniformly. `{modalities}` in an
    instruction is filled with `human_readable_subset(subset)`; system variants never contain it.

    `rng` is the caller's seeded generator (data/dataset.py seeds one per (object_id, idx) so the
    draw is reproducible per epoch but varies across objects and epochs).
    """
    systems = list(prompt_cfg.system_variants)
    instructions = list(prompt_cfg.instruction_variants)
    system = str(systems[int(rng.integers(len(systems)))])
    instruction = str(instructions[int(rng.integers(len(instructions)))])
    if "{modalities}" in instruction:
        instruction = instruction.format(modalities=human_readable_subset(subset))
    return system, instruction


def build_wrapper_text(prompt_cfg, system: str, instruction: str) -> tuple[str, str]:
    """`(pre_text, post_text)` — everything before the observation vectors, and everything after
    them up to and including the `<|im_start|>assistant\\n` header. Tokenize each with
    `add_special_tokens=False`; the Q-Former vectors go between the two.
    """
    pre_text = str(prompt_cfg.wrapper_pre).format(system=system)
    post_text = str(prompt_cfg.wrapper_post).format(instruction=instruction)
    return pre_text, post_text
