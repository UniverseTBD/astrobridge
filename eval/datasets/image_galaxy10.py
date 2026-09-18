"""Loader for `astronolan/galaxy10-aion` — a Galaxy Zoo 10-class morphology benchmark, already
pre-built for AION specifically (not the generic Galaxy10 DECaLS release), by the same person
("Nolan") behind the spectra crossmatch file used elsewhere in this project. Has a real, proper
train/test split already (`data/train-*.parquet` x11, `data/test-*.parquet` x2) — this loader
only ever reads the `test` split.

Confirmed directly against the live HF repo: 796 test rows, columns `image_rgb` (struct
`{bytes, path}`, pre-rendered picture), `ra`, `dec`, `Galaxy10_DECals_index`, `label` (int 0-9),
`label_name` (string), `image_bands` (nested list). Real confirmed 10-class distribution on test:
Round Smooth Galaxies 148, Edge-on Galaxies with Bulge 123, Merging Galaxies 87, Unbarred Tight
Spiral Galaxies 90, Unbarred Loose Spiral Galaxies 90, Barred Spiral Galaxies 86, In-between Round
Smooth Galaxies 69, Disturbed Galaxies 49, Edge-on Galaxies without Bulge 46, Cigar Shaped Smooth
Galaxies 8.

**Real, unresolved-until-verified band question**: `image_bands` has 4 bands (96x96 each), not
the 3 (`[DES-G, DES-R, DES-Z]`) `AionImageEncoder` is configured for — no `image_band_names`-style
column exists to disambiguate directly from the schema. Working hypothesis (not yet empirically
confirmed): griz — g, r, i, z, ordered `[g, r, i, z]` — dropping index 2 (`i`) reproduces a grz
triple in the right order for AION's encoder, since this is the same survey convention our grz
encoder already targets. `load_galaxy10_aion_bands` implements this hypothesis and raises loudly
if `image_bands` doesn't have exactly 4 bands (so a future schema change fails fast rather than
silently mis-selecting), but does NOT itself verify the hypothesis is correct — that's a real,
one-off empirical check (render the resulting grz composite for a handful of rows and compare
against those same rows' `image_rgb` field, itself a Lupton-style grz composite) to run before
trusting predictions from this function; if it disagrees, re-open the question with Nolan
directly rather than continuing to guess.

`load_galaxy10_rgb_only` has no such blocker — it feeds the base-model comparison side (which
never touches AION) directly from the pre-rendered `image_rgb` field, so it's usable immediately.
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

GALAXY10_HF_PATH = "astronolan/galaxy10-aion"

# Real, confirmed `label_name` values, in `label` int order (0-9) — see module docstring. Order
# matters for eval/metrics/caption_to_label.py's tie-break behaviour; kept in sync with that
# file's GALAXY10_LABEL_SYNONYMS key order deliberately.
GALAXY10_LABELS = [
    "Disturbed Galaxies",
    "Merging Galaxies",
    "Round Smooth Galaxies",
    "In-between Round Smooth Galaxies",
    "Cigar Shaped Smooth Galaxies",
    "Barred Spiral Galaxies",
    "Unbarred Tight Spiral Galaxies",
    "Unbarred Loose Spiral Galaxies",
    "Edge-on Galaxies without Bulge",
    "Edge-on Galaxies with Bulge",
]

# Digit-code label mapping, `label` int order — matches GALAXY10_LABELS position-for-position.
# Confirmed real via a live prompt-engineering pass (eval/prompt_playground.py): a short, closed
# digit-code answer space is both far easier for a model to emit compliantly (one token, no
# free-text drift) and far easier to parse reliably (eval/metrics/caption_to_label.py's
# predict_label_from_code) than matching label-name keywords in free text. This is the format
# eval/runners/collect_image_labels.py actually uses now, not GALAXY10_LABEL_SYNONYMS-style
# free-text matching, which was the original (worse) design.
CLASS_CODES: dict[str, str] = {str(i): label for i, label in enumerate(GALAXY10_LABELS)}

CLASS_CODE_LEGEND = "\n".join(f"{code}={name}" for code, name in CLASS_CODES.items())

# The exact prompt confirmed live to work on both sides — ending mid-sentence ("Image class
# code:") matters for the equipped side specifically, which never goes through a chat template at
# all (see generate_caption's docstring): it's a genuine raw-text continuation there, not a
# stylistic touch, so the model completes it directly rather than starting a fresh turn.
CLASS_CODE_PROMPT = (
    "Galaxy morphology classifier. Output ONLY the digit code, nothing else.\n"
    f"{CLASS_CODE_LEGEND}\n"
    "Image class code:"
)

# A real, diagnostic finding (confirmed live via a full n=150 collect run) motivates this: the
# equipped model's answers, under CLASS_CODE_PROMPT above, never once emitted digits 0/3/8/9
# across 150 real objects, regardless of image content — a hard gap, not scattered wrong guesses.
# Whether that's a bias in the specific DIGIT TOKENS (fixable with prompting) or genuine
# confusion between the CLASSES those digits happened to be assigned to (needs a training-side
# fix) can't be told apart without re-running under a different digit assignment. SHUFFLED_CLASS_
# CODES is a full derangement (cyclic shift by 5 — every class gets a different digit than
# CLASS_CODES assigned it; confirmed no class keeps its original code) for exactly that test.
# collect_image_labels.py's --shuffle-codes flag uses this instead of CLASS_CODES.
SHUFFLED_CLASS_CODES: dict[str, str] = {str((i + 5) % 10): label for i, label in enumerate(GALAXY10_LABELS)}
assert all(SHUFFLED_CLASS_CODES[code] != CLASS_CODES[code] for code in CLASS_CODES), (
    "SHUFFLED_CLASS_CODES must be a true derangement of CLASS_CODES — every code maps to a "
    "DIFFERENT class than the default, or the whole point of the shuffle experiment is broken."
)

SHUFFLED_CLASS_CODE_LEGEND = "\n".join(f"{code}={name}" for code, name in SHUFFLED_CLASS_CODES.items())
SHUFFLED_CLASS_CODE_PROMPT = (
    "Galaxy morphology classifier. Output ONLY the digit code, nothing else.\n"
    f"{SHUFFLED_CLASS_CODE_LEGEND}\n"
    "Image class code:"
)

# --- Deterministic classification (the default path, see eval.datasets.lightcurve_yse's module
# docstring for the full rationale — the same digit-emission compliance failure showed up here
# too: CLASS_CODE_PROMPT's digit-code answers never once used 4 of the 10 digits across a real
# n=150 run, regardless of image content). REASONING_PROMPT never asks for a digit or a class
# name at all; `eval/backend.py`'s `.classify(...)` scores CANDIDATES as a teacher-forced
# continuation of the model's free reasoning and takes the argmax — always one of the 10 labels
# by construction, no parsing, no digit-token bias possible.
REASONING_PROMPT = (
    "Examine this galaxy image: describe its overall shape (round, cigar-shaped, or disk-like), "
    "whether spiral arms are visible and how tightly or loosely wound they are, whether a central "
    "bar is present, whether a disk is seen edge-on or face-on and whether it has a prominent "
    "central bulge, and whether the galaxy shows signs of disturbance or merging with a companion. "
    "Based on the evidence above, the classification is:"
)

# Leading space on each candidate, same reason as eval.datasets.lightcurve_yse.SN_CANDIDATES: a
# clean BPE boundary regardless of what precedes it, so scoring is length-invariant and robust to
# the reasoning text's exact ending.
CANDIDATES = [f" {label}" for label in GALAXY10_LABELS]

# Working hypothesis only (see module docstring) — verify before trusting load_galaxy10_aion_bands.
_HYPOTHESIZED_BAND_ORDER = ["g", "r", "i", "z"]
_KEEP_BAND_INDICES = [0, 1, 3]  # g, r, z — dropping index 2 ("i")


def _test_parquet_paths(hf_path: str = GALAXY10_HF_PATH) -> list[str]:
    from huggingface_hub import list_repo_files

    files = sorted(
        f for f in list_repo_files(hf_path, repo_type="dataset")
        if f.startswith("data/test-") and f.endswith(".parquet")
    )
    if not files:
        raise FileNotFoundError(
            f"No 'data/test-*.parquet' files found in {hf_path!r} — check the repo layout hasn't changed."
        )
    return files


def _download_and_read(hf_path: str, columns: list[str], cache_dir: Path | None = None) -> pd.DataFrame:
    from huggingface_hub import hf_hub_download

    import pyarrow.parquet as pq

    frames = []
    for f in _test_parquet_paths(hf_path):
        local_path = hf_hub_download(
            repo_id=hf_path, filename=f, repo_type="dataset", cache_dir=str(cache_dir) if cache_dir else None,
        )
        table = pq.read_table(local_path, columns=columns)
        frames.append(table.to_pandas(ignore_metadata=True))
    return pd.concat(frames, ignore_index=True, sort=False)


def load_galaxy10_rgb_only(hf_path: str = GALAXY10_HF_PATH, cache_dir: Path | None = None) -> pd.DataFrame:
    """One row per test-set object: `label`, `label_name`, `ra`/`dec` (needed later for
    `eval.datasets.gz_decals_votes`'s crossmatch — not used by the base-model side itself), and
    the pre-rendered `image_rgb` struct (`{bytes, path}` — decode with `decode_rgb_image` below).
    Feeds the base-model comparison side only; never touches `image_bands`/AION.
    """
    return _download_and_read(
        hf_path, ["Galaxy10_DECals_index", "ra", "dec", "image_rgb", "label", "label_name"], cache_dir,
    )


def decode_rgb_image(image_rgb_struct: dict) -> Image.Image:
    """`image_rgb_struct["bytes"]` is a PNG/JPEG-encoded picture (HF's standard `Image` feature
    struct) — decode it into a plain `PIL.Image` for `eval.backend`'s base-side `{"image": ...}`
    raw_inputs contract.
    """
    return Image.open(io.BytesIO(image_rgb_struct["bytes"]))


def load_galaxy10_aion_bands(hf_path: str = GALAXY10_HF_PATH, cache_dir: Path | None = None) -> pd.DataFrame:
    """One row per test-set object: `label`, `label_name`, and `image_bands` — the raw multi-band
    array, untouched (band selection happens in `build_raw_inputs`, not here, so the "not yet
    verified" hypothesis stays isolated to one function).
    """
    return _download_and_read(hf_path, ["Galaxy10_DECals_index", "image_bands", "label", "label_name"], cache_dir)


def stratified_sample(
    table: pd.DataFrame, n_total: int, seed: int, min_per_class: int = 2, label_col: str = "label_name",
) -> pd.DataFrame:
    """Deterministic, seeded sample covering all 10 classes — every random decision here is
    driven by exactly one `numpy.random.default_rng(seed)` instance, consumed in a fixed order
    (`GALAXY10_LABELS`, never dict/groupby iteration order, which isn't guaranteed stable across
    pandas versions), so the same `seed` always draws the same objects.

    Per-class target counts use the "largest remainder" apportionment method (a standard,
    well-known deterministic algorithm — the same one used for allocating parliamentary seats
    proportionally, not something invented for this): allocate each class
    `floor(n_total * class_size / total_size)`, raised to `min_per_class` (capped at how many
    that class actually has — `Cigar Shaped Smooth Galaxies` only has 8 in the full test set, so
    `min_per_class` beyond that is silently capped, not padded with duplicates), then hand out
    the remaining slots one at a time to the classes with the largest fractional remainder,
    skipping any class already at its cap. This guarantees every one of the 10 classes appears
    (as long as `min_per_class >= 1`) and the total sampled is exactly `n_total` (or less, only
    if `n_total` exceeds the full table size).
    """
    rng = np.random.default_rng(seed)
    groups = {label: table.index[table[label_col] == label].to_numpy() for label in GALAXY10_LABELS}
    sizes = {label: len(idx) for label, idx in groups.items()}
    total_available = sum(sizes.values())
    if n_total > total_available:
        raise ValueError(f"n_total={n_total} exceeds the {total_available} objects available across all 10 classes.")
    min_feasible = sum(min(min_per_class, sizes[label]) for label in GALAXY10_LABELS)
    if n_total < min_feasible:
        raise ValueError(
            f"n_total={n_total} is below {min_feasible}, the minimum needed to give every class at "
            f"least min_per_class={min_per_class} (capped by classes with fewer available, e.g. "
            f"Cigar Shaped Smooth Galaxies has only {sizes['Cigar Shaped Smooth Galaxies']}). "
            "Raise n_total or lower min_per_class — returning a silently-larger-than-requested "
            "sample here would break the 'exactly n_total, reproducibly' contract."
        )

    raw_share = {label: n_total * sizes[label] / total_available for label in GALAXY10_LABELS}
    target = {label: min(sizes[label], max(min_per_class, int(np.floor(raw_share[label])))) for label in GALAXY10_LABELS}

    # Largest-remainder-first order for handing out slots; smallest-remainder-first (reverse) for
    # taking them back — a class least entitled to its share proportionally gives one back first.
    by_largest_remainder = sorted(GALAXY10_LABELS, key=lambda label: raw_share[label] - np.floor(raw_share[label]), reverse=True)
    by_smallest_remainder = list(reversed(by_largest_remainder))

    # The `min_per_class` floor (applied above) can push the initial sum ABOVE n_total on its own
    # — e.g. min_per_class=2 across 10 classes floors to 20 regardless of n_total, so a small
    # n_total could already be exceeded before any remainder-based allocation happens. Correct in
    # whichever direction is needed, deterministically, rather than only ever adding.
    while sum(target.values()) > n_total:
        progressed = False
        for label in by_smallest_remainder:
            if sum(target.values()) <= n_total:
                break
            if target[label] > min_per_class:  # never reduce below the floor the caller asked for
                target[label] -= 1
                progressed = True
        if not progressed:
            break  # every class already at its own min_per_class floor — can't reduce further

    while sum(target.values()) < n_total:
        progressed = False
        for label in by_largest_remainder:
            if sum(target.values()) >= n_total:
                break
            if target[label] < sizes[label]:
                target[label] += 1
                progressed = True
        if not progressed:
            break  # every class already at its cap — n_total unreachable, already validated above not to exceed total

    sampled_indices = []
    for label in GALAXY10_LABELS:  # fixed order — see docstring
        chosen = rng.choice(groups[label], size=target[label], replace=False)
        sampled_indices.extend(chosen.tolist())

    return table.loc[sampled_indices].reset_index(drop=True)


def build_raw_inputs(row: pd.Series) -> dict:
    """Applies the griz-minus-`i` hypothesis (see module docstring) to turn one
    `load_galaxy10_aion_bands` row's `image_bands` into the `{"image": {"pixel_values": ...}}`
    shape `generate_caption` expects. Raises loudly if `image_bands` doesn't have exactly 4 bands
    — a real schema change should fail fast here, not silently select the wrong 3.

    `image_bands` deserializes from parquet as triple-nested `dtype=object` numpy arrays (outer
    (4,) object array -> each element a (96,) object array -> each element a real (96,) float32
    array), not one contiguous numeric block — confirmed real via a live check, not assumed.
    `np.asarray(b, dtype=np.float32)` on the middle layer directly raises `ValueError: setting an
    array element with a sequence` (numpy can't auto-flatten a doubly-nested object array in one
    call), so each level is stacked explicitly here instead.
    """
    bands = np.stack([np.stack([np.asarray(px, dtype=np.float32) for px in band]) for band in row["image_bands"]])
    if bands.shape[0] != 4:
        raise ValueError(
            f"Expected 4 bands (hypothesized {_HYPOTHESIZED_BAND_ORDER}), got {bands.shape[0]} — "
            "the griz-minus-i selection below assumes exactly 4; re-check the real schema before "
            "trusting this function."
        )
    grz = bands[_KEEP_BAND_INDICES]  # (3, H, W), hypothesized [g, r, z] order
    return {"image": {"pixel_values": torch.from_numpy(grz).unsqueeze(0)}}
