"""Soft, crowd-vote-grounded scoring for Galaxy10 classification — a wrong prediction gets partial
credit proportional to how plausible it actually was, according to real Galaxy Zoo DECaLS
volunteer votes, rather than flat 0.

**The class-defining rules below are not invented** — they're transcribed directly from
`henrysky/Galaxy10`'s own dataset-construction notebook (`Compile-Galaxy10-DES.ipynb`,
`gz_decals_volunteers_c` campaign variant), confirmed by fetching and reading that notebook's real
source cells. Each of the 10 Galaxy10 classes is defined there as a boolean AND/OR combination of
specific vote-fraction thresholds — e.g. "Barred Spiral" = `has-spiral-arms_yes_fraction > 0.8`
AND `bar_no_fraction < 0.2`. `_RULES` below encodes those same conditions, using the `_debiased`
column variants (from `astronolan/gz-decals-embeddings`) instead of the raw `_fraction` ones,
since debiased is the more appropriate estimate of "what the crowd would really think" for our
purposes (see the debiasing explanation from this project's own discussion — corrects for
distant/faint galaxies looking artificially smoother than they really are).

**What's a deliberate simplification, not an oversight**: Leung's original rules also gate on
`*_total-votes > N` (a statistical-reliability floor, not a "closeness" signal). Rather than
reimplementing those exact vote-count thresholds, this module relies on `gz-decals-embeddings`'s
own `_debiased.mask` columns as the reliability signal — if a class's defining branch is masked
for an object, that class becomes unscoreable for that object (see `class_margin`), rather than
silently trusting an unreliable fraction. This is a real simplification: worth reconciling with
Leung's exact vote-count gates later if this scoring bench's numbers ever need bit-for-bit parity
with the original hard-label assignment, but not needed for a *relative* base-vs-equipped
comparison, which is all this eval bench claims to measure.

Pipeline:
    class_margin(row, class_name)      -> float | None   (per-class "how well does this object
                                                            satisfy that class's real rule")
    soft_label_vector(row, true_label) -> dict[str, float] | None   (rescaled so true class = 1.0)
    score_prediction(vector, predicted_label) -> float | None
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from eval.datasets.image_galaxy10 import GALAXY10_LABELS

# Each class: a list of AND-terms. Each AND-term is a list of (column, op, threshold)
# alternatives — more than one alternative means OR (take whichever is more satisfied).
# `op` is ">" or "<". A threshold of 1.0 with ">" is special-cased in `_condition_margin` (see
# there) since Leung's original rule for Round Smooth Galaxies is `== 1.0`, not a plain `>`.
_RULES: dict[str, list[list[tuple[str, str, float]]]] = {
    "Disturbed Galaxies": [
        [("merging_major-disturbance_debiased", ">", 0.5), ("merging_minor-disturbance_debiased", ">", 0.7)],
        [("merging_merger_debiased", "<", 0.2)],
    ],
    "Merging Galaxies": [
        [("merging_merger_debiased", ">", 0.7)],
    ],
    "Round Smooth Galaxies": [
        [("how-rounded_round_debiased", ">", 1.0)],
    ],
    "In-between Round Smooth Galaxies": [
        [("smooth-or-featured_smooth_debiased", ">", 0.9)],
        [("how-rounded_in-between_debiased", ">", 0.8)],
    ],
    "Cigar Shaped Smooth Galaxies": [
        [("smooth-or-featured_smooth_debiased", ">", 0.9)],
        [("how-rounded_cigar-shaped_debiased", ">", 0.5)],
    ],
    "Barred Spiral Galaxies": [
        [("has-spiral-arms_yes_debiased", ">", 0.8)],
        [("bar_no_debiased", "<", 0.2)],
    ],
    "Unbarred Tight Spiral Galaxies": [
        [("smooth-or-featured_featured-or-disk_debiased", ">", 0.5)],
        [("spiral-winding_tight_debiased", ">", 0.65)],
        [("bar_no_debiased", ">", 0.8)],
    ],
    "Unbarred Loose Spiral Galaxies": [
        [("smooth-or-featured_featured-or-disk_debiased", ">", 0.6)],
        [("spiral-winding_tight_debiased", "<", 0.5)],
        [("has-spiral-arms_yes_debiased", ">", 0.8)],
        [("bar_no_debiased", ">", 0.8)],
    ],
    "Edge-on Galaxies without Bulge": [
        [("smooth-or-featured_featured-or-disk_debiased", ">", 0.5)],
        [("edge-on-bulge_none_debiased", ">", 0.6)],
    ],
    "Edge-on Galaxies with Bulge": [
        [("smooth-or-featured_featured-or-disk_debiased", ">", 0.6)],
        [("edge-on-bulge_none_debiased", "<", 0.1)],
    ],
}

assert set(_RULES) == set(GALAXY10_LABELS), "every Galaxy10 class must have a rule defined, and vice versa"


def _condition_margin(row: pd.Series, column: str, op: str, threshold: float) -> float | None:
    """Continuous [0,1] "how well satisfied" score for one leaf condition. `None` if the debiased
    value for `column` is masked (`{column}.mask` is True) or missing — an unreliable/absent
    estimate must not silently participate in a score, see module docstring.
    """
    mask_col = f"{column}.mask"
    if mask_col in row.index and bool(row[mask_col]):
        return None
    if column not in row.index or pd.isna(row[column]):
        return None

    x = float(row[column])
    if op == ">":
        if threshold >= 1.0:
            # Leung's real rule here is `== 1.0` (Round Smooth Galaxies only) — dividing by
            # (1 - threshold) would be a division by zero for a plain ">" margin formula, and a
            # hard step function (1.0 only at exactly x==1.0) throws away real information for
            # x close to but not exactly 1.0. Using x itself as the margin is a deliberate,
            # smooth softening of that discrete rule: "how close to perfectly round," not just
            # "is it exactly perfectly round."
            return float(np.clip(x, 0.0, 1.0))
        return float(np.clip((x - threshold) / (1.0 - threshold), 0.0, 1.0))
    if op == "<":
        return float(np.clip((threshold - x) / threshold, 0.0, 1.0))
    raise ValueError(f"Unknown op {op!r} — expected '>' or '<'.")


def class_margin(row: pd.Series, class_name: str) -> float | None:
    """Combines `class_name`'s AND-terms via min (an AND-conjunction is only as strong as its
    weakest term) and each term's OR-alternatives via max (take whichever alternative is more
    satisfied). Returns `None` if any AND-term has every one of its alternatives masked/missing —
    the whole class becomes unscoreable for this object rather than silently treating a missing
    piece of evidence as a pass or a fail.
    """
    term_margins = []
    for alternatives in _RULES[class_name]:
        alt_margins = [_condition_margin(row, col, op, thr) for col, op, thr in alternatives]
        real_alt_margins = [m for m in alt_margins if m is not None]
        if not real_alt_margins:
            return None
        term_margins.append(max(real_alt_margins))
    return min(term_margins)


def soft_label_vector(row: pd.Series, true_label: str) -> dict[str, float] | None:
    """The per-object 10-float vector: `true_label` always maps to exactly 1.0, every other class
    is `class_margin(row, c) / class_margin(row, true_label)`, clipped to `[0, 1]` (a class could
    in principle score higher than the true label's own margin — clip so it never exceeds the
    true class rather than implying "more true than the truth").

    Returns `None` if `true_label`'s own margin is unscoreable (masked/missing evidence for the
    object's own real class) — the object should be excluded from the eval sample entirely in
    that case (see `eval/README.md`), not scored with a distorted vector.
    """
    true_margin = class_margin(row, true_label)
    if true_margin is None or true_margin <= 0.0:
        return None

    vector = {}
    for label in GALAXY10_LABELS:
        if label == true_label:
            vector[label] = 1.0
            continue
        margin = class_margin(row, label)
        vector[label] = float(np.clip(margin / true_margin, 0.0, 1.0)) if margin is not None else None
    return vector


def score_prediction(vector: dict[str, float | None], predicted_label: str | None) -> float | None:
    """`vector[predicted_label]` — the soft score for whatever the model actually predicted.
    `None` if the model's answer didn't parse to any known label at all (`predict_label` returned
    `None`) or if that specific class's margin was itself unscoreable for this object — both
    excluded from the aggregate average, not counted as a silent 0.
    """
    if predicted_label is None:
        return None
    return vector.get(predicted_label)
