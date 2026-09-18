"""Coarse, group-level partial credit — a deliberately simpler complement to `vote_fraction_
scoring.py`, not a replacement. Same/similar-group misclassifications get 0.5, wrong-group
misclassifications get 0.0, exact matches get 1.0.

**Where this genuinely differs from the vote-fraction score, know both before trusting either
alone**:
- This is a fixed, hand-picked partition with a fixed, hand-picked reward (0.5) — no crowd data
  says a same-group miss is worth exactly half credit, it's a reasonable heuristic, not an
  empirical measurement. `vote_fraction_scoring.py`'s score is grounded in that specific object's
  own real Galaxy Zoo DECaLS vote fractions instead.
- This is coarser: predicting "In-between Round Smooth" when the truth is "Round Smooth" (a real
  near-miss) scores the same 0.5 as predicting "Cigar Shaped" when the truth is "Round Smooth" (a
  real miss) — both are "same group, wrong class." The vote-fraction score can tell these apart;
  this one can't, by design.
- The group boundary itself can occasionally be a source of unfairness in the OTHER direction —
  an edge-on spiral can look close to a face-on spiral, but "Spiral" and "Edge-On" are different
  groups here, so that confusion scores a flat 0.0 even where the vote-fraction score (having no
  fixed boundary at all) might not.

The four groups map directly onto the real Galaxy Zoo decision tree's own top-level branches
(`smooth-or-featured`, then `disk-edge-on` vs. spiral-arm/bar/winding questions for
featured-or-disk, plus the mostly-independent `merging` branch) — not an arbitrary split, though
still a discrete simplification of a continuous, branching structure. See `vote_fraction_
scoring.py`'s module docstring for that real tree's structure in full.
"""
from __future__ import annotations

GALAXY10_GROUPS: dict[str, str] = {
    "Round Smooth Galaxies": "Smooth",
    "In-between Round Smooth Galaxies": "Smooth",
    "Cigar Shaped Smooth Galaxies": "Smooth",
    "Barred Spiral Galaxies": "Spiral",
    "Unbarred Tight Spiral Galaxies": "Spiral",
    "Unbarred Loose Spiral Galaxies": "Spiral",
    "Edge-on Galaxies without Bulge": "Edge-On",
    "Edge-on Galaxies with Bulge": "Edge-On",
    "Disturbed Galaxies": "Disturbed/Merging",
    "Merging Galaxies": "Disturbed/Merging",
}


def group_score(true_label: str, predicted_label: str | None) -> float:
    """1.0 exact match, 0.5 same group / different class, 0.0 different group or unparseable
    (`predicted_label is None`, e.g. `caption_to_label.predict_label*` found nothing) — an
    unparseable answer is scored as wrong, not excluded, matching how the hard/soft metrics
    elsewhere in this eval bench already treat a non-answer (never silently dropped).
    """
    if predicted_label is None:
        return 0.0
    if predicted_label == true_label:
        return 1.0
    if GALAXY10_GROUPS.get(predicted_label) == GALAXY10_GROUPS.get(true_label):
        return 0.5
    return 0.0


def group_report(y_true: list[str], y_pred: list[str | None]) -> dict:
    """Aggregate report: overall mean group-score, plus a per-group breakdown (mean score and
    count) restricted to objects whose TRUE label falls in that group — so a group's number
    answers "when the object really was e.g. a Spiral, how well did predictions do," not
    contaminated by objects from other groups.
    """
    if len(y_true) != len(y_pred):
        raise ValueError(f"y_true ({len(y_true)}) and y_pred ({len(y_pred)}) must be the same length.")

    scores = [group_score(t, p) for t, p in zip(y_true, y_pred)]

    per_group: dict[str, dict] = {}
    for group_name in set(GALAXY10_GROUPS.values()):
        group_scores = [s for t, s in zip(y_true, scores) if GALAXY10_GROUPS.get(t) == group_name]
        per_group[group_name] = {
            "mean_score": (sum(group_scores) / len(group_scores)) if group_scores else None,
            "support": len(group_scores),
        }

    return {
        "mean_score": (sum(scores) / len(scores)) if scores else None,
        "n": len(scores),
        "per_group": per_group,
    }
