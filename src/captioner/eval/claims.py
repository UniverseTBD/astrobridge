"""Claim-level analysis over a caption set, reused by both the caption-decomposition CI gate
(§4/§9 step 3) and the eval report (§8).
"""
from __future__ import annotations

from captioner.data.captions import Caption, validate_no_leakage


def validate_all(captions: list[Caption]) -> list[str]:
    violations = []
    for c in captions:
        violations.extend(validate_no_leakage(c))
    return violations


def claim_kind_histogram(captions: list[Caption]) -> dict[str, int]:
    hist: dict[str, int] = {}
    for c in captions:
        for claim in c.claims:
            hist[claim.kind] = hist.get(claim.kind, 0) + 1
    return hist
