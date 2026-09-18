"""Caption decomposition: splits `mention_summary`/`evidence_quotes` (object-level, paper-derived)
into per-claim, per-tier captions (§4). This is decomposition, not generation — claims that no
available modality supports are dropped, not reassigned.

`decompose_object` below is a rule-based baseline extractor (generator="rule_based_v0" in
configs/data.yaml). It is a deliberately swappable extension point: replace it with an
LLM-backed extractor (generator="llm_v1", say) without touching Caption/Claim or the leakage
validator, which are generator-agnostic.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

SPECTRA_KEYWORDS = ("redshift", "z =", "z=", "emission line", "absorption line", "spectrum", "spectroscop")
IMAGE_KEYWORDS = ("morpholog", "extended", "compact", "disk", "spiral", "elliptical", "bright", "flux", "magnitude", "color")
RELATIONAL_MARKERS = ("consistent with", "confirming", "corroborat", "combined with", "together with")

# Time-domain photometry vocabulary. Note the deliberate avoidance of a bare "mag": the substring
# also occurs inside "image".
LIGHTCURVE_KEYWORDS = (
    "light curve", "lightcurve", "mjd", "magnitude", " mag", "photometr", "peak", "rise", "rising",
    "rose", "declin", "fad", "brighten", "outburst", "epoch", "cadence", "plateau", "amplitude",
    "brightness", "day",
)

# NOTE: an earlier `LIGHTCURVE_UNSUPPORTABLE` regex veto lived here — it dropped any
# lightcurve sentence naming a designation, redshift, "literature", "classif",
# "spectroscop", or "Type I[abc]". Removed deliberately (RETRAIN_SKETCH.md B3): transient
# captions are now used RAW in scripts/01_generate_captions.py, so the model sees the explicit
# (lightcurve -> SN type) signal, at the cost of also seeing claims a light curve alone cannot
# ground. The lightcurve branch of `_tag_support` below is now a plain keyword check, symmetric
# with spectra/image.


@dataclass
class Claim:
    text: str
    supporting: frozenset[str]     # len > 1 => joint claim; empty => dropped
    kind: Literal["observation", "inference", "relation"]
    provenance: str                # arxiv_id + quote_id where available


@dataclass
class Caption:
    object_id: str
    subset: frozenset[str]
    text: str
    claims: list[Claim] = field(default_factory=list)
    source: Literal["paper_decomposed", "llm_generated", "human_checked"] = "paper_decomposed"
    generator: str = "rule_based_v0"


def _split_sentences(text: str) -> list[str]:
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _tag_support(sentence: str, available_modalities: frozenset[str]) -> frozenset[str]:
    """Rule-based tagging against structured-field keyword priors (§4). A sentence supported by
    neither available modality returns an empty frozenset and the caller drops it.
    """
    low = sentence.lower()
    supports: set[str] = set()
    if "spectra" in available_modalities and any(k in low for k in SPECTRA_KEYWORDS):
        supports.add("spectra")
    if "image" in available_modalities and any(k in low for k in IMAGE_KEYWORDS):
        supports.add("image")
    if "lightcurve" in available_modalities and any(k in low for k in LIGHTCURVE_KEYWORDS):
        supports.add("lightcurve")
    if len(supports) > 1 or (len(supports) == 1 and any(m in low for m in RELATIONAL_MARKERS)):
        pass  # relational marker alone doesn't add support beyond what keywords already found
    return frozenset(supports)


def decompose_object(
    object_id: str,
    mention_summary: str,
    evidence_quotes: list[str] | None,
    arxiv_id: str | None,
    available_modalities: frozenset[str],
    generator: str = "rule_based_v0",
) -> list[Claim]:
    """Splits mention_summary into sentence-level claims and tags each with the modality subset
    it is supported by, restricted to `available_modalities` (what this build actually has —
    not what the paper discusses). Claims supported by neither are dropped entirely.
    """
    sentences = _split_sentences(mention_summary)
    claims: list[Claim] = []
    for i, sent in enumerate(sentences):
        supporting = _tag_support(sent, available_modalities)
        if not supporting:
            continue  # multi-wavelength or otherwise unsupportable claim — dropped, not reassigned
        kind = "relation" if len(supporting) > 1 else "observation"
        provenance = f"{arxiv_id or 'unknown'}#sent{i}"
        claims.append(Claim(text=sent, supporting=supporting, kind=kind, provenance=provenance))
    return claims


def compose_captions(object_id: str, claims: list[Claim], available_modalities: frozenset[str]) -> list[Caption]:
    """One Caption per tier the object actually supports: `single` per present modality, plus
    `joint` if both are present (§4 table). Unpaired objects (available_modalities has one
    element) only get their single-modality caption.
    """
    captions: list[Caption] = []
    for modality in sorted(available_modalities):
        subset = frozenset({modality})
        subset_claims = [c for c in claims if c.supporting == subset]
        if subset_claims:
            text = " ".join(c.text for c in subset_claims)
            captions.append(Caption(object_id=object_id, subset=subset, text=text, claims=subset_claims))

    if len(available_modalities) > 1:
        subset = frozenset(available_modalities)
        subset_claims = [c for c in claims if c.supporting.issubset(subset)]
        if subset_claims:
            text = " ".join(c.text for c in subset_claims)
            captions.append(Caption(object_id=object_id, subset=subset, text=text, claims=subset_claims))

    return captions


def validate_no_leakage(c: Caption) -> list[str]:
    """Violation if any claim.supporting is not a subset of c.subset."""
    violations = []
    for claim in c.claims:
        if not claim.supporting.issubset(c.subset):
            violations.append(
                f"object={c.object_id} tier={sorted(c.subset)}: claim {claim.provenance!r} "
                f"supported by {sorted(claim.supporting)}, which is not a subset of the tier."
            )
    return violations


def claim_survival_rate(n_source_sentences: int, n_surviving_claims: int) -> float:
    if n_source_sentences == 0:
        return 0.0
    return n_surviving_claims / n_source_sentences
