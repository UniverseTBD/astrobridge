"""Numerical-claim extraction and deterministic caption checks for the v8 benchmark.

Gemini is used only to turn caption prose into a fixed numerical schema.  It never receives the
light curve or true class and never decides whether a claim is correct.  Reference values and
errors are calculated locally from the immutable retained-observation table.
"""
from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass, field
from typing import Literal

from eval.benchmark_protocol import SN_LABELS, fingerprint_contract, sha256_bytes


EXTRACTOR_VERSION = "v8-caption-numeric-claims-v2"
SCORING_VERSION = "v8-caption-numeric-scoring-v3"
DEFAULT_EXTRACTOR_MODEL = "gemini-3.7-flash"
DEFAULT_MAX_OUTPUT_TOKENS = 2048
MANUAL_REVIEW_SEED = 42
MANUAL_REVIEW_COUNTS = {"SN Ia": 7, "SN II": 7, "SN Ibc": 6}

# The first ten were sampled before this audit was designed.  Keeping them makes the expanded
# review a genuine extension rather than a replacement chosen after reading the outputs.
INITIAL_SEED42_REVIEW_IDS = (
    "SN2020tzs",
    "SN2020aatr",
    "SN2020tjd",
    "SN2021sje",
    "SN2020shf",
    "SN2021tux",
    "SN2021uhl",
    "SN2020azk",
    "SN2020eci",
    "SN2019yvr",
)

# A person checked these three claim types against the captions for the fixed 20-object sample.
# Every explicit claim was transcribed correctly, with no unsupported or missed claims.  Keeping
# the object-level counts here makes that manual result reproducible instead of recording only an
# aggregate statement in the paper.
MANUAL_VALIDATED_CLAIM_COUNTS = {
    "SN2021urb": {"duration": 1, "peaks": 2, "gaps": 1},
    "SN2020tlf": {"duration": 1, "peaks": 2, "gaps": 1},
    "SN2021us": {"duration": 1, "peaks": 2, "gaps": 0},
    "SN2020azk": {"duration": 1, "peaks": 2, "gaps": 1},
    "SN2020shf": {"duration": 1, "peaks": 2, "gaps": 0},
    "SN2021uhl": {"duration": 1, "peaks": 2, "gaps": 1},
    "SN2021tux": {"duration": 1, "peaks": 2, "gaps": 1},
    "SN2020aazk": {"duration": 1, "peaks": 2, "gaps": 1},
    "SN2020evu": {"duration": 1, "peaks": 2, "gaps": 0},
    "SN2020tfb": {"duration": 1, "peaks": 2, "gaps": 1},
    "SN2019yvr": {"duration": 1, "peaks": 2, "gaps": 1},
    "SN2020eci": {"duration": 1, "peaks": 2, "gaps": 1},
    "SN2020ej": {"duration": 1, "peaks": 1, "gaps": 1},
    "SN2021sje": {"duration": 1, "peaks": 2, "gaps": 1},
    "SN2020jfo": {"duration": 1, "peaks": 2, "gaps": 0},
    "SN2020tzs": {"duration": 1, "peaks": 2, "gaps": 0},
    "SN2020aatr": {"duration": 1, "peaks": 2, "gaps": 0},
    "SN2021abam": {"duration": 1, "peaks": 2, "gaps": 0},
    "SN2021xbg": {"duration": 1, "peaks": 2, "gaps": 1},
    "SN2020tjd": {"duration": 1, "peaks": 2, "gaps": 0},
}

EXTRACTION_SYSTEM_PROMPT = (
    "You extract explicitly stated numerical light-curve claims from a model-written caption. "
    "Do not assess correctness, infer missing values, calculate new values, classify the object, "
    "or extract qualitative morphology. Copy an exact supporting quote for every claim."
)

EXTRACTION_USER_TEMPLATE = """Extract only the following claims when they are explicitly stated:
- total observation duration in days;
- per-band brightest-sampled or peak time and magnitude;
- observation-gap start, end, and duration;

Use null for an unstated scalar and an empty list when a claim type is absent. Never derive an
endpoint from a duration. Quotes must be exact substrings of the caption. Ignore all class labels
and qualitative statements.

<caption>
{caption}
</caption>"""


@dataclass
class NumericClaim:
    value: float
    quote: str


@dataclass
class PeakClaim:
    band: Literal["g", "r"]
    quote: str
    time_days: float | None = None
    magnitude: float | None = None


@dataclass
class GapClaim:
    quote: str
    start_days: float | None = None
    end_days: float | None = None
    duration_days: float | None = None


@dataclass
class CaptionNumericClaims:
    duration_days: NumericClaim | None = None
    peaks: list[PeakClaim] = field(default_factory=list)
    gaps: list[GapClaim] = field(default_factory=list)


_NULLABLE_NUMBER = {"anyOf": [{"type": "number"}, {"type": "null"}]}
CLAIMS_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "duration_days": {
            "anyOf": [
                {
                    "type": "object",
                    "properties": {"value": {"type": "number"}, "quote": {"type": "string"}},
                    "required": ["value", "quote"],
                },
                {"type": "null"},
            ],
        },
        "peaks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "band": {"type": "string", "enum": ["g", "r"]},
                    "time_days": _NULLABLE_NUMBER,
                    "magnitude": _NULLABLE_NUMBER,
                    "quote": {"type": "string"},
                },
                "required": ["band", "time_days", "magnitude", "quote"],
            },
        },
        "gaps": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "start_days": _NULLABLE_NUMBER,
                    "end_days": _NULLABLE_NUMBER,
                    "duration_days": _NULLABLE_NUMBER,
                    "quote": {"type": "string"},
                },
                "required": ["start_days", "end_days", "duration_days", "quote"],
            },
        },
    },
    "required": ["duration_days", "peaks", "gaps"],
}


def claims_from_dict(payload: dict) -> CaptionNumericClaims:
    if not isinstance(payload, dict):
        raise ValueError("structured extraction must be a JSON object")
    unknown = set(payload) - {"duration_days", "peaks", "gaps"}
    if unknown:
        raise ValueError(f"structured extraction has unknown fields: {sorted(unknown)}")

    def optional_float(value, name: str) -> float | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be numeric or null")
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
        return value

    def quote(item: dict) -> str:
        value = item.get("quote")
        if not isinstance(value, str) or not value:
            raise ValueError("every extracted claim requires a non-empty quote")
        return value

    duration_payload = payload.get("duration_days")
    duration = None
    if duration_payload is not None:
        duration = NumericClaim(
            value=optional_float(duration_payload.get("value"), "duration_days.value"),  # type: ignore[arg-type]
            quote=quote(duration_payload),
        )
        if duration.value is None:
            raise ValueError("duration_days.value cannot be null")

    peaks = []
    for item in payload.get("peaks", []):
        if item.get("band") not in {"g", "r"}:
            raise ValueError("peak band must be g or r")
        peaks.append(PeakClaim(
            band=item["band"],
            time_days=optional_float(item.get("time_days"), "peak.time_days"),
            magnitude=optional_float(item.get("magnitude"), "peak.magnitude"),
            quote=quote(item),
        ))
    gaps = [
        GapClaim(
            start_days=optional_float(item.get("start_days"), "gap.start_days"),
            end_days=optional_float(item.get("end_days"), "gap.end_days"),
            duration_days=optional_float(item.get("duration_days"), "gap.duration_days"),
            quote=quote(item),
        )
        for item in payload.get("gaps", [])
    ]
    return CaptionNumericClaims(duration_days=duration, peaks=peaks, gaps=gaps)


def build_extraction_prompt(caption: str) -> str:
    if not caption or not caption.strip():
        raise ValueError("caption must be a non-empty string")
    return EXTRACTION_USER_TEMPLATE.format(caption=caption)


def extraction_contract(
    benchmark_protocol_fingerprint: str,
    *,
    model: str = DEFAULT_EXTRACTOR_MODEL,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
) -> dict:
    return {
        "extractor_version": EXTRACTOR_VERSION,
        "benchmark_protocol_fingerprint": benchmark_protocol_fingerprint,
        "model": model,
        "max_output_tokens": max_output_tokens,
        "thinking_level": "low",
        "tools": [],
        "system_prompt": EXTRACTION_SYSTEM_PROMPT,
        "user_template": EXTRACTION_USER_TEMPLATE,
        "schema": CLAIMS_JSON_SCHEMA,
    }


def extraction_fingerprint(
    benchmark_protocol_fingerprint: str,
    *,
    model: str = DEFAULT_EXTRACTOR_MODEL,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
) -> str:
    return fingerprint_contract(extraction_contract(
        benchmark_protocol_fingerprint,
        model=model,
        max_output_tokens=max_output_tokens,
    ))


def scoring_contract(benchmark_protocol_fingerprint: str) -> dict:
    return {
        "scoring_version": SCORING_VERSION,
        "benchmark_protocol_fingerprint": benchmark_protocol_fingerprint,
        "duration": "absolute and relative error against retained-observation span",
        "peaks": "absolute time and magnitude errors against maximum signed FLUXCAL per band",
        "gaps": (
            "requires both claimed start and end; match by minimum summed endpoint distance and "
            "report endpoint errors; duration-only claims are unscorable"
        ),
    }


def scoring_fingerprint(benchmark_protocol_fingerprint: str) -> str:
    return fingerprint_contract(scoring_contract(benchmark_protocol_fingerprint))


def claims_to_dict(claims: CaptionNumericClaims) -> dict:
    return asdict(claims)


def validate_claim_quotes(caption: str, claims: CaptionNumericClaims) -> None:
    quoted = []
    if claims.duration_days is not None:
        quoted.append(claims.duration_days.quote)
    quoted.extend(claim.quote for claim in claims.peaks)
    quoted.extend(claim.quote for claim in claims.gaps)
    invalid = [quote for quote in quoted if not quote or quote not in caption]
    if invalid:
        raise ValueError(f"extracted quotes must be exact caption substrings; invalid={invalid[:3]!r}")


def parse_prepared_table(text: str) -> list[dict]:
    lines = [line for line in text.splitlines() if line.strip()]
    expected_header = "TIME_DAYS BAND FLUXCAL FLUXCAL_ERR DIFF_MAG DIFF_MAG_ERR"
    if not lines or lines[0] != expected_header:
        raise ValueError("prepared light-curve table has an unexpected header")
    rows = []
    for line_number, line in enumerate(lines[1:], start=2):
        fields = line.split()
        if len(fields) != 6 or fields[1] not in {"g", "r"}:
            raise ValueError(f"invalid prepared table row {line_number}: {line!r}")
        rows.append({
            "time_days": float(fields[0]),
            "band": fields[1],
            "fluxcal": float(fields[2]),
            "fluxcal_error": float(fields[3]),
            "diff_mag": float(fields[4]),
            "diff_mag_error": float(fields[5]),
        })
    if not rows:
        raise ValueError("prepared light-curve table contains no observations")
    if [row["time_days"] for row in rows] != sorted(row["time_days"] for row in rows):
        raise ValueError("prepared light-curve table is not sorted by time")
    return rows


def calculate_table_facts(text: str) -> dict:
    rows = parse_prepared_table(text)
    times = [row["time_days"] for row in rows]
    peaks = {}
    for band in ("g", "r"):
        band_rows = [row for row in rows if row["band"] == band]
        if band_rows:
            # Signed flux is authoritative.  Ties retain the earlier sorted observation.
            peak = max(band_rows, key=lambda row: row["fluxcal"])
            peaks[band] = dict(peak)
    gaps = [
        {"start_days": left, "end_days": right, "duration_days": right - left}
        for left, right in zip(times, times[1:])
    ]
    return {
        "duration_days": times[-1] - times[0],
        "observation_count": len(rows),
        "peaks": peaks,
        "gaps": gaps,
        "rows": rows,
    }


def _finite(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def score_numeric_claims(claims: CaptionNumericClaims, facts: dict) -> dict:
    duration = None
    if claims.duration_days is not None:
        claimed = claims.duration_days.value
        actual = facts["duration_days"]
        duration = {
            "claim": asdict(claims.duration_days),
            "actual_days": actual,
            "absolute_error_days": abs(claimed - actual),
            "relative_error": abs(claimed - actual) / actual if actual else None,
        }

    peaks = []
    for claim in claims.peaks:
        actual = facts["peaks"].get(claim.band)
        item = {"claim": asdict(claim), "actual": actual}
        if actual is None:
            item["status"] = "band_absent"
        else:
            item["status"] = "compared"
            if claim.time_days is not None:
                item["absolute_time_error_days"] = abs(claim.time_days - actual["time_days"])
            if claim.magnitude is not None:
                actual_mag = _finite(actual["diff_mag"])
                item["absolute_magnitude_error"] = (
                    abs(claim.magnitude - actual_mag) if actual_mag is not None else None
                )
        peaks.append(item)

    gaps = []
    for claim in claims.gaps:
        item = {"claim": asdict(claim)}
        actual_gaps = facts["gaps"]
        if not actual_gaps:
            item.update(status="no_actual_gap", matched_actual=None)
        elif claim.start_days is not None and claim.end_days is not None:
            matched = min(
                actual_gaps,
                key=lambda gap: abs(claim.start_days - gap["start_days"]) + abs(claim.end_days - gap["end_days"]),
            )
            item.update({
                "status": "compared",
                "matched_actual": matched,
                "absolute_start_error_days": abs(claim.start_days - matched["start_days"]),
                "absolute_end_error_days": abs(claim.end_days - matched["end_days"]),
            })
            item["maximum_endpoint_error_days"] = max(
                item["absolute_start_error_days"], item["absolute_end_error_days"],
            )
            if claim.duration_days is not None:
                item["absolute_duration_error_days"] = abs(claim.duration_days - matched["duration_days"])
        elif claim.duration_days is not None:
            item.update({
                "status": "unscorable_without_gap_endpoints",
                "matched_actual": None,
            })
        else:
            item.update(status="insufficient_numeric_anchor", matched_actual=None)
        gaps.append(item)

    return {"duration": duration, "peaks": peaks, "gaps": gaps}


def select_manual_review_objects(
    manifest_objects: list[dict],
    *,
    seed: int = MANUAL_REVIEW_SEED,
    target_counts: dict[str, int] | None = None,
    initial_ids: tuple[str, ...] = INITIAL_SEED42_REVIEW_IDS,
) -> list[dict]:
    """Expand the original seed-42 sample without consulting model outputs or correctness."""
    targets = dict(target_counts or MANUAL_REVIEW_COUNTS)
    if set(targets) != set(SN_LABELS):
        raise ValueError("manual-review targets must contain exactly the three benchmark classes")
    by_id = {str(record["object_id"]): record for record in manifest_objects}
    if len(by_id) != len(manifest_objects):
        raise ValueError("manifest contains duplicate object IDs")
    missing = [object_id for object_id in initial_ids if object_id not in by_id]
    if missing:
        raise ValueError(f"original seed-42 review IDs are absent from the manifest: {missing}")

    selected_ids = list(initial_ids)
    selected = set(selected_ids)
    rng = random.Random(seed)
    for label in SN_LABELS:
        have = sum(str(by_id[object_id]["true_label"]) == label for object_id in selected_ids)
        needed = targets[label] - have
        if needed < 0:
            raise ValueError(f"initial review sample already exceeds target for {label}")
        candidates = sorted(
            object_id for object_id, record in by_id.items()
            if object_id not in selected and str(record["true_label"]) == label
        )
        if len(candidates) < needed:
            raise ValueError(f"not enough remaining {label} objects for manual review")
        additions = rng.sample(candidates, needed)
        selected_ids.extend(additions)
        selected.update(additions)
    rng.shuffle(selected_ids)
    return [by_id[object_id] for object_id in selected_ids]


def build_manual_validation_report(
    manual_objects: list[dict],
    extraction_records: dict[str, dict],
) -> dict:
    """Record the completed human transcription check at object and claim-type level."""
    selected_ids = [str(record["object_id"]) for record in manual_objects]
    if set(selected_ids) != set(MANUAL_VALIDATED_CLAIM_COUNTS):
        raise ValueError("manual validation annotations do not match the fixed 20-object sample")

    totals = {
        name: {
            "explicit_claims": 0,
            "correctly_extracted": 0,
            "unsupported_extractions": 0,
            "missed_claims": 0,
        }
        for name in ("duration", "peaks", "gaps")
    }
    annotations = []
    for review_order, manifest_record in enumerate(manual_objects, start=1):
        object_id = str(manifest_record["object_id"])
        extraction = extraction_records.get(object_id)
        if extraction is None or extraction.get("failure_reason") is not None:
            raise ValueError(f"manual validation object={object_id} lacks a successful extraction")
        claims = extraction.get("extracted_claims")
        if not isinstance(claims, dict):
            raise ValueError(f"manual validation object={object_id} lacks structured claims")
        observed = {
            "duration": int(claims.get("duration_days") is not None),
            "peaks": len(claims.get("peaks", [])),
            "gaps": len(claims.get("gaps", [])),
        }
        expected = MANUAL_VALIDATED_CLAIM_COUNTS[object_id]
        if observed != expected:
            raise ValueError(
                f"manual validation object={object_id} has extracted counts {observed}, "
                f"but the reviewed extraction has {expected}"
            )
        claim_annotations = {}
        for name, count in expected.items():
            result = {
                "explicit_claims": count,
                "correctly_extracted": count,
                "unsupported_extractions": 0,
                "missed_claims": 0,
            }
            claim_annotations[name] = result
            for field, value in result.items():
                totals[name][field] += value
        annotations.append({
            "review_order": review_order,
            "object_id": object_id,
            "true_label": str(manifest_record["true_label"]),
            "claims": claim_annotations,
        })

    return {
        "status": "single-reviewer_manual_validation",
        "sample": "fixed_seed42_class_balanced_n20",
        "scope": (
            "Check whether Gemini transcribed every explicit duration, peak, and observation-gap "
            "claim with an exact supporting quote. Scientific correctness is scored separately "
            "from retained numerical tables."
        ),
        "objects": annotations,
        "totals": totals,
    }


def caption_sha256(caption: str) -> str:
    return sha256_bytes(caption.encode("utf-8"))
