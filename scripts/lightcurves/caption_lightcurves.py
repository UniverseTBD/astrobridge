"""Generate ATCAT-aligned transient captions with Gemini.

This is the portable part of the Exp002 Run014 caption generator. It loads the
published transient dataset, applies the same row selection used by ATCAT,
builds the final class-conditioned prompt and chart, and optionally submits the
prepared requests to Gemini. Preparation is the default; API use requires the
explicit ``--execute`` flag and a matching approval environment variable.

Run from the repository root, for example::

    python -m scripts.lightcurves.caption_lightcurves \
        --output-dir outputs/lightcurve_captions --limit 10

The original Exp002 run also supplied private literature passages. A portable
run can omit them, or provide a JSONL file with one ``object_id`` and ``context``
string per row through ``--context-jsonl``.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from captioner.data.transients_dataset import (
    ATCAT_SEQ_LEN,
    load_transients_table,
    prepare_lightcurve_arrays,
)


DATASET_ID = "BuildNg/astrobridge-transients-dataset"
DATASET_REVISION = "9eb915d5c025d7c98dcbe6e7d7fa98b31551a441"
MODEL = "gemini-3.7-flash"
APPROVAL_ENV = "ASTROBRIDGE_GEMINI_APPROVED_RUN_ID"
VALID_CLASSES = frozenset({"SN Ia", "SN II", "SN Ibc"})
PROMPT_VERSION = "atcat-aligned-transient-caption-v1"
CHART_VERSION = "atcat-aligned-lightcurve-chart-v1"
MAX_CONTEXT_CHARACTERS = 25_000
MEDIA_TOKENS_HIGH_RESOLUTION = 1_120
REQUEST_OVERHEAD_TOKENS = 128


OUTPUT_INSTRUCTION = """Return only a self-contained caption of 2-4 sentences. Use the first sentences to describe the observed photometric evolution without stating a class. In the final sentence, connect only the photometric features already described to the required broad class and conclude using wording of the form '...is consistent with the [class] class.' Never use 'confirms', 'proves', 'establishes', or similarly strong wording. Do not invent, exaggerate, or selectively omit observations to make the required class appear better supported. If the features are not individually diagnostic, state this explicitly.

If the supplied light-curve evidence does not support a meaningful object-specific description, return exactly INSUFFICIENT_PHOTOMETRIC_DATA."""


PROMPT_TEMPLATE = """You are an expert time-domain astrophysicist. Write a concise scientific caption describing the optical light curve of one transient.

The required broad class is:
<target_class>
{target_class}
</target_class>

The class will be exactly one of SN Ia, SN II, or SN Ibc. It is an output constraint, not observational evidence. Do not mention that it was supplied, required, catalogued, or obtained from metadata. Do not state another class or a more specific subtype.

The following deterministic summary was calculated only from the retained light-curve rows below:
<photometry_summary>
{summary}
</photometry_summary>

The complete retained light-curve table is:
<light_curve>
{table}
</light_curve>

TIME_DAYS is measured from the first retained row. FLUXCAL and FLUXCAL_ERR use the SNANA zero point 27.5. FLUXCAL is signed difference flux. DIFF_MAG is 27.5 - 2.5 log10(abs(FLUXCAL)), and DIFF_MAG_ERR is its propagated uncertainty. The flux sign remains authoritative for whether a residual is positive or negative.

An image of exactly the same retained rows is attached. Use it to inspect the overall morphology and use the table as the authority for exact measurements. The image has an inverted difference-magnitude axis, separates g and r, shows uncertainty, and distinguishes positive and negative residuals. The summary, table, and image contain no independent measurements beyond one another.

{context_block}
Keep the caption focused on the retained light curve. Do not report an object or host name, coordinate, redshift, distance, absolute magnitude, spectrum, spectral feature, multiwavelength observation, progenitor, explosion mechanism, or event phase. Do not present any such information as the reason for the final classification.

Use the following general guidance cautiously:
- SN Ia light curves commonly show a relatively smooth rise and decline. The bluer band may fade faster after the brightest sampled region, while the redder band may show a slower decline or shoulder.
- SN II light curves may show a prolonged plateau, a sustained slow decline, or a longer-lasting evolution.
- SN Ibc light curves commonly show a comparatively rapid, single-peaked rise and decline without a prolonged plateau, but their photometric evolution can overlap substantially with SN Ia and SN II.
- These are population-level tendencies, not diagnostic rules. A missing feature is evidence only when the relevant phase was adequately sampled. Sparse or gapped coverage means that the feature is unknown, not absent.

Describe only supported behavior such as the retained time span, sampled rise or decline, changes in slope, shoulders or multiple sampled maxima, band-dependent evolution, gaps, negative residuals, and measurement uncertainty. Refer to the 'brightest sampled point' rather than the true peak unless the rows clearly sample a turnover on both sides. Compare bands as colour information only when their measurements are sufficiently close in time.

{output_instruction}"""


PRIVATE_CONTEXT_TEMPLATE = """Additional literature passages are supplied below:
<private_context>
{context}
</private_context>

The private context is background only. It may guide which behavior to inspect in the retained light curve, but no output claim about a photometric value, time, duration, upper limit, band, light-curve behavior, or sampling may rely on it. Every factual or quantitative light-curve claim in the caption must be directly checkable from the retained table. Do not mention the private context or its sources.

"""


@dataclass(frozen=True)
class RetainedPoint:
    time_days: float
    band: str
    fluxcal: float
    fluxcal_err: float
    diff_mag: float | None
    diff_mag_err: float | None


@dataclass(frozen=True)
class PreparedCaption:
    object_id: str
    target_class: str
    prompt: str
    prompt_sha256: str
    chart_path: Path
    chart_sha256: str
    retained_rows: int
    accepted_rows: int
    rows_in_window: int
    downsampled: bool


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Malformed JSONL at {path}:{line_number}") from error
        if not isinstance(row, dict):
            raise ValueError(f"Expected a JSON object at {path}:{line_number}")
        rows.append(row)
    return rows


def _append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    payload = "".join(
        json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    )
    _atomic_write(path, payload.encode("utf-8"))


def _load_contexts(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    contexts: dict[str, str] = {}
    for row in _read_jsonl(path):
        object_id = str(row.get("object_id", "")).strip()
        context = str(row.get("context", "")).strip()
        if not object_id or not context:
            raise ValueError("Each context row needs non-empty object_id and context fields")
        if object_id in contexts:
            raise ValueError(f"Duplicate context for {object_id}")
        if len(context) > MAX_CONTEXT_CHARACTERS:
            raise ValueError(
                f"Context for {object_id} exceeds {MAX_CONTEXT_CHARACTERS:,} characters"
            )
        contexts[object_id] = context
    return contexts


def retained_points(row: Mapping[str, Any]) -> tuple[tuple[RetainedPoint, ...], dict[str, int]]:
    """Return exactly the unpadded rows selected for ATCAT."""
    object_id = str(row["object_id"])
    arrays, info = prepare_lightcurve_arrays(
        row["lc_mjd"],
        row["atcat_flux"],
        row["atcat_flux_error"],
        row["atcat_band_id"],
        row["atcat_use"],
        object_id=object_id,
    )
    count = int(info["n_selected"])
    points: list[RetainedPoint] = []
    for time_days, flux, error, band_id in zip(
        arrays["time"][:count],
        arrays["flux"][:count],
        arrays["flux_err"][:count],
        arrays["channel_index"][:count],
        strict=True,
    ):
        absolute_flux = abs(float(flux))
        magnitude = None
        magnitude_error = None
        if absolute_flux > 0:
            magnitude = 27.5 - 2.5 * math.log10(absolute_flux)
            magnitude_error = (2.5 / math.log(10.0)) * float(error) / absolute_flux
        try:
            band = {1: "g", 2: "r"}[int(band_id)]
        except KeyError as error:
            raise ValueError(
                f"Selected row for {object_id} has unsupported ATCAT band {int(band_id)}"
            ) from error
        points.append(
            RetainedPoint(
                time_days=float(time_days),
                band=band,
                fluxcal=float(flux),
                fluxcal_err=float(error),
                diff_mag=magnitude,
                diff_mag_err=magnitude_error,
            )
        )
    if not points:
        raise ValueError(f"No retained light-curve rows for {object_id}")
    return tuple(points), info


def format_table(points: Sequence[RetainedPoint]) -> str:
    lines = ["TIME_DAYS BAND FLUXCAL FLUXCAL_ERR DIFF_MAG DIFF_MAG_ERR"]
    for point in points:
        magnitude = "NA" if point.diff_mag is None else f"{point.diff_mag:.4f}"
        magnitude_error = "NA" if point.diff_mag_err is None else f"{point.diff_mag_err:.4f}"
        lines.append(
            f"{point.time_days:.5f} {point.band} {point.fluxcal:.6f} "
            f"{point.fluxcal_err:.6f} {magnitude} {magnitude_error}"
        )
    return "\n".join(lines)


def build_summary(points: Sequence[RetainedPoint]) -> str:
    counts = {band: sum(point.band == band for point in points) for band in ("g", "r")}
    gaps = [
        (right.time_days - left.time_days, left.time_days, right.time_days)
        for left, right in zip(points, points[1:])
    ]
    largest_gap = max(gaps, default=(0.0, 0.0, 0.0))
    brightest: list[str] = []
    for band in ("g", "r"):
        candidates = [
            point
            for point in points
            if point.band == band and point.fluxcal > 0 and point.diff_mag is not None
        ]
        if not candidates:
            brightest.append(f"{band}: no retained positive-residual point")
            continue
        point = min(candidates, key=lambda item: (float(item.diff_mag), item.time_days))
        brightest.append(
            f"{band}: {point.diff_mag:.4f} +/- {point.diff_mag_err:.4f} "
            f"at day {point.time_days:.5f}"
        )
    return "\n".join(
        (
            f"Retained baseline: {points[-1].time_days:.5f} days, from day 0.00000 "
            f"to day {points[-1].time_days:.5f}.",
            f"Retained observations by band: g={counts['g']}, r={counts['r']}.",
            f"Largest gap between consecutive retained rows: {largest_gap[0]:.5f} days, "
            f"from day {largest_gap[1]:.5f} to day {largest_gap[2]:.5f}.",
            "Brightest retained positive-residual point by band: "
            + "; ".join(brightest)
            + ".",
        )
    )


def build_prompt(
    points: Sequence[RetainedPoint], target_class: str, context: str | None = None
) -> str:
    if target_class not in VALID_CLASSES:
        raise ValueError(f"Unsupported class label: {target_class!r}")
    context_block = ""
    if context:
        context_block = PRIVATE_CONTEXT_TEMPLATE.format(context=context.strip())
    return PROMPT_TEMPLATE.format(
        target_class=target_class,
        summary=build_summary(points),
        table=format_table(points),
        context_block=context_block,
        output_instruction=OUTPUT_INSTRUCTION,
    )


def render_chart(points: Sequence[RetainedPoint]) -> bytes:
    """Render the same retained rows shown in the prompt table."""
    with matplotlib.rc_context(
        {"font.family": "DejaVu Sans", "figure.dpi": 180, "savefig.dpi": 180}
    ):
        figure, axes = plt.subplots(figsize=(8.4, 5.2), constrained_layout=True)
        for band, color in (("g", "#239b56"), ("r", "#e53935")):
            for positive in (True, False):
                selected = [
                    point
                    for point in points
                    if point.band == band
                    and (point.fluxcal > 0) == positive
                    and point.diff_mag is not None
                ]
                if selected:
                    axes.errorbar(
                        [point.time_days for point in selected],
                        [point.diff_mag for point in selected],
                        yerr=[point.diff_mag_err for point in selected],
                        fmt="o",
                        ms=5.5,
                        color=color,
                        markerfacecolor=color if positive else "white",
                        markeredgecolor=color,
                        markeredgewidth=1.3,
                        ecolor=color,
                        elinewidth=0.8,
                        capsize=1.8,
                        alpha=0.94,
                    )
        axes.invert_yaxis()
        axes.set_xlabel("Days from first retained observation")
        axes.set_ylabel("Difference magnitude (zero point 27.5)")
        axes.set_title("Retained g/r light curve", loc="left", fontweight="bold")
        axes.grid(True, color="#dbe4ef", linewidth=0.7, alpha=0.75)
        axes.legend(
            handles=[
                Line2D([0], [0], color="#239b56", marker="o", linestyle="", label="g band"),
                Line2D([0], [0], color="#e53935", marker="o", linestyle="", label="r band"),
                Line2D(
                    [0],
                    [0],
                    color="#111827",
                    marker="o",
                    markerfacecolor="#111827",
                    linestyle="",
                    label="positive residual",
                ),
                Line2D(
                    [0],
                    [0],
                    color="#111827",
                    marker="o",
                    markerfacecolor="white",
                    linestyle="",
                    label="negative residual",
                ),
            ],
            frameon=False,
            ncols=2,
        )
        buffer = io.BytesIO()
        figure.savefig(
            buffer,
            format="png",
            facecolor="white",
            metadata={"Software": "AstroBridge", "Creation Time": None},
        )
        plt.close(figure)
    return buffer.getvalue()


def prepare_items(args: argparse.Namespace) -> list[PreparedCaption]:
    frame = load_transients_table(args.dataset, args.revision, args.cache_dir)
    if args.object_id:
        requested = list(dict.fromkeys(args.object_id))
        available = set(frame["object_id"].astype(str))
        missing = sorted(set(requested) - available)
        if missing:
            raise ValueError(f"Unknown object IDs: {missing}")
        order = {object_id: index for index, object_id in enumerate(requested)}
        frame = frame[frame["object_id"].astype(str).isin(requested)].copy()
        frame["_order"] = frame["object_id"].astype(str).map(order)
        frame = frame.sort_values("_order")
    else:
        frame = frame.sort_values("object_id")
    if args.limit is not None:
        frame = frame.head(args.limit)
    if frame.empty:
        raise ValueError("No objects selected")

    contexts = _load_contexts(args.context_jsonl)
    output_dir = args.output_dir.resolve()
    chart_dir = output_dir / "charts"
    items: list[PreparedCaption] = []
    for row in frame.astype(object).where(frame.notna(), None).to_dict(orient="records"):
        object_id = str(row["object_id"])
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", object_id):
            raise ValueError(f"Unsafe object ID for chart filename: {object_id!r}")
        target_class = str(row["class_label"])
        points, info = retained_points(row)
        prompt = build_prompt(points, target_class, contexts.get(object_id))
        chart = render_chart(points)
        chart_path = chart_dir / f"{object_id}.png"
        _atomic_write(chart_path, chart)
        items.append(
            PreparedCaption(
                object_id=object_id,
                target_class=target_class,
                prompt=prompt,
                prompt_sha256=_sha256_text(prompt),
                chart_path=chart_path,
                chart_sha256=_sha256_bytes(chart),
                retained_rows=len(points),
                accepted_rows=int(info["n_accepted"]),
                rows_in_window=int(info["n_in_window"]),
                downsampled=bool(info["downsampled"]),
            )
        )

    _write_jsonl(
        output_dir / "prepared.jsonl",
        (
            {
                "schema_version": 1,
                "object_id": item.object_id,
                "target_class": item.target_class,
                "prompt": item.prompt,
                "prompt_sha256": item.prompt_sha256,
                "chart_path": str(item.chart_path.relative_to(output_dir)),
                "chart_sha256": item.chart_sha256,
                "retained_rows": item.retained_rows,
                "accepted_rows": item.accepted_rows,
                "rows_in_window": item.rows_in_window,
                "downsampled": item.downsampled,
                "dataset": args.dataset,
                "dataset_revision": args.revision,
                "prompt_version": PROMPT_VERSION,
                "chart_version": CHART_VERSION,
                "atcat_sequence_length": ATCAT_SEQ_LEN,
                "private_context_supplied": item.object_id in contexts,
            }
            for item in items
        ),
    )
    return items


def _load_token(credentials_env_file: Path | None) -> str:
    token = os.environ.get("GEMINI_API_TOKEN", "").strip()
    if token:
        return token
    if credentials_env_file is not None:
        for line in credentials_env_file.read_text(encoding="utf-8").splitlines():
            if "=" not in line or line.lstrip().startswith("#"):
                continue
            name, value = line.split("=", 1)
            if name.strip() == "GEMINI_API_TOKEN" and value.strip().strip("\"'"):
                return value.strip().strip("\"'")
    raise RuntimeError(
        "GEMINI_API_TOKEN is not set and was not found in --credentials-env-file"
    )


def _usage(response: Any, args: argparse.Namespace) -> dict[str, Any]:
    usage = response.usage_metadata
    input_tokens = int(getattr(usage, "prompt_token_count", 0) or 0)
    output_tokens = int(getattr(usage, "candidates_token_count", 0) or 0)
    thinking_tokens = int(getattr(usage, "thoughts_token_count", 0) or 0)
    cost = (
        input_tokens * args.input_price_per_million_usd
        + (output_tokens + thinking_tokens) * args.output_price_per_million_usd
    ) / 1_000_000
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "thinking_tokens": thinking_tokens,
        "cached_tokens": int(getattr(usage, "cached_content_token_count", 0) or 0),
        "total_tokens": int(getattr(usage, "total_token_count", 0) or 0),
        "estimated_cost_usd": round(cost, 8),
        "cost_is_upper_bound": False,
    }


def _finish_reason(response: Any) -> str:
    candidates = getattr(response, "candidates", None)
    if not candidates:
        return ""
    reason = getattr(candidates[0], "finish_reason", None)
    return str(getattr(reason, "name", reason) or "")


def execute(items: Sequence[PreparedCaption], args: argparse.Namespace) -> None:
    if not args.run_id:
        raise ValueError("--run-id is required with --execute")
    if os.environ.get(APPROVAL_ENV) != args.run_id:
        raise RuntimeError(
            f"Set {APPROVAL_ENV} to the reviewed --run-id before API execution"
        )
    for name in (
        "input_price_per_million_usd",
        "output_price_per_million_usd",
        "hard_usd_stop",
    ):
        value = getattr(args, name)
        if value is None or value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive with --execute")
    if args.max_output_tokens <= 0:
        raise ValueError("--max-output-tokens must be positive")
    if args.max_requests is not None and args.max_requests < 1:
        raise ValueError("--max-requests must be positive")
    if args.usage_ledger is None:
        raise ValueError("--usage-ledger is required with --execute")

    prior_results = _read_jsonl(args.output_dir / "captions.jsonl")
    completed: dict[str, dict[str, Any]] = {}
    for row in prior_results:
        if row.get("run_id") != args.run_id or row.get("status") != "ok":
            continue
        object_id = str(row["object_id"])
        if object_id in completed:
            raise RuntimeError(f"Duplicate successful result for {object_id}")
        completed[object_id] = row
    by_id = {item.object_id: item for item in items}
    for object_id, result in completed.items():
        item = by_id.get(object_id)
        if item is not None and (
            result.get("prompt_sha256") != item.prompt_sha256
            or result.get("chart_sha256") != item.chart_sha256
        ):
            raise RuntimeError(f"Completed input hash mismatch for {object_id}")
    pending = [item for item in items if item.object_id not in completed]
    if args.max_requests is not None:
        pending = pending[: args.max_requests]
    if not pending:
        print("All selected objects already have successful results.")
        return

    ledger = [
        row for row in _read_jsonl(args.usage_ledger) if row.get("run_id") == args.run_id
    ]
    spent = sum(float(row.get("estimated_cost_usd", 0.0)) for row in ledger)
    token = _load_token(args.credentials_env_file)
    try:
        from google import genai
        from google.genai import types
    except ImportError as error:
        raise RuntimeError("Install the repository dependencies before API execution") from error

    client = genai.Client(
        api_key=token,
        http_options=types.HttpOptions(api_version="v1beta"),
    )
    try:
        for item in pending:
            image = item.chart_path.read_bytes()
            if _sha256_bytes(image) != item.chart_sha256:
                raise RuntimeError(f"Chart hash mismatch for {item.object_id}")
            input_upper_bound = (
                len(item.prompt.encode("utf-8"))
                + MEDIA_TOKENS_HIGH_RESOLUTION
                + REQUEST_OVERHEAD_TOKENS
            )
            worst_cost = (
                input_upper_bound * args.input_price_per_million_usd
                + args.max_output_tokens * args.output_price_per_million_usd
            ) / 1_000_000
            if spent + worst_cost > args.hard_usd_stop:
                raise RuntimeError(
                    f"Hard USD stop would be exceeded before {item.object_id}: "
                    f"spent={spent:.6f}, reserved={worst_cost:.6f}, "
                    f"limit={args.hard_usd_stop:.6f}"
                )

            timestamp = datetime.now(timezone.utc).isoformat()
            response: Any | None = None
            try:
                response = client.models.generate_content(
                    model=MODEL,
                    contents=[
                        item.prompt,
                        types.Part.from_bytes(data=image, mime_type="image/png"),
                    ],
                    config=types.GenerateContentConfig(
                        service_tier="standard",
                        media_resolution=types.MediaResolution.MEDIA_RESOLUTION_HIGH,
                        response_mime_type="text/plain",
                        response_modalities=["TEXT"],
                        thinking_config=types.ThinkingConfig(thinking_level="low"),
                        max_output_tokens=args.max_output_tokens,
                    ),
                )
                usage = _usage(response, args)
                spent += float(usage["estimated_cost_usd"])
                raw_caption = response.text or ""
                finish_reason = _finish_reason(response)
                status = "ok" if raw_caption.strip() and finish_reason == "STOP" else "invalid"
                ledger_row = {
                    "timestamp_utc": timestamp,
                    "run_id": args.run_id,
                    "object_id": item.object_id,
                    "model": getattr(response, "model_version", None) or MODEL,
                    "request_id": getattr(response, "response_id", None) or "",
                    "status": status,
                    "finish_reason": finish_reason,
                    "prompt_sha256": item.prompt_sha256,
                    "chart_sha256": item.chart_sha256,
                    **usage,
                }
                _append_jsonl(args.usage_ledger, ledger_row)
                _append_jsonl(
                    args.output_dir / "captions.jsonl",
                    {
                        **ledger_row,
                        "target_class": item.target_class,
                        "caption": raw_caption.strip(),
                        "raw_response": raw_caption,
                    },
                )
            except Exception as error:
                spent += worst_cost
                failure = {
                    "timestamp_utc": timestamp,
                    "run_id": args.run_id,
                    "object_id": item.object_id,
                    "model": getattr(response, "model_version", None) or MODEL,
                    "request_id": getattr(response, "response_id", None) or "",
                    "status": f"error:{type(error).__name__}",
                    "finish_reason": _finish_reason(response) if response is not None else "",
                    "prompt_sha256": item.prompt_sha256,
                    "chart_sha256": item.chart_sha256,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "thinking_tokens": 0,
                    "cached_tokens": 0,
                    "total_tokens": 0,
                    "estimated_cost_usd": round(worst_cost, 8),
                    "cost_is_upper_bound": True,
                    "error": str(error),
                }
                _append_jsonl(args.usage_ledger, failure)
                _append_jsonl(args.output_dir / "captions.jsonl", failure)
                raise
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DATASET_ID)
    parser.add_argument("--revision", default=DATASET_REVISION)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--object-id", action="append", help="Repeat to select exact objects")
    parser.add_argument("--limit", type=int, help="Keep the first N selected objects")
    parser.add_argument(
        "--context-jsonl",
        type=Path,
        help="Optional JSONL with object_id and private literature context fields",
    )
    parser.add_argument("--execute", action="store_true", help="Submit prepared requests")
    parser.add_argument("--run-id")
    parser.add_argument("--credentials-env-file", type=Path)
    parser.add_argument("--usage-ledger", type=Path)
    parser.add_argument("--max-requests", type=int)
    parser.add_argument("--max-output-tokens", type=int, default=512)
    parser.add_argument("--input-price-per-million-usd", type=float)
    parser.add_argument("--output-price-per-million-usd", type=float)
    parser.add_argument("--hard-usd-stop", type=float)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    args.output_dir = args.output_dir.resolve()
    for name in ("cache_dir", "context_jsonl", "credentials_env_file", "usage_ledger"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.resolve())
    items = prepare_items(args)
    print(
        f"Prepared {len(items)} caption input(s) in {args.output_dir.resolve()} "
        f"using {PROMPT_VERSION}."
    )
    if args.execute:
        execute(items, args)
    else:
        print("No API request was made. Add --execute only after the run is approved.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
