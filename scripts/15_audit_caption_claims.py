#!/usr/bin/env python
"""Extract numerical caption claims with Gemini and check them against saved YSE tables.

The default action is local preparation only.  API execution requires ``--run-api`` plus explicit
pricing and a hard USD stop.  The model sees only each saved caption; it never sees a table, chart,
true class, or qualitative-review decision.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import statistics

from captioner.utils.config import load_config, remaining_argv
from eval.benchmark_artifacts import load_yse_manifest, manifest_records_by_id
from eval.benchmark_caption_claims import (
    CaptionNumericClaims,
    CLAIMS_JSON_SCHEMA,
    DEFAULT_EXTRACTOR_MODEL,
    DEFAULT_MAX_OUTPUT_TOKENS,
    EXTRACTOR_VERSION,
    EXTRACTION_SYSTEM_PROMPT,
    SCORING_VERSION,
    build_manual_validation_report,
    build_extraction_prompt,
    calculate_table_facts,
    caption_sha256,
    claims_to_dict,
    claims_from_dict,
    extraction_contract,
    extraction_fingerprint,
    score_numeric_claims,
    scoring_contract,
    scoring_fingerprint,
    select_manual_review_objects,
    validate_claim_quotes,
)
from eval.benchmark_generation import validate_generation_records
from eval.benchmark_protocol import (
    ATCAT_REVISION,
    GENERATIVE_ROUTES,
    JsonlLedger,
    atomic_write_json,
    benchmark_protocol_fingerprint,
    require_successful_generation_records,
    sha256_bytes,
)


def _append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _usage_value(metadata, name: str) -> int:
    value = getattr(metadata, name, 0) if metadata is not None else 0
    return int(value or 0)


def _estimated_cost(input_tokens: int, output_tokens: int, input_price: float, output_price: float) -> float:
    return input_tokens * input_price / 1_000_000 + output_tokens * output_price / 1_000_000


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = fraction * (len(ordered) - 1)
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _summarize(latest: dict[str, dict]) -> dict:
    successful = [record for record in latest.values() if record.get("failure_reason") is None]
    errors: dict[str, list[float]] = {
        "duration_absolute_error_days": [],
        "peak_absolute_time_error_days": [],
        "peak_absolute_magnitude_error": [],
        "gap_maximum_endpoint_error_days": [],
    }
    claim_counts = Counter()
    for record in successful:
        claims = record["extracted_claims"]
        claim_counts["duration"] += int(claims.get("duration_days") is not None)
        claim_counts["peaks"] += len(claims.get("peaks", []))
        claim_counts["gaps"] += len(claims.get("gaps", []))
        score = record["numeric_scores"]
        if score.get("duration") is not None:
            errors["duration_absolute_error_days"].append(score["duration"]["absolute_error_days"])
        for item in score.get("peaks", []):
            if item.get("absolute_time_error_days") is not None:
                errors["peak_absolute_time_error_days"].append(item["absolute_time_error_days"])
            if item.get("absolute_magnitude_error") is not None:
                errors["peak_absolute_magnitude_error"].append(item["absolute_magnitude_error"])
        for item in score.get("gaps", []):
            if item.get("maximum_endpoint_error_days") is not None:
                errors["gap_maximum_endpoint_error_days"].append(item["maximum_endpoint_error_days"])
    return {
        "objects_with_successful_extraction": len(successful),
        "claim_counts": dict(claim_counts),
        "error_summaries": {
            name: {
                "count": len(values),
                "mean": statistics.fmean(values) if values else None,
                "median": statistics.median(values) if values else None,
                "p90": _percentile(values, 0.90),
            }
            for name, values in errors.items()
        },
    }


def _write_manual_validation_if_complete(
    root: Path,
    manual_objects: list[dict],
    latest: dict[str, dict],
) -> Path | None:
    manual_ids = [str(record["object_id"]) for record in manual_objects]
    complete = all(
        object_id in latest
        and latest[object_id].get("failure_reason") is None
        and isinstance(latest[object_id].get("extracted_claims"), dict)
        for object_id in manual_ids
    )
    if not complete:
        return None
    path = root / "caption_claims" / "manual_extraction_validation_seed42_n20.json"
    atomic_write_json(path, build_manual_validation_report(manual_objects, latest))
    return path


def _validate_existing(
    latest: dict[str, dict],
    *,
    route: str,
    generation_records: dict[str, dict],
    manifest_by_id: dict[str, dict],
    benchmark_fingerprint: str,
    extractor_fingerprint: str,
    scoring_fingerprint_value: str | None = None,
) -> None:
    for object_id, record in latest.items():
        if object_id not in generation_records or object_id not in manifest_by_id:
            raise ValueError(f"claim ledger object={object_id!r} is outside the current benchmark")
        # A failed API attempt contains no scientific result.  Preserve it in the append-only
        # ledger but allow a corrected extraction protocol to retry the object under a new
        # fingerprint.  Successful records must always match the current protocol exactly.
        if (
            record.get("failure_reason") is not None
            and record.get("extraction_fingerprint") != extractor_fingerprint
        ):
            continue
        expected = {
            "route": route,
            "caption_sha256": caption_sha256(generation_records[object_id]["raw_answer"]),
            "table_sha256": manifest_by_id[object_id]["table_sha256"],
            "benchmark_protocol_fingerprint": benchmark_fingerprint,
            "extraction_fingerprint": extractor_fingerprint,
            "extractor_version": EXTRACTOR_VERSION,
        }
        for key, value in expected.items():
            if record.get(key) != value:
                raise ValueError(
                    f"stale claim ledger object={object_id!r}: {key}={record.get(key)!r}, expected {value!r}"
                )
        if record.get("failure_reason") is None:
            claims = claims_from_dict(record["extracted_claims"])
            validate_claim_quotes(generation_records[object_id]["raw_answer"], claims)
            if scoring_fingerprint_value is not None:
                scoring_expected = {
                    "scoring_version": SCORING_VERSION,
                    "scoring_fingerprint": scoring_fingerprint_value,
                }
                for key, value in scoring_expected.items():
                    if record.get(key) != value:
                        raise ValueError(
                            f"stale local score object={object_id!r}: "
                            f"{key}={record.get(key)!r}, expected {value!r}"
                        )


def _rescore_existing(
    ledger: JsonlLedger,
    latest: dict[str, dict],
    *,
    root: Path,
    manifest_by_id: dict[str, dict],
    extractor_fingerprint: str,
    scoring_fingerprint_value: str,
) -> int:
    rescored = 0
    for object_id, record in latest.items():
        if record.get("failure_reason") is not None:
            continue
        if record.get("extraction_fingerprint") != extractor_fingerprint:
            continue
        if record.get("scoring_fingerprint") == scoring_fingerprint_value:
            continue
        table_text = (root / manifest_by_id[object_id]["table_path"]).read_text(encoding="utf-8")
        observed_sha = sha256_bytes(table_text.encode("utf-8"))
        if observed_sha != manifest_by_id[object_id]["table_sha256"]:
            raise ValueError(f"object={object_id}: table checksum differs before deterministic rescore")
        claims = claims_from_dict(record["extracted_claims"])
        updated = dict(record)
        updated.update({
            "numeric_scores": score_numeric_claims(claims, calculate_table_facts(table_text)),
            "scoring_version": SCORING_VERSION,
            "scoring_fingerprint": scoring_fingerprint_value,
            "rescored_utc": datetime.now(timezone.utc).isoformat(),
        })
        ledger.append(updated)
        rescored += 1
    return rescored


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", default="outputs/benchmark_v8")
    parser.add_argument("--route", choices=GENERATIVE_ROUTES, default="astrobridge_v8_prefix")
    parser.add_argument("--scope", choices=("manual20", "all"), default="manual20")
    parser.add_argument("--run-api", action="store_true")
    parser.add_argument("--run-id")
    parser.add_argument("--model", default=DEFAULT_EXTRACTOR_MODEL)
    parser.add_argument("--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS)
    parser.add_argument("--credentials-env-file")
    parser.add_argument("--api-key-env", default="GEMINI_API_TOKEN")
    parser.add_argument("--input-price-per-million-usd", type=float)
    parser.add_argument("--output-price-per-million-usd", type=float)
    parser.add_argument("--hard-usd-stop", type=float)
    args = parser.parse_args(remaining_argv())

    root = Path(args.artifact_root)
    cfg = load_config("base", "data", "modalities", "model", "stage2")
    cfg.modalities.lightcurve.encoder.revision = ATCAT_REVISION
    benchmark_fingerprint = benchmark_protocol_fingerprint(cfg)
    manifest = load_yse_manifest(root, expected_protocol_fingerprint=benchmark_fingerprint)
    manifest_by_id = manifest_records_by_id(manifest)

    generation_path = root / "generations" / f"{args.route}.jsonl"
    generation_records = JsonlLedger(generation_path).latest()
    ordered_ids = [str(record["object_id"]) for record in manifest["objects"]]
    missing = [object_id for object_id in ordered_ids if object_id not in generation_records]
    if missing:
        raise ValueError(f"route={args.route} is missing {len(missing)} generation records")
    validate_generation_records(
        generation_records, args.route, manifest_by_id, root, benchmark_fingerprint,
    )
    require_successful_generation_records(
        [generation_records[object_id] for object_id in ordered_ids], args.route,
    )

    manual_objects = select_manual_review_objects(manifest["objects"])
    manual_ids = [str(record["object_id"]) for record in manual_objects]
    manual_payload = {
        "seed": 42,
        "selection_rule": (
            "Preserve the original seed-42 ten, then randomly sample from sorted remaining IDs "
            "without consulting generations or correctness."
        ),
        "target_class_counts": {"SN Ia": 7, "SN II": 7, "SN Ibc": 6},
        "benchmark_protocol_fingerprint": benchmark_fingerprint,
        "objects": [
            {
                "review_order": index,
                "object_id": object_id,
                "true_label": manifest_by_id[object_id]["true_label"],
                "chart_path": manifest_by_id[object_id]["chart_path"],
                "table_path": manifest_by_id[object_id]["table_path"],
                "raw_caption": generation_records[object_id]["raw_answer"],
                "final_sentence": generation_records[object_id]["final_sentence"],
                "parsed_label": generation_records[object_id]["parsed_label"],
            }
            for index, object_id in enumerate(manual_ids, start=1)
        ],
    }
    manual_path = root / "caption_claims" / "manual_review_seed42_n20.json"
    atomic_write_json(manual_path, manual_payload)

    extractor_fingerprint = extraction_fingerprint(
        benchmark_fingerprint, model=args.model, max_output_tokens=args.max_output_tokens,
    )
    contract_path = root / "caption_claims" / "extraction_protocol.json"
    atomic_write_json(contract_path, {
        "extraction_fingerprint": extractor_fingerprint,
        "contract": extraction_contract(
            benchmark_fingerprint, model=args.model, max_output_tokens=args.max_output_tokens,
        ),
    })
    scoring_fingerprint_value = scoring_fingerprint(benchmark_fingerprint)
    scoring_path = root / "caption_claims" / "scoring_protocol.json"
    atomic_write_json(scoring_path, {
        "scoring_fingerprint": scoring_fingerprint_value,
        "contract": scoring_contract(benchmark_fingerprint),
    })

    ledger_path = root / "caption_claims" / f"{args.route}.jsonl"
    usage_path = root / "caption_claims" / "api_usage.jsonl"
    ledger = JsonlLedger(ledger_path)
    latest = ledger.latest()
    _validate_existing(
        latest,
        route=args.route,
        generation_records=generation_records,
        manifest_by_id=manifest_by_id,
        benchmark_fingerprint=benchmark_fingerprint,
        extractor_fingerprint=extractor_fingerprint,
    )
    rescored = _rescore_existing(
        ledger,
        latest,
        root=root,
        manifest_by_id=manifest_by_id,
        extractor_fingerprint=extractor_fingerprint,
        scoring_fingerprint_value=scoring_fingerprint_value,
    )
    latest = ledger.latest()
    _validate_existing(
        latest,
        route=args.route,
        generation_records=generation_records,
        manifest_by_id=manifest_by_id,
        benchmark_fingerprint=benchmark_fingerprint,
        extractor_fingerprint=extractor_fingerprint,
        scoring_fingerprint_value=scoring_fingerprint_value,
    )
    completed = {
        object_id for object_id, record in latest.items()
        if record.get("failure_reason") is None and isinstance(record.get("extracted_claims"), dict)
    }
    selected_ids = manual_ids if args.scope == "manual20" else ordered_ids
    pending_ids = [object_id for object_id in selected_ids if object_id not in completed]
    manual_validation_path = _write_manual_validation_if_complete(root, manual_objects, latest)
    worklist_path = root / "caption_claims" / f"{args.route}_{args.scope}_pending.json"
    atomic_write_json(worklist_path, {
        "route": args.route,
        "scope": args.scope,
        "benchmark_protocol_fingerprint": benchmark_fingerprint,
        "extraction_fingerprint": extractor_fingerprint,
        "scoring_fingerprint": scoring_fingerprint_value,
        "completed_count": len(selected_ids) - len(pending_ids),
        "pending_count": len(pending_ids),
        "objects": [
            {
                "object_id": object_id,
                "caption_sha256": caption_sha256(generation_records[object_id]["raw_answer"]),
                "caption_characters": len(generation_records[object_id]["raw_answer"]),
                "table_path": manifest_by_id[object_id]["table_path"],
                "table_sha256": manifest_by_id[object_id]["table_sha256"],
            }
            for object_id in pending_ids
        ],
    })

    if not args.run_api:
        existing_summary = {
            "api_called": False,
            "route": args.route,
            "scope": args.scope,
            "manual_review_path": str(manual_path),
            "manual_review_objects": len(manual_ids),
            "manual_validation_path": (
                str(manual_validation_path) if manual_validation_path is not None else None
            ),
            "extraction_protocol_path": str(contract_path),
            "scoring_protocol_path": str(scoring_path),
            "rescored_existing_records": rescored,
            "completed_count": len(selected_ids) - len(pending_ids),
            "pending_count": len(pending_ids),
            "pending_worklist_path": str(worklist_path),
            **_summarize({object_id: latest[object_id] for object_id in selected_ids if object_id in latest}),
        }
        atomic_write_json(
            root / "caption_claims" / f"{args.route}_{args.scope}_summary.json", existing_summary,
        )
        print(json.dumps(existing_summary), flush=True)
        return

    required = {
        "--run-id": args.run_id,
        "--input-price-per-million-usd": args.input_price_per_million_usd,
        "--output-price-per-million-usd": args.output_price_per_million_usd,
        "--hard-usd-stop": args.hard_usd_stop,
    }
    absent = [name for name, value in required.items() if value is None]
    if absent:
        raise ValueError(f"API execution requires explicit values for {absent}")
    if args.hard_usd_stop <= 0:
        raise ValueError("--hard-usd-stop must be positive")
    if args.credentials_env_file:
        from dotenv import load_dotenv
        load_dotenv(args.credentials_env_file, override=False)
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise ValueError(f"API key environment variable {args.api_key_env!r} is not set")

    from google import genai
    from google.genai import types

    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            retry_options=types.HttpRetryOptions(attempts=1),
            timeout=120_000,
        ),
    )
    config = types.GenerateContentConfig(
        system_instruction=EXTRACTION_SYSTEM_PROMPT,
        response_mime_type="application/json",
        response_schema=CLAIMS_JSON_SCHEMA,
        max_output_tokens=args.max_output_tokens,
        thinking_config=types.ThinkingConfig(thinking_level="low"),
    )

    spent = 0.0
    if usage_path.exists():
        for line in usage_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                usage = json.loads(line)
                if usage.get("run_id") == args.run_id:
                    spent += float(usage.get("estimated_cost_usd") or 0.0)

    processed = 0
    for object_id in selected_ids:
        if object_id in completed:
            continue
        caption = generation_records[object_id]["raw_answer"]
        if len(caption) > 12_000:
            raise ValueError(f"object={object_id}: caption exceeds the 12,000-character input limit")
        table_path = root / manifest_by_id[object_id]["table_path"]
        table_text = table_path.read_text(encoding="utf-8")
        observed_table_sha = sha256_bytes(table_text.encode("utf-8"))
        if observed_table_sha != manifest_by_id[object_id]["table_sha256"]:
            raise ValueError(f"object={object_id}: table checksum differs from immutable manifest")

        # This upper bound intentionally assumes one token per input character and the full output
        # allowance.  It prevents starting a request that could cross the approved hard stop.
        input_upper_tokens = len(EXTRACTION_SYSTEM_PROMPT) + len(build_extraction_prompt(caption))
        reserve = _estimated_cost(
            input_upper_tokens,
            args.max_output_tokens,
            args.input_price_per_million_usd,
            args.output_price_per_million_usd,
        )
        if spent + reserve > args.hard_usd_stop:
            raise RuntimeError(
                f"hard USD stop would be exceeded before object={object_id}; "
                f"spent={spent:.6f}, reserve={reserve:.6f}, stop={args.hard_usd_stop:.6f}"
            )

        timestamp = datetime.now(timezone.utc).isoformat()
        response = None
        input_tokens = output_tokens = total_tokens = 0
        cost = 0.0
        cost_added = False
        try:
            response = client.models.generate_content(
                model=args.model,
                contents=build_extraction_prompt(caption),
                config=config,
            )
            usage_metadata = response.usage_metadata
            input_tokens = _usage_value(usage_metadata, "prompt_token_count")
            output_tokens = _usage_value(usage_metadata, "candidates_token_count") + _usage_value(
                usage_metadata, "thoughts_token_count",
            )
            total_tokens = _usage_value(usage_metadata, "total_token_count")
            cost = _estimated_cost(
                input_tokens,
                output_tokens,
                args.input_price_per_million_usd,
                args.output_price_per_million_usd,
            )
            spent += cost
            cost_added = True
            parsed = response.parsed if response.parsed is not None else json.loads(response.text)
            claims = claims_from_dict(parsed)
            validate_claim_quotes(caption, claims)
            facts = calculate_table_facts(table_text)
            scores = score_numeric_claims(claims, facts)
            usage_record = {
                "run_id": args.run_id,
                "item_id": object_id,
                "attempt": 1,
                "timestamp_utc": timestamp,
                "status": "success",
                "model": args.model,
                "input_tokens": input_tokens,
                "output_tokens_including_thinking": output_tokens,
                "total_tokens": total_tokens,
                "estimated_cost_usd": cost,
            }
            _append_jsonl(usage_path, usage_record)
            ledger.append({
                "object_id": object_id,
                "true_label": manifest_by_id[object_id]["true_label"],
                "parsed_label": generation_records[object_id]["parsed_label"],
                "route": args.route,
                "caption_sha256": caption_sha256(caption),
                "table_sha256": observed_table_sha,
                "benchmark_protocol_fingerprint": benchmark_fingerprint,
                "extraction_fingerprint": extractor_fingerprint,
                "extractor_version": EXTRACTOR_VERSION,
                "extractor_model": args.model,
                "scoring_version": SCORING_VERSION,
                "scoring_fingerprint": scoring_fingerprint_value,
                "extracted_claims": claims_to_dict(claims),
                "numeric_scores": scores,
                "failure_reason": None,
                "usage": usage_record,
            })
            processed += 1
        except Exception as exc:
            if response is not None and not cost_added:
                usage_metadata = response.usage_metadata
                input_tokens = _usage_value(usage_metadata, "prompt_token_count")
                output_tokens = _usage_value(usage_metadata, "candidates_token_count") + _usage_value(
                    usage_metadata, "thoughts_token_count",
                )
                total_tokens = _usage_value(usage_metadata, "total_token_count")
                cost = _estimated_cost(
                    input_tokens,
                    output_tokens,
                    args.input_price_per_million_usd,
                    args.output_price_per_million_usd,
                )
                spent += cost
            _append_jsonl(usage_path, {
                "run_id": args.run_id,
                "item_id": object_id,
                "attempt": 1,
                "timestamp_utc": timestamp,
                "status": "failed",
                "model": args.model,
                "input_tokens": input_tokens,
                "output_tokens_including_thinking": output_tokens,
                "total_tokens": total_tokens,
                "estimated_cost_usd": cost,
                "error_type": type(exc).__name__,
                "error_message": str(exc)[:500],
            })
            ledger.append({
                "object_id": object_id,
                "true_label": manifest_by_id[object_id]["true_label"],
                "parsed_label": generation_records[object_id]["parsed_label"],
                "route": args.route,
                "caption_sha256": caption_sha256(caption),
                "table_sha256": observed_table_sha,
                "benchmark_protocol_fingerprint": benchmark_fingerprint,
                "extraction_fingerprint": extractor_fingerprint,
                "extractor_version": EXTRACTOR_VERSION,
                "extractor_model": args.model,
                "extracted_claims": None,
                "numeric_scores": None,
                "failure_reason": f"{type(exc).__name__}: {exc}",
            })
            raise

    latest = ledger.latest()
    _validate_existing(
        latest,
        route=args.route,
        generation_records=generation_records,
        manifest_by_id=manifest_by_id,
        benchmark_fingerprint=benchmark_fingerprint,
        extractor_fingerprint=extractor_fingerprint,
        scoring_fingerprint_value=scoring_fingerprint_value,
    )
    manual_validation_path = _write_manual_validation_if_complete(root, manual_objects, latest)
    summary = {
        "route": args.route,
        "scope": args.scope,
        "extractor_model": args.model,
        "extraction_fingerprint": extractor_fingerprint,
        "scoring_fingerprint": scoring_fingerprint_value,
        "run_id": args.run_id,
        "processed_this_invocation": processed,
        "rescored_existing_records": rescored,
        "estimated_cost_usd_this_run_id": spent,
        "manual_validation_path": (
            str(manual_validation_path) if manual_validation_path is not None else None
        ),
        **_summarize({object_id: latest[object_id] for object_id in selected_ids if object_id in latest}),
    }
    atomic_write_json(root / "caption_claims" / f"{args.route}_{args.scope}_summary.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
