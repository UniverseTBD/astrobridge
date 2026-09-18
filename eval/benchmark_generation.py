"""Batched, resumable execution for one v8 benchmark generative route per process."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import pandas as pd

from eval.datasets.lightcurve_yse import build_raw_inputs_lightcurve
from eval.benchmark_artifacts import load_yse_manifest, manifest_records_by_id
from eval.benchmark_protocol import (
    CHART_ROUTES,
    GENERATIVE_ROUTES,
    JsonlLedger,
    MAX_NEW_TOKENS,
    BASE_MODEL_REVISIONS,
    MODEL_IDS,
    MODEL_REVISIONS,
    SYSTEM_PROMPT,
    TABLE_ROUTES,
    V8_REPO_ID,
    V8_USER_PROMPT,
    YSE_REVISION,
    benchmark_protocol_fingerprint,
    full_prompt_text,
    parse_final_sentence,
    user_prompt_for_route,
    validate_local_snapshot,
)


@dataclass
class BatchResult:
    answers: list[str]
    elapsed_seconds: float


class V8BatchGenerator:
    def __init__(self, cfg, base_model_path: str, device: str) -> None:
        from captioner.inference import load_inference_model_from_hub

        cfg.llm.name = base_model_path
        self.cfg = cfg
        self.device = device
        self.model, self.tokenizer, self.encoders = load_inference_model_from_hub(
            cfg,
            V8_REPO_ID,
            device=device,
            modality_names=["lightcurve"],
            revision=MODEL_REVISIONS["astrobridge_v8_prefix"],
        )
        self.out_dims = {name: int(modality.out_dim) for name, modality in cfg.modalities.items()}
        self.max_tokens = {name: int(modality.max_tokens) for name, modality in cfg.modalities.items()}

    def generate(self, rows: list[pd.Series], _prompts: list[str], _images: list | None = None) -> BatchResult:
        from captioner.inference import generate_captions_batched

        started = time.perf_counter()
        raw_inputs = [build_raw_inputs_lightcurve(row, self.cfg) for row in rows]
        answers = generate_captions_batched(
            self.model,
            self.tokenizer,
            self.encoders,
            self.out_dims,
            self.max_tokens,
            self.cfg.prompt,
            self.device,
            raw_inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            question=V8_USER_PROMPT,
            system=SYSTEM_PROMPT,
        )
        return BatchResult(answers=answers, elapsed_seconds=time.perf_counter() - started)


class NativeBatchGenerator:
    def __init__(self, model_path: str, device: str) -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.device = device
        self.processor = AutoProcessor.from_pretrained(
            model_path, local_files_only=True, trust_remote_code=True,
        )
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            local_files_only=True,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map={"": device},
        )
        self.model.eval()

    def generate(self, _rows: list[pd.Series], prompts: list[str], images: list | None = None) -> BatchResult:
        import torch

        started = time.perf_counter()
        if images is not None and len(images) != len(prompts):
            raise ValueError("images/prompts batch lengths differ")
        messages = native_messages(prompts, images)

        previous_side = self.processor.tokenizer.padding_side
        self.processor.tokenizer.padding_side = "left"
        try:
            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                padding=True,
                enable_thinking=False,
            ).to(self.device)
        finally:
            self.processor.tokenizer.padding_side = previous_side

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            generated = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
            )
        prompt_length = inputs["input_ids"].shape[1]
        answers = self.processor.batch_decode(
            generated[:, prompt_length:], skip_special_tokens=True,
        )
        return BatchResult(answers=answers, elapsed_seconds=time.perf_counter() - started)


def native_messages(prompts: list[str], images: list | None = None) -> list[list[dict]]:
    """Build the structured multimodal message form validated by the interface smoke test."""
    if images is not None and len(images) != len(prompts):
        raise ValueError("images/prompts batch lengths differ")
    messages = []
    for index, prompt in enumerate(prompts):
        content = [{"type": "text", "text": prompt}]
        if images is not None:
            content.insert(0, {"type": "image", "image": images[index]})
        messages.append([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ])
    return messages


def input_checksum_for_route(route: str, manifest_record: dict) -> str:
    if route in TABLE_ROUTES:
        return manifest_record["table_sha256"]
    if route in CHART_ROUTES:
        return manifest_record["chart_sha256"]
    return manifest_record["v8_input_sha256"]


def user_prompt_for_manifest_record(route: str, manifest_record: dict, artifact_root: Path) -> str:
    if route in TABLE_ROUTES:
        table = (artifact_root / manifest_record["table_path"]).read_text(encoding="utf-8").rstrip("\n")
        return user_prompt_for_route(route, table=table)
    return user_prompt_for_route(route)


def validate_generation_records(
    records: dict[str, dict],
    route: str,
    manifest_by_id: dict[str, dict],
    artifact_root: Path,
    protocol_fingerprint: str,
) -> None:
    for object_id, record in records.items():
        if object_id not in manifest_by_id:
            raise ValueError(f"existing route record object={object_id!r} is absent from the YSE manifest")
        manifest_record = manifest_by_id[object_id]
        expected_prompt = full_prompt_text(user_prompt_for_manifest_record(route, manifest_record, artifact_root))
        checks = {
            "route": route,
            "model_id": MODEL_IDS[route],
            "model_revision": MODEL_REVISIONS[route],
            "base_model_revision": BASE_MODEL_REVISIONS[route],
            "dataset_revision": YSE_REVISION,
            "max_new_tokens": MAX_NEW_TOKENS,
            "protocol_fingerprint": protocol_fingerprint,
            "true_label": manifest_record["true_label"],
            "full_prompt": expected_prompt,
            "input_checksum": input_checksum_for_route(route, manifest_record),
        }
        for key, expected in checks.items():
            if record.get(key) != expected:
                raise ValueError(
                    f"existing record object={object_id!r} has {key}={record.get(key)!r}; "
                    f"expected {expected!r}. Refusing to mix protocols."
                )
        if isinstance(record.get("raw_answer"), str):
            expected_sentence, expected_label = parse_final_sentence(record["raw_answer"])
            if record.get("final_sentence") != expected_sentence or record.get("parsed_label") != expected_label:
                raise ValueError(
                    f"existing record object={object_id!r} has parser outputs inconsistent with its raw answer"
                )


def generation_record(
    route: str,
    manifest_record: dict,
    user_prompt: str,
    raw_answer: str | None,
    final_sentence: str | None,
    parsed_label: str | None,
    runtime_seconds: float,
    batch_runtime_seconds: float,
    batch_size: int,
    protocol_fingerprint: str,
    *,
    failure_reason: str | None,
) -> dict:
    return {
        "object_id": str(manifest_record["object_id"]),
        "true_label": str(manifest_record["true_label"]),
        "model_id": MODEL_IDS[route],
        "model_revision": MODEL_REVISIONS[route],
        "base_model_revision": BASE_MODEL_REVISIONS[route],
        "dataset_revision": YSE_REVISION,
        "route": route,
        "protocol_fingerprint": protocol_fingerprint,
        "full_prompt": full_prompt_text(user_prompt),
        "input_checksum": input_checksum_for_route(route, manifest_record),
        "max_new_tokens": MAX_NEW_TOKENS,
        "raw_answer": raw_answer,
        "final_sentence": final_sentence,
        "parsed_label": parsed_label,
        "runtime_seconds": runtime_seconds,
        "batch_runtime_seconds": batch_runtime_seconds,
        "batch_size": batch_size,
        "failure_reason": failure_reason,
    }


def run_generative_route(
    *,
    route: str,
    artifact_root: Path,
    output_jsonl: Path,
    cfg,
    model_path: str,
    batch_size: int,
    device: str = "cuda",
    limit: int | None = None,
) -> dict:
    """Run exactly one route; successful JSONL records are never generated again."""
    if route not in GENERATIVE_ROUTES:
        raise ValueError(f"unknown route={route!r}")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    validate_local_snapshot(route, model_path)
    if route.startswith("gemma4_31b_") and batch_size != 1:
        raise ValueError("Gemma 4 31B is validated only at batch size 1; batch size 2 exhausted A100 memory")
    if batch_size == 1 and not route.startswith("gemma4_31b_"):
        raise ValueError("routes other than Gemma 4 31B require batched generation with batch_size > 1")
    protocol_fingerprint = benchmark_protocol_fingerprint(cfg)

    manifest = load_yse_manifest(artifact_root, expected_protocol_fingerprint=protocol_fingerprint)
    manifest_by_id = manifest_records_by_id(manifest)
    ledger = JsonlLedger(output_jsonl)
    latest = ledger.latest()
    validate_generation_records(latest, route, manifest_by_id, artifact_root, protocol_fingerprint)
    completed = ledger.completed_ids()
    selected = manifest["objects"][:limit] if limit is not None else manifest["objects"]
    pending = [record for record in selected if str(record["object_id"]) not in completed]
    if not pending:
        return {"route": route, "completed": len(completed), "generated": 0}

    rows_path = artifact_root / manifest["rows_path"]
    rows = pd.read_parquet(rows_path).set_index("object_id", drop=False)
    if route == "astrobridge_v8_prefix":
        generator = V8BatchGenerator(cfg, model_path, device)
    else:
        generator = NativeBatchGenerator(model_path, device)

    generated_count = 0
    for start in range(0, len(pending), batch_size):
        batch_records = pending[start:start + batch_size]
        batch_rows = [rows.loc[str(record["object_id"])] for record in batch_records]
        prompts: list[str] = []
        images = None
        if route in TABLE_ROUTES:
            for record in batch_records:
                prompts.append(user_prompt_for_manifest_record(route, record, artifact_root))
        elif route in CHART_ROUTES:
            from PIL import Image

            prompts = [user_prompt_for_manifest_record(route, record, artifact_root) for record in batch_records]
            images = [Image.open(artifact_root / record["chart_path"]).convert("RGB") for record in batch_records]
        else:
            prompts = [user_prompt_for_manifest_record(route, record, artifact_root) for record in batch_records]

        batch_started = time.perf_counter()
        try:
            result = generator.generate(batch_rows, prompts, images)
            if len(result.answers) != len(batch_records):
                raise RuntimeError(
                    f"model returned {len(result.answers)} answers for batch of {len(batch_records)}"
                )
        except Exception as exc:
            failed_runtime = time.perf_counter() - batch_started
            for record, prompt in zip(batch_records, prompts):
                ledger.append(generation_record(
                    route, record, prompt, None, None, None,
                    failed_runtime / len(batch_records), failed_runtime, len(batch_records),
                    protocol_fingerprint,
                    failure_reason=f"{type(exc).__name__}: {exc}",
                ))
            raise
        finally:
            if images is not None:
                for image in images:
                    image.close()

        per_object = result.elapsed_seconds / len(batch_records)
        for record, prompt, answer in zip(batch_records, prompts, result.answers):
            final_sentence, parsed = parse_final_sentence(answer)
            ledger.append(generation_record(
                route,
                manifest_by_id[str(record["object_id"])],
                prompt,
                answer,
                final_sentence,
                parsed,
                per_object,
                result.elapsed_seconds,
                len(batch_records),
                protocol_fingerprint,
                failure_reason=None,
            ))
            generated_count += 1

    return {"route": route, "completed": len(ledger.completed_ids()), "generated": generated_count}
