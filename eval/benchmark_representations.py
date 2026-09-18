"""Extract the frozen pre-adapter v8 Q-former representation without loading Qwen or LoRA."""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import torch

from captioner.encoders.registry import build_encoder
from captioner.model.captioner import FusionStack
from eval.datasets.lightcurve_yse import build_raw_inputs_lightcurve
from eval.benchmark_protocol import (
    ATCAT_REVISION,
    DEVELOPMENT_DATASET_ID,
    DEVELOPMENT_REVISION,
    V8_REPO_ID,
    V8_REVISION,
    YSE_DATASET_ID,
    YSE_REVISION,
    assert_dev_yse_disjoint,
    benchmark_protocol_fingerprint,
    deterministic_cv_folds,
    lightcurve_preprocessing_contract,
)


def load_frozen_v8_stack(cfg, *, cache_dir: Path | None, device: str) -> tuple[FusionStack, object]:
    """Load only ``middle.pt`` and ATCAT; the decoder and published LoRA are not touched."""
    from huggingface_hub import hf_hub_download

    cfg.modalities.lightcurve.encoder.revision = ATCAT_REVISION
    middle_path = hf_hub_download(
        repo_id=V8_REPO_ID,
        filename="middle.pt",
        revision=V8_REVISION,
        cache_dir=str(cache_dir) if cache_dir else None,
    )
    state = torch.load(middle_path, map_location="cpu", weights_only=False)
    d_llm = int(state["adapter.out_norm.weight"].numel())
    out_dims = {name: int(modality.out_dim) for name, modality in cfg.modalities.items()}
    stack = FusionStack(
        modality_out_dims=out_dims,
        d_shared=int(cfg.d_shared),
        d_llm=d_llm,
        qformer_cfg=dict(cfg.qformer),
        projector_hidden_mult=int(cfg.projector.hidden_mult),
        projector_dropout=float(cfg.projector.dropout),
        adapter_target_norm=1.0,
    )
    stack.load_state_dict(state)
    stack.to(device).eval()
    for parameter in stack.parameters():
        parameter.requires_grad_(False)
    atcat = build_encoder("lightcurve", cfg.modalities.lightcurve, device=device)
    return stack, atcat


@torch.no_grad()
def extract_table_representations(table: pd.DataFrame, cfg, stack, atcat, *, batch_size: int, device: str) -> torch.Tensor:
    outputs: list[torch.Tensor] = []
    out_dims = {name: int(modality.out_dim) for name, modality in cfg.modalities.items()}
    max_tokens = {name: int(modality.max_tokens) for name, modality in cfg.modalities.items()}
    for start in range(0, len(table), batch_size):
        rows = [row for _, row in table.iloc[start:start + batch_size].iterrows()]
        raw = [build_raw_inputs_lightcurve(row, cfg)["lightcurve"] for row in rows]
        atcat_batch = {name: torch.cat([entry[name] for entry in raw], dim=0) for name in raw[0]}
        lightcurve_tokens = atcat.encode(atcat_batch).to(device=device, dtype=torch.float32)

        token_chunks: list[torch.Tensor] = []
        mask_chunks: list[torch.Tensor] = []
        batch_n = len(rows)
        device_type = "cuda" if str(device).startswith("cuda") else "cpu"
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            for name in stack.modality_names:
                token_count = max_tokens[name]
                tokens = torch.zeros((batch_n, token_count, out_dims[name]), dtype=torch.float32, device=device)
                mask = torch.ones((batch_n, token_count), dtype=torch.bool, device=device)
                if name == "lightcurve":
                    n = min(token_count, lightcurve_tokens.shape[1])
                    tokens[:, :n] = lightcurve_tokens[:, :n]
                    mask[:, :n] = False
                projected = stack.projectors[name](tokens)
                token_chunks.append(stack.modality_identity(projected, name))
                mask_chunks.append(mask)
            queries = stack.qformer(
                torch.cat(token_chunks, dim=1),
                key_padding_mask=torch.cat(mask_chunks, dim=1),
            )
        if tuple(queries.shape[1:]) != (64, 384):
            raise ValueError(f"unexpected pre-adapter representation shape {tuple(queries.shape)}")
        if not torch.isfinite(queries).all():
            raise ValueError("non-finite values in frozen Q-former representations")
        outputs.append(queries.float().cpu())
    return torch.cat(outputs, dim=0)


def _save_payload(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _validate_cache(
    path: Path,
    *,
    dataset_id: str,
    revision: str,
    object_ids: list[str],
    labels: list[str],
    preprocessing: dict,
    protocol_fingerprint: str,
) -> None:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    expected = {
        "dataset_id": dataset_id,
        "dataset_revision": revision,
        "v8_revision": V8_REVISION,
        "atcat_revision": ATCAT_REVISION,
        "representation_shape": [64, 384],
        "object_ids": object_ids,
        "labels": labels,
        "preprocessing": preprocessing,
        "protocol_fingerprint": protocol_fingerprint,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"representation cache {path} has incompatible {key}")
    if tuple(payload["representations"].shape) != (len(object_ids), 64, 384):
        raise ValueError(f"representation cache {path} has an incompatible tensor shape")


def extract_and_save_representations(
    development: pd.DataFrame,
    yse: pd.DataFrame,
    cfg,
    *,
    output_dir: Path,
    cache_dir: Path | None,
    batch_size: int,
    device: str,
) -> dict:
    """Create separate immutable development and YSE caches, or reuse both if already complete."""
    development_path = output_dir / "development_qformer.pt"
    yse_path = output_dir / "yse_qformer.pt"
    development_ids = development["object_id"].astype(str).tolist()
    development_labels = development["class_label"].astype(str).tolist()
    yse_ids = yse["object_id"].astype(str).tolist()
    yse_labels = yse["class_label"].astype(str).tolist()
    assert_dev_yse_disjoint(development_ids, yse_ids)
    expected_development_counts = {"SN Ia": 563, "SN II": 314, "SN Ibc": 110}
    development_counts = development["class_label"].value_counts().to_dict()
    if development_counts != expected_development_counts:
        raise ValueError(
            f"unexpected development class counts: {development_counts}; "
            f"expected {expected_development_counts}"
        )
    folds = deterministic_cv_folds(development_ids, development_labels)
    protocol_fingerprint = benchmark_protocol_fingerprint(cfg)

    common = {
        "v8_repo_id": V8_REPO_ID,
        "v8_revision": V8_REVISION,
        "atcat_revision": ATCAT_REVISION,
        "atcat_provenance_note": (
            "Execution-compatible with v8; not provably the original training revision from the published files."
        ),
        "representation_shape": [64, 384],
        "preprocessing": lightcurve_preprocessing_contract(cfg),
        "protocol_fingerprint": protocol_fingerprint,
    }
    if development_path.exists():
        _validate_cache(
            development_path,
            dataset_id=DEVELOPMENT_DATASET_ID,
            revision=DEVELOPMENT_REVISION,
            object_ids=development_ids,
            labels=development_labels,
            preprocessing=common["preprocessing"],
            protocol_fingerprint=protocol_fingerprint,
        )
    if yse_path.exists():
        _validate_cache(
            yse_path,
            dataset_id=YSE_DATASET_ID,
            revision=YSE_REVISION,
            object_ids=yse_ids,
            labels=yse_labels,
            preprocessing=common["preprocessing"],
            protocol_fingerprint=protocol_fingerprint,
        )
    missing_development = not development_path.exists()
    missing_yse = not yse_path.exists()
    if not missing_development and not missing_yse:
        return {"development": str(development_path), "yse": str(yse_path), "status": "reused"}

    stack, atcat = load_frozen_v8_stack(cfg, cache_dir=cache_dir, device=device)
    if missing_development:
        development_tensor = extract_table_representations(
            development, cfg, stack, atcat, batch_size=batch_size, device=device,
        )
        _save_payload(development_path, {
            **common,
            "dataset_id": DEVELOPMENT_DATASET_ID,
            "dataset_revision": DEVELOPMENT_REVISION,
            "object_ids": development_ids,
            "labels": development_labels,
            "cv_folds": [folds[object_id] for object_id in development_ids],
            "records": [
                {"object_id": oid, "class_label": label, "split": f"cv_fold_{folds[oid]}"}
                for oid, label in zip(development_ids, development_labels)
            ],
            "representations": development_tensor,
        })
    if missing_yse:
        yse_tensor = extract_table_representations(
            yse, cfg, stack, atcat, batch_size=batch_size, device=device,
        )
        _save_payload(yse_path, {
            **common,
            "dataset_id": YSE_DATASET_ID,
            "dataset_revision": YSE_REVISION,
            "object_ids": yse_ids,
            "labels": yse_labels,
            "records": [
                {"object_id": oid, "class_label": label, "split": "held_out_yse_test"}
                for oid, label in zip(yse_ids, yse_labels)
            ],
            "representations": yse_tensor,
        })
    return {"development": str(development_path), "yse": str(yse_path), "status": "created"}
