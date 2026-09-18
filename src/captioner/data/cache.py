"""Embedding cache. Encoders are loaded exactly once, here — training never imports them (§5).

Cache path includes the encoder hash (impl, hf_path, revision, kwargs), not just the modality
name, so swapping an encoder can never silently reuse stale embeddings (§3).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig

from captioner.encoders.base import ModalityEncoder
from captioner.encoders.registry import encoder_hash, encoder_spec
from captioner.utils.logging import get_logger

logger = get_logger(__name__)


def cache_dir_for(base_out_dir: Path, modality: str, spec_hash: str) -> Path:
    return base_out_dir / modality / spec_hash


def cache_modality(
    name: str,
    encoder: ModalityEncoder,
    modality_cfg: DictConfig,
    manifest: pd.DataFrame,
    batch_loader,  # callable: list[object_id] -> encoder-specific batch dict, e.g. {"pixel_values": Tensor}
    out_dir: Path,
    shard: int = 256,
) -> tuple[Path, list[str]]:
    """Writes float16 shards keyed by object_id + an index parquet including the full encoder
    spec, so training can assert the cache matches the config it was built against.

    Returns `(target_dir, excluded_object_ids)`. An object's embedding can come back all-NaN even
    though the encoder call itself succeeds — the confirmed real cause is `_flux_to_array` in
    02_cache_embeddings.py NaN-filling image rows that were `None` (edge-clipped cutouts) rather
    than raising, on the (now known to be wrong) assumption that AION's encoder would tolerate
    NaN input. It doesn't: NaN propagates through the encoder into the cached embedding, then
    through the Q-Former/adapter/LLM into the training loss, presenting downstream as a NaN
    train or val loss with no visible cause at the point it's noticed. Per this codebase's own
    rule that absent modalities must be truly excluded rather than fed in as placeholder content
    (see model/captioner.py's FusionStack docstring), a corrupted embedding is excluded from the
    index here — never written as if valid — and returned so the caller can flip that object's
    `has_<name>` manifest flag to False; otherwise `_availability` in data/dataset.py would still
    believe the modality is present (it only reads the manifest flag) and later raise a KeyError
    trying to look up an object missing from this index.
    """
    spec = encoder_spec(modality_cfg)
    spec_hash = encoder_hash(spec)
    target_dir = cache_dir_for(out_dir, name, spec_hash)
    target_dir.mkdir(parents=True, exist_ok=True)

    flag_col = f"has_{name}"
    if flag_col not in manifest.columns:
        flag_col = None
    object_ids = manifest["object_id"].tolist() if flag_col is None else manifest.loc[manifest[flag_col], "object_id"].tolist()

    index_rows = []
    excluded_ids: list[str] = []
    for shard_start in range(0, len(object_ids), shard):
        shard_ids = object_ids[shard_start : shard_start + shard]
        batch = batch_loader(shard_ids)
        with torch.no_grad():
            out = encoder.encode(batch)  # (B, T, out_dim), token-level — never pooled
        arr = out.to(torch.float16).cpu().numpy()

        shard_idx = shard_start // shard
        shard_path = target_dir / f"shard_{shard_idx:05d}.npy"
        np.save(shard_path, arr)

        bad = ~np.isfinite(arr).all(axis=tuple(range(1, arr.ndim)))  # (B,) True = NaN/Inf anywhere in that row
        for i, obj_id in enumerate(shard_ids):
            if bad[i]:
                excluded_ids.append(obj_id)
                continue
            index_rows.append(
                {
                    "object_id": obj_id,
                    "shard_file": shard_path.name,
                    "row_in_shard": i,
                    "n_tokens": int(arr.shape[1]),
                    "out_dim": int(arr.shape[2]),
                    **{f"spec_{k}": str(v) for k, v in spec.items()},
                }
            )

    if excluded_ids:
        logger.warning(
            f"[{name}] {len(excluded_ids)}/{len(object_ids)} objects produced a non-finite "
            f"(NaN/Inf) embedding and were excluded from the cache index: {excluded_ids[:10]}"
            f"{'...' if len(excluded_ids) > 10 else ''}. The caller must flip has_{name}=False "
            "for these in the manifest so they're never sampled as this modality again."
        )

    index_df = pd.DataFrame(index_rows)
    index_path = target_dir / "index.parquet"
    index_df.to_parquet(index_path, index=False)
    logger.info(f"Cached {len(index_rows)} objects for modality={name} at {target_dir}")
    return target_dir, excluded_ids


def verify_fp16_roundtrip(encoder: ModalityEncoder, batch, n: int = 50) -> float:
    """float32 vs float16 max abs error on `n` objects — recorded in the run log (§5)."""
    with torch.no_grad():
        out_fp32 = encoder.encode(batch).to(torch.float32)
        out_fp16_roundtrip = out_fp32.to(torch.float16).to(torch.float32)
    max_abs_err = (out_fp32 - out_fp16_roundtrip).abs().max().item()
    return max_abs_err


def load_cache_index(out_dir: Path, modality: str, spec_hash: str) -> pd.DataFrame:
    index_path = cache_dir_for(out_dir, modality, spec_hash) / "index.parquet"
    if not index_path.exists():
        raise FileNotFoundError(
            f"No cache index at {index_path}. Run scripts/02_cache_embeddings.py for modality="
            f"{modality!r} with the current configs/modalities.yaml first."
        )
    return pd.read_parquet(index_path)


def assert_cache_matches_config(index_df: pd.DataFrame, modality_cfg: DictConfig) -> None:
    spec = encoder_spec(modality_cfg)
    if len(index_df) == 0:
        return
    row = index_df.iloc[0]
    for k, v in spec.items():
        cached_val = row.get(f"spec_{k}")
        if cached_val is not None and cached_val != str(v):
            raise ValueError(
                f"Cache index spec_{k}={cached_val!r} does not match current config value {v!r}. "
                "The embedding cache is stale for this encoder config — re-run 02_cache_embeddings.py."
            )
