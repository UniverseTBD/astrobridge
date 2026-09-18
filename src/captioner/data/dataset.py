"""Dataset: samples the TARGET SUBSET first, then intersects with availability (§6) — sampling
per-modality independently would under-sample exactly the subsets that matter (joint).

Iterates over `self.modality_names` (from configs/modalities.yaml) everywhere; adding a third
modality requires no change here.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset

from captioner.data.cache import assert_cache_matches_config, cache_dir_for, load_cache_index
from captioner.encoders.registry import encoder_hash, encoder_spec
from captioner.utils.logging import get_logger
from captioner.utils.prompt import build_wrapper_text, pick_prompt_variants

logger = get_logger(__name__)


class ModalityCacheReader:
    """Lazily mmaps npy shards for one modality and returns (T, out_dim) float32 arrays by object_id."""

    def __init__(self, cache_root: Path, modality: str, spec_hash: str) -> None:
        self.dir = cache_dir_for(cache_root, modality, spec_hash)
        self.index = load_cache_index(cache_root, modality, spec_hash).set_index("object_id")
        self._shards: dict[str, np.ndarray] = {}

    def get(self, object_id: str) -> np.ndarray:
        row = self.index.loc[object_id]
        shard_file = row["shard_file"]
        if shard_file not in self._shards:
            self._shards[shard_file] = np.load(self.dir / shard_file, mmap_mode="r")
        arr = self._shards[shard_file][int(row["row_in_shard"])]
        return np.asarray(arr, dtype=np.float32)


class CaptionerDataset(Dataset):
    def __init__(
        self,
        manifest: pd.DataFrame,
        captions: pd.DataFrame,
        modalities_cfg: DictConfig,
        cache_root: Path,
        split: str,
        tokenizer,
        prompt_cfg: DictConfig,
        # Was 128 with no documented reasoning anywhere (unchanged since the initial commit, never
        # overridden at any of its 4 call sites) — and silently truncated captions in training.
        # Measured directly against every tier's real caption text (Qwen3.5-9B tokenizer,
        # add_special_tokens=False): lightcurve max 297 tok, image max 213 tok, spectra max 174
        # tok. Lightcurve was hit hardest — 99.6% of its 987 captions got truncated, and since
        # every one of them puts its classification sentence in the FINAL sentence (confirmed:
        # position 0.98 of the caption, 100% of the time), truncation was cutting exactly the part
        # of the caption that says which SN class and why, on nearly every training example.
        # True max + suffix is 298 (297, lightcurve, + 1 for <|im_end|>) — the minimum value that
        # truncates NOTHING in any tier as of this measurement; set to 300 for a couple tokens of
        # headroom. Caption sources get rewritten in place across revisions (see configs/data.yaml),
        # so re-measure this against the live max whenever captions are regenerated, rather than
        # assuming it holds.
        max_caption_tokens: int = 300,
    ) -> None:
        self.manifest = manifest[manifest["split"] == split].reset_index(drop=True)
        self.captions = captions
        self.modality_names = list(modalities_cfg.modalities.keys())
        self.modality_cfg = modalities_cfg.modalities
        self.out_dims = {n: int(modalities_cfg.modalities[n].out_dim) for n in self.modality_names}
        self.max_tokens = {n: int(modalities_cfg.modalities[n].max_tokens) for n in self.modality_names}

        self.readers: dict[str, ModalityCacheReader] = {}
        for name in self.modality_names:
            spec = encoder_spec(self.modality_cfg[name])
            spec_hash = encoder_hash(spec)
            index_df = load_cache_index(cache_root, name, spec_hash)
            assert_cache_matches_config(index_df, self.modality_cfg[name])
            self.readers[name] = ModalityCacheReader(cache_root, name, spec_hash)

        self.subset_weights: list[tuple[frozenset[str], float]] = [
            (frozenset(entry["subset"]), float(entry["weight"]))
            for entry in modalities_cfg.dropout.subset_weights
        ]

        self._epoch = 0
        self.tokenizer = tokenizer
        self.prompt_cfg = prompt_cfg
        self.max_caption_tokens = max_caption_tokens
        # Tokenized once — appended to every caption target so the model learns to stop.
        self._caption_suffix_ids = tokenizer(
            str(prompt_cfg.caption_suffix), add_special_tokens=False, return_tensors="pt",
        )["input_ids"][0]

        self._captions_by_key = {
            (row["object_id"], frozenset(row["subset"])): row["text"]
            for _, row in captions.iterrows()
        }
        # Per-object set of subsets that actually have a caption — NOT the same as "data is
        # present for this subset". Confirmed real gap: spectra-tier captions are restricted to
        # only the ~2,021 Gemini-covered objects (deliberate, no rule-based fallback — see
        # 01_generate_captions.py), so most has_spectra=True objects have spectrum embeddings
        # but no caption for subset={"spectra"}. Sampling `available` alone (has_<modality>
        # flags) would pick that subset anyway, land on an empty caption_text, and produce a
        # sample whose labels are entirely IGNORE_INDEX — a real, confirmed cause of NaN loss
        # when enough such samples land in the same micro-batch (see train/loop.py's degenerate-
        # batch guard for the other half of this fix).
        self._caption_subsets_by_object: dict[str, set[frozenset]] = {}
        for object_id, subset in self._captions_by_key:
            self._caption_subsets_by_object.setdefault(object_id, set()).add(subset)

        # Drop objects that can never produce a real training example: no subset of their
        # available modalities has a caption at all, so `_sample_target_subset` would fall back
        # to frozenset() (empty prompt, empty caption) on every single visit, every epoch —
        # dead weight that only dilutes the loss. Object-level, not modality-specific, so this
        # needs no change when a third modality is added.
        def _has_any_usable_subset(row: pd.Series) -> bool:
            available = self._availability(row)
            captioned = self._caption_subsets_by_object.get(row["object_id"], set())
            return any(s.issubset(available) and s in captioned for s, _ in self.subset_weights)

        usable_mask = self.manifest.apply(_has_any_usable_subset, axis=1)
        n_dropped = int((~usable_mask).sum())
        if n_dropped:
            logger.warning(
                f"[{split}] Dropping {n_dropped}/{len(self.manifest)} objects with no captioned "
                "subset at all (data present but no caption exists for any subset of it) — they "
                "would only ever produce empty-caption training examples."
            )
        self.manifest = self.manifest[usable_mask].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.manifest)

    def set_epoch(self, epoch: int) -> None:
        """Re-roll the per-object subset and (system, instruction) draw for a new epoch.

        train/loop.py calls this at the top of every epoch. Without it the draw is fixed for the
        whole run: the pre-fix code seeded on `hash((object_id, idx))` with no epoch term, so
        each object saw exactly ONE prompt variant and ONE modality subset across all 30 epochs,
        despite the docstring below claiming otherwise. configs/model.yaml ships 4 system x 10
        instruction variants specifically so the model does not become brittle to one wording;
        none of that cross-product was ever reached.
        """
        self._epoch = int(epoch)

    def _seed_for(self, object_id: str) -> int:
        """Stable across processes and runs, unlike `hash()`.

        Python randomises `hash()` of a str per process via PYTHONHASHSEED unless it is pinned,
        so the pre-fix seeding gave different subsets for the same object on different ranks —
        real on the 2-node v6 run — and was not reproducible between runs at all.
        """
        key = f"{object_id}:{self._epoch}".encode()
        return int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big")

    def _availability(self, row: pd.Series) -> frozenset[str]:
        present = set()
        for name in self.modality_names:
            flag_col = f"has_{name}"
            if flag_col in row and bool(row[flag_col]):
                present.add(name)
        return frozenset(present)

    def _sample_target_subset(
        self, available: frozenset[str], rng: np.random.Generator, captioned: set[frozenset]
    ) -> frozenset[str]:
        """Sample the target subset first from configured weights, restricted to subsets that
        are both actually available AND actually have a caption for this object, then
        renormalize — never sample per-modality independently, and never pick a subset with no
        real caption to train against.
        """
        candidates = [(s, w) for s, w in self.subset_weights if s.issubset(available) and s in captioned]
        if not candidates:
            return frozenset()
        weights = np.array([w for _, w in candidates], dtype=np.float64)
        weights = weights / weights.sum()
        idx = rng.choice(len(candidates), p=weights)
        return candidates[idx][0]

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.manifest.iloc[idx]
        object_id = row["object_id"]
        available = self._availability(row)

        rng = np.random.default_rng(self._seed_for(object_id))
        captioned = self._caption_subsets_by_object.get(object_id, set())
        shown = self._sample_target_subset(available, rng, captioned)

        modality_arrays: dict[str, np.ndarray | None] = {}
        for name in self.modality_names:
            if name in shown:
                modality_arrays[name] = self.readers[name].get(object_id)
            else:
                modality_arrays[name] = None

        caption_text = self._captions_by_key.get((object_id, shown), "")

        # Same seeded rng that picked `shown` — reproducible per (object_id, epoch), varying
        # across objects and genuinely re-rolling each epoch via set_epoch(), which is what gives
        # the model the whole system x instruction cross-product over a run.
        system, instruction = pick_prompt_variants(self.prompt_cfg, shown, rng)
        pre_text, post_text = build_wrapper_text(self.prompt_cfg, system, instruction)

        pre_ids = self.tokenizer(pre_text, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
        post_ids = self.tokenizer(post_text, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
        # Truncate the caption itself so caption + suffix stays within the budget — never let the
        # <|im_end|> stop token be the thing that gets cut.
        if caption_text:
            cap_ids = self.tokenizer(
                caption_text,
                add_special_tokens=False,
                truncation=True,
                max_length=max(1, self.max_caption_tokens - len(self._caption_suffix_ids)),
                return_tensors="pt",
            )["input_ids"][0]
            caption_ids = torch.cat([cap_ids, self._caption_suffix_ids])
        else:
            # Defensive: `_sample_target_subset` only picks captioned subsets, but keeps an empty
            # fallback. Leave caption_ids empty (not just <|im_end|>) so the degenerate-batch
            # guard in train/loop.py still recognises a caption-less example.
            caption_ids = torch.empty(0, dtype=torch.long)

        return {
            "object_id": object_id,
            "shown": shown,
            "modality_arrays": modality_arrays,
            "pre_ids": pre_ids,
            "post_ids": post_ids,
            "caption_ids": caption_ids,
        }
