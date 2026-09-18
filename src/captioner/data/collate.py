"""Collate: builds the per-modality padded tensors + key_padding_mask the model consumes.

Absent-modality and within-modality padding use the same mechanism (True = ignore in the
mask), which is what `test_garbage_in_absent_slot_is_bitwise_noop` exercises — see
tests/test_masking.py and §6/§10 point 3 for why this must never be an unmasked zero vector.
"""
from __future__ import annotations

from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence


def collate_batch(
    examples: list[dict[str, Any]],
    modality_names: list[str],
    out_dims: dict[str, int],
    max_tokens: dict[str, int],
    pad_token_id: int,
) -> dict[str, Any]:
    B = len(examples)
    modality_batch: dict[str, dict[str, torch.Tensor]] = {}

    for name in modality_names:
        T_m = max_tokens[name]
        D = out_dims[name]
        tokens = torch.zeros((B, T_m, D), dtype=torch.float32)
        mask = torch.ones((B, T_m), dtype=torch.bool)  # True = pad/absent by default

        for i, ex in enumerate(examples):
            arr = ex["modality_arrays"].get(name)
            if arr is None:
                continue  # modality not shown for this example — stays fully masked, zero rows
            t = torch.from_numpy(arr).to(torch.float32)
            n = min(t.shape[0], T_m)
            tokens[i, :n] = t[:n]
            mask[i, :n] = False

        modality_batch[name] = {"tokens": tokens, "mask": mask}

    # `pre_ids` / `post_ids` are the chat-template text before and after the observation vectors
    # (see data/dataset.py and configs/model.yaml's prompt block). Padded independently — the
    # vectors sit between them in Captioner.forward, so they can't be one contiguous sequence.
    pre_ids = pad_sequence([ex["pre_ids"] for ex in examples], batch_first=True, padding_value=pad_token_id)
    post_ids = pad_sequence([ex["post_ids"] for ex in examples], batch_first=True, padding_value=pad_token_id)
    caption_ids = pad_sequence([ex["caption_ids"] for ex in examples], batch_first=True, padding_value=pad_token_id)

    pre_attn_mask = torch.zeros_like(pre_ids)
    post_attn_mask = torch.zeros_like(post_ids)
    caption_attn_mask = torch.zeros_like(caption_ids)
    for i, ex in enumerate(examples):
        pre_attn_mask[i, : ex["pre_ids"].shape[0]] = 1
        post_attn_mask[i, : ex["post_ids"].shape[0]] = 1
        caption_attn_mask[i, : ex["caption_ids"].shape[0]] = 1

    return {
        "modality_batch": modality_batch,
        "pre_ids": pre_ids,
        "pre_attn_mask": pre_attn_mask,
        "post_ids": post_ids,
        "post_attn_mask": post_attn_mask,
        "caption_ids": caption_ids,
        "caption_attn_mask": caption_attn_mask,
        "object_id": [ex["object_id"] for ex in examples],
        "shown": [ex["shown"] for ex in examples],
    }


def make_collate_fn(modality_names: list[str], out_dims: dict[str, int], max_tokens: dict[str, int], pad_token_id: int):
    def _fn(examples: list[dict[str, Any]]) -> dict[str, Any]:
        return collate_batch(examples, modality_names, out_dims, max_tokens, pad_token_id)

    return _fn
