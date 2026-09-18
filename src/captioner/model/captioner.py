"""Fusion stack + sequence assembly against the frozen-or-LoRA LLM decoder.

Iterates over the modality registry (`modality_names`) everywhere — no `if image ... elif
spectra ...` branching (§0). Training never loads raw encoders; it consumes cached embeddings
via the batch dict produced by data/collate.py.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from captioner.model.adapter import Adapter
from captioner.model.modality_embed import ModalityIdentity
from captioner.model.projectors import ModalityProjector
from captioner.model.qformer import SharedQFormer

IGNORE_INDEX = -100


class FusionStack(nn.Module):
    """Everything upstream of the LLM: projectors -> modality identity -> Q-Former -> adapter.
    This is exactly what gets saved as middle.pt (§11).
    """

    def __init__(
        self,
        modality_out_dims: dict[str, int],
        d_shared: int,
        d_llm: int,
        qformer_cfg: dict,
        projector_hidden_mult: int,
        projector_dropout: float,
        adapter_target_norm: float = 1.0,
    ) -> None:
        super().__init__()
        self.modality_names = list(modality_out_dims.keys())
        self.projectors = nn.ModuleDict(
            {
                name: ModalityProjector(out_dim, d_shared, projector_hidden_mult, projector_dropout)
                for name, out_dim in modality_out_dims.items()
            }
        )
        self.modality_identity = ModalityIdentity(self.modality_names, d_shared)
        self.qformer = SharedQFormer(**qformer_cfg)
        # adapter_target_norm: the LLM's own median token-embedding norm. See Adapter's
        # docstring — the shipped v6 prefix was 34x this, which is a large part of why the LLM
        # learned to ignore it. Callers that have the real LLM must pass it.
        self.adapter = Adapter(
            d_shared,
            d_llm,
            dropout=qformer_cfg.get("dropout", 0.1),
            target_norm=adapter_target_norm,
        )

    def forward(self, modality_batch: dict[str, dict[str, Tensor]]) -> Tensor:
        """modality_batch: {name: {"tokens": (B, T_m, out_dim), "mask": (B, T_m) bool, True=pad/absent}}
        for every modality in the registry, present or not — absent examples arrive as an
        all-True mask row, never as unmasked zero content (§6).

        Returns adapter(qformer_out): (B, n_queries, d_llm).
        """
        token_chunks: list[Tensor] = []
        mask_chunks: list[Tensor] = []
        for name in self.modality_names:
            entry = modality_batch[name]
            projected = self.projectors[name](entry["tokens"])          # (B, T_m, d_shared)
            projected = self.modality_identity(projected, name)
            token_chunks.append(projected)
            mask_chunks.append(entry["mask"])

        tokens = torch.cat(token_chunks, dim=1)          # (B, T_total, d_shared)
        key_padding_mask = torch.cat(mask_chunks, dim=1)  # (B, T_total) bool, True = ignore

        queries = self.qformer(tokens, key_padding_mask=key_padding_mask)  # (B, n_queries, d_shared)
        return self.adapter(queries)                                       # (B, n_queries, d_llm)

    def trainable_named_parameters(self, groups: list[str]):
        """`groups` are names from configs/stage*.yaml `trainable` / `param_groups` lists —
        projectors, modality_identity, qformer, adapter."""
        modules = {
            "projectors": self.projectors,
            "modality_identity": self.modality_identity,
            "qformer": self.qformer,
            "adapter": self.adapter,
        }
        for group in groups:
            for n, p in modules[group].named_parameters():
                yield f"{group}.{n}", p


class Captioner(nn.Module):
    """Wraps FusionStack + the LLM and performs the inputs_embeds/labels assembly from §6."""

    def __init__(self, fusion_stack: FusionStack, llm: nn.Module, n_queries: int) -> None:
        super().__init__()
        self.fusion_stack = fusion_stack
        self.llm = llm
        self.n_queries = n_queries

    def forward(
        self,
        modality_batch: dict[str, dict[str, Tensor]],
        pre_ids: Tensor,          # (B, Ppre)  chat-template text BEFORE the observation vectors
        post_ids: Tensor,         # (B, Ppost) chat-template text AFTER the vectors, incl. the assistant header
        caption_ids: Tensor,      # (B, C)     the loss target
        pre_attn_mask: Tensor | None = None,      # (B, Ppre) 1 real / 0 pad
        post_attn_mask: Tensor | None = None,     # (B, Ppost) 1 real / 0 pad
        caption_attn_mask: Tensor | None = None,  # (B, C) 1 real / 0 pad
    ):
        device = pre_ids.device
        B, Ppre = pre_ids.shape
        Ppost = post_ids.shape[1]
        C = caption_ids.shape[1]

        prefix = self.fusion_stack(modality_batch)                        # (B, n_queries, d_llm)
        embed_fn = self.llm.get_input_embeddings()
        pre_embeds = embed_fn(pre_ids)                                     # (B, Ppre, d_llm)
        post_embeds = embed_fn(post_ids)                                   # (B, Ppost, d_llm)
        target_embeds = embed_fn(caption_ids)                             # (B, C, d_llm)

        # [ pre text | 48 observation vectors | post text (+ assistant header) | caption ]
        inputs_embeds = torch.cat([pre_embeds, prefix, post_embeds, target_embeds], dim=1)

        labels = torch.cat(
            [
                torch.full((B, Ppre), IGNORE_INDEX, dtype=torch.long, device=device),
                torch.full((B, self.n_queries), IGNORE_INDEX, dtype=torch.long, device=device),
                torch.full((B, Ppost), IGNORE_INDEX, dtype=torch.long, device=device),
                caption_ids.masked_fill(caption_attn_mask == 0, IGNORE_INDEX)
                if caption_attn_mask is not None
                else caption_ids,
            ],
            dim=1,
        )

        pre_mask = pre_attn_mask if pre_attn_mask is not None else torch.ones((B, Ppre), dtype=torch.long, device=device)
        prefix_mask = torch.ones((B, self.n_queries), dtype=torch.long, device=device)
        post_mask = post_attn_mask if post_attn_mask is not None else torch.ones((B, Ppost), dtype=torch.long, device=device)
        cap_mask = caption_attn_mask if caption_attn_mask is not None else torch.ones((B, C), dtype=torch.long, device=device)
        attention_mask = torch.cat([pre_mask, prefix_mask, post_mask, cap_mask], dim=1)

        return self.llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)


def llm_embedding_norm(llm: nn.Module) -> float:
    """Median L2 norm of the LLM's input token embeddings — the scale a prefix vector must match
    to read as a token rather than as an outlier. Pass into FusionStack(adapter_target_norm=...).
    """
    with torch.no_grad():
        weight = llm.get_input_embeddings().weight
        return float(weight.float().norm(dim=1).median())
