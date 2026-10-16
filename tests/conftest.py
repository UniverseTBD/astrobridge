"""Shared fixtures for the training-pipeline tests (test_data_pipeline.py,
test_modality_loaders.py, test_model_training.py). Tests must run on CPU in seconds — a
TinyCausalLM stands in for Qwen so the model-correctness tests (masking, labels, overfit) don't
depend on real weights or a GPU.

Ported from Infonioknight/Astrobridge-Train's own tests/conftest.py — astrobridge's `src/captioner`
package is the same training code, so the same fixtures apply unchanged.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

from captioner.model.captioner import IGNORE_INDEX, Captioner, FusionStack

VOCAB_SIZE = 64
D_LLM = 32


def make_prompt_cfg(
    wrapper_pre: str = "<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n<observation>",
    wrapper_post: str = "</observation>\n{instruction}<|im_end|>\n<|im_start|>assistant\n",
    caption_suffix: str = "<|im_end|>",
    system_variants: list[str] | None = None,
    instruction_variants: list[str] | None = None,
):
    """The `prompt:` config block CaptionerDataset / generate_caption expect. Two variants each
    by default so the sampling helpers have something to choose between.
    """
    return OmegaConf.create(
        {
            "wrapper_pre": wrapper_pre,
            "wrapper_post": wrapper_post,
            "caption_suffix": caption_suffix,
            "system_variants": system_variants or ["You are an astronomy assistant.", "You are an expert astronomy assistant."],
            "instruction_variants": instruction_variants
            or ["Describe this observation.", "Describe the object shown, using only {modalities}."],
        }
    )


class TinyCausalLM(nn.Module):
    """Minimal HF-compatible causal LM: forward(inputs_embeds, attention_mask, labels) ->
    object with .loss and .logits, matching the contract Captioner.forward relies on.
    """

    def __init__(self, vocab_size: int = VOCAB_SIZE, d_model: int = D_LLM, n_layers: int = 2, n_heads: int = 4) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=d_model * 2, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, vocab_size)
        self.config = SimpleNamespace(hidden_size=d_model)

    def get_input_embeddings(self):
        return self.embed

    def forward(self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor):
        B, L, _ = inputs_embeds.shape
        causal_mask = torch.triu(torch.full((L, L), float("-inf")), diagonal=1).to(inputs_embeds.device)
        key_padding_mask = attention_mask == 0
        combined_mask = causal_mask.unsqueeze(0).expand(inputs_embeds.shape[0] * self.encoder.layers[0].self_attn.num_heads, -1, -1).clone()
        combined_mask = combined_mask.masked_fill(key_padding_mask.repeat_interleave(self.encoder.layers[0].self_attn.num_heads, dim=0).unsqueeze(1), float("-inf"))

        hidden = self.encoder(inputs_embeds, mask=combined_mask)
        logits = self.head(hidden)

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=IGNORE_INDEX
        )
        return SimpleNamespace(loss=loss, logits=logits)


@pytest.fixture
def modality_out_dims():
    return {"image": 8, "spectra": 6}


@pytest.fixture
def qformer_cfg():
    return dict(n_queries=4, d_model=16, n_layers=2, n_heads=2, ffn_mult=2, dropout=0.0)


@pytest.fixture
def fusion_stack(modality_out_dims, qformer_cfg):
    return FusionStack(
        modality_out_dims=modality_out_dims,
        d_shared=16,
        d_llm=D_LLM,
        qformer_cfg=qformer_cfg,
        projector_hidden_mult=2,
        projector_dropout=0.0,
    )


@pytest.fixture
def tiny_llm():
    return TinyCausalLM()


@pytest.fixture
def captioner(fusion_stack, tiny_llm, qformer_cfg):
    return Captioner(fusion_stack, tiny_llm, n_queries=qformer_cfg["n_queries"])


def make_modality_batch(B: int, out_dims: dict[str, int], T_max: dict[str, int], present: dict[str, torch.Tensor] | None = None):
    """present: {modality: bool tensor of shape (B,)} — which examples have that modality shown.
    Defaults to all-present for every modality.
    """
    batch = {}
    for name, D in out_dims.items():
        T = T_max[name]
        tokens = torch.randn(B, T, D)
        mask = torch.zeros(B, T, dtype=torch.bool)
        if present is not None and name in present:
            for i, is_present in enumerate(present[name]):
                if not is_present:
                    tokens[i] = 0.0
                    mask[i] = True
        batch[name] = {"tokens": tokens, "mask": mask}
    return batch
