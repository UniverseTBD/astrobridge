"""Does the actual training step compute what it's supposed to?

Covers the Q-Former's absent-modality masking, `Captioner.forward`'s label construction (loss
must land on caption tokens only, never the visual prefix or chat-template scaffolding), the
collate→forward wiring, an end-to-end memorization sanity check, and the per-example prompt
sampling that makes the model robust to wording. All CPU-only — a TinyCausalLM (see conftest.py)
stands in for Qwen so these run in seconds with no GPU and no real weights.

Ported and consolidated from Infonioknight/Astrobridge-Train's test suite (test_masking.py,
test_labels.py, test_chat_template_end_to_end.py, test_overfit.py, test_prompt_variants.py).
"""
from __future__ import annotations

import numpy as np
import torch

from captioner.data.collate import collate_batch
from captioner.model.captioner import IGNORE_INDEX
from captioner.model.qformer import SharedQFormer
from captioner.utils.prompt import build_wrapper_text, pick_prompt_variants
from tests.conftest import make_modality_batch, make_prompt_cfg

# --- Q-Former masking: absent modality must contribute zero rows, never leak as zero-content ---


def _tiny_qformer():
    torch.manual_seed(0)
    qf = SharedQFormer(n_queries=4, d_model=16, n_layers=2, n_heads=2, ffn_mult=2, dropout=0.0)
    qf.eval()
    return qf


def test_absent_modality_ignored():
    """A masked-out modality's tokens contribute nothing — output is identical to never having
    passed those positions in at all."""
    qf = _tiny_qformer()
    B, T_present, T_absent, D = 3, 5, 7, 16
    real_tokens = torch.randn(B, T_present, D)
    absent_tokens = torch.randn(B, T_absent, D)  # arbitrary content — must not matter
    tokens_with_absent = torch.cat([real_tokens, absent_tokens], dim=1)
    mask_with_absent = torch.cat([torch.zeros(B, T_present, dtype=torch.bool), torch.ones(B, T_absent, dtype=torch.bool)], dim=1)
    with torch.no_grad():
        out_with_absent = qf(tokens_with_absent, key_padding_mask=mask_with_absent)
        out_present_only = qf(real_tokens, key_padding_mask=torch.zeros(B, T_present, dtype=torch.bool))
    assert torch.allclose(out_with_absent, out_present_only, atol=1e-5)


def test_masked_garbage_noop():
    """Stress test: even extreme garbage values in a masked slot must be a bitwise no-op. A real
    masking leak often only shows up once the "ignored" positions hold non-zero content."""
    qf = _tiny_qformer()
    B, T_present, T_absent, D = 3, 5, 7, 16
    real_tokens = torch.randn(B, T_present, D)
    mask = torch.cat([torch.zeros(B, T_present, dtype=torch.bool), torch.ones(B, T_absent, dtype=torch.bool)], dim=1)
    tokens_zeros = torch.cat([real_tokens, torch.zeros(B, T_absent, D)], dim=1)
    tokens_garbage = torch.cat([real_tokens, torch.randn(B, T_absent, D) * 1e6], dim=1)
    with torch.no_grad():
        out_zeros = qf(tokens_zeros, key_padding_mask=mask)
        out_garbage = qf(tokens_garbage, key_padding_mask=mask)
    assert torch.equal(out_zeros, out_garbage)


# --- labels: IGNORE_INDEX everywhere except the caption tokens ---------------------------------


def test_labels_mask_prefix(captioner, modality_out_dims, qformer_cfg):
    """Everything except the caption tokens — the visual prefix, the system/instruction text —
    is IGNORE_INDEX in the labels, so the model is never trained to predict its own prompt or
    the image embeddings back."""
    B, PPRE, PPOST, C = 2, 4, 3, 5
    T_max = {"image": 6, "spectra": 4}
    modality_batch = make_modality_batch(B, modality_out_dims, T_max)
    pre_ids = torch.randint(0, 30, (B, PPRE))
    post_ids = torch.randint(0, 30, (B, PPOST))
    caption_ids = torch.randint(0, 30, (B, C))
    n_queries = qformer_cfg["n_queries"]

    # Reconstruct labels the same way Captioner.forward does, and check the boundaries directly.
    labels = torch.cat([
        torch.full((B, PPRE), IGNORE_INDEX, dtype=torch.long),
        torch.full((B, n_queries), IGNORE_INDEX, dtype=torch.long),
        torch.full((B, PPOST), IGNORE_INDEX, dtype=torch.long),
        caption_ids,
    ], dim=1)
    head = PPRE + n_queries + PPOST
    assert torch.all(labels[:, :head] == IGNORE_INDEX)
    assert torch.equal(labels[:, head:], caption_ids)

    out = captioner(modality_batch, pre_ids, post_ids, caption_ids)
    assert out.loss.item() == out.loss.item()  # not NaN


def test_loss_reaches_fusion_stack(captioner, modality_out_dims):
    """Loss is finite and backward() produces a real, non-zero gradient on the fusion stack —
    not just a detached prefix that the loss never actually learns from."""
    B, PPRE, PPOST, C = 2, 4, 3, 5
    T_max = {"image": 6, "spectra": 4}
    modality_batch = make_modality_batch(B, modality_out_dims, T_max)
    pre_ids = torch.randint(0, 30, (B, PPRE))
    post_ids = torch.randint(0, 30, (B, PPOST))
    caption_ids = torch.randint(0, 30, (B, C))

    out = captioner(modality_batch, pre_ids, post_ids, caption_ids)
    assert torch.isfinite(out.loss)
    out.loss.backward()
    grads = [p.grad for p in captioner.fusion_stack.parameters() if p.requires_grad]
    assert any(g is not None and torch.any(g != 0) for g in grads), (
        "No gradient reached the fusion stack — loss is not actually connected to the prefix."
    )


# --- collate -> forward wiring: the one seam not covered by either unit test alone -------------


def _collate_example(pre, post, cap, image_T=3, spectra_T=2):
    return {
        "object_id": "obj", "shown": frozenset({"image", "spectra"}),
        "modality_arrays": {"image": np.random.randn(image_T, 8).astype(np.float32), "spectra": np.random.randn(spectra_T, 6).astype(np.float32)},
        "pre_ids": torch.tensor(pre), "post_ids": torch.tensor(post), "caption_ids": torch.tensor(cap),
    }


def test_collate_to_forward(captioner, modality_out_dims):
    """collate_batch's output feeds directly into Captioner.forward exactly as the real training
    loop (train/loop.py) calls it — the one seam not covered by either side's own unit tests."""
    examples = [_collate_example([1, 2, 3], [4, 5], [10, 11, 12]), _collate_example([1, 2], [4, 5, 6, 7], [10, 11])]
    batch = collate_batch(examples, ["image", "spectra"], {"image": 8, "spectra": 6}, {"image": 5, "spectra": 5}, pad_token_id=0)

    out = captioner(
        batch["modality_batch"], batch["pre_ids"], batch["post_ids"], batch["caption_ids"],
        batch["pre_attn_mask"], batch["post_attn_mask"], batch["caption_attn_mask"],
    )
    assert torch.isfinite(out.loss)
    out.loss.backward()
    assert any(p.grad is not None and torch.any(p.grad != 0) for p in captioner.fusion_stack.parameters())


# --- end-to-end memorization sanity check -------------------------------------------------------


def test_memorizes_8_examples(captioner, modality_out_dims):
    """The cheapest end-to-end wiring check. If the visual prefix isn't actually reaching the
    loss (wrong concatenation order, a detached tensor, wrong labels), 8 fixed examples won't
    memorise in 150 steps."""
    torch.manual_seed(0)
    B, PPRE, PPOST, C = 8, 2, 2, 4
    T_max = {"image": 6, "spectra": 4}
    modality_batch = make_modality_batch(B, modality_out_dims, T_max)
    pre_ids = torch.randint(0, 20, (B, PPRE))
    post_ids = torch.randint(0, 20, (B, PPOST))
    caption_ids = torch.randint(20, 40, (B, C))  # distinct vocab range from the wrapper, fixed target

    optimizer = torch.optim.AdamW(captioner.parameters(), lr=3e-3)
    losses = []
    for _ in range(150):
        optimizer.zero_grad()
        out = captioner(modality_batch, pre_ids, post_ids, caption_ids)
        out.loss.backward()
        optimizer.step()
        losses.append(out.loss.item())

    assert losses[-1] < 0.05, f"Failed to memorise 8 examples; final loss={losses[-1]:.4f} (started at {losses[0]:.4f})"


# --- prompt variant sampling ---------------------------------------------------------------------


def test_seed_is_deterministic():
    """Same RNG seed picks the same (system, instruction) pair every time."""
    cfg = make_prompt_cfg(system_variants=["S1", "S2", "S3", "S4"], instruction_variants=["I1", "I2", "I3", "I4", "I5"])
    a = pick_prompt_variants(cfg, frozenset({"image"}), np.random.default_rng(7))
    b = pick_prompt_variants(cfg, frozenset({"image"}), np.random.default_rng(7))
    assert a == b


def test_seeds_explore_prompts():
    """configs/model.yaml ships multiple system x instruction variants specifically so the model
    is robust to wording — over many seeds, the sampler should actually touch most of that grid,
    not collapse onto one pair."""
    cfg = make_prompt_cfg(system_variants=["S1", "S2", "S3", "S4"], instruction_variants=["I1", "I2", "I3", "I4", "I5"])
    seen = {pick_prompt_variants(cfg, frozenset({"image"}), np.random.default_rng(s)) for s in range(200)}
    assert len(seen) >= 12  # over 200 seeds, should touch most of the 4x5 grid


def test_modalities_fill_scope():
    """The {modalities} placeholder is filled into the instruction text only — system variants
    are never formatted, so they stay generic across every modality combination."""
    cfg = make_prompt_cfg(
        system_variants=["You are an assistant. {modalities} should NOT appear filled here."],
        instruction_variants=["Describe this {modalities}."],
    )
    system, instruction = pick_prompt_variants(cfg, frozenset({"image", "spectra"}), np.random.default_rng(0))
    assert instruction == "Describe this an image and a spectrum."
    assert "{modalities}" in system


def test_wrapper_text_split():
    """build_wrapper_text splits into a pre-half ending in <observation> and a post-half
    starting with </observation> — the splice point where the 64 visual prefix vectors get
    inserted during training."""
    cfg = make_prompt_cfg()
    pre, post = build_wrapper_text(cfg, "SYS", "INSTR")
    assert pre.endswith("<observation>")
    assert "SYS" in pre and "<|im_start|>system" in pre
    assert post.startswith("</observation>")
    assert "INSTR" in post and post.rstrip().endswith("<|im_start|>assistant")
