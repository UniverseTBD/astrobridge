"""Stage 2: fusion stack (from stage1) + LoRA on the LLM. Encoders still not loaded — cached
embeddings only. Asserts quantization matches the stage1 checkpoint (§10 pitfall 6) before doing
any work.

Same single-node multi-GPU (DDP, via `accelerate`) pattern as stage1.py — see that file's
docstring and `captioner/README.md`'s cluster section.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import torch
from accelerate import Accelerator
from omegaconf import DictConfig
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader

from captioner.data.collate import make_collate_fn
from captioner.data.dataset import CaptionerDataset
from captioner.model.captioner import Captioner, FusionStack, llm_embedding_norm
from captioner.train.checkpoint import assert_quantization_matches
from captioner.train.loop import run_training
from captioner.train.stage1 import _cosine_with_warmup, build_llm_staggered, get_llm_hidden_size
from captioner.utils.logging import get_logger
from captioner.utils.seeding import seed_everything

logger = get_logger(__name__)


def run_stage2(cfg: DictConfig) -> None:
    seed_everything(int(cfg.seed))

    stage1_state_json = Path(cfg.init_from).parent / "state.json"
    assert_quantization_matches(stage1_state_json, cfg)

    # `batch_size` is a GLOBAL budget, so the per-process accumulation has to divide by the
    # world size. Without the `num_processes` term (the pre-fix behaviour) `batch_size: 32` meant
    # 32 PER RANK: the v6 run on 8 ranks trained at an effective batch of 4 x 8 x 8 = 256, which
    # left 4144 train examples / 256 = 17 optimizer steps per epoch and 510 steps for all of
    # stage 1. That is the single biggest reason the fusion stack never learned to condition on
    # anything — see DIAGNOSIS.md fault 2.
    world = max(1, int(os.environ.get("WORLD_SIZE", "1")))
    grad_accum = max(1, int(cfg.batch_size) // (int(cfg.micro_batch_size) * world))
    accelerator = Accelerator(gradient_accumulation_steps=grad_accum)

    manifest = pd.read_parquet(cfg.manifest.parquet)
    captions = pd.read_parquet(cfg.captions.parquet)
    tier_histogram = json.loads(Path(cfg.manifest.stats).read_text())["tier_histogram"]

    llm, tokenizer = build_llm_staggered(cfg, accelerator)
    d_llm = get_llm_hidden_size(llm)

    lora_config = LoraConfig(
        r=int(cfg.lora.r),
        lora_alpha=int(cfg.lora.alpha),
        lora_dropout=float(cfg.lora.dropout),
        target_modules=list(cfg.lora.target_modules),
        task_type="CAUSAL_LM",
    )
    llm = get_peft_model(llm, lora_config)

    n_lora_params = sum(p.numel() for n, p in llm.named_parameters() if "lora_" in n and p.requires_grad)
    if n_lora_params == 0:
        raise RuntimeError(
            f"target_modules={list(cfg.lora.target_modules)} matched zero LoRA-wrapped parameters "
            f"on {cfg.llm.name} — PEFT applies LoRA silently to whatever matches, so a wrong name "
            "here would otherwise train nothing on the LLM side without any error. Qwen3.5's "
            "Gated DeltaNet layers may not use the standard q_proj/k_proj/v_proj/o_proj naming — "
            "inspect `[n for n, _ in llm.named_modules()]` on the base model and update "
            "configs/stage2.yaml's lora.target_modules to match."
        )
    logger.info(f"LoRA applied to {n_lora_params:,} parameters across the LLM")

    out_dims = {n: int(c.out_dim) for n, c in cfg.modalities.items()}
    max_tokens = {n: int(c.max_tokens) for n, c in cfg.modalities.items()}

    fusion_stack = FusionStack(
        modality_out_dims=out_dims,
        d_shared=int(cfg.d_shared),
        d_llm=d_llm,
        qformer_cfg=dict(cfg.qformer),
        projector_hidden_mult=int(cfg.projector.hidden_mult),
        projector_dropout=float(cfg.projector.dropout),
        adapter_target_norm=llm_embedding_norm(llm),
    )
    logger.info(f"Loading stage 1 fusion stack checkpoint from {cfg.init_from} ...")
    fusion_stack.load_state_dict(torch.load(cfg.init_from, map_location="cpu", weights_only=False))
    logger.info("Stage 1 checkpoint loaded.")

    for p in fusion_stack.parameters():
        p.requires_grad = True

    model = Captioner(fusion_stack, llm, n_queries=int(cfg.qformer.n_queries))

    cache_root = Path(cfg.get("cache", {}).get("out_dir", "outputs/cache"))
    train_ds = CaptionerDataset(manifest, captions, cfg, cache_root, "train", tokenizer, cfg.prompt)
    val_ds = CaptionerDataset(manifest, captions, cfg, cache_root, "val", tokenizer, cfg.prompt)
    logger.info(f"Stage 2 datasets ready: train={len(train_ds)} val={len(val_ds)}")

    collate_fn = make_collate_fn(train_ds.modality_names, out_dims, max_tokens, tokenizer.pad_token_id)
    micro_bs = int(cfg.micro_batch_size)
    train_loader = DataLoader(train_ds, batch_size=micro_bs, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=micro_bs, shuffle=False, collate_fn=collate_fn)

    param_groups = []
    for group in cfg.param_groups:
        if group.params == "lora":
            params = [p for p in llm.parameters() if p.requires_grad]
        else:
            params = [p for _, p in fusion_stack.trainable_named_parameters(list(group.params))]
        param_groups.append({"params": params, "lr": float(group.lr)})

    optimizer = torch.optim.AdamW(param_groups, weight_decay=float(cfg.weight_decay))
    total_steps = (len(train_loader) // grad_accum) * int(cfg.epochs)
    scheduler = _cosine_with_warmup(optimizer, total_steps, float(cfg.warmup_frac))

    def lora_state_fn():
        # Captures `llm` directly (not the accelerator-prepared/possibly-DDP-wrapped `model`) —
        # the underlying parameter tensors are updated in place by the optimizer either way, so
        # this stays correct after accelerator.prepare() below.
        return {k: v for k, v in llm.state_dict().items() if "lora_" in k}

    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )

    run_dir = Path(cfg.checkpoint.dir)
    result = run_training(
        accelerator, model, train_loader, val_loader, optimizer, scheduler, cfg, run_dir,
        epochs=int(cfg.epochs), early_stop_patience=int(cfg.early_stop.patience),
        tier_histogram=tier_histogram, lora_state_fn=lora_state_fn,
    )
    if accelerator.is_main_process:
        logger.info(f"Stage 2 complete: {result}")
