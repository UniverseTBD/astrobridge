"""Stage 1: LLM frozen, encoders not loaded (training reads cached embeddings only). Trainable:
projectors, modality_identity, qformer, adapter — exactly `configs/stage1.yaml: trainable`.

Single-node multi-GPU via `accelerate` (DDP): each process loads a full copy of the model
normally (no `device_map="auto"` — that's model-parallel, incompatible with DDP) and
`accelerator.prepare(...)` wraps it for gradient synchronization. GPU placement is done by
`build_llm_staggered` one rank at a time instead of by `prepare`, which would otherwise have
every rank pay a ~15 GiB host-RAM transient simultaneously and OOM under a memory-capped
Slurm cgroup — see that function's docstring for the measurements. Launch with `accelerate launch scripts/03_train_stage1.py ...` — see
`captioner/README.md`'s cluster section for the `accelerate config` this expects.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import torch
from accelerate import Accelerator
from omegaconf import DictConfig
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from captioner.data.collate import make_collate_fn
from captioner.data.dataset import CaptionerDataset
from captioner.model.captioner import Captioner, FusionStack, llm_embedding_norm
from captioner.train.loop import run_training
from captioner.utils.logging import get_logger
from captioner.utils.seeding import seed_everything

logger = get_logger(__name__)


def _cosine_with_warmup(optimizer, total_steps: int, warmup_frac: float):
    warmup_steps = max(1, int(total_steps * warmup_frac))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        import math

        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


def build_llm(model_cfg: DictConfig, device=None):
    """Loads the frozen LLM. `device` (when given) places it before returning — see
    `build_llm_staggered` for why the placement belongs here rather than being left to
    `accelerator.prepare()` under multi-GPU DDP.
    """
    logger.info(f"Loading tokenizer for {model_cfg.llm.name} ...")
    trust_remote_code = bool(model_cfg.llm.get("trust_remote_code", False))
    tokenizer = AutoTokenizer.from_pretrained(model_cfg.llm.name, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    logger.info("Tokenizer loaded.")

    quant_config = None
    if model_cfg.llm.quantization == "nf4":
        from transformers import BitsAndBytesConfig

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
        )

    logger.info(
        f"Loading {model_cfg.llm.name} weights (dtype={model_cfg.llm.dtype}, "
        f"quantization={model_cfg.llm.quantization}) — for a model this size, this can take "
        "several minutes with no further output until it's done; that's normal, not stuck."
    )
    load_kwargs = dict(
        torch_dtype=torch.bfloat16 if model_cfg.llm.dtype == "bfloat16" else torch.float32,
        attn_implementation=model_cfg.llm.attn_impl,
        quantization_config=quant_config,
        trust_remote_code=trust_remote_code,
    )
    try:
        llm = AutoModelForCausalLM.from_pretrained(model_cfg.llm.name, **load_kwargs)
    except (ValueError, KeyError) as e:
        # Qwen3.5-9B's real config (confirmed via a live check) carries a `vision_config` —
        # architecture "Qwen3_5ForConditionalGeneration" is a native vision-language model, not
        # a plain causal LM, so AutoModelForCausalLM may not have it registered even once
        # transformers recognizes the "qwen3_5" model_type at all. This build never uses Qwen's
        # own vision pathway (LegacySurveyImage/DESISpectrum go through AION, not here) — a
        # generic AutoModel load still exposes .get_input_embeddings() and forward(inputs_embeds=
        # ...), which is all Captioner.forward actually needs.
        logger.warning(
            f"AutoModelForCausalLM.from_pretrained({model_cfg.llm.name!r}) failed "
            f"({type(e).__name__}: {e}); retrying with AutoModel (this architecture may be "
            "registered as vision-language, not causal-LM, but this build never uses its "
            "native vision pathway so a generic load should still work for our use — verify the "
            "resulting model actually has get_input_embeddings()/generate() before trusting it)."
        )
        from transformers import AutoModel

        llm = AutoModel.from_pretrained(model_cfg.llm.name, **load_kwargs)

    n_params = sum(p.numel() for p in llm.parameters())
    logger.info(f"LLM weights loaded ({n_params:,} parameters). Freezing and continuing.")
    for p in llm.parameters():
        p.requires_grad = False
    if device is not None:
        llm = llm.to(device)
    return llm, tokenizer


def build_llm_staggered(model_cfg: DictConfig, accelerator):
    """Loads the LLM one rank at a time, each placing it on its own GPU before the next starts.

    Measured directly on this build (Qwen3.5-9B bf16, 8.95B params, GH200): `from_pretrained`
    itself is nearly free in host RAM — the weights come back mmap-backed, ~0.35 GiB resident.
    Moving them to the GPU is what costs: RssAnon peaks at **14.66 GiB** during the transfer and
    falls back to ~0.38 GiB once it completes, because the CPU-side weights stay alive until the
    whole move finishes.

    Left to `accelerator.prepare()`, every rank pays that transient at the same moment, so peak
    host RAM is num_processes x 14.66 GiB — 58.6 GiB at 4 GPUs. Slurm caps this job's cgroup at
    SLURM_MEM_PER_NODE (64 GiB here), so that plus the session's own footprint trips the kernel
    OOM killer. It SIGKILLs a single rank; torch elastic then SIGTERMs the rest and reports a
    bare `ChildFailedError` with `exitcode -9` and no traceback, naming nothing about memory.

    Serializing just this step holds the peak at one rank's 14.66 GiB whatever the GPU count, at
    the cost of a slower startup (each rank's transfer runs alone rather than all at once).
    Nothing else in setup has a transient near this size, so nothing else is staggered.

    Two alternatives were measured and rejected: `device_map={"": rank}` is *worse* (17.49 GiB
    peak, since it stages through host memory too), and dropping to bf16 changes nothing because
    the weights are already bf16. Raising the job's memory request is the other real fix — it
    keeps the parallel load and costs nothing at runtime, so prefer it if you can get the RAM.
    """
    for turn in range(accelerator.num_processes):
        if accelerator.local_process_index == turn:
            if accelerator.num_processes > 1:
                logger.info(
                    f"Loading the LLM on rank {turn} ({turn + 1}/{accelerator.num_processes}) — "
                    "ranks load one at a time so the ~15 GiB host-RAM transient of the CPU->GPU "
                    "move is paid once rather than once per GPU. Startup is slower by design."
                )
            llm, tokenizer = build_llm(model_cfg, device=accelerator.device)
        accelerator.wait_for_everyone()
    return llm, tokenizer


def get_llm_hidden_size(llm) -> int:
    """`Qwen3_5ForConditionalGeneration`-style architectures often nest text config under
    `config.text_config` rather than exposing `hidden_size` flatly — check both rather than
    assume, so a config layout change fails loudly instead of picking up the wrong `d_llm`.
    """
    if hasattr(llm.config, "hidden_size"):
        return int(llm.config.hidden_size)
    if hasattr(llm.config, "text_config") and hasattr(llm.config.text_config, "hidden_size"):
        return int(llm.config.text_config.hidden_size)
    raise AttributeError(
        f"Could not find hidden_size on {type(llm.config).__name__} (checked both "
        "config.hidden_size and config.text_config.hidden_size) — inspect the real config "
        "object for this model and adjust get_llm_hidden_size()."
    )


def run_stage1(cfg: DictConfig) -> None:
    seed_everything(int(cfg.seed))

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
    model = Captioner(fusion_stack, llm, n_queries=int(cfg.qformer.n_queries))

    cache_root = Path(cfg.get("cache", {}).get("out_dir", "outputs/cache"))
    train_ds = CaptionerDataset(manifest, captions, cfg, cache_root, "train", tokenizer, cfg.prompt)
    val_ds = CaptionerDataset(manifest, captions, cfg, cache_root, "val", tokenizer, cfg.prompt)
    logger.info(f"Stage 1 datasets ready: train={len(train_ds)} val={len(val_ds)}")

    collate_fn = make_collate_fn(train_ds.modality_names, out_dims, max_tokens, tokenizer.pad_token_id)
    micro_bs = int(cfg.micro_batch_size)
    train_loader = DataLoader(train_ds, batch_size=micro_bs, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=micro_bs, shuffle=False, collate_fn=collate_fn)

    trainable_params = [p for _, p in fusion_stack.trainable_named_parameters(list(cfg.trainable))]
    optimizer = torch.optim.AdamW(trainable_params, lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))
    total_steps = (len(train_loader) // grad_accum) * int(cfg.epochs)
    scheduler = _cosine_with_warmup(optimizer, total_steps, float(cfg.schedule.warmup_frac))

    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )

    run_dir = Path(cfg.checkpoint.dir)
    result = run_training(
        accelerator, model, train_loader, val_loader, optimizer, scheduler, cfg, run_dir,
        epochs=int(cfg.epochs), early_stop_patience=int(cfg.early_stop.patience),
        tier_histogram=tier_histogram,
    )
    if accelerator.is_main_process:
        logger.info(f"Stage 1 complete: {result}")
