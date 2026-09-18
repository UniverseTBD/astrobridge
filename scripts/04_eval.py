#!/usr/bin/env python
"""Groundedness gate (§9 step 7 / §9 step 8): shuffle + ablation tests must be non-null for
every modality before stage 2 is allowed to start, and again after stage 2 for comparison.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from peft import PeftModel

from captioner.data.dataset import CaptionerDataset
from captioner.eval.report import build_groundedness_report
from captioner.model.captioner import Captioner, FusionStack, llm_embedding_norm
from captioner.train.stage1 import build_llm, get_llm_hidden_size
from captioner.utils.config import load_config, remaining_argv
from captioner.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", required=True, help="e.g. outputs/checkpoints/stage1/best")
    parser.add_argument("--lora-dir", default=None, help="outputs/checkpoints/stage2/best/lora, if evaluating stage 2")
    parser.add_argument("--split", default="test")
    parser.add_argument("--out", default="outputs/eval/groundedness_report.json")
    args = parser.parse_args(remaining_argv())

    cfg = load_config("base", "data", "modalities", "model", "stage2")
    manifest = pd.read_parquet(cfg.manifest.parquet)
    captions = pd.read_parquet(cfg.captions.parquet)

    llm, tokenizer = build_llm(cfg)
    if args.lora_dir:
        # Two different on-disk LoRA layouts reach this flag, and only one of them is PEFT's.
        # train/checkpoint.py writes this repo's own format — a raw filtered state_dict at
        # lora/adapter.pt with no adapter_config.json — while 06_publish_model.py converts that
        # into PEFT's save_pretrained layout for the Hub. `PeftModel.from_pretrained` only reads
        # the latter and dies with "Can't find 'adapter_config.json'" on a training checkpoint,
        # which is exactly what this gate is normally pointed at. Rebuild the config from
        # configs/stage2.yaml in that case, the same way inference.py already does.
        lora_dir = Path(args.lora_dir)
        if (lora_dir / "adapter_config.json").exists():
            llm = PeftModel.from_pretrained(llm, str(lora_dir))
        else:
            from peft import LoraConfig, get_peft_model

            from captioner.publish import filter_missing_lora_keys

            llm = get_peft_model(
                llm,
                LoraConfig(
                    r=int(cfg.lora.r), lora_alpha=int(cfg.lora.alpha),
                    lora_dropout=float(cfg.lora.dropout),
                    target_modules=list(cfg.lora.target_modules), task_type="CAUSAL_LM",
                ),
            )
            missing, _ = llm.load_state_dict(
                torch.load(lora_dir / "adapter.pt", map_location="cpu", weights_only=False),
                strict=False,
            )
            real_missing = filter_missing_lora_keys(missing)
            if real_missing:
                raise RuntimeError(
                    f"LoRA state from {lora_dir / 'adapter.pt'} didn't fully load — missing "
                    f"{real_missing[:10]}. Check it matches configs/stage2.yaml's lora.* settings."
                )
    d_llm = get_llm_hidden_size(llm)

    out_dims = {n: int(c.out_dim) for n, c in cfg.modalities.items()}
    fusion_stack = FusionStack(
        modality_out_dims=out_dims,
        d_shared=int(cfg.d_shared),
        d_llm=d_llm,
        qformer_cfg=dict(cfg.qformer),
        projector_hidden_mult=int(cfg.projector.hidden_mult),
        projector_dropout=float(cfg.projector.dropout),
        adapter_target_norm=llm_embedding_norm(llm),
    )
    fusion_stack.load_state_dict(
        torch.load(Path(args.checkpoint_dir) / "middle.pt", map_location="cpu", weights_only=False)
    )
    model = Captioner(fusion_stack, llm, n_queries=int(cfg.qformer.n_queries))
    model.to(cfg.get("device", "cuda"))

    cache_root = Path(cfg.get("cache", {}).get("out_dir", "outputs/cache"))
    dataset = CaptionerDataset(manifest, captions, cfg, cache_root, args.split, tokenizer, cfg.prompt)

    report = build_groundedness_report(
        model, dataset, dataset.modality_names, tokenizer, cfg.get("device", "cuda"), Path(args.out)
    )
    logger.info(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
