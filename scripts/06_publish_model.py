#!/usr/bin/env python
"""Publishes a trained checkpoint to a private HF Hub model repo, so the team can pull it the
same way this pipeline already pulls everything else (AION, Qwen, AstroBridge-Data) — via
huggingface_hub, no separate storage account needed.

Publishes to `repo_id`:
  - adapter_config.json + adapter_model.safetensors — PEFT's own save_pretrained format for the
    LoRA deltas, loadable the standard way via `PeftModel.from_pretrained(base_llm, repo_id)`.
    Converted from the raw filtered state_dict this repo's checkpoints actually save
    (train/checkpoint.py's lora/adapter.pt) into that standard format here, not re-uploaded as-is.
  - middle.pt — the fusion stack (projectors, modality identity, Q-Former, adapter). Not a
    native HF/PEFT concept, so it's a raw torch.save'd state_dict; the generated README explains
    exactly how to reload it.
  - README.md — model card with the base model, training config hash/quantization/git sha, and
    the groundedness eval report if one exists at outputs/eval/groundedness_report.json.

Does NOT publish the base Qwen3.5-9B weights — those are frozen/untouched and stay sourced from
the original `Qwen/Qwen3.5-9B` repo.

Usage:
    python scripts/06_publish_model.py --checkpoint-dir outputs/checkpoints/stage2/best \\
        --repo-id your-org/astrobridge-captioner-v1
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from huggingface_hub import HfApi
from peft import LoraConfig, get_peft_model

from captioner.publish import build_model_card, filter_missing_lora_keys
from captioner.train.stage1 import build_llm
from captioner.utils.config import load_config, remaining_argv
from captioner.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", required=True, help="e.g. outputs/checkpoints/stage2/best")
    parser.add_argument("--repo-id", required=True, help="e.g. your-org/astrobridge-captioner-v1")
    parser.add_argument("--public", dest="private", action="store_false", default=True)
    parser.add_argument(
        "--eval-report", default="outputs/eval/groundedness_report.json",
        help="groundedness report to embed in the model card; must be newer than the checkpoint",
    )
    parser.add_argument(
        "--notes-file", default=None,
        help="markdown file appended to the model card under 'Release notes'",
    )
    parser.add_argument(
        "--allow-stale-report", action="store_true",
        help="publish even if the eval report predates the checkpoint (it will be wrong)",
    )
    args = parser.parse_args(remaining_argv())

    cfg = load_config("base", "data", "modalities", "model", "stage2")
    ckpt_dir = Path(args.checkpoint_dir)
    lora_dir = ckpt_dir / "lora"
    if not lora_dir.exists():
        raise FileNotFoundError(
            f"{lora_dir} not found — this publishes a stage 2 checkpoint (has a LoRA adapter). "
            "For a stage-1-only checkpoint there's no adapter to convert; just upload middle.pt "
            "directly instead of running this script."
        )

    logger.info(f"Loading base LLM ({cfg.llm.name}) and wrapping with the training-time LoRA config...")
    llm, tokenizer = build_llm(cfg)
    lora_config = LoraConfig(
        r=int(cfg.lora.r), lora_alpha=int(cfg.lora.alpha), lora_dropout=float(cfg.lora.dropout),
        target_modules=list(cfg.lora.target_modules), task_type="CAUSAL_LM",
    )
    llm = get_peft_model(llm, lora_config)

    lora_state = torch.load(lora_dir / "adapter.pt", map_location="cpu", weights_only=False)
    missing, _unexpected = llm.load_state_dict(lora_state, strict=False)
    real_missing = filter_missing_lora_keys(missing)
    if real_missing:
        raise RuntimeError(
            f"LoRA state from {lora_dir / 'adapter.pt'} didn't fully load onto a freshly "
            f"LoRA-wrapped {cfg.llm.name} — missing keys: {real_missing[:10]}. This means either "
            "the checkpoint doesn't match configs/stage2.yaml's current lora.* settings, or the "
            "checkpoint is corrupt."
        )
    logger.info("LoRA adapter loaded onto base model.")

    publish_dir = Path("outputs/publish") / args.repo_id.replace("/", "__")
    if publish_dir.exists():
        shutil.rmtree(publish_dir)
    publish_dir.mkdir(parents=True)

    logger.info(f"Saving PEFT adapter (standard format) + tokenizer to {publish_dir} ...")
    llm.save_pretrained(publish_dir)
    tokenizer.save_pretrained(publish_dir)
    shutil.copy(ckpt_dir / "middle.pt", publish_dir / "middle.pt")

    state = json.loads((ckpt_dir / "state.json").read_text())
    # The report path is an argument, not a constant, and it is checked against the checkpoint's
    # own mtime. Hardcoding "outputs/eval/groundedness_report.json" meant a publish silently
    # embedded whatever report happened to be sitting there — including a previous version's,
    # since 04_eval.py can be pointed at any --out path. Shipping v6's gate numbers on v7's model
    # card is worse than shipping none.
    eval_report_path = Path(args.eval_report)
    eval_report = None
    if eval_report_path.exists():
        if eval_report_path.stat().st_mtime < (ckpt_dir / "middle.pt").stat().st_mtime:
            if not args.allow_stale_report:
                raise SystemExit(
                    f"{eval_report_path} is OLDER than {ckpt_dir / 'middle.pt'} — it almost "
                    "certainly describes a previous checkpoint. Re-run scripts/04_eval.py against "
                    f"this checkpoint with --out {eval_report_path}, point --eval-report at the "
                    "right file, or pass --allow-stale-report if you really mean it."
                )
            logger.warning(f"{eval_report_path} predates the checkpoint; embedding it anyway.")
        eval_report = json.loads(eval_report_path.read_text())
    else:
        logger.warning(
            f"No eval report at {eval_report_path} — the model card will ship without gate "
            "numbers. Run scripts/04_eval.py first if you want them included."
        )
    notes = Path(args.notes_file).read_text() if args.notes_file else None
    (publish_dir / "README.md").write_text(
        build_model_card(cfg.llm.name, args.repo_id, state, eval_report, notes)
    )

    api = HfApi()
    api.create_repo(args.repo_id, private=args.private, exist_ok=True)
    logger.info(f"Uploading {publish_dir} to {args.repo_id} (private={args.private}) ...")
    api.upload_folder(folder_path=str(publish_dir), repo_id=args.repo_id)
    logger.info(f"Published: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
