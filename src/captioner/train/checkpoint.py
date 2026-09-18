"""Checkpoint layout from §11:

outputs/checkpoints/{run}/{tag}/
├── middle.pt        # projectors + identity + qformer + adapter — small, keep all of them
├── lora/             # stage 2 only
├── optimizer.pt
├── rng_state.pt      # python/numpy/torch RNG state, for reproducible resume
└── state.json        # step, epoch, config hash, quantization, tier histogram, git sha
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf

from captioner.model.captioner import FusionStack
from captioner.utils.seeding import rng_state


def config_hash(cfg: DictConfig) -> str:
    canonical = OmegaConf.to_yaml(cfg, resolve=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent).decode().strip()
    except Exception:
        return "unknown"


def save_checkpoint(
    run_dir: Path,
    fusion_stack: FusionStack,
    optimizer: torch.optim.Optimizer,
    step: int,
    epoch: int,
    cfg: DictConfig,
    tier_histogram: dict[str, int],
    lora_state_dict: dict[str, Any] | None = None,
    fusion_state_dict_override: dict[str, Any] | None = None,
) -> None:
    """`fusion_state_dict_override`: pass this under `accelerate` (DDP) so the saved weights come
    from `accelerator.get_state_dict(...)` rather than a possibly-wrapped module's raw
    `.state_dict()` — see train/loop.py:_save.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(fusion_state_dict_override or fusion_stack.state_dict(), run_dir / "middle.pt")
    torch.save(optimizer.state_dict(), run_dir / "optimizer.pt")

    if lora_state_dict is not None:
        lora_dir = run_dir / "lora"
        lora_dir.mkdir(parents=True, exist_ok=True)
        torch.save(lora_state_dict, lora_dir / "adapter.pt")

    state = {
        "step": step,
        "epoch": epoch,
        "config_hash": config_hash(cfg),
        "quantization": cfg.llm.quantization if "llm" in cfg else None,
        "tier_histogram": tier_histogram,
        "git_sha": git_sha(),
    }
    torch.save(rng_state(), run_dir / "rng_state.pt")
    (run_dir / "state.json").write_text(json.dumps(state, indent=2, default=str))


def load_checkpoint(run_dir: Path) -> dict[str, Any]:
    state = json.loads((run_dir / "state.json").read_text())
    state["middle_path"] = run_dir / "middle.pt"
    state["optimizer_path"] = run_dir / "optimizer.pt"
    state["rng_path"] = run_dir / "rng_state.pt"
    lora_path = run_dir / "lora" / "adapter.pt"
    state["lora_path"] = lora_path if lora_path.exists() else None
    return state


def assert_quantization_matches(stage1_state_json: Path, stage2_cfg: DictConfig) -> None:
    """Pitfall #6: stage1 and stage2 must be trained against the same LLM quantization.
    Fail loudly on mismatch — this otherwise looks like an unrelated training regression.
    """
    stage1_state = json.loads(stage1_state_json.read_text())
    stage1_quant = stage1_state.get("quantization")
    stage2_quant = stage2_cfg.llm.quantization if "llm" in stage2_cfg else None
    if stage1_quant != stage2_quant:
        raise ValueError(
            f"Quantization mismatch: stage1 checkpoint was trained with quantization="
            f"{stage1_quant!r} but stage2 config specifies {stage2_quant!r}. The fusion stack "
            "was tuned against a different LLM embedding space than it would now feed — fix "
            "configs/model.yaml before continuing (§10 pitfall 6)."
        )
