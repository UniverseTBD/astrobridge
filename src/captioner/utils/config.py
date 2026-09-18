"""Config loading with Hydra-style dotted CLI overrides, e.g.

    python scripts/03_train_stage1.py modalities.image.encoder.hf_path=other-org/other-encoder \
        modalities.image.out_dim=1024 checkpoint.dir=outputs/checkpoints/stage1_abl_image_encoder_v2

No modality name or count is ever hardcoded here — overrides are applied generically by dotted
path against whatever configs are loaded.
"""
from __future__ import annotations

import sys
from pathlib import Path

from omegaconf import OmegaConf, DictConfig

CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs"


def _parse_cli_overrides(argv: list[str]) -> list[str]:
    """Only `key.path=value` tokens are treated as overrides; anything else is left for argparse."""
    return [a for a in argv if "=" in a and not a.startswith("-")]


def load_config(*names: str, argv: list[str] | None = None) -> DictConfig:
    """Load and merge one or more named configs from configs/, then apply CLI overrides.

    `names` are config stems, e.g. load_config("base", "data", "modalities", "model", "stage1").
    Later configs override earlier ones on key collision.
    """
    argv = sys.argv[1:] if argv is None else argv
    cfgs = [OmegaConf.load(CONFIG_DIR / f"{name}.yaml") for name in names]
    merged = OmegaConf.merge(*cfgs) if len(cfgs) > 1 else cfgs[0]

    overrides = _parse_cli_overrides(argv)
    if overrides:
        merged = OmegaConf.merge(merged, OmegaConf.from_dotlist(overrides))

    OmegaConf.resolve(merged)
    return merged


def remaining_argv(argv: list[str] | None = None) -> list[str]:
    """argv with dotted overrides stripped, for scripts that also take argparse flags."""
    argv = sys.argv[1:] if argv is None else argv
    overrides = set(_parse_cli_overrides(argv))
    return [a for a in argv if a not in overrides]
