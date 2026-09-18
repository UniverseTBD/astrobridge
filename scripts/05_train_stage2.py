#!/usr/bin/env python
from __future__ import annotations

from captioner.train.stage2 import run_stage2
from captioner.utils.config import load_config


def main() -> None:
    cfg = load_config("base", "data", "modalities", "model", "stage2")
    run_stage2(cfg)


if __name__ == "__main__":
    main()
