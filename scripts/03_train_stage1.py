#!/usr/bin/env python
from __future__ import annotations

from captioner.train.stage1 import run_stage1
from captioner.utils.config import load_config


def main() -> None:
    cfg = load_config("base", "data", "modalities", "model", "stage1")
    run_stage1(cfg)


if __name__ == "__main__":
    main()
