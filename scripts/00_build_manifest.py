#!/usr/bin/env python
from __future__ import annotations

from captioner.data.manifest import write_manifest
from captioner.utils.config import load_config


def main() -> None:
    cfg = load_config("base", "data")
    write_manifest(cfg)


if __name__ == "__main__":
    main()
