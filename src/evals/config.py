from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

DEFAULT_HF_DATA_REPO: str = "UniverseTBD/AstroBridge-Data"
DEFAULT_HF_EVALS_SUBDIR: str = "evals/spectra"

def get_repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def get_hf_data_repo() -> str:
    return os.environ.get("ASTROBRIDGE_HF_DATA_REPO", DEFAULT_HF_DATA_REPO)


def get_hf_evals_subdir() -> str:
    return os.environ.get("ASTROBRIDGE_HF_EVALS_SUBDIR", DEFAULT_HF_EVALS_SUBDIR)


def get_hf_benchmark_subpath(filename: str, subdir: Optional[str] = None) -> str:
    base_subdir = subdir or get_hf_evals_subdir()
    return f"{base_subdir.rstrip('/')}/{filename}"


def get_default_cache_dir() -> Path:
    env_dir = os.environ.get("ASTROBRIDGE_DATA_DIR")
    if env_dir:
        return Path(env_dir).resolve()
    return get_repo_root() / "data" / "benchmarks"
