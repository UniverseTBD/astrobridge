#!/usr/bin/env python
"""Checks HF access to everything this pipeline needs — two datasets (one of them gated, with a
specific file this build depends on) and two gated-or-not models — before you sink real time
into `make manifest`/`make cache`/`make stage1`. Metadata-only calls, no downloads.

Run this first on any machine/account that hasn't used this project before: `make check-access`.
"""
from __future__ import annotations

import sys

from captioner.utils.config import load_config
from captioner.utils.logging import get_logger

logger = get_logger(__name__)


def _check(label: str, fn) -> bool:
    try:
        fn()
        logger.info(f"OK    {label}")
        return True
    except Exception as e:
        logger.error(f"FAIL  {label}")
        logger.error(f"      {type(e).__name__}: {e}")
        return False


def main() -> None:
    cfg = load_config("base", "data", "modalities", "model")
    from huggingface_hub import HfApi

    api = HfApi()
    results = {}

    results["spectra dataset"] = _check(
        f"dataset access: {cfg.sources.spectra.hf_path}",
        lambda: api.dataset_info(cfg.sources.spectra.hf_path),
    )

    results["image dataset (repo)"] = _check(
        f"dataset access: {cfg.sources.image.hf_path}",
        lambda: api.dataset_info(cfg.sources.image.hf_path),
    )
    results["image caption dataset (repo)"] = _check(
        f"dataset access: {cfg.sources.image_captions.hf_path}",
        lambda: api.dataset_info(cfg.sources.image_captions.hf_path),
    )

    def _check_image_files():
        files = api.list_repo_files(cfg.sources.image.hf_path, repo_type="dataset")
        if "legacy_south_all_images.parquet" not in files:
            raise FileNotFoundError(
                "legacy_south_all_images.parquet not found in repo file listing — "
                "the image pixel-data pipeline (02_cache_embeddings.py) depends on this exact "
                "filename; check configs/data.yaml and data/image_dataset.py:FLUX_PARQUET_FILENAME."
            )
        cap_files = api.list_repo_files(cfg.sources.image_captions.hf_path, repo_type="dataset")
        if not any(f.startswith("data/") and f.endswith(".parquet") for f in cap_files):
            raise FileNotFoundError(
                f"No data/*.parquet in {cfg.sources.image_captions.hf_path} — image-tier "
                "captions (caption_fused) depend on these."
            )

    results["image dataset (required files)"] = _check(
        "legacy_south_all_images.parquet + image_captions data/*.parquet present in the repo listings",
        _check_image_files,
    )

    if "transients" in cfg.sources:
        results["transients dataset"] = _check(
            f"dataset access: {cfg.sources.transients.hf_path}",
            lambda: api.dataset_info(cfg.sources.transients.hf_path),
        )

    if "lightcurve" in cfg.modalities:
        lc_encoder = cfg.modalities.lightcurve.encoder
        onnx_file = lc_encoder.get("kwargs", {}).get("onnx_file", "atcat_f32.onnx")

        def _check_atcat():
            files = api.list_repo_files(lc_encoder.hf_path)
            if onnx_file not in files:
                raise FileNotFoundError(
                    f"{onnx_file} not found in {lc_encoder.hf_path} — the lightcurve encoder "
                    "depends on this exact filename; check configs/modalities.yaml."
                )

        results["ATCAT model"] = _check(
            f"model access: {lc_encoder.hf_path} ({onnx_file})", _check_atcat
        )

    aion_path = cfg.modalities.image.encoder.hf_path  # same repo for both modalities today
    results["AION model"] = _check(
        f"model access: {aion_path}",
        lambda: api.model_info(aion_path),
    )

    results["LLM"] = _check(
        f"model access: {cfg.llm.name}",
        lambda: api.model_info(cfg.llm.name),
    )

    logger.info("")
    n_ok = sum(results.values())
    n_total = len(results)
    if n_ok == n_total:
        logger.info(f"All {n_total} checks passed — ready to run the pipeline.")
    else:
        logger.error(f"{n_total - n_ok}/{n_total} checks failed.")
        logger.error(
            "For gated repos: request access on the HF page, wait for approval, then re-run "
            "this check. Also confirm you're logged in (`hf auth login` or `HF_TOKEN` "
            "set) on THIS machine — access granted on your account doesn't help if this "
            "environment isn't authenticated."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
