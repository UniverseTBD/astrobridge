"""Assembles the eval report (§8): shuffle test, ablation test, and joint_claim_fraction are
logged as first-class scalars alongside loss for every eval, for every modality in the registry.
"""
from __future__ import annotations

import json
from pathlib import Path

from torch.utils.data import Dataset

from captioner.eval.groundedness import ablation_test, diversity_test, shuffle_test
from captioner.model.captioner import Captioner
from captioner.utils.logging import get_logger

logger = get_logger(__name__)


def build_groundedness_report(
    model: Captioner, dataset: Dataset, modality_names: list[str], tokenizer, device: str, out_path: Path
) -> dict:
    logger.info(
        f"Running groundedness gate for {len(modality_names)} modalities ({list(modality_names)}) "
        "— each runs a shuffle test and an ablation test, generating a caption per example "
        "(up to n=200/400 generations each); this can take a while with a real LLM."
    )
    report: dict = {"per_modality": {}}
    for modality in modality_names:
        logger.info(f"[{modality}] shuffle_test starting...")
        shuffle_result = shuffle_test(model, dataset, modality, tokenizer, device)
        logger.info(f"[{modality}] ablation_test starting...")
        ablation_result = ablation_test(model, dataset, modality, tokenizer, device)
        logger.info(f"[{modality}] diversity_test starting...")
        diversity_result = diversity_test(model, dataset, modality, tokenizer, device)
        report["per_modality"][modality] = {
            "shuffle_test": shuffle_result,
            "ablation_test": ablation_result,
            "diversity_test": diversity_result,
        }
        if diversity_result.get("null_result"):
            logger.warning(
                f"[{modality}] DIVERSITY GATE FAILED: distinct_fraction="
                f"{diversity_result.get('distinct_fraction'):.3f} "
                f"top1_share={diversity_result.get('top1_share'):.3f}. The model is writing the "
                "same caption for different objects — check the prefix collapse ratio with "
                "diag/prefix_health.py before trusting anything else in this report."
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    return report
