#!/usr/bin/env python
"""Debiased crowd-vote-fraction soft score — parked as its own script, separate from
`score_image_eval.py`, since it needs a real network crossmatch against
`astronolan/gz-decals-embeddings` and is meaningfully slower than the hard/group metrics (no GPU
either, but a real download the first time it runs). Run this alongside `score_image_eval.py`
whenever the finer-grained, empirically-grounded partial credit is wanted, not by default every
time.

A wrong prediction scores how plausible it actually was, according to real Galaxy Zoo DECaLS
volunteer votes for that specific object (`eval.metrics.vote_fraction_scoring`), not a flat 0 —
see that module's docstring for the real mechanism (each of Galaxy10's 10 classes has a real,
published defining rule, transcribed from `henrysky/Galaxy10`'s own construction notebook; a
continuous "how well does this object satisfy that rule" margin per class, rescaled so the true
class lands at exactly 1.0).

**Real, structural coverage limit, not a bug**: confirmed live against the full 796-object test
set, `galaxy10-aion` and `gz-decals-embeddings` are only ~41.5% overlapping populations, flat
across crossmatch radii from 1 to 10 arcsec — so at any sample size, roughly 58% of objects will
be excluded via `n_excluded_no_crossmatch`, reported explicitly, never silently dropped.

Usage:
    uv run python -m eval.runners.score_image_eval_debiased --in outputs/eval/raw_generations/galaxy10_seed0_n150.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from captioner.utils.logging import get_logger
from eval.datasets.gz_decals_votes import crossmatch_to_sample, load_vote_fractions
from eval.datasets.image_galaxy10 import CLASS_CODES, GALAXY10_LABELS
from eval.metrics.caption_to_label import GALAXY10_LABEL_SYNONYMS, make_predictor
from eval.metrics.vote_fraction_scoring import score_prediction, soft_label_vector

logger = get_logger(__name__)


def _soft_report(crossmatched: pd.DataFrame, answer_key: str, predict) -> dict:
    scores = []
    n_excluded_no_crossmatch = 0
    n_excluded_true_class_unscoreable = 0
    for _, row in crossmatched.iterrows():
        if not row["_crossmatched"]:
            n_excluded_no_crossmatch += 1
            continue
        vector = soft_label_vector(row, row["label_name"])
        if vector is None:
            n_excluded_true_class_unscoreable += 1
            continue
        predicted = predict(row[answer_key])
        score = score_prediction(vector, predicted)
        if score is not None:
            scores.append(score)

    return {
        "n_scored": len(scores),
        "n_excluded_no_crossmatch": n_excluded_no_crossmatch,
        "n_excluded_true_class_unscoreable": n_excluded_true_class_unscoreable,
        "n_excluded_predicted_class_unscoreable_or_unparseable": len(crossmatched) - len(scores) - n_excluded_no_crossmatch - n_excluded_true_class_unscoreable,
        "mean_soft_score": (sum(scores) / len(scores)) if scores else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", required=True, help="a collect_image_labels.py output JSON")
    parser.add_argument("--crossmatch-radius-arcsec", type=float, default=1.0)
    parser.add_argument("--out", default=None, help="defaults alongside --in, suffixed _debiased_scored.json")
    args = parser.parse_args()

    data = json.loads(Path(args.in_path).read_text())
    objects = data["objects"]
    answer_format = data.get("answer_format", "free_text")  # older collect files predate this key
    # Read the mapping the collect run ACTUALLY used (see score_image_eval.py's matching comment
    # for why this isn't optional) — falls back to the default only for files that predate this
    # field entirely (never used --shuffle-codes).
    file_class_codes = data.get("class_codes", CLASS_CODES)
    predict = make_predictor(answer_format, GALAXY10_LABELS, GALAXY10_LABEL_SYNONYMS, file_class_codes)
    logger.info(
        f"Debiased-scoring {len(objects)} objects from {args.in_path} "
        f"(sampling seed={data['sampling']['seed']}, answer_format={answer_format!r}, "
        f"shuffle_codes={data.get('shuffle_codes', False)})."
    )

    logger.info("Crossmatching sample against astronolan/gz-decals-embeddings for soft scoring...")
    sample_df = pd.DataFrame(objects)
    votes = load_vote_fractions()
    crossmatched = crossmatch_to_sample(sample_df, votes, radius_arcsec=args.crossmatch_radius_arcsec)

    report = {
        "source": args.in_path,
        "dataset": data["dataset"],
        "repo_id": data["repo_id"],
        "answer_format": answer_format,
        "class_codes": file_class_codes,
        "shuffle_codes": data.get("shuffle_codes", False),
        "sampling": data["sampling"],
        "crossmatch_radius_arcsec": args.crossmatch_radius_arcsec,
        "soft": {
            "base": _soft_report(crossmatched, "base_answer", predict),
            "equipped": _soft_report(crossmatched, "equipped_answer", predict),
        },
    }

    out_path = Path(args.out) if args.out else Path(args.in_path).with_name(Path(args.in_path).stem + "_debiased_scored.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    logger.info(json.dumps(report, indent=2))
    logger.info(f"Wrote debiased-scored report to {out_path}")


if __name__ == "__main__":
    main()
