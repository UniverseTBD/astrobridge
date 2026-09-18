#!/usr/bin/env python
"""Step 2/2 of the image eval (default/primary path): score a `collect_image_labels.py` output
file with the two metrics that need no external data at all. Pure post-processing — no GPU, no
Modal, no model, no network — so this is fast and can be re-run as many times as needed without
ever re-paying for inference.

Reports two metrics side by side, both for base and equipped:
  - **hard**: hard-label accuracy + per-class precision/recall/F1 (`eval.metrics.classification`)
    — the model's answer either exactly matches the true label or it doesn't.
  - **group**: coarse 1.0/0.5/0.0 partial credit by the 4 top-level morphology groups (Smooth,
    Spiral, Edge-On, Disturbed/Merging — `eval.metrics.group_scoring`). A same-group miss (e.g.
    Round Smooth predicted as In-between Round Smooth) scores 0.5 instead of a flat 0.

For `logprob_argmax` files, `hard` also carries a `margin` block (see
`eval.runners.score_lightcurve_eval`'s identical metric): the gap between the winning and
runner-up candidate log-prob, a calibration signal orthogonal to raw accuracy.

For the crowd-vote-fraction-grounded soft score (a finer-grained, empirically-grounded partial
credit, using real Galaxy Zoo DECaLS volunteer votes — genuinely different from `group`, not a
duplicate), see the separate `score_image_eval_debiased.py`.

Usage:
    uv run python -m eval.runners.score_image_eval --in outputs/eval/raw_generations/galaxy10_logprob_argmax_seed0_n150.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from captioner.utils.logging import get_logger
from eval.datasets.image_galaxy10 import CLASS_CODES, GALAXY10_LABELS
from eval.metrics.caption_to_label import GALAXY10_LABEL_SYNONYMS, make_predictor
from eval.metrics.classification import classification_report
from eval.metrics.group_scoring import group_report

logger = get_logger(__name__)


def _predictions_for(objects: list[dict], answer_format: str, side: str, predict) -> list[str | None]:
    if answer_format == "logprob_argmax":
        return [o[side]["predicted_label"] for o in objects]
    return [predict(o[f"{side}_answer"]) for o in objects]


def _margin_block(objects: list[dict], side: str) -> dict:
    margins = []
    for o in objects:
        scores = sorted(o[side]["logprobs"].values(), reverse=True)
        margins.append(scores[0] - scores[1])
    return {"mean": sum(margins) / len(margins) if margins else 0.0, "n_below_0.5": sum(1 for m in margins if m < 0.5)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", required=True, help="a collect_image_labels.py output JSON")
    parser.add_argument("--out", default=None, help="defaults alongside --in, suffixed _scored.json")
    args = parser.parse_args()

    data = json.loads(Path(args.in_path).read_text())
    objects = data["objects"]
    answer_format = data.get("answer_format", "free_text")  # older collect files predate this key
    file_class_codes = data.get("class_codes") or CLASS_CODES
    predict = make_predictor(answer_format, GALAXY10_LABELS, GALAXY10_LABEL_SYNONYMS, file_class_codes)
    logger.info(
        f"Scoring {len(objects)} objects from {args.in_path} (sampling seed={data['sampling']['seed']}, "
        f"answer_format={answer_format!r}, shuffle_codes={data.get('shuffle_codes', False)})."
    )

    y_true = [o["label_name"] for o in objects]
    hard = {}
    group = {}
    for side in ("base", "equipped"):
        y_pred = _predictions_for(objects, answer_format, side, predict)
        side_report = classification_report(y_true, y_pred, GALAXY10_LABELS)
        n_unparsed = sum(1 for p in y_pred if p is None)
        side_report["parsing"] = {
            "n_unparsed": n_unparsed,
            "parse_rate": (len(y_pred) - n_unparsed) / len(y_pred) if y_pred else 0.0,
        }
        if answer_format == "logprob_argmax":
            side_report["margin"] = _margin_block(objects, side)
        hard[side] = side_report
        group[side] = group_report(y_true, y_pred)

    report = {
        "source": args.in_path,
        "dataset": data["dataset"],
        "repo_id": data["repo_id"],
        "answer_format": answer_format,
        "class_codes": file_class_codes if answer_format == "digit_code" else None,
        "shuffle_codes": data.get("shuffle_codes", False),
        "sampling": data["sampling"],
        "hard": hard,
        "group": group,
    }

    out_path = Path(args.out) if args.out else Path(args.in_path).with_name(Path(args.in_path).stem + "_scored.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    logger.info(json.dumps(report, indent=2))
    logger.info(f"Wrote scored report to {out_path}")


if __name__ == "__main__":
    main()
