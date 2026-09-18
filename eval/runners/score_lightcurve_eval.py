#!/usr/bin/env python
"""Step 2/2 of the lightcurve eval: score a `collect_lightcurve_labels.py` output file. Pure
post-processing — no GPU, no Modal, no model, no network — mirrors `score_image_eval.py`, minus
the group/debiased-vote-fraction metrics (Galaxy-Zoo-specific, no SN-typing analog): hard-label
accuracy + per-class precision/recall/F1, both sides, branched on `answer_format`.

Usage:
    uv run python -m eval.runners.score_lightcurve_eval --in outputs/eval/raw_generations/yse_lightcurve_only_logprob_argmax_seed0_bal15.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from captioner.utils.logging import get_logger
from eval.datasets.lightcurve_yse import SN_CLASS_CODES, SN_LABELS
from eval.metrics.caption_to_label import SN_TYPE_SYNONYMS, make_predictor
from eval.metrics.classification import classification_report

logger = get_logger(__name__)


def _hard_report_from_predictions(y_true: list[str], y_pred: list[str | None]) -> dict:
    report = classification_report(y_true, y_pred, SN_LABELS)
    # Parse rate is a first-class number, not a footnote: unparsed answers are already counted as
    # wrong in the metrics above (classification_report treats None as incorrect) — this reports
    # them separately as well, never instead. logprob_argmax's predicted_label is never None (the
    # argmax is always one of the fixed candidates by construction), so its parse_rate is always
    # exactly 1.0 — reported for both formats so the two are directly comparable on this axis too.
    n_unparsed = sum(1 for p in y_pred if p is None)
    report["parsing"] = {
        "n_unparsed": n_unparsed,
        "parse_rate": (len(y_pred) - n_unparsed) / len(y_pred) if y_pred else 0.0,
    }
    return report


def _score_logprob_argmax(objects: list[dict]) -> dict:
    y_true = [o["label_name"] for o in objects]
    report = {}
    for side in ("base", "equipped"):
        y_pred = [o[side]["predicted_label"] for o in objects]
        side_report = _hard_report_from_predictions(y_true, y_pred)

        # Calibration: the gap between the winning and runner-up candidate log-prob. A narrow
        # margin marks a genuinely ambiguous object; a wide margin on a WRONG prediction marks a
        # confidently-wrong one — worth telling apart before concluding anything from accuracy
        # alone.
        margins = []
        for o in objects:
            scores = sorted(o[side]["logprobs"].values(), reverse=True)
            margins.append(scores[0] - scores[1])
        side_report["margin"] = {
            "mean": sum(margins) / len(margins) if margins else 0.0,
            "n_below_0.5": sum(1 for m in margins if m < 0.5),
        }
        report[side] = side_report
    return report


def _score_digit_code(objects: list[dict], file_class_codes: dict) -> dict:
    predict = make_predictor("digit_code", SN_LABELS, SN_TYPE_SYNONYMS, file_class_codes)
    y_true = [o["label_name"] for o in objects]
    return {
        "base": _hard_report_from_predictions(y_true, [predict(o["base_answer"]) for o in objects]),
        "equipped": _hard_report_from_predictions(y_true, [predict(o["equipped_answer"]) for o in objects]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", required=True, help="a collect_lightcurve_labels.py output JSON")
    parser.add_argument("--out", default=None, help="defaults alongside --in, suffixed _scored.json")
    args = parser.parse_args()

    data = json.loads(Path(args.in_path).read_text())
    objects = data["objects"]
    # "verbose_class"/"free_text" are older formats this file no longer collects but may still
    # need to score against a pre-existing output file.
    answer_format = data.get("answer_format", "logprob_argmax")
    logger.info(
        f"Scoring {len(objects)} objects from {args.in_path} (track={data.get('track')}, "
        f"sampling={data.get('sampling')}, answer_format={answer_format!r})."
    )

    if answer_format == "logprob_argmax":
        hard = _score_logprob_argmax(objects)
    else:
        file_class_codes = data.get("class_codes") or SN_CLASS_CODES
        hard = _score_digit_code(objects, file_class_codes)

    report = {
        "source": args.in_path,
        "dataset": data["dataset"],
        "track": data.get("track"),
        "repo_id": data["repo_id"],
        "answer_format": answer_format,
        "sampling": data["sampling"],
        "hard": hard,
    }

    out_path = Path(args.out) if args.out else Path(args.in_path).with_name(Path(args.in_path).stem + "_scored.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    logger.info(json.dumps(report, indent=2))
    logger.info(f"Wrote scored report to {out_path}")


if __name__ == "__main__":
    main()
