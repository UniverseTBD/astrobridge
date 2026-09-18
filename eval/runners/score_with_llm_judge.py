#!/usr/bin/env python
"""Score a collect-style output file (base/equipped captions vs. ground truth) using Gemini as an
automated judge, instead of `eval/metrics/caption_to_label.py`'s regex/keyword extractor. Pure
API calls, no GPU/Modal — this is a text-comprehension task, not something that needs the trained
model or a rented GPU.

Why this exists: a live manual read of 30 real Galaxy10 captions (see `eval/metrics/llm_judge.py`'s
module docstring) found the regex extractor badly undercounts real signal — e.g. equipped's own
captions genuinely describe the right morphology in many cases, just not in the exact words the
synonym list expects. This asks a small, cheap LLM (`gemini-2.5-flash` by default — reading a
paragraph and picking one of ~10 options needs no frontier model) to read each caption the way a
human would and pick the best-matching class, using Gemini's structured-output mode so the answer
is always a valid class index.

Auto-detects the input file's shape — works with any collect script's output that has ground-truth
`label_name` and one or more caption/reasoning fields per object:
  - `collect_lightcurve_labels.py`/`collect_image_labels.py --answer-format logprob_argmax`:
    `{"base": {"reasoning": ...}, "equipped": {"reasoning": ...}}`
  - `collect_image_labels.py --answer-format digit_code`/`verbose_class`:
    `"base_answer"`/`"equipped_answer"` strings
  - `collect_image_captions.py`: equipped-only `"caption"` string

Usage:
    uv run python -m eval.runners.score_with_llm_judge --in outputs/eval/raw_generations/gz10_images.json
    uv run python -m eval.runners.score_with_llm_judge --in outputs/eval/raw_generations/galaxy10_logprob_argmax_seed0_n30.json
    uv run python -m eval.runners.score_with_llm_judge --in outputs/eval/raw_generations/yse_lightcurve_only_logprob_argmax_seed0_bal15.json --label-vocabulary lightcurve
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tqdm import tqdm

from captioner.utils.logging import get_logger
from eval.datasets.image_galaxy10 import GALAXY10_LABELS
from eval.datasets.lightcurve_yse import SN_LABELS
from eval.metrics.classification import classification_report
from eval.metrics.llm_judge import DEFAULT_JUDGE_MODEL, judge_caption

logger = get_logger(__name__)

_LABEL_VOCABULARIES = {"galaxy10": GALAXY10_LABELS, "lightcurve": SN_LABELS}


def _extract_captions(obj: dict, single_side_label: str = "equipped") -> dict[str, str]:
    """`{side: caption_text}` for whichever sides this object actually has — see module docstring
    for the three known shapes this auto-detects. `single_side_label`: which side a bare
    `"caption"` field (collect_image_captions.py's shape) belongs to — that script's output is
    genuinely single-sided (one whole file is either base or equipped, per its `--side` flag), so
    the label must come from the file's own top-level `"side"` field, not be hardcoded, or a
    standalone base-captions file would get silently mislabeled as equipped.
    """
    captions: dict[str, str] = {}
    for side in ("base", "equipped"):
        if isinstance(obj.get(side), dict) and "reasoning" in obj[side]:
            captions[side] = obj[side]["reasoning"]
        elif isinstance(obj.get(f"{side}_answer"), str):
            captions[side] = obj[f"{side}_answer"]
    if not captions and isinstance(obj.get("caption"), str):
        captions[single_side_label] = obj["caption"]
    return captions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", required=True, help="a collect_*.py output JSON with base and/or equipped captions")
    parser.add_argument("--label-vocabulary", choices=list(_LABEL_VOCABULARIES), default="galaxy10")
    parser.add_argument("--model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument(
        "--workers", type=int, default=8,
        help="parallel Gemini calls — I/O-bound (network), not compute-bound, so threads are enough; raise if you aren't hitting rate limits, lower if you are",
    )
    parser.add_argument("--out", default=None, help="defaults alongside --in, suffixed _llm_judge_scored.json")
    args = parser.parse_args()

    labels = _LABEL_VOCABULARIES[args.label_vocabulary]
    data = json.loads(Path(args.in_path).read_text())
    objects = data["objects"]
    single_side_label = data.get("side", "equipped")
    logger.info(f"Judging {len(objects)} objects from {args.in_path} with model={args.model!r}, label_vocabulary={args.label_vocabulary!r}.")

    per_object_captions = [_extract_captions(o, single_side_label) for o in objects]
    sides = sorted({side for c in per_object_captions for side in c})
    if not sides:
        raise ValueError(f"No recognisable caption field found in {args.in_path} — see this module's docstring for the shapes it expects.")

    predictions: dict[str, list[str | None]] = {side: [None] * len(objects) for side in sides}
    tasks = [(i, side) for i in range(len(objects)) for side in sides if side in per_object_captions[i]]

    def _judge_one(i_side: tuple[int, str]) -> tuple[int, str, str | None]:
        i, side = i_side
        return i, side, judge_caption(per_object_captions[i][side], labels, model=args.model)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, side, pred in tqdm(pool.map(_judge_one, tasks), total=len(tasks), desc="llm judge"):
            predictions[side][i] = pred

    y_true = [o["label_name"] for o in objects]
    hard = {}
    for side in sides:
        y_pred = predictions[side]
        report = classification_report(y_true, y_pred, labels)
        n_unparsed = sum(1 for p in y_pred if p is None)
        report["parsing"] = {
            "n_unparsed": n_unparsed,
            "parse_rate": (len(y_pred) - n_unparsed) / len(y_pred) if y_pred else 0.0,
        }
        hard[side] = report

    result = {
        "source": args.in_path,
        "judge_model": args.model,
        "label_vocabulary": args.label_vocabulary,
        "hard": hard,
        "objects": [
            {
                "label_name": y_true[i],
                **{f"{side}_predicted_label": predictions[side][i] for side in sides if side in per_object_captions[i]},
            }
            for i in range(len(objects))
        ],
    }

    out_path = Path(args.out) if args.out else Path(args.in_path).with_name(Path(args.in_path).stem + "_llm_judge_scored.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    logger.info(json.dumps(hard, indent=2))
    logger.info(f"Wrote scored report to {out_path}")


if __name__ == "__main__":
    main()
