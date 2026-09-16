from __future__ import annotations

import re
import string
from typing import Any, Callable, Dict, List, Optional
import pandas as pd

from ..prompts import render_prompt
from .base import Task, parse_multi_choice

CANONICAL_LINES: List[str] = [
    "HALPHA",
    "HBETA",
    "HGAMMA",
    "OIII_5007",
    "OIII_4959",
    "OII_3726",
    "OII_3729",
    "NII_6584",
    "NII_6548",
    "SII_6716",
    "SII_6731",
    "MGII_2796",
    "MGII_2803",
    "CIV_1549",
]


class EmissionLineTask(Task):
    name: str = "emission_lines"
    benchmark_name: str = "emission_lines"

    def __init__(self, **kwargs: Any):
        self.canonical_lines = list(CANONICAL_LINES)

    def _get_candidate_lines(self, item: Optional[Dict[str, Any]] = None) -> List[str]:
        if item is not None and "candidate_query_lines" in item:
            cand = item["candidate_query_lines"]
            if cand is not None and len(cand) > 0:
                return [str(k) for k in cand]
        return list(self.canonical_lines)

    def build_frontier_prompt(
        self, caption: str, item: Optional[Dict[str, Any]] = None
    ) -> str:
        candidate_keys = self._get_candidate_lines(item)
        items_text = "\n".join(
            f"{string.ascii_uppercase[i]}: {k}" for i, k in enumerate(candidate_keys)
        )
        return render_prompt(
            "caption_eval/emission_lines.jinja2",
            candidate_lines=items_text,
            caption=caption.strip(),
        )

    def fallback_tag(self) -> str:
        return "\n\nFINAL ANSWER: "

    def get_parse_fn(
        self, item: Optional[Dict[str, Any]] = None
    ) -> Callable[[str], Optional[List[str]]]:
        candidate_lines = self._get_candidate_lines(item)
        letter_to_line = {
            string.ascii_uppercase[i]: line for i, line in enumerate(candidate_lines)
        }
        allowed_letters = set(letter_to_line.keys())

        def parse(raw_text: str) -> Optional[List[str]]:
            chosen_letters = parse_multi_choice(
                raw_text, allowed_letters=allowed_letters
            )
            if chosen_letters is None:
                return None
            return [letter_to_line[l] for l in chosen_letters if l in letter_to_line]

        return parse

    def default_parse(
        self, raw_text: str, item: Optional[Dict[str, Any]] = None, **kwargs: Any
    ) -> Optional[List[str]]:
        parse_fn = self.get_parse_fn(item)
        return parse_fn(raw_text)

    def extract_ground_truth(self, item: Any) -> Dict[str, float]:
        if item is None:
            return {}

        if isinstance(item, (dict, pd.Series)):
            gt = item.get("ground_truth", item)
            if isinstance(gt, dict):
                det_lines = gt.get("detected_lines")
                if det_lines is not None:
                    details = gt.get("line_details", {})
                    gt_dict: Dict[str, float] = {}
                    for line in det_lines:
                        line_str = str(line)
                        snr = 1.0
                        if isinstance(details, dict) and line_str in details:
                            d = details[line_str]
                            if isinstance(d, dict) and d.get("snr") is not None:
                                snr = float(d["snr"])
                        gt_dict[line_str] = snr
                    return gt_dict
                return {
                    str(k): float(v)
                    for k, v in gt.items()
                    if isinstance(v, (int, float))
                }
            if isinstance(gt, (list, set)):
                return {str(l): 1.0 for l in gt}

        if isinstance(item, (list, set)):
            return {str(l): 1.0 for l in item}

        return {}

    def get_config(self) -> Dict[str, Any]:
        return {
            "task_name": self.name,
            "benchmark_name": self.benchmark_name,
            "num_canonical_lines": len(self.canonical_lines),
            "canonical_lines": self.canonical_lines,
        }
