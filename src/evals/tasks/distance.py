from __future__ import annotations

import re
import string
from collections import defaultdict
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from ..prompts import render_prompt
from .base import Task, parse_single_choice


class DistanceTask(Task):
    name: str = "distance"
    benchmark_name: str = "redshift"

    def __init__(
        self,
        benchmark_df: Optional[pd.DataFrame] = None,
        labels: Optional[List[str]] = None,
        **kwargs: Any,
    ):
        if labels is not None:
            self._set_ordered_labels(labels)
        elif benchmark_df is not None:
            self.init_from_dataframe(benchmark_df)
        else:
            from ..data import load_benchmark_dataset

            self.init_from_dataframe(load_benchmark_dataset(self.benchmark_name))

    def init_from_dataframe(self, df: pd.DataFrame) -> None:
        grouped: Dict[str, List[float]] = defaultdict(list)
        for _, row in df.iterrows():
            gt = row.get("ground_truth", row)
            lbl = None
            if isinstance(gt, dict):
                lbl = gt.get("redshift_bin", gt.get("label", gt.get("bin")))
            if lbl is None:
                lbl = row.get("redshift_bin", row.get("label", row.get("bin")))

            z_val = None
            if isinstance(gt, dict) and "z" in gt and gt["z"] is not None:
                z_val = float(gt["z"])
            elif "z" in row and row["z"] is not None:
                z_val = float(row["z"])

            if lbl is not None and z_val is not None:
                grouped[str(lbl)].append(z_val)

        if not grouped:
            raise ValueError("Could not extract any (label, z) pairs from dataframe.")

        sorted_labels = sorted(grouped.keys(), key=lambda l: float(np.mean(grouped[l])))
        self._set_ordered_labels(sorted_labels)

    def _set_ordered_labels(self, sorted_labels: List[str]) -> None:
        self.ordered_labels = list(sorted_labels)
        self.label_to_letter = {
            lbl: string.ascii_uppercase[i] for i, lbl in enumerate(self.ordered_labels)
        }
        self.letter_to_label = {
            letter: lbl for lbl, letter in self.label_to_letter.items()
        }
        self.letters = [self.label_to_letter[lbl] for lbl in self.ordered_labels]
        self._options_text = "\n".join(
            f"{self.label_to_letter[lbl]}: {lbl}" for lbl in self.ordered_labels
        )

    def build_frontier_prompt(
        self, caption: str, item: Optional[Dict[str, Any]] = None
    ) -> str:
        return render_prompt(
            "caption_eval/distance.jinja2",
            options=self._options_text,
            caption=caption.strip(),
        )

    def fallback_tag(self) -> str:
        return "\n\nFINAL ANSWER: "

    def default_parse(self, raw_text: str, **kwargs: Any) -> Optional[str]:
        letter = parse_single_choice(
            raw_text, allowed_letters=set(self.letters)
        )
        if letter:
            return letter
        if not raw_text:
            return None
        for lbl, let in self.label_to_letter.items():
            if re.search(r"\b" + re.escape(lbl) + r"\b", raw_text, re.IGNORECASE):
                return let
        return None

    def get_parse_fn(self, item: Optional[Dict[str, Any]] = None) -> Any:
        return self.default_parse

    def extract_ground_truth(self, item: Any) -> str:
        if isinstance(item, str):
            if item in self.label_to_letter:
                return self.label_to_letter[item]
            if item in self.letter_to_label:
                return item

        if isinstance(item, (dict, pd.Series)):
            gt = item.get("ground_truth", item)
            lbl = None
            if isinstance(gt, dict):
                lbl = gt.get("redshift_bin", gt.get("label", gt.get("bin")))
            if lbl is None:
                lbl = item.get("redshift_bin", item.get("label", item.get("bin")))

            if lbl is not None and str(lbl) in self.label_to_letter:
                return self.label_to_letter[str(lbl)]

        raise ValueError(
            f"Cannot extract ground truth redshift label from: {item}. Available labels: {list(self.label_to_letter.keys())}"
        )

    def get_config(self) -> Dict[str, Any]:
        return {
            "task_name": self.name,
            "benchmark_name": self.benchmark_name,
            "label_to_letter": self.label_to_letter,
            "ordered_labels": self.ordered_labels,
        }
