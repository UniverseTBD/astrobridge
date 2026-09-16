from __future__ import annotations

import re
import string
from typing import Any, Dict, Optional
import pandas as pd

from ..prompts import render_prompt
from .base import Task, parse_single_choice


class SourceTask(Task):
    name: str = "source"
    benchmark_name: str = "source_class"

    def __init__(
        self,
        active_classes: Optional[Dict[str, str]] = None,
        name: str = "source",
        **kwargs: Any,
    ):
        self.name = name
        self.active_classes = active_classes or {
            "GALAXY": "Galaxy",
            "QSO": "Quasar",
            "QUASAR": "Quasar",
        }
        self.categories = list(dict.fromkeys(self.active_classes.values()))
        self.letter_to_label = {
            string.ascii_uppercase[i]: cat for i, cat in enumerate(self.categories)
        }
        self.label_to_letter = {
            cat: letter for letter, cat in self.letter_to_label.items()
        }
        self._options_text = "\n".join(
            f"{letter}: {cat}" for letter, cat in self.letter_to_label.items()
        )

    def build_frontier_prompt(
        self, caption: str, item: Optional[Dict[str, Any]] = None
    ) -> str:
        return render_prompt(
            "caption_eval/source_class.jinja2",
            options=self._options_text,
            caption=caption.strip(),
        )

    def fallback_tag(self) -> str:
        return "\n\nFINAL ANSWER: "

    def default_parse(self, raw_text: str, **kwargs: Any) -> Optional[str]:
        letter = parse_single_choice(
            raw_text, allowed_letters=set(self.letter_to_label.keys())
        )
        if letter and letter in self.letter_to_label:
            return self.letter_to_label[letter]
        if not raw_text:
            return None
        for cat in self.categories:
            if re.search(r"\b" + re.escape(cat) + r"\b", raw_text, re.IGNORECASE):
                return cat
        for key, cat in self.active_classes.items():
            if re.search(r"\b" + re.escape(key) + r"\b", raw_text, re.IGNORECASE):
                return cat
        return None

    def get_parse_fn(self, item: Optional[Dict[str, Any]] = None) -> Any:
        return self.default_parse

    def extract_ground_truth(self, item: Any) -> str:
        if isinstance(item, str):
            raw_class = item
        elif isinstance(item, (dict, pd.Series)):
            gt = item.get("ground_truth", item)
            raw_class = "UNKNOWN"
            candidates = ("source_class", "class", "source_type")
            if isinstance(gt, dict):
                for c in candidates:
                    if c in gt and gt[c]:
                        raw_class = gt[c]
                        break
            if raw_class == "UNKNOWN":
                for c in candidates:
                    if c in item and item[c]:
                        raw_class = item[c]
                        break
        else:
            raise ValueError(
                f"Cannot extract ground truth from item of type {type(item)}"
            )

        if raw_class in self.active_classes:
            return self.active_classes[raw_class]
        for k, v in self.active_classes.items():
            if k.lower() == str(raw_class).lower():
                return v
        for v in self.categories:
            if v.lower() == str(raw_class).lower():
                return v
        return "UNKNOWN"

    def get_config(self) -> Dict[str, Any]:
        return {
            "task_name": self.name,
            "benchmark_name": self.benchmark_name,
            "active_classes": self.active_classes,
            "categories": self.categories,
            "letter_to_label": self.letter_to_label,
            "label_to_letter": self.label_to_letter,
        }
