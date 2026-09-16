from __future__ import annotations

from typing import Any, Dict, List, Optional

from .base import (
    BaseResponder,
    GeneratedCaption,
    resolve_caption_prompt,
)
from .sample import SpectrumSample


class MockCaptionResponder(BaseResponder):
    def __init__(self, config: Optional[dict] = None, device: str = "cpu"):
        self.config = config or {}
        self.caption_prompt = resolve_caption_prompt(self.config, "caption_generation/default.jinja2")
        self.model_id = self.config.get("model_id", "mock-caption-model")

    def get_config(self) -> Dict[str, Any]:
        return {
            "type": "MockCaptionResponder",
            "model_id": self.model_id,
            "caption_prompt": self.caption_prompt,
        }

    def generate_captions(
        self,
        samples: List[SpectrumSample],
        prompt_override: Optional[str] = None,
    ) -> List[GeneratedCaption]:
        prompt = prompt_override or self.caption_prompt
        custom_caption = self.config.get("mock_caption")
        captions = []
        for s in samples:
            text = custom_caption or (
                f"Synthetic astronomical spectrum description for {s.sample_id}. "
                "The spectrum exhibits a moderately flat continuum with strong broad Hα and Hβ emission lines, "
                "consistent with an active galactic nucleus at redshift z ~ 0.15."
            )
            captions.append(
                GeneratedCaption(
                    sample_id=s.sample_id,
                    caption=text,
                    responder_type="mock",
                    model_id=self.model_id,
                    caption_prompt=prompt,
                    survey=s.survey,
                )
            )
        return captions
