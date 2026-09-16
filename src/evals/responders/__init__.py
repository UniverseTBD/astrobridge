from __future__ import annotations

from typing import Any, Dict

from .sample import SpectrumSample
from .mock import MockCaptionResponder
from .base import (
    BaseResponder,
    GeneratedCaption,
    DEFAULT_SPECTRUM_CAPTION_PROMPT,
    DEFAULT_SYSTEM_CAPTION_PROMPT,
    resolve_caption_prompt,
    resolve_system_prompt,
)


def get_responder(config: Dict[str, Any], device: str) -> BaseResponder:
    responder_type = config.get("responder_type")
    if responder_type == "mock":
        return MockCaptionResponder(config, device)
    elif responder_type == "astrobridge":
        from .astrobridge import AstroBridgeResponder
        return AstroBridgeResponder(config, device)
    elif responder_type == "hf_vision":
        from .hf_vision import HFVisionResponder
        return HFVisionResponder(config, device)
    elif responder_type == "hf_text":
        from .hf_text import HFTextResponder
        return HFTextResponder(config, device)
    else:
        raise ValueError(
            f"Unknown responder '{responder_type}'. Available: ['astrobridge', 'hf_vision', 'hf_text', 'mock']"
        )


__all__ = [
    "SpectrumSample",
    "BaseResponder",
    "GeneratedCaption",
    "MockCaptionResponder",
    "DEFAULT_SPECTRUM_CAPTION_PROMPT",
    "DEFAULT_SYSTEM_CAPTION_PROMPT",
    "resolve_caption_prompt",
    "resolve_system_prompt",
    "get_responder",
]
