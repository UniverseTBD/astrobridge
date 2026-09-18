"""Shared AION model/codec loading. Both `image` and `spectra` modalities in this build point at
the same `polymathic-ai/aion-base` checkpoint (one multimodal model, two input codecs) — loading
it twice would double GPU memory for no reason, so instances are cached by (hf_path, revision).
"""
from __future__ import annotations

_CACHE: dict[tuple[str, str | None], tuple] = {}


def load_aion(hf_path: str, revision: str | None, device: str):
    key = (hf_path, revision)
    if key in _CACHE:
        return _CACHE[key]

    from aion import AION
    from aion.codecs import CodecManager

    model = AION.from_pretrained(hf_path, revision=revision) if revision else AION.from_pretrained(hf_path)
    model = model.to(device).eval().requires_grad_(False)
    codec_manager = CodecManager(device=device)

    _CACHE[key] = (model, codec_manager)
    return model, codec_manager
