"""AION image wrapper (impl=aion_image), against the real `aion` package API:

    img = LegacySurveyImage(flux=pixel_values, bands=BANDS)
    tokens = codec_manager.encode(img)
    emb = model.encode(tokens, num_encoder_tokens=N)   # (B, N, 768) — token-level

Confirmed against a working linear-probe script that mean-pooled `emb` over dim=1 for a
fixed-size probe input; this wrapper is that same call with the pooling step removed, since the
Q-Former needs the full token grid, not a pooled summary (§5).
"""
from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from captioner.encoders.aion_common import load_aion
from captioner.encoders.base import EncoderLoadError

AION_OUT_DIM = 768  # AION-base's shared embedding width — same for every modality's tokens


class AionImageEncoder:
    def __init__(
        self,
        name: str,
        hf_path: str,
        revision: str | None,
        kwargs: dict[str, Any],
        num_encoder_tokens: int,
    ) -> None:
        self.name = name
        self.hf_path = hf_path
        self.revision = revision
        self.bands: list[str] = list(kwargs.get("bands", []))
        self.num_encoder_tokens = num_encoder_tokens
        self.out_dim = AION_OUT_DIM
        self._model = None
        self._codec_manager = None
        self._device = "cpu"

    def load(self, device: str = "cpu") -> None:
        self._device = device
        self._model, self._codec_manager = load_aion(self.hf_path, self.revision, device)

    @torch.no_grad()
    def encode(self, batch: dict[str, Tensor]) -> Tensor:
        """batch['pixel_values']: (B, n_bands, H, W). Band order must match self.bands —
        channel order and label order travel together (§5).
        """
        if self._model is None:
            raise EncoderLoadError("encode() called before load()")

        from aion.modalities import LegacySurveyImage

        pixel_values = batch["pixel_values"].to(self._device)
        if pixel_values.shape[1] != len(self.bands):
            raise ValueError(
                f"pixel_values has {pixel_values.shape[1]} band channels but {len(self.bands)} "
                f"bands are declared in configs/modalities.yaml (kwargs.bands={self.bands})."
            )

        img = LegacySurveyImage(flux=pixel_values, bands=self.bands)
        tokens = self._codec_manager.encode(img)
        emb = self._model.encode(tokens, num_encoder_tokens=self.num_encoder_tokens)

        if emb.dim() != 3:
            raise EncoderLoadError(
                f"Expected token-level (B, T, out_dim) from AION.encode, got shape "
                f"{tuple(emb.shape)}. Do not mean-pool in this wrapper — see §5/§10."
            )
        return emb
