"""Encoder contract. Every modality encoder must satisfy this — token-level output, no pooling.

Mean-pooling is wrong here: the Q-Former exists to cross-attend over structure that pooling
destroys (05_build_spec_image_spectra.md §5).
"""
from __future__ import annotations

from typing import Any, Protocol

from torch import Tensor


class ModalityEncoder(Protocol):
    name: str
    out_dim: int

    def encode(self, batch: Any) -> Tensor:
        """Returns (B, T, out_dim) token-level embeddings. T may vary by batch."""
        ...


class EncoderLoadError(RuntimeError):
    pass


def assert_out_dim(declared: int, actual: int, impl: str, hf_path: str) -> None:
    """out_dim is asserted, not trusted — silent dimension drift after a checkpoint swap is
    otherwise very hard to spot."""
    if declared != actual:
        raise EncoderLoadError(
            f"Declared out_dim={declared} for encoder impl={impl!r} hf_path={hf_path!r} "
            f"does not match the real output dimension {actual}. Fix configs/modalities.yaml."
        )
