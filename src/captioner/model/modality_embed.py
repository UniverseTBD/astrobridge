from __future__ import annotations

import torch.nn as nn
from torch import Tensor


class ModalityIdentity(nn.Module):
    """Learned (n_modalities, d_shared) embedding, ADDED to every token of that modality.

    Indexed by position in `modality_names`, which is read from configs/modalities.yaml at
    construction time — never a hardcoded modality count or name (§0).
    """

    def __init__(self, modality_names: list[str], d_shared: int) -> None:
        super().__init__()
        self.modality_names = list(modality_names)
        self.index = {name: i for i, name in enumerate(self.modality_names)}
        self.embed = nn.Embedding(len(self.modality_names), d_shared)
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)

    def forward(self, x: Tensor, modality: str) -> Tensor:
        """x: (B, T, d_shared) tokens of a single modality -> same shape, identity added."""
        idx = self.index[modality]
        vec = self.embed.weight[idx].view(1, 1, -1)
        return x + vec
