from __future__ import annotations

import torch.nn as nn
from torch import Tensor


class ModalityProjector(nn.Module):
    """(B, T, out_dim) -> (B, T, d_shared). 2-layer MLP, GELU, LayerNorm out, dropout."""

    def __init__(self, in_dim: int, d_shared: int, hidden_mult: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        hidden = in_dim * hidden_mult
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_shared),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(d_shared)

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(self.net(x))
