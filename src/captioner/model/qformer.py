from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class QFormerLayer(nn.Module):
    """self-attn(queries) -> cross-attn(queries -> tokens) -> FFN, each with a residual + norm."""

    def __init__(self, d_model: int, n_heads: int, ffn_mult: int, dropout: float) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.self_norm = nn.LayerNorm(d_model)

        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.cross_norm = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ffn_mult, d_model),
        )
        self.ffn_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries: Tensor, tokens: Tensor, key_padding_mask: Tensor | None) -> Tensor:
        q = queries
        attn_out, _ = self.self_attn(q, q, q, need_weights=False)
        q = self.self_norm(q + self.dropout(attn_out))

        attn_out, _ = self.cross_attn(q, tokens, tokens, key_padding_mask=key_padding_mask, need_weights=False)
        q = self.cross_norm(q + self.dropout(attn_out))

        q = self.ffn_norm(q + self.dropout(self.ffn(q)))
        return q


class SharedQFormer(nn.Module):
    """(B, T_total, d_model) + key_padding_mask -> (B, n_queries, d_model).

    key_padding_mask follows torch.nn.MultiheadAttention convention: True = ignore (padding).
    An absent modality must never appear here as zero-padded rows with mask=True over real
    positions — it must contribute zero rows to T_total in the first place (assembled upstream
    in collate.py). This module only respects whatever mask it is given.
    """

    def __init__(self, n_queries: int, d_model: int, n_layers: int, n_heads: int, ffn_mult: int, dropout: float) -> None:
        super().__init__()
        self.n_queries = n_queries
        self.query_embed = nn.Parameter(torch.empty(n_queries, d_model))
        nn.init.normal_(self.query_embed, mean=0.0, std=0.02)
        self.layers = nn.ModuleList(
            [QFormerLayer(d_model, n_heads, ffn_mult, dropout) for _ in range(n_layers)]
        )

    def forward(self, tokens: Tensor, key_padding_mask: Tensor | None = None) -> Tensor:
        B = tokens.shape[0]
        queries = self.query_embed.unsqueeze(0).expand(B, -1, -1).contiguous()
        for layer in self.layers:
            queries = layer(queries, tokens, key_padding_mask)
        return queries
