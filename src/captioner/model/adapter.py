from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor


class Adapter(nn.Module):
    """(B, n_queries, d_shared) -> (B, n_queries, d_llm). Maps into the LLM's embedding space —
    exactly the shape the embedding layer would have produced for n_queries real tokens.

    `target_norm` is what makes "the LLM's embedding space" literally true rather than merely
    the right shape, and it is not cosmetic. Measured on the shipped v6 checkpoint: prefix
    vectors came out at norm **30.6** against Qwen3.5-9B's own token-embedding norm of **0.89**
    — 34x out of distribution. The old adapter ended in a bare `nn.Linear` fed from a LayerNorm'd
    Q-Former (norm ~ sqrt(d_shared) ~ 19.6) and nothing ever pulled that down, because Qwen
    RMSNorms every position at each sublayer input: the magnitude is divided out before it can
    produce a gradient, while the residual stream still carries it, so the prefix positions
    behave as extreme outlier/sink tokens. The ratio sat at ~30 from stage-1 step 500 through the
    end of stage 2 — it never self-corrected, and it cannot be expected to.

    So the output is explicitly renormalised and rescaled to the LLM's own embedding scale. The
    scale stays learnable (in log space, so it can only ever be positive) because the right value
    is "about the same as a real token", not exactly the median — training can move it, it just
    starts somewhere sane instead of 34x away.

    Pass `target_norm` from the real model:
        llm.get_input_embeddings().weight.norm(dim=1).median()
    The 1.0 default is only for tests that have no LLM to measure.
    """

    def __init__(
        self, d_shared: int, d_llm: int, dropout: float = 0.1, target_norm: float = 1.0
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_shared, d_llm),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_llm, d_llm),
        )
        self.out_norm = nn.LayerNorm(d_llm)
        # out_norm gives unit variance per component, i.e. a vector of norm ~ sqrt(d_llm).
        # Scale that to target_norm.
        self.log_scale = nn.Parameter(
            torch.tensor(math.log(max(float(target_norm), 1e-6) / math.sqrt(d_llm)))
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.out_norm(self.net(x)) * self.log_scale.exp()
