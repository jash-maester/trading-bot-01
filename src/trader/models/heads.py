"""Actor and critic heads for the RL policy."""
from __future__ import annotations

import torch
import torch.nn as nn


def _mlp(in_dim: int, hidden: int, out_dim: int, layers: int = 2) -> nn.Sequential:
    parts: list[nn.Module] = []
    d = in_dim
    for _ in range(layers - 1):
        parts += [nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.GELU()]
        d = hidden
    parts.append(nn.Linear(d, out_dim))
    return nn.Sequential(*parts)


class ActorHead(nn.Module):
    """Produces (N+1,) raw logits from per-ticker embeddings and portfolio.

    Per-ticker logit: MLP([z_i, portfolio_weight_i]) → scalar
    Cash logit:       MLP([mean_pool(z), cash_weight]) → scalar
    """

    def __init__(self, embed_dim: int, hidden: int = 64) -> None:
        super().__init__()
        # equity branch: z_i (d) || portfolio_i (1) → scalar
        self.equity_mlp = _mlp(embed_dim + 1, hidden, 1)
        # cash branch: mean_pool(z) (d) || cash_weight (1) → scalar
        self.cash_mlp = _mlp(embed_dim + 1, hidden, 1)
        # log-std for Gaussian policy (learnable, per-dim)
        self._log_std_init = -1.0

    def forward(
        self,
        z: torch.Tensor,           # [B, N, d]
        portfolio: torch.Tensor,   # [B, N+1]   index 0 = cash
    ) -> torch.Tensor:
        B, N, d = z.shape
        # Equity logits
        pw = portfolio[:, 1:].unsqueeze(-1)          # [B, N, 1]
        eq_in = torch.cat([z, pw], dim=-1)           # [B, N, d+1]
        eq_logits = self.equity_mlp(eq_in).squeeze(-1)  # [B, N]
        # Cash logit
        z_mean = z.mean(dim=1)                       # [B, d]
        cash_w = portfolio[:, :1]                    # [B, 1]
        cash_logit = self.cash_mlp(
            torch.cat([z_mean, cash_w], dim=-1)
        )                                            # [B, 1]
        return torch.cat([cash_logit, eq_logits], dim=1)  # [B, N+1]


class CriticHead(nn.Module):
    """Scalar value estimate from pooled embeddings and portfolio summary."""

    def __init__(self, embed_dim: int, hidden: int = 64) -> None:
        super().__init__()
        # Summary stats: mean_weight, max_weight, cash_weight, t_frac = 4 scalars
        self.mlp = _mlp(embed_dim + 4, hidden, 1)

    def forward(
        self,
        z: torch.Tensor,           # [B, N, d]
        portfolio: torch.Tensor,   # [B, N+1]
        t_frac: torch.Tensor,      # [B]
    ) -> torch.Tensor:
        z_mean = z.mean(dim=1)                            # [B, d]
        eq_w = portfolio[:, 1:]                           # [B, N]
        summary = torch.stack(
            [
                eq_w.mean(dim=1),
                eq_w.max(dim=1).values,
                portfolio[:, 0],   # cash weight
                t_frac,
            ],
            dim=1,
        )                                                 # [B, 4]
        v = self.mlp(torch.cat([z_mean, summary], dim=1))  # [B, 1]
        return v.squeeze(-1)  # type: ignore[no-any-return]  # [B]
