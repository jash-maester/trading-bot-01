"""Actor and critic heads for the RL policy."""
from __future__ import annotations

import torch
import torch.nn as nn

from trader.models.encoders import RegimeNormalizer


def _mlp(in_dim: int, hidden: int, out_dim: int, layers: int = 2) -> nn.Sequential:
    parts: list[nn.Module] = []
    d = in_dim
    for _ in range(layers - 1):
        parts += [nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.GELU()]
        d = hidden
    parts.append(nn.Linear(d, out_dim))
    return nn.Sequential(*parts)


def _init_last_linear(seq: nn.Sequential, gain: float) -> None:
    """Apply orthogonal init with the given gain to the last Linear layer.

    Hidden Linear layers keep PyTorch's default init (Kaiming-uniform) which
    pairs well with GELU/LayerNorm. Only the *output* layer is rescaled:

    - Actor output: ``gain=0.01`` so initial logits are O(0.01).  After the
      Gaussian policy and masked softmax this gives a near-uniform
      allocation at step 0, exactly as recommended by the CleanRL/PPO
      conventions.  Big initial logit spread → big initial KL → harder to
      keep the policy on the same trust region.
    - Critic output: ``gain=1.0`` (standard).
    """
    for layer in reversed(seq):
        if isinstance(layer, nn.Linear):
            nn.init.orthogonal_(layer.weight, gain=gain)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
            return


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

        # Small-init last linear layer (CleanRL convention).
        _init_last_linear(self.equity_mlp, gain=0.01)
        _init_last_linear(self.cash_mlp, gain=0.01)

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


class ReturnPredictionHead(nn.Module):
    """Auxiliary self-supervised head: predict next-day per-stock log returns.

    Phase 2 of the architecture upgrade.  Adds a dense gradient signal to
    the encoder that PPO's policy/value losses do not provide:

    * **PPO value loss** sees only one scalar per batch element (portfolio
      value), so the critic gradient pressure on per-stock representations
      is weak (mediated by ``mean(z, dim=1)``).
    * **PPO policy loss** is sparse and noisy — the advantage signal is
      the *combined* effect of all stock decisions, not per-stock truth.
    * **This head** supervises every per-stock embedding directly with a
      regression target known one step into the future.  The encoder is
      pushed to encode features that are predictive of the cross-section,
      independent of how the policy chooses to act on them.

    The literature (UNREAL, Jaderberg+ 2017; auxiliary heads in financial
    RL — Theate & Ernst 2021) consistently finds this gives faster
    convergence and better representations even when the auxiliary loss
    weight is small (~0.1).

    Architecture: a small per-stock MLP applied identically to each
    embedding, output a scalar return.  No cross-stock mixing inside the
    head (that's already done upstream by CrossStockAttention).
    """

    def __init__(self, embed_dim: int, hidden: int = 32) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        # Small init on the output layer so initial predictions are O(0.001),
        # i.e. on the same order as actual daily log returns.  Without this
        # the head's initial MSE swamps PPO losses for the first 100s of
        # updates.
        _init_last_linear(self.mlp, gain=0.01)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Predict next-day log return for each stock.

        Parameters
        ----------
        z : Tensor [B, N, D]
            Per-stock embeddings (post encoder + cross-attn + FiLM).

        Returns
        -------
        Tensor [B, N]
            Predicted log returns.
        """
        return self.mlp(z).squeeze(-1)  # type: ignore[no-any-return]


def aux_return_loss(
    pred: torch.Tensor,           # [B, N]
    target: torch.Tensor,         # [B, N]
    tradeable_mask: torch.Tensor, # [B, N] bool / int8
) -> torch.Tensor:
    """Masked MSE between predicted and actual next-day log returns.

    Only tradeable stocks contribute.  If a batch element has *no*
    tradeable stocks (degenerate; shouldn't happen in practice) we
    fall back to a zero loss to avoid NaNs from divide-by-zero.
    """
    m = tradeable_mask.bool().to(pred.dtype)
    diff_sq = (pred - target).pow(2) * m
    denom = m.sum().clamp_min(1.0)
    return diff_sq.sum() / denom


class CriticHead(nn.Module):
    """Scalar value estimate from pooled embeddings and portfolio summary.

    Inputs:
        z (encoder output, [B, N, d]) — mean-pooled.
        portfolio ([B, N+1])         — current weights inc. cash.
        t_frac ([B])                 — episode progress.
        sector_ids ([B, N])          — used to compute per-sector exposure
                                        via scatter-add (8 dims).
        recent_return_1d ([B])       — last day's portfolio log return.
        recent_vol_20d   ([B])       — std of the last 20 daily log returns.
        nav_log_progress ([B])       — log(current_nav / initial_cash).
        regime ([B, R], optional)    — market regime vector, when
                                        ``regime_dim > 0``.  Concatenated
                                        directly so the value function can
                                        distinguish e.g. high-vol vs low-vol
                                        forward expectations independent of
                                        what the actor does.

    The richer summary lets the value function distinguish, e.g., a portfolio
    that is up 5 % at high vol from one that is up 5 % at low vol — both
    have the same `nav_log_progress` but very different forward expectations.
    Lower V variance ⇒ lower advantage variance ⇒ faster, more stable PPO.
    """

    def __init__(
        self,
        embed_dim: int,
        num_sectors: int = 8,
        hidden: int = 64,
        regime_dim: int = 0,
        regime_normalizer: RegimeNormalizer | None = None,
    ) -> None:
        super().__init__()
        self.num_sectors = num_sectors
        self.regime_dim = regime_dim
        # The critic owns its own normaliser handle (often shared with FiLM).
        # When `regime_dim==0` this attribute is None and `forward` ignores
        # the regime arg entirely.
        self.regime_normalizer: RegimeNormalizer | None = (
            regime_normalizer if regime_dim > 0 else None
        )
        # z_mean (d) || summary scalars (7) || sector exposure (S) || regime (R)
        in_dim = embed_dim + 7 + num_sectors + regime_dim
        self.mlp = _mlp(in_dim, hidden, 1)
        _init_last_linear(self.mlp, gain=1.0)

    def forward(
        self,
        z: torch.Tensor,           # [B, N, d]
        portfolio: torch.Tensor,   # [B, N+1]
        t_frac: torch.Tensor,      # [B]
        sector_ids: torch.Tensor,  # [B, N], 1-indexed (0 = unknown)
        recent_return_1d: torch.Tensor,   # [B]
        recent_vol_20d: torch.Tensor,     # [B]
        nav_log_progress: torch.Tensor,   # [B]
        regime: torch.Tensor | None = None,   # [B, R]
    ) -> torch.Tensor:
        B, _, _ = z.shape
        z_mean = z.mean(dim=1)                            # [B, d]
        eq_w = portfolio[:, 1:]                           # [B, N]

        # Per-sector exposure via scatter_add. sector_id == 0 means
        # unclassified — those weights are zeroed before scatter so they
        # don't pile into bucket 0 by accident.
        sec_idx = (sector_ids - 1).clamp(min=0).long()    # [B, N]
        valid = (sector_ids > 0).to(eq_w.dtype)           # [B, N]
        masked_w = eq_w * valid
        sector_exp = torch.zeros(
            B, self.num_sectors, device=z.device, dtype=eq_w.dtype
        )
        sector_exp.scatter_add_(1, sec_idx, masked_w)     # [B, S]

        summary = torch.stack(
            [
                eq_w.mean(dim=1),
                eq_w.max(dim=1).values,
                portfolio[:, 0],     # cash weight
                t_frac,
                recent_return_1d,
                recent_vol_20d,
                nav_log_progress,
            ],
            dim=1,
        )                                                 # [B, 7]

        parts = [z_mean, summary, sector_exp]
        if self.regime_dim > 0 and regime is not None:
            r = regime
            if r.dim() == 1:
                r = r.unsqueeze(0)
            if self.regime_normalizer is not None:
                r = self.regime_normalizer(r)
            parts.append(r)
        elif self.regime_dim > 0:
            # Defensive: no regime supplied but the head expects one — pad
            # with zeros so the linear receives the right shape.
            parts.append(
                torch.zeros(
                    B, self.regime_dim, device=z.device, dtype=z.dtype
                )
            )

        x = torch.cat(parts, dim=1)
        v = self.mlp(x)                                   # [B, 1]
        return v.squeeze(-1)  # type: ignore[no-any-return]  # [B]
