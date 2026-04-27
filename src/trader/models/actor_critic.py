"""ActorCritic model: TCN encoder → mean-pool → actor/critic heads (no GNN for M5)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torch.distributions import Normal

from trader.models.encoders import CrossStockAttention, TCNEncoder
from trader.models.heads import ActorHead, CriticHead


@dataclass
class ModelConfig:
    in_features: int
    n_tickers: int
    embed_dim: int = 64
    num_channels: list[int] | None = None   # TCN channels per layer
    kernel_size: int = 3
    dropout: float = 0.1
    head_hidden: int = 64
    # log_std_init = -2.0 → σ=0.135 at start (was -1.0 → σ=0.37 in r6/r7).
    # With 164 action dims, smaller σ means each *sample* is closer to the
    # policy's mean, which is what we actually want to evaluate on. The
    # entropy bonus + ent_coef will still let the policy widen σ if there
    # is real exploration value to it.
    log_std_init: float = -2.0
    num_sectors: int = 8                     # used by the critic for sector exposure
    # Cross-stock self-attention layer between encoder and heads.
    # Provides the inductive bias needed for portfolio ranking — see
    # encoders.CrossStockAttention.  Default ON for r9+; flip to False
    # via Hydra override for ablation runs.
    use_cross_attn: bool = True
    cross_attn_heads: int = 4


class ActorCritic(nn.Module):
    """M5 no-GNN actor-critic: TCN → mean-pool → heads.

    The model consumes the obs dict from PanelTradingEnv and returns
    action distributions and value estimates.
    """

    def __init__(
        self,
        cfg: ModelConfig,
        feat_mean: torch.Tensor | None = None,
        feat_std: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = TCNEncoder(
            in_features=cfg.in_features,
            embed_dim=cfg.embed_dim,
            num_channels=cfg.num_channels,
            kernel_size=cfg.kernel_size,
            dropout=cfg.dropout,
            feat_mean=feat_mean,
            feat_std=feat_std,
        )
        d = self.encoder.out_dim
        # Optional cross-stock attention BEFORE the heads (see ModelConfig).
        # Lets each stock's representation see the entire universe.
        self.cross_attn: CrossStockAttention | None = (
            CrossStockAttention(
                embed_dim=d,
                num_heads=cfg.cross_attn_heads,
                dropout=cfg.dropout,
            )
            if cfg.use_cross_attn
            else None
        )
        self.actor = ActorHead(d, hidden=cfg.head_hidden)
        self.critic = CriticHead(d, num_sectors=cfg.num_sectors, hidden=cfg.head_hidden)
        # Learnable log-std per action dimension (N+1)
        self.log_std = nn.Parameter(
            torch.full((cfg.n_tickers + 1,), cfg.log_std_init)
        )

    def forward(
        self,
        obs: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Returns
        -------
        action_mean : (B, N+1)
        value       : (B,)
        log_std     : (N+1,)  — for constructing the Normal distribution
        """
        features = obs["features"].float()              # [B, L, N, F]  env convention
        features = features.permute(0, 2, 1, 3)         # → [B, N, L, F] for TCN
        portfolio = obs["portfolio"].float()             # [B, N+1]
        t_frac = obs["t_frac"].float()                   # [B] or scalar
        sector_ids = obs["sector_ids"].long()            # [B, N]

        if t_frac.dim() == 0:
            t_frac = t_frac.unsqueeze(0)

        # Critic-only state summary (with backward-compatible defaults so any
        # callers using older obs dicts still work — though training always
        # supplies the keys).
        zero_b = torch.zeros_like(t_frac)
        recent_return = obs.get("recent_return_1d", zero_b).float()
        recent_vol = obs.get("recent_vol_20d", zero_b).float()
        nav_log_progress = obs.get("nav_log_progress", zero_b).float()
        if recent_return.dim() == 0:
            recent_return = recent_return.unsqueeze(0)
        if recent_vol.dim() == 0:
            recent_vol = recent_vol.unsqueeze(0)
        if nav_log_progress.dim() == 0:
            nav_log_progress = nav_log_progress.unsqueeze(0)

        z = self.encoder(features)                         # [B, N, d]
        if self.cross_attn is not None:
            mask_raw = obs.get("mask")
            tradeable: torch.Tensor | None = (
                mask_raw.bool() if mask_raw is not None else None
            )
            z = self.cross_attn(z, tradeable_mask=tradeable)  # [B, N, d]
        action_mean = self.actor(z, portfolio)             # [B, N+1]
        value = self.critic(
            z, portfolio, t_frac, sector_ids,
            recent_return, recent_vol, nav_log_progress,
        )                                                  # [B]
        return action_mean, value, self.log_std

    def get_action_and_value(
        self,
        obs: dict[str, torch.Tensor],
        action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample or evaluate an action; return (action, log_prob, entropy, value)."""
        mean, value, log_std = self.forward(obs)
        std = log_std.exp().expand_as(mean)
        dist = Normal(mean, std)

        if action is None:
            action = dist.sample()  # type: ignore[no-untyped-call]

        log_prob = dist.log_prob(action).sum(-1)   # type: ignore[no-untyped-call]  # [B]
        entropy = dist.entropy().sum(-1)           # type: ignore[no-untyped-call]  # [B]
        return action, log_prob, entropy, value

    def get_value(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Return value estimate only (used for bootstrap at rollout end)."""
        _, value, _ = self.forward(obs)
        return value

    @classmethod
    def from_obs_space(
        cls,
        obs_space: Any,
        cfg_overrides: dict[str, Any] | None = None,
        feat_mean: torch.Tensor | None = None,
        feat_std: torch.Tensor | None = None,
    ) -> ActorCritic:
        """Construct from a gymnasium observation space."""
        feat_shape = obs_space["features"].shape   # (lookback, N, F)
        n_tickers = feat_shape[1]
        in_features = feat_shape[2]
        overrides = cfg_overrides or {}
        cfg = ModelConfig(
            in_features=in_features,
            n_tickers=n_tickers,
            **overrides,
        )
        return cls(cfg, feat_mean=feat_mean, feat_std=feat_std)
