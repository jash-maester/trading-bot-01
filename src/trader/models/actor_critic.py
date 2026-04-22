"""ActorCritic model: TCN encoder → mean-pool → actor/critic heads (no GNN for M5)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torch.distributions import Normal

from trader.models.encoders import TCNEncoder
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
    log_std_init: float = -1.0


class ActorCritic(nn.Module):
    """M5 no-GNN actor-critic: TCN → mean-pool → heads.

    The model consumes the obs dict from PanelTradingEnv and returns
    action distributions and value estimates.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = TCNEncoder(
            in_features=cfg.in_features,
            embed_dim=cfg.embed_dim,
            num_channels=cfg.num_channels,
            kernel_size=cfg.kernel_size,
            dropout=cfg.dropout,
        )
        d = self.encoder.out_dim
        self.actor = ActorHead(d, hidden=cfg.head_hidden)
        self.critic = CriticHead(d, hidden=cfg.head_hidden)
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
        features = obs["features"].float()    # [B, N, L, F]
        portfolio = obs["portfolio"].float()  # [B, N+1]
        t_frac = obs["t_frac"].float()        # [B] or scalar

        if t_frac.dim() == 0:
            t_frac = t_frac.unsqueeze(0)

        z = self.encoder(features)                         # [B, N, d]
        action_mean = self.actor(z, portfolio)             # [B, N+1]
        value = self.critic(z, portfolio, t_frac)          # [B]
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
        return cls(cfg)
