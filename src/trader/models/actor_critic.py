"""ActorCritic model: TCN encoder → (FiLM) → CrossStockAttention → (FiLM) → heads."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torch.distributions import Normal

from trader.models.encoders import CrossStockAttention, FiLM, RegimeNormalizer, TCNEncoder
from trader.models.heads import ActorHead, CriticHead, ReturnPredictionHead


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
    # ── Regime conditioning (Phase 1: targets the val/test corr=−0.86 problem) ─
    # Each flag is independently toggleable so we can ablate the locations
    # cleanly:
    #   regime_film_encoder=True  → FiLM after the TCN encoder
    #   regime_film_attn=True     → FiLM after CrossStockAttention
    #   regime_in_critic=True     → concatenate raw regime to critic input
    # When all three are False the model is bit-exact identical to pre-Phase-1.
    regime_dim: int = 0                       # set automatically by from_obs_space
    regime_film_encoder: bool = False
    regime_film_attn: bool = False
    regime_in_critic: bool = False
    regime_film_hidden: int = 32
    # ── Auxiliary supervised return-prediction head (Phase 2) ────────────────
    # When enabled, an extra small head predicts per-stock next-day log
    # returns from the same z used by actor/critic.  Trained jointly with
    # PPO via an MSE loss; weight controlled by `train.aux_return_loss_coef`.
    use_aux_return_head: bool = False
    aux_return_hidden: int = 32


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
        regime_mean: torch.Tensor | None = None,
        regime_std: torch.Tensor | None = None,
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

        # ── Regime FiLM conditioning ──────────────────────────────────────
        # A single shared `RegimeNormalizer` is used by all FiLM blocks and
        # by the critic-side regime concat — they all need the same z-score
        # transform.  When stats are absent the normaliser is identity.
        self._regime_norm = RegimeNormalizer(regime_mean, regime_std)
        def _make_film() -> FiLM:
            return FiLM(
                cond_dim=cfg.regime_dim,
                embed_dim=d,
                hidden=cfg.regime_film_hidden,
                normalizer=self._regime_norm,
            )
        # Each FiLM block is independently toggleable for ablation.
        # When regime_dim==0 (no regime in obs), FiLM is silently disabled.
        regime_active = cfg.regime_dim > 0
        self.film_encoder: FiLM | None = (
            _make_film()
            if regime_active and cfg.regime_film_encoder
            else None
        )
        self.film_attn: FiLM | None = (
            _make_film()
            if regime_active and cfg.regime_film_attn and cfg.use_cross_attn
            else None
        )
        critic_regime_dim = (
            cfg.regime_dim if regime_active and cfg.regime_in_critic else 0
        )

        self.actor = ActorHead(d, hidden=cfg.head_hidden)
        self.critic = CriticHead(
            d,
            num_sectors=cfg.num_sectors,
            hidden=cfg.head_hidden,
            regime_dim=critic_regime_dim,
            regime_normalizer=self._regime_norm if critic_regime_dim > 0 else None,
        )
        # Optional auxiliary head — predicts next-day per-stock log returns.
        # Disabled (None) means zero param overhead and no aux gradient path.
        self.aux_return_head: ReturnPredictionHead | None = (
            ReturnPredictionHead(d, hidden=cfg.aux_return_hidden)
            if cfg.use_aux_return_head
            else None
        )
        # Learnable log-std per action dimension (N+1)
        self.log_std = nn.Parameter(
            torch.full((cfg.n_tickers + 1,), cfg.log_std_init)
        )

    def _forward_shared(
        self,
        obs: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Internal forward that returns ``(action_mean, value, log_std, aux_pred)``.

        ``aux_pred`` is ``None`` when the auxiliary head is disabled.

        Centralising this makes ``forward`` and ``get_action_value_and_aux``
        share *all* encoder/attention/FiLM compute — the aux head is the
        only extra op when enabled.
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

        # Regime conditioning vector (only used if any FiLM block / critic
        # consumer is active; otherwise it's silently ignored).  Shape [B, R].
        regime: torch.Tensor | None = None
        if self.cfg.regime_dim > 0:
            regime_raw = obs.get("regime")
            if regime_raw is not None:
                regime = regime_raw.float()
                if regime.dim() == 1:
                    regime = regime.unsqueeze(0)

        z = self.encoder(features)                         # [B, N, d]
        if self.film_encoder is not None and regime is not None:
            z = self.film_encoder(z, regime)
        if self.cross_attn is not None:
            mask_raw = obs.get("mask")
            tradeable: torch.Tensor | None = (
                mask_raw.bool() if mask_raw is not None else None
            )
            z = self.cross_attn(z, tradeable_mask=tradeable)  # [B, N, d]
            if self.film_attn is not None and regime is not None:
                z = self.film_attn(z, regime)
        action_mean = self.actor(z, portfolio)             # [B, N+1]
        value = self.critic(
            z, portfolio, t_frac, sector_ids,
            recent_return, recent_vol, nav_log_progress,
            regime=regime if self.cfg.regime_in_critic else None,
        )                                                  # [B]
        aux_pred: torch.Tensor | None = (
            self.aux_return_head(z) if self.aux_return_head is not None else None
        )
        return action_mean, value, self.log_std, aux_pred

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
        action_mean, value, log_std, _ = self._forward_shared(obs)
        return action_mean, value, log_std

    def get_action_and_value(
        self,
        obs: dict[str, torch.Tensor],
        action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample or evaluate an action; return (action, log_prob, entropy, value)."""
        mean, value, log_std, _ = self._forward_shared(obs)
        std = log_std.exp().expand_as(mean)
        dist = Normal(mean, std)

        if action is None:
            action = dist.sample()  # type: ignore[no-untyped-call]

        log_prob = dist.log_prob(action).sum(-1)   # type: ignore[no-untyped-call]  # [B]
        entropy = dist.entropy().sum(-1)           # type: ignore[no-untyped-call]  # [B]
        return action, log_prob, entropy, value

    def get_action_value_and_aux(
        self,
        obs: dict[str, torch.Tensor],
        action: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None
    ]:
        """Like ``get_action_and_value`` but also returns the auxiliary
        next-day-return prediction.  Used during the PPO update step when
        ``aux_return_loss_coef > 0``; rollout uses the cheaper version
        above and discards the aux output entirely.

        Returns ``(action, log_prob, entropy, value, aux_pred)``.
        ``aux_pred`` is ``None`` when the model has no aux head.
        """
        mean, value, log_std, aux_pred = self._forward_shared(obs)
        std = log_std.exp().expand_as(mean)
        dist = Normal(mean, std)
        if action is None:
            action = dist.sample()                       # type: ignore[no-untyped-call]
        log_prob = dist.log_prob(action).sum(-1)         # type: ignore[no-untyped-call]
        entropy = dist.entropy().sum(-1)                 # type: ignore[no-untyped-call]
        return action, log_prob, entropy, value, aux_pred

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
        regime_mean: torch.Tensor | None = None,
        regime_std: torch.Tensor | None = None,
    ) -> ActorCritic:
        """Construct from a gymnasium observation space.

        Auto-detects ``regime_dim`` from the obs space if a ``regime`` key
        is present; callers can still override by passing
        ``cfg_overrides={"regime_dim": ...}``.
        """
        feat_shape = obs_space["features"].shape   # (lookback, N, F)
        n_tickers = feat_shape[1]
        in_features = feat_shape[2]
        overrides = dict(cfg_overrides or {})
        if "regime_dim" not in overrides and "regime" in obs_space.spaces:
            overrides["regime_dim"] = int(obs_space["regime"].shape[0])
        cfg = ModelConfig(
            in_features=in_features,
            n_tickers=n_tickers,
            **overrides,
        )
        return cls(
            cfg,
            feat_mean=feat_mean,
            feat_std=feat_std,
            regime_mean=regime_mean,
            regime_std=regime_std,
        )
