"""R4 — the supervised cross-sectional signal model.

``TCNEncoder`` (encoders.py) → one ``ReturnPredictionHead`` (heads.py) per
horizon, sharing the encoder.  Nothing here is new modelling: both pieces exist,
and the head's own docstring already makes the case for supervising every
per-stock embedding with a dense regression target instead of one scalar reward
per day.  `10_architecture_revamp.md` §5 promotes that auxiliary head to the
*primary* model; this module is that composition and nothing more.

Conventions
-----------
Input ``features`` is ``[B, L, N, F]`` — the env's layout (see
``actor_critic.py:157``), permuted to ``[B, N, L, F]`` for the encoder.  A
prediction is *dated* by the last row of its window: ``r_hat[t]`` is made from
feature rows ``t-L+1 .. t`` (information through the close of ``t``) and is a
forecast of the cross-sectionally standardised return over ``t+1 .. t+h``.

The env's observation on decision day ``d`` is rows ``d-L .. d-1``
(``panel_env.py:_build_obs``), i.e. information through the close of ``d-1``.
So a signal dated ``t`` corresponds to the env's decision on day ``t+1``.  The
allocator must trade on or after the open of ``t+1`` — never on ``t`` itself.

There is deliberately no ``CrossStockAttention`` in this model.  Cross-sectional
comparison happens in the *target* (the per-date z-score) and in the *metric*
(rank IC), not in the network; each ``r_hat_i`` is a function of stock ``i``'s
own window only.  That keeps the embedding cache (``encode``) valid for the RL
layer regardless of which subset of the universe is tradeable on a given day.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from trader.models.encoders import TCNEncoder
from trader.models.heads import ReturnPredictionHead

DEFAULT_HORIZONS: tuple[int, ...] = (5, 20)


def horizon_key(h: int) -> str:
    """Output-column name for a horizon: ``r_hat_5d``, ``r_hat_20d`` …

    This is also the column name in ``predictions.parquet`` — a pinned contract
    consumed by the allocator and the RL layer, so it lives in exactly one place.
    """
    return f"r_hat_{h}d"


@dataclass
class SignalConfig:
    """Hyper-parameters for :class:`SignalModel`."""

    in_features: int
    embed_dim: int = 128
    num_channels: list[int] | None = None   # TCN channels per layer; None → [embed_dim]*4
    kernel_size: int = 3
    dropout: float = 0.1
    head_hidden: int = 32
    horizons: tuple[int, ...] = field(default_factory=lambda: DEFAULT_HORIZONS)

    def __post_init__(self) -> None:
        if not self.horizons:
            raise ValueError("SignalConfig.horizons must not be empty")
        if any(h <= 0 for h in self.horizons):
            raise ValueError(f"horizons must be positive, got {self.horizons}")
        if len(set(self.horizons)) != len(self.horizons):
            raise ValueError(f"horizons must be unique, got {self.horizons}")


class SignalModel(nn.Module):
    """Shared TCN encoder with one return-prediction head per horizon.

    Parameters
    ----------
    cfg:
        See :class:`SignalConfig`.
    feat_mean, feat_std:
        Per-feature normalisation stats ``(F,)`` computed from the **training**
        split only (``trader.data.feature_stats``).  They are stored as buffers
        inside the encoder, so a saved ``state_dict`` carries them and the
        cached embeddings are reproducible from the weights alone.
    """

    def __init__(
        self,
        cfg: SignalConfig,
        feat_mean: torch.Tensor | None = None,
        feat_std: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.horizons: tuple[int, ...] = tuple(cfg.horizons)
        self.encoder = TCNEncoder(
            in_features=cfg.in_features,
            embed_dim=cfg.embed_dim,
            num_channels=cfg.num_channels,
            kernel_size=cfg.kernel_size,
            dropout=cfg.dropout,
            feat_mean=feat_mean,
            feat_std=feat_std,
        )
        self.embed_dim: int = self.encoder.out_dim
        self.heads = nn.ModuleDict(
            {
                horizon_key(h): ReturnPredictionHead(self.embed_dim, hidden=cfg.head_hidden)
                for h in self.horizons
            }
        )

    # ── forward ──────────────────────────────────────────────────────────────

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        """Per-stock embeddings ``z`` from an env-layout feature window.

        Parameters
        ----------
        features : Tensor [B, L, N, F]

        Returns
        -------
        Tensor [B, N, D]
        """
        if features.dim() != 4:
            raise ValueError(
                f"features must be [B, L, N, F], got shape {tuple(features.shape)}"
            )
        x = features.float().permute(0, 2, 1, 3)      # → [B, N, L, F] for the TCN
        # nn.Module.__call__ is typed `-> Any`, so bind through an annotated
        # local rather than returning Any from a Tensor-typed function.
        z: torch.Tensor = self.encoder(x)
        return z

    def predict_from_embeddings(
        self,
        z: torch.Tensor,                          # [B, N, D]
        mask: torch.Tensor | None = None,         # [B, N] bool/int8, True = tradeable
    ) -> dict[str, torch.Tensor]:
        """Run every horizon head on cached embeddings.

        Untradeable positions are zeroed so a consumer that forgets to mask
        cannot read a real-looking number off a sentinel-filled row.
        """
        out: dict[str, torch.Tensor] = {}
        m: torch.Tensor | None = None
        if mask is not None:
            m = mask.bool().to(z.dtype)
        for key, head in self.heads.items():
            pred: torch.Tensor = head(z)              # [B, N]
            out[key] = pred if m is None else pred * m
        return out

    def forward(
        self,
        features: torch.Tensor,                   # [B, L, N, F]
        mask: torch.Tensor | None = None,         # [B, N]
    ) -> dict[str, torch.Tensor]:
        """Return ``{"r_hat_5d": [B, N], "r_hat_20d": [B, N]}`` (one per horizon)."""
        return self.predict_from_embeddings(self.encode(features), mask)

    # ── provenance ───────────────────────────────────────────────────────────

    def encoder_state_sha256(self) -> str:
        """Deterministic digest of the encoder weights *and* normaliser buffers.

        Written into ``index.json`` next to the cached embeddings so a consumer
        can tell which encoder produced them.  Keys are sorted and every tensor
        is serialised as contiguous float32 bytes, so the digest is independent
        of device, dtype promotion, and insertion order.
        """
        return state_dict_sha256(self.encoder.state_dict())


def state_dict_sha256(state: dict[str, torch.Tensor]) -> str:
    """sha256 over a state dict; see :meth:`SignalModel.encoder_state_sha256`."""
    h = hashlib.sha256()
    for key in sorted(state):
        t = state[key].detach().to("cpu", torch.float32).contiguous()
        h.update(key.encode("utf-8"))
        h.update(str(tuple(t.shape)).encode("utf-8"))
        h.update(t.numpy().tobytes())
    return h.hexdigest()
