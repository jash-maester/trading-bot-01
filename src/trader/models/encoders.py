"""Per-ticker time-series encoders and cross-stock attention."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossStockAttention(nn.Module):
    """All-pairs self-attention over the universe of stocks.

    Inserted between the per-ticker encoder (TCN, optionally GNN) and the
    actor / critic heads.  Lets each stock's representation be informed by
    *every* other stock in the universe — critical for portfolio
    allocation, where relative merit determines the weight assigned.

    Without this layer, the actor head sees ``logit_i = MLP(z_i)`` where
    ``z_i`` depends only on stock-i features (and at most its sector
    neighbours via the GNN).  The softmax that turns logits into weights
    is a *normalisation*, not a *comparison* — so the model has no
    inductive bias for ranking.  Adding cross-stock attention fixes that:
    each ``z_i`` after this layer is a function of all 163 stock
    representations, so per-stock logits can express genuinely relative
    decisions.

    Untradeable stocks are masked from attention via ``key_padding_mask``
    so they don't contaminate other stocks' refined representations.
    Their own outputs still get masked at action time by `masked_softmax`
    in the env.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x: torch.Tensor,                                # [B, N, d]
        tradeable_mask: torch.Tensor | None = None,     # [B, N] bool, True = tradeable
    ) -> torch.Tensor:
        kpm: torch.Tensor | None = None
        if tradeable_mask is not None:
            # MultiheadAttention `key_padding_mask`: True ⇒ ignore that key.
            kpm = ~tradeable_mask.bool()
            # Numerical-safety guard: if every key is masked for a batch
            # element, MHA emits NaN.  Force at least one key to be
            # un-masked.  In practice this never fires (the universe always
            # has tradeable stocks on every training day), but it costs
            # nothing and prevents a class of silent NaN bugs.
            all_masked = kpm.all(dim=1)
            if bool(all_masked.any().item()):
                kpm = kpm.clone()
                kpm[all_masked, 0] = False

        attn_out, _ = self.attn(x, x, x, key_padding_mask=kpm, need_weights=False)
        return self.norm(x + attn_out)  # type: ignore[no-any-return]


class RegimeNormalizer(nn.Module):
    """Frozen z-score standardisation for the regime conditioning vector.

    Mirrors `FeatureNormalizer` but operates on the [B, R] regime tensor
    rather than the [B, N, L, F] feature tensor.  Using fixed train-set
    stats means val/test envs see the same regime-feature distribution
    the FiLM net was trained on — without normalisation, FiLM would have
    to spend capacity absorbing the natural scale of each feature
    (e.g., `mkt_vol_20d` ~ 0.02, `mkt_breadth_20d` ~ 0.5).

    If `mean=None or std=None`, this becomes an identity (used for tests
    and ablation toggles).
    """

    def __init__(
        self,
        mean: torch.Tensor | None = None,
        std: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if mean is None or std is None:
            self.register_buffer("mean", torch.zeros(1))
            self.register_buffer("std", torch.ones(1))
            self._identity = True
        else:
            self.register_buffer("mean", mean.float())
            self.register_buffer("std", std.float().clamp_min(1e-6))
            self._identity = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._identity:
            return x
        mean: torch.Tensor = self.mean      # type: ignore[assignment]
        std: torch.Tensor = self.std        # type: ignore[assignment]
        return (x - mean) / std


class FiLM(nn.Module):
    """Feature-wise Linear Modulation (Perez et al. 2018).

    Given a per-batch conditioning vector ``c ∈ R^{cond_dim}`` (the regime
    vector here) and a per-stock embedding ``x ∈ R^{B×N×D}``, FiLM produces
    affine modulation parameters ``(γ, β) ∈ R^{B×D}`` from c via a small
    MLP and applies them as::

        out = γ.unsqueeze(1) * x + β.unsqueeze(1)

    The same (γ, β) is applied to every stock — appropriate here because
    the regime is a market-wide property; per-stock adjustments are
    handled later by the actor head and (in this architecture) by
    CrossStockAttention.

    Why FiLM over concatenation?
    ----------------------------
    Concatenating the regime vector to per-stock features grows the
    encoder input dimension and forces every layer to relearn the
    regime-dependence.  FiLM is *multiplicative*: it gates which features
    matter in which regime.  Empirically this gives much stronger
    conditional behaviour than a residual concat with the same parameter
    budget.

    Initialisation
    --------------
    The last linear's weights and biases are initialised to zero so the
    layer outputs ``(Δγ, Δβ) = 0`` at start, with γ = 1 + Δγ → identity
    at step 0.  This means the network behaves exactly like the
    non-FiLM model at initialisation; FiLM only gradually learns
    regime-dependent modulations as gradients flow.
    """

    def __init__(
        self,
        cond_dim: int,
        embed_dim: int,
        hidden: int = 32,
        normalizer: RegimeNormalizer | None = None,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.normalizer = normalizer if normalizer is not None else RegimeNormalizer()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2 * embed_dim),
        )
        # Identity-init: zeroing the final layer makes Δγ=Δβ=0,
        # so γ=1 and β=0 at step 0 (out = x unchanged).
        last = self.net[-1]
        assert isinstance(last, nn.Linear)
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Apply FiLM to a per-stock embedding tensor.

        Parameters
        ----------
        x : Tensor [B, N, D]
            Per-stock embeddings.
        cond : Tensor [B, R]
            Regime conditioning vector (one per batch element).

        Returns
        -------
        Tensor [B, N, D]
        """
        if cond.dim() == 1:
            cond = cond.unsqueeze(0)
        cond = self.normalizer(cond)
        gb = self.net(cond)                                   # [B, 2D]
        gamma_delta, beta = gb.chunk(2, dim=-1)               # each [B, D]
        gamma = 1.0 + gamma_delta
        out: torch.Tensor = gamma.unsqueeze(1) * x + beta.unsqueeze(1)
        return out                                            # [B, N, D]


class FeatureNormalizer(nn.Module):
    """Fixed per-feature standardisation: ``(x - mean) / std``.

    Stats are stored as buffers (so they save/load with the model) and are
    NOT updated during training.  This is critical: features in this project
    span ~11 orders of magnitude in raw scale, so without standardisation
    the TCN's first conv layer is dominated by `dollar_volume_20` and the
    model is effectively blind to the returns / RSI / vol features.

    Parameters
    ----------
    mean, std : Tensor of shape ``(F,)``
        Computed once from the **training** panel via
        ``trader.data.feature_stats.compute_feature_stats``.
    """

    def __init__(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("mean", mean.float())
        self.register_buffer("std", std.float().clamp_min(1e-6))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x can be any shape ending in F.  Broadcasting handles the rest.
        # `register_buffer` is typed as Tensor | Module by torch stubs,
        # so help mypy with explicit casts.
        mean: torch.Tensor = self.mean  # type: ignore[assignment]
        std: torch.Tensor = self.std    # type: ignore[assignment]
        return (x - mean) / std


class _CausalConv1d(nn.Module):
    """Left-pad so the output length equals the input length (causal)."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int, dilation: int) -> None:
        super().__init__()
        self._pad = (kernel - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, (self._pad, 0)))  # type: ignore[no-any-return]


class _TCNBlock(nn.Module):
    def __init__(
        self, in_ch: int, out_ch: int, kernel: int, dilation: int, dropout: float
    ) -> None:
        super().__init__()
        self.conv1 = _CausalConv1d(in_ch, out_ch, kernel, dilation)
        self.conv2 = _CausalConv1d(out_ch, out_ch, kernel, dilation)
        self.norm1 = nn.LayerNorm(out_ch)
        self.norm2 = nn.LayerNorm(out_ch)
        self.drop = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x if self.downsample is None else self.downsample(x)
        # [B, C, L] → conv → [B, C, L]
        out = self.conv1(x)
        out = self.norm1(out.transpose(1, 2)).transpose(1, 2)
        out = F.gelu(out)
        out = self.drop(out)
        out = self.conv2(out)
        out = self.norm2(out.transpose(1, 2)).transpose(1, 2)
        out = F.gelu(out)
        out = self.drop(out)
        return out + res  # type: ignore[no-any-return]


class TCNEncoder(nn.Module):
    """Temporal Convolutional Network encoder for per-ticker feature windows.

    Input:  (B, N, L, F) — batch, tickers, lookback, features
    Output: (B, N, embed_dim)
    """

    def __init__(
        self,
        in_features: int,
        embed_dim: int = 64,
        num_channels: list[int] | None = None,
        kernel_size: int = 3,
        dropout: float = 0.1,
        feat_mean: torch.Tensor | None = None,
        feat_std: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        channels = num_channels if num_channels is not None else [embed_dim] * 4
        dilations = [1, 2, 4, 8][: len(channels)]

        # ── Optional input normaliser ─────────────────────────────────────
        # Strongly recommended. Without it the conv weights are pulled hard
        # toward whichever raw feature has the largest magnitude.
        self.input_norm: FeatureNormalizer | None
        if feat_mean is not None and feat_std is not None:
            self.input_norm = FeatureNormalizer(feat_mean, feat_std)
        else:
            self.input_norm = None

        blocks: list[nn.Module] = []
        in_ch = in_features
        for out_ch, dil in zip(channels, dilations):
            blocks.append(_TCNBlock(in_ch, out_ch, kernel_size, dil, dropout))
            in_ch = out_ch
        self.blocks = nn.Sequential(*blocks)
        self.out_dim = in_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, L, F]
        if self.input_norm is not None:
            x = self.input_norm(x)
        B, N, L, F = x.shape
        h = x.reshape(B * N, L, F).permute(0, 2, 1)   # [B*N, F, L]
        h = self.blocks(h)                              # [B*N, d, L]
        h = h[:, :, -1]                                 # last timestep → [B*N, d]
        return h.reshape(B, N, -1)  # type: ignore[no-any-return]  # [B, N, d]
