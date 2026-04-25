"""Per-ticker time-series encoders."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


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
