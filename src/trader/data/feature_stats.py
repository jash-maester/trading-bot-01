"""Per-feature normalisation statistics.

Computed once from the **training** panel and frozen.  Validation/test envs
must use the same stats — using their own statistics would be a (subtle)
form of data leakage on the policy, and would also produce a different
input distribution from what the model was trained on.

The features in this project span ~11 orders of magnitude in raw scale
(``dollar_volume_20`` is in INR units, ``log_return_1d`` is around 0.02).
Without standardisation the TCN's first convolution is dominated by the
largest-scale features and the model is effectively blind to returns,
RSI, volatility and the rest.  Normalising to zero-mean unit-variance
fixes this completely.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import torch


def compute_feature_stats(
    panel_path: Path,
    feature_cols: list[str],
) -> dict[str, tuple[float, float]]:
    """Compute (mean, std) per feature over **tradeable** rows of the panel.

    Only `is_tradeable=True` rows are used so that warm-up rows (which were
    sentinel-filled with 0 in `compute_features`) don't pollute the stats.

    Returns a dict keyed by feature name; values are ``(mean, std)`` floats.
    """
    df = pl.read_parquet(panel_path).filter(pl.col("is_tradeable"))
    stats: dict[str, tuple[float, float]] = {}
    for col in feature_cols:
        if col not in df.columns:
            stats[col] = (0.0, 1.0)
            continue
        vals = df[col].drop_nulls().drop_nans().to_numpy()
        if vals.size == 0 or float(np.std(vals)) < 1e-8:
            # Constant or empty feature — neutral pass-through.
            stats[col] = (0.0, 1.0)
        else:
            stats[col] = (float(np.mean(vals)), float(np.std(vals)))
    return stats


def stats_to_tensors(
    stats: dict[str, tuple[float, float]],
    feature_cols: list[str],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert a stats dict into ordered ``(mean, std)`` 1-D tensors.

    Order matches `feature_cols` so the tensors line up with the feature
    axis of the encoder input.
    """
    means = torch.tensor([stats[c][0] for c in feature_cols], dtype=torch.float32)
    stds = torch.tensor(
        [max(stats[c][1], 1e-6) for c in feature_cols], dtype=torch.float32
    )
    return means, stds


def save_stats(
    stats: dict[str, tuple[float, float]],
    path: Path,
) -> None:
    """Persist stats as JSON for reproducibility (alongside panel SHA256)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump({k: list(v) for k, v in stats.items()}, f, indent=2, sort_keys=True)


def load_stats(path: Path) -> dict[str, tuple[float, float]]:
    """Load stats from a JSON file produced by `save_stats`."""
    with path.open() as f:
        raw = json.load(f)
    return {k: (float(v[0]), float(v[1])) for k, v in raw.items()}
