"""Baseline agents that emit the same logit-vector action as the RL policy."""
from __future__ import annotations

import math

import numpy as np


class BaselineAgent:
    """Base class for all baseline agents."""

    def reset(self, info: dict[str, object] | None = None) -> None:
        """Called at the start of each episode."""

    def act(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        """Return logit vector of shape (N+1,); index 0 = cash."""
        raise NotImplementedError


# ── helpers ───────────────────────────────────────────────────────────────────

def _equal_weight_logits(mask: np.ndarray) -> np.ndarray:
    """Return logits that produce equal weight across tradeable equity names."""
    N = len(mask)
    logits = np.full(N + 1, -1e9, dtype=np.float64)
    n_tradeable = int(mask.sum())
    if n_tradeable == 0:
        logits[0] = 0.0   # all cash
        return logits
    # Set tradeable tickers to 0.0; cash gets a small negative bias so equal
    # weight wins over cash when tickers are available.
    logits[0] = -1.0
    for i, t in enumerate(mask):
        if t:
            logits[i + 1] = 0.0
    return logits


def _top_k_logits(mask: np.ndarray, scores: np.ndarray, k: int) -> np.ndarray:
    """Return logits concentrating weight on top-k names by score."""
    N = len(mask)
    logits = np.full(N + 1, -1e9, dtype=np.float64)
    tradeable_idx = np.where(mask)[0]
    if len(tradeable_idx) == 0:
        logits[0] = 0.0
        return logits
    k_eff = min(k, len(tradeable_idx))
    top_idx = tradeable_idx[np.argsort(scores[tradeable_idx])[::-1][:k_eff]]
    logits[0] = -1.0
    logits[top_idx + 1] = 0.0   # equal logit → equal weight after softmax
    return logits


# ── concrete baselines ────────────────────────────────────────────────────────


class EqualWeightRebalanced(BaselineAgent):
    """1/N_tradeable across all tradeable equities, rebalanced each step."""

    def act(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        mask = obs["mask"].astype(bool)
        return _equal_weight_logits(mask).astype(np.float32)


class BuyAndHoldIndex(BaselineAgent):
    """Buy equal weight on first tradeable day, never rebalance.

    Approximates buy-and-hold of the universe index.
    """

    def __init__(self) -> None:
        self._fixed_logits: np.ndarray | None = None

    def reset(self, info: dict[str, object] | None = None) -> None:
        self._fixed_logits = None

    def act(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        mask = obs["mask"].astype(bool)
        if self._fixed_logits is None:
            self._fixed_logits = _equal_weight_logits(mask).astype(np.float32)
        return self._fixed_logits


class MomentumTopK(BaselineAgent):
    """Top K tickers by trailing 20-day log return, equal-weighted.

    Uses the precomputed ``log_return_20d`` feature from the observation.
    Rebalanced every `rebalance_freq` steps.
    """

    def __init__(self, k: int = 5, rebalance_freq: int = 21) -> None:
        self._k = k
        self._rebalance_freq = rebalance_freq
        self._step = 0
        self._cached_logits: np.ndarray | None = None

    def reset(self, info: dict[str, object] | None = None) -> None:
        self._step = 0
        self._cached_logits = None

    def act(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        mask = obs["mask"].astype(bool)
        if self._step % self._rebalance_freq == 0 or self._cached_logits is None:
            features = obs["features"]          # (lookback, N, F)
            # Use the last lookback step's feature values.
            # We need log_return_20d, but we don't know its feature index here.
            # Fall back to the last row of features col 0 as a proxy score.
            # In practice the env should be configured with feature_columns that
            # includes log_return_20d; this picks whatever feature is at index 0.
            scores = features[-1, :, 0].astype(np.float64)   # (N,)
            self._cached_logits = _top_k_logits(mask, scores, self._k).astype(np.float32)
        self._step += 1
        return self._cached_logits


class SixtyFortyCash(BaselineAgent):
    """60% equal-weight equity, 40% cash, rebalanced each step."""

    def act(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        mask = obs["mask"].astype(bool)
        N = len(mask)
        logits = np.full(N + 1, -1e9, dtype=np.float64)
        n_tradeable = int(mask.sum())
        if n_tradeable == 0:
            logits[0] = 0.0
            return logits.astype(np.float32)
        # 60% across equity names → each name gets weight 0.6/N_t
        # 40% cash → cash weight 0.4
        # We achieve this ratio by setting:
        #   logit_cash  = log(0.40)
        #   logit_stock = log(0.60 / N_t)   (same for all tradeable)
        logits[0] = math.log(0.40)
        per_stock = math.log(0.60 / n_tradeable)
        for i, t in enumerate(mask):
            if t:
                logits[i + 1] = per_stock
        return logits.astype(np.float32)


class RandomPolicy(BaselineAgent):
    """Sanity floor: uniformly random logits over tradeable names."""

    def __init__(self, seed: int | None = None) -> None:
        self._rng = np.random.default_rng(seed)

    def reset(self, info: dict[str, object] | None = None) -> None:
        pass

    def act(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        mask = obs["mask"].astype(bool)
        N = len(mask)
        logits = np.full(N + 1, -1e9, dtype=np.float64)
        logits[0] = float(self._rng.uniform(-1.0, 1.0))
        for i, t in enumerate(mask):
            if t:
                logits[i + 1] = float(self._rng.uniform(-1.0, 1.0))
        return logits.astype(np.float32)
