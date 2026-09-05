"""Baseline agents that emit the same logit-vector action as the RL policy.

A note on what a baseline can and cannot express here
----------------------------------------------------
`PanelTradingEnv.step` turns the logit vector into *target weights* and then
into target share counts on every **rebalance day**.  An agent that returns
the same logits twice therefore does not "hold" — the env marks the book to
the new NAV and trades back to those weights.  Nothing an agent returns
through this interface can produce a lower-than-daily cadence on its own.

Cadence lives in the env, not the agent.  Pass
``rebalance_schedule=RebalanceSchedule("monthly")`` (`trader.allocator`) to
`PanelTradingEnv` and it holds the book on every non-rebalance day — the
action is ignored, nothing trades, no cost is charged.  Every baseline below
honours that automatically because none of them carries state that assumes a
daily cadence; `MomentumTopK` re-ranks on every call precisely so that the
selection is fresh on whichever day the env actually trades.  Class docstrings
say what each baseline actually does rather than what it was named after.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from trader.data.features import FEATURE_COLS

# The column MomentumTopK ranks on.  Resolved by *name* against the feature
# list the env was built with — never by hardcoded position.
_MOMENTUM_COL = "log_return_20d"


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
    """1/N_tradeable across all tradeable equities, rebalanced on every
    **env rebalance day**.

    With the env's default ``rebalance_schedule=None`` that is daily, not
    monthly.  `03_environment.md` and `05_training.md` described this baseline
    as monthly-rebalanced and used that to argue the RL agent was unfairly
    compared against a lower-turnover, lower-tax benchmark; that premise was
    never true of this code.  It pays exactly the turnover drag the env's
    schedule allows, the same as the agent.

    A monthly variant cannot be built agent-side — see the module docstring —
    so this class deliberately has no `rebalance_freq` knob.  Build the env
    with ``RebalanceSchedule("monthly")`` instead.
    """

    def act(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        mask = obs["mask"].astype(bool)
        return _equal_weight_logits(mask).astype(np.float32)


class EqualWeightFrozenUniverse(BaselineAgent):
    """Equal weight over the set of names tradeable on the episode's **first**
    day, held fixed for the rest of the episode.

    Renamed from `BuyAndHoldIndex`, which misdescribed it twice:

    * **It tracks no index.**  There is no index membership anywhere in this
      class — it holds the entire tradeable universe (a ~500-name midcap-tilted
      NSE list), which is not a proxy for the NIFTY 50 in the expected-ordering
      table of `05_training.md`.  Nothing here is comparable to a real index
      until an index definition (constituents + weights) exists in the data
      layer; that is a data-side change, not a baseline-side one.
    * **It is not buy-and-hold.**  Returning frozen logits freezes only the
      *universe*; the env re-derives target shares from those weights on every
      rebalance day and trades the drift back out (see the module docstring).
      The only difference from `EqualWeightRebalanced` is that names becoming
      tradeable mid-episode are never added, and names dropping out are never
      removed from the target.  Under a monthly env schedule the drift is
      traded out once a month, which is as close to buy-and-hold as this
      interface gets.

    The old name is kept as a module-level alias below purely because
    `scripts/evaluate.py` and `scripts/paper_run.py` import it.
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


# Deprecated alias.  `BuyAndHoldIndex` neither tracks an index nor holds; use
# `EqualWeightFrozenUniverse`.  Retained only so that `scripts/evaluate.py:29`
# and `scripts/paper_run.py:113` keep importing — both are outside this
# change's file ownership and should be switched over, along with their
# "buy_and_hold" display key.
BuyAndHoldIndex = EqualWeightFrozenUniverse


class MomentumTopK(BaselineAgent):
    """Top-K tickers by trailing 20-day log return, equal-weighted.

    Ranking column
    --------------
    Scores are the most recent row of the observation's feature window for
    ``log_return_20d``.  ``obs["features"]`` is an unlabelled ``(lookback, N,
    F)`` array, so the column index is resolved by name against the same list
    the env was built from — ``trader.data.features.FEATURE_COLS`` by default,
    or an explicit ``feature_columns`` for an env configured with a different
    list.  Never by position: this previously read ``features[-1, :, 0]``,
    which is ``log_return_1d``, so "momentum top-K" was a one-day
    reversal/continuation signal with no 20-day content at all.  ``act`` also
    checks the observation's F against the resolved list and raises on a
    mismatch, so a future edit to ``FEATURE_COLS`` fails loudly instead of
    silently sliding onto another column.

    K and the per-name weight cap
    -----------------------------
    K must be chosen against the env's ``max_weight_per_name`` (0.10 by
    default, `panel_env.py:50`).  K equal-weighted names can hold at most
    ``K * cap`` of the book; the remainder is forced into cash by
    ``_cap_and_renormalize``.  So K < 1/cap = 10 leaves the "momentum"
    baseline part money-market fund.  Measured at N=40 tradeable, cap=0.10:
    **K=5 → 49.8% cash; K=20 → 1.8% cash** (the residual is the deliberate
    ``logit_cash = -1.0`` tilt in ``_top_k_logits``, not the cap).  Hence the
    default K=20.  Raise `max_weight_per_name` if you want a smaller K.

    Rebalance cadence
    -----------------
    Cadence belongs to the env's ``rebalance_schedule``, not to this class.
    ``rebalance_freq`` (default 1 = re-rank on every call) freezes the
    *selection* for that many *calls*; it was 21 when the env could not hold,
    as a stand-in for monthly.  With an env schedule that stand-in is actively
    wrong: a 21-call counter and the calendar's first-trading-day-of-month
    drift apart, so the env would trade a selection up to twenty days stale.
    Leave it at 1 and let the env decide when to trade.
    """

    def __init__(
        self,
        k: int = 20,
        rebalance_freq: int = 1,
        feature_columns: Sequence[str] | None = None,
        momentum_col: str = _MOMENTUM_COL,
    ) -> None:
        cols = list(FEATURE_COLS if feature_columns is None else feature_columns)
        if momentum_col not in cols:
            raise ValueError(
                f"MomentumTopK ranks on {momentum_col!r}, which is not in the "
                f"feature columns it was given ({cols!r}). Pass the same "
                f"feature_columns list the env was built with."
            )
        self._k = k
        self._rebalance_freq = rebalance_freq
        self._momentum_col = momentum_col
        self._score_idx = cols.index(momentum_col)
        self._n_features = len(cols)
        self._step = 0
        self._cached_logits: np.ndarray | None = None

    def reset(self, info: dict[str, object] | None = None) -> None:
        self._step = 0
        self._cached_logits = None

    def act(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        mask = obs["mask"].astype(bool)
        if self._step % self._rebalance_freq == 0 or self._cached_logits is None:
            features = obs["features"]          # (lookback, N, F)
            if features.shape[-1] != self._n_features:
                raise ValueError(
                    f"MomentumTopK resolved {self._momentum_col!r} to column "
                    f"{self._score_idx} of a {self._n_features}-column feature "
                    f"list, but the observation has {features.shape[-1]} "
                    f"columns. Pass feature_columns=<the env's list>."
                )
            scores = features[-1, :, self._score_idx].astype(np.float64)   # (N,)
            self._cached_logits = _top_k_logits(mask, scores, self._k).astype(np.float32)
        self._step += 1
        return self._cached_logits


class SixtyFortyCash(BaselineAgent):
    """60% equal-weight equity, 40% cash, rebalanced on each env rebalance day."""

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
