"""PanelTradingEnv — a Gymnasium env that steps one trading day at a time."""

from __future__ import annotations

import math
from datetime import date
from pathlib import Path
from typing import Any, SupportsFloat

import numpy as np
import polars as pl
from gymnasium import Env, spaces

from trader.allocator.rebalance import RebalanceSchedule
from trader.data.regime_features import REGIME_DIM, compute_regime_features
from trader.env.costs import (
    DEFAULT_MIN_TRADE_VALUE,
    CostModel,
    ZerodhaEquityDeliveryCostModel,
)
from trader.env.reward import LogReturn, RewardFn

_SLIPPAGE_K = 0.1
_NAV_HIST_LEN = 21      # holds 20 daily returns (NAV[t-20:t+1])


class PanelTradingEnv(Env):  # type: ignore[type-arg]
    """Gymnasium env wrapping a precomputed feature panel.

    Observation keys
    ----------------
    features    (lookback, N, F)  float32   panel window ending at t-1
    mask        (N,)              int8      is_tradeable at day t
    sector_ids  (N,)              int32
    portfolio   (N+1,)            float32   index 0 = cash weight
    cash        ()                float32
    nav         ()                float32
    t_frac      ()                float32   episode progress in [0, 1]

    Action space: Box(N+1,) logits; env applies masked softmax + cap.

    Rebalance cadence
    -----------------
    ``rebalance_schedule`` (default ``None`` = trade every step, the historical
    behaviour) restricts trading to the schedule's rebalance days.  On every
    other day the env **holds**: the incoming action is ignored, no shares
    change, no costs are charged and ``info["turnover"]`` is 0.0 — but NAV
    still marks to the close and the reward is still computed.  The first
    step of an episode always trades, so a monthly schedule does not leave a
    fresh episode in cash for up to twenty days.  See :meth:`step`.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        panel_path: Path,
        universe: list[str],
        feature_columns: list[str],
        lookback: int = 60,
        episode_length: int = 252,
        initial_cash: float = 1_000_000.0,
        cost_model: CostModel | None = None,
        reward_fn: RewardFn | None = None,
        allow_short: bool = False,
        max_weight_per_name: float = 0.10,
        turnover_penalty: float = 0.0,
        use_excess_returns: bool = False,
        seed: int | None = None,
        rebalance_schedule: RebalanceSchedule | None = None,
        min_trade_value: float = DEFAULT_MIN_TRADE_VALUE,
    ) -> None:
        super().__init__()

        self._cost_model: CostModel = cost_model or ZerodhaEquityDeliveryCostModel()
        if min_trade_value < 0.0:
            raise ValueError(f"min_trade_value must be >= 0, got {min_trade_value}")
        self._min_trade_value = float(min_trade_value)
        self._schedule = rebalance_schedule
        self._reward_fn: RewardFn = reward_fn or LogReturn()
        self._use_excess_returns = bool(use_excess_returns)
        self._lookback = lookback
        # NOTE: self._episode_length is assigned AFTER the panel-length clamp below.
        self._initial_cash = initial_cash
        self._allow_short = allow_short
        self._max_weight = max_weight_per_name
        self._turnover_penalty = turnover_penalty
        self._universe = list(universe)
        N = len(universe)

        # ── Load and prepare panel ────────────────────────────────────────────
        panel = pl.read_parquet(panel_path).sort(["date", "ticker"])
        self._dates: list[Any] = sorted(panel["date"].unique().to_list())
        self._feat_cols: list[str] = [c for c in feature_columns if c in panel.columns]

        # Auto-clamp episode_length so val/test panels (which are shorter than
        # train) don't crash.  Need: lookback + episode_length + 1 ≤ n_dates.
        _max_ep = len(self._dates) - lookback - 1
        if _max_ep < 1:
            raise ValueError(
                f"Panel too short for any episode: {len(self._dates)} days, "
                f"need at least lookback+2={lookback + 2}"
            )
        if episode_length > _max_ep:
            from loguru import logger as _log

            _log.warning(
                f"episode_length={episode_length} capped to {_max_ep} "
                f"(panel has {len(self._dates)} days, lookback={lookback})"
            )
            episode_length = _max_ep

        # Assign AFTER clamping so self._episode_length reflects the capped value.
        self._episode_length = episode_length

        # [T] bool — which calendar days the book may trade on.  None means
        # every day, and the step path is then byte-for-byte the pre-schedule
        # code (regression-tested).
        self._rebalance_mask: np.ndarray | None = (
            None if self._schedule is None else self._schedule.mask(self._dates)
        )

        # Pre-build per-(col, ticker) dense time-series arrays for O(1) access
        self._arrays = _build_ticker_arrays(
            panel, self._universe, self._dates, self._feat_cols
        )

        # ── Stacked [T, N, ...] arrays for vectorised step / obs ───────────────
        # The per-(col, ticker) dict was kept for compatibility but is far too
        # slow on the hot path: 60 × 163 × 15 dict lookups per step.  A single
        # numpy slice over a pre-stacked tensor is ~50× faster.
        T = len(self._dates)
        F = len(self._feat_cols)
        self._stacked_features = np.zeros((T, N, F), dtype=np.float32)
        for fi, col in enumerate(self._feat_cols):
            for ni, ticker in enumerate(self._universe):
                key = (col, ticker)
                if key in self._arrays:
                    self._stacked_features[:, ni, fi] = self._arrays[key]

        # Price / volume / metadata stacks ([T, N])
        def _stack(col: str, dtype: type[np.generic]) -> np.ndarray:
            arr = np.zeros((T, N), dtype=dtype)
            for ni, ticker in enumerate(self._universe):
                key = (col, ticker)
                if key in self._arrays:
                    arr[:, ni] = self._arrays[key]
            return arr

        self._stk_open = _stack("open", np.float64)
        self._stk_close = _stack("close", np.float64)
        self._stk_atr = _stack("atr_14", np.float64)
        self._stk_dollar_vol = _stack("dollar_volume_20", np.float64)
        self._stk_mask = _stack("is_tradeable", np.bool_)
        self._stk_sector_ids = _stack("sector_id", np.int32)

        # Equal-weight benchmark log return per day, used when
        # `use_excess_returns=True`. Computed cross-sectionally: mean of
        # `log_return_1d` across tradeable stocks each day.  Kept in raw
        # space (no normalisation) — that's a model-side concern.
        # Also kept on `self._stk_log_ret` for the auxiliary next-day
        # return prediction target (Phase 2).
        self._stk_log_ret = _stack("log_return_1d", np.float64)  # [T, N]
        log_ret_raw = self._stk_log_ret
        trd_f = self._stk_mask.astype(np.float64)
        trd_count = np.maximum(trd_f.sum(axis=1), 1.0)
        self._benchmark_log_ret = (log_ret_raw * trd_f).sum(axis=1) / trd_count  # [T]

        # Market-regime features ([T, R]) — exogenous signal for FiLM
        # conditioning in the model.  Always computed (cheap, ~O(T·N)
        # at construction); the model decides whether to consume them
        # via `obs["regime"]`.  Strictly backward-looking.
        self._regime_features = compute_regime_features(
            log_ret_raw, self._stk_mask
        )                                                         # [T, R] float32

        _sid_max = panel["sector_id"].max() if "sector_id" in panel.columns else None
        S: int = int(_sid_max) if isinstance(_sid_max, (int, float)) else 0
        F = len(self._feat_cols)

        self.observation_space = spaces.Dict(
            {
                "features": spaces.Box(-np.inf, np.inf, (lookback, N, F), np.float32),
                "mask": spaces.MultiBinary(N),
                "sector_ids": spaces.Box(0, S, (N,), np.int32),
                "portfolio": spaces.Box(0.0, 1.0, (N + 1,), np.float32),
                "cash": spaces.Box(0.0, np.inf, (), np.float32),
                "nav": spaces.Box(0.0, np.inf, (), np.float32),
                "t_frac": spaces.Box(0.0, 1.0, (), np.float32),
                # Critic enrichment: portfolio-level summary stats
                "recent_return_1d": spaces.Box(-np.inf, np.inf, (), np.float32),
                "recent_vol_20d": spaces.Box(0.0, np.inf, (), np.float32),
                "nav_log_progress": spaces.Box(-np.inf, np.inf, (), np.float32),
                # Market-regime conditioning vector (FiLM input).  Models
                # that don't use it simply ignore the key — the env always
                # emits it for forward compatibility.
                "regime": spaces.Box(-np.inf, np.inf, (REGIME_DIM,), np.float32),
                # Auxiliary supervised target: the next-day cross-section of
                # log returns (the "today" close-to-close move from the obs's
                # POV — the agent has features through yesterday's close).
                # This is the *target* for the optional ReturnPredictionHead;
                # the main policy/value path NEVER reads this key (it would
                # be a future leak).  See PPOTrainer for how it's consumed.
                "next_day_returns": spaces.Box(
                    -np.inf, np.inf, (N,), np.float32
                ),
            }
        )
        self.action_space = spaces.Box(-10.0, 10.0, (N + 1,), np.float32)

        # Episode state (reset before each episode)
        self._cash = initial_cash
        self._shares = np.zeros(N, dtype=np.float64)
        self._prev_nav = initial_cash
        self._t = 0
        self._start_idx = lookback  # safe default
        # Rolling NAV history for recent_return / recent_vol obs.  Size
        # _NAV_HIST_LEN holds (LEN-1) = 20 daily log returns.
        self._nav_history = np.full(_NAV_HIST_LEN, initial_cash, dtype=np.float64)

        self._np_rng = np.random.default_rng(seed)

    # ── Gymnasium API ─────────────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._np_rng = np.random.default_rng(seed)

        N = len(self._universe)
        min_idx = self._lookback
        # episode_length was already clamped in __init__; this should never fire
        max_idx = max(min_idx, len(self._dates) - self._episode_length - 1)

        self._start_idx = int(self._np_rng.integers(min_idx, max_idx + 1))
        self._t = 0
        self._cash = float(self._initial_cash)
        self._shares = np.zeros(N, dtype=np.float64)
        self._prev_nav = self._initial_cash
        # Reset rolling NAV history to flat baseline so recent stats start at 0.
        self._nav_history = np.full(
            _NAV_HIST_LEN, self._initial_cash, dtype=np.float64
        )
        self._reward_fn.reset()

        return self._build_obs(), {}

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], SupportsFloat, bool, bool, dict[str, Any]]:
        """Advance one trading day on a logit action.

        The action is turned into target weights by :func:`masked_softmax`
        (masked softmax + per-name cap) and traded at the open, **unless today
        is a hold day** under ``rebalance_schedule`` — then the action is
        ignored and the book is carried unchanged: no share changes, zero
        ``trade_val``, zero costs, ``info["turnover"] == 0.0``.  NAV still
        marks to today's close and the reward is computed as usual.
        ``info["rebalanced"]`` says which happened.
        """
        day_idx = self._start_idx + self._t
        if self.is_rebalance_step():
            mask = self._mask_at(day_idx)
            target_w = masked_softmax(action.astype(np.float64), mask, self._max_weight)
            return self._step_target(target_w[1:], day_idx, rebalance=True)
        return self._step_target(None, day_idx, rebalance=False)

    def step_weights(
        self, target_w: np.ndarray
    ) -> tuple[dict[str, np.ndarray], SupportsFloat, bool, bool, dict[str, Any]]:
        """Advance one trading day on an explicit **weight** target.

        ``target_w`` is ``[N+1]`` with cash at index 0, entries ≥ 0 and summing
        to 1 — exactly what :func:`trader.allocator.allocate` returns.  Weights
        are used as-is (no softmax, no cap; the allocator already applied its
        own), except that untradeable names are forced to zero as in
        :meth:`step`.  Hold-day semantics are identical to :meth:`step`.

        Feeding weights through :meth:`step` as ``log(w)`` does not work:
        :func:`masked_softmax` clips logits to ``[-10, 10]``, so every
        zero-weight name would receive ``e^-10`` relative mass — a dust
        position in hundreds of names, each paying a per-scrip DP charge on
        the way out.
        """
        day_idx = self._start_idx + self._t
        if not self.is_rebalance_step():
            return self._step_target(None, day_idx, rebalance=False)
        w = np.asarray(target_w, dtype=np.float64)
        N = len(self._universe)
        if w.shape != (N + 1,):
            raise ValueError(f"target_w must be shape [N+1]={N + 1}, got {w.shape}")
        if np.any(w < -1e-9) or not np.isfinite(w).all():
            raise ValueError("target_w must be finite and non-negative")
        total = float(w.sum())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"target_w must sum to 1, got {total}")
        return self._step_target(np.maximum(w[1:], 0.0), day_idx, rebalance=True)

    def is_rebalance_step(self) -> bool:
        """Whether the *next* call to :meth:`step` / :meth:`step_weights` may trade."""
        if self._rebalance_mask is None or self._t == 0:
            return True
        return bool(self._rebalance_mask[self._start_idx + self._t])

    @property
    def dates(self) -> list[date]:
        """The env's trading calendar (every date in the panel, sorted)."""
        return list(self._dates)

    @property
    def universe(self) -> list[str]:
        return list(self._universe)

    @property
    def day_index(self) -> int:
        """Calendar index (into :attr:`dates`) of the day the next step trades."""
        return self._start_idx + self._t

    @property
    def rebalance_schedule(self) -> RebalanceSchedule | None:
        return self._schedule

    # The three below exist so a wrapper can verify this env's reward
    # configuration without reaching into private attributes.  AllocatorEnv
    # sums the daily rewards and calls the sum "the period's excess log
    # return"; that identity is only true for a particular configuration, and
    # a `getattr(env, "_turnover_penalty", 0.0)` style check fails *open* — a
    # rename here would silently disable the guard rather than break it.

    @property
    def use_excess_returns(self) -> bool:
        """Whether the daily reward subtracts the equal-weight benchmark."""
        return self._use_excess_returns

    @property
    def turnover_penalty(self) -> float:
        """Per-day turnover charge applied inside the reward."""
        return self._turnover_penalty

    @property
    def reward_fn(self) -> RewardFn:
        """The reward function object the env steps."""
        return self._reward_fn

    def _step_target(
        self,
        eq_target_frac: np.ndarray | None,
        day_idx: int,
        *,
        rebalance: bool,
    ) -> tuple[dict[str, np.ndarray], SupportsFloat, bool, bool, dict[str, Any]]:
        """Shared step body.  ``eq_target_frac`` is ``[N]`` equity target
        weights on a rebalance day, ``None`` on a hold day (carry the book)."""
        mask = self._mask_at(day_idx)

        opens = self._price_at(day_idx, "open")
        closes = self._price_at(day_idx, "close")
        prev_closes = self._price_at(day_idx - 1, "close")
        atrs = self._col_at(day_idx, "atr_14")
        dollar_vol = self._col_at(day_idx, "dollar_volume_20")

        # NAV before today's trades (marked at yesterday's close)
        current_nav = max(self._cash + float(np.sum(self._shares * prev_closes)), 1e-8)

        # Target integer shares
        if rebalance and eq_target_frac is not None:
            target_equity_value = current_nav * np.where(mask, eq_target_frac, 0.0)
            requested = np.where(
                opens > 0, target_equity_value / np.maximum(opens, 1e-8), 0.0
            )
            target_shares = np.floor(requested)
            # A request that ROUNDS to the position we already hold is not an
            # order.  `floor` is kept for the *size* of a trade that does happen
            # — it is what stops a buy from spending more than the target value
            # — but it must not decide *whether* one happens, because it turns
            # an arbitrarily small request into a whole share:
            #
            #   * `obs["portfolio"]` is float32 (`_build_obs` below), so an
            #     allocator pinning a name to "its current weight" (P3's
            #     `no_trade_band`) hands back a weight good to ~6e-8 relative.
            #     floor(400 x (1 - 6e-8)) = 399 — a one-share SELL nobody asked
            #     for, which clears the 0.5-share integrality guard (it is a
            #     whole share) and clears `min_trade_value` on any name above
            #     ₹500, and pays the flat ₹15.34 demat debit.  Measured: 25.6%
            #     of pinned names still traded with open == prev_close exactly.
            #   * weights are marked at the previous close and filled at the
            #     open, so holding a weight across an overnight gap asks for a
            #     move of gap x position.  Under half a share that is not
            #     executable and rounding it up costs more than it corrects.
            #
            # Only ever REMOVES orders (|requested - held| < 0.5 could not have
            # produced a delta above one share anyway), never adds or enlarges
            # one, and never blocks a liquidation: a full exit requests 0.0
            # against a position of at least one share.
            # Regression: tests/unit/test_weight_to_share_orders.py.
            target_shares = np.where(
                np.abs(requested - self._shares) < 0.5, self._shares, target_shares
            )
        else:
            # Hold day: carry the book.  delta_shares is identically zero, so
            # every downstream quantity — fills, trade_val, costs, turnover —
            # is exactly zero without a special case.
            target_shares = self._shares.copy()
        delta_shares = target_shares - self._shares

        # Slippage
        adv_20 = np.where(closes > 0, dollar_vol / np.maximum(closes, 1e-8), 1.0)
        adv_20 = np.maximum(adv_20, 1.0)
        atr_frac = np.where(closes > 0, atrs / np.maximum(closes, 1e-8), 0.0)
        slip = (
            _SLIPPAGE_K
            * atr_frac
            * np.sqrt(np.abs(delta_shares) / adv_20)
            * np.sign(delta_shares)
        )
        fill_prices = np.where(opens > 0, opens * (1.0 + slip), 0.0)

        # Vectorised cost / cash accounting.  The previous loop was the
        # single biggest CPU hot-spot in step() — N=163 Python iterations
        # every day across 16 envs × 252 steps × ~2000 updates.
        # Two independent gates, and they are not the same test.
        #   * the 0.5-share guard is an INTEGRALITY check — target_shares is
        #     floored, so a sub-share delta is rounding noise, not an order;
        #   * the value guard is an ECONOMIC one. The demat debit fee is flat
        #     (see costs.DEFAULT_MIN_TRADE_VALUE), so a small enough trade pays
        #     more in fees than it moves in exposure.
        # `PaperBroker._emit_target_orders` drops on `abs(delta) * open_px`, so
        # this must too — on `opens`, NOT on `fill_prices`, or the two disagree
        # by the slippage term for trades sitting on the boundary.
        traded = (np.abs(delta_shares) >= 0.5) & (
            np.abs(delta_shares) * opens >= self._min_trade_value
        )
        trade_val = np.where(traded, np.abs(delta_shares) * fill_prices, 0.0)
        is_buy = delta_shares > 0
        n_sold = np.where(
            traded & (~is_buy) & (self._shares > 0), 1, 0
        ).astype(np.int64)

        cost_per_leg = self._cost_model.cost_vec(trade_val, is_buy, n_sold)
        signed_cash = np.where(traded, -delta_shares * fill_prices, 0.0)
        self._cash += float(signed_cash.sum() - cost_per_leg.sum())
        costs_paid = float(cost_per_leg.sum())

        # ── Turnover ─────────────────────────────────────────────────────────
        # Definition: gross traded value / pre-trade NAV.  Units: dimensionless
        # fraction of NAV per day (multiply by 252 for the annual figure — see
        # `eval_metrics.compute_episode_metrics`).  Scale:
        #     0.0  nothing traded
        #     1.0  one side of the book replaced (all-buy or all-sell)
        #     2.0  a full rotation — sell everything, buy something else
        # `trade_val` is the same rupee volume the cost model is charged on, so
        # `turnover_penalty * turnover` is proportional to the fees the trade
        # actually incurs; that proportionality is the entire point of the
        # penalty term.  This is the two-sided (gross) convention; the one-way
        # figure is half of it.
        #
        # The denominator is NAV *before* today's trades (marked at yesterday's
        # close, `current_nav` above), so the metric is a function of what was
        # traded and nothing else: a price move with no trading leaves it at
        # exactly 0.0.
        #
        # This MUST be computed before `self._shares` is overwritten below.
        # It previously was not: both weight vectors were built from the
        # post-trade share vector and differed only by their NAV denominator,
        # making the "turnover" a pure function of the day's NAV drift — a full
        # rotation reported 0.0 and a flat book through a -5% day reported 0.05.
        turnover = float(trade_val.sum()) / current_nav

        # Only positions that actually TRADED move. A delta suppressed by the
        # value guard (or the integrality guard) is an order that was never
        # sent, so the position stays exactly where it was — matching
        # `PaperBroker._emit_target_orders`, which drops the order and leaves
        # the holding untouched.
        #
        # Assigning `target_shares` unconditionally here moved the position
        # without the matching cash leg above (`signed_cash` is masked by
        # `traded`), i.e. it created shares for free. The bug was dormant while
        # the only guard was `>= 0.5` shares, which is true for every nonzero
        # integer delta; adding the value guard made it live and it inverted the
        # whole R5 grid before it was caught. Tested by
        # `test_suppressed_trade_leaves_the_position_and_the_cash_untouched`.
        self._shares = np.where(traded, target_shares, self._shares)

        # Mark-to-close
        new_nav = max(self._cash + float(np.sum(self._shares * closes)), 1e-8)
        log_return = math.log(new_nav / max(self._prev_nav, 1e-8))

        # Optional: subtract equal-weight benchmark from log_return so the
        # reward function sees *excess* return.  This gives the agent a
        # direct incentive to beat the universe equal-weight rather than
        # just earn positive returns in a bull market.
        reward_input = log_return
        if self._use_excess_returns:
            reward_input -= float(self._benchmark_log_ret[day_idx])
        reward = float(self._reward_fn(reward_input)) - self._turnover_penalty * turnover
        self._prev_nav = new_nav
        # Roll NAV history forward by one step (drop oldest, append new).
        # `np.roll` would also work but a slice is faster and allocation-free.
        self._nav_history[:-1] = self._nav_history[1:]
        self._nav_history[-1] = new_nav

        self._t += 1
        truncated = self._t >= self._episode_length
        terminated = new_nav < self._initial_cash * 0.10

        obs = self._build_obs()

        # Vectorised sector exposure (np.bincount over sector ids)
        sids = self._stk_sector_ids[day_idx]
        weights = self._shares * closes / max(new_nav, 1e-8)
        n_sec = int(sids.max()) + 1 if sids.size else 0
        sec_w = np.bincount(sids, weights=weights, minlength=n_sec) if n_sec else np.zeros(0)
        sector_exp: dict[int, float] = {
            int(s): float(sec_w[s]) for s in range(n_sec) if sec_w[s] != 0.0
        }

        info: dict[str, Any] = {
            "nav": new_nav,
            "turnover": turnover,
            "gross_exposure": float(np.sum(self._shares * closes)) / max(new_nav, 1e-8),
            "sector_exposure": sector_exp,
            "weights": np.concatenate(
                [
                    [self._cash / max(new_nav, 1e-8)],
                    self._shares * closes / max(new_nav, 1e-8),
                ]
            ).astype(np.float32),
            "date": self._dates[day_idx],
            "costs_paid": costs_paid,
            # The quantity the flat ₹15.34 demat debit is billed on: distinct
            # scrips sold today (`costs.py`, per scrip per selling day).  It is
            # 83-98% of all cost measured, and until this was exposed the only
            # way to count it was to subclass the cost model.  Weight-space
            # estimates of it — including `allocator.band_suppression` — are
            # upper bounds; this is the number the ledger pays.
            "n_scrips_sold": int(n_sold.sum()),
            "n_legs": int(np.count_nonzero(trade_val)),
            "log_return": log_return,
            "rebalanced": rebalance,
        }
        return obs, reward, bool(terminated), bool(truncated), info

    # ── obs builder ───────────────────────────────────────────────────────────

    def _build_obs(self) -> dict[str, np.ndarray]:
        day_idx = self._start_idx + self._t
        start = day_idx - self._lookback

        # Single slice into the pre-stacked [T, N, F] tensor (no Python loop).
        feat_arr = self._stacked_features[start:day_idx]   # [L, N, F]

        mask = self._stk_mask[day_idx]
        prev_closes = self._stk_close[day_idx - 1]
        current_nav = max(self._cash + float(np.sum(self._shares * prev_closes)), 1e-8)
        eq_w = (self._shares * prev_closes) / current_nav
        portfolio = np.concatenate(
            [[self._cash / current_nav], eq_w]
        ).astype(np.float32)
        portfolio = np.clip(portfolio, 0.0, 1.0)

        # ── Portfolio-level summary stats for the critic ─────────────────────
        # log_rets length = _NAV_HIST_LEN - 1 = 20.  At reset all entries equal
        # initial_cash so the diff is exactly zero — recent_return and
        # recent_vol are both 0.0 in the first observation, which is correct.
        log_rets = np.diff(np.log(np.maximum(self._nav_history, 1e-8)))
        recent_return_1d = float(log_rets[-1])
        recent_vol_20d = float(np.std(log_rets))
        nav_log_progress = float(
            np.log(max(current_nav, 1e-8) / max(self._initial_cash, 1e-8))
        )

        # Auxiliary target: next-day cross-section of log returns.
        # `day_idx` is the day being traded; obs features end at day_idx-1's
        # close, so the *next* return (from day_idx-1 close to day_idx close)
        # is `_stk_log_ret[day_idx]` — exactly the cross-section the agent
        # is implicitly betting on.  Always emitted; the main forward path
        # never consumes this key.  Untradeable rows get the raw return value
        # (the trainer applies a tradeable-mask before computing MSE).
        next_day_returns = self._stk_log_ret[day_idx].astype(np.float32)

        return {
            "features": feat_arr,
            "mask": mask.astype(np.int8),
            "sector_ids": self._stk_sector_ids[day_idx].astype(np.int32),
            "portfolio": portfolio,
            "cash": np.array(self._cash, dtype=np.float32),
            "nav": np.array(current_nav, dtype=np.float32),
            "t_frac": np.array(self._t / self._episode_length, dtype=np.float32),
            "recent_return_1d": np.array(recent_return_1d, dtype=np.float32),
            "recent_vol_20d": np.array(recent_vol_20d, dtype=np.float32),
            "nav_log_progress": np.array(nav_log_progress, dtype=np.float32),
            "regime": self._regime_features[day_idx].astype(np.float32),
            "next_day_returns": next_day_returns,
        }

    # ── fast array accessors (vectorised — kept for backward compatibility) ───

    def _price_at(self, day_idx: int, col: str) -> np.ndarray:
        if col == "open":
            return np.asarray(self._stk_open[day_idx], dtype=np.float64).copy()
        if col == "close":
            return np.asarray(self._stk_close[day_idx], dtype=np.float64).copy()
        if col == "atr_14":
            return np.asarray(self._stk_atr[day_idx], dtype=np.float64).copy()
        if col == "dollar_volume_20":
            return np.asarray(self._stk_dollar_vol[day_idx], dtype=np.float64).copy()
        # Fallback: dict path (rarely needed)
        N = len(self._universe)
        arr = np.zeros(N, dtype=np.float64)
        for ni, ticker in enumerate(self._universe):
            key = (col, ticker)
            if key in self._arrays and 0 <= day_idx < len(self._arrays[key]):
                arr[ni] = float(self._arrays[key][day_idx])
        return arr

    def _col_at(self, day_idx: int, col: str) -> np.ndarray:
        return self._price_at(day_idx, col)

    def _mask_at(self, day_idx: int) -> np.ndarray:
        return np.asarray(self._stk_mask[day_idx], dtype=np.bool_).copy()

    def _sector_ids_at(self, day_idx: int) -> np.ndarray:
        return np.asarray(self._stk_sector_ids[day_idx], dtype=np.int32).copy()


# ── module helpers ────────────────────────────────────────────────────────────


def _build_ticker_arrays(
    panel: pl.DataFrame,
    universe: list[str],
    dates: list[Any],
    feat_cols: list[str],
) -> dict[tuple[str, str], np.ndarray]:
    """Build (T,) dense arrays per (col, ticker) for fast random access."""
    T = len(dates)
    date_index: dict[Any, int] = {d: i for i, d in enumerate(dates)}

    extra = ["open", "close", "is_tradeable", "sector_id", "atr_14", "dollar_volume_20"]
    all_cols = list(dict.fromkeys(feat_cols + extra))

    arrays: dict[tuple[str, str], np.ndarray] = {}

    for ticker in universe:
        sub = panel.filter(pl.col("ticker") == ticker).sort("date")
        if sub.is_empty():
            continue
        sub_dates = sub["date"].to_list()
        for col in all_cols:
            if col not in sub.columns:
                continue
            if col == "is_tradeable":
                dtype: type[np.generic] = np.bool_
            elif col == "sector_id":
                dtype = np.int32
            else:
                dtype = np.float32
            full: np.ndarray = np.zeros(T, dtype=dtype)
            vals = sub[col].to_numpy()
            for row_i, d in enumerate(sub_dates):
                idx = date_index.get(d)
                if idx is not None:
                    full[idx] = vals[row_i]
            arrays[(col, ticker)] = full

    return arrays


def masked_softmax(
    logits: np.ndarray,
    mask: np.ndarray,
    max_weight: float,
) -> np.ndarray:
    """Apply masked softmax then cap-and-renormalize.

    Parameters
    ----------
    logits:
        Shape (N+1,). Index 0 is cash (always allowed).
    mask:
        Shape (N,) bool. True = equity ticker is tradeable today.
    max_weight:
        Per-name weight cap applied after softmax.
    """
    full_mask = np.concatenate([[True], mask])  # cash always allowed
    clipped = np.clip(logits, -10.0, 10.0)
    clipped = np.where(full_mask, clipped, -np.inf)

    valid = full_mask & np.isfinite(clipped)
    if not np.any(valid):
        w = np.zeros(len(logits))
        w[0] = 1.0
        return w

    shift = np.max(clipped[valid])
    exp_l = np.where(valid, np.exp(np.clip(clipped - shift, -500.0, 0.0)), 0.0)
    total = exp_l.sum()
    if total < 1e-12:
        w = np.zeros(len(logits))
        w[0] = 1.0
        return w

    w = exp_l / total
    return _cap_and_renormalize(w, max_weight)


def _cap_and_renormalize(w: np.ndarray, cap: float) -> np.ndarray:
    """Cap equity weights (index 1+) at `cap`; redistribute excess to all uncapped positions.

    Cash (index 0) is never capped — it can absorb the full portfolio.
    """
    w = w.copy()
    for _ in range(len(w) + 1):
        # Only equity positions are subject to the per-name cap
        equity_over = np.zeros(len(w), dtype=bool)
        equity_over[1:] = w[1:] > cap
        if not np.any(equity_over):
            break
        excess = float(np.sum(w[equity_over] - cap))
        w[equity_over] = cap
        # Free positions: cash + any uncapped equity
        free = ~equity_over
        free_sum = float(w[free].sum())
        if free_sum < 1e-12:
            break
        w[free] += excess * (w[free] / free_sum)
    return w
