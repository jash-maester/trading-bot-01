"""PanelTradingEnv — a Gymnasium env that steps one trading day at a time."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, SupportsFloat

import numpy as np
import polars as pl
from gymnasium import Env, spaces

from trader.data.regime_features import REGIME_DIM, compute_regime_features
from trader.env.costs import CostModel, ZerodhaEquityDeliveryCostModel
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
    ) -> None:
        super().__init__()

        self._cost_model: CostModel = cost_model or ZerodhaEquityDeliveryCostModel()
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
        day_idx = self._start_idx + self._t

        mask = self._mask_at(day_idx)
        target_w = masked_softmax(action.astype(np.float64), mask, self._max_weight)

        eq_target_frac = target_w[1:]  # (N,)

        opens = self._price_at(day_idx, "open")
        closes = self._price_at(day_idx, "close")
        prev_closes = self._price_at(day_idx - 1, "close")
        atrs = self._col_at(day_idx, "atr_14")
        dollar_vol = self._col_at(day_idx, "dollar_volume_20")

        # NAV before today's trades (marked at yesterday's close)
        current_nav = max(self._cash + float(np.sum(self._shares * prev_closes)), 1e-8)

        # Target integer shares
        target_equity_value = current_nav * np.where(mask, eq_target_frac, 0.0)
        target_shares = np.where(
            opens > 0,
            np.floor(target_equity_value / np.maximum(opens, 1e-8)),
            0.0,
        )
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
        traded = np.abs(delta_shares) >= 0.5
        trade_val = np.where(traded, np.abs(delta_shares) * fill_prices, 0.0)
        is_buy = delta_shares > 0
        n_sold = np.where(
            traded & (~is_buy) & (self._shares > 0), 1, 0
        ).astype(np.int64)

        cost_per_leg = self._cost_model.cost_vec(trade_val, is_buy, n_sold)
        signed_cash = np.where(traded, -delta_shares * fill_prices, 0.0)
        self._cash += float(signed_cash.sum() - cost_per_leg.sum())
        costs_paid = float(cost_per_leg.sum())

        self._shares = target_shares

        # Mark-to-close
        new_nav = max(self._cash + float(np.sum(self._shares * closes)), 1e-8)
        log_return = math.log(new_nav / max(self._prev_nav, 1e-8))

        # Turnover (equity weight change)
        prev_eq_w = (self._shares * closes) / max(self._prev_nav, 1e-8)
        new_eq_w = (self._shares * closes) / max(new_nav, 1e-8)
        turnover = float(np.sum(np.abs(new_eq_w - prev_eq_w)))

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
            "log_return": log_return,
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
