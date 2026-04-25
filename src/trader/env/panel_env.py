"""PanelTradingEnv — a Gymnasium env that steps one trading day at a time."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, SupportsFloat

import numpy as np
import polars as pl
from gymnasium import Env, spaces

from trader.env.costs import CostModel, ZerodhaEquityDeliveryCostModel
from trader.env.reward import DifferentialSharpe, RewardFn

_SLIPPAGE_K = 0.1


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
        seed: int | None = None,
    ) -> None:
        super().__init__()

        self._cost_model: CostModel = cost_model or ZerodhaEquityDeliveryCostModel()
        self._reward_fn: RewardFn = reward_fn or DifferentialSharpe()
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
            }
        )
        self.action_space = spaces.Box(-10.0, 10.0, (N + 1,), np.float32)

        # Episode state (reset before each episode)
        self._cash = initial_cash
        self._shares = np.zeros(N, dtype=np.float64)
        self._prev_nav = initial_cash
        self._t = 0
        self._start_idx = lookback  # safe default

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

        costs_paid = 0.0
        for i in range(len(self._universe)):
            if abs(delta_shares[i]) < 0.5:
                continue
            trade_val = abs(delta_shares[i]) * fill_prices[i]
            is_buy = delta_shares[i] > 0
            n_sold = 1 if (not is_buy and self._shares[i] > 0) else 0
            c = self._cost_model.cost(trade_val, is_buy, n_scrips_sold=n_sold)
            signed_cash = -delta_shares[i] * fill_prices[i]  # positive when selling
            self._cash += signed_cash - c
            costs_paid += c

        self._shares = target_shares

        # Mark-to-close
        new_nav = max(self._cash + float(np.sum(self._shares * closes)), 1e-8)
        log_return = math.log(new_nav / max(self._prev_nav, 1e-8))

        # Turnover (equity weight change)
        prev_eq_w = (self._shares * closes) / max(self._prev_nav, 1e-8)
        new_eq_w = (self._shares * closes) / max(new_nav, 1e-8)
        turnover = float(np.sum(np.abs(new_eq_w - prev_eq_w)))

        reward = float(self._reward_fn(log_return)) - self._turnover_penalty * turnover
        self._prev_nav = new_nav

        self._t += 1
        truncated = self._t >= self._episode_length
        terminated = new_nav < self._initial_cash * 0.10

        obs = self._build_obs()

        sector_exp: dict[int, float] = {}
        for i, sid in enumerate(self._sector_ids_at(day_idx)):
            w = float(self._shares[i] * closes[i]) / max(new_nav, 1e-8)
            sector_exp[int(sid)] = sector_exp.get(int(sid), 0.0) + w

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
        N = len(self._universe)
        F = len(self._feat_cols)

        feat_arr = np.zeros((self._lookback, N, F), dtype=np.float32)
        start = day_idx - self._lookback
        for fi, col in enumerate(self._feat_cols):
            for ni, ticker in enumerate(self._universe):
                key = (col, ticker)
                if key in self._arrays:
                    feat_arr[:, ni, fi] = self._arrays[key][start:day_idx]

        mask = self._mask_at(day_idx)
        prev_closes = self._price_at(day_idx - 1, "close")
        current_nav = max(self._cash + float(np.sum(self._shares * prev_closes)), 1e-8)
        eq_w = (self._shares * prev_closes) / current_nav
        portfolio = np.concatenate([[self._cash / current_nav], eq_w]).astype(
            np.float32
        )
        portfolio = np.clip(portfolio, 0.0, 1.0)

        return {
            "features": feat_arr,
            "mask": mask.astype(np.int8),
            "sector_ids": self._sector_ids_at(day_idx).astype(np.int32),
            "portfolio": portfolio,
            "cash": np.array(self._cash, dtype=np.float32),
            "nav": np.array(current_nav, dtype=np.float32),
            "t_frac": np.array(self._t / self._episode_length, dtype=np.float32),
        }

    # ── fast array accessors ──────────────────────────────────────────────────

    def _price_at(self, day_idx: int, col: str) -> np.ndarray:
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
        N = len(self._universe)
        arr = np.zeros(N, dtype=bool)
        for ni, ticker in enumerate(self._universe):
            key = ("is_tradeable", ticker)
            if key in self._arrays and 0 <= day_idx < len(self._arrays[key]):
                arr[ni] = bool(self._arrays[key][day_idx])
        return arr

    def _sector_ids_at(self, day_idx: int) -> np.ndarray:
        N = len(self._universe)
        arr = np.zeros(N, dtype=np.int32)
        for ni, ticker in enumerate(self._universe):
            key = ("sector_id", ticker)
            if key in self._arrays and 0 <= day_idx < len(self._arrays[key]):
                arr[ni] = int(self._arrays[key][day_idx])
        return arr


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
