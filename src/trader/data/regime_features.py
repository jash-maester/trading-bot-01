"""Market-regime features — Phase 1 of MLP architecture upgrade.

The walk-forward MLP results revealed a `corr(val, test) = -0.86` pathology:
the policy specialises to whichever regime dominates the train+val window,
and the very next year (test) tends to be in the opposite regime, so the
specialised policy fails.  The fix is to give the model an *exogenous*
regime signal it can condition on, via FiLM modulation in the encoder.

This module computes a daily market-regime vector from the cross-section
of stock returns + tradeability mask.  Six scalars per trading day:

    mkt_vol_20d        annualised std of the equal-weight benchmark
                       log-return over the last 20 trading days
    mkt_breadth_20d    fraction of stocks with cumulative log_return_20d > 0
                       (using the per-day cross-section)
    mkt_dispersion_20d cross-sectional std of log_return_1d, mean of last 20d
                       (how much stocks disagree)
    mkt_trend_20d      cumulative equal-weight log return over last 20 days
    mkt_acceleration   `mkt_trend_20d` minus `mkt_trend_60d` — regime-change
                       indicator (positive when momentum is accelerating)
    mkt_vol_of_vol_60d 60-day rolling std of mkt_vol_20d — measures whether
                       the vol regime is itself stable or shifting

Strictly backward-looking: ``regime_features[t]`` is computed from data
through ``log_return_1d[..t-1]`` (i.e., information known at the **start**
of trading day t), matching the env's `features` lookback convention.

The features are computed once at env construction from the panel.  Per-day
lookup is then O(1).  For walk-forward, each window's panel produces its
own regime tensor and (mean, std) stats — no leakage.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import torch

# Public name list — referenced by tests, configs, and the model config.
REGIME_COLS: list[str] = [
    "mkt_vol_20d",
    "mkt_breadth_20d",
    "mkt_dispersion_20d",
    "mkt_trend_20d",
    "mkt_acceleration",
    "mkt_vol_of_vol_60d",
]

REGIME_DIM: int = len(REGIME_COLS)

_SQRT_252 = np.sqrt(252.0)
_VOL_WIN = 20
_TREND_WIN_SHORT = 20
_TREND_WIN_LONG = 60
_VOV_WIN = 60


def compute_regime_features(
    log_return_1d: np.ndarray,        # [T, N]  — daily log returns
    tradeable_mask: np.ndarray,       # [T, N]  — bool, True = tradeable
) -> np.ndarray:
    """Compute the [T, R] regime tensor from raw panel arrays.

    Output column order matches `REGIME_COLS`.  Row ``t`` reflects info
    available at the **start** of trading day t (i.e., uses returns
    through day ``t-1``).  Warm-up rows (insufficient history) are set
    to zero — they're outside the env's ``lookback`` start window.
    """
    if log_return_1d.shape != tradeable_mask.shape:
        raise ValueError(
            f"shape mismatch: log_return_1d {log_return_1d.shape} vs "
            f"tradeable_mask {tradeable_mask.shape}"
        )
    T, _ = log_return_1d.shape
    out = np.zeros((T, REGIME_DIM), dtype=np.float32)

    trd = tradeable_mask.astype(np.float64)
    rets = log_return_1d.astype(np.float64) * trd                   # zero out non-tradeable

    # Equal-weight cross-sectional benchmark return per day (only over tradeable).
    n_trd = np.maximum(trd.sum(axis=1), 1.0)
    bench = rets.sum(axis=1) / n_trd                                # [T]

    # Cross-sectional dispersion per day: weighted std of log_return_1d
    # across tradeable stocks (mean=bench[t]).  Use a small numerical
    # safeguard for zero-tradeable days (shouldn't happen in practice).
    dev2 = ((log_return_1d - bench[:, None]) ** 2) * trd
    csd_var = dev2.sum(axis=1) / n_trd
    csd_std_today = np.sqrt(np.maximum(csd_var, 0.0))               # [T]

    # ── Rolling helpers ────────────────────────────────────────────────────
    # All rolling windows are strictly backward-looking using a left shift:
    # regime[t] uses bench[t-W : t] (no day-t info).
    def _shift_then_roll_mean(x: np.ndarray, w: int) -> np.ndarray:
        """Mean of x[t-w:t] (length w window ending at t-1)."""
        if w <= 0:
            return np.zeros_like(x)
        # Cumulative sum trick for fast rolling mean.
        c = np.concatenate([[0.0], np.cumsum(x)])                   # length T+1
        # mean over [t-w, t-1] = (c[t] - c[t-w]) / w  for t ≥ w
        out = np.zeros_like(x)
        if T > w:
            out[w:] = (c[w:T] - c[: T - w]) / w
        return out

    def _shift_then_roll_std(x: np.ndarray, w: int) -> np.ndarray:
        """Std of x[t-w:t] (length w window ending at t-1)."""
        if w <= 0:
            return np.zeros_like(x)
        c1 = np.concatenate([[0.0], np.cumsum(x)])
        c2 = np.concatenate([[0.0], np.cumsum(x ** 2)])
        out = np.zeros_like(x)
        if T > w:
            mean = (c1[w:T] - c1[: T - w]) / w
            mean_sq = (c2[w:T] - c2[: T - w]) / w
            var = np.maximum(mean_sq - mean ** 2, 0.0)
            out[w:] = np.sqrt(var)
        return out

    def _shift_then_roll_sum(x: np.ndarray, w: int) -> np.ndarray:
        """Sum of x[t-w:t] (length w window ending at t-1)."""
        if w <= 0:
            return np.zeros_like(x)
        c = np.concatenate([[0.0], np.cumsum(x)])
        out = np.zeros_like(x)
        if T > w:
            out[w:] = c[w:T] - c[: T - w]
        return out

    # ── 1. mkt_vol_20d (annualised) ────────────────────────────────────────
    mkt_vol = _shift_then_roll_std(bench, _VOL_WIN) * _SQRT_252

    # ── 2. mkt_breadth_20d ────────────────────────────────────────────────
    # cumulative 20d log return per stock = sum of last 20 daily log returns.
    # We then look at the fraction of TRADEABLE stocks with positive cum ret.
    # Use the panel column "log_return_20d" if present is conceptually the
    # same; here we recompute from log_return_1d for self-containment.
    cumret_20 = np.zeros_like(log_return_1d)
    if T > _TREND_WIN_SHORT:
        # Per-stock 20d cumulative return = sum of last 20 daily log returns
        # ending day t-1.
        c = np.concatenate(
            [np.zeros((1, log_return_1d.shape[1])), np.cumsum(rets, axis=0)],
            axis=0,
        )
        cumret_20[_TREND_WIN_SHORT:] = (
            c[_TREND_WIN_SHORT:T] - c[: T - _TREND_WIN_SHORT]
        )
    pos = (cumret_20 > 0.0) & tradeable_mask
    n_pos = pos.sum(axis=1)
    breadth = n_pos / np.maximum(tradeable_mask.sum(axis=1), 1.0)
    breadth = breadth.astype(np.float64)
    # breadth above is "as of t" using rets up to t-1 ✓.

    # ── 3. mkt_dispersion_20d ─────────────────────────────────────────────
    # Average of the last 20 days' cross-sectional std, ending at t-1.
    mkt_disp = _shift_then_roll_mean(csd_std_today, _VOL_WIN)

    # ── 4. mkt_trend_20d (sum of last 20 daily benchmark returns, t-20..t-1).
    mkt_trend = _shift_then_roll_sum(bench, _TREND_WIN_SHORT)

    # ── 5. mkt_acceleration = trend_20 − trend_60 ─────────────────────────
    mkt_trend_60 = _shift_then_roll_sum(bench, _TREND_WIN_LONG)
    mkt_accel = mkt_trend - mkt_trend_60

    # ── 6. mkt_vol_of_vol_60d ─────────────────────────────────────────────
    # Std of the mkt_vol_20d series over the last 60 days, ending at t-1.
    # mkt_vol[t] uses bench[t-20:t] — so when we then take std of mkt_vol
    # over [t-60:t] we're not using future info.
    mkt_vov = _shift_then_roll_std(mkt_vol, _VOV_WIN)

    out[:, 0] = mkt_vol.astype(np.float32)
    out[:, 1] = breadth.astype(np.float32)
    out[:, 2] = mkt_disp.astype(np.float32)
    out[:, 3] = mkt_trend.astype(np.float32)
    out[:, 4] = mkt_accel.astype(np.float32)
    out[:, 5] = mkt_vov.astype(np.float32)
    # Replace any nans (e.g. from edge cases on day 0) with zeros.
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


# ── Stats: computed once on the training panel and frozen for val/test ────────


def compute_regime_stats(
    panel_path: Path,
) -> dict[str, tuple[float, float]]:
    """Compute (mean, std) for each regime feature on the **training** panel.

    The input is the same parquet panel used by `compute_feature_stats`.
    We extract `log_return_1d` and `is_tradeable` as [T, N] arrays, run
    `compute_regime_features`, and aggregate per-column mean/std excluding
    the warm-up zeros (where regime would otherwise be 0 by construction).
    """
    df = pl.read_parquet(panel_path).sort(["date", "ticker"])
    if "log_return_1d" not in df.columns or "is_tradeable" not in df.columns:
        raise ValueError(
            "panel must have 'log_return_1d' and 'is_tradeable' columns"
        )

    dates = df["date"].unique().sort().to_list()
    tickers = df["ticker"].unique().sort().to_list()
    T = len(dates)
    N = len(tickers)
    if T == 0 or N == 0:
        return {k: (0.0, 1.0) for k in REGIME_COLS}

    date_idx = {d: i for i, d in enumerate(dates)}
    tick_idx = {t: i for i, t in enumerate(tickers)}

    log_ret = np.zeros((T, N), dtype=np.float64)
    trd = np.zeros((T, N), dtype=bool)

    di = df["date"].to_list()
    ti = df["ticker"].to_list()
    lv = df["log_return_1d"].to_numpy()
    tv = df["is_tradeable"].to_numpy()
    for k in range(len(di)):
        i = date_idx.get(di[k])
        j = tick_idx.get(ti[k])
        if i is None or j is None:
            continue
        log_ret[i, j] = lv[k]
        trd[i, j] = bool(tv[k])

    regime = compute_regime_features(log_ret, trd)                  # [T, R]
    # Skip the first ``_VOV_WIN`` rows — they're warm-up zeros.
    valid = regime[_VOV_WIN:]
    stats: dict[str, tuple[float, float]] = {}
    for ri, name in enumerate(REGIME_COLS):
        col = valid[:, ri]
        if col.size == 0 or float(np.std(col)) < 1e-8:
            stats[name] = (0.0, 1.0)
        else:
            stats[name] = (float(np.mean(col)), float(np.std(col)))
    return stats


def regime_stats_to_tensors(
    stats: dict[str, tuple[float, float]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert a regime-stats dict into ordered ``(mean, std)`` 1-D tensors.

    Order matches `REGIME_COLS`.
    """
    means = torch.tensor(
        [stats.get(c, (0.0, 1.0))[0] for c in REGIME_COLS], dtype=torch.float32
    )
    stds = torch.tensor(
        [max(stats.get(c, (0.0, 1.0))[1], 1e-6) for c in REGIME_COLS],
        dtype=torch.float32,
    )
    return means, stds


def save_regime_stats(
    stats: dict[str, tuple[float, float]],
    path: Path,
) -> None:
    """Persist regime stats as JSON (mirrors `feature_stats.save_stats`)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump({k: list(v) for k, v in stats.items()}, f, indent=2, sort_keys=True)


def load_regime_stats(path: Path) -> dict[str, tuple[float, float]]:
    """Load regime stats from a JSON file produced by `save_regime_stats`."""
    with path.open() as f:
        raw = json.load(f)
    return {k: (float(v[0]), float(v[1])) for k, v in raw.items()}
