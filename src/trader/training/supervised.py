"""R4 — supervised training of the cross-sectional signal model.

Plain PyTorch.  No PPO, no env, no reward.  What lives here:

* **Targets** — forward log returns over ``t+1 .. t+h``, z-scored across the
  tradeable cross-section at each date.  Rows whose forward window runs off the
  end of a split, or whose stock is untradeable anywhere in that window, are
  *unlabelled* (NaN) and contribute nothing to the loss.  They are never
  zero-filled.
* **Feature-liveness gate** — every input column must have nonzero variance on
  the training split's tradeable rows, or training refuses to start and names
  the dead column.  ``beta_nifty_60d`` was constant 1.0 for months and nothing
  caught it; this does.
* **Training loop** — Adam, cosine or constant LR, early stopping on validation
  *rank IC* (not on the loss; the loss is not the objective).
* **Evaluation** — per-day Spearman IC per horizon, ICIR, hit rate, bootstrap
  95 % CI on the mean IC (resampling days), decile spread.  The R4 gate.
* **Walk-forward orchestration** — over ``compute_windows()`` with the purge
  the guard there insists on; feature stats are computed from each window's
  train split only (mirrors ``runner.py``).  Every window logs to MLflow.
* **Artefacts** — a pinned contract other agents build against:
  ``predictions.parquet``, ``embeddings.npy``, ``index.json`` (see
  :func:`write_artefacts`) and ``gate.json`` (see :func:`write_gate_json`).

Leakage discipline, stated once
-------------------------------
A prediction dated ``t`` is a function of feature rows ``t-L+1 .. t`` of *its
own split* and of normalisation stats from the *train* split.  Its label is a
function of ``log_return_1d`` at ``t+1 .. t+h`` only.  ``tests/unit/
test_supervised.py`` perturbs a future return and asserts the prediction at
``t`` is bit-identical.
"""
from __future__ import annotations

import copy
import json
import math
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch
from loguru import logger

from trader.data.feature_stats import compute_feature_stats, save_stats, stats_to_tensors
from trader.models.heads import aux_return_loss
from trader.models.signal import (
    DEFAULT_HORIZONS,
    SignalConfig,
    SignalModel,
    horizon_key,
    state_dict_sha256,
)
from trader.training.walk_forward import (
    WindowConfig,
    _average_ranks,
    _pearson_r,
    materialise_window,
)
from trader.utils.seeding import get_device, seed_everything

# ── The gate ───────────────────────────────────────────────────────────────────

#: Materiality floor on the mean of the per-window OOS rank ICs.  `12_gate_decision.md`.
GATE_MIN_MEAN_IC: float = 0.02
#: Fewest windows on which a window-level test means anything.  A t-test on two
#: or three numbers is a coin toss with a decimal point on it.
GATE_MIN_WINDOWS: int = 4
#: Fraction of windows whose OOS mean IC must be positive — a robustness check
#: that one strong era is not carrying the mean.
GATE_MIN_POSITIVE_FRACTION: float = 0.75
#: Names the rule that produced a verdict; written into gate.json so a consumer
#: can tell a verdict under this rule from one under the retired every-window rule.
GATE_RULE: str = "window-level-t/v2"

# Two-sided 95% Student-t critical values by degrees of freedom.  scipy is not
# a dependency (see `spearman`); these are the standard table values.  Above 30
# the df=30 value is used, which is conservative.
_T_CRIT_95: dict[int, float] = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
    8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145,
    15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056,
    27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
}


def t_critical_95(df: int) -> float:
    """Two-sided 95% Student-t critical value for ``df`` degrees of freedom."""
    if df < 1:
        raise ValueError(f"df must be >= 1, got {df}")
    return _T_CRIT_95[df] if df in _T_CRIT_95 else _T_CRIT_95[30]
# Minimum scorable OOS days per window before a verdict means anything.  Below
# this the mean IC is an artefact of the sample size: a single scorable day
# with IC 0.9 passed the gate outright before this floor existed.
GATE_MIN_SCORABLE_DAYS: int = 30
# The pinned `gate.json` contract names these two horizons explicitly.  A
# consumer reading the pinned schema must find both keys whatever was run.
PINNED_GATE_HORIZONS: tuple[int, ...] = (5, 20)


class DeadFeatureError(ValueError):
    """An input column has zero variance on the training split's tradeable rows.

    Its own type so the liveness gate cannot be confused with any other
    ``ValueError`` on the way out — this one must stop the run.
    """


# ── Feature liveness ───────────────────────────────────────────────────────────


def assert_feature_liveness(
    panel: pl.DataFrame,
    feature_cols: list[str],
    *,
    where: str = "train",
    min_std: float = 1e-8,
) -> dict[str, float]:
    """Refuse to proceed unless every feature varies on tradeable rows.

    Returns the per-column standard deviation so the caller can log it.
    Raises :class:`DeadFeatureError` naming the first dead column (and listing
    all of them) — a dead channel normalises to a constant bias into every
    convolution, and a model trained on it is not measuring what you think.
    """
    if "is_tradeable" not in panel.columns:
        raise ValueError("panel has no is_tradeable column")
    trd = panel.filter(pl.col("is_tradeable"))
    if trd.is_empty():
        raise DeadFeatureError(f"{where}: no tradeable rows at all — nothing to train on")

    stds: dict[str, float] = {}
    dead: list[str] = []
    for col in feature_cols:
        if col not in trd.columns:
            dead.append(f"{col} (missing)")
            stds[col] = float("nan")
            continue
        vals = trd[col].drop_nulls().drop_nans().to_numpy()
        s = float(np.std(vals)) if vals.size else 0.0
        stds[col] = s
        if not math.isfinite(s) or s < min_std:
            dead.append(col)
    if dead:
        raise DeadFeatureError(
            f"{where}: dead feature column(s) — zero variance over "
            f"{trd.shape[0]:,} tradeable rows: {', '.join(dead)}. "
            "A constant input is a bias term, not a feature (beta_nifty_60d was "
            "1.0 for months). Rebuild the panel; do not train on it."
        )
    return stds


# ── Dense panel tensors ────────────────────────────────────────────────────────


@dataclass
class PanelTensors:
    """One split of the panel as dense ``[T, N, ...]`` arrays.

    ``targets[h]`` is the cross-sectionally z-scored forward return (NaN where
    unlabelled); ``fwd_raw[h]`` is the un-standardised forward log return over
    the same window, used for the decile spread.  ``mask`` is ``is_tradeable``.
    """

    dates: list[date]
    tickers: list[str]
    feature_cols: list[str]
    features: np.ndarray            # [T, N, F] float32
    mask: np.ndarray                # [T, N]    bool
    targets: dict[int, np.ndarray]  # h -> [T, N] float32, NaN = unlabelled
    fwd_raw: dict[int, np.ndarray]  # h -> [T, N] float32, NaN = unlabelled
    #: Leading dates that are feature CONTEXT ONLY — never labelled, never
    #: predicted, never scored. Set from the `is_warmup` column written by
    #: `materialise_window`. See its docstring for why they exist.
    n_warmup: int = 0

    @property
    def n_days(self) -> int:
        return len(self.dates)

    @property
    def n_tickers(self) -> int:
        return len(self.tickers)


def _dense_column(
    panel: pl.DataFrame,
    col: str,
    dates: list[date],
    tickers: list[str],
    *,
    fill: float,
    dtype: type[np.generic],
) -> np.ndarray:
    """Pivot one panel column into a ``[T, N]`` array in (`dates`, `tickers`) order.

    Tickers absent from the panel (the on-disk panels are stale — 163 tickers
    against 504 in ``active_tickers()``) become constant `fill` columns; for
    ``is_tradeable`` that is ``False``, so they never enter a loss or a metric.

    Everything is pivoted through Float64 before `fill`.  ``is_tradeable`` is a
    Boolean column and ``fill_null`` on a Boolean with a float refuses ("invalid
    or ambiguous dtypes"), so the cast is not cosmetic — without it every panel
    with a gap in it raises here.

    Nulls **and** NaNs both take `fill`.  On a tradeable row that would be a
    silent zero-fill, which is exactly what this module refuses to do, so
    :func:`build_panel_tensors` checks tradeable rows for non-finite values
    separately and raises.  Here the fill only ever lands on masked rows.
    """
    wide = (
        panel.select(["date", "ticker", col])
        .pivot(on="ticker", index="date", values=col)
    )
    spine = pl.DataFrame({"date": dates})
    wide = spine.join(wide, on="date", how="left")
    exprs: list[pl.Expr] = []
    for t in tickers:
        if t in wide.columns:
            exprs.append(
                pl.col(t).cast(pl.Float64).fill_nan(fill).fill_null(fill).alias(t)
            )
        else:
            exprs.append(pl.lit(fill, dtype=pl.Float64).alias(t))
    return wide.select(exprs).to_numpy().astype(dtype)


def _assert_finite_on_tradeable(panel: pl.DataFrame, cols: list[str]) -> None:
    """Raise unless every named column is finite on every ``is_tradeable`` row."""
    trd = panel.filter(pl.col("is_tradeable"))
    if trd.is_empty():
        return
    bad: dict[str, int] = {}
    for col in cols:
        n = int(
            trd.select(
                (~pl.col(col).cast(pl.Float64).is_finite().fill_null(False))
                .sum()
                .alias("n")
            ).item()
        )
        if n:
            bad[col] = n
    if bad:
        detail = ", ".join(f"{c}: {n:,}" for c, n in bad.items())
        raise ValueError(
            f"non-finite value(s) on tradeable rows — {detail}. These would be "
            "zero-filled into the encoder (or into a label), which is a "
            "fabricated observation. Fix the panel; do not train on it."
        )


def forward_returns(
    log_return_1d: np.ndarray,   # [T, N]
    mask: np.ndarray,            # [T, N] bool
    horizon: int,
) -> np.ndarray:
    """``fwd_h[t, i] = sum(log_return_1d[t+1 .. t+h, i])``, NaN where unlabelled.

    A row is labelled only when the stock is tradeable at ``t`` (so it belongs
    to the cross-section) **and** on every day of ``t+1 .. t+h`` (so the sum is
    a real holding-period return, not a run of sentinel zeros through a
    suspension or delisting).  The last ``h`` rows of any split are unlabelled
    by construction — their window runs off the end.
    """
    if log_return_1d.shape != mask.shape:
        raise ValueError(
            f"shape mismatch: log_return_1d {log_return_1d.shape} vs mask {mask.shape}"
        )
    if horizon <= 0:
        raise ValueError(f"horizon must be positive, got {horizon}")
    T, N = log_return_1d.shape
    out = np.full((T, N), np.nan, dtype=np.float32)
    if T <= horizon:
        return out

    m = mask.astype(bool)
    r = np.where(m, log_return_1d.astype(np.float64), 0.0)
    zero = np.zeros((1, N), dtype=np.float64)
    cum_r = np.vstack([zero, np.cumsum(r, axis=0)])                 # cum_r[k] = Σ r[0..k-1]
    cum_m = np.vstack([zero, np.cumsum(m.astype(np.float64), axis=0)])

    n_lab = T - horizon
    # Σ r[t+1 .. t+h] = cum_r[t+h+1] - cum_r[t+1]
    fwd = cum_r[horizon + 1 : T + 1] - cum_r[1 : n_lab + 1]         # [n_lab, N]
    all_trd = (cum_m[horizon + 1 : T + 1] - cum_m[1 : n_lab + 1]) == horizon
    ok = all_trd & m[:n_lab]
    out[:n_lab] = np.where(ok, fwd, np.nan).astype(np.float32)
    return out


def cross_sectional_zscore(
    raw: np.ndarray,             # [T, N], NaN = unlabelled
    min_cross_section: int = 10,
) -> np.ndarray:
    """Per-date z-score over the labelled cross-section.

    This is what turns a return-regression into a *ranking* target: the model
    is asked where a stock sits relative to its peers that day, not what the
    market did.  Dates with fewer than `min_cross_section` labelled stocks, or
    a degenerate spread, are left entirely unlabelled.
    """
    valid = np.isfinite(raw)
    n = valid.sum(axis=1)                                            # [T]
    x = np.where(valid, raw.astype(np.float64), 0.0)
    denom = np.maximum(n, 1).astype(np.float64)
    mean = x.sum(axis=1) / denom
    var = (((x - mean[:, None]) ** 2) * valid).sum(axis=1) / denom
    std = np.sqrt(var)
    ok_row = (n >= max(min_cross_section, 2)) & (std > 1e-12)
    z = np.where(
        valid & ok_row[:, None],
        (raw - mean[:, None]) / np.where(ok_row, std, 1.0)[:, None],
        np.nan,
    )
    return z.astype(np.float32)


def build_panel_tensors(
    panel: pl.DataFrame,
    tickers: list[str],
    feature_cols: list[str],
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    *,
    min_cross_section: int = 10,
) -> PanelTensors:
    """Dense arrays for one split, with targets computed from *this split only*.

    Targets never look past the split's last row, so a train-split label
    cannot contain a purge-gap or validation return.
    """
    for col in ("date", "ticker", "is_tradeable", "log_return_1d"):
        if col not in panel.columns:
            raise ValueError(f"panel is missing required column {col!r}")
    missing = [c for c in feature_cols if c not in panel.columns]
    if missing:
        raise ValueError(f"panel is missing feature column(s): {missing}")

    dates = sorted(panel["date"].unique().to_list())
    mask = _dense_column(panel, "is_tradeable", dates, tickers, fill=0.0, dtype=np.bool_)
    feats = np.stack(
        [
            _dense_column(panel, c, dates, tickers, fill=0.0, dtype=np.float32)
            for c in feature_cols
        ],
        axis=2,
    )                                                                # [T, N, F]
    lr = _dense_column(panel, "log_return_1d", dates, tickers, fill=0.0, dtype=np.float32)

    # A null or NaN on a tradeable row is filled with 0.0 by `_dense_column` —
    # a fabricated observation the model would learn from, or a fabricated
    # return a label would be built on.  Check the *source* rows, before the
    # fill has hidden them.  (Masked rows may be null; nothing reads them.)
    _assert_finite_on_tradeable(panel, [*feature_cols, "log_return_1d"])

    # Leading feature-context days, if `materialise_window` prepended any. They
    # must be contiguous at the front: anything else means the column was built
    # by something other than the warm-up path and is not safe to trust.
    n_warmup = 0
    if "is_warmup" in panel.columns:
        warm_dates = set(
            panel.filter(pl.col("is_warmup"))["date"].unique().to_list()
        )
        if warm_dates:
            n_warmup = len(warm_dates)
            if set(dates[:n_warmup]) != warm_dates:
                raise ValueError(
                    "is_warmup rows are not a contiguous prefix of the split's "
                    "dates; refusing to guess which days are context."
                )

    targets: dict[int, np.ndarray] = {}
    fwd_raw: dict[int, np.ndarray] = {}
    for h in horizons:
        raw = forward_returns(lr, mask, h)
        fwd_raw[h] = raw
        tgt = cross_sectional_zscore(raw, min_cross_section)
        # Blank the warm-up rows outright. `_scorable_days` already skips them,
        # but a label that cannot be read is safer than one that must not be:
        # this path is the one where a mistake IS contamination.
        if n_warmup:
            raw[:n_warmup, :] = np.nan
            tgt[:n_warmup, :] = np.nan
        targets[h] = tgt
    return PanelTensors(
        dates=dates,
        tickers=list(tickers),
        feature_cols=list(feature_cols),
        features=feats,
        mask=mask,
        targets=targets,
        fwd_raw=fwd_raw,
        n_warmup=n_warmup,
    )


def _gather_windows(features: torch.Tensor, idx: np.ndarray, lookback: int) -> torch.Tensor:
    """``[B, L, N, F]`` windows ending at (and including) each day index in `idx`."""
    return torch.stack([features[t - lookback + 1 : t + 1] for t in idx.tolist()])


# ── Metrics ────────────────────────────────────────────────────────────────────


def spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    """Spearman rank correlation without scipy (reuses walk_forward's tie-aware ranks)."""
    if x.shape != y.shape or x.ndim != 1 or x.size < 3:
        return None
    return _pearson_r(_average_ranks(x.astype(np.float64)), _average_ranks(y.astype(np.float64)))


@dataclass
class DailyScores:
    """Per-day IC and decile spread for one horizon on one split."""

    day_index: np.ndarray        # [n_days] int — index into the split's dates
    ic: np.ndarray               # [n_days]
    decile_spread: np.ndarray    # [n_days] mean raw fwd return, top decile − bottom decile
    # Days that existed in the split but could not be scored.  A model that
    # collapses to a constant cross-section on half its days would otherwise
    # report the other half's mean IC with no trace of the omission.
    n_skipped_thin: int = 0      # cross-section below `min_cross_section`
    n_skipped_degenerate: int = 0  # Spearman undefined (constant pred or target)


def daily_scores(
    pred: np.ndarray,            # [T, N], NaN where no prediction
    target: np.ndarray,          # [T, N], NaN where unlabelled (any monotone transform works)
    fwd_raw: np.ndarray,         # [T, N], raw forward return for the spread
    *,
    min_cross_section: int = 10,
) -> DailyScores:
    """Spearman IC and top-minus-bottom-decile spread for every scorable day."""
    valid = np.isfinite(pred) & np.isfinite(target) & np.isfinite(fwd_raw)
    days: list[int] = []
    ics: list[float] = []
    spreads: list[float] = []
    n_thin = 0
    n_degenerate = 0
    for t in range(pred.shape[0]):
        v = valid[t]
        n = int(v.sum())
        if n < max(min_cross_section, 3):
            n_thin += 1
            continue
        p = pred[t, v]
        rho = spearman(p, target[t, v])
        if rho is None:
            n_degenerate += 1
            continue
        order = np.argsort(p, kind="stable")
        k = max(1, n // 10)
        r = fwd_raw[t, v].astype(np.float64)
        spread = float(r[order[-k:]].mean() - r[order[:k]].mean())
        days.append(t)
        ics.append(rho)
        spreads.append(spread)
    return DailyScores(
        day_index=np.asarray(days, dtype=np.int64),
        ic=np.asarray(ics, dtype=np.float64),
        decile_spread=np.asarray(spreads, dtype=np.float64),
        n_skipped_thin=n_thin,
        n_skipped_degenerate=n_degenerate,
    )


def bootstrap_mean_ci(
    values: np.ndarray,
    *,
    block: int = 1,
    n_boot: int = 1000,
    rng_seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile CI on the mean of a **serially correlated** daily series.

    ``block`` is the **overlap length**: pass the forward-return horizon ``h``.
    Consecutive daily ICs are not independent for two compounding reasons — the
    ``h``-day forward returns of days ``t`` and ``t+1`` share ``h-1`` days of
    price path, and any fitted signal is persistent day to day (a 60-day
    lookback TCN scores consecutive windows that share 59 of 60 rows).  This
    resamples contiguous **blocks** of ``block`` days (moving-block bootstrap,
    wrapped so every start index is equally likely) rather than individual
    days, so the resamples carry the same within-block dependence as the data.

    Why this is not a detail
    -----------------------
    The i.i.d. version this replaced was measured against a strict null
    (i.i.d. returns, a predictor carrying zero information, 150 names x 250 OOS
    days): with a *persistent* predictor its 95% CI excluded zero in **33.7%**
    of trials at ``h=5`` and **63.0%** at ``h=20``, against a nominal 5%.
    Lag-1 autocorrelation of the daily IC series was +0.697 and +0.830.  With a
    fresh-random-each-day predictor the same test gave 7.3% / 6.3% and ACF
    ~0.00 — i.e. the i.i.d. bootstrap is only valid for a predictor with no
    day-to-day persistence, which the model this gate guards never is.
    ``gate_verdict``'s "CI excludes zero" condition therefore rests on this
    function being block-aware; with ``block=1`` it is not a significance test.

    ``block=1`` reproduces the old i.i.d. behaviour exactly and is kept only so
    the function can be exercised against it.
    """
    n = int(values.size)
    if n == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(rng_seed)
    b = max(1, min(int(block), n))
    if b == 1:
        samples = rng.choice(values, size=(n_boot, n), replace=True).mean(axis=1)
    else:
        n_blocks = -(-n // b)                       # ceil, then trim to n
        starts = rng.integers(0, n, size=(n_boot, n_blocks))
        idx = (starts[:, :, None] + np.arange(b)[None, None, :]) % n
        samples = values[idx.reshape(n_boot, -1)[:, :n]].mean(axis=1)
    lo, hi = np.percentile(samples, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


@dataclass
class HorizonMetrics:
    """The R4 summary for one horizon on one split (or pooled across windows)."""

    horizon: int
    n_days: int
    mean_ic: float
    std_ic: float
    icir: float
    t_stat: float
    hit_rate: float
    ci_lo: float
    ci_hi: float
    decile_spread: float
    # Persistence diagnostics for the CI above.  `ic_acf_lag1` is the lag-1
    # autocorrelation of the daily IC series and `n_eff` the effective sample
    # size n*(1-r)/(1+r).  They exist because the CI's calibration degrades
    # with persistence and a reader must be able to see how much: see
    # `bootstrap_mean_ci` and `gate_verdict`.
    ic_acf_lag1: float = float("nan")
    n_eff: float = float("nan")
    n_skipped_thin: int = 0
    n_skipped_degenerate: int = 0
    daily: DailyScores = field(repr=False, default_factory=lambda: DailyScores(
        np.zeros(0, dtype=np.int64), np.zeros(0), np.zeros(0)
    ))

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("daily")
        return d


def summarise_daily(
    horizon: int,
    daily: DailyScores,
    *,
    n_boot: int = 1000,
    rng_seed: int = 0,
) -> HorizonMetrics:
    ic = daily.ic
    n = int(ic.size)
    if n == 0:
        nan = float("nan")
        return HorizonMetrics(
            horizon, 0, nan, nan, nan, nan, nan, nan, nan, nan,
            ic_acf_lag1=nan, n_eff=nan,
            n_skipped_thin=daily.n_skipped_thin,
            n_skipped_degenerate=daily.n_skipped_degenerate,
            daily=daily,
        )
    mean = float(ic.mean())
    std = float(ic.std(ddof=1)) if n > 1 else float("nan")
    icir = mean / std if n > 1 and std > 0 else float("nan")
    t_stat = mean / (std / math.sqrt(n)) if n > 1 and std > 0 else float("nan")
    # Block length = the horizon: that is the overlap between consecutive
    # forward returns, and the scale of the signal's own persistence.
    lo, hi = bootstrap_mean_ci(ic, block=horizon, n_boot=n_boot, rng_seed=rng_seed)
    acf1 = float(np.corrcoef(ic[:-1], ic[1:])[0, 1]) if n > 2 and std > 0 else float("nan")
    n_eff = (
        float(n * (1.0 - acf1) / (1.0 + acf1))
        if math.isfinite(acf1) and -1.0 < acf1 < 1.0
        else float(n)
    )
    return HorizonMetrics(
        horizon=horizon,
        n_days=n,
        mean_ic=mean,
        std_ic=std,
        icir=icir,
        t_stat=t_stat,
        hit_rate=float((ic > 0).mean()),
        ci_lo=lo,
        ci_hi=hi,
        decile_spread=float(daily.decile_spread.mean()),
        ic_acf_lag1=acf1,
        n_eff=n_eff,
        n_skipped_thin=daily.n_skipped_thin,
        n_skipped_degenerate=daily.n_skipped_degenerate,
        daily=daily,
    )


def evaluate_predictions(
    preds: dict[int, np.ndarray],
    tensors: PanelTensors,
    *,
    min_cross_section: int = 10,
    n_boot: int = 1000,
    rng_seed: int = 0,
) -> dict[int, HorizonMetrics]:
    """Rank-IC report per horizon for a set of ``[T, N]`` predictions on a split."""
    out: dict[int, HorizonMetrics] = {}
    for h, p in preds.items():
        d = daily_scores(
            p, tensors.targets[h], tensors.fwd_raw[h], min_cross_section=min_cross_section
        )
        out[h] = summarise_daily(h, d, n_boot=n_boot, rng_seed=rng_seed)
    return out


def pool_daily(
    per_window: list[dict[int, HorizonMetrics]],
    *,
    n_boot: int = 1000,
    rng_seed: int = 0,
) -> dict[int, HorizonMetrics]:
    """Concatenate day-level scores across windows and summarise once."""
    horizons: list[int] = sorted({h for w in per_window for h in w})
    pooled: dict[int, HorizonMetrics] = {}
    for h in horizons:
        parts = [w[h].daily for w in per_window if h in w]
        daily = DailyScores(
            day_index=np.concatenate([p.day_index for p in parts]),
            ic=np.concatenate([p.ic for p in parts]),
            decile_spread=np.concatenate([p.decile_spread for p in parts]),
            n_skipped_thin=sum(p.n_skipped_thin for p in parts),
            n_skipped_degenerate=sum(p.n_skipped_degenerate for p in parts),
        )
        pooled[h] = summarise_daily(h, daily, n_boot=n_boot, rng_seed=rng_seed)
    return pooled


# ── The gate verdict ───────────────────────────────────────────────────────────


@dataclass
class WindowLevelStats:
    """The window-level test behind a verdict, for one horizon.

    Each window's OOS mean rank IC is one observation.  Windows' test periods do
    not overlap, so these are the closest thing to independent draws the
    walk-forward produces, and a plain t-test on them is honest about the
    within-window autocorrelation in a way no daily-level interval can be.
    """

    n_windows: int
    mean_ic: float
    sd_ic: float
    t_stat: float
    t_crit: float
    n_positive: int
    positive_fraction: float
    min_window_ic: float
    max_window_ic: float
    window_ics: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_windows": int(self.n_windows),
            "mean_ic": _json_float(self.mean_ic),
            "sd_ic": _json_float(self.sd_ic),
            "t_stat": _json_float(self.t_stat),
            "t_crit": _json_float(self.t_crit),
            "n_positive": int(self.n_positive),
            "positive_fraction": _json_float(self.positive_fraction),
            "min_window_ic": _json_float(self.min_window_ic),
            "max_window_ic": _json_float(self.max_window_ic),
            "window_ics": {k: _json_float(v) for k, v in self.window_ics.items()},
        }


@dataclass
class GateResult:
    passed: bool
    min_mean_ic: float
    per_horizon: dict[int, bool]
    reasons: list[str]
    #: The window-level statistics the verdict rests on, per horizon.
    window_level: dict[int, WindowLevelStats] = field(default_factory=dict)
    #: The retired every-window rule, kept as a DIAGNOSTIC: would each window
    #: individually have cleared mean IC > floor AND ci_lo > 0?  Reported, never
    #: decisive.  See `12_gate_decision.md` for why it was retired.
    strict_every_window: dict[int, bool] = field(default_factory=dict)
    strict_reasons: list[str] = field(default_factory=list)
    #: Positive statements of what a passing horizon showed, so a PASS carries
    #: its evidence the way a FAIL carries its reasons.
    evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "rule": GATE_RULE,
            "min_mean_ic": self.min_mean_ic,
            "per_horizon": {str(h): v for h, v in self.per_horizon.items()},
            "reasons": list(self.reasons),
            "evidence": list(self.evidence),
            "window_level": {str(h): w.to_dict() for h, w in self.window_level.items()},
            "strict_every_window": {str(h): v for h, v in self.strict_every_window.items()},
            "strict_reasons": list(self.strict_reasons),
        }


def gate_verdict(
    per_window_test: dict[str, dict[int, HorizonMetrics]],
    *,
    min_mean_ic: float = GATE_MIN_MEAN_IC,
    min_days: int = GATE_MIN_SCORABLE_DAYS,
    min_windows: int = GATE_MIN_WINDOWS,
    min_positive_fraction: float = GATE_MIN_POSITIVE_FRACTION,
) -> GateResult:
    """PASS iff some horizon clears a **window-level** test.

    The unit of evidence is a walk-forward window's OOS mean rank IC.  Per
    horizon, over the windows with a usable OOS record:

    * at least ``min_windows`` such windows (data-integrity floor);
    * every window has at least ``min_days`` scorable days — a window with
      fewer is a data problem and fails the horizon outright, as before;
    * the **mean of the window ICs** exceeds ``min_mean_ic`` (materiality:
      is the effect large enough to survive costs at all);
    * a one-sample t-test on the window ICs clears the two-sided 95% critical
      value for ``n_windows - 1`` degrees of freedom (agreement: the windows
      say the same thing to a degree chance would not);
    * at least ``min_positive_fraction`` of windows are positive (robustness:
      one strong era is not carrying the mean).

    One clean horizon is a PASS — the allocator consumes one.  No windows at
    all is a FAIL, not a vacuous pass.

    Why this replaced the every-window rule
    ---------------------------------------
    The rule it replaces required *each* window to clear the IC floor AND a
    daily-level bootstrap interval excluding zero.  Simulated at this run's
    measured shape (8 windows x 242 days, daily-IC autocorrelation 0.75, daily
    sd 0.094, the real moving-block bootstrap), it passed a signal with a
    **true** IC of 0.04 only 33.5% of the time and one of 0.03 only 4.8% of
    the time, while this rule passes them 100% and 97% of the time.  Both
    rules reject a null signal 100% of the time.  Regenerate with
    ``scripts/profiling/gate_power.py``; the decision and its evidence are in
    ``12_gate_decision.md``.

    The reason is structural, not a tuning matter.  A 242-day window at that
    autocorrelation carries an effective sample of roughly 35 days, so each
    window's interval is a low-resolution instrument, and demanding that eight
    of them *independently* resolve a modest effect multiplies a large miss
    rate eight times over.  Aggregating to the window level uses each window's
    mean, which is a perfectly good statistic, and lets the eight of them
    speak together.

    The retired rule is still evaluated and reported as ``strict_every_window``
    so nothing it would have said is hidden — it is simply no longer decisive.
    """
    reasons: list[str] = []
    evidence: list[str] = []
    strict_reasons: list[str] = []
    if not per_window_test:
        return GateResult(False, min_mean_ic, {}, ["no windows evaluated"])
    horizons = sorted({h for m in per_window_test.values() for h in m})
    per_h: dict[int, bool] = {}
    strict: dict[int, bool] = {}
    window_level: dict[int, WindowLevelStats] = {}
    for h in horizons:
        ok = True
        strict_ok = True
        ics: dict[str, float] = {}
        for wname, metrics in per_window_test.items():
            m = metrics.get(h)
            if m is None or m.n_days == 0 or not math.isfinite(m.mean_ic):
                reasons.append(f"{wname} h={h}: no scorable OOS days")
                ok = False
                strict_ok = False
                continue
            if m.n_days < min_days:
                # One scorable day with IC 0.9 used to pass the gate outright.
                reasons.append(
                    f"{wname} h={h}: only {m.n_days} scorable OOS day(s), "
                    f"need >= {min_days} "
                    f"(skipped: {m.n_skipped_thin} thin, "
                    f"{m.n_skipped_degenerate} degenerate)"
                )
                ok = False
                strict_ok = False
                continue
            ics[wname] = float(m.mean_ic)
            # The retired rule, as a diagnostic only.
            if m.mean_ic <= min_mean_ic:
                strict_reasons.append(
                    f"{wname} h={h}: mean IC {m.mean_ic:+.4f} <= {min_mean_ic:.3f}"
                )
                strict_ok = False
            if not (m.ci_lo > 0.0):
                strict_reasons.append(
                    f"{wname} h={h}: 95% CI [{m.ci_lo:+.4f}, {m.ci_hi:+.4f}] includes zero"
                )
                strict_ok = False
        strict[h] = strict_ok and ok

        n = len(ics)
        if n == 0:
            per_h[h] = False
            continue
        vals = np.array(list(ics.values()), dtype=np.float64)
        mean = float(vals.mean())
        sd = float(vals.std(ddof=1)) if n > 1 else float("nan")
        if n > 1 and sd > 0.0:
            t_stat = mean / (sd / math.sqrt(n))
        elif n > 1:
            # Identical windows: infinitely consistent, in whichever direction.
            t_stat = math.inf if mean > 0 else (-math.inf if mean < 0 else 0.0)
        else:
            t_stat = float("nan")
        t_crit = t_critical_95(n - 1) if n > 1 else float("nan")
        n_pos = int((vals > 0.0).sum())
        frac = n_pos / n
        window_level[h] = WindowLevelStats(
            n_windows=n, mean_ic=mean, sd_ic=sd, t_stat=t_stat, t_crit=t_crit,
            n_positive=n_pos, positive_fraction=frac,
            min_window_ic=float(vals.min()), max_window_ic=float(vals.max()),
            window_ics=dict(ics),
        )

        if n < min_windows:
            reasons.append(
                f"h={h}: only {n} window(s) with a usable OOS record, need >= "
                f"{min_windows} for a window-level test"
            )
            ok = False
        if not (mean > min_mean_ic):
            reasons.append(
                f"h={h}: mean of window OOS ICs {mean:+.4f} <= {min_mean_ic:.3f}"
            )
            ok = False
        if n >= min_windows and not (t_stat > t_crit):
            reasons.append(
                f"h={h}: window-level t {t_stat:.2f} <= t_crit {t_crit:.3f} "
                f"(df={n - 1}); the windows do not agree beyond chance"
            )
            ok = False
        if frac < min_positive_fraction:
            reasons.append(
                f"h={h}: only {n_pos}/{n} windows positive, "
                f"need >= {min_positive_fraction:.0%}"
            )
            ok = False
        if ok:
            evidence.append(
                f"h={h}: {n} windows, mean IC {mean:+.4f} > {min_mean_ic:.3f}, "
                f"t {t_stat:.2f} > {t_crit:.3f} (df={n - 1}), {n_pos}/{n} positive, "
                f"min {vals.min():+.4f}, max {vals.max():+.4f}"
            )
        per_h[h] = ok
    passed = any(per_h.values())
    return GateResult(
        passed, min_mean_ic, per_h, reasons,
        window_level=window_level, strict_every_window=strict,
        strict_reasons=strict_reasons, evidence=evidence,
    )


def _json_float(x: float) -> float | None:
    """``None`` for a non-finite metric — ``NaN`` is not portable JSON."""
    return float(x) if math.isfinite(x) else None


def gate_horizon(
    gate: GateResult, pooled: dict[int, HorizonMetrics]
) -> int | None:
    """The horizon the verdict rests on.

    ``gate.json`` reports one CI and one ICIR (pinned schema), so it has to name
    *which* horizon they describe.  On a PASS that is the passing horizon with
    the highest pooled OOS mean IC; on a FAIL it is the least-bad horizon, so a
    reader can see how far short it fell rather than a blank.
    """
    if not pooled:
        return None
    candidates = [h for h, ok in gate.per_horizon.items() if ok and h in pooled]
    pool = candidates if candidates else list(pooled)
    finite = [h for h in pool if math.isfinite(pooled[h].mean_ic)]
    if not finite:
        return min(pool)
    return max(finite, key=lambda h: pooled[h].mean_ic)


def gate_json_payload(
    gate: GateResult,
    pooled: dict[int, HorizonMetrics],
    n_windows: int,
) -> dict[str, Any]:
    """The pinned ``gate.json`` contract, plus provenance keys after it.

    ::

        {"verdict": "PASS"|"FAIL", "mean_ic_5d": f, "mean_ic_20d": f,
         "ic_ci_low": f, "ic_ci_high": f, "icir": f, "n_windows": int}

    ``mean_ic_5d`` and ``mean_ic_20d`` are always present (``null`` if that
    horizon was not run); any other configured horizon is reported alongside
    them.  ``mean_ic_{h}d`` is the **pooled OOS** mean rank IC — every scorable test day
    of every window, concatenated, then averaged.  ``ic_ci_low``/``ic_ci_high``
    (percentile bootstrap over days) and ``icir`` describe the single horizon
    named by ``gate_horizon``; the per-horizon breakdown follows in
    ``per_horizon`` so nothing is lost to the flattening.

    A non-finite metric is written as ``null`` rather than the unparseable
    ``NaN``.  That can only happen when a horizon had no scorable OOS day, which
    is a FAIL by construction (:func:`gate_verdict`).
    """
    gh = gate_horizon(gate, pooled)
    m = pooled.get(gh) if gh is not None else None
    payload: dict[str, Any] = {"verdict": "PASS" if gate.passed else "FAIL"}
    # The pinned keys are emitted UNCONDITIONALLY (null when that horizon was
    # not run), so a consumer written to the pinned schema cannot KeyError.
    # They used to be synthesised from whatever horizons were configured: with
    # horizons=(3,) the file carried mean_ic_3d and no mean_ic_5d/mean_ic_20d.
    for h in PINNED_GATE_HORIZONS:
        m_h = pooled.get(h)
        payload[f"mean_ic_{h}d"] = _json_float(m_h.mean_ic) if m_h is not None else None
    for h in sorted(pooled):
        if h not in PINNED_GATE_HORIZONS:
            payload[f"mean_ic_{h}d"] = _json_float(pooled[h].mean_ic)
    payload["ic_ci_low"] = _json_float(m.ci_lo) if m is not None else None
    payload["ic_ci_high"] = _json_float(m.ci_hi) if m is not None else None
    payload["icir"] = _json_float(m.icir) if m is not None else None
    payload["n_windows"] = int(n_windows)
    # Provenance — additive, so a reader of the pinned keys is unaffected.
    payload["gate_horizon"] = gh
    payload["min_mean_ic"] = float(gate.min_mean_ic)
    payload["per_horizon"] = {
        str(h): {
            "passed": bool(gate.per_horizon.get(h, False)),
            "mean_ic": _json_float(pooled[h].mean_ic),
            "ci_low": _json_float(pooled[h].ci_lo),
            "ci_high": _json_float(pooled[h].ci_hi),
            "icir": _json_float(pooled[h].icir),
            "hit_rate": _json_float(pooled[h].hit_rate),
            "decile_spread": _json_float(pooled[h].decile_spread),
            "n_days": int(pooled[h].n_days),
            # How many days the scorer could not use, and how persistent the
            # scored series was.  A model that collapses to a constant
            # cross-section on half its days is visible here and nowhere else.
            "n_skipped_thin": int(pooled[h].n_skipped_thin),
            "n_skipped_degenerate": int(pooled[h].n_skipped_degenerate),
            "ic_acf_lag1": _json_float(pooled[h].ic_acf_lag1),
            "n_eff": _json_float(pooled[h].n_eff),
        }
        for h in sorted(pooled)
    }
    payload["reasons"] = list(gate.reasons)
    # The rule and the window-level test behind the verdict, plus the retired
    # every-window rule as a diagnostic.  All additive.
    payload["gate_rule"] = GATE_RULE
    payload["evidence"] = list(gate.evidence)
    payload["window_level"] = {
        str(h): w.to_dict() for h, w in sorted(gate.window_level.items())
    }
    payload["strict_every_window"] = {
        str(h): bool(v) for h, v in sorted(gate.strict_every_window.items())
    }
    payload["strict_reasons"] = list(gate.strict_reasons)
    return payload


def write_gate_json(
    out_dir: Path,
    gate: GateResult,
    pooled: dict[int, HorizonMetrics],
    n_windows: int,
) -> Path:
    """Write ``<out_dir>/gate.json`` and return its path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "gate.json"
    path.write_text(
        json.dumps(gate_json_payload(gate, pooled, n_windows), indent=2) + "\n"
    )
    return path


def format_verdict(gate: GateResult) -> str:
    head = "R4 GATE: PASS" if gate.passed else "R4 GATE: FAIL"
    lines = [f"{head}  (rule: {GATE_RULE})"]
    for h, ok in sorted(gate.per_horizon.items()):
        w = gate.window_level.get(h)
        detail = ""
        if w is not None:
            detail = (
                f"  windows={w.n_windows} mean={w.mean_ic:+.4f} t={w.t_stat:.2f}"
                f"/{w.t_crit:.3f} positive={w.n_positive}/{w.n_windows}"
                f" min={w.min_window_ic:+.4f}"
            )
        strict = gate.strict_every_window.get(h)
        strict_s = (
            ""
            if strict is None
            else f"  [strict every-window: {'pass' if strict else 'fail'}]"
        )
        lines.append(f"  horizon {h:>3}d: {'pass' if ok else 'fail'}{detail}{strict_s}")
    for e in gate.evidence:
        lines.append(f"  + {e}")
    for r in gate.reasons:
        lines.append(f"  - {r}")
    return "\n".join(lines)


# ── Training ───────────────────────────────────────────────────────────────────


@dataclass
class SupervisedConfig:
    lookback: int = 60
    horizons: tuple[int, ...] = field(default_factory=lambda: DEFAULT_HORIZONS)
    batch_days: int = 32
    eval_batch_days: int = 64
    max_epochs: int = 30
    max_steps: int | None = None          # hard cap on optimiser steps (smoke runs)
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    lr_schedule: str = "cosine"           # "cosine" | "constant"
    patience: int = 5                     # epochs without val-IC improvement
    max_grad_norm: float = 1.0
    min_cross_section: int = 10
    n_boot: int = 1000
    seed: int = 42
    device: str = "auto"

    def __post_init__(self) -> None:
        if self.lr_schedule not in ("cosine", "constant"):
            raise ValueError(
                f"lr_schedule must be 'cosine' or 'constant', got {self.lr_schedule!r}"
            )
        if self.lookback < 1:
            raise ValueError("lookback must be >= 1")


@dataclass
class EpochRecord:
    epoch: int
    step: int
    train_loss: float
    val_ic: dict[int, float]
    val_ic_mean: float
    lr: float


@dataclass
class TrainHistory:
    epochs: list[EpochRecord]
    best_epoch: int
    best_val_ic: float
    n_steps: int
    stopped_early: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "best_epoch": self.best_epoch,
            "best_val_ic": self.best_val_ic,
            "n_steps": self.n_steps,
            "stopped_early": self.stopped_early,
            "epochs": [
                {
                    "epoch": e.epoch,
                    "step": e.step,
                    "train_loss": e.train_loss,
                    "val_ic_mean": e.val_ic_mean,
                    "lr": e.lr,
                    **{f"val_ic_{h}d": v for h, v in e.val_ic.items()},
                }
                for e in self.epochs
            ],
        }


def _scorable_days(tensors: PanelTensors, lookback: int) -> np.ndarray:
    """Day indices with a full lookback window and at least one label on some horizon."""
    T = tensors.n_days
    if T < lookback:
        return np.zeros(0, dtype=np.int64)
    labelled = np.zeros(T, dtype=bool)
    for tgt in tensors.targets.values():
        labelled |= np.isfinite(tgt).any(axis=1)
    idx = np.arange(max(lookback - 1, tensors.n_warmup), T)
    return idx[labelled[idx]]


def _predictable_days(tensors: PanelTensors, lookback: int) -> np.ndarray:
    """Day indices with a full input window AND past any feature-context prefix.

    With a warm-up of exactly ``lookback - 1`` the two bounds coincide and the
    first prediction lands on the first real OOS date, which is the point.
    """
    T = tensors.n_days
    if T < lookback:
        return np.zeros(0, dtype=np.int64)
    return np.arange(max(lookback - 1, tensors.n_warmup), T)


def predict_panel(
    model: SignalModel,
    tensors: PanelTensors,
    *,
    lookback: int,
    device: torch.device,
    batch_days: int = 64,
) -> dict[int, np.ndarray]:
    """``[T, N]`` predictions per horizon; NaN on days without a full lookback."""
    T, N = tensors.mask.shape
    out = {h: np.full((T, N), np.nan, dtype=np.float32) for h in model.horizons}
    idx = _predictable_days(tensors, lookback)
    if idx.size == 0:
        return out
    feats = torch.from_numpy(tensors.features)
    mask_t = torch.from_numpy(tensors.mask)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for s in range(0, idx.size, batch_days):
            b = idx[s : s + batch_days]
            x = _gather_windows(feats, b, lookback).to(device)
            m = mask_t[b].to(device)
            preds = model(x, m)
            for h in model.horizons:
                p = preds[horizon_key(h)].cpu().numpy()
                out[h][b] = np.where(tensors.mask[b], p, np.nan)
    model.train(was_training)
    return out


def _mean_val_ic(
    model: SignalModel,
    val: PanelTensors,
    cfg: SupervisedConfig,
    device: torch.device,
) -> dict[int, float]:
    preds = predict_panel(
        model, val, lookback=cfg.lookback, device=device, batch_days=cfg.eval_batch_days
    )
    out: dict[int, float] = {}
    for h in model.horizons:
        d = daily_scores(
            preds[h], val.targets[h], val.fwd_raw[h], min_cross_section=cfg.min_cross_section
        )
        out[h] = float(d.ic.mean()) if d.ic.size else float("nan")
    return out


def train_signal_window(
    model: SignalModel,
    train: PanelTensors,
    val: PanelTensors,
    cfg: SupervisedConfig,
    *,
    device: torch.device,
    on_epoch: Any | None = None,
) -> TrainHistory:
    """Fit `model` on `train`, early-stopping on mean validation rank IC.

    The model is left holding the *best* validation-IC weights, not the last.
    `on_epoch(record)` is called after each epoch (MLflow hook).
    """
    L = cfg.lookback
    train_idx = _scorable_days(train, L)
    if train_idx.size == 0:
        raise ValueError(
            f"train split has no labelled day with a full {L}-day lookback "
            f"({train.n_days} days, horizons {cfg.horizons})"
        )
    val_scorable = _scorable_days(val, L).size > 0
    if not val_scorable:
        logger.warning(
            "val split has no scorable day — early stopping is disabled and the "
            "final epoch's weights are kept"
        )

    feats = torch.from_numpy(train.features)
    mask_t = torch.from_numpy(train.mask)
    tgt_t = {h: torch.from_numpy(np.nan_to_num(train.targets[h], nan=0.0)) for h in cfg.horizons}
    lab_t = {h: torch.from_numpy(np.isfinite(train.targets[h])) for h in cfg.horizons}

    opt = torch.optim.Adam(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    steps_per_epoch = math.ceil(train_idx.size / cfg.batch_days)
    total_steps = steps_per_epoch * cfg.max_epochs
    if cfg.max_steps is not None:
        total_steps = min(total_steps, cfg.max_steps)
    sched: torch.optim.lr_scheduler.LRScheduler | None = None
    if cfg.lr_schedule == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(total_steps, 1))

    rng = np.random.default_rng(cfg.seed)
    best_state = copy.deepcopy(model.state_dict())
    best_ic = -math.inf
    best_epoch = 0
    bad_epochs = 0
    step = 0
    history: list[EpochRecord] = []
    stopped_early = False

    model.train()
    for epoch in range(1, cfg.max_epochs + 1):
        perm = rng.permutation(train_idx)
        losses: list[float] = []
        for s in range(0, perm.size, cfg.batch_days):
            if cfg.max_steps is not None and step >= cfg.max_steps:
                break
            b = perm[s : s + cfg.batch_days]
            x = _gather_windows(feats, b, L).to(device)
            m = mask_t[b].to(device)
            preds = model(x, m)
            loss = torch.zeros((), device=device)
            for h in cfg.horizons:
                valid = (lab_t[h][b] & mask_t[b]).to(device)
                loss = loss + aux_return_loss(
                    preds[horizon_key(h)], tgt_t[h][b].to(device), valid
                )
            opt.zero_grad(set_to_none=True)
            # `Tensor.backward` is untyped in the torch stubs; the free function
            # is the same call and keeps `mypy --strict` honest.
            torch.autograd.backward(loss)
            if cfg.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            opt.step()
            if sched is not None:
                sched.step()
            step += 1
            losses.append(float(loss.item()))

        val_ic = _mean_val_ic(model, val, cfg, device) if val_scorable else {}
        finite = [v for v in val_ic.values() if math.isfinite(v)]
        val_mean = float(np.mean(finite)) if finite else float("nan")
        rec = EpochRecord(
            epoch=epoch,
            step=step,
            train_loss=float(np.mean(losses)) if losses else float("nan"),
            val_ic=val_ic,
            val_ic_mean=val_mean,
            lr=float(opt.param_groups[0]["lr"]),
        )
        history.append(rec)
        logger.info(
            f"  epoch {epoch:>3}  step {step:>6}  loss {rec.train_loss:.5f}  "
            f"val IC {val_mean:+.4f}  "
            + " ".join(f"{h}d {v:+.4f}" for h, v in val_ic.items())
        )
        if on_epoch is not None:
            on_epoch(rec)

        if val_scorable and math.isfinite(val_mean):
            if val_mean > best_ic:
                best_ic, best_epoch = val_mean, epoch
                best_state = copy.deepcopy(model.state_dict())
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= cfg.patience:
                    stopped_early = True
                    logger.info(
                        f"  early stop: no val-IC improvement for {cfg.patience} epochs "
                        f"(best {best_ic:+.4f} at epoch {best_epoch})"
                    )
                    break
        else:
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
        if cfg.max_steps is not None and step >= cfg.max_steps:
            logger.info(f"  max_steps={cfg.max_steps} reached")
            break

    model.load_state_dict(best_state)
    model.eval()
    return TrainHistory(
        epochs=history,
        best_epoch=best_epoch,
        best_val_ic=best_ic if math.isfinite(best_ic) else float("nan"),
        n_steps=step,
        stopped_early=stopped_early,
    )


# ── Artefacts (pinned contract) ────────────────────────────────────────────────


def assert_unique_date_ticker(predictions: pl.DataFrame) -> None:
    """Enforce the pinned "one row per (date, tradeable ticker)" contract.

    The walk-forward concatenates one OOS frame per window.  When two windows'
    **test** segments overlap — reachable by lowering ``step_months`` below
    ``test_months``, which nothing forbids — the concatenation carries the same
    (date, ticker) twice with *different* ``r_hat`` values from two different
    models.  A downstream join on (date, ticker) then fans out or picks
    arbitrarily, and the pooled metrics double-count those days.  Both are
    completely silent, so this raises instead.

    Demonstrated on the real orchestrator with two overlapping windows:
    2019-02-01/T4 appeared twice carrying r_hat 0.005160 and 0.000173.
    """
    if predictions.height == 0:
        return
    keys = predictions.select(["date", "ticker"])
    n_unique = keys.unique().height
    if n_unique == predictions.height:
        return
    dupes = (
        keys.group_by(["date", "ticker"])
        .len()
        .filter(pl.col("len") > 1)
        .sort(["date", "ticker"])
    )
    sample = dupes.head(5).rows()
    raise ValueError(
        f"predictions.parquet violates the pinned one-row-per-(date, ticker) "
        f"contract: {predictions.height - n_unique} duplicate row(s) across "
        f"{dupes.height} (date, ticker) pair(s). First few (date, ticker, count): "
        f"{sample}. This means two walk-forward windows' test segments overlap "
        f"(check step_months >= test_months in the walk config); the duplicated "
        f"rows carry predictions from different models."
    )


def predictions_schema(horizons: tuple[int, ...]) -> dict[str, Any]:
    """The pinned ``predictions.parquet`` schema — one place, one definition."""
    schema: dict[str, Any] = {"date": pl.Date, "ticker": pl.Utf8}
    for h in horizons:
        schema[horizon_key(h)] = pl.Float64
    return schema


def empty_predictions_frame(horizons: tuple[int, ...]) -> pl.DataFrame:
    """Zero-row frame carrying the pinned schema (no windows produced OOS rows)."""
    return pl.DataFrame(schema=predictions_schema(horizons))


def predictions_frame(
    preds: dict[int, np.ndarray],
    tensors: PanelTensors,
    horizons: tuple[int, ...],
) -> pl.DataFrame:
    """Long-format OOS predictions: one row per (date, tradeable ticker).

    Columns: ``date`` (Date), ``ticker`` (str), then ``r_hat_{h}d`` (f64) per
    horizon.  Days without a prediction (no full lookback) emit no rows.
    """
    h0 = horizons[0]
    has = np.isfinite(preds[h0]) & tensors.mask                      # [T, N]
    t_idx, n_idx = np.nonzero(has)
    data: dict[str, Any] = {
        "date": [tensors.dates[t] for t in t_idx.tolist()],
        "ticker": [tensors.tickers[n] for n in n_idx.tolist()],
    }
    for h in horizons:
        data[horizon_key(h)] = preds[h][t_idx, n_idx].astype(np.float64)
    return pl.DataFrame(data, schema=predictions_schema(horizons))


def embed_panel_to(
    model: SignalModel,
    tensors: PanelTensors,
    out_path: Path,
    *,
    lookback: int,
    device: torch.device,
    batch_days: int = 64,
) -> list[date]:
    """Write ``[T', N, D]`` float16 embeddings from the frozen encoder to `out_path`.

    ``T'`` counts only days with a full lookback; the returned date list is the
    axis-0 index and goes into ``index.json["dates"]``.  Written through a
    memmap so a 504-ticker, 3000-day panel never has to sit in RAM twice.
    """
    idx = _predictable_days(tensors, lookback)
    N, D = tensors.n_tickers, model.embed_dim
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mm = np.lib.format.open_memmap(
        str(out_path), mode="w+", dtype=np.float16, shape=(int(idx.size), N, D)
    )
    feats = torch.from_numpy(tensors.features)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for s in range(0, idx.size, batch_days):
            b = idx[s : s + batch_days]
            z = model.encode(_gather_windows(feats, b, lookback).to(device))
            mm[s : s + b.size] = z.cpu().numpy().astype(np.float16)
    mm.flush()
    del mm
    model.train(was_training)
    return [tensors.dates[t] for t in idx.tolist()]


def write_artefacts(
    out_dir: Path,
    *,
    model: SignalModel,
    predictions: pl.DataFrame,
    full_tensors: PanelTensors,
    lookback: int,
    device: torch.device,
    batch_days: int = 64,
) -> dict[str, Path]:
    """Write the pinned R4 artefact set under `out_dir`.

    ::

        predictions.parquet   date (Date), ticker (str), r_hat_5d (f64), r_hat_20d (f64)
                              one row per (date, tradeable ticker); OOS only
        embeddings.npy        float16 [T, N, D] from the FROZEN encoder over the full panel
        index.json            {"dates": [...ISO], "tickers": [...], "embed_dim": D,
                               "feature_cols": [...], "encoder_state_sha256": "..."}
        signal_model.pt       state_dict of the model that produced embeddings.npy

    Axis 1 of ``embeddings.npy`` is ``index.json["tickers"]`` — the
    ``active_tickers()`` order the caller passed in — exactly.

    ``gate.json`` completes the pinned set and is written separately by
    :func:`write_gate_json`: it needs the pooled OOS metrics, which only exist
    once every window has been scored.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "predictions.parquet"
    predictions.sort(["date", "ticker"]).write_parquet(pred_path)

    emb_path = out_dir / "embeddings.npy"
    dates = embed_panel_to(
        model, full_tensors, emb_path, lookback=lookback, device=device, batch_days=batch_days
    )
    index: dict[str, Any] = {
        "dates": [d.isoformat() for d in dates],
        "tickers": list(full_tensors.tickers),
        "embed_dim": int(model.embed_dim),
        "feature_cols": list(full_tensors.feature_cols),
        "encoder_state_sha256": model.encoder_state_sha256(),
        # Additive, and load-bearing for anyone who trains on these.
        # `predictions.parquet` is OOS only; `embeddings.npy` is not. It covers
        # the whole panel, and the encoder that produced it was fitted on the
        # last window's train split — so embeddings for dates inside that span
        # are in-sample. A downstream model fitted on all of them, and evaluated
        # on any of them, is reporting an in-sample number.
        "note": (
            "embeddings.npy spans the FULL panel and is in-sample over the "
            "encoder's own train window; predictions.parquet is OOS only"
        ),
    }
    index_path = out_dir / "index.json"
    index_path.write_text(json.dumps(index, indent=2))

    model_path = out_dir / "signal_model.pt"
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, model_path)
    return {
        "predictions": pred_path,
        "embeddings": emb_path,
        "index": index_path,
        "model": model_path,
    }


# ── Walk-forward orchestration ─────────────────────────────────────────────────


@dataclass
class WindowResult:
    window: WindowConfig
    history: TrainHistory
    val_metrics: dict[int, HorizonMetrics]
    test_metrics: dict[int, HorizonMetrics]
    feature_std: dict[str, float]
    mlflow_run_id: str | None
    n_train_days: int
    n_val_days: int
    n_test_days: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "window": self.window.asdict_iso(),
            "mlflow_run_id": self.mlflow_run_id,
            "n_days": {
                "train": self.n_train_days,
                "val": self.n_val_days,
                "test": self.n_test_days,
            },
            "train": self.history.to_dict(),
            "val": {str(h): m.to_dict() for h, m in self.val_metrics.items()},
            "test": {str(h): m.to_dict() for h, m in self.test_metrics.items()},
            "feature_std_train": self.feature_std,
        }


@dataclass
class SignalRunSummary:
    tag: str
    out_dir: Path
    gate: GateResult
    windows: list[WindowResult]
    pooled_test: dict[int, HorizonMetrics]
    artefacts: dict[str, Path]
    summary_path: Path


def _log_metrics_prefixed(
    mlflow: Any, prefix: str, metrics: dict[int, HorizonMetrics]
) -> None:
    flat: dict[str, float] = {}
    for h, m in metrics.items():
        for k, v in m.to_dict().items():
            if k == "horizon":
                continue
            if isinstance(v, float) and math.isfinite(v):
                flat[f"{prefix}_{h}d_{k}"] = v
            elif isinstance(v, int):
                flat[f"{prefix}_{h}d_{k}"] = float(v)
    if flat:
        mlflow.log_metrics(flat)


def run_signal_walk_forward(
    *,
    full_panel: pl.DataFrame,
    windows: list[WindowConfig],
    tickers: list[str],
    universe_fn: Callable[[WindowConfig], list[str]] | None = None,
    feature_cols: list[str],
    model_cfg: SignalConfig,
    train_cfg: SupervisedConfig,
    out_dir: Path,
    tag: str,
    mlflow_port: int | None = 5555,
    mlflow_experiment: str = "signal",
    mlflow_params: dict[str, Any] | None = None,
) -> SignalRunSummary:
    """Train, evaluate and gate the signal model over walk-forward windows.

    Per window: materialise the three splits, run the feature-liveness gate on
    train, compute feature stats from train only, build tensors per split (so
    no label or lookback crosses a split boundary), train with early stopping
    on val IC, score val and test, log to MLflow.  Then pool the OOS (test)
    days, print the gate verdict, and write the artefacts.

    ``embeddings.npy`` comes from the *last* window's trained encoder — the one
    fitted on the most recent train span — applied frozen to the full panel.
    ``mlflow_port=None`` disables MLflow entirely (unit tests).
    """
    if not windows:
        raise ValueError("no walk-forward windows — nothing to run")
    seed_everything(train_cfg.seed)
    device = get_device(train_cfg.device)
    out_dir.mkdir(parents=True, exist_ok=True)
    horizons = tuple(train_cfg.horizons)
    if tuple(model_cfg.horizons) != horizons:
        raise ValueError(
            f"model horizons {model_cfg.horizons} != train horizons {horizons}"
        )
    logger.info(
        f"R4 signal walk-forward: tag={tag} device={device} windows={len(windows)} "
        f"tickers={len(tickers)} features={len(feature_cols)} lookback={train_cfg.lookback} "
        f"horizons={horizons}"
    )

    mlflow: Any | None = None
    if mlflow_port is not None:
        import mlflow as _mlflow

        mlflow = _mlflow
        mlflow.set_tracking_uri(f"http://localhost:{mlflow_port}")
        mlflow.set_experiment(mlflow_experiment)

    results: list[WindowResult] = []
    oos_frames: list[pl.DataFrame] = []
    last_model: SignalModel | None = None

    window_universes: dict[str, list[str]] = {}
    for window in windows:
        logger.info(f"── {window.name}: train {window.train_start}..{window.train_end}  "
                    f"val {window.val_start}..{window.val_end}  "
                    f"test {window.test_start}..{window.test_end}")
        # A POINT-IN-TIME universe is chosen per window, from data strictly
        # before that window's test span. Fixed at the window boundary and held
        # for its duration, which is how an index reconstitutes — not lookahead,
        # and it keeps the ticker axis a constant width so the observation cost
        # per run does not move with the universe.
        #
        # `tickers` remains the default so every run predating this reproduces.
        win_tickers = universe_fn(window) if universe_fn is not None else tickers
        if not win_tickers:
            raise ValueError(
                f"{window.name}: the universe function returned no tickers. A "
                "window with an empty universe trains on nothing and scores "
                "nothing, which downstream reads as a window that simply had no "
                "signal."
            )
        if universe_fn is not None:
            logger.info(f"   universe: {len(win_tickers)} names, "
                        f"{win_tickers[0]} … {win_tickers[-1]}")
        window_universes[window.name] = list(win_tickers)

        # Release the previous window's CUDA blocks before allocating this
        # window's. r4_v2 held a CONSTANT 504-name universe, so its peak was set
        # in W1 and every later window reused the same blocks — it ran at a flat
        # 122-125 s/epoch for all eight. A point-in-time universe GROWS (254 ->
        # 306 -> 373 -> ...), so each window needs strictly more than the last
        # and cannot reuse what the caching allocator is holding. The result is
        # fragmentation that reaches the 8 GiB ceiling early: measured 7,938 of
        # 8,188 MiB at W3, with the epoch time going 25s -> 8min. That is the
        # step-function penalty WSL2 CUDA oversubscription produces.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            free_b, total_b = torch.cuda.mem_get_info()
            logger.info(
                f"   VRAM before {window.name}: "
                f"{(total_b - free_b) / 2**20:,.0f} / {total_b / 2**20:,.0f} MiB used"
            )

        wdir = out_dir / "windows" / window.name
        # The test segment gets `lookback - 1` days of feature context from the
        # purge gap, so the first prediction lands on the first real OOS date
        # instead of 59 days into it. Train and val get none: they are scored
        # over their whole span and losing their opening days costs nothing that
        # a longer segment does not already supply.
        paths = materialise_window(
            full_panel, window, wdir, warmup_days=train_cfg.lookback - 1
        )
        train_df = pl.read_parquet(paths["train"])
        val_df = pl.read_parquet(paths["val"])
        test_df = pl.read_parquet(paths["test"])

        # Gate 0: every input column must be alive on the train split.
        feature_std = assert_feature_liveness(train_df, feature_cols, where=f"{window.name}/train")

        # Normalisation stats from train only (runner.py:111 convention).
        stats_path = wdir / "train.feature_stats.json"
        feat_stats = compute_feature_stats(paths["train"], feature_cols)
        save_stats(feat_stats, stats_path)
        feat_mean, feat_std_t = stats_to_tensors(feat_stats, feature_cols)

        mcs = train_cfg.min_cross_section
        train_t = build_panel_tensors(
            train_df, win_tickers, feature_cols, horizons, min_cross_section=mcs
        )
        val_t = build_panel_tensors(
            val_df, win_tickers, feature_cols, horizons, min_cross_section=mcs
        )
        test_t = build_panel_tensors(
            test_df, win_tickers, feature_cols, horizons, min_cross_section=mcs
        )
        for name, t in (("train", train_t), ("val", val_t), ("test", test_t)):
            lab = {h: int(np.isfinite(t.targets[h]).sum()) for h in horizons}
            skipped = max(train_cfg.lookback - 1, t.n_warmup)
            note = (
                f"{t.n_warmup} warm-up day(s) as feature context, unlabelled"
                if t.n_warmup
                else f"first {min(skipped, t.n_days)} days have no full lookback"
            )
            logger.info(
                f"  {name}: {t.n_days} days, {int(t.mask.sum()):,} tradeable rows, "
                f"labelled rows {lab}; {note}; "
                f"{max(t.n_days - skipped, 0)} predictable day(s)"
            )

        seed_everything(train_cfg.seed)
        model = SignalModel(model_cfg, feat_mean=feat_mean, feat_std=feat_std_t).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        logger.info(f"  SignalModel: {n_params:,} params, embed_dim={model.embed_dim}")

        run_id: str | None = None
        run_ctx: Any = None
        if mlflow is not None:
            run_ctx = mlflow.start_run(run_name=f"{tag}_{window.name}")
            run = run_ctx.__enter__()
            run_id = str(run.info.run_id)
            mlflow.set_tags(
                {
                    "window": window.name,
                    "tag": tag,
                    "seed": str(train_cfg.seed),
                    "stage": "R4",
                    **{k: v for k, v in window.asdict_iso().items() if k != "name"},
                }
            )
            params: dict[str, Any] = {
                "seed": train_cfg.seed,
                "n_tickers": len(win_tickers),
                "n_params": n_params,
                "device": str(device),
                **{f"train.{k}": str(v) for k, v in asdict(train_cfg).items()},
                **{f"model.{k}": str(v) for k, v in asdict(model_cfg).items()},
                **(mlflow_params or {}),
            }
            mlflow.log_params(params)
            mlflow.log_artifact(str(stats_path), artifact_path="feature_stats")

        def _on_epoch(rec: EpochRecord, _mlflow: Any = mlflow) -> None:
            if _mlflow is None:
                return
            m = {"train_loss": rec.train_loss, "lr": rec.lr}
            if math.isfinite(rec.val_ic_mean):
                m["val_ic_mean"] = rec.val_ic_mean
            for h, v in rec.val_ic.items():
                if math.isfinite(v):
                    m[f"val_ic_{h}d"] = v
            _mlflow.log_metrics(m, step=rec.epoch)

        try:
            history = train_signal_window(
                model, train_t, val_t, train_cfg, device=device, on_epoch=_on_epoch
            )
            val_preds = predict_panel(
                model, val_t, lookback=train_cfg.lookback, device=device,
                batch_days=train_cfg.eval_batch_days,
            )
            test_preds = predict_panel(
                model, test_t, lookback=train_cfg.lookback, device=device,
                batch_days=train_cfg.eval_batch_days,
            )
            val_m = evaluate_predictions(
                val_preds, val_t, min_cross_section=mcs, n_boot=train_cfg.n_boot,
                rng_seed=train_cfg.seed,
            )
            test_m = evaluate_predictions(
                test_preds, test_t, min_cross_section=mcs, n_boot=train_cfg.n_boot,
                rng_seed=train_cfg.seed,
            )
            for split, ms in (("val", val_m), ("test", test_m)):
                for h, m in ms.items():
                    logger.info(
                        f"  {window.name} {split} {h:>2}d: IC {m.mean_ic:+.4f} "
                        f"[{m.ci_lo:+.4f}, {m.ci_hi:+.4f}]  ICIR {m.icir:+.3f}  "
                        f"hit {m.hit_rate:.3f}  decile {m.decile_spread:+.5f}  n={m.n_days}"
                    )
            if mlflow is not None:
                _log_metrics_prefixed(mlflow, "val", val_m)
                _log_metrics_prefixed(mlflow, "test", test_m)
                mlflow.log_metrics(
                    {"best_epoch": float(history.best_epoch), "n_steps": float(history.n_steps)}
                )

            wres = WindowResult(
                window=window,
                history=history,
                val_metrics=val_m,
                test_metrics=test_m,
                feature_std=feature_std,
                mlflow_run_id=run_id,
                n_train_days=train_t.n_days,
                n_val_days=val_t.n_days,
                n_test_days=test_t.n_days,
            )
            (wdir / "summary.json").write_text(json.dumps(wres.to_dict(), indent=2))
            torch.save(
                {k: v.cpu() for k, v in model.state_dict().items()},
                wdir / "signal_model.pt",
            )
            if mlflow is not None:
                mlflow.log_artifact(str(wdir / "summary.json"))
        finally:
            if run_ctx is not None:
                run_ctx.__exit__(None, None, None)

        results.append(wres)
        oos_frames.append(predictions_frame(test_preds, test_t, horizons))
        last_model = model

    assert last_model is not None
    pooled = pool_daily(
        [r.test_metrics for r in results], n_boot=train_cfg.n_boot, rng_seed=train_cfg.seed
    )
    gate = gate_verdict({r.window.name: r.test_metrics for r in results})

    # `windows` is non-empty (checked above) and every iteration appends, so this
    # is a schema guard, not a live branch — but an empty parquet with the wrong
    # columns is exactly the kind of thing a downstream reader trusts silently.
    predictions = (
        pl.concat(oos_frames) if oos_frames else empty_predictions_frame(horizons)
    )
    assert_unique_date_ticker(predictions)
    # The saved encoder is the LAST window's, so its embeddings must be taken
    # over that window's ticker axis. Using the default list here would emit an
    # embedding matrix whose rows do not correspond to the model that produced
    # them — a mismatch nothing downstream could detect.
    embed_tickers = win_tickers if universe_fn is not None else tickers
    full_t = build_panel_tensors(
        full_panel, embed_tickers, feature_cols, horizons,
        min_cross_section=train_cfg.min_cross_section,
    )
    artefacts = write_artefacts(
        out_dir,
        model=last_model,
        predictions=predictions,
        full_tensors=full_t,
        lookback=train_cfg.lookback,
        device=device,
        batch_days=train_cfg.eval_batch_days,
    )
    artefacts["gate"] = write_gate_json(out_dir, gate, pooled, len(results))

    summary: dict[str, Any] = {
        "tag": tag,
        "gate": gate.to_dict(),
        "pooled_test": {str(h): m.to_dict() for h, m in pooled.items()},
        "windows": [r.to_dict() for r in results],
        # Which names each window actually traded. Without this a point-in-time
        # run is unauditable: two windows can report different ICs simply
        # because they ranked different universes, and nothing in the metrics
        # would say so.
        "window_universes": window_universes,
        "universe_is_point_in_time": universe_fn is not None,
        "encoder_window": results[-1].window.name,
        "encoder_state_sha256": last_model.encoder_state_sha256(),
        "model_state_sha256": state_dict_sha256(last_model.state_dict()),
        "train_cfg": asdict(train_cfg),
        "model_cfg": asdict(model_cfg),
        "n_tickers": len(tickers),
        "feature_cols": list(feature_cols),
        "artefacts": {k: str(v) for k, v in artefacts.items()},
        "convention": (
            "r_hat dated t uses feature rows t-L+1..t (close of t) and forecasts the "
            "cross-sectionally z-scored log return over t+1..t+h; trade on/after the "
            "open of t+1. Equivalent to the env's decision day t+1."
        ),
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    for h, m in pooled.items():
        logger.info(
            f"POOLED OOS {h:>2}d: IC {m.mean_ic:+.4f} [{m.ci_lo:+.4f}, {m.ci_hi:+.4f}]  "
            f"ICIR {m.icir:+.3f}  hit {m.hit_rate:.3f}  decile {m.decile_spread:+.5f}  "
            f"n_days={m.n_days}"
        )
    logger.info(format_verdict(gate))

    if mlflow is not None:
        with mlflow.start_run(run_name=f"{tag}_summary"):
            mlflow.set_tags({"tag": tag, "stage": "R4", "kind": "summary"})
            mlflow.log_params(
                {
                    "n_windows": len(results),
                    "gate_passed": gate.passed,
                    "encoder_window": results[-1].window.name,
                    **(mlflow_params or {}),
                }
            )
            _log_metrics_prefixed(mlflow, "pooled_test", pooled)
            mlflow.log_artifact(str(summary_path))
            mlflow.log_artifact(str(artefacts["index"]))
            mlflow.log_artifact(str(artefacts["gate"]))

    return SignalRunSummary(
        tag=tag,
        out_dir=out_dir,
        gate=gate,
        windows=results,
        pooled_test=pooled,
        artefacts=artefacts,
        summary_path=summary_path,
    )
