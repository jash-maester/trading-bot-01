"""QuantStats metric logging and tearsheets for every evaluated run.

The project's own :class:`~trader.training.eval_metrics.EpisodeMetrics` carries
eight numbers.  QuantStats computes ~50 more from the same daily return series —
tail ratio, Omega, Ulcer index, CVaR, Kelly, probabilistic Sharpe and so on —
plus the HTML tearsheet in the upstream README.  This module bridges the two so
that every run records the full set rather than the eight.

Two things to know about the inputs:

* The env produces **log** returns; QuantStats expects **simple** returns.  The
  conversion is ``expm1`` and it is applied here, once, at the boundary.
* :class:`EpisodeMetrics` does not carry calendar dates, so a synthetic
  business-day index is attached.  Ratio metrics are unaffected.  Time-dependent
  metrics (CAGR, annualised vol) depend only on the *number* of periods and the
  252-day convention, both of which the synthetic index preserves — but the
  tearsheet's x-axis is therefore nominal, not the real trading calendar.
"""
from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

# Scalar metrics worth recording per run.  Deliberately curated rather than
# `dir(qs.stats)`: that also returns plotting helpers, DataFrame-valued
# functions and internal utilities.  Every name here returns one float.
QS_METRICS: tuple[str, ...] = (
    # returns
    "cagr", "comp", "expected_return", "geometric_mean", "ghpr", "avg_return",
    "best", "worst", "exposure",
    # risk-adjusted
    "sharpe", "smart_sharpe", "probabilistic_sharpe_ratio", "adjusted_sortino",
    "sortino", "smart_sortino", "probabilistic_sortino_ratio", "calmar",
    "omega", "rar", "risk_return_ratio", "serenity_index",
    # drawdown / risk
    "max_drawdown", "ulcer_index", "ulcer_performance_index", "recovery_factor",
    "risk_of_ruin", "value_at_risk", "conditional_value_at_risk",
    "expected_shortfall", "volatility",
    # distribution
    "skew", "kurtosis", "tail_ratio", "outlier_win_ratio", "outlier_loss_ratio",
    # win/loss
    "win_rate", "avg_win", "avg_loss", "win_loss_ratio", "payoff_ratio",
    "profit_factor", "profit_ratio", "gain_to_pain_ratio", "cpc_index",
    "common_sense_ratio", "kelly_criterion",
)

# Metrics that additionally accept a benchmark series and are only meaningful
# against one.
QS_BENCHMARK_METRICS: tuple[str, ...] = ("information_ratio", "r_squared", "greeks")


def log_returns_to_series(log_returns: list[float]) -> pd.Series:
    """Convert a list of daily **log** returns to a QuantStats-ready Series.

    Returns simple returns on a synthetic business-day index — see module
    docstring for why the index is synthetic and what it does not affect.
    """
    import pandas as pd

    arr = np.asarray(log_returns, dtype=np.float64)
    simple = np.expm1(arr)
    idx = pd.bdate_range("2000-01-03", periods=len(simple))
    return pd.Series(simple, index=idx, dtype="float64")


def _scalar(value: Any) -> float | None:
    """Coerce a QuantStats return value to a finite float, or None."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float, np.floating, np.integer)):
        f = float(value)
        return f if math.isfinite(f) else None
    return None


def compute_quantstats_metrics(
    log_returns: list[float],
    periods_per_year: int = 252,
) -> dict[str, float]:
    """Compute every metric in :data:`QS_METRICS` for one return series.

    Metrics that raise, or that return a non-finite / non-scalar value, are
    omitted rather than logged as NaN — MLflow treats NaN as a real datapoint
    and it pollutes the run comparison view.
    """
    if len(log_returns) < 2:
        return {}

    import quantstats as qs

    series = log_returns_to_series(log_returns)
    out: dict[str, float] = {}

    with warnings.catch_warnings():
        # QuantStats is noisy on short series (degrees-of-freedom, empty slices).
        warnings.simplefilter("ignore")
        for name in QS_METRICS:
            fn = getattr(qs.stats, name, None)
            if fn is None:
                continue
            try:
                try:
                    value = fn(series, periods=periods_per_year)
                except TypeError:
                    value = fn(series)
            except Exception:
                continue
            scalar = _scalar(value)
            if scalar is not None:
                out[name] = scalar
    return out


def aggregate_quantstats(
    episodes: list[list[float]],
    periods_per_year: int = 252,
) -> dict[str, float]:
    """Mean / std of every QuantStats metric across independent episodes.

    Episodes are separate sample paths, so they are summarised the way
    :func:`~trader.training.eval_metrics.aggregate_metrics` summarises the
    built-in metrics — not concatenated into one pseudo-track-record.

    Keys are returned bare (``cagr``, ``cagr_std``, ...).  Callers prefix them —
    ``runner.py`` logs them as ``qs/<split>/<name>`` so they group in the MLflow
    UI and never collide with the existing ``mean_sharpe`` / ``mean_cagr``
    names, which are computed differently and must stay comparable to
    historical runs.
    """
    per_episode = [
        m for m in (compute_quantstats_metrics(e, periods_per_year) for e in episodes) if m
    ]
    if not per_episode:
        return {}

    keys: set[str] = set()
    for m in per_episode:
        keys |= m.keys()

    agg: dict[str, float] = {}
    for k in sorted(keys):
        vals = np.array([m[k] for m in per_episode if k in m], dtype=np.float64)
        if vals.size == 0:
            continue
        agg[k] = float(vals.mean())
        if vals.size > 1:
            agg[f"{k}_std"] = float(vals.std(ddof=1))
    return agg


def save_tearsheet(
    log_returns: list[float],
    out_path: Path,
    title: str,
    benchmark_log_returns: list[float] | None = None,
) -> Path | None:
    """Write the QuantStats HTML tearsheet. Returns the path, or None on failure.

    Tearsheet generation is best-effort: it pulls in matplotlib and is the most
    fragile part of the dependency, and a plotting failure must never abort a
    training run that has already produced its metrics.
    """
    if len(log_returns) < 2:
        return None

    import quantstats as qs

    series = log_returns_to_series(log_returns)
    bench = None
    if benchmark_log_returns is not None and len(benchmark_log_returns) >= 2:
        bench = log_returns_to_series(benchmark_log_returns)
        # A benchmark of a different length cannot be aligned to the strategy's
        # synthetic index; truncate both to the overlap.
        n = min(len(series), len(bench))
        series, bench = series.iloc[:n], bench.iloc[:n]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib

        matplotlib.use("Agg")  # headless — no display on a training box
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            qs.reports.html(  # type: ignore[no-untyped-call]  # quantstats ships no stubs
                series,
                benchmark=bench,
                output=str(out_path),
                title=title,
                download_filename=str(out_path),
            )
    except Exception:
        return None
    return out_path if out_path.exists() else None
