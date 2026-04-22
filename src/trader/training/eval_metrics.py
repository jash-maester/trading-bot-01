"""Evaluation metrics for trading strategies."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class EpisodeMetrics:
    """Metrics computed from a single episode's daily log returns and turnovers."""

    cagr: float = 0.0
    sharpe: float = 0.0
    sortino: float = 0.0
    max_drawdown: float = 0.0
    calmar: float = 0.0
    turnover_ann: float = 0.0
    total_return: float = 0.0
    cost_drag: float = 0.0
    n_days: int = 0
    daily_returns: list[float] = field(default_factory=list)


def compute_episode_metrics(
    nav_series: list[float],
    turnovers: list[float],
    gross_nav_series: list[float] | None = None,
    trading_days_per_year: int = 252,
) -> EpisodeMetrics:
    """Compute all standard metrics from a NAV time-series.

    Parameters
    ----------
    nav_series:
        Daily NAV values (length T+1, starting with initial NAV).
    turnovers:
        Daily one-way turnover fractions (length T).
    gross_nav_series:
        Optional NAV without costs — used to compute cost_drag.
    trading_days_per_year:
        252 for NSE.
    """
    if len(nav_series) < 2:
        return EpisodeMetrics()

    navs = np.array(nav_series, dtype=np.float64)
    log_rets = np.log(navs[1:] / np.maximum(navs[:-1], 1e-12))
    T = len(log_rets)

    total_return = float(navs[-1] / navs[0] - 1.0)
    cagr = float((navs[-1] / navs[0]) ** (trading_days_per_year / T) - 1.0)

    mean_r = float(log_rets.mean())
    std_r = float(log_rets.std(ddof=1)) if T > 1 else 1e-12
    sharpe = float(mean_r / (std_r + 1e-12) * math.sqrt(trading_days_per_year))

    downside = log_rets[log_rets < 0]
    down_std = float(downside.std(ddof=1)) if len(downside) > 1 else 1e-12
    sortino = float(mean_r / (down_std + 1e-12) * math.sqrt(trading_days_per_year))

    # Max drawdown from peak
    running_max = np.maximum.accumulate(navs)
    drawdowns = (navs - running_max) / np.maximum(running_max, 1e-12)
    max_drawdown = float(drawdowns.min())

    calmar = float(cagr / (abs(max_drawdown) + 1e-12))

    turnover_ann = float(np.mean(turnovers)) * trading_days_per_year if turnovers else 0.0

    cost_drag = 0.0
    if gross_nav_series is not None and len(gross_nav_series) == len(nav_series):
        gross_navs = np.array(gross_nav_series, dtype=np.float64)
        gross_cagr = float(
            (gross_navs[-1] / gross_navs[0]) ** (trading_days_per_year / T) - 1.0
        )
        cost_drag = gross_cagr - cagr

    return EpisodeMetrics(
        cagr=cagr,
        sharpe=sharpe,
        sortino=sortino,
        max_drawdown=max_drawdown,
        calmar=calmar,
        turnover_ann=turnover_ann,
        total_return=total_return,
        cost_drag=cost_drag,
        n_days=T,
        daily_returns=log_rets.tolist(),
    )


def aggregate_metrics(episodes: list[EpisodeMetrics]) -> dict[str, float]:
    """Aggregate metrics across multiple episodes (mean ± std)."""
    if not episodes:
        return {}

    result: dict[str, float] = {}
    fields = ["cagr", "sharpe", "sortino", "max_drawdown", "calmar", "turnover_ann", "total_return"]
    for f in fields:
        vals = [getattr(e, f) for e in episodes]
        result[f"mean_{f}"] = float(np.mean(vals))
        result[f"std_{f}"] = float(np.std(vals))
    return result


def compute_alpha_beta(
    strategy_log_rets: list[float],
    benchmark_log_rets: list[float],
    trading_days_per_year: int = 252,
) -> tuple[float, float]:
    """OLS alpha and beta vs benchmark (annualized alpha)."""
    s = np.array(strategy_log_rets)
    b = np.array(benchmark_log_rets)
    n = min(len(s), len(b))
    if n < 2:
        return 0.0, 1.0
    s, b = s[:n], b[:n]
    cov = np.cov(s, b)
    beta = float(cov[0, 1] / (cov[1, 1] + 1e-12))
    alpha_daily = float(s.mean() - beta * b.mean())
    alpha_ann = alpha_daily * trading_days_per_year
    return alpha_ann, beta
