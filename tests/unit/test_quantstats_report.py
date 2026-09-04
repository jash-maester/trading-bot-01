"""Tests for the QuantStats metric bridge."""
from __future__ import annotations

import math

import numpy as np
import pytest

from trader.training.quantstats_report import (
    QS_METRICS,
    aggregate_quantstats,
    compute_quantstats_metrics,
    log_returns_to_series,
    save_tearsheet,
)


def _episode(seed: int, n: int = 252, mu: float = 0.0004, sd: float = 0.011) -> list[float]:
    rng = np.random.default_rng(seed)
    return list(rng.normal(mu, sd, n))


# ── conversion ────────────────────────────────────────────────────────────────


def test_log_to_simple_return_conversion() -> None:
    """QuantStats wants simple returns; the env emits log returns."""
    log_rets = [0.0, 0.01, -0.02, 0.005]
    s = log_returns_to_series(log_rets)
    assert len(s) == 4
    for got, lr in zip(s.to_numpy(), log_rets, strict=True):
        assert got == pytest.approx(math.expm1(lr))


def test_series_index_is_business_days_and_unique() -> None:
    s = log_returns_to_series(_episode(0, n=30))
    assert s.index.is_monotonic_increasing
    assert s.index.is_unique
    assert len(s) == 30


def test_compounded_series_matches_log_sum() -> None:
    """expm1/log1p round-trips: compounding simple returns == sum of log returns."""
    log_rets = _episode(7, n=100)
    s = log_returns_to_series(log_rets)
    compounded = float((1.0 + s).prod())
    assert math.log(compounded) == pytest.approx(sum(log_rets), rel=1e-9)


# ── metric computation ────────────────────────────────────────────────────────


def test_every_curated_metric_computes() -> None:
    """The curated list must not drift away from the installed quantstats."""
    m = compute_quantstats_metrics(_episode(1))
    missing = [k for k in QS_METRICS if k not in m]
    assert not missing, f"curated metrics no longer computable: {missing}"


def test_all_values_are_finite_floats() -> None:
    """NaN/inf must be dropped — MLflow stores them as real datapoints."""
    m = compute_quantstats_metrics(_episode(2))
    assert m
    for k, v in m.items():
        assert isinstance(v, float), k
        assert math.isfinite(v), k


def test_short_series_returns_empty_not_error() -> None:
    assert compute_quantstats_metrics([]) == {}
    assert compute_quantstats_metrics([0.01]) == {}


def test_positive_drift_beats_negative_drift_on_sharpe() -> None:
    good = compute_quantstats_metrics(_episode(3, mu=0.0015))
    bad = compute_quantstats_metrics(_episode(3, mu=-0.0015))
    assert good["sharpe"] > bad["sharpe"]
    assert good["cagr"] > bad["cagr"]


def test_max_drawdown_is_non_positive() -> None:
    m = compute_quantstats_metrics(_episode(4))
    assert m["max_drawdown"] <= 0.0


# ── aggregation ───────────────────────────────────────────────────────────────


def test_aggregate_emits_mean_and_std_per_metric() -> None:
    eps = [_episode(s) for s in range(5)]
    agg = aggregate_quantstats(eps)
    assert "sharpe" in agg
    assert "sharpe_std" in agg
    assert agg["sharpe_std"] >= 0.0


def test_aggregate_mean_matches_manual_mean() -> None:
    eps = [_episode(s) for s in range(4)]
    agg = aggregate_quantstats(eps)
    manual = float(np.mean([compute_quantstats_metrics(e)["sharpe"] for e in eps]))
    assert agg["sharpe"] == pytest.approx(manual, rel=1e-9)


def test_single_episode_emits_no_std() -> None:
    agg = aggregate_quantstats([_episode(0)])
    assert "sharpe" in agg
    assert "sharpe_std" not in agg


def test_aggregate_keys_are_unprefixed() -> None:
    """runner.py prefixes with qs/<split>/ — the module must not double up."""
    agg = aggregate_quantstats([_episode(0), _episode(1)])
    assert not any(k.startswith("qs/") for k in agg)


def test_aggregate_of_nothing_is_empty() -> None:
    assert aggregate_quantstats([]) == {}
    assert aggregate_quantstats([[], [0.01]]) == {}


# ── tearsheet ─────────────────────────────────────────────────────────────────


def test_tearsheet_written_with_benchmark(tmp_path) -> None:  # type: ignore[no-untyped-def]
    out = tmp_path / "ts.html"
    got = save_tearsheet(_episode(0), out, "test", benchmark_log_returns=_episode(1))
    assert got is not None
    assert got.exists()
    assert got.stat().st_size > 10_000


def test_tearsheet_short_series_returns_none(tmp_path) -> None:  # type: ignore[no-untyped-def]
    assert save_tearsheet([], tmp_path / "a.html", "t") is None
    assert save_tearsheet([0.01], tmp_path / "b.html", "t") is None
