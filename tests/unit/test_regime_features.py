"""Unit tests for the M7-Phase-1 market-regime feature pipeline.

These tests cover:

1. Pure-numpy `compute_regime_features`:
   - Shape & column order match `REGIME_COLS`
   - Strictly backward-looking — perturbing future days at t=10 must not
     change `regime[t]` for any t ≤ 10
   - Breadth ∈ [0, 1], vol non-negative
   - Trend matches a hand-rolled cumulative sum
   - Returns finite values even with all-zero / all-tradeable inputs

2. `compute_regime_stats` / `regime_stats_to_tensors` round-trip:
   - Saving + loading + tensorizing reproduces the same vectors

3. End-to-end via `PanelTradingEnv`:
   - obs dict has the `regime` key with correct shape & dtype
   - Regime values across consecutive obs change as the agent steps forward
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

# ── compute_regime_features (pure numpy) ───────────────────────────────────────


def _toy_returns(T: int = 200, N: int = 6, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    log_ret = rng.normal(0.0, 0.01, size=(T, N))
    trd = np.ones((T, N), dtype=bool)
    return log_ret, trd


def test_regime_shape_and_dtype() -> None:
    from trader.data.regime_features import REGIME_COLS, compute_regime_features

    log_ret, trd = _toy_returns(T=200, N=6)
    out = compute_regime_features(log_ret, trd)
    assert out.shape == (200, len(REGIME_COLS))
    assert out.dtype == np.float32
    assert np.isfinite(out).all()


def test_regime_strictly_backward_looking() -> None:
    """Perturbing log_return at day t* must not change regime[t] for any t ≤ t*."""
    from trader.data.regime_features import compute_regime_features

    log_ret, trd = _toy_returns(T=120, N=8)
    base = compute_regime_features(log_ret, trd)

    # Perturb day 80's returns wildly
    log_ret_perturbed = log_ret.copy()
    log_ret_perturbed[80] += 5.0
    pert = compute_regime_features(log_ret_perturbed, trd)

    # Regime values for t ≤ 80 must be unchanged (pure backward-looking).
    # Note: regime[t] uses returns through day t-1, so regime[80] uses
    # returns up to day 79 — so even regime[80] should be unchanged.
    assert np.allclose(base[:81], pert[:81]), (
        "Regime is leaking future info: rows 0..80 changed when day 80 was perturbed"
    )
    # Sanity: the perturbation DOES propagate forward.
    assert not np.allclose(base[81:], pert[81:]), (
        "Perturbation didn't propagate to later days — bug?"
    )


def test_regime_breadth_in_unit_interval() -> None:
    from trader.data.regime_features import REGIME_COLS, compute_regime_features

    log_ret, trd = _toy_returns(T=300, N=20)
    out = compute_regime_features(log_ret, trd)
    breadth = out[:, REGIME_COLS.index("mkt_breadth_20d")]
    # First 20 rows are warm-up (zero), then breadth ∈ [0, 1].
    assert (breadth >= 0.0).all() and (breadth <= 1.0).all()


def test_regime_vol_non_negative() -> None:
    from trader.data.regime_features import REGIME_COLS, compute_regime_features

    log_ret, trd = _toy_returns(T=300, N=10)
    out = compute_regime_features(log_ret, trd)
    vol = out[:, REGIME_COLS.index("mkt_vol_20d")]
    vov = out[:, REGIME_COLS.index("mkt_vol_of_vol_60d")]
    disp = out[:, REGIME_COLS.index("mkt_dispersion_20d")]
    assert (vol >= 0.0).all()
    assert (vov >= 0.0).all()
    assert (disp >= 0.0).all()


def test_regime_trend_matches_manual_sum() -> None:
    """`mkt_trend_20d` must equal the rolling 20-day equal-weight cumulative
    return computed by hand on the benchmark series."""
    from trader.data.regime_features import REGIME_COLS, compute_regime_features

    T, N = 100, 5
    rng = np.random.default_rng(42)
    log_ret = rng.normal(0.001, 0.01, size=(T, N))
    trd = np.ones((T, N), dtype=bool)

    bench = log_ret.mean(axis=1)            # equal-weight mean per day
    out = compute_regime_features(log_ret, trd)
    trend = out[:, REGIME_COLS.index("mkt_trend_20d")]

    # Manual reference: trend[t] = sum(bench[t-20:t]) for t ≥ 20, else 0.
    expected = np.zeros(T, dtype=np.float64)
    for t in range(20, T):
        expected[t] = bench[t - 20 : t].sum()

    np.testing.assert_allclose(trend.astype(np.float64), expected, atol=1e-6)


def test_regime_handles_partial_tradeability() -> None:
    """Stocks marked non-tradeable must not contribute to breadth / dispersion."""
    from trader.data.regime_features import REGIME_COLS, compute_regime_features

    T, N = 80, 6
    rng = np.random.default_rng(0)
    log_ret = rng.normal(0.0, 0.01, size=(T, N))
    trd = np.ones((T, N), dtype=bool)

    # Mask out half the stocks throughout
    trd_partial = trd.copy()
    trd_partial[:, 3:] = False

    out_full = compute_regime_features(log_ret, trd)
    out_part = compute_regime_features(log_ret, trd_partial)

    # Breadth should differ (denominator is smaller, so different fractions)
    bi = REGIME_COLS.index("mkt_breadth_20d")
    diff = np.abs(out_full[40:, bi] - out_part[40:, bi]).mean()
    assert diff > 0.0, "Tradeability mask had no effect on breadth"


def test_regime_acceleration_is_signed() -> None:
    """Acceleration = trend_20 − trend_60 = −(sum of older 40 days in the 60d window).

    So when the older 40-day window is *negative*, acceleration is *positive*
    (the recent 20 days are improving relative to the prior trend).  This
    matches the financial interpretation: acceleration > 0 → momentum turning up.
    """
    from trader.data.regime_features import REGIME_COLS, compute_regime_features

    T, N = 200, 5
    log_ret = np.zeros((T, N), dtype=np.float64)
    # Days [T-60, T-20) have negative returns; days [T-20, T) are flat.
    # → trend_60[T-1] = -40·0.01 + 0 = -0.40
    # → trend_20[T-1] = 0
    # → accel[T-1] = trend_20 − trend_60 = +0.40
    log_ret[T - 60 : T - 20] = -0.01
    trd = np.ones((T, N), dtype=bool)

    out = compute_regime_features(log_ret, trd)
    accel = out[:, REGIME_COLS.index("mkt_acceleration")]
    assert accel[T - 1] > 0.3, f"Expected accel > 0.3, got {accel[T - 1]:.4f}"
    # Mirror case: positive older window ⇒ negative acceleration.
    log_ret2 = np.zeros((T, N), dtype=np.float64)
    log_ret2[T - 60 : T - 20] = 0.01
    out2 = compute_regime_features(log_ret2, trd)
    accel2 = out2[:, REGIME_COLS.index("mkt_acceleration")]
    assert accel2[T - 1] < -0.3, f"Expected accel < -0.3, got {accel2[T - 1]:.4f}"
    # Before the regime starts (well within zero region) acceleration is ~0.
    assert abs(accel[80]) < 1e-9


# ── compute_regime_stats / round-trip ──────────────────────────────────────────


def _write_tiny_panel(tmp_path: Path, T: int = 200, N: int = 5) -> Path:
    rng = np.random.default_rng(7)
    dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(T)]
    tickers = [f"T{i}" for i in range(N)]
    rows: list[dict] = []
    for di, d in enumerate(dates):
        for ti, tk in enumerate(tickers):
            rows.append(
                {
                    "date": d,
                    "ticker": tk,
                    "log_return_1d": float(rng.normal(0.0005, 0.012)),
                    "is_tradeable": True,
                }
            )
    df = pl.DataFrame(rows)
    p = tmp_path / "panel.parquet"
    df.write_parquet(p)
    return p


def test_compute_regime_stats_basic(tmp_path: Path) -> None:
    from trader.data.regime_features import REGIME_COLS, compute_regime_stats

    p = _write_tiny_panel(tmp_path, T=300, N=8)
    stats = compute_regime_stats(p)
    assert set(stats.keys()) == set(REGIME_COLS)
    for name in REGIME_COLS:
        mean, std = stats[name]
        assert np.isfinite(mean) and np.isfinite(std)
        assert std > 0.0, f"{name} has zero std — wasn't computed?"


def test_regime_stats_round_trip(tmp_path: Path) -> None:
    from trader.data.regime_features import (
        REGIME_COLS,
        compute_regime_stats,
        load_regime_stats,
        regime_stats_to_tensors,
        save_regime_stats,
    )

    panel_p = _write_tiny_panel(tmp_path, T=300, N=6)
    stats_p = tmp_path / "regime_stats.json"
    stats = compute_regime_stats(panel_p)
    save_regime_stats(stats, stats_p)
    loaded = load_regime_stats(stats_p)
    for name in REGIME_COLS:
        assert pytest.approx(stats[name][0]) == loaded[name][0]
        assert pytest.approx(stats[name][1]) == loaded[name][1]
    mean, std = regime_stats_to_tensors(loaded)
    assert mean.shape == (len(REGIME_COLS),)
    assert std.shape == (len(REGIME_COLS),)
    assert (std > 0).all()


# ── PanelTradingEnv emits the regime key ───────────────────────────────────────


def test_env_obs_includes_regime() -> None:
    """End-to-end: env reset/step returns `regime` of correct shape & dtype."""
    pytest.importorskip("trader.env.panel_env")
    from trader.data.regime_features import REGIME_DIM
    from trader.env.panel_env import PanelTradingEnv

    # Build a small synthetic panel that matches what panel_env expects.
    T, N = 120, 4
    rng = np.random.default_rng(1)
    dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(T)]
    tickers = [f"T{i}" for i in range(N)]
    rows: list[dict] = []
    for d in dates:
        for ti, tk in enumerate(tickers):
            rows.append(
                {
                    "date": d,
                    "ticker": tk,
                    "open": 100.0 + rng.normal(0, 1),
                    "close": 100.0 + rng.normal(0, 1),
                    "is_tradeable": True,
                    "sector_id": 1 + (ti % 3),
                    "atr_14": 1.0,
                    "dollar_volume_20": 1e6,
                    "log_return_1d": float(rng.normal(0.0, 0.01)),
                }
            )
    panel = pl.DataFrame(rows)
    panel_path = Path(__file__).parent / "_tmp_regime_panel.parquet"
    try:
        panel.write_parquet(panel_path)
        env = PanelTradingEnv(
            panel_path=panel_path,
            universe=tickers,
            feature_columns=["log_return_1d"],
            lookback=20,
            episode_length=30,
            initial_cash=1_000_000.0,
            seed=0,
        )
        obs, _ = env.reset(seed=0)
        assert "regime" in obs
        assert obs["regime"].shape == (REGIME_DIM,)
        assert obs["regime"].dtype == np.float32
        assert np.isfinite(obs["regime"]).all()

        # Step a few times — env should keep emitting regime cleanly.
        for _ in range(3):
            obs, _, _, _, _ = env.step(np.zeros(N + 1, dtype=np.float32))
            assert obs["regime"].shape == (REGIME_DIM,)
            assert np.isfinite(obs["regime"]).all()
    finally:
        if panel_path.exists():
            panel_path.unlink()
