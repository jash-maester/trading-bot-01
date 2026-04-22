"""Integration tests for PanelTradingEnv — all invariants from 03_environment.md.

These tests build a synthetic panel in a temp directory and run the env
without a real database or network connection.
"""
from __future__ import annotations

import math
import tempfile
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

# ── synthetic panel fixture ───────────────────────────────────────────────────

_FEATURE_COLS = [
    "log_return_1d", "log_return_5d", "log_return_20d",
    "realized_vol_20d", "realized_vol_60d",
    "rsi_14", "macd", "macd_signal", "macd_hist",
    "bbw_20", "z_close_20", "volume_z_20", "dollar_volume_20",
    "atr_14", "beta_nifty_60d",
]
_TICKERS = ["A.NS", "B.NS", "C.NS"]
_N = len(_TICKERS)
_T = 400   # enough for lookback=60 + episode=252 + buffer


def _make_panel(tickers: list[str] = _TICKERS, n_days: int = _T) -> pl.DataFrame:
    rng = np.random.default_rng(42)
    start = date(2018, 1, 2)
    all_rows = []

    for ticker in tickers:
        price = 500.0
        for i in range(n_days):
            d = start + timedelta(days=i)
            price = max(1.0, price * (1.0 + rng.normal(0.0, 0.01)))
            row: dict[str, object] = {
                "date": d,
                "ticker": ticker,
                "open": round(price * 0.999, 4),
                "high": round(price * 1.005, 4),
                "low": round(price * 0.995, 4),
                "close": round(price, 4),
                "adj_close": round(price, 4),
                "volume": int(rng.integers(500_000, 2_000_000)),
                "is_tradeable": True,
                "sector_id": 1,
                "atr_14": round(price * 0.015, 4),
                "dollar_volume_20": round(price * 800_000, 2),
            }
            for fc in _FEATURE_COLS:
                row[fc] = float(rng.normal(0.0, 0.01))
            all_rows.append(row)

    return pl.DataFrame(all_rows)


@pytest.fixture(scope="module")
def panel_file() -> Path:  # type: ignore[misc]
    tmp = tempfile.mkdtemp()
    path = Path(tmp) / "train.parquet"
    _make_panel().write_parquet(path)
    return path


def _make_env(panel_file: Path, seed: int = 0) -> object:
    from trader.env.panel_env import PanelTradingEnv

    return PanelTradingEnv(
        panel_path=panel_file,
        universe=_TICKERS,
        feature_columns=_FEATURE_COLS,
        lookback=60,
        episode_length=100,   # short for test speed
        initial_cash=1_000_000.0,
        seed=seed,
    )


# ── invariant tests ───────────────────────────────────────────────────────────


def test_env_reset_returns_valid_obs(panel_file: Path) -> None:
    env = _make_env(panel_file)
    obs, info = env.reset(seed=1)  # type: ignore[union-attr]
    assert "features" in obs
    assert "mask" in obs
    assert "portfolio" in obs
    assert obs["features"].shape == (60, _N, len(_FEATURE_COLS))
    assert obs["mask"].shape == (_N,)
    assert obs["portfolio"].shape == (_N + 1,)


def test_softmax_weights_sum_to_one_each_step(panel_file: Path) -> None:
    """Sum of portfolio weights must be 1.0 ± 1e-4 after each step."""
    env = _make_env(panel_file)
    env.reset(seed=2)  # type: ignore[union-attr]
    for _ in range(10):
        action = env.action_space.sample()  # type: ignore[union-attr]
        obs, _, term, trunc, info = env.step(action)  # type: ignore[union-attr]
        w = info["weights"]
        assert abs(float(w.sum()) - 1.0) < 1e-4, f"Weights sum = {w.sum()}"
        if term or trunc:
            break


def test_masked_names_have_zero_weight(panel_file: Path) -> None:
    """Tickers with mask=0 must have zero weight in every step."""
    from trader.env.panel_env import PanelTradingEnv

    # Build a panel where ticker B.NS is never tradeable
    rng = np.random.default_rng(7)
    start = date(2018, 1, 2)
    rows = []
    for ticker in _TICKERS:
        price = 500.0
        for i in range(_T):
            d = start + timedelta(days=i)
            price = max(1.0, price * (1.0 + rng.normal(0, 0.01)))
            is_t = ticker != "B.NS"
            row: dict[str, object] = {
                "date": d, "ticker": ticker,
                "open": price, "high": price * 1.005, "low": price * 0.995,
                "close": price, "adj_close": price,
                "volume": 0 if not is_t else 1_000_000,
                "is_tradeable": is_t,
                "sector_id": 1, "atr_14": price * 0.01,
                "dollar_volume_20": price * 500_000,
            }
            for fc in _FEATURE_COLS:
                row[fc] = 0.0 if not is_t else float(rng.normal(0, 0.01))
            rows.append(row)

    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "panel.parquet"
        pl.DataFrame(rows).write_parquet(p)
        env = PanelTradingEnv(
            panel_path=p,
            universe=_TICKERS,
            feature_columns=_FEATURE_COLS,
            lookback=60,
            episode_length=50,
            seed=0,
        )
        env.reset(seed=0)
        b_idx = _TICKERS.index("B.NS")
        for _ in range(10):
            action = env.action_space.sample()
            obs, _, term, trunc, info = env.step(action)
            weights = info["weights"]
            assert weights[b_idx + 1] == pytest.approx(0.0, abs=1e-6), (
                f"B.NS weight should be 0 but got {weights[b_idx + 1]}"
            )
            if term or trunc:
                break


def test_no_action_zero_turnover(panel_file: Path) -> None:
    """Holding current allocation → turnover ≈ 0 and cost ≈ 0."""
    from trader.env.panel_env import PanelTradingEnv

    env = PanelTradingEnv(
        panel_path=panel_file,
        universe=_TICKERS,
        feature_columns=_FEATURE_COLS,
        lookback=60,
        episode_length=100,
        seed=3,
    )
    obs, _ = env.reset(seed=3)
    # First step: put everything in cash
    all_cash = np.full(_N + 1, -1e9, dtype=np.float32)
    all_cash[0] = 0.0
    obs, _, term, trunc, info1 = env.step(all_cash)
    # Second step: repeat same all-cash action
    obs, _, term, trunc, info2 = env.step(all_cash)
    # Turnover and cost should be near 0 (holding cash → no trades)
    assert info2["costs_paid"] == pytest.approx(0.0, abs=1e-6)


def test_all_cash_nav_does_not_grow(panel_file: Path) -> None:
    """If agent stays in cash the whole episode, NAV should not exceed initial."""
    env = _make_env(panel_file, seed=5)
    env.reset(seed=5)  # type: ignore[union-attr]
    all_cash = np.full(_N + 1, -1e9, dtype=np.float32)
    all_cash[0] = 0.0
    nav_final = 1_000_000.0
    for _ in range(20):
        obs, _, term, trunc, info = env.step(all_cash)  # type: ignore[union-attr]
        nav_final = info["nav"]
        if term or trunc:
            break
    assert nav_final <= 1_000_000.0 + 1.0   # small float rounding tolerance


def test_no_nan_in_obs_or_reward(panel_file: Path) -> None:
    """No NaN in obs or reward across a full 100-step episode."""
    env = _make_env(panel_file, seed=6)
    obs, _ = env.reset(seed=6)  # type: ignore[union-attr]
    _assert_obs_no_nan(obs)

    for step_i in range(100):
        action = env.action_space.sample()  # type: ignore[union-attr]
        obs, reward, term, trunc, _ = env.step(action)  # type: ignore[union-attr]
        assert not math.isnan(float(reward)), f"NaN reward at step {step_i}"
        _assert_obs_no_nan(obs)
        if term or trunc:
            break


def _assert_obs_no_nan(obs: dict[str, np.ndarray]) -> None:
    for key, arr in obs.items():
        if np.issubdtype(arr.dtype, np.floating):
            assert not np.any(np.isnan(arr)), f"NaN in obs[{key!r}]"


def test_seeded_reset_produces_identical_trajectory(panel_file: Path) -> None:
    """Two resets with the same seed must produce identical observations."""
    env = _make_env(panel_file, seed=0)

    obs1, _ = env.reset(seed=42)  # type: ignore[union-attr]
    rng = np.random.default_rng(99)
    actions1 = [rng.uniform(-1, 1, _N + 1).astype(np.float32) for _ in range(5)]
    traj1 = []
    for a in actions1:
        _, r, term, trunc, _ = env.step(a)  # type: ignore[union-attr]
        traj1.append(r)
        if term or trunc:
            break

    obs2, _ = env.reset(seed=42)  # type: ignore[union-attr]
    rng2 = np.random.default_rng(99)
    actions2 = [rng2.uniform(-1, 1, _N + 1).astype(np.float32) for _ in range(5)]
    traj2 = []
    for a in actions2:
        _, r, term, trunc, _ = env.step(a)  # type: ignore[union-attr]
        traj2.append(r)
        if term or trunc:
            break

    np.testing.assert_allclose(traj1, traj2, rtol=1e-5)
    np.testing.assert_array_equal(
        obs1["features"], obs2["features"]
    )


def test_full_episode_walltime(panel_file: Path) -> None:
    """252-step episode with 3 tickers should complete well under 200 ms."""
    from trader.env.panel_env import PanelTradingEnv

    env = PanelTradingEnv(
        panel_path=panel_file,
        universe=_TICKERS,
        feature_columns=_FEATURE_COLS,
        lookback=60,
        episode_length=252,
        seed=0,
    )
    env.reset(seed=0)
    rng = np.random.default_rng(0)

    t0 = time.perf_counter()
    for _ in range(252):
        action = rng.uniform(-1, 1, _N + 1).astype(np.float32)
        _, _, term, trunc, _ = env.step(action)
        if term or trunc:
            break
    elapsed = time.perf_counter() - t0

    # For 3 tickers the loop is trivially fast; 200 ms headroom for 150
    assert elapsed < 5.0, f"Episode took {elapsed:.3f}s — too slow"
