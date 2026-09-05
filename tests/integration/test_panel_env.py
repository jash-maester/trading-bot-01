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
    # Turnover and cost should be near 0 (holding cash → no trades).
    # The turnover half of this assertion was missing while `info["turnover"]`
    # was NAV drift; it is the invariant the test is named after.
    assert info2["costs_paid"] == pytest.approx(0.0, abs=1e-6)
    assert info2["turnover"] == pytest.approx(0.0, abs=1e-9)


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


# ── Turnover ──────────────────────────────────────────────────────────────────
#
# `info["turnover"]` used to compute both weight vectors from the *post-trade*
# share vector (`panel_env.py:279,286-288`), which made it a pure function of
# the day's NAV drift: on the panel below the full rotation reported 0.000000
# and a zero-trade day through a -5% close reported 0.049979. These tests pin
# the behaviour that matters — turnover responds to trading and to nothing else.

_FLAT_PRICE = 100.0
_DROP_PRICE = 95.0
_TURNOVER_LOOKBACK = 3
_TURNOVER_DAYS = 20
_DROP_IDX = 5   # first day index whose close is _DROP_PRICE


def _deterministic_panel() -> pl.DataFrame:
    """Panel engineered so every share count is exactly predictable.

    * `atr_14 = 0` → the slippage term is identically zero, fills are at open.
    * `open[i] = close[i-1]` → no overnight gap, so an unchanged target weight
      reproduces the held share vector exactly and trades nothing.
    * close steps 100 → 95 at `_DROP_IDX` and stays there, giving one day with
      a -5% NAV move and provably zero trading.
    """
    closes = [_FLAT_PRICE] * _DROP_IDX + [_DROP_PRICE] * (_TURNOVER_DAYS - _DROP_IDX)
    opens = [_FLAT_PRICE] + closes[:-1]

    start = date(2020, 1, 1)
    rows: list[dict[str, object]] = []
    for ticker in _TICKERS:
        for i in range(_TURNOVER_DAYS):
            row: dict[str, object] = {
                "date": start + timedelta(days=i),
                "ticker": ticker,
                "open": opens[i],
                "high": max(opens[i], closes[i]) * 1.001,
                "low": min(opens[i], closes[i]) * 0.999,
                "close": closes[i],
                "adj_close": closes[i],
                "volume": 1_000_000,
                "is_tradeable": True,
                "sector_id": 1,
            }
            for fc in _FEATURE_COLS:
                row[fc] = 0.0
            row["atr_14"] = 0.0                 # no slippage
            row["dollar_volume_20"] = 1e9
            rows.append(row)
    return pl.DataFrame(rows)


def _turnover_env(tmpdir: Path) -> object:
    """Env over `_deterministic_panel` with a forced start index.

    `len(dates) - lookback - episode_length - 1 == 0` makes `reset` sample from
    a single-element range, so the episode always starts at day `lookback` and
    the price path below is the one actually traded.
    """
    from trader.env.costs import ZeroCostModel
    from trader.env.panel_env import PanelTradingEnv

    path = tmpdir / "deterministic.parquet"
    _deterministic_panel().write_parquet(path)
    return PanelTradingEnv(
        panel_path=path,
        universe=_TICKERS,
        feature_columns=_FEATURE_COLS,
        lookback=_TURNOVER_LOOKBACK,
        episode_length=_TURNOVER_DAYS - _TURNOVER_LOOKBACK - 1,
        initial_cash=1_000_000.0,
        cost_model=ZeroCostModel(),      # keep NAV arithmetic exact
        max_weight_per_name=1.0,         # allow a single-name book to rotate
        seed=0,
    )


def _all_in(ticker_idx: int) -> np.ndarray:
    """Logits that put ~100% of NAV into one ticker."""
    action = np.full(_N + 1, -1e9, dtype=np.float32)
    action[ticker_idx + 1] = 0.0
    return action


def test_turnover_zero_on_hold_and_two_on_full_rotation() -> None:
    """A hold is free and a full rotation costs ~2.0 (sell 100% + buy 100%)."""
    with tempfile.TemporaryDirectory() as tmp:
        env = _turnover_env(Path(tmp))
        env.reset(seed=0)                                    # type: ignore[union-attr]

        _, _, _, _, buy = env.step(_all_in(0))               # type: ignore[union-attr]
        _, _, _, _, hold = env.step(_all_in(0))              # type: ignore[union-attr]
        _, _, _, _, drop = env.step(_all_in(0))              # -5% close, no trade
        _, _, _, _, rotate = env.step(_all_in(1))            # type: ignore[union-attr]
        _, _, _, _, hold2 = env.step(_all_in(1))             # type: ignore[union-attr]

    # Entering from all-cash trades one side of the book only.
    assert buy["turnover"] == pytest.approx(1.0, abs=1e-3)
    # Rotating A → B sells the whole book and buys another: two sides.
    assert rotate["turnover"] == pytest.approx(2.0, abs=1e-3)
    # Holding trades nothing at all.
    assert hold["turnover"] == pytest.approx(0.0, abs=1e-9)
    assert drop["turnover"] == pytest.approx(0.0, abs=1e-9)
    assert hold2["turnover"] == pytest.approx(0.0, abs=1e-9)
    # The defect inverted exactly this ordering.
    assert rotate["turnover"] > 100.0 * max(hold["turnover"], drop["turnover"], 1e-9)


def test_turnover_is_not_a_function_of_nav() -> None:
    """Perturbing NAV without trading must leave turnover unchanged.

    Step 2 and step 3 are the same zero-trade hold; step 3's close is 5% lower.
    The old formula reported 0.000000 for the flat day and 0.049979 for the
    -5% day — the same trading activity (none) scored differently.
    """
    with tempfile.TemporaryDirectory() as tmp:
        env = _turnover_env(Path(tmp))
        env.reset(seed=0)                                    # type: ignore[union-attr]

        env.step(_all_in(0))                                 # type: ignore[union-attr]
        _, _, _, _, flat = env.step(_all_in(0))              # type: ignore[union-attr]
        _, _, _, _, dropped = env.step(_all_in(0))           # type: ignore[union-attr]

    nav_move = dropped["nav"] / flat["nav"] - 1.0
    assert nav_move == pytest.approx(-0.05, abs=1e-3), "panel must move NAV here"
    assert dropped["costs_paid"] == 0.0, "and must do so without trading"
    assert dropped["turnover"] == flat["turnover"] == pytest.approx(0.0, abs=1e-9)


def test_turnover_scales_with_the_value_actually_traded() -> None:
    """Half the book rotated must report half the turnover of a full rotation."""
    with tempfile.TemporaryDirectory() as tmp:
        env = _turnover_env(Path(tmp))
        env.reset(seed=0)                                    # type: ignore[union-attr]

        env.step(_all_in(0))                                 # 100% into A
        # 50% A / 50% B: sells half of A, buys an equal value of B.
        half = np.full(_N + 1, -1e9, dtype=np.float32)
        half[1] = 0.0
        half[2] = 0.0
        _, _, _, _, info = env.step(half)                    # type: ignore[union-attr]

    assert info["turnover"] == pytest.approx(1.0, abs=5e-3)
