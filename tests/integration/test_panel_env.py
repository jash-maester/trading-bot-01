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


# ── R5: rebalance schedule ────────────────────────────────────────────────────
#
# `rebalance_schedule` is the second-biggest profit lever in the system
# (`10_architecture_revamp.md` §4): delivery STT is 0.1% on both legs and STCG
# is 20% under twelve months, so a rebalance the agent did not need is paid for
# twice.  The env — not the agent — owns the cadence, because `step` re-derives
# target shares from the action every call, so an agent returning cached logits
# still trades the drift back out.  These tests pin the two things that has to
# mean: a hold day trades *nothing*, and `rebalance_schedule=None` is the old
# behaviour unchanged.

_SCHED_TICKERS = [f"T{i:02d}.NS" for i in range(40)]
_SCHED_LOOKBACK = 20
_SCHED_EPISODE = 252
_SCHED_DAYS = 400


def _business_days(start: date, n: int) -> list[date]:
    """`n` consecutive weekdays — a stand-in NSE calendar with no holidays."""
    out: list[date] = []
    d = start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def _schedule_panel() -> pl.DataFrame:
    """A weekday-indexed panel wide enough for a top-20 selection to bite.

    Weekdays, not `_make_panel`'s consecutive calendar days: a monthly schedule
    over calendar days would put ~30 days in a month and the trading-day
    arithmetic below (≈21 days per month) would not hold.
    """
    rng = np.random.default_rng(7)
    dates = _business_days(date(2018, 1, 1), _SCHED_DAYS)
    rows: list[dict[str, object]] = []
    for ti, ticker in enumerate(_SCHED_TICKERS):
        price = 500.0
        for d in dates:
            price = max(1.0, price * (1.0 + rng.normal(0.0004, 0.015)))
            row: dict[str, object] = {
                "date": d,
                "ticker": ticker,
                "open": price * 0.999,
                "high": price * 1.005,
                "low": price * 0.995,
                "close": price,
                "adj_close": price,
                "volume": 1_000_000,
                "is_tradeable": True,
                "sector_id": (ti % 4) + 1,
            }
            for fc in _FEATURE_COLS:
                row[fc] = float(rng.normal(0.0, 0.02))
            row["atr_14"] = price * 0.015
            row["dollar_volume_20"] = price * 5e6
            row["realized_vol_20d"] = float(rng.uniform(0.15, 0.45))
            rows.append(row)
    return pl.DataFrame(rows)


@pytest.fixture(scope="module")
def schedule_panel_file() -> Path:  # type: ignore[misc]
    tmp = tempfile.mkdtemp()
    path = Path(tmp) / "sched.parquet"
    _schedule_panel().write_parquet(path)
    return path


def _sched_env(path: Path, schedule: object = None, **kw: object) -> object:
    from trader.env.panel_env import PanelTradingEnv

    return PanelTradingEnv(
        panel_path=path,
        universe=_SCHED_TICKERS,
        feature_columns=_FEATURE_COLS,
        lookback=_SCHED_LOOKBACK,
        episode_length=_SCHED_EPISODE,
        initial_cash=1_000_000.0,
        seed=0,
        rebalance_schedule=schedule,   # type: ignore[arg-type]
        **kw,                          # type: ignore[arg-type]
    )


def _random_actions(n: int, width: int, seed: int = 5) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [rng.normal(0.0, 1.0, width).astype(np.float32) for _ in range(n)]


def _trajectory(env: object, actions: list[np.ndarray]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for a in actions:
        _, reward, term, trunc, info = env.step(a)   # type: ignore[union-attr]
        out.append({
            "nav": info["nav"],
            "turnover": info["turnover"],
            "costs": info["costs_paid"],
            "reward": float(reward),
            "weights": info["weights"].copy(),
            "rebalanced": info["rebalanced"],
        })
        if term or trunc:
            break
    return out


def test_schedule_none_is_identical_to_an_explicit_daily_schedule(
    schedule_panel_file: Path,
) -> None:
    """`rebalance_schedule=None` must be the pre-schedule env, exactly.

    A large suite of pre-existing tests exercises the `None` path; this asserts
    the stronger property that `None` and `RebalanceSchedule("daily")` produce
    bit-identical NAV, turnover, cost, reward and weight streams — i.e. the
    hold branch is genuinely unreachable rather than merely usually skipped.
    """
    from trader.allocator import RebalanceSchedule

    actions = _random_actions(60, len(_SCHED_TICKERS) + 1)

    a = _sched_env(schedule_panel_file, None)
    a.reset(seed=3)                                          # type: ignore[union-attr]
    traj_none = _trajectory(a, actions)

    b = _sched_env(schedule_panel_file, RebalanceSchedule("daily"))
    b.reset(seed=3)                                          # type: ignore[union-attr]
    traj_daily = _trajectory(b, actions)

    assert len(traj_none) == len(traj_daily) == 60
    for i, (x, y) in enumerate(zip(traj_none, traj_daily, strict=True)):
        assert x["nav"] == y["nav"], i          # exact float equality, not approx
        assert x["turnover"] == y["turnover"], i
        assert x["costs"] == y["costs"], i
        assert x["reward"] == y["reward"], i
        np.testing.assert_array_equal(x["weights"], y["weights"])
        assert x["rebalanced"] is True and y["rebalanced"] is True, i


def test_hold_day_trades_nothing_but_still_marks_to_market(
    schedule_panel_file: Path,
) -> None:
    """On a non-rebalance day: no trade, no cost, turnover exactly 0.0 — and a
    NAV that still moves with the close, and a reward that is still computed."""
    from trader.allocator import RebalanceSchedule

    env = _sched_env(schedule_panel_file, RebalanceSchedule("monthly"))
    env.reset(seed=3)                                        # type: ignore[union-attr]
    actions = _random_actions(80, len(_SCHED_TICKERS) + 1)
    traj = _trajectory(env, actions)

    holds = [r for r in traj if not r["rebalanced"]]
    trades = [r for r in traj if r["rebalanced"]]
    assert len(holds) > 50, "a monthly schedule over 80 days must mostly hold"
    assert len(trades) >= 3

    for r in holds:
        assert r["turnover"] == 0.0          # exactly, not approximately
        assert r["costs"] == 0.0
    # NAV still moves on hold days, and the reward is a real number.
    navs = [float(r["nav"]) for r in holds]
    assert len(set(navs)) > 10, "hold days must still mark the book to close"
    assert all(math.isfinite(float(r["reward"])) for r in holds)
    assert any(float(r["reward"]) != 0.0 for r in holds)


def test_hold_day_ignores_the_action_entirely(schedule_panel_file: Path) -> None:
    """Two runs that differ only in the actions fed on hold days must agree.

    This is the property that makes the cadence real: if the action leaked
    through on a hold day the env would trade, and a "monthly" run would pay
    daily STT while reporting a monthly schedule.
    """
    from trader.allocator import RebalanceSchedule

    n = 80
    base = _random_actions(n, len(_SCHED_TICKERS) + 1, seed=5)

    probe = _sched_env(schedule_panel_file, RebalanceSchedule("monthly"))
    probe.reset(seed=3)                                      # type: ignore[union-attr]
    flags = [bool(r["rebalanced"]) for r in _trajectory(probe, base)]

    rng = np.random.default_rng(999)
    scrambled = [
        a if flags[i] else rng.normal(0.0, 50.0, len(a)).astype(np.float32)
        for i, a in enumerate(base)
    ]

    e1 = _sched_env(schedule_panel_file, RebalanceSchedule("monthly"))
    e1.reset(seed=3)                                         # type: ignore[union-attr]
    t1 = _trajectory(e1, base)
    e2 = _sched_env(schedule_panel_file, RebalanceSchedule("monthly"))
    e2.reset(seed=3)                                         # type: ignore[union-attr]
    t2 = _trajectory(e2, scrambled)

    assert any(not f for f in flags), "the probe must have found some hold days"
    for i, (x, y) in enumerate(zip(t1, t2, strict=True)):
        assert x["nav"] == y["nav"], i
        assert x["turnover"] == y["turnover"], i
        np.testing.assert_array_equal(x["weights"], y["weights"])


def test_first_step_of_an_episode_always_trades(schedule_panel_file: Path) -> None:
    """A monthly episode starting mid-month must not sit in cash for 20 days."""
    from trader.allocator import RebalanceSchedule

    env = _sched_env(schedule_panel_file, RebalanceSchedule("monthly"))
    for seed in range(6):
        env.reset(seed=seed)                                 # type: ignore[union-attr]
        assert env.is_rebalance_step() is True               # type: ignore[union-attr]
        _, _, _, _, info = env.step(                         # type: ignore[union-attr]
            np.zeros(len(_SCHED_TICKERS) + 1, dtype=np.float32)
        )
        assert info["rebalanced"] is True
        assert info["turnover"] > 0.0, "the first step must actually enter the book"


def test_step_weights_takes_allocator_output_verbatim(
    schedule_panel_file: Path,
) -> None:
    """`step_weights` must place the allocator's weights without a softmax."""
    from trader.allocator import AllocatorParams, allocate

    env = _sched_env(schedule_panel_file, None, max_weight_per_name=1.0)
    obs, _ = env.reset(seed=3)                               # type: ignore[union-attr]
    n = len(_SCHED_TICKERS)
    rng = np.random.default_rng(1)
    params = AllocatorParams(
        k=10, max_name_weight=0.20, max_sector_weight=0.40, turnover_budget=2.0
    )
    target = allocate(
        rng.normal(0.0, 1.0, n),
        np.full(n, 0.25),
        obs["mask"].astype(bool),
        obs["sector_ids"].astype(np.int64),
        obs["portfolio"].astype(np.float64),
        params,
    )
    _, _, _, _, info = env.step_weights(target)              # type: ignore[union-attr]

    realised = info["weights"][1:]
    assert int((realised > 1e-6).sum()) == 10, "exactly the 10 chosen names"
    # Integer share rounding is the only difference; 0.2% is generous slack.
    np.testing.assert_allclose(realised, target[1:], atol=2e-3)
    assert np.all(realised <= 0.20 + 2e-3), "no softmax cap was re-applied"


def test_step_weights_holds_on_a_non_rebalance_day(schedule_panel_file: Path) -> None:
    from trader.allocator import RebalanceSchedule

    n = len(_SCHED_TICKERS)
    env = _sched_env(schedule_panel_file, RebalanceSchedule("monthly"))
    obs, _ = env.reset(seed=3)                               # type: ignore[union-attr]
    flat = np.full(n + 1, 1.0 / (n + 1))
    env.step_weights(flat)                                   # type: ignore[union-attr]
    assert env.is_rebalance_step() is False                  # type: ignore[union-attr]
    # A wildly different target on a hold day must still trade nothing.
    all_cash = np.zeros(n + 1)
    all_cash[0] = 1.0
    _, _, _, _, info = env.step_weights(all_cash)            # type: ignore[union-attr]
    assert info["rebalanced"] is False
    assert info["turnover"] == 0.0
    assert info["costs_paid"] == 0.0
    assert info["weights"][0] < 0.5, "the book must still be invested"


def test_step_weights_rejects_malformed_targets(schedule_panel_file: Path) -> None:
    n = len(_SCHED_TICKERS)
    env = _sched_env(schedule_panel_file, None)
    env.reset(seed=3)                                        # type: ignore[union-attr]
    good = np.full(n + 1, 1.0 / (n + 1))
    for bad in (good[:-1], good * 2.0, np.where(np.arange(n + 1) == 1, -0.1, good)):
        with pytest.raises(ValueError):
            env.step_weights(bad)                            # type: ignore[union-attr]


# ── R5: every baseline honours the env's cadence ──────────────────────────────


def _annual_turnover(path: Path, agent: object, schedule: object) -> tuple[float, int]:
    """(annualised gross turnover, number of days the env actually traded)."""
    env = _sched_env(path, schedule)
    obs, _ = env.reset(seed=11)                              # type: ignore[union-attr]
    agent.reset()                                            # type: ignore[attr-defined]
    turns: list[float] = []
    n_traded = 0
    done = False
    while not done:
        obs, _, term, trunc, info = env.step(agent.act(obs))  # type: ignore[union-attr,attr-defined]
        turns.append(float(info["turnover"]))
        n_traded += int(bool(info["rebalanced"]))
        done = term or trunc
    return float(np.sum(turns)) * 252.0 / len(turns), n_traded


def _baselines() -> dict[str, object]:
    from trader.env.baselines import (
        EqualWeightFrozenUniverse,
        EqualWeightRebalanced,
        MomentumTopK,
        RandomPolicy,
        SixtyFortyCash,
    )

    return {
        "equal_weight": EqualWeightRebalanced(),
        "equal_weight_frozen": EqualWeightFrozenUniverse(),
        "momentum_topk": MomentumTopK(),
        "sixty_forty": SixtyFortyCash(),
        "random": RandomPolicy(seed=0),
    }


def test_every_baseline_trades_only_on_schedule_days(schedule_panel_file: Path) -> None:
    """The cadence is enforced for every baseline, whatever it returns.

    `EqualWeightRebalanced` cannot rebalance monthly by returning cached
    logits — the env re-derives target shares every step, so cached logits
    still trade the drift out.  The fix is at the env level, so it applies to
    every agent uniformly; this asserts the day counts, which are a property
    of the schedule and nothing else.

    Over a 252-step episode on a weekday calendar: 12 month-starts plus the
    forced first step = 13 trading days (252 / 13 = 19.4 fewer trading days),
    and 51 week-starts.
    """
    from trader.allocator import RebalanceSchedule

    for name, agent in _baselines().items():
        daily = _annual_turnover(schedule_panel_file, agent, None)[1]
        weekly = _annual_turnover(
            schedule_panel_file, agent, RebalanceSchedule("weekly")
        )[1]
        monthly = _annual_turnover(
            schedule_panel_file, agent, RebalanceSchedule("monthly")
        )[1]
        assert daily == _SCHED_EPISODE, name
        assert weekly == 51, name
        assert monthly == 13, name
        assert _SCHED_EPISODE / monthly == pytest.approx(19.4, abs=0.1), name


def test_monthly_cadence_cuts_turnover_by_the_day_ratio_for_a_churning_agent(
    schedule_panel_file: Path,
) -> None:
    """A churning agent's turnover falls by ~the rebalance-day ratio (≈21x).

    This is the case `10_architecture_revamp.md` §1.1 is about: a policy whose
    target portfolio does not persist rebalances the *whole book* whenever it
    is allowed to, so its annual turnover is proportional to the number of days
    it is allowed to trade.  Measured on this panel (numbers regenerated by
    the assertions below): `momentum_topk` 241.8 → 12.9 (18.7x), `random`
    151.4 → 8.2 (18.5x), against a ceiling of 252/13 = 19.4x.

    The theoretical figure is 21x — 252 trading days / 12 month-starts — and
    the measured 18.7x is that ceiling minus the episode's forced first trade
    (13 trading days, not 12) minus the drift each held book still shows.
    """
    from trader.allocator import RebalanceSchedule

    for name in ("momentum_topk", "random"):
        daily = _annual_turnover(schedule_panel_file, _baselines()[name], None)[0]
        monthly = _annual_turnover(
            schedule_panel_file, _baselines()[name], RebalanceSchedule("monthly")
        )[0]
        ratio = daily / monthly
        assert 15.0 < ratio < 19.5, f"{name}: {daily:.1f} -> {monthly:.1f} = {ratio:.1f}x"


def test_monthly_cadence_cuts_a_persistent_baselines_turnover_far_less_than_21x(
    schedule_panel_file: Path,
) -> None:
    """Equal-weight does NOT get a 21x cut, and the reason is not a bug.

    Its target barely moves, so what it trades is accumulated drift — and drift
    on a random walk grows like √t, not t.  Twenty-one days of drift is ≈ √21
    ≈ 4.6 times one day's, so twelve monthly rebalances cost ≈ 12·√21 / 252 ≈
    1/4.6 of the daily bill, not 1/21.

    Measured, after `min_trade_value` began gating execution AND after the env
    stopped letting `floor` decide whether a trade happens (the half-share snap,
    `_step_target`; `tests/unit/test_weight_to_share_orders.py`):

        equal_weight          2.930 → 1.600   1.83x   monthly is 55% of daily
        equal_weight_frozen   2.930 → 1.600   1.83x
        sixty_forty           1.457 → 0.942   1.55x

    The bounds have now been widened downwards twice, both times for the same
    reason and both times because a real cost defect was removed rather than
    because a result drifted:

        pre-min_trade_value    equal_weight 3.96 → 1.63,  bounds 2.0 < r < 3.5
        post-min_trade_value   equal_weight 3.28 → 1.61,  bounds 1.7 < r < 3.5
        post-half-share snap   equal_weight 2.93 → 1.60,  bounds 1.4 < r < 3.5

    Each fix takes far more off the DAILY leg than the monthly one (-10.6% and
    -17.8% here against -0.5% and -0.7%), which narrows the ratio.  That is the
    mechanism, not a coincidence: a trade too small to carry a flat ₹15.34 fee —
    whether "too small" means under ₹500 or under half a share — is
    overwhelmingly a daily-cadence phenomenon, because one day of drift is
    ≈ √21 times smaller than twenty-one days of it.  See
    `11_cost_defect_and_fix_plan.md`, `tests/unit/test_min_trade_value.py` and
    `scripts/probe_share_rounding.py`.

    Recorded because the 21x figure is a real property of the *schedule* (see
    the day-count test above) and of a churning agent, and it would be wrong to
    quote it as equal-weight's saving.  Monthly still removes about half of
    equal-weight's turnover — a large number honestly stated.
    """
    from trader.allocator import RebalanceSchedule

    for name in ("equal_weight", "equal_weight_frozen", "sixty_forty"):
        daily = _annual_turnover(schedule_panel_file, _baselines()[name], None)[0]
        monthly = _annual_turnover(
            schedule_panel_file, _baselines()[name], RebalanceSchedule("monthly")
        )[0]
        ratio = daily / monthly
        assert 1.4 < ratio < 3.5, f"{name}: {daily:.2f} -> {monthly:.2f} = {ratio:.1f}x"
        assert monthly < 0.70 * daily, f"{name}: monthly {monthly:.3f} vs daily {daily:.3f}"

# ── R5: the allocator driving the env ─────────────────────────────────────────
#
# One number to keep in view in both tests below.  `allocate` budgets turnover
# against `current_w`, which the env reports marked at **yesterday's close**
# (`info["weights"]` / `obs["portfolio"]`), but the env fills at **today's
# open**.  A held position worth `s · prev_close` yesterday is worth `s · open`
# when it trades, so the value actually traded differs from the value budgeted
# by the overnight gap on the names being traded, and `info["turnover"]` can
# sit slightly *above* `turnover_budget`.  Measured on this panel (≈1.5% daily
# vol, `open = 0.999 · close`): 0.300 budgeted → 0.305 realised, and 0.250 →
# 0.260 — a 2–4% overshoot.
#
# It is not fixable inside the pinned `allocate(r_hat, vol, mask, sector_ids,
# current_w, params)` signature, which is given no price at which the trade
# will fill, and it is second-order next to what the budget is for (stopping a
# signal flip from rotating the entire book, a 2.0). So both tests assert the
# allocator's exact promise on the weights it was given, and the realised
# figure against a tolerance that names the mechanism.

# realised / budgeted. The documented and measured overshoot is 2-4% (0.300 ->
# 0.305, 0.250 -> 0.260); 1.15 was 4x looser than the mechanism it names, so it
# would have absorbed a real budget bug without failing.
_GAP_TOL = 1.05


def test_allocator_drives_a_full_episode_within_its_own_constraints(
    schedule_panel_file: Path,
) -> None:
    """`allocate()` → `step_weights()` over a whole monthly episode.

    This is the loop `scripts/run_allocator.py` runs, minus Hydra and MLflow.
    It asserts the constraints the allocator promises are still true *after*
    the env has rounded to integer shares and marked to close — the caps and
    the turnover budget are only worth something if they survive that.
    """
    from trader.allocator import AllocatorParams, RebalanceSchedule, allocate

    n = len(_SCHED_TICKERS)
    rng = np.random.default_rng(17)
    params = AllocatorParams(
        k=15, max_name_weight=0.10, max_sector_weight=0.25, turnover_budget=0.30
    )
    env = _sched_env(
        schedule_panel_file, RebalanceSchedule("monthly"), max_weight_per_name=1.0
    )
    obs, _ = env.reset(seed=3)                               # type: ignore[union-attr]

    n_reb, n_hold, steps = 0, 0, 0
    done = False
    while not done and steps < 120:
        if env.is_rebalance_step():                          # type: ignore[union-attr]
            current = obs["portfolio"].astype(np.float64)
            target = allocate(
                rng.normal(0.0, 1.0, n),
                np.full(n, 0.25),
                obs["mask"].astype(bool),
                obs["sector_ids"].astype(np.int64),
                current,
                params,
            )
            assert np.all(target >= 0.0)
            assert float(target.sum()) == pytest.approx(1.0, abs=1e-9)
            # The exact promise: gross planned move over equities, against the
            # weights the allocator was actually handed.
            planned = float(np.abs(target[1:] - current[1:]).sum())
            assert planned <= params.turnover_budget + 1e-6
            assert np.all(target[1:] <= params.max_name_weight + 1e-9)
            sec_w = np.bincount(
                obs["sector_ids"].astype(np.int64), weights=target[1:]
            )
            assert np.all(sec_w <= params.max_sector_weight + 1e-9)

            obs, reward, term, trunc, info = env.step_weights(target)  # type: ignore[union-attr]
            n_reb += 1
            assert info["turnover"] <= params.turnover_budget * _GAP_TOL
        else:
            obs, reward, term, trunc, info = env.step(       # type: ignore[union-attr]
                np.zeros(n + 1, dtype=np.float32)
            )
            n_hold += 1
            assert info["turnover"] == 0.0
            assert info["costs_paid"] == 0.0
        w = info["weights"]
        assert np.all(w[1:] <= params.max_name_weight + 5e-3)
        assert math.isfinite(float(reward)) and math.isfinite(float(info["nav"]))
        assert not np.isnan(obs["portfolio"]).any()
        done = bool(term or trunc)
        steps += 1

    assert n_reb >= 5 and n_hold > 90, (n_reb, n_hold)


def test_allocator_never_rotates_the_whole_book_on_a_signal_flip(
    schedule_panel_file: Path,
) -> None:
    """A full signal reversal must cost the budget, not a full rotation (2.0).

    This is the case the budget exists for.  Without it a flipped signal sells
    everything and buys something else in one day, which at the verified
    delivery round trip (`costs.py`) is the most expensive single thing this
    system can do.
    """
    from trader.allocator import AllocatorParams, allocate

    n = len(_SCHED_TICKERS)
    rng = np.random.default_rng(23)
    signal = rng.normal(0.0, 1.0, n)
    params = AllocatorParams(
        k=10, max_name_weight=0.15, max_sector_weight=0.40, turnover_budget=0.25
    )
    env = _sched_env(schedule_panel_file, None, max_weight_per_name=1.0)
    obs, _ = env.reset(seed=3)                               # type: ignore[union-attr]

    def one(sig: np.ndarray) -> tuple[dict[str, np.ndarray], float, float]:
        current = obs["portfolio"].astype(np.float64)
        target = allocate(
            sig, np.full(n, 0.25), obs["mask"].astype(bool),
            obs["sector_ids"].astype(np.int64), current, params,
        )
        # Against the *normalised* book, which is what `allocate` budgets
        # against.  `obs["portfolio"]` does not always sum to 1: its space is
        # `Box(0, 1)`, so when the env's cash goes slightly negative — it buys
        # at the open against a NAV marked at the previous close, so a gap up
        # overspends — the clipped cash reads 0.0 and the vector sums to more
        # than 1.  Measured here at step 12: cash 0.0, equity 1.009760.  The
        # allocator renormalises before budgeting (`_finish`), which is the
        # only sane reading of "30% of the book".
        norm = current / max(float(current.sum()), 1e-12)
        planned = float(np.abs(target[1:] - norm[1:]).sum())
        nxt, _, _, _, inf = env.step_weights(target)         # type: ignore[union-attr]
        return nxt, planned, float(inf["turnover"])

    for _ in range(12):                                      # settle onto the signal
        obs, _, realised = one(signal)
    assert realised < 0.02, "a stable signal must eventually stop trading"

    for _ in range(3):                                       # now reverse it
        obs, planned, realised = one(-signal)
        assert planned <= params.turnover_budget + 1e-6
        assert realised <= params.turnover_budget * _GAP_TOL
        # The point of the whole exercise: nowhere near a 2.0 rotation.
        assert realised < 0.5
