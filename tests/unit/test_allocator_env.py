"""R6 — AllocatorEnv: one step is one rebalance period, one action is a few scalars.

Every panel here is synthetic.  The panels on disk are stale relative to
``active_tickers()`` (504 vs 163), so nothing in this project may test against
the real panel matching the real universe; these tests build their own calendar
and their own universe and check the env's invariants against those.
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from trader.allocator.deterministic import AllocatorParams, allocate
from trader.allocator.rebalance import RebalanceSchedule
from trader.env.allocator_env import (
    N_SECTORS,
    ActionRanges,
    AllocatorEnv,
    AllocatorEnvConfig,
    InnerEnvMisconfigured,
    SignalPanel,
)
from trader.env.panel_env import PanelTradingEnv
from trader.env.reward import LogReturn

FEATURES = ["log_return_1d"]
LOOKBACK = 20


# ── synthetic panel ───────────────────────────────────────────────────────────


def _trading_dates(n: int, start: date = date(2020, 1, 1)) -> list[date]:
    out: list[date] = []
    d = start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def _write_panel(
    path: Path, T: int = 420, N: int = 24, seed: int = 0, gap_sd: float = 0.0
) -> tuple[list[str], list[date], np.ndarray]:
    """A panel that is gap-free by default: ``open[t] == close[t-1]``.

    ``gap_sd > 0`` injects a lognormal overnight gap between the previous close
    and today's open, which is what makes the turnover budget slack real: the
    env budgets against NAV at the previous close and fills at the open.

    The gap-free choice is deliberate.  The env sizes the trade off NAV marked
    at yesterday's close and executes at today's open, so with an overnight gap
    the realised turnover differs from the allocator's ``|Δw|`` by the gap
    itself.  Removing the gap lets the turnover-budget test assert a tight
    bound and attribute any breach to the allocator, not to the market.
    """
    rng = np.random.default_rng(seed)
    dates = _trading_dates(T)
    tickers = [f"T{i:02d}" for i in range(N)]
    rets = rng.normal(0.0004, 0.012, size=(T, N))
    rets[0] = 0.0
    close = 100.0 * np.exp(np.cumsum(rets, axis=0))
    open_ = np.vstack([close[:1], close[:-1]])
    if gap_sd > 0.0:
        open_ = open_ * np.exp(rng.normal(0.0, gap_sd, size=open_.shape))
    vol = np.full((T, N), np.nan)
    for t in range(T):
        lo = max(0, t - 19)
        vol[t] = np.std(rets[lo : t + 1], axis=0) * np.sqrt(252.0)
    vol = np.maximum(np.nan_to_num(vol, nan=0.2), 0.05)

    rows: list[dict[str, object]] = []
    for ti, d in enumerate(dates):
        for ni, tk in enumerate(tickers):
            rows.append(
                {
                    "date": d,
                    "ticker": tk,
                    "open": float(open_[ti, ni]),
                    "close": float(close[ti, ni]),
                    "is_tradeable": True,
                    "sector_id": 1 + (ni % N_SECTORS),
                    "atr_14": float(close[ti, ni] * 0.01),
                    "dollar_volume_20": 1e10,
                    "log_return_1d": float(rets[ti, ni]),
                    "realized_vol_20d": float(vol[ti, ni]),
                }
            )
    pl.DataFrame(rows).write_parquet(path)
    return tickers, dates, vol


def _signal(
    tickers: list[str], dates: list[date], vol: np.ndarray, seed: int = 1
) -> SignalPanel:
    """A SignalPanel over the fixture panel, with vol lagged one day.

    ``_write_panel`` writes ``realized_vol_20d`` the way the REAL panel carries
    it — a window ending on date t inclusive, so it embeds date t's close.  The
    SignalPanel contract (see the CONTRACT note on ``SignalPanel.vol``) is that
    ``vol[t]`` is knowable *before* date t's open, so anything building one
    directly has to lag it, exactly as ``from_artifacts`` does.

    The fixture is deliberately left faithful rather than made exclusive: a
    fixture that pre-lags would agree with a broken ``from_artifacts`` and hide
    the leak instead of catching it.
    """
    rng = np.random.default_rng(seed)
    r_hat = rng.normal(0.0, 1.0, size=(len(dates), len(tickers)))
    lagged = np.full_like(vol, np.nan)
    lagged[1:] = vol[:-1]
    return SignalPanel(dates=list(dates), tickers=list(tickers), r_hat=r_hat, vol=lagged)


def _inner(
    path: Path,
    tickers: list[str],
    *,
    freq: str = "monthly",
    use_excess_returns: bool = True,
    turnover_penalty: float = 0.0,
    episode_length: int = 300,
) -> PanelTradingEnv:
    return PanelTradingEnv(
        panel_path=path,
        universe=tickers,
        feature_columns=FEATURES,
        lookback=LOOKBACK,
        episode_length=episode_length,
        initial_cash=1_000_000.0,
        reward_fn=LogReturn(),
        turnover_penalty=turnover_penalty,
        use_excess_returns=use_excess_returns,
        seed=0,
        rebalance_schedule=RebalanceSchedule(freq=freq),  # type: ignore[arg-type]
    )


def _env(
    tmp_path: Path, *, freq: str = "monthly", **cfg_kw: object
) -> tuple[AllocatorEnv, list[str]]:
    p = tmp_path / "panel.parquet"
    tickers, dates, vol = _write_panel(p)
    inner = _inner(p, tickers, freq=freq)
    cfg = AllocatorEnvConfig(**cfg_kw)  # type: ignore[arg-type]
    return AllocatorEnv(inner, _signal(tickers, dates, vol), cfg, seed=0), tickers


# ── action encoding ───────────────────────────────────────────────────────────


def test_action_dim_and_ranges() -> None:
    r = ActionRanges()
    assert r.action_dim == 3 + N_SECTORS
    lo, _ = r.decode(np.zeros(r.action_dim), max_name_weight=0.1,
                     max_sector_weight=0.25, vol_lookback=20)
    hi, hi_tilt = r.decode(np.ones(r.action_dim), max_name_weight=0.1,
                           max_sector_weight=0.25, vol_lookback=20)
    assert (lo.k, hi.k) == (10, 60)
    assert lo.turnover_budget == pytest.approx(0.05)
    assert hi.turnover_budget == pytest.approx(1.0)
    assert lo.cash_floor == pytest.approx(0.0)
    assert hi.cash_floor == pytest.approx(0.5)
    assert hi_tilt.min() == pytest.approx(1.0)


def test_decode_encode_round_trips() -> None:
    r = ActionRanges()
    a = np.linspace(0.05, 0.95, r.action_dim)
    params, tilt = r.decode(a, max_name_weight=0.1, max_sector_weight=0.25, vol_lookback=20)
    back = r.encode(params, tilt)
    # K is rounded to an integer, so its round trip is exact only to 1/50th.
    np.testing.assert_allclose(back[1:], a[1:], atol=1e-6)
    assert abs(float(back[0]) - a[0]) <= 0.5 / (r.k_max - r.k_min) + 1e-9


def test_out_of_range_actions_are_clipped_not_raised() -> None:
    """A replayed or hand-written action must not blow up mid-rollout."""
    r = ActionRanges()
    params, tilt = r.decode(
        np.full(r.action_dim, 5.0), max_name_weight=0.1,
        max_sector_weight=0.25, vol_lookback=20,
    )
    assert params.k == 60 and params.turnover_budget == pytest.approx(1.0)
    assert float(tilt.max()) == pytest.approx(1.0)


# ── construction refusals ─────────────────────────────────────────────────────


def test_raw_log_return_inner_env_is_refused(tmp_path: Path) -> None:
    """§1.3: with raw log return the agent is paid for being long, not for selecting."""
    p = tmp_path / "panel.parquet"
    tickers, dates, vol = _write_panel(p, T=120, N=8)
    inner = _inner(p, tickers, use_excess_returns=False, episode_length=80)
    with pytest.raises(InnerEnvMisconfigured, match="use_excess_returns"):
        AllocatorEnv(inner, _signal(tickers, dates, vol))


def test_double_turnover_charge_is_refused(tmp_path: Path) -> None:
    p = tmp_path / "panel.parquet"
    tickers, dates, vol = _write_panel(p, T=120, N=8)
    inner = _inner(p, tickers, turnover_penalty=0.001, episode_length=80)
    with pytest.raises(InnerEnvMisconfigured, match="turnover_penalty"):
        AllocatorEnv(inner, _signal(tickers, dates, vol))


def test_mismatched_ticker_order_is_refused(tmp_path: Path) -> None:
    """Axis-1 order is a pinned contract; a silent mis-join picks the wrong stock."""
    p = tmp_path / "panel.parquet"
    tickers, dates, vol = _write_panel(p, T=120, N=8)
    inner = _inner(p, tickers, episode_length=80)
    shuffled = list(reversed(tickers))
    with pytest.raises(ValueError, match="ticker order"):
        AllocatorEnv(inner, _signal(shuffled, dates, vol))


# ── observation ───────────────────────────────────────────────────────────────


def test_observation_is_allocator_state_only(tmp_path: Path) -> None:
    """No per-stock tensor: §1.2's mean-pooled critic is what happens otherwise."""
    from trader.data.regime_features import REGIME_DIM

    env, _ = _env(tmp_path)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (env.obs_dim,)
    assert env.obs_dim == ActionRanges().action_dim + 5 + N_SECTORS + REGIME_DIM
    assert obs.dtype == np.float32
    assert np.isfinite(obs).all()
    assert env.observation_space.contains(obs)


def test_observation_width_does_not_scale_with_the_universe(tmp_path: Path) -> None:
    """The old critic's input was a mean-pool over N because N was in it (§1.2).

    Here N never enters the observation, so widening the traded universe from
    163 to 504 changes nothing about the policy or the critic.
    """
    widths: list[int] = []
    for i, n in enumerate((8, 40)):
        p = tmp_path / f"panel_{n}.parquet"
        tickers, dates, vol = _write_panel(p, T=200, N=n, seed=i)
        inner = _inner(p, tickers, episode_length=150)
        widths.append(AllocatorEnv(inner, _signal(tickers, dates, vol)).obs_dim)
    assert widths[0] == widths[1]


def test_observation_layout_tracks_the_book(tmp_path: Path) -> None:
    env, _ = _env(tmp_path)
    env.reset(seed=0)
    A, S = env.ranges.action_dim, env.ranges.n_sectors
    obs, *_ = env.step(np.full(A, 0.5, dtype=np.float32))
    cash, max_name = float(obs[A]), float(obs[A + 1])
    sectors = obs[A + 2 : A + 2 + S]
    assert 0.0 <= cash <= 1.0
    assert 0.0 <= max_name <= env.cfg.max_name_weight + 1e-6
    assert float(sectors.sum()) == pytest.approx(1.0 - cash, abs=1e-4)
    assert float(obs[A + 2 + S]) >= 0.0        # realised turnover
    assert 0.0 <= float(obs[A + 3 + S]) < 1.0  # drawdown
    assert float(obs[-1]) == pytest.approx(1.0 / env.cfg.periods_per_episode)


def test_action_in_force_is_echoed_in_the_observation(tmp_path: Path) -> None:
    env, _ = _env(tmp_path)
    env.reset(seed=0)
    A = env.ranges.action_dim
    action = np.linspace(0.1, 0.9, A).astype(np.float32)
    obs, *_ = env.step(action)
    np.testing.assert_allclose(obs[:A], action, atol=1e-6)


# ── step semantics ────────────────────────────────────────────────────────────


def test_one_step_is_one_month_and_trades_once(tmp_path: Path) -> None:
    """Monthly schedule: ~21 inner days per step, one of them a trading day."""
    env, _ = _env(tmp_path, freq="monthly")
    env.reset(seed=0)
    A = env.ranges.action_dim
    action = np.full(A, 0.5, dtype=np.float32)
    # The episode's random start lands mid-month, so the FIRST period runs only
    # to the next month boundary — the env always trades on step 0 rather than
    # leaving a fresh episode in cash for up to twenty days.  Steady state is
    # the second period onwards.
    env.step(action)
    _, _, _, _, info = env.step(action)
    assert 15 <= info["n_days"] <= 25, info["n_days"]
    assert info["costs_paid"] == 0.0, "the last day of the period must be a hold day"


def test_daily_schedule_makes_a_period_one_day(tmp_path: Path) -> None:
    env, _ = _env(tmp_path, freq="daily")
    env.reset(seed=0)
    _, _, _, _, info = env.step(np.full(env.ranges.action_dim, 0.5, dtype=np.float32))
    assert info["n_days"] == 1


def test_episode_truncates_after_the_configured_number_of_periods(tmp_path: Path) -> None:
    env, _ = _env(tmp_path, periods_per_episode=6)
    env.reset(seed=0)
    A = env.ranges.action_dim
    for i in range(6):
        _, _, term, trunc, _ = env.step(np.full(A, 0.5, dtype=np.float32))
        assert trunc == (i == 5), f"period {i}: truncated={trunc}"
        assert not term


def test_realised_turnover_never_exceeds_the_sampled_budget(tmp_path: Path) -> None:
    """The structural claim: the sampled budget IS the bound on the trade.

    Measured through the *env*, not the allocator, so integer share rounding,
    slippage and the open-vs-previous-close basis are all included.  The panel
    is gap-free (see ``_write_panel``), so the only slack left is slippage,
    which is ~1e-4 here.
    """
    rng = np.random.default_rng(0)
    env, _ = _env(tmp_path, periods_per_episode=10)
    env.reset(seed=0)
    A = env.ranges.action_dim
    worst = -1.0
    for _ in range(10):
        action = rng.uniform(0.0, 1.0, A).astype(np.float32)
        _, _, term, trunc, info = env.step(action)
        worst = max(worst, info["turnover"] - info["turnover_budget"])
        if term or trunc:
            break
    assert worst <= 1e-3, f"realised turnover exceeded its budget by {worst:.2e}"


def test_reward_is_excess_return_minus_the_two_penalties(tmp_path: Path) -> None:
    """Reproduce the reward from the info dict, term by term."""
    env, _ = _env(tmp_path, periods_per_episode=8, turnover_penalty=0.5,
                  drawdown_penalty=2.0, drawdown_threshold=0.0)
    env.reset(seed=0)
    A = env.ranges.action_dim
    for _ in range(4):
        _, reward, term, trunc, info = env.step(
            np.random.default_rng(0).uniform(0, 1, A).astype(np.float32)
        )
        expected = (
            info["excess_log_return"]
            - 0.5 * info["turnover"]
            - 2.0 * max(0.0, info["drawdown"])
        )
        assert float(reward) == pytest.approx(expected, abs=1e-9)
        if term or trunc:
            break


def test_a_zero_turnover_budget_freezes_the_book(tmp_path: Path) -> None:
    """Sanity on the bound's lower end: budget 0.05 keeps the trade tiny."""
    env, _ = _env(tmp_path, periods_per_episode=4)
    env.reset(seed=0)
    A = env.ranges.action_dim
    action = np.full(A, 0.5, dtype=np.float32)
    action[1] = 0.0                              # turnover_budget -> its minimum, 0.05
    _, _, _, _, info = env.step(action)
    assert info["turnover_budget"] == pytest.approx(0.05)
    assert info["turnover"] <= 0.05 + 1e-3


def test_no_signal_on_a_date_holds_rather_than_liquidating(tmp_path: Path) -> None:
    """A signal gap must not be traded on — it is the absence of information."""
    p = tmp_path / "panel.parquet"
    tickers, dates, vol = _write_panel(p)
    inner = _inner(p, tickers)
    sig = _signal(tickers, dates, vol)
    blank = SignalPanel(
        dates=sig.dates, tickers=sig.tickers,
        r_hat=np.full_like(sig.r_hat, np.nan), vol=sig.vol,
    )
    env = AllocatorEnv(inner, blank, AllocatorEnvConfig(periods_per_episode=3))
    env.reset(seed=0)
    A = env.ranges.action_dim
    _, _, _, _, info = env.step(np.full(A, 0.5, dtype=np.float32))
    assert info["had_signal"] is False
    assert info["turnover"] == pytest.approx(0.0, abs=1e-9)
    assert info["costs_paid"] == pytest.approx(0.0)


def test_signal_coverage_is_reported(tmp_path: Path) -> None:
    """A partially covered calendar must be visible, not silently absorbed."""
    p = tmp_path / "panel.parquet"
    tickers, dates, vol = _write_panel(p)
    half = len(dates) // 2
    sig = SignalPanel(
        dates=list(dates[half:]),
        tickers=tickers,
        r_hat=np.random.default_rng(0).normal(size=(len(dates) - half, len(tickers))),
        vol=vol[half:],
    )
    env = AllocatorEnv(_inner(p, tickers), sig)
    assert env.n_signal_dates == len(dates) - half


def test_seeded_reset_is_reproducible(tmp_path: Path) -> None:
    env, _ = _env(tmp_path, periods_per_episode=4)
    A = env.ranges.action_dim
    actions = np.random.default_rng(3).uniform(0, 1, (4, A)).astype(np.float32)

    def rollout() -> list[float]:
        env.reset(seed=17)
        return [float(env.step(a)[1]) for a in actions]

    assert rollout() == rollout()


# ── artefact loading ──────────────────────────────────────────────────────────


def test_signal_panel_from_artifacts(tmp_path: Path) -> None:
    p = tmp_path / "panel.parquet"
    tickers, dates, _ = _write_panel(p, T=60, N=6)
    sig_dir = tmp_path / "signal"
    sig_dir.mkdir()
    rng = np.random.default_rng(0)
    oos = dates[30:]
    pl.DataFrame(
        {
            "date": [d for d in oos for _ in tickers],
            "ticker": [t for _ in oos for t in tickers],
            "r_hat_5d": rng.normal(size=len(oos) * len(tickers)),
            "r_hat_20d": rng.normal(size=len(oos) * len(tickers)),
        }
    ).write_parquet(sig_dir / "predictions.parquet")

    sig = SignalPanel.from_artifacts(sig_dir, p, tickers)
    assert sig.dates == list(oos)
    assert sig.tickers == tickers
    assert sig.r_hat.shape == (len(oos), len(tickers))
    # Row 0 is NaN by design: `from_artifacts` lags vol one PANEL row, and the
    # signal's first date has a predecessor in the panel (the panel starts 30
    # days earlier), so here every row including the first is finite.
    assert np.isfinite(sig.vol).all(), "vol must align onto the signal's dates"
    # And it must be the PREVIOUS trading day's vol, not the same day's.
    panel = pl.read_parquet(p)
    v = panel.pivot(on="ticker", index="date", values="realized_vol_20d").sort("date")
    v_np = v.drop("date").to_numpy()
    prev_row = v_np[29]                 # panel row before oos[0] == dates[30]
    assert np.allclose(sig.vol[0], prev_row), (
        "from_artifacts must lag the vol column by one trading day: vol[t] is "
        "read before date t's open, but realized_vol_20d[t] embeds close_t"
    )
    assert not np.allclose(sig.vol[0], v_np[30]), "vol[0] is the SAME-day value"

    with pytest.raises(KeyError, match="r_hat_60d"):
        SignalPanel.from_artifacts(sig_dir, p, tickers, horizon="r_hat_60d")


def test_perturbing_a_dates_own_vol_cannot_move_that_dates_target(tmp_path: Path) -> None:
    """The lookahead regression test.

    ``realized_vol_20d[t]`` is computed from a window ending at date t's CLOSE
    (features.py:241-244, 261-265), but the allocator's trade fills at date t's
    OPEN.  Reading it same-day sizes today's trade with a price that has not
    printed — measured before the fix: perturbing only ``realized_vol_20d`` on
    the trade date moved the executed target by L1 0.268551.

    So: scramble one date's vol in the panel, and that date's allocation must
    be **bit-identical**.  The NEXT date's must change, which is what proves the
    perturbation was real and the test is not passing vacuously.
    """
    base = tmp_path / "panel.parquet"
    tickers, dates, _ = _write_panel(base, T=60, N=16)
    sig_dir = tmp_path / "signal"
    sig_dir.mkdir()
    rng = np.random.default_rng(0)
    oos = dates[30:]
    pl.DataFrame(
        {
            "date": [d for d in oos for _ in tickers],
            "ticker": [t for _ in oos for t in tickers],
            "r_hat_5d": rng.normal(size=len(oos) * len(tickers)),
            "r_hat_20d": rng.normal(size=len(oos) * len(tickers)),
        }
    ).write_parquet(sig_dir / "predictions.parquet")

    victim = oos[5]
    bumped = tmp_path / "panel_bumped.parquet"
    # The perturbation must be CROSS-SECTIONAL, not a uniform scale: inverse-vol
    # weights renormalise, so multiplying every name's vol by one constant is a
    # no-op and would make this test vacuous.  Give each ticker its own factor.
    factor = {t: 1.0 + 0.5 * i for i, t in enumerate(tickers)}
    pl.read_parquet(base).with_columns(
        pl.when(pl.col("date") == victim)
        .then(pl.col("realized_vol_20d") * pl.col("ticker").replace_strict(factor))
        .otherwise(pl.col("realized_vol_20d"))
        .alias("realized_vol_20d")
    ).write_parquet(bumped)

    a = SignalPanel.from_artifacts(sig_dir, base, tickers)
    b = SignalPanel.from_artifacts(sig_dir, bumped, tickers)
    i = a.dates.index(victim)

    # Caps deliberately loose: at the pinned defaults 16 names all sit on the
    # sector cap at 0.05 each, the allocation is vol-independent, and the test
    # cannot detect anything.  Here inverse-vol sizing actually drives weights.
    params = AllocatorParams(k=8, max_name_weight=0.5, max_sector_weight=1.0)
    mask = np.ones(len(tickers), dtype=bool)
    sids = np.array([1 + (n % N_SECTORS) for n in range(len(tickers))], dtype=np.int64)
    cur = np.r_[1.0, np.zeros(len(tickers))]

    w_a = allocate(a.r_hat[i], a.vol[i], mask, sids, cur, params)
    w_b = allocate(b.r_hat[i], b.vol[i], mask, sids, cur, params)
    assert np.array_equal(w_a, w_b), (
        f"date {victim}'s target moved by L1 "
        f"{np.abs(w_a - w_b).sum():.6f} when only that date's own vol changed — "
        f"the allocator is reading same-day vol, which embeds that date's close"
    )

    # The perturbation must land SOMEWHERE, or the test above proves nothing.
    w_a1 = allocate(a.r_hat[i + 1], a.vol[i + 1], mask, sids, cur, params)
    w_b1 = allocate(b.r_hat[i + 1], b.vol[i + 1], mask, sids, cur, params)
    assert not np.array_equal(w_a1, w_b1), (
        "the next date's target did not move either: the perturbation never "
        "reached the allocator and this test is vacuous"
    )


def test_turnover_budget_slack_on_a_gapped_panel_is_measured_not_assumed(
    tmp_path: Path,
) -> None:
    """`turnover_budget` is a bound *up to the overnight gap*, not a hard one.

    The env sizes against NAV marked at the previous close and fills at the
    open, so the value actually traded differs from the value budgeted by the
    gap on the traded names.  Every other budget test in this suite runs on the
    deliberately gap-free ``_write_panel``, so the slack was documented
    (deterministic.py: 0.300 -> 0.305) but never under measurement.  This puts
    it under measurement.
    """
    overruns: dict[float, float] = {}
    for gap_sd in (0.0, 0.01, 0.03):
        worst = -np.inf
        for sd in range(6):                       # one seed is far too few to
            p = tmp_path / f"panel_gap_{gap_sd}_{sd}.parquet"   # find the worst case
            tickers, dates, vol = _write_panel(p, T=200, N=10, seed=sd, gap_sd=gap_sd)
            env = AllocatorEnv(_inner(p, tickers), _signal(tickers, dates, vol, seed=sd), seed=sd)
            env.reset(seed=sd)
            for _ in range(8):
                _, _, term, trunc, info = env.step(env.action_space.sample())
                worst = max(worst, float(info["turnover"]) - float(info["turnover_budget"]))
                if term or trunc:
                    break
        overruns[gap_sd] = float(worst)
    print(f"turnover overrun vs sampled budget by overnight-gap sd: {overruns}")
    # Measured on this fixture (48 steps per gap level):
    #   gap sd 0.00 -> +0.000637   0.01 -> +0.003732   0.03 -> +0.010731
    # The bound is the gap, so it must stay small and must scale WITH the gap.
    assert overruns[0.0] <= 2e-3, overruns
    assert overruns[0.03] <= 3e-2, overruns
    assert overruns[0.03] > overruns[0.0], overruns


# ── end-to-end PPO smoke run (synthetic data, CPU, no GPU) ────────────────────


def test_ppo_loop_runs_end_to_end_on_synthetic_data(tmp_path: Path) -> None:
    """Proof the whole R6 stack executes: env -> policy -> GAE -> update.

    Not a performance claim of any kind.  Two envs, four periods, one update —
    it asserts that the loop completes, that the weights move, and that the
    model comes back in ``eval()``.  Any statement about whether this beats
    equal-weight requires a real run and an MLflow run id (CLAUDE.md rule 2).
    """
    import torch

    from trader.models.allocator_policy import AllocatorPolicy, AllocatorPolicyConfig
    from trader.training.ppo_allocator import PPOAllocatorConfig, PPOAllocatorTrainer

    p = tmp_path / "panel.parquet"
    tickers, dates, vol = _write_panel(p, T=420, N=20)
    sig = _signal(tickers, dates, vol)
    cfg = AllocatorEnvConfig(periods_per_episode=4)
    envs = [AllocatorEnv(_inner(p, tickers), sig, cfg, seed=i) for i in range(2)]

    torch.manual_seed(0)
    model = AllocatorPolicy(
        AllocatorPolicyConfig(obs_dim=envs[0].obs_dim, action_dim=envs[0].ranges.action_dim)
    )
    ppo_cfg = PPOAllocatorConfig(
        total_steps=8, n_envs=2, n_steps=4, n_epochs=2, n_minibatches=2,
        checkpoint_dir=tmp_path / "ckpt", log_interval=0, checkpoint_interval=0,
        target_kl=None,
    )
    before = {k: v.clone() for k, v in model.state_dict().items()}
    trainer = PPOAllocatorTrainer(envs, model, ppo_cfg, torch.device("cpu"))
    episodes = trainer.train()

    assert not model.training, "the trainer must leave the model in eval()"
    assert any(
        not torch.equal(v, before[k]) for k, v in model.state_dict().items()
    ), "one update produced no weight change"
    assert episodes, "n_steps == periods_per_episode, so every env finished an episode"
    for ep in episodes:
        assert ep.n_periods == 4
        assert np.isfinite(ep.excess_log_return)
        assert ep.mean_turnover >= 0.0
    assert set(trainer.last_metrics) >= {"pg_loss", "v_loss", "entropy", "approx_kl"}


def test_ppo_checkpoint_round_trips(tmp_path: Path) -> None:
    import torch

    from trader.models.allocator_policy import AllocatorPolicy, AllocatorPolicyConfig
    from trader.training.ppo_allocator import PPOAllocatorConfig, PPOAllocatorTrainer

    p = tmp_path / "panel.parquet"
    tickers, dates, vol = _write_panel(p, T=200, N=10)
    envs = [
        AllocatorEnv(
            _inner(p, tickers, episode_length=150),
            _signal(tickers, dates, vol),
            AllocatorEnvConfig(periods_per_episode=3),
        )
    ]
    model = AllocatorPolicy(
        AllocatorPolicyConfig(obs_dim=envs[0].obs_dim, action_dim=envs[0].ranges.action_dim)
    )
    cfg = PPOAllocatorConfig(
        total_steps=0, n_envs=1, n_steps=3, n_minibatches=1, checkpoint_dir=tmp_path / "ck"
    )
    path = PPOAllocatorTrainer(envs, model, cfg, torch.device("cpu")).save_checkpoint(1)
    blob = torch.load(path, weights_only=False)
    assert blob["update"] == 1
    assert set(blob["model_state"]) == set(model.state_dict())


# ── the R4 gate, enforced in code (CLAUDE.md rule 1) ──────────────────────────


def _load_train_script():  # type: ignore[no-untyped-def]
    import importlib.util

    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "train_allocator_rl", root / "scripts" / "train_allocator_rl.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _gate_dir(tmp_path: Path, payload: dict[str, object], with_preds: bool = True) -> Path:
    import json

    d = tmp_path / "signal"
    d.mkdir(exist_ok=True)
    (d / "gate.json").write_text(json.dumps(payload))
    if with_preds:
        pl.DataFrame(
            {"date": [date(2020, 1, 1)], "ticker": ["T00"], "r_hat_5d": [0.0], "r_hat_20d": [0.0]}
        ).write_parquet(d / "predictions.parquet")
    return d


def test_training_refuses_without_a_gate_file(tmp_path: Path) -> None:
    mod = _load_train_script()
    with pytest.raises(mod.SignalGateNotPassed, match="does not exist"):
        mod.require_passing_gate(tmp_path / "nope")


def test_training_refuses_on_a_failed_gate(tmp_path: Path) -> None:
    """A failed gate is a result, not an obstacle — and not a licence to train."""
    mod = _load_train_script()
    d = _gate_dir(tmp_path, {"verdict": "FAIL", "mean_ic_5d": 0.001, "n_windows": 6})
    with pytest.raises(mod.SignalGateNotPassed, match="'FAIL'"):
        mod.require_passing_gate(d)


def test_training_refuses_on_a_missing_verdict(tmp_path: Path) -> None:
    mod = _load_train_script()
    d = _gate_dir(tmp_path, {"mean_ic_5d": 0.05})
    with pytest.raises(mod.SignalGateNotPassed):
        mod.require_passing_gate(d)


def test_training_refuses_when_predictions_are_absent(tmp_path: Path) -> None:
    mod = _load_train_script()
    d = _gate_dir(tmp_path, {"verdict": "PASS"}, with_preds=False)
    with pytest.raises(mod.SignalGateNotPassed, match="predictions.parquet"):
        mod.require_passing_gate(d)


def test_training_accepts_a_passing_gate(tmp_path: Path) -> None:
    mod = _load_train_script()
    d = _gate_dir(
        tmp_path,
        {
            "verdict": "PASS", "mean_ic_5d": 0.031, "mean_ic_20d": 0.042,
            "ic_ci_low": 0.011, "ic_ci_high": 0.055, "icir": 0.42, "n_windows": 8,
        },
    )
    assert mod.require_passing_gate(d)["verdict"] == "PASS"


def test_there_is_no_force_flag() -> None:
    """Rule 1 in code has no escape hatch; keep it that way."""
    root = Path(__file__).resolve().parents[2]
    source = (root / "scripts" / "train_allocator_rl.py").read_text()
    for flag in ("--force", "--skip-gate", "--no-gate", "--ignore-gate"):
        assert f'add_argument("{flag}' not in source, f"{flag} is an escape hatch"
    # And the prose says so, so the next reader knows it was a decision.
    assert "no ``--force``" in source


def test_build_envs_from_the_shipped_config_and_a_synthetic_artefact(tmp_path: Path) -> None:
    """The config file, the artefact layout and the env constructor, together.

    Catches the class of bug where ``configs/env/allocator.yaml`` drifts from
    what ``build_envs`` reads — a dead config key is exactly the trap
    ``min_trade_value: 500`` set (declared, never read, 11% NAV divergence).
    """
    import json

    import yaml

    mod = _load_train_script()
    root = Path(__file__).resolve().parents[2]
    env_cfg = yaml.safe_load((root / "configs" / "env" / "allocator.yaml").read_text())

    p = tmp_path / "panel.parquet"
    tickers, dates, _ = _write_panel(p, T=420, N=16)
    sig_dir = tmp_path / "signal"
    sig_dir.mkdir()
    oos = dates[100:]
    rng = np.random.default_rng(0)
    pl.DataFrame(
        {
            "date": [d for d in oos for _ in tickers],
            "ticker": [t for _ in oos for t in tickers],
            "r_hat_5d": rng.normal(size=len(oos) * len(tickers)),
            "r_hat_20d": rng.normal(size=len(oos) * len(tickers)),
        }
    ).write_parquet(sig_dir / "predictions.parquet")
    (sig_dir / "index.json").write_text(
        json.dumps({"dates": [d.isoformat() for d in oos], "tickers": tickers,
                    "embed_dim": 8, "feature_cols": FEATURES,
                    "encoder_state_sha256": "0" * 64})
    )
    (sig_dir / "gate.json").write_text(json.dumps({"verdict": "PASS", "mean_ic_20d": 0.03}))
    mod.require_passing_gate(sig_dir)

    # The shipped config asks for a 60-day lookback and a 756-day episode; the
    # synthetic panel is 420 days, so override just those two.
    env_cfg["inner"]["lookback_days"] = LOOKBACK
    env_cfg["inner"]["episode_length"] = 300
    envs = mod.build_envs(
        signal_dir=sig_dir, panel_path=p, n_envs=2, env_cfg=env_cfg, seed=0
    )
    assert len(envs) == 2
    assert envs[0].n_signal_dates == len(oos)
    assert envs[0].ranges.action_dim == 3 + N_SECTORS
    assert envs[0].cfg.periods_per_episode == env_cfg["periods_per_episode"]
    assert envs[0].cfg.turnover_penalty == env_cfg["turnover_penalty"]
    assert envs[0].ranges.k_max == env_cfg["ranges"]["k_max"]
    obs, _ = envs[0].reset(seed=0)
    assert obs.shape == (envs[0].obs_dim,)
