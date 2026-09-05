"""Unit tests for M4 — cost model, reward, and env utilities."""
from __future__ import annotations

import math

import numpy as np
import pytest

# ── Cost model ────────────────────────────────────────────────────────────────


def test_zero_trade_value_returns_zero() -> None:
    from trader.env.costs import ZerodhaEquityDeliveryCostModel

    model = ZerodhaEquityDeliveryCostModel()
    assert model.cost(0.0, is_buy=True) == 0.0
    assert model.cost(-1.0, is_buy=False) == 0.0


def test_buy_has_stamp_and_stt() -> None:
    from trader.env.costs import ZerodhaEquityDeliveryCostModel

    model = ZerodhaEquityDeliveryCostModel()
    buy_cost = model.cost(100_000.0, is_buy=True)
    sell_cost = model.cost(100_000.0, is_buy=False, n_scrips_sold=0)
    # Delivery STT is 0.1% on *both* legs; only the buy also pays 0.015% stamp
    # duty, so with no DP charge the buy leg is the dearer of the two.
    assert buy_cost - sell_cost == pytest.approx(0.00015 * 100_000.0)


def test_dp_charge_on_sell() -> None:
    from trader.env.costs import ZerodhaEquityDeliveryCostModel

    model = ZerodhaEquityDeliveryCostModel()
    no_dp = model.cost(100_000.0, is_buy=False, n_scrips_sold=0)
    with_dp = model.cost(100_000.0, is_buy=False, n_scrips_sold=2)
    # ₹15.34 per distinct scrip per sell day (₹3.5 CDSL + ₹9.5 broker + GST).
    assert with_dp - no_dp == pytest.approx(2 * 15.34, abs=1e-6)


def test_intraday_brokerage_cap_at_20() -> None:
    from trader.env.costs import ZerodhaEquityDeliveryCostModel

    # Delivery is brokerage-free; the ₹20 cap only bites on intraday orders.
    model = ZerodhaEquityDeliveryCostModel(intraday=True)
    delivery = ZerodhaEquityDeliveryCostModel()
    assert delivery.cost(2_000_000.0, is_buy=True) == pytest.approx(
        2.0 * delivery.cost(1_000_000.0, is_buy=True)
    )
    # Doubling turnover must not double the (capped) intraday brokerage, so the
    # intraday premium over delivery shrinks in relative terms.
    premium_1m = model.cost(1_000_000.0, is_buy=True) - delivery.cost(1_000_000.0, is_buy=True)
    premium_2m = model.cost(2_000_000.0, is_buy=True) - delivery.cost(2_000_000.0, is_buy=True)
    assert premium_2m - premium_1m == pytest.approx(
        # Only the rate-driven STT/stamp differences move; brokerage is pinned.
        (0.00003 - 0.00015) * 1_000_000.0 + (0.0 - 0.001) * 1_000_000.0
    )


def test_zero_cost_model() -> None:
    from trader.env.costs import ZeroCostModel

    model = ZeroCostModel()
    assert model.cost(1_000_000.0, is_buy=True) == 0.0
    assert model.cost(1_000_000.0, is_buy=False, n_scrips_sold=5) == 0.0


# ── Reward: DifferentialSharpe ────────────────────────────────────────────────


def test_dsr_reset_clears_state() -> None:
    from trader.env.reward import DifferentialSharpe

    dsr = DifferentialSharpe()
    for _ in range(10):
        dsr(0.01)
    dsr.reset()
    assert dsr._A == 0.0
    assert dsr._B == 0.0
    assert dsr._step == 0


def test_dsr_returns_zero_on_first_two_steps() -> None:
    from trader.env.reward import DifferentialSharpe

    dsr = DifferentialSharpe()
    r0 = dsr(0.05)
    r1 = dsr(0.05)
    assert r0 == 0.0
    assert r1 == 0.0


def test_dsr_no_nan_on_flat_returns() -> None:
    from trader.env.reward import DifferentialSharpe

    dsr = DifferentialSharpe()
    for _ in range(100):
        r = dsr(0.0)
        assert not math.isnan(r)
        assert not math.isinf(r)


def test_log_return_reward_passthrough() -> None:
    from trader.env.reward import LogReturn

    fn = LogReturn()
    assert fn(0.05) == pytest.approx(0.05)
    fn.reset()  # should not raise


def test_excess_log_return_passthrough() -> None:
    """ExcessLogReturn is functionally identical to LogReturn — the env
    is the one that subtracts the benchmark before calling it."""
    from trader.env.reward import ExcessLogReturn

    fn = ExcessLogReturn()
    assert fn(0.05) == pytest.approx(0.05)
    assert fn(-0.02) == pytest.approx(-0.02)
    fn.reset()  # should not raise


# ── masked_softmax ────────────────────────────────────────────────────────────


def test_softmax_weights_sum_to_one() -> None:
    from trader.env.panel_env import masked_softmax

    rng = np.random.default_rng(0)
    mask = rng.integers(0, 2, size=9).astype(bool)
    logits = rng.uniform(-3, 3, size=10)
    w = masked_softmax(logits, mask, max_weight=0.10)
    assert w.sum() == pytest.approx(1.0, abs=1e-6)


def test_softmax_masked_names_are_zero() -> None:
    from trader.env.panel_env import masked_softmax

    mask = np.array([True, False, True, False, True])
    logits = np.ones(6)   # N+1 = 6
    w = masked_softmax(logits, mask, max_weight=0.5)
    # Untradeable tickers (indices 2, 4 in w = mask indices 1, 3) must be 0
    assert w[2] == pytest.approx(0.0, abs=1e-9)
    assert w[4] == pytest.approx(0.0, abs=1e-9)


def test_softmax_max_weight_respected() -> None:
    from trader.env.panel_env import masked_softmax

    N = 20
    mask = np.ones(N, dtype=bool)
    logits = np.zeros(N + 1)
    w = masked_softmax(logits, mask, max_weight=0.10)
    assert np.all(w <= 0.10 + 1e-9)
    assert w.sum() == pytest.approx(1.0, abs=1e-6)


def test_softmax_all_untradeable_goes_to_cash() -> None:
    from trader.env.panel_env import masked_softmax

    mask = np.zeros(5, dtype=bool)
    logits = np.ones(6)
    w = masked_softmax(logits, mask, max_weight=0.10)
    assert w[0] == pytest.approx(1.0, abs=1e-6)


def test_softmax_no_nan_with_extreme_logits() -> None:
    from trader.env.panel_env import masked_softmax

    mask = np.ones(4, dtype=bool)
    logits = np.array([10.0, 10.0, -10.0, -10.0, 10.0])
    w = masked_softmax(logits, mask, max_weight=0.5)
    assert not np.any(np.isnan(w))
    assert w.sum() == pytest.approx(1.0, abs=1e-6)


# ── Baseline agents ───────────────────────────────────────────────────────────


def _fake_obs(n: int = 5, n_tradeable: int = 3, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    mask = np.zeros(n, dtype=np.int8)
    idx = rng.choice(n, size=n_tradeable, replace=False)
    mask[idx] = 1
    return {
        "mask": mask,
        "features": rng.standard_normal((60, n, 15)).astype(np.float32),
        "portfolio": np.zeros(n + 1, dtype=np.float32),
        "cash": np.float32(1_000_000.0),
        "nav": np.float32(1_000_000.0),
        "t_frac": np.float32(0.0),
        "sector_ids": np.zeros(n, dtype=np.int32),
    }


def test_equal_weight_sums_to_one_after_softmax() -> None:
    from trader.env.baselines import EqualWeightRebalanced
    from trader.env.panel_env import masked_softmax

    agent = EqualWeightRebalanced()
    obs = _fake_obs(n=10, n_tradeable=6)
    logits = agent.act(obs).astype(np.float64)
    mask = obs["mask"].astype(bool)
    w = masked_softmax(logits, mask, max_weight=0.10)
    assert w.sum() == pytest.approx(1.0, abs=1e-6)


def test_all_baselines_return_correct_shape() -> None:
    from trader.env.baselines import (
        EqualWeightFrozenUniverse,
        EqualWeightRebalanced,
        MomentumTopK,
        RandomPolicy,
        SixtyFortyCash,
    )

    N = 8
    obs = _fake_obs(n=N, n_tradeable=5)
    for AgentCls in [
        EqualWeightRebalanced,
        EqualWeightFrozenUniverse,
        MomentumTopK,
        SixtyFortyCash,
        RandomPolicy,
    ]:
        agent = AgentCls()  # type: ignore[call-arg]
        agent.reset()
        logits = agent.act(obs)
        assert logits.shape == (N + 1,), f"{AgentCls.__name__} shape mismatch"


def test_frozen_universe_logits_do_not_change_after_first_step() -> None:
    from trader.env.baselines import EqualWeightFrozenUniverse

    agent = EqualWeightFrozenUniverse()
    obs1 = _fake_obs(n=5, n_tradeable=3)
    obs2 = _fake_obs(n=5, n_tradeable=5, seed=99)  # different tradeable set
    agent.reset()
    l1 = agent.act(obs1).copy()
    l2 = agent.act(obs2)
    np.testing.assert_array_equal(l1, l2)


def test_sixty_forty_cash_weight_approx_040() -> None:
    from trader.env.baselines import SixtyFortyCash
    from trader.env.panel_env import masked_softmax

    agent = SixtyFortyCash()
    N = 10
    obs = _fake_obs(n=N, n_tradeable=N)   # all tradeable
    logits = agent.act(obs).astype(np.float64)
    mask = obs["mask"].astype(bool)
    w = masked_softmax(logits, mask, max_weight=0.10)
    # Cash weight should be ≈ 0.40
    assert w[0] == pytest.approx(0.40, abs=0.02)


# ── MomentumTopK: ranking column and K vs the per-name weight cap ─────────────


def _momentum_obs(n: int, feats: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "mask": np.ones(n, dtype=np.int8),
        "features": feats,
        "portfolio": np.zeros(n + 1, dtype=np.float32),
        "cash": np.float32(1_000_000.0),
        "nav": np.float32(1_000_000.0),
        "t_frac": np.float32(0.0),
        "sector_ids": np.zeros(n, dtype=np.int32),
    }


def _opposed_rank_features(n: int) -> np.ndarray:
    """Feature block whose 1-day and 20-day rankings are exact opposites.

    Ticker i gets log_return_1d = +i and log_return_20d = -i, so the 1-day
    winners are the last K tickers and the 20-day winners are the first K.
    """
    from trader.data.features import FEATURE_COLS

    feats = np.zeros((60, n, len(FEATURE_COLS)), dtype=np.float32)
    feats[-1, :, FEATURE_COLS.index("log_return_1d")] = np.arange(n, dtype=np.float32)
    feats[-1, :, FEATURE_COLS.index("log_return_20d")] = -np.arange(n, dtype=np.float32)
    return feats


def _picked(logits: np.ndarray) -> set[int]:
    """Ticker indices the top-K logits select (non-picked names get -1e9)."""
    return set(np.flatnonzero(logits[1:] > -1e8).tolist())


def test_momentum_ranks_on_the_20d_column_not_the_1d_one() -> None:
    """MomentumTopK must rank on `log_return_20d`.

    It used to read `features[-1, :, 0]`, and `FEATURE_COLS[0]` is
    `log_return_1d` — so "momentum top-K" was a one-day signal. With the two
    rankings made exact opposites, the old code picks {n-3, n-2, n-1}.
    """
    from trader.env.baselines import MomentumTopK

    n, k = 10, 3
    obs = _momentum_obs(n, _opposed_rank_features(n))

    agent = MomentumTopK(k=k)
    agent.reset()
    picked = _picked(agent.act(obs))

    assert picked == {0, 1, 2}, f"ranked on the wrong column: picked {sorted(picked)}"
    assert picked.isdisjoint({n - 3, n - 2, n - 1}), "these are the 1-day winners"


def test_momentum_column_index_is_resolved_by_name_not_position() -> None:
    """A caller-supplied feature list must be honoured positionally-correctly,
    so a reordering of FEATURE_COLS cannot silently move the ranking column."""
    from trader.data.features import FEATURE_COLS
    from trader.env.baselines import MomentumTopK

    n, k = 8, 2
    # Same data, feature axis reversed → log_return_20d lands on a new index.
    reversed_cols = list(reversed(FEATURE_COLS))
    feats = _opposed_rank_features(n)[:, :, ::-1].copy()
    obs = _momentum_obs(n, feats)

    agent = MomentumTopK(k=k, feature_columns=reversed_cols)
    agent.reset()
    assert _picked(agent.act(obs)) == {0, 1}

    with pytest.raises(ValueError, match="log_return_20d"):
        MomentumTopK(feature_columns=["log_return_1d", "rsi_14"])


def test_momentum_raises_on_feature_width_mismatch() -> None:
    """Wrong-width observations must fail loudly, not rank on whatever column
    happens to sit at the resolved index."""
    from trader.data.features import FEATURE_COLS
    from trader.env.baselines import MomentumTopK

    n = 6
    obs = _momentum_obs(n, np.zeros((60, n, len(FEATURE_COLS) + 3), dtype=np.float32))
    agent = MomentumTopK()
    agent.reset()
    with pytest.raises(ValueError, match="feature list"):
        agent.act(obs)


def test_momentum_default_k_leaves_under_5pct_cash_at_the_10pct_cap() -> None:
    """K must be >= 1/max_weight_per_name or the cap parks the rest in cash.

    Measured at N=40 tradeable, cap=0.10: K=20 → 1.81% cash, K=5 → 49.79%.
    """
    from trader.env.baselines import MomentumTopK
    from trader.env.panel_env import masked_softmax

    n, cap = 40, 0.10
    obs = _momentum_obs(n, _opposed_rank_features(n))
    mask = obs["mask"].astype(bool)

    agent = MomentumTopK()
    agent.reset()
    w = masked_softmax(agent.act(obs).astype(np.float64), mask, cap)
    assert w[0] < 0.05, f"default K leaves {w[0]:.2%} in cash"
    assert np.all(w[1:] <= cap + 1e-9)
    assert w.sum() == pytest.approx(1.0, abs=1e-6)

    # The old default: five names capped at 10% each cannot hold more than
    # half the book, so the "momentum" baseline was half money-market fund.
    agent_k5 = MomentumTopK(k=5)
    agent_k5.reset()
    w5 = masked_softmax(agent_k5.act(obs).astype(np.float64), mask, cap)
    assert w5[0] > 0.45


def test_buy_and_hold_index_alias_points_at_the_renamed_class() -> None:
    """`BuyAndHoldIndex` tracks no index and does not hold; the honest name is
    `EqualWeightFrozenUniverse`. The alias exists only because
    `scripts/evaluate.py` and `scripts/paper_run.py` still import the old name.
    """
    from trader.env import baselines

    assert baselines.BuyAndHoldIndex is baselines.EqualWeightFrozenUniverse
