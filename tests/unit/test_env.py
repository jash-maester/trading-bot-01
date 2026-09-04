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
        BuyAndHoldIndex,
        EqualWeightRebalanced,
        MomentumTopK,
        RandomPolicy,
        SixtyFortyCash,
    )

    N = 8
    obs = _fake_obs(n=N, n_tradeable=5)
    for AgentCls in [
        EqualWeightRebalanced,
        BuyAndHoldIndex,
        MomentumTopK,
        SixtyFortyCash,
        RandomPolicy,
    ]:
        agent = AgentCls()  # type: ignore[call-arg]
        agent.reset()
        logits = agent.act(obs)
        assert logits.shape == (N + 1,), f"{AgentCls.__name__} shape mismatch"


def test_buy_and_hold_does_not_change_after_first_step() -> None:
    from trader.env.baselines import BuyAndHoldIndex

    agent = BuyAndHoldIndex()
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
