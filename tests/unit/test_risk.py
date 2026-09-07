"""The three risk overlays: volatility target, drawdown brake, per-name stop.

Written before the sweep that decides whether any of them belongs in the
strategy, so the measurement is not also debugging the instrument.

The property that matters most is the last one: with every control disabled the
overlay must be the identity, or the sweep's control arm is not a control.
"""
from __future__ import annotations

import numpy as np
import pytest

from trader.allocator.risk import RiskOverlay, RiskParams


def _flat(n: int) -> np.ndarray:
    w = np.zeros(n + 1)
    w[1:] = 1.0 / n
    return w


# ── configuration ────────────────────────────────────────────────────────────


def test_disabled_by_default() -> None:
    p = RiskParams()
    assert not p.any_enabled
    assert not p.needs_intraperiod_trading


def test_only_the_stop_needs_intraperiod_trading() -> None:
    """A vol target or brake can wait for the next rebalance; a stop cannot."""
    assert not RiskParams(vol_target=0.15).needs_intraperiod_trading
    assert not RiskParams(drawdown_threshold=0.15).needs_intraperiod_trading
    assert RiskParams(stop_loss=0.15).needs_intraperiod_trading


@pytest.mark.parametrize(
    "kw",
    [
        {"vol_target": 0.0},
        {"vol_target": -0.1},
        {"drawdown_threshold": 1.5},
        {"stop_loss": 0.0},
        {"vol_lookback": 1},
        {"vol_floor": 1.5},
        {"stop_cooldown_steps": -1},
        {"drawdown_threshold": 0.3, "drawdown_full": 0.2},
    ],
)
def test_invalid_params_are_refused(kw: dict) -> None:  # type: ignore[type-arg]
    with pytest.raises(ValueError):
        RiskParams(**kw)


# ── the identity property ────────────────────────────────────────────────────


def test_all_controls_off_is_the_identity() -> None:
    """The control arm of the sweep must be bit-identical to no overlay at all."""
    o = RiskOverlay(RiskParams(), 6)
    rng = np.random.default_rng(0)
    px = np.full(6, 100.0)
    for i in range(40):
        px = px * (1.0 + rng.normal(0, 0.02, 6))
        o.update(1e6 * (0.9 ** (i / 40)), px, _flat(6)[1:])
        target = _flat(6)
        np.testing.assert_array_equal(o.apply(target), target)
    assert o.exposure_scale() == 1.0
    assert o.stops_to_execute().sum() == 0


# ── volatility target ────────────────────────────────────────────────────────


def test_vol_target_cuts_exposure_when_realised_vol_is_high() -> None:
    o = RiskOverlay(RiskParams(vol_target=0.10, vol_lookback=30, vol_floor=0.0), 4)
    rng = np.random.default_rng(1)
    nav = 1e6
    for _ in range(40):                      # ~48% annualised: far above target
        nav *= float(np.exp(rng.normal(0, 0.03)))
        o.update(nav, np.full(4, 100.0), _flat(4)[1:])
    rv = o.realised_vol()
    assert rv > 0.10
    assert o.exposure_scale() == pytest.approx(0.10 / rv, rel=1e-9)


def test_vol_target_never_raises_exposure_above_one() -> None:
    """Below-target vol must not lever the book up. Long-only, no leverage."""
    o = RiskOverlay(RiskParams(vol_target=0.50, vol_lookback=30), 4)
    nav = 1e6
    for i in range(40):
        nav *= 1.0001 if i % 2 else 0.9999   # almost no volatility
        o.update(nav, np.full(4, 100.0), _flat(4)[1:])
    assert o.exposure_scale() == 1.0


def test_vol_floor_binds() -> None:
    o = RiskOverlay(RiskParams(vol_target=0.01, vol_lookback=30, vol_floor=0.4), 4)
    rng = np.random.default_rng(2)
    nav = 1e6
    for _ in range(40):
        nav *= float(np.exp(rng.normal(0, 0.05)))
        o.update(nav, np.full(4, 100.0), _flat(4)[1:])
    assert o.exposure_scale() == pytest.approx(0.4)


# ── drawdown brake ───────────────────────────────────────────────────────────


def test_brake_is_inactive_above_the_threshold() -> None:
    o = RiskOverlay(RiskParams(drawdown_threshold=0.20, drawdown_full=0.40), 4)
    o.update(1e6, np.full(4, 100.0), _flat(4)[1:])
    o.update(0.9e6, np.full(4, 100.0), _flat(4)[1:])     # 10% down, under 20%
    assert o.drawdown() == pytest.approx(0.10)
    assert o.exposure_scale() == 1.0


def test_brake_tapers_linearly_and_reaches_its_floor() -> None:
    p = RiskParams(drawdown_threshold=0.20, drawdown_full=0.40, drawdown_floor=0.25)
    o = RiskOverlay(p, 4)
    o.update(1e6, np.full(4, 100.0), _flat(4)[1:])
    o.update(0.70e6, np.full(4, 100.0), _flat(4)[1:])    # 30% down = halfway
    assert o.exposure_scale() == pytest.approx(1.0 - 0.5 * (1.0 - 0.25))
    o.update(0.55e6, np.full(4, 100.0), _flat(4)[1:])    # 45% down, past full
    assert o.exposure_scale() == pytest.approx(0.25)


def test_brake_moves_the_cut_exposure_into_cash_not_into_other_names() -> None:
    """The point is to hold less equity, not to concentrate the same exposure."""
    o = RiskOverlay(RiskParams(drawdown_threshold=0.10, drawdown_full=0.20,
                               drawdown_floor=0.5), 4)
    o.update(1e6, np.full(4, 100.0), _flat(4)[1:])
    o.update(0.8e6, np.full(4, 100.0), _flat(4)[1:])     # past full -> 0.5
    out = o.apply(_flat(4))
    assert out.sum() == pytest.approx(1.0)
    assert out[0] == pytest.approx(0.5)                  # half in cash
    np.testing.assert_allclose(out[1:], np.full(4, 0.125))


# ── per-name stop ────────────────────────────────────────────────────────────


def test_stop_fires_on_a_loss_from_entry_and_not_on_a_gain() -> None:
    o = RiskOverlay(RiskParams(stop_loss=0.15), 3)
    px = np.array([100.0, 100.0, 100.0])
    o.update(1e6, px, np.array([1 / 3, 1 / 3, 1 / 3]))   # entry at 100 each
    o.update(1e6, np.array([84.0, 100.0, 130.0]), np.array([1 / 3, 1 / 3, 1 / 3]))
    stops = o.stops_to_execute()
    assert stops.tolist() == [True, False, False]


def test_a_name_not_held_cannot_be_stopped() -> None:
    o = RiskOverlay(RiskParams(stop_loss=0.10), 3)
    o.update(1e6, np.full(3, 100.0), np.array([0.5, 0.5, 0.0]))
    o.update(1e6, np.array([50.0, 100.0, 10.0]), np.array([0.5, 0.5, 0.0]))
    assert o.stops_to_execute().tolist() == [True, False, False]


def test_re_entry_measures_from_the_new_price_not_the_old_one() -> None:
    """Forgetting the old entry is what stops a re-bought name stopping instantly."""
    o = RiskOverlay(RiskParams(stop_loss=0.20), 2)
    o.update(1e6, np.array([100.0, 100.0]), np.array([0.5, 0.5]))
    o.update(1e6, np.array([50.0, 100.0]), np.array([0.0, 1.0]))   # name 0 exited
    o.update(1e6, np.array([50.0, 100.0]), np.array([0.5, 0.5]))   # re-entered at 50
    o.update(1e6, np.array([48.0, 100.0]), np.array([0.5, 0.5]))   # -4% from 50
    assert o.stops_to_execute().tolist() == [False, False]


def test_a_stopped_name_is_quarantined_then_released() -> None:
    o = RiskOverlay(RiskParams(stop_loss=0.10, stop_cooldown_steps=3), 2)
    o.update(1e6, np.array([100.0, 100.0]), np.array([0.5, 0.5]))
    o.update(1e6, np.array([80.0, 100.0]), np.array([0.5, 0.5]))
    assert o.register_stops() == 1
    for _ in range(3):
        assert o.apply(_flat(2))[1] == 0.0                 # barred
        o.update(1e6, np.array([80.0, 100.0]), np.array([0.0, 0.5]))
    assert o.apply(_flat(2))[1] > 0.0                      # released


def test_register_stops_clears_them_so_one_breach_is_one_sale() -> None:
    o = RiskOverlay(RiskParams(stop_loss=0.10, stop_cooldown_steps=0), 2)
    o.update(1e6, np.array([100.0, 100.0]), np.array([0.5, 0.5]))
    o.update(1e6, np.array([80.0, 100.0]), np.array([0.5, 0.5]))
    assert o.register_stops() == 1
    assert o.stops_to_execute().sum() == 0


# ── composition ──────────────────────────────────────────────────────────────


def test_apply_always_returns_a_valid_weight_vector() -> None:
    o = RiskOverlay(
        RiskParams(vol_target=0.10, drawdown_threshold=0.10, drawdown_full=0.3,
                   stop_loss=0.10, stop_cooldown_steps=5),
        5,
    )
    rng = np.random.default_rng(7)
    px = np.full(5, 100.0)
    nav = 1e6
    for _ in range(60):
        px = np.maximum(px * (1.0 + rng.normal(0, 0.04, 5)), 1.0)
        nav *= float(np.exp(rng.normal(0, 0.02)))
        o.update(nav, px, _flat(5)[1:])
        o.register_stops()
        out = o.apply(_flat(5))
        assert out.shape == (6,)
        assert out.sum() == pytest.approx(1.0)
        assert np.all(out >= -1e-12)


def test_apply_rejects_a_wrongly_shaped_target() -> None:
    o = RiskOverlay(RiskParams(), 4)
    with pytest.raises(ValueError, match="shape"):
        o.apply(np.zeros(4))


def test_entry_is_the_fill_price_not_the_close() -> None:
    """A stop measures loss from what the position COST.

    Recording the entry-day close instead is wrong by one intraday move: it
    fires the stop early on a name that rose after the open and late on one
    that fell. Here the fill is 100 and the entry-day close is 120, so a stop
    at 15% must trigger at 85, not at 102.
    """
    o = RiskOverlay(RiskParams(stop_loss=0.15), 2)
    closes = np.array([120.0, 100.0])
    fills = np.array([100.0, 100.0])
    o.update(1e6, closes, np.array([0.5, 0.5]), fill_prices=fills)

    # 90 is 25% below the close but only 10% below the fill: must NOT stop.
    o.update(1e6, np.array([90.0, 100.0]), np.array([0.5, 0.5]))
    assert o.stops_to_execute().tolist() == [False, False]

    # 84 is 16% below the fill: must stop.
    o.update(1e6, np.array([84.0, 100.0]), np.array([0.5, 0.5]))
    assert o.stops_to_execute().tolist() == [True, False]


def test_a_zero_fill_falls_back_to_the_close() -> None:
    """Callers that cannot supply fills must still get a usable entry."""
    o = RiskOverlay(RiskParams(stop_loss=0.10), 2)
    o.update(1e6, np.array([100.0, 100.0]), np.array([0.5, 0.5]),
             fill_prices=np.array([0.0, 0.0]))
    o.update(1e6, np.array([85.0, 100.0]), np.array([0.5, 0.5]))
    assert o.stops_to_execute().tolist() == [True, False]


# ── dynamic, volatility-scaled threshold ─────────────────────────────────────


def test_vol_scaled_threshold_gives_a_volatile_name_more_room() -> None:
    """One fixed percentage asks a different question of a utility and a small-cap."""
    o = RiskOverlay(RiskParams(stop_vol_mult=1.0, stop_vol_horizon_days=20), 2)
    # 15% annualised vs 60% annualised.
    o.update(1e6, np.array([100.0, 100.0]), np.array([0.5, 0.5]),
             vol_ann=np.array([0.15, 0.60]))
    thr = o.stop_thresholds()
    assert thr[0] < thr[1], "the calmer name must be stopped sooner"
    # 0.15/sqrt(252)*sqrt(20) = 0.0423 -> clipped up to the 0.05 floor.
    assert thr[0] == pytest.approx(0.05)
    # 0.60/sqrt(252)*sqrt(20) = 0.1690, inside the band.
    assert thr[1] == pytest.approx(0.169, abs=1e-3)


def test_default_mult_is_calibrated_near_the_fixed_ten_percent_arm() -> None:
    """So the dynamic arm is comparable to the fixed one, not a different animal."""
    o = RiskOverlay(RiskParams(stop_vol_mult=1.0, stop_vol_horizon_days=20), 1)
    o.update(1e6, np.array([100.0]), np.array([1.0]), vol_ann=np.array([0.35]))
    assert o.stop_thresholds()[0] == pytest.approx(0.0986, abs=0.005)


def test_vol_scaled_stop_fires_on_the_right_name_only() -> None:
    o = RiskOverlay(RiskParams(stop_vol_mult=1.0, stop_vol_horizon_days=20), 2)
    o.update(1e6, np.array([100.0, 100.0]), np.array([0.5, 0.5]),
             vol_ann=np.array([0.15, 0.60]))
    # -8%: past the calm name's 5% threshold, inside the volatile name's 16.9%.
    o.update(1e6, np.array([92.0, 92.0]), np.array([0.5, 0.5]))
    assert o.stops_to_execute().tolist() == [True, False]


def test_a_name_with_no_usable_vol_gets_the_loosest_threshold() -> None:
    """An unknown is not evidence that a position is in trouble."""
    o = RiskOverlay(RiskParams(stop_vol_mult=1.0, stop_vol_max=0.30), 2)
    o.update(1e6, np.array([100.0, 100.0]), np.array([0.5, 0.5]),
             vol_ann=np.array([np.nan, 0.35]))
    assert o.stop_thresholds()[0] == pytest.approx(0.30)


def test_fixed_and_dynamic_stops_cannot_both_be_set() -> None:
    with pytest.raises(ValueError, match="not both"):
        RiskParams(stop_loss=0.10, stop_vol_mult=1.0)


def test_a_dynamic_stop_without_vol_raises_rather_than_pretending() -> None:
    o = RiskOverlay(RiskParams(stop_vol_mult=1.0), 2)
    with pytest.raises(ValueError, match="volatility-scaled stop with no volatility"):
        o.update(1e6, np.array([100.0, 100.0]), np.array([0.5, 0.5]))
