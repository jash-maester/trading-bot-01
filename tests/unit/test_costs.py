"""Unit tests for the Zerodha cost model.

The rupee figures asserted here are taken from zerodha.com/charges (Sept 2026)
for a ₹1,00,000 buy plus a ₹1,00,000 sell in one scrip on NSE.  They are exact
to the paisa on purpose: a cost model that drifts by a few basis points changes
the sign of a high-turnover strategy's backtest.
"""
from __future__ import annotations

import numpy as np
import pytest

from trader.env.costs import (
    ZeroCostModel,
    ZerodhaEquityDeliveryCostModel,
    demat_amc_annual,
)

TRADE = 100_000.0


# ── Verification targets ──────────────────────────────────────────────────────


def test_delivery_buy_leg_exact() -> None:
    model = ZerodhaEquityDeliveryCostModel()
    assert round(model.cost(TRADE, is_buy=True), 2) == 118.74


def test_delivery_sell_leg_exact() -> None:
    model = ZerodhaEquityDeliveryCostModel()
    assert round(model.cost(TRADE, is_buy=False, n_scrips_sold=1), 2) == 119.08


def test_delivery_round_trip_exact() -> None:
    model = ZerodhaEquityDeliveryCostModel()
    total = model.cost(TRADE, is_buy=True) + model.cost(TRADE, is_buy=False, n_scrips_sold=1)
    assert round(total, 2) == 237.82


def test_delivery_round_trip_component_breakdown() -> None:
    """Each published component, reconstructed from first principles."""
    model = ZerodhaEquityDeliveryCostModel()
    buy = model.cost(TRADE, is_buy=True)
    sell = model.cost(TRADE, is_buy=False, n_scrips_sold=1)

    brokerage = 0.0
    stt = 0.001 * TRADE * 2  # both legs
    exchange = 0.0000307 * TRADE * 2
    sebi = 0.000001 * TRADE * 2
    gst = 0.18 * (brokerage + sebi + exchange)
    stamp = 0.00015 * TRADE  # buy only
    dp = 15.34  # one scrip, one sell day

    assert round(stt, 2) == 200.00
    assert round(exchange, 2) == 6.14
    assert round(sebi, 2) == 0.20
    assert round(gst, 2) == 1.14
    assert round(stamp, 2) == 15.00
    assert buy + sell == pytest.approx(brokerage + stt + exchange + sebi + gst + stamp + dp)


def test_intraday_round_trip_exact() -> None:
    model = ZerodhaEquityDeliveryCostModel(intraday=True)
    total = model.cost(TRADE, is_buy=True) + model.cost(TRADE, is_buy=False, n_scrips_sold=1)
    assert round(total, 2) == 82.68


def test_intraday_round_trip_component_breakdown() -> None:
    model = ZerodhaEquityDeliveryCostModel(intraday=True)
    total = model.cost(TRADE, is_buy=True) + model.cost(TRADE, is_buy=False, n_scrips_sold=1)

    brokerage = 2 * min(0.0003 * TRADE, 20.0)
    stt = 0.00025 * TRADE  # sell side only
    exchange = 0.0000307 * TRADE * 2
    sebi = 0.000001 * TRADE * 2
    gst = 0.18 * (brokerage + sebi + exchange)
    stamp = 0.00003 * TRADE

    assert round(brokerage, 2) == 40.00
    assert round(stt, 2) == 25.00
    assert round(gst, 2) == 8.34
    assert round(stamp, 2) == 3.00
    assert total == pytest.approx(brokerage + stt + exchange + sebi + gst + stamp)


# ── Individual levies ─────────────────────────────────────────────────────────


def test_zero_or_negative_trade_value_is_free() -> None:
    model = ZerodhaEquityDeliveryCostModel()
    assert model.cost(0.0, is_buy=True) == 0.0
    assert model.cost(-1.0, is_buy=False, n_scrips_sold=1) == 0.0


def test_delivery_brokerage_is_zero() -> None:
    """Delivery is brokerage-free, so cost must be exactly linear in turnover
    apart from the flat DP charge."""
    model = ZerodhaEquityDeliveryCostModel()
    small = model.cost(10_000.0, is_buy=True)
    large = model.cost(10_000_000.0, is_buy=True)
    assert large == pytest.approx(small * 1000.0)


def test_intraday_brokerage_caps_at_20() -> None:
    model = ZerodhaEquityDeliveryCostModel(intraday=True)

    def expected_buy(turnover: float, brokerage: float) -> float:
        exchange = 0.0000307 * turnover
        sebi = 0.000001 * turnover
        gst = 0.18 * (brokerage + sebi + exchange)
        return brokerage + exchange + sebi + gst + 0.00003 * turnover

    # Below ~₹66,667 of turnover the 0.03% rate binds...
    assert model.cost(50_000.0, is_buy=True) == pytest.approx(expected_buy(50_000.0, 15.0))
    # ...above it the flat ₹20 cap does, however large the order.
    assert model.cost(10_000_000.0, is_buy=True) == pytest.approx(
        expected_buy(10_000_000.0, 20.0)
    )


def test_stt_charged_on_both_delivery_legs() -> None:
    """The headline bug in the previous model: STT is 0.1% each way."""
    model = ZerodhaEquityDeliveryCostModel()
    # Buy cost minus its non-STT parts must still leave 0.1% of turnover.
    buy = model.cost(TRADE, is_buy=True)
    non_stt = 0.0000307 * TRADE + 0.000001 * TRADE + 0.00015 * TRADE
    non_stt += 0.18 * (0.000001 * TRADE + 0.0000307 * TRADE)
    assert buy - non_stt == pytest.approx(0.001 * TRADE)


def test_intraday_stt_is_sell_side_only() -> None:
    model = ZerodhaEquityDeliveryCostModel(intraday=True)
    buy = model.cost(TRADE, is_buy=True)
    sell = model.cost(TRADE, is_buy=False)
    # Sell has 0.025% STT but no 0.003% stamp duty; buy has stamp but no STT.
    assert sell - buy == pytest.approx(0.00025 * TRADE - 0.00003 * TRADE)


def test_stamp_duty_is_buy_side_only() -> None:
    model = ZerodhaEquityDeliveryCostModel()
    buy = model.cost(TRADE, is_buy=True)
    sell_no_dp = model.cost(TRADE, is_buy=False, n_scrips_sold=0)
    assert buy - sell_no_dp == pytest.approx(0.00015 * TRADE)


def test_gst_base_includes_sebi_fee() -> None:
    """GST is 18% of (brokerage + SEBI + exchange), not of (brokerage +
    exchange).  Omitting SEBI is small but systematic."""
    model = ZerodhaEquityDeliveryCostModel()
    cost = model.cost(TRADE, is_buy=True)
    without_sebi_in_gst = (
        0.0000307 * TRADE
        + 0.000001 * TRADE
        + 0.001 * TRADE
        + 0.00015 * TRADE
        + 0.18 * (0.0000307 * TRADE)
    )
    assert cost - without_sebi_in_gst == pytest.approx(0.18 * 0.000001 * TRADE)


def test_dp_charge_is_per_scrip_per_sell_day() -> None:
    model = ZerodhaEquityDeliveryCostModel()
    no_dp = model.cost(TRADE, is_buy=False, n_scrips_sold=0)
    two_scrips = model.cost(TRADE, is_buy=False, n_scrips_sold=2)
    assert two_scrips - no_dp == pytest.approx(2 * 15.34)


def test_no_dp_charge_on_buys_or_intraday() -> None:
    delivery = ZerodhaEquityDeliveryCostModel()
    intraday = ZerodhaEquityDeliveryCostModel(intraday=True)
    assert delivery.cost(TRADE, is_buy=True, n_scrips_sold=1) == delivery.cost(
        TRADE, is_buy=True, n_scrips_sold=0
    )
    assert intraday.cost(TRADE, is_buy=False, n_scrips_sold=3) == intraday.cost(
        TRADE, is_buy=False, n_scrips_sold=0
    )


def test_bse_transaction_charge_is_higher_than_nse() -> None:
    nse = ZerodhaEquityDeliveryCostModel(exchange="NSE")
    bse = ZerodhaEquityDeliveryCostModel(exchange="BSE")
    delta = bse.cost(TRADE, is_buy=True) - nse.cost(TRADE, is_buy=True)
    # Rate difference plus the GST that rides on it.
    assert delta == pytest.approx((0.0000375 - 0.0000307) * TRADE * 1.18)


def test_unknown_exchange_rejected() -> None:
    with pytest.raises(ValueError, match="unknown exchange"):
        ZerodhaEquityDeliveryCostModel(exchange="MCX")  # type: ignore[arg-type]


def test_per_call_intraday_override() -> None:
    """The paper broker decides delivery-vs-intraday at fill time."""
    model = ZerodhaEquityDeliveryCostModel()
    assert model.cost(TRADE, is_buy=True, intraday=True) == pytest.approx(
        ZerodhaEquityDeliveryCostModel(intraday=True).cost(TRADE, is_buy=True)
    )
    assert model.exchange == "NSE"
    assert model.intraday is False


# ── Vectorised path ───────────────────────────────────────────────────────────


def _reference_loop(
    model: ZerodhaEquityDeliveryCostModel,
    tv: np.ndarray,
    buy: np.ndarray,
    sold: np.ndarray,
) -> np.ndarray:
    return np.array(
        [
            model.cost(float(tv[i]), bool(buy[i]), n_scrips_sold=int(sold[i]))
            for i in range(tv.shape[0])
        ]
    )


def test_cost_vec_matches_scalar_cost() -> None:
    rng = np.random.default_rng(0)
    tv = rng.uniform(0.0, 5_000_000.0, size=200)
    tv[::7] = 0.0  # no-trade legs
    buy = rng.random(200) > 0.5
    sold = ((~buy) & (tv > 0)).astype(np.int64)

    for model in (
        ZerodhaEquityDeliveryCostModel(),
        ZerodhaEquityDeliveryCostModel(intraday=True),
        ZerodhaEquityDeliveryCostModel(exchange="BSE"),
    ):
        np.testing.assert_allclose(
            model.cost_vec(tv, buy, sold), _reference_loop(model, tv, buy, sold)
        )


def test_cost_vec_signature_unchanged_for_env() -> None:
    """panel_env calls cost_vec positionally; that must keep working."""
    model = ZerodhaEquityDeliveryCostModel()
    tv = np.array([TRADE, TRADE, 0.0])
    buy = np.array([True, False, False])
    sold = np.array([0, 1, 0], dtype=np.int64)
    out = model.cost_vec(tv, buy, sold)
    assert out.dtype == np.float64
    assert out.shape == (3,)
    assert round(float(out.sum()), 2) == 237.82
    assert out[2] == 0.0


def test_cost_vec_intraday_keyword() -> None:
    model = ZerodhaEquityDeliveryCostModel()
    tv = np.array([TRADE, TRADE])
    buy = np.array([True, False])
    sold = np.array([0, 1], dtype=np.int64)
    assert round(float(model.cost_vec(tv, buy, sold, intraday=True).sum()), 2) == 82.68


def test_cost_vec_stays_vectorised() -> None:
    """Guard the ~1000× env speedup: no Python loop over the ticker axis."""
    import dis

    src = ZerodhaEquityDeliveryCostModel.cost_vec
    opnames = {instr.opname for instr in dis.get_instructions(src)}
    assert "FOR_ITER" not in opnames, "cost_vec must not iterate per-name in Python"
    assert "CALL_FUNCTION_EX" not in opnames


# ── Demat AMC ─────────────────────────────────────────────────────────────────


def test_demat_amc_bsda_tiers() -> None:
    assert demat_amc_annual(0.0) == 0.0
    assert demat_amc_annual(399_999.0) == 0.0
    assert demat_amc_annual(400_000.0) == 100.0
    assert demat_amc_annual(999_999.0) == 100.0
    assert demat_amc_annual(1_000_000.0) == 100.0
    assert demat_amc_annual(1_000_001.0) == pytest.approx(354.0)  # 300 + 18% GST


def test_demat_amc_is_not_a_trade_cost() -> None:
    """AMC must not leak into per-order cost — it is an annual accrual."""
    model = ZerodhaEquityDeliveryCostModel()
    assert model.cost(TRADE, is_buy=True) < 200.0


# ── Zero-cost ablation model ──────────────────────────────────────────────────


def test_zero_cost_model() -> None:
    model = ZeroCostModel()
    assert model.cost(1_000_000.0, is_buy=True) == 0.0
    assert model.cost(1_000_000.0, is_buy=False, n_scrips_sold=5) == 0.0
    out = model.cost_vec(
        np.array([1e6, 2e6]), np.array([True, False]), np.array([0, 1], dtype=np.int64)
    )
    np.testing.assert_array_equal(out, np.zeros(2))
