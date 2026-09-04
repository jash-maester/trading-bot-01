"""Unit tests for the paper broker — all in memory, no Postgres required.

The interesting cases here are the ones where a plausible-looking
implementation is quietly wrong: a DP fee charged per order instead of per
scrip per day, a sale funding a purchase on the day of the sale, a fractional
share surviving a floor.  Each of those is a small rupee error per event that
compounds into a wrong backtest, so they are asserted exactly rather than
approximately.

Every test that trades also checks the accounting identity, because a broker
that gets one fill right and loses a rupee somewhere else is not usable.
"""
from __future__ import annotations

from datetime import date

import pytest

from trader.broker.base import (
    InvalidOrderState,
    OrderNotFound,
    OrderRequest,
)
from trader.broker.paper_broker import (
    REJECT_INSUFFICIENT_FUNDS,
    REJECT_INSUFFICIENT_HOLDING,
    REJECT_NO_PRICE,
    SLIPPAGE_K,
    AccountingError,
    ImpactInputs,
    IntradayRoundTripError,
    PaperBroker,
    PaperBrokerConfig,
    weights_from_logits,
)
from trader.env.costs import ZerodhaEquityDeliveryCostModel

D1 = date(2025, 1, 6)
D2 = date(2025, 1, 7)
D3 = date(2025, 1, 8)

COSTS = ZerodhaEquityDeliveryCostModel()


def make_broker(**overrides: object) -> PaperBroker:
    """A started, no-slippage broker so price arithmetic stays exact."""
    defaults: dict[str, object] = {
        "initial_cash": 100_000.0,
        "slippage_model": "none",
    }
    defaults.update(overrides)
    broker = PaperBroker(PaperBrokerConfig(**defaults))  # type: ignore[arg-type]
    broker.start_run(strategy_id="unit-test")
    return broker


def assert_books_balance(broker: PaperBroker) -> None:
    """Both statements of NAV must agree to the paisa."""
    nav_from_assets = broker.cash + broker.unsettled_cash + broker.equity_value
    nav_from_flows = (
        broker.config.initial_cash
        + broker.realised_pnl
        + broker.unrealised_pnl
        - broker.fees_paid
    )
    assert nav_from_assets == pytest.approx(broker.nav, abs=1e-9)
    assert nav_from_flows == pytest.approx(broker.nav, abs=1e-6)
    assert broker.cash >= 0.0


# ── sizing: integer shares, the per-name cap, the dust filter ─────────────────


def test_target_weight_floors_to_whole_shares() -> None:
    """₹10,000 of a ₹333 share is 30.03 shares, and 30 is what gets bought."""
    broker = make_broker()
    broker.submit_target_weights({"INFY.NS": 0.10})
    result = broker.open_session(D1, {"INFY.NS": 333.0})

    assert len(result.fills) == 1
    fill = result.fills[0]
    assert fill.quantity == 30
    assert isinstance(fill.quantity, int)
    assert fill.price == 333.0
    assert_books_balance(broker)


def test_weight_above_the_cap_is_clipped_not_redistributed() -> None:
    """A 50% target on a 10%-capped account buys 10% of NAV, and no more."""
    broker = make_broker(max_weight_per_name=0.10)
    broker.submit_target_weights({"INFY.NS": 0.50})
    result = broker.open_session(D1, {"INFY.NS": 100.0})

    # 10% of ₹1,00,000 at ₹100 = 100 shares, not 500.
    assert result.fills[0].quantity == 100
    assert_books_balance(broker)


def test_weights_summing_above_one_are_rejected() -> None:
    """The account has no leverage, so an over-allocated basket is a caller bug."""
    broker = make_broker(max_weight_per_name=1.0)
    with pytest.raises(ValueError, match="no leverage"):
        broker.submit_target_weights({"A.NS": 0.7, "B.NS": 0.5})


def test_negative_weight_is_rejected() -> None:
    broker = make_broker()
    with pytest.raises(ValueError, match="long-only"):
        broker.submit_target_weights({"A.NS": -0.1})


def test_delta_below_min_trade_value_produces_no_order() -> None:
    """₹400 of turnover is dust; the ledger should not record an order at all."""
    broker = make_broker(min_trade_value=500.0)
    # 0.4% of ₹1,00,000 = ₹400 → 4 shares at ₹100 → below the floor.
    broker.submit_target_weights({"A.NS": 0.004})
    result = broker.open_session(D1, {"A.NS": 100.0})

    assert result.fills == ()
    assert broker.get_orders() == ()
    assert broker.cash == 100_000.0


def test_delta_at_min_trade_value_does_trade() -> None:
    """The same setup one rupee over the floor trades — the filter is the rule."""
    broker = make_broker(min_trade_value=500.0)
    broker.submit_target_weights({"A.NS": 0.006})  # ₹600 → 6 shares
    result = broker.open_session(D1, {"A.NS": 100.0})

    assert [(f.symbol, f.quantity) for f in result.fills] == [("A.NS", 6)]
    assert_books_balance(broker)


# ── T+1 settlement ────────────────────────────────────────────────────────────


def test_sale_proceeds_cannot_fund_a_same_day_purchase() -> None:
    """The whole point of the settlement pipeline: today's sale is not cash.

    The account is fully invested in A; the strategy rotates into B on the same
    day.  The sell fills, the buy is rejected for want of settled funds — which
    is exactly what a real account does and what the environment, with its one
    cash float, cannot show.
    """
    broker = make_broker(max_weight_per_name=1.0, settlement_days=1)
    broker.submit_target_weights({"A.NS": 0.95})
    broker.open_session(D1, {"A.NS": 100.0, "B.NS": 50.0})
    broker.mark_to_market(D1, {"A.NS": 100.0, "B.NS": 50.0})
    assert broker.cash < 5_100.0  # essentially fully deployed

    broker.submit_target_weights({"B.NS": 0.95})
    result = broker.open_session(D2, {"A.NS": 100.0, "B.NS": 50.0})

    sides = {f.side for f in result.fills}
    assert sides == {"SELL"}
    assert [o.reject_reason for o in result.rejected] == [REJECT_INSUFFICIENT_FUNDS]
    assert broker.unsettled_cash > 0.0
    assert_books_balance(broker)


def test_proceeds_settle_on_the_next_session() -> None:
    broker = make_broker(max_weight_per_name=1.0, settlement_days=1)
    broker.submit_target_weights({"A.NS": 0.95})
    broker.open_session(D1, {"A.NS": 100.0})
    broker.mark_to_market(D1, {"A.NS": 100.0})

    broker.submit_target_weights({})
    broker.open_session(D2, {"A.NS": 100.0})
    broker.mark_to_market(D2, {"A.NS": 100.0})
    pending = broker.unsettled_cash
    assert pending > 0.0

    result = broker.open_session(D3, {"A.NS": 100.0})
    assert result.cash_settled == pytest.approx(pending)
    assert broker.unsettled_cash == 0.0
    assert_books_balance(broker)


def test_a_hundred_percent_target_leaves_no_room_for_charges() -> None:
    """Charges have to come from somewhere, and this account has no overdraft.

    A basket summing to exactly 1.0 asks to spend every rupee on stock, leaving
    nothing for the ~12 bps of statutory charges — so the buy is rejected whole
    rather than half-filled.  Callers should leave a sliver in cash; the
    environment's masked softmax normally does.
    """
    broker = make_broker(max_weight_per_name=1.0)
    broker.submit_target_weights({"A.NS": 1.0})
    result = broker.open_session(D1, {"A.NS": 100.0})
    assert result.fills == ()
    assert [o.reject_reason for o in result.rejected] == [REJECT_INSUFFICIENT_FUNDS]


def test_settlement_days_zero_credits_immediately() -> None:
    """T+0 is not how India settles, but the knob has to actually work."""
    broker = make_broker(max_weight_per_name=1.0, settlement_days=0)
    broker.submit_target_weights({"A.NS": 0.95})
    broker.open_session(D1, {"A.NS": 100.0})
    broker.mark_to_market(D1, {"A.NS": 100.0})

    broker.submit_target_weights({})
    broker.open_session(D2, {"A.NS": 100.0})
    assert broker.unsettled_cash == 0.0
    assert broker.cash > 99_000.0


# ── depository charges ────────────────────────────────────────────────────────


def test_dp_is_charged_once_per_symbol_per_day_not_per_order() -> None:
    """Exiting one scrip in three clips costs the same as exiting it in one.

    Every other component of the Indian charge stack is linear in turnover, so
    if DP is charged once per scrip per day the two paths must agree exactly.
    A per-order DP would overcharge this by 2 × ₹15.34.
    """
    broker = make_broker(max_weight_per_name=1.0)
    broker.submit_target_weights({"A.NS": 0.90})
    broker.open_session(D1, {"A.NS": 100.0})
    broker.mark_to_market(D1, {"A.NS": 100.0})
    held = broker.get_positions()[0].quantity
    fees_after_buy = broker.fees_paid

    for qty in (held // 3, held // 3, held - 2 * (held // 3)):
        broker.place_order(OrderRequest(symbol="A.NS", side="SELL", quantity=qty))
    result = broker.open_session(D2, {"A.NS": 100.0})

    assert len(result.fills) == 3
    sell_fees = broker.fees_paid - fees_after_buy
    one_shot = COSTS.cost(held * 100.0, False, n_scrips_sold=1)
    assert sell_fees == pytest.approx(one_shot, abs=1e-9)
    # And the DP fee is in there exactly once.
    without_dp = COSTS.cost(held * 100.0, False, n_scrips_sold=0)
    assert sell_fees - without_dp == pytest.approx(15.34, abs=1e-9)
    assert_books_balance(broker)


def test_dp_is_charged_again_on_a_later_sell_day() -> None:
    """Once per scrip per *day* — a second day of selling pays again."""
    broker = make_broker(max_weight_per_name=1.0)
    broker.submit_target_weights({"A.NS": 0.90})
    broker.open_session(D1, {"A.NS": 100.0})
    broker.mark_to_market(D1, {"A.NS": 100.0})
    base = broker.fees_paid

    broker.place_order(OrderRequest(symbol="A.NS", side="SELL", quantity=100))
    broker.open_session(D2, {"A.NS": 100.0})
    broker.mark_to_market(D2, {"A.NS": 100.0})
    broker.place_order(OrderRequest(symbol="A.NS", side="SELL", quantity=100))
    broker.open_session(D3, {"A.NS": 100.0})

    two_days = broker.fees_paid - base
    assert two_days == pytest.approx(2 * COSTS.cost(10_000.0, False, n_scrips_sold=1))


def test_two_symbols_sold_on_the_same_day_each_pay_dp() -> None:
    """Per *scrip*, not per portfolio: two names leaving the demat is two fees."""
    broker = make_broker()
    broker.submit_target_weights({"A.NS": 0.10, "B.NS": 0.10})
    broker.open_session(D1, {"A.NS": 100.0, "B.NS": 100.0})
    broker.mark_to_market(D1, {"A.NS": 100.0, "B.NS": 100.0})
    base = broker.fees_paid

    broker.submit_target_weights({})
    broker.open_session(D2, {"A.NS": 100.0, "B.NS": 100.0})
    assert broker.fees_paid - base == pytest.approx(
        2 * COSTS.cost(10_000.0, False, n_scrips_sold=1)
    )


# ── the intraday guard ────────────────────────────────────────────────────────


def test_same_day_round_trip_raises_rather_than_mispricing() -> None:
    """A daily simulator cannot produce this, so it must not price it silently."""
    broker = make_broker(max_weight_per_name=1.0)
    broker.submit_target_weights({"A.NS": 0.50})
    broker.open_session(D1, {"A.NS": 100.0})
    broker.mark_to_market(D1, {"A.NS": 100.0})

    broker.place_order(OrderRequest(symbol="A.NS", side="SELL", quantity=100))
    broker.place_order(OrderRequest(symbol="A.NS", side="BUY", quantity=100))
    with pytest.raises(IntradayRoundTripError, match="intraday rates"):
        broker.open_session(D2, {"A.NS": 100.0})


def test_target_weights_can_never_trip_the_intraday_guard() -> None:
    """One delta per symbol per day means the guard is unreachable by design."""
    broker = make_broker()
    broker.submit_target_weights({"A.NS": 0.10})
    broker.open_session(D1, {"A.NS": 100.0})
    broker.mark_to_market(D1, {"A.NS": 100.0})
    broker.submit_target_weights({"A.NS": 0.05})
    result = broker.open_session(D2, {"A.NS": 100.0})
    assert all(f.side == "SELL" for f in result.fills)


# ── FIFO lots and tax ─────────────────────────────────────────────────────────


def test_fifo_consumes_the_oldest_lot_first_across_a_partial() -> None:
    """Two buys, one sell that eats the first lot and part of the second."""
    broker = make_broker(max_weight_per_name=1.0)
    broker.place_order(OrderRequest(symbol="A.NS", side="BUY", quantity=100))
    broker.open_session(D1, {"A.NS": 100.0})
    broker.mark_to_market(D1, {"A.NS": 100.0})
    broker.place_order(OrderRequest(symbol="A.NS", side="BUY", quantity=100))
    broker.open_session(D2, {"A.NS": 200.0})
    broker.mark_to_market(D2, {"A.NS": 200.0})

    broker.place_order(OrderRequest(symbol="A.NS", side="SELL", quantity=150))
    broker.open_session(D3, {"A.NS": 300.0})

    # 100 @ 100 and 50 @ 200 consumed at 300 → 20,000 + 5,000 realised.
    assert broker.realised_pnl == pytest.approx(25_000.0)
    remaining = broker.open_lots("A.NS")
    assert len(remaining) == 1
    assert remaining[0].quantity == pytest.approx(50.0)
    assert remaining[0].cost_basis_per_share == pytest.approx(200.0)
    assert remaining[0].buy_date == D2
    assert_books_balance(broker)


def test_exactly_twelve_months_is_taxed_as_short_term() -> None:
    """The statute says "more than" twelve months; the anniversary is short."""
    broker = make_broker(max_weight_per_name=1.0)
    broker.place_order(OrderRequest(symbol="A.NS", side="BUY", quantity=100))
    broker.open_session(date(2024, 1, 15), {"A.NS": 100.0})
    broker.mark_to_market(date(2024, 1, 15), {"A.NS": 100.0})

    broker.place_order(OrderRequest(symbol="A.NS", side="SELL", quantity=100))
    broker.open_session(date(2025, 1, 15), {"A.NS": 200.0})

    short_term, long_term = broker.tax.realised(2024)
    assert short_term == pytest.approx(10_000.0)
    assert long_term == 0.0


def test_one_day_past_twelve_months_is_long_term() -> None:
    broker = make_broker(max_weight_per_name=1.0)
    broker.place_order(OrderRequest(symbol="A.NS", side="BUY", quantity=100))
    broker.open_session(date(2024, 1, 15), {"A.NS": 100.0})
    broker.mark_to_market(date(2024, 1, 15), {"A.NS": 100.0})

    broker.place_order(OrderRequest(symbol="A.NS", side="SELL", quantity=100))
    broker.open_session(date(2025, 1, 16), {"A.NS": 200.0})

    short_term, long_term = broker.tax.realised(2024)
    assert short_term == 0.0
    assert long_term == pytest.approx(10_000.0)
    # Below the ₹1.25L §112A exemption, so nothing is actually owed.
    assert broker.tax.liability(2024).total == 0.0


def test_tax_is_accrued_not_deducted_from_cash() -> None:
    """A liability on the financial year is not a per-trade charge."""
    broker = make_broker(max_weight_per_name=1.0)
    broker.place_order(OrderRequest(symbol="A.NS", side="BUY", quantity=100))
    broker.open_session(D1, {"A.NS": 100.0})
    broker.mark_to_market(D1, {"A.NS": 100.0})
    broker.place_order(OrderRequest(symbol="A.NS", side="SELL", quantity=100))
    broker.open_session(D2, {"A.NS": 200.0})
    snapshot = broker.mark_to_market(D2, {"A.NS": 200.0})

    liabilities = broker.tax_liabilities()
    assert len(liabilities) == 1
    assert liabilities[0].total > 0.0
    # NAV is unaffected by the accrual: cash + equity still explains it.
    assert snapshot.nav == pytest.approx(
        broker.config.initial_cash
        + broker.realised_pnl
        + broker.unrealised_pnl
        - broker.fees_paid
    )
    assert_books_balance(broker)


# ── cash safety ───────────────────────────────────────────────────────────────


def test_cash_never_goes_negative_under_repeated_rebalancing() -> None:
    broker = make_broker(max_weight_per_name=0.10, slippage_model="percentage")
    prices = {f"S{i}.NS": 100.0 + 7.0 * i for i in range(10)}
    for step, day in enumerate(
        [date(2025, 3, d) for d in (3, 4, 5, 6, 7, 10, 11, 12, 13, 14)]
    ):
        weight = 0.09 if step % 2 == 0 else 0.045
        broker.submit_target_weights({s: weight for s in prices})
        broker.open_session(day, prices)
        assert broker.cash >= 0.0
        assert_books_balance(broker)
        broker.mark_to_market(day, prices)
        assert_books_balance(broker)


def test_buy_beyond_settled_cash_is_rejected_whole() -> None:
    """No partial fills: the simulator must not re-size the strategy's order."""
    broker = make_broker(initial_cash=10_000.0)
    broker.place_order(OrderRequest(symbol="A.NS", side="BUY", quantity=1_000))
    result = broker.open_session(D1, {"A.NS": 100.0})

    assert result.fills == ()
    assert [o.reject_reason for o in result.rejected] == [REJECT_INSUFFICIENT_FUNDS]
    assert broker.cash == 10_000.0
    assert broker.get_positions() == ()


def test_selling_more_than_held_is_rejected() -> None:
    broker = make_broker()
    broker.place_order(OrderRequest(symbol="A.NS", side="SELL", quantity=10))
    result = broker.open_session(D1, {"A.NS": 100.0})
    assert [o.reject_reason for o in result.rejected] == [REJECT_INSUFFICIENT_HOLDING]


def test_missing_price_rejects_rather_than_filling_at_zero() -> None:
    broker = make_broker()
    broker.place_order(OrderRequest(symbol="A.NS", side="BUY", quantity=10))
    result = broker.open_session(D1, {"B.NS": 100.0})
    assert [o.reject_reason for o in result.rejected] == [REJECT_NO_PRICE]


# ── the accounting identity ───────────────────────────────────────────────────


def test_identity_holds_after_every_fill_of_a_multi_day_run() -> None:
    broker = make_broker(max_weight_per_name=0.20, slippage_model="percentage")
    prices = {"A.NS": 100.0, "B.NS": 250.0, "C.NS": 1_000.0}
    days = [date(2025, 2, d) for d in (3, 4, 5, 6, 7)]
    for i, day in enumerate(days):
        prices = {k: v * (1.0 + 0.01 * (-1) ** i) for k, v in prices.items()}
        broker.submit_target_weights({"A.NS": 0.2, "B.NS": 0.15, "C.NS": 0.1 * (i % 2)})
        broker.open_session(day, prices)
        assert_books_balance(broker)
        snapshot = broker.mark_to_market(day, prices)
        assert snapshot.nav == pytest.approx(
            snapshot.cash + snapshot.equity_value, abs=1e-9
        )
        assert_books_balance(broker)

    # Daily P&L must decompose the NAV path exactly.
    snapshots = broker.snapshots
    for prev, cur in zip(snapshots[:-1], snapshots[1:], strict=True):
        delta = (
            (cur.realised_pnl - prev.realised_pnl)
            + (cur.unrealised_pnl - prev.unrealised_pnl)
            - (cur.fees_paid - prev.fees_paid)
        )
        assert cur.nav - prev.nav == pytest.approx(delta, abs=1e-6)


def test_identity_check_catches_a_corrupted_book() -> None:
    """The guard is worth having only if it actually fires."""
    broker = make_broker()
    broker.submit_target_weights({"A.NS": 0.10})
    broker.open_session(D1, {"A.NS": 100.0})
    broker._cash += 1_000.0  # a rupee leak, injected  # noqa: SLF001
    with pytest.raises(AccountingError, match="NAV identity broken"):
        broker.mark_to_market(D1, {"A.NS": 100.0})


# ── order lifecycle ───────────────────────────────────────────────────────────


def test_orders_do_not_fill_in_the_session_they_were_submitted_in() -> None:
    broker = make_broker()
    order_id = broker.place_order(OrderRequest(symbol="A.NS", side="BUY", quantity=10))
    assert broker.get_orders()[0].status == "PENDING"
    assert broker.get_positions() == ()

    broker.open_session(D1, {"A.NS": 100.0})
    filled = {o.order_id: o for o in broker.get_orders()}[order_id]
    assert filled.status == "FILLED"
    assert filled.filled_quantity == 10


def test_cancel_removes_a_pending_order_from_the_queue() -> None:
    broker = make_broker()
    order_id = broker.place_order(OrderRequest(symbol="A.NS", side="BUY", quantity=10))
    broker.cancel_order(order_id)
    result = broker.open_session(D1, {"A.NS": 100.0})
    assert result.fills == ()
    assert broker.get_orders()[0].status == "CANCELLED"


def test_cancelling_a_filled_order_raises() -> None:
    broker = make_broker()
    order_id = broker.place_order(OrderRequest(symbol="A.NS", side="BUY", quantity=10))
    broker.open_session(D1, {"A.NS": 100.0})
    with pytest.raises(InvalidOrderState):
        broker.cancel_order(order_id)


def test_cancelling_an_unknown_order_raises() -> None:
    broker = make_broker()
    with pytest.raises(OrderNotFound):
        broker.cancel_order("nope")


def test_fractional_quantity_is_a_type_error() -> None:
    with pytest.raises(TypeError, match="no fractional shares"):
        OrderRequest(symbol="A.NS", side="BUY", quantity=10.5)  # type: ignore[arg-type]


def test_limit_order_fills_at_the_limit_when_marketable() -> None:
    broker = make_broker(slippage_model="percentage", slippage_pct=0.05)
    broker.place_order(
        OrderRequest(
            symbol="A.NS", side="BUY", quantity=10, order_type="LIMIT", limit_price=101.0
        )
    )
    result = broker.open_session(D1, {"A.NS": 100.0})
    # Slippage would have taken it to 105; the limit caps it at 101.
    assert result.fills[0].price == pytest.approx(101.0)


def test_limit_order_that_is_not_marketable_at_the_open_is_cancelled() -> None:
    """A daily bar cannot prove it would have filled later, so it must not."""
    broker = make_broker()
    broker.place_order(
        OrderRequest(
            symbol="A.NS", side="BUY", quantity=10, order_type="LIMIT", limit_price=90.0
        )
    )
    result = broker.open_session(D1, {"A.NS": 100.0})
    assert result.fills == ()
    assert broker.get_orders()[0].status == "CANCELLED"


def test_mis_product_is_refused() -> None:
    broker = make_broker()
    with pytest.raises(Exception, match="delivery only"):
        broker.place_order(
            OrderRequest(symbol="A.NS", side="BUY", quantity=10, product="MIS")
        )


def test_sessions_must_advance() -> None:
    broker = make_broker()
    broker.open_session(D2, {"A.NS": 100.0})
    with pytest.raises(Exception, match="must advance"):
        broker.open_session(D1, {"A.NS": 100.0})


def test_trading_before_start_run_is_refused() -> None:
    broker = PaperBroker(PaperBrokerConfig())
    with pytest.raises(Exception, match="start_run"):
        broker.place_order(OrderRequest(symbol="A.NS", side="BUY", quantity=1))


# ── views ─────────────────────────────────────────────────────────────────────


def test_holdings_report_the_unsettled_quantity_bought_today() -> None:
    broker = make_broker()
    broker.submit_target_weights({"A.NS": 0.10})
    broker.open_session(D1, {"A.NS": 100.0})
    broker.mark_to_market(D1, {"A.NS": 110.0})

    holding = broker.get_holdings()[0]
    assert holding.quantity == 100
    assert holding.t1_quantity == 100
    assert holding.pnl == pytest.approx(1_000.0)

    broker.submit_target_weights({"A.NS": 0.10})
    broker.open_session(D2, {"A.NS": 110.0})
    # Nothing bought today, so nothing is in the T+1 bucket any more.
    assert broker.get_holdings()[0].t1_quantity == 0


def test_margins_split_settled_from_unsettled() -> None:
    broker = make_broker(max_weight_per_name=1.0)
    broker.submit_target_weights({"A.NS": 0.95})
    broker.open_session(D1, {"A.NS": 100.0})
    broker.mark_to_market(D1, {"A.NS": 100.0})
    broker.submit_target_weights({})
    broker.open_session(D2, {"A.NS": 100.0})

    margins = broker.get_margins()
    assert margins.unsettled_cash > 0.0
    assert margins.opening_balance == 100_000.0
    assert margins.total_cash == pytest.approx(
        margins.available_cash + margins.unsettled_cash
    )


def test_snapshot_series_feeds_quantstats() -> None:
    broker = make_broker()
    for day, price in ((D1, 100.0), (D2, 110.0), (D3, 121.0)):
        broker.submit_target_weights({"A.NS": 0.10})
        broker.open_session(day, {"A.NS": price})
        broker.mark_to_market(day, {"A.NS": price})
    returns = broker.daily_log_returns()
    assert len(returns) == len(broker.snapshots) - 1
    assert all(isinstance(r, float) for r in returns)


# ── consistency with the backtest environment ─────────────────────────────────


def test_slippage_coefficient_matches_the_environment() -> None:
    """Duplicated constant, so it needs a test that notices the drift."""
    from trader.env.panel_env import _SLIPPAGE_K

    assert SLIPPAGE_K == _SLIPPAGE_K


def test_atr_impact_slippage_matches_the_environment_formula() -> None:
    broker = make_broker(slippage_model="atr_impact")
    broker.place_order(OrderRequest(symbol="A.NS", side="BUY", quantity=100))
    impact = {"A.NS": ImpactInputs(atr_fraction=0.02, adv_shares=10_000.0)}
    result = broker.open_session(D1, {"A.NS": 100.0}, impact=impact)

    expected = SLIPPAGE_K * 0.02 * (100 / 10_000.0) ** 0.5
    assert result.fills[0].slippage == pytest.approx(expected)
    assert result.fills[0].price == pytest.approx(100.0 * (1.0 + expected))


def test_weights_from_logits_uses_the_environment_softmax() -> None:
    import numpy as np

    from trader.env.panel_env import masked_softmax

    logits = np.array([0.0, 1.0, 2.0, -1.0], dtype=np.float64)
    mask = np.array([True, True, False])
    expected = masked_softmax(logits, mask, 0.10)
    weights = weights_from_logits(logits, ["A.NS", "B.NS", "C.NS"], mask, 0.10)

    assert weights == {"A.NS": pytest.approx(expected[1]), "B.NS": pytest.approx(expected[2])}
    assert "C.NS" not in weights


def test_target_shares_match_the_environment_formula() -> None:
    """floor(nav * weight / open) — the same expression PanelTradingEnv uses."""
    import numpy as np

    broker = make_broker(max_weight_per_name=0.25)
    nav = 100_000.0
    weights = {"A.NS": 0.25, "B.NS": 0.13, "C.NS": 0.07}
    opens = {"A.NS": 137.0, "B.NS": 2_411.5, "C.NS": 19.35}
    broker.submit_target_weights(weights)
    result = broker.open_session(D1, opens)

    filled = {f.symbol: f.quantity for f in result.fills}
    for symbol, weight in weights.items():
        expected = int(np.floor(nav * weight / opens[symbol]))
        assert filled[symbol] == expected
