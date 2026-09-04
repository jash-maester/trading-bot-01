"""Golden-file test: a fixed scenario must produce byte-identical books forever.

The unit tests pin individual rules — DP once per scrip per day, FIFO across a
partial lot, T+1 blocking a same-day repurchase.  What they cannot catch is a
change that is locally reasonable and globally wrong: a different tie-break in
the execution order, a rounding that moves a fill by a paisa, a settlement
release that happens one session early.  Each of those leaves every unit test
green and silently moves the NAV curve.

So this runs a whole strategy — a deterministic price panel, a baseline agent
from :mod:`trader.env.baselines`, 300 sessions spanning fourteen months — and
asserts the entire result against numbers committed below.  Fourteen months is
deliberate: it is the shortest span that realises both short-term and long-term
capital gains, in two different financial years, from lots that FIFO has to
match across a partial consumption.

The golden lives in this file rather than a sidecar because the scenario and
its expected output are useless apart; regenerate it with::

    PAPER_GOLDEN_UPDATE=1 uv run pytest tests/integration/test_paper_broker_golden.py

which prints a replacement block and then fails, so a regeneration can never be
mistaken for a passing run.  Regenerating is an admission that behaviour
changed: the diff has to be explainable before it is committed.

Requires Postgres — ``conftest.py`` skips the module when the DB is unreachable.
"""
from __future__ import annotations

import hashlib
import os
from datetime import date, timedelta
from typing import Any

import numpy as np
import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from trader.broker import schema
from trader.broker.paper_broker import PaperBroker, PaperBrokerConfig, weights_from_logits
from trader.env.baselines import EqualWeightRebalanced

pytestmark = pytest.mark.usefixtures("db_session")

SEED = 20_250_904
START = date(2024, 1, 1)
N_SESSIONS = 300
SYMBOLS: tuple[str, ...] = (
    "ADANIENT.NS", "ASIANPAINT.NS", "AXISBANK.NS", "BHARTIARTL.NS",
    "CIPLA.NS", "HDFCBANK.NS", "INFY.NS", "ITC.NS",
    "MARUTI.NS", "RELIANCE.NS", "SBIN.NS", "TCS.NS",
)

# The agent is held constant and the *cap* is the phase control, so the policy
# driving this run really is the committed baseline and nothing else:
#   session 0   — build the book at 8% a name
#   session 60  — halve it, realising short-term gains on part of every lot
#   session 290 — rebuild, creating a second generation of lots
#   session 299 — liquidate, realising long-term gains on what survived
REBALANCES: dict[int, float] = {0: 0.08, 60: 0.04, 290: 0.08, 299: 0.0}


def _sessions(start: date, count: int) -> list[date]:
    """``count`` weekdays from ``start``.  Holidays are irrelevant here — the
    broker settles per *session*, not per calendar day, so a fake calendar with
    the right shape is enough and keeps the fixture free of data files."""
    days: list[date] = []
    day = start
    while len(days) < count:
        if day.weekday() < 5:
            days.append(day)
        day += timedelta(days=1)
    return days


def _price_panel(
    days: list[date], seed: int
) -> tuple[dict[date, dict[str, float]], dict[date, dict[str, float]]]:
    """A seeded geometric walk, rounded to paise so the golden is exact."""
    rng = np.random.default_rng(seed)
    level = {s: 100.0 + 37.5 * i for i, s in enumerate(SYMBOLS)}
    opens: dict[date, dict[str, float]] = {}
    closes: dict[date, dict[str, float]] = {}
    for day in days:
        o: dict[str, float] = {}
        c: dict[str, float] = {}
        for symbol in SYMBOLS:
            base = level[symbol]
            o[symbol] = round(base * (1.0 + float(rng.normal(0.0, 0.004))), 2)
            close = round(base * (1.0 + float(rng.normal(0.0004, 0.012))), 2)
            c[symbol] = close
            level[symbol] = close
        opens[day] = o
        closes[day] = c
    return opens, closes


def run_scenario(session: Session | None) -> PaperBroker:
    """Drive the whole run.  ``session=None`` runs it entirely in memory."""
    days = _sessions(START, N_SESSIONS)
    opens, closes = _price_panel(days, SEED)

    broker = PaperBroker(
        PaperBrokerConfig(
            initial_cash=100_000.0,
            settlement_days=1,
            max_weight_per_name=0.10,
            min_trade_value=500.0,
            slippage_model="percentage",
            slippage_pct=0.001,
        ),
        session=session,
        strategy_id="golden-equal-weight",
    )
    broker.start_run(notes="golden-file integration scenario")

    agent = EqualWeightRebalanced()
    agent.reset()
    mask = np.ones(len(SYMBOLS), dtype=np.int8)

    for i, day in enumerate(days):
        cap = REBALANCES.get(i)
        if cap is not None:
            logits = agent.act({"mask": mask})
            broker.submit_target_weights(
                {} if cap == 0.0 else weights_from_logits(logits, SYMBOLS, mask, cap)
            )
        broker.open_session(day, opens[day])
        broker.mark_to_market(day, closes[day])

    broker.end_run()
    return broker


def _fills_digest(broker: PaperBroker) -> str:
    """One hash over every fill: date, symbol, side, quantity, price, fees."""
    orders = {o.order_id: o for o in broker.get_orders()}
    lines: list[str] = []
    for snapshot in sorted(orders.values(), key=lambda o: o.order_id):
        if snapshot.status != "FILLED":
            continue
        lines.append(
            f"{snapshot.ts_updated.date()}|{snapshot.symbol}|{snapshot.side}"
            f"|{snapshot.filled_quantity}|{snapshot.average_price:.4f}"
        )
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def summarise(broker: PaperBroker) -> dict[str, Any]:
    orders = broker.get_orders()
    liabilities = broker.tax_liabilities()
    navs = [round(s.nav, 2) for s in broker.snapshots]
    return {
        "n_orders": len(orders),
        "n_filled": sum(1 for o in orders if o.status == "FILLED"),
        "n_rejected": sum(1 for o in orders if o.status == "REJECTED"),
        "final_nav": round(broker.nav, 2),
        "final_cash": round(broker.cash, 2),
        # The final liquidation settles T+1, so on the last session almost the
        # whole book is money the account owns but cannot yet spend.
        "final_unsettled": round(broker.unsettled_cash, 2),
        "final_equity": round(broker.equity_value, 2),
        "realised_pnl": round(broker.realised_pnl, 2),
        "fees_paid": round(broker.fees_paid, 2),
        "financial_years": [liability.financial_year for liability in liabilities],
        "short_term_gain": [round(liability.short_term_gain, 2) for liability in liabilities],
        "long_term_gain": [round(liability.long_term_gain, 2) for liability in liabilities],
        "tax_total": round(sum(liability.total for liability in liabilities), 2),
        "nav_first": navs[0],
        "nav_mid": navs[len(navs) // 2],
        "nav_last": navs[-1],
        "nav_min": min(navs),
        "nav_max": max(navs),
        "fills_digest": _fills_digest(broker),
    }


# ── GOLDEN — regenerate with PAPER_GOLDEN_UPDATE=1, never by hand ────────────
GOLDEN: dict[str, Any] = {
    "fees_paid": 683.59,
    "fills_digest": "8d6502386c4c46fe8b070d0ad27cb5553f1ffe3330de348c372df52c90732d26",
    "final_cash": 5409.26,
    "final_equity": 0.0,
    "final_nav": 100813.45,
    "final_unsettled": 95404.19,
    "financial_years": [2023, 2024],
    "long_term_gain": [0.0, 1718.36],
    "n_filled": 48,
    "n_orders": 48,
    "n_rejected": 0,
    "nav_first": 99519.16,
    "nav_last": 100813.45,
    "nav_max": 101931.01,
    "nav_mid": 99103.24,
    "nav_min": 97544.95,
    "realised_pnl": 1497.03,
    "short_term_gain": [-608.21, 386.88],
    "tax_total": 80.47,
}
# ── end GOLDEN ───────────────────────────────────────────────────────────────


def test_scenario_matches_the_golden(db_session: Session) -> None:
    broker = run_scenario(db_session)
    actual = summarise(broker)

    if os.getenv("PAPER_GOLDEN_UPDATE"):
        import pprint

        print("\nGOLDEN: dict[str, Any] = " + pprint.pformat(actual, width=96, sort_dicts=True))
        pytest.fail("PAPER_GOLDEN_UPDATE set — golden printed above, not asserted")

    assert actual == GOLDEN


def test_in_memory_mode_produces_the_same_books_as_postgres(db_session: Session) -> None:
    """Persistence must be a side effect, never an input to the arithmetic."""
    assert summarise(run_scenario(None)) == summarise(run_scenario(db_session))


def test_ledger_rows_reconcile_with_the_in_memory_book(db_session: Session) -> None:
    """Whatever the broker believes, the ledger has to be able to prove it."""
    broker = run_scenario(db_session)
    run_id = broker.run_id
    assert run_id is not None

    run = db_session.get(schema.StrategyRun, run_id)
    assert run is not None
    assert run.mode == "paper"
    assert float(run.initial_cash) == pytest.approx(100_000.0)
    assert run.ended_at is not None

    orders = list(
        db_session.scalars(
            select(schema.Order).where(schema.Order.strategy_run_id == run_id)
        )
    )
    assert len(orders) == len(broker.get_orders())
    assert {o.status for o in orders} <= {"FILLED", "REJECTED", "CANCELLED", "PENDING"}

    n_fills = db_session.scalar(
        select(func.count())
        .select_from(schema.Fill)
        .join(schema.Order, schema.Fill.order_id == schema.Order.id)
        .where(schema.Order.strategy_run_id == run_id)
    )
    assert n_fills == sum(1 for o in broker.get_orders() if o.status == "FILLED")

    # Every fill's fees and price survived the Numeric(18,4) round trip.
    fee_total = db_session.scalar(
        select(func.sum(schema.Fill.fees))
        .select_from(schema.Fill)
        .join(schema.Order, schema.Fill.order_id == schema.Order.id)
        .where(schema.Order.strategy_run_id == run_id)
    )
    assert float(fee_total or 0.0) == pytest.approx(broker.fees_paid, abs=0.01)

    # One snapshot and one P&L row per session, and the snapshots carry the
    # same NAV path the broker reported.
    snapshots = list(
        db_session.scalars(
            select(schema.PortfolioSnapshot)
            .where(schema.PortfolioSnapshot.strategy_run_id == run_id)
            .order_by(schema.PortfolioSnapshot.ts)
        )
    )
    assert len(snapshots) == N_SESSIONS
    for row, snapshot in zip(snapshots, broker.snapshots, strict=True):
        assert float(row.total_value) == pytest.approx(snapshot.nav, abs=0.01)
        assert float(row.cash) == pytest.approx(snapshot.cash, abs=0.01)
        assert float(row.equity_value) == pytest.approx(snapshot.equity_value, abs=0.01)

    pnl_rows = list(
        db_session.scalars(
            select(schema.PnlDaily)
            .where(schema.PnlDaily.strategy_run_id == run_id)
            .order_by(schema.PnlDaily.date)
        )
    )
    assert len(pnl_rows) == N_SESSIONS
    # Daily totals must telescope into the whole run's P&L.
    assert float(sum(row.total for row in pnl_rows)) == pytest.approx(
        broker.nav - 100_000.0, abs=0.05
    )

    # The lot book is fully closed out by the final liquidation, and the ledger
    # agrees: no scrip left, no open remaining_quantity anywhere.
    open_qty = db_session.scalar(
        select(func.sum(schema.Lot.remaining_quantity)).where(
            schema.Lot.strategy_run_id == run_id
        )
    )
    assert int(open_qty or 0) == 0
    assert broker.get_positions() == ()

    positions = list(
        db_session.scalars(
            select(schema.Position).where(schema.Position.strategy_run_id == run_id)
        )
    )
    assert {p.symbol for p in positions} == set(SYMBOLS)
    assert all(p.quantity == 0 for p in positions)
