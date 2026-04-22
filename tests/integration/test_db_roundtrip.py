"""Round-trip insert/select tests for every table in M1 schema.

Requires: make db-up && make db-migrate
Each test is wrapped in a rolled-back session so the DB stays clean.
"""

from datetime import UTC, date, datetime
from decimal import Decimal

import pytest
from sqlalchemy.orm import Session

from trader.broker.schema import Fill, Order, PnlDaily, PortfolioSnapshot, Position, StrategyRun
from trader.db.market_models import (
    CorporateAction,
    DatasetVersion,
    Instrument,
    TradingCalendarDay,
    UniverseSnapshot,
)

pytestmark = pytest.mark.usefixtures("db_session")

NOW = datetime.now(tz=UTC)
TODAY = date.today()


# ------------------------------------------------------------------ helpers


def _flush_expire_get(session: Session, model: type, pk: int) -> object:
    session.flush()
    session.expire_all()
    obj = session.get(model, pk)
    assert obj is not None
    return obj


# ------------------------------------------------------------------ market.*


def test_instrument_roundtrip(db_session: Session) -> None:
    inst = Instrument(
        symbol="RELIANCE.NS",
        name="Reliance Industries",
        sector="Oil Gas Power",
        sector_id=4,
        isin="INE002A01018",
        listed_on=date(1995, 11, 29),
        lot_size=1,
    )
    db_session.add(inst)
    db_session.flush()
    pk = inst.id

    db_session.expire_all()
    fetched = db_session.get(Instrument, pk)
    assert fetched is not None
    assert fetched.symbol == "RELIANCE.NS"
    assert fetched.sector_id == 4
    assert fetched.isin == "INE002A01018"
    assert fetched.lot_size == 1


def test_universe_snapshot_roundtrip(db_session: Session) -> None:
    snap = UniverseSnapshot(
        universe_name="universe_v1",
        effective_date=date(2024, 1, 1),
        symbols=["RELIANCE.NS", "TCS.NS", "INFY.NS"],
    )
    db_session.add(snap)
    db_session.flush()
    pk = snap.id

    db_session.expire_all()
    fetched = db_session.get(UniverseSnapshot, pk)
    assert fetched is not None
    assert fetched.universe_name == "universe_v1"
    assert "TCS.NS" in fetched.symbols
    assert len(fetched.symbols) == 3


def test_corporate_action_roundtrip(db_session: Session) -> None:
    action = CorporateAction(
        symbol="RELIANCE.NS",
        action_date=date(2017, 9, 19),
        action_type="BONUS",
        ratio=Decimal("1.0"),
        notes="1:1 bonus issue",
    )
    db_session.add(action)
    db_session.flush()
    pk = action.id

    db_session.expire_all()
    fetched = db_session.get(CorporateAction, pk)
    assert fetched is not None
    assert fetched.action_type == "BONUS"
    assert fetched.symbol == "RELIANCE.NS"


def test_trading_calendar_roundtrip(db_session: Session) -> None:
    cal = TradingCalendarDay(date=date(2024, 1, 15), is_trading_day=True)
    db_session.add(cal)
    db_session.flush()

    db_session.expire_all()
    fetched = db_session.get(TradingCalendarDay, date(2024, 1, 15))
    assert fetched is not None
    assert fetched.is_trading_day is True


def test_dataset_version_roundtrip(db_session: Session) -> None:
    dv = DatasetVersion(
        name="universe_v1",
        split="train",
        file_path="data/panels/train.parquet",
        sha256="abc123" * 10,
        num_dates=1826,
        num_tickers=150,
        num_features=15,
    )
    db_session.add(dv)
    db_session.flush()
    pk = dv.id

    db_session.expire_all()
    fetched = db_session.get(DatasetVersion, pk)
    assert fetched is not None
    assert fetched.split == "train"
    assert fetched.num_tickers == 150


# ------------------------------------------------------------------ ledger.*


def _make_strategy_run(session: Session) -> StrategyRun:
    run = StrategyRun(
        strategy_id="test-strategy",
        config_hash="deadbeef",
        initial_cash=Decimal("1000000.00"),
        mode="paper",
    )
    session.add(run)
    session.flush()
    return run


def test_strategy_run_roundtrip(db_session: Session) -> None:
    run = _make_strategy_run(db_session)
    pk = run.id

    db_session.expire_all()
    fetched = db_session.get(StrategyRun, pk)
    assert fetched is not None
    assert fetched.strategy_id == "test-strategy"
    assert fetched.mode == "paper"
    assert fetched.initial_cash == Decimal("1000000.00")


def test_order_roundtrip(db_session: Session) -> None:
    run = _make_strategy_run(db_session)
    order = Order(
        strategy_run_id=run.id,
        client_order_id="ord-001",
        ts_submitted=NOW,
        symbol="TCS.NS",
        side="BUY",
        quantity=10,
        order_type="MARKET",
        status="PENDING",
    )
    db_session.add(order)
    db_session.flush()
    pk = order.id

    db_session.expire_all()
    fetched = db_session.get(Order, pk)
    assert fetched is not None
    assert fetched.client_order_id == "ord-001"
    assert fetched.side == "BUY"
    assert fetched.quantity == 10


def test_fill_roundtrip(db_session: Session) -> None:
    run = _make_strategy_run(db_session)
    order = Order(
        strategy_run_id=run.id,
        client_order_id="ord-002",
        ts_submitted=NOW,
        symbol="INFY.NS",
        side="BUY",
        quantity=5,
        order_type="MARKET",
        status="FILLED",
    )
    db_session.add(order)
    db_session.flush()

    fill = Fill(
        order_id=order.id,
        ts=NOW,
        quantity=5,
        price=Decimal("1800.50"),
        fees=Decimal("18.00"),
        slippage=Decimal("0.000125"),
    )
    db_session.add(fill)
    db_session.flush()
    pk = fill.id

    db_session.expire_all()
    fetched = db_session.get(Fill, pk)
    assert fetched is not None
    assert fetched.price == Decimal("1800.50")
    assert fetched.quantity == 5


def test_position_roundtrip(db_session: Session) -> None:
    run = _make_strategy_run(db_session)
    pos = Position(
        strategy_run_id=run.id,
        symbol="HDFC.NS",
        quantity=20,
        avg_price=Decimal("1600.00"),
    )
    db_session.add(pos)
    db_session.flush()
    pk = pos.id

    db_session.expire_all()
    fetched = db_session.get(Position, pk)
    assert fetched is not None
    assert fetched.symbol == "HDFC.NS"
    assert fetched.quantity == 20


def test_portfolio_snapshot_roundtrip(db_session: Session) -> None:
    run = _make_strategy_run(db_session)
    snap = PortfolioSnapshot(
        strategy_run_id=run.id,
        ts=NOW,
        cash=Decimal("800000.00"),
        equity_value=Decimal("200000.00"),
        total_value=Decimal("1000000.00"),
        realized_pnl=Decimal("0.00"),
        unrealized_pnl=Decimal("5000.00"),
        fees_paid=Decimal("18.00"),
        turnover=Decimal("9002.50"),
        metrics_json={},
    )
    db_session.add(snap)
    db_session.flush()
    pk = snap.id

    db_session.expire_all()
    fetched = db_session.get(PortfolioSnapshot, pk)
    assert fetched is not None
    assert fetched.total_value == Decimal("1000000.00")
    assert fetched.metrics_json == {}


def test_pnl_daily_roundtrip(db_session: Session) -> None:
    run = _make_strategy_run(db_session)
    pnl = PnlDaily(
        strategy_run_id=run.id,
        date=TODAY,
        realized=Decimal("0.00"),
        unrealized=Decimal("5000.00"),
        fees=Decimal("18.00"),
        total=Decimal("4982.00"),
    )
    db_session.add(pnl)
    db_session.flush()

    db_session.expire_all()
    fetched = db_session.get(PnlDaily, (run.id, TODAY))
    assert fetched is not None
    assert fetched.total == Decimal("4982.00")
    assert fetched.fees == Decimal("18.00")
