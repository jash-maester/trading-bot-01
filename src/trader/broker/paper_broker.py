"""A paper broker: real prices in, dummy money out, everything on the ledger.

This is the object that stands in for the user's Zerodha account until live
trading is switched on.  It implements :class:`~trader.broker.base.Broker`, so
the runner that drives it cannot tell the difference between this and a Kite
adapter, and it writes the same rows to ``ledger.*`` that a live run would.

The design is dictated by one constraint: **the paper run and the backtest must
agree**.  If :class:`~trader.env.panel_env.PanelTradingEnv` says a strategy made
14% and the paper broker says 9%, neither number means anything.  So the
mechanics that decide *what* gets traded are copied from the environment rather
than reinvented — NAV marked at the previous close, target weights times NAV
divided by the next open, floored to whole shares:

    target_shares = floor(nav * weight / open)

Everything the environment cannot model, because it runs on a NumPy array with
one cash scalar, is added here and is the reason this file exists:

**T+1 cash settlement.**  Sale proceeds are not money you can spend today.  The
environment keeps a single ``cash`` float and lets a sale fund a purchase in the
same step; this broker keeps a settlement pipeline, and a rebalance that funds
itself out of today's sales will get its buy legs rejected for want of settled
cash.  That is not a bug in the simulator — it is the constraint a real account
is under, and it is the main reason a paper run underperforms its backtest.

**DP charges are per scrip per day, not per order.**  The depository debits
₹15.34 when a scrip leaves the demat account, once, however many sell orders it
took to empty the position.  Charging it per order is the single easiest thing
to get wrong here, so the count of distinct scrips sold is tracked per session
and only the first sell order of a symbol carries the fee.

**Intraday round trips are a pricing trap.**  A buy and a sell of the same
scrip on the same day is taxed as a speculative intraday trade even when both
orders were tagged CNC — different STT, different brokerage, no DP charge.  A
daily-bar simulator that fills once, at the open, physically cannot produce
one, so rather than silently pricing an impossible trade at the wrong rates
this broker raises :class:`IntradayRoundTripError`.  If that exception ever
fires, the caller has invented a fill the daily panel cannot justify.

**Tax is a liability, not a cost.**  Realised gains are matched FIFO against
``ledger.lots`` and accumulated per financial year by
:class:`~trader.env.tax.TaxModel`.  Capital-gains tax is assessed on the year,
after netting, with a ₹1.25L long-term exemption — deducting it per trade would
be both wrong and roughly 20% too pessimistic on a losing year.  It is reported
alongside NAV and never subtracted from cash.

Persistence is optional.  Constructed without a SQLAlchemy ``Session`` the
broker runs entirely in memory, which is what the unit tests use; given one, it
writes ``strategy_runs`` → ``orders`` → ``fills`` → ``lots`` → ``positions`` →
``portfolio_snapshots`` → ``pnl_daily``.

Accounting invariant, checked after every session and every mark::

    nav == initial_cash + realised_pnl + unrealised_pnl - fees_paid
    nav == settled_cash + unsettled_cash + equity_value

This holds exactly (to float tolerance) because a lot's cost basis is the gross
fill price and every charge is expensed to cash — see :meth:`_assert_identity`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal
from math import floor, isfinite
from typing import Any, Literal

from trader.broker import schema
from trader.broker.base import (
    Broker,
    BrokerError,
    BrokerOrder,
    Exchange,
    Fill,
    Holding,
    InvalidOrderState,
    Margins,
    OrderNotFound,
    OrderRequest,
    OrderStatus,
    Position,
    Side,
)
from trader.env.costs import (
    DEFAULT_MIN_TRADE_VALUE,
    CostModel,
    ZerodhaEquityDeliveryCostModel,
)
from trader.env.tax import (
    FifoLotBook,
    Lot,
    RealisedGain,
    TaxLiability,
    TaxModel,
    financial_year_label,
)

# NSE trades in IST and the ledger columns are timestamptz, so every timestamp
# this module writes is an aware IST datetime at a real session boundary.  A
# naive datetime here would be silently reinterpreted as UTC by Postgres and
# would drop every fill into the previous trading day.
IST = timezone(timedelta(hours=5, minutes=30), name="IST")
MARKET_OPEN = time(9, 15)
MARKET_CLOSE = time(15, 30)

# ₹1,00,000 — the account this is standing in for.  `configs/broker/paper.yaml`
# overrides it; the default is what a bare PaperBroker() starts with.
DEFAULT_INITIAL_CASH = 100_000.0
DEFAULT_MAX_WEIGHT = 0.10
# DEFAULT_MIN_TRADE_VALUE is imported from trader.env.costs above so the broker
# and the environment gate on one constant. They disagreed for months.
DEFAULT_SETTLEMENT_DAYS = 1

# Impact coefficient for the ``atr_impact`` slippage model.  This MUST stay
# equal to ``trader.env.panel_env._SLIPPAGE_K`` or the paper run and the
# backtest fill at different prices; it is duplicated rather than imported so
# that this module does not drag gymnasium and polars in behind it, and
# ``tests/unit/test_paper_broker.py`` asserts the two are equal.
SLIPPAGE_K = 0.1

SlippageModel = Literal["none", "percentage", "atr_impact"]

# Reject reasons, stored verbatim in ``ledger.orders.reject_reason``.
REJECT_NO_PRICE = "NO_PRICE"
REJECT_INSUFFICIENT_FUNDS = "INSUFFICIENT_FUNDS"
REJECT_INSUFFICIENT_HOLDING = "INSUFFICIENT_HOLDING"
CANCEL_LIMIT_NOT_MARKETABLE = "LIMIT_NOT_MARKETABLE"

# Relative tolerance for the accounting identity.  Rupee sums of ~1e6 over a
# few thousand fills accumulate float error in the 1e-9 relative range; 1e-7
# is loose enough never to false-positive and tight enough that a genuinely
# lost paisa (1e-8 relative on a ₹1L account) still trips it.
_IDENTITY_RTOL = 1e-7


class AccountingError(BrokerError):
    """The books do not balance — raised by the invariant check, never caught.

    Every path that moves money in this module is supposed to preserve
    ``nav == initial + realised + unrealised - fees``.  If it does not, some
    cash has been created or destroyed and every number the run reports is
    suspect, so the run stops rather than continuing with a quiet leak.
    """


class IntradayRoundTripError(BrokerError):
    """A symbol was bought and sold in the same session.

    See the module docstring: a daily-bar simulator cannot produce this, and
    pricing it at CNC delivery rates would be wrong on STT, brokerage and DP
    simultaneously.  Raised rather than silently mispriced.
    """


@dataclass(frozen=True, slots=True)
class ImpactInputs:
    """Per-symbol inputs for the ``atr_impact`` slippage model.

    Both are computed by the caller so the formula here is *identical* to the
    environment's rather than merely similar: the environment divides by the
    session's close, and a broker that only sees the open would divide by the
    open and quietly fill at a different price.
    """

    atr_fraction: float  # ATR(14) / close
    adv_shares: float  # 20-day dollar volume / close


@dataclass(frozen=True, slots=True)
class PaperBrokerConfig:
    """Everything `configs/broker/paper.yaml` can set, plus the sizing rules."""

    initial_cash: float = DEFAULT_INITIAL_CASH
    settlement_days: int = DEFAULT_SETTLEMENT_DAYS
    max_weight_per_name: float = DEFAULT_MAX_WEIGHT
    min_trade_value: float = DEFAULT_MIN_TRADE_VALUE
    slippage_model: SlippageModel = "percentage"
    slippage_pct: float = 0.001
    exchange: Exchange = "NSE"

    def __post_init__(self) -> None:
        if self.initial_cash <= 0.0:
            raise ValueError(f"initial_cash must be positive, got {self.initial_cash}")
        if self.settlement_days < 0:
            raise ValueError(f"settlement_days must be >= 0, got {self.settlement_days}")
        if not 0.0 < self.max_weight_per_name <= 1.0:
            raise ValueError(
                f"max_weight_per_name must be in (0, 1], got {self.max_weight_per_name}"
            )
        if self.min_trade_value < 0.0:
            raise ValueError(f"min_trade_value must be >= 0, got {self.min_trade_value}")
        if self.slippage_pct < 0.0:
            raise ValueError(f"slippage_pct must be >= 0, got {self.slippage_pct}")
        if self.slippage_model not in ("none", "percentage", "atr_impact"):
            raise ValueError(f"unknown slippage_model {self.slippage_model!r}")


@dataclass(frozen=True, slots=True)
class DaySnapshot:
    """The state of the account at one session's close."""

    day: date
    settled_cash: float
    unsettled_cash: float
    equity_value: float
    nav: float
    realised_pnl: float
    unrealised_pnl: float
    fees_paid: float
    turnover: float

    @property
    def cash(self) -> float:
        return self.settled_cash + self.unsettled_cash


@dataclass(frozen=True, slots=True)
class SessionResult:
    """What one call to :meth:`PaperBroker.open_session` did."""

    day: date
    fills: tuple[Fill, ...]
    rejected: tuple[BrokerOrder, ...]
    cash_settled: float
    fees: float
    turnover: float


# ── ledger persistence ────────────────────────────────────────────────────────


class LedgerRecorder(ABC):
    """Where the broker's lifecycle goes.

    Split out so the broker itself has no ``if self._session is not None``
    scattered through it, and so the unit tests exercise exactly the same code
    path the Postgres runs do — only the recorder changes.
    """

    @abstractmethod
    def start_run(
        self,
        *,
        strategy_id: str,
        config_hash: str,
        initial_cash: float,
        started_at: datetime,
        notes: str | None,
    ) -> int: ...

    @abstractmethod
    def end_run(self, run_id: int, ended_at: datetime) -> None: ...

    @abstractmethod
    def record_order(self, run_id: int, order: BrokerOrder) -> int: ...

    @abstractmethod
    def update_order(self, row_id: int, order: BrokerOrder) -> None: ...

    @abstractmethod
    def record_fill(self, order_row_id: int, fill: Fill) -> None: ...

    @abstractmethod
    def record_lot(
        self, run_id: int, symbol: str, buy_date: date, quantity: int, basis: float
    ) -> int: ...

    @abstractmethod
    def update_lot(self, row_id: int, remaining: int) -> None: ...

    @abstractmethod
    def upsert_position(
        self,
        run_id: int,
        symbol: str,
        quantity: int,
        avg_price: float,
        last_mark: float | None,
        ts: datetime,
    ) -> None: ...

    @abstractmethod
    def record_snapshot(
        self, run_id: int, snapshot: DaySnapshot, ts: datetime, metrics: dict[str, Any]
    ) -> None: ...

    @abstractmethod
    def record_pnl(
        self,
        run_id: int,
        day: date,
        realized: float,
        unrealized: float,
        fees: float,
        total: float,
    ) -> None: ...

    @abstractmethod
    def flush(self) -> None: ...


class NullRecorder(LedgerRecorder):
    """In-memory mode: hands out row ids and forgets everything else."""

    def __init__(self) -> None:
        self._next_id = 1

    def _issue(self) -> int:
        row_id = self._next_id
        self._next_id += 1
        return row_id

    def start_run(
        self,
        *,
        strategy_id: str,
        config_hash: str,
        initial_cash: float,
        started_at: datetime,
        notes: str | None,
    ) -> int:
        return self._issue()

    def end_run(self, run_id: int, ended_at: datetime) -> None:
        return None

    def record_order(self, run_id: int, order: BrokerOrder) -> int:
        return self._issue()

    def update_order(self, row_id: int, order: BrokerOrder) -> None:
        return None

    def record_fill(self, order_row_id: int, fill: Fill) -> None:
        return None

    def record_lot(
        self, run_id: int, symbol: str, buy_date: date, quantity: int, basis: float
    ) -> int:
        return self._issue()

    def update_lot(self, row_id: int, remaining: int) -> None:
        return None

    def upsert_position(
        self,
        run_id: int,
        symbol: str,
        quantity: int,
        avg_price: float,
        last_mark: float | None,
        ts: datetime,
    ) -> None:
        return None

    def record_snapshot(
        self, run_id: int, snapshot: DaySnapshot, ts: datetime, metrics: dict[str, Any]
    ) -> None:
        return None

    def record_pnl(
        self,
        run_id: int,
        day: date,
        realized: float,
        unrealized: float,
        fees: float,
        total: float,
    ) -> None:
        return None

    def flush(self) -> None:
        return None


def _dec(value: float, places: int) -> Decimal:
    """Round a rupee float to a ``Numeric(_, places)`` column's scale.

    Postgres would round on insert anyway; doing it here means the value that
    comes back out of the DB is the value the golden tests asserted on.
    """
    return Decimal(f"{value:.{places}f}")


class SqlRecorder(LedgerRecorder):
    """Writes the run to ``ledger.*`` through a SQLAlchemy session.

    Flushes rather than commits: transaction scope belongs to whoever opened
    the session, which lets the integration tests roll the whole run back.
    """

    def __init__(self, session: Any) -> None:
        self._session = session
        self._positions: dict[tuple[int, str], schema.Position] = {}

    def start_run(
        self,
        *,
        strategy_id: str,
        config_hash: str,
        initial_cash: float,
        started_at: datetime,
        notes: str | None,
    ) -> int:
        run = schema.StrategyRun(
            strategy_id=strategy_id,
            started_at=started_at,
            config_hash=config_hash,
            initial_cash=_dec(initial_cash, 2),
            mode="paper",
            notes=notes,
        )
        self._session.add(run)
        self._session.flush()
        return int(run.id)

    def end_run(self, run_id: int, ended_at: datetime) -> None:
        run = self._session.get(schema.StrategyRun, run_id)
        if run is not None:
            run.ended_at = ended_at
        self._session.flush()

    def record_order(self, run_id: int, order: BrokerOrder) -> int:
        row = schema.Order(
            strategy_run_id=run_id,
            client_order_id=order.order_id,
            ts_submitted=order.ts_submitted,
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            order_type=order.order_type,
            limit_price=None if order.limit_price is None else _dec(order.limit_price, 4),
            status=order.status,
            reject_reason=order.reject_reason,
            tag=order.tag,
        )
        self._session.add(row)
        self._session.flush()
        return int(row.id)

    def update_order(self, row_id: int, order: BrokerOrder) -> None:
        row = self._session.get(schema.Order, row_id)
        if row is None:
            return
        row.status = order.status
        row.reject_reason = order.reject_reason

    def record_fill(self, order_row_id: int, fill: Fill) -> None:
        self._session.add(
            schema.Fill(
                order_id=order_row_id,
                ts=fill.ts,
                quantity=fill.quantity,
                price=_dec(fill.price, 4),
                fees=_dec(fill.fees, 4),
                slippage=_dec(fill.slippage, 6),
            )
        )

    def record_lot(
        self, run_id: int, symbol: str, buy_date: date, quantity: int, basis: float
    ) -> int:
        row = schema.Lot(
            strategy_run_id=run_id,
            symbol=symbol,
            buy_date=buy_date,
            quantity=quantity,
            remaining_quantity=quantity,
            cost_basis_per_share=_dec(basis, 4),
        )
        self._session.add(row)
        self._session.flush()
        return int(row.id)

    def update_lot(self, row_id: int, remaining: int) -> None:
        row = self._session.get(schema.Lot, row_id)
        if row is not None:
            row.remaining_quantity = remaining

    def upsert_position(
        self,
        run_id: int,
        symbol: str,
        quantity: int,
        avg_price: float,
        last_mark: float | None,
        ts: datetime,
    ) -> None:
        # ``ledger.positions`` is unique on (run, symbol), so a closed position
        # keeps its row at quantity 0 rather than being deleted — the audit
        # trail of "we held this once" is worth one row.
        row = self._positions.get((run_id, symbol))
        if row is None:
            row = schema.Position(
                strategy_run_id=run_id,
                symbol=symbol,
                quantity=quantity,
                avg_price=_dec(avg_price, 4),
                last_mark=None if last_mark is None else _dec(last_mark, 4),
                updated_at=ts,
            )
            self._session.add(row)
            self._positions[(run_id, symbol)] = row
            return
        row.quantity = quantity
        row.avg_price = _dec(avg_price, 4)
        row.last_mark = None if last_mark is None else _dec(last_mark, 4)
        row.updated_at = ts

    def record_snapshot(
        self, run_id: int, snapshot: DaySnapshot, ts: datetime, metrics: dict[str, Any]
    ) -> None:
        self._session.add(
            schema.PortfolioSnapshot(
                strategy_run_id=run_id,
                ts=ts,
                cash=_dec(snapshot.cash, 2),
                equity_value=_dec(snapshot.equity_value, 2),
                total_value=_dec(snapshot.nav, 2),
                realized_pnl=_dec(snapshot.realised_pnl, 2),
                unrealized_pnl=_dec(snapshot.unrealised_pnl, 2),
                fees_paid=_dec(snapshot.fees_paid, 2),
                turnover=_dec(snapshot.turnover, 2),
                metrics_json=metrics,
            )
        )

    def record_pnl(
        self,
        run_id: int,
        day: date,
        realized: float,
        unrealized: float,
        fees: float,
        total: float,
    ) -> None:
        self._session.add(
            schema.PnlDaily(
                strategy_run_id=run_id,
                date=day,
                realized=_dec(realized, 2),
                unrealized=_dec(unrealized, 2),
                fees=_dec(fees, 2),
                total=_dec(total, 2),
            )
        )

    def flush(self) -> None:
        self._session.flush()


# ── internal order bookkeeping ────────────────────────────────────────────────


@dataclass(slots=True)
class _WorkingOrder:
    """An order plus the mutable execution state the public dataclass omits."""

    request: OrderRequest
    order_id: str
    row_id: int
    ts_submitted: datetime
    status: OrderStatus = "PENDING"
    filled_quantity: int = 0
    average_price: float = 0.0
    reject_reason: str | None = None
    ts_updated: datetime = field(default_factory=lambda: datetime.now(tz=IST))

    def snapshot(self) -> BrokerOrder:
        r = self.request
        return BrokerOrder(
            order_id=self.order_id,
            symbol=r.symbol,
            side=r.side,
            quantity=r.quantity,
            product=r.product,
            order_type=r.order_type,
            status=self.status,
            ts_submitted=self.ts_submitted,
            ts_updated=self.ts_updated,
            filled_quantity=self.filled_quantity,
            average_price=self.average_price,
            limit_price=r.limit_price,
            exchange=r.exchange,
            tag=r.tag,
            reject_reason=self.reject_reason,
        )


class PaperBroker(Broker):
    """Simulated Zerodha account, executing daily bars against the ledger.

    Drive it one session at a time::

        broker.start_run(strategy_id="mlp_regime_seed42")
        broker.submit_target_weights({"INFY.NS": 0.08, ...})   # decided at t-1 close
        broker.open_session(day, opens)                        # fills at t open
        broker.mark_to_market(day, closes)                     # snapshot at t close

    Orders placed through :meth:`place_order` join the same queue and execute
    at the next open — nothing fills in the session it was submitted in, which
    is the whole point of a next-open simulator.
    """

    def __init__(
        self,
        config: PaperBrokerConfig | None = None,
        *,
        session: Any | None = None,
        cost_model: CostModel | None = None,
        strategy_id: str = "paper",
    ) -> None:
        self._cfg = config or PaperBrokerConfig()
        self._cost_model: CostModel = cost_model or ZerodhaEquityDeliveryCostModel(
            exchange=self._cfg.exchange
        )
        self._recorder: LedgerRecorder = (
            NullRecorder() if session is None else SqlRecorder(session)
        )
        self._strategy_id = strategy_id
        self._run_id: int | None = None

        # ── money ────────────────────────────────────────────────────────────
        self._cash = float(self._cfg.initial_cash)
        # (session index at which it becomes spendable, amount) — the T+1 pipe.
        self._pending: list[tuple[int, float]] = []
        self._fees_paid = 0.0
        self._realised_pnl = 0.0

        # ── positions and lots ───────────────────────────────────────────────
        self._shares: dict[str, int] = {}
        self._marks: dict[str, float] = {}
        self._lots = FifoLotBook()
        # Row ids mirroring each symbol's FIFO deque, oldest first, so a
        # consumed lot can be written back to ``ledger.lots`` without
        # re-deriving the matching that FifoLotBook already did.
        self._lot_rows: dict[str, list[int]] = {}
        self._tax = TaxModel()

        # ── order book ───────────────────────────────────────────────────────
        self._orders: dict[str, _WorkingOrder] = {}
        self._order_seq = 0
        self._queue: list[str] = []
        self._targets: dict[str, float] | None = None
        self._target_ts: datetime | None = None

        # ── session state ────────────────────────────────────────────────────
        self._session_index = -1
        self._day: date | None = None
        self._last_marked_day: date | None = None
        self._nav = float(self._cfg.initial_cash)
        self._prev_nav = float(self._cfg.initial_cash)
        self._prev_realised = 0.0
        self._prev_unrealised = 0.0
        self._prev_fees = 0.0
        self._sold_today: set[str] = set()
        self._bought_today: dict[str, int] = {}
        self._session_fees = 0.0
        self._session_turnover = 0.0
        self._snapshots: list[DaySnapshot] = []

    # ── run lifecycle ────────────────────────────────────────────────────────

    def start_run(
        self,
        *,
        strategy_id: str | None = None,
        config_hash: str | None = None,
        notes: str | None = None,
        started_at: datetime | None = None,
    ) -> int:
        """Open a ``ledger.strategy_runs`` row and return its id."""
        if self._run_id is not None:
            raise BrokerError("run already started")
        self._strategy_id = strategy_id or self._strategy_id
        self._run_id = self._recorder.start_run(
            strategy_id=self._strategy_id,
            config_hash=config_hash or self._config_hash(),
            initial_cash=self._cfg.initial_cash,
            started_at=started_at or datetime.now(tz=IST),
            notes=notes,
        )
        return self._run_id

    def end_run(self, ended_at: datetime | None = None) -> None:
        """Stamp ``ended_at`` and flush.  Committing is the caller's job.

        Wall clock, not the last session's close: ``strategy_runs.started_at``
        is wall clock too, and a replay of 2022 data stamped with a 2022
        ``ended_at`` would make every "how long did this run take" query return
        a negative duration.  The trading period is in ``pnl_daily``.
        """
        if self._run_id is None:
            return
        self._recorder.end_run(self._run_id, ended_at or datetime.now(tz=IST))
        self._recorder.flush()

    def _config_hash(self) -> str:
        """Stable digest of the sizing and cost configuration.

        Cheap identity for "was this run produced by the same rules": two runs
        with the same hash are comparable, two with different hashes are not.
        """
        import hashlib

        payload = "|".join(
            f"{k}={v}"
            for k, v in (
                ("initial_cash", self._cfg.initial_cash),
                ("settlement_days", self._cfg.settlement_days),
                ("max_weight", self._cfg.max_weight_per_name),
                ("min_trade_value", self._cfg.min_trade_value),
                ("slippage_model", self._cfg.slippage_model),
                ("slippage_pct", self._cfg.slippage_pct),
                ("exchange", self._cfg.exchange),
                ("cost_model", type(self._cost_model).__name__),
            )
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    @property
    def run_id(self) -> int | None:
        return self._run_id

    def _require_run(self) -> int:
        if self._run_id is None:
            # Auto-starting would hide a missing start_run() until the ledger
            # came up empty, so make it loud instead.
            raise BrokerError("start_run() must be called before trading")
        return self._run_id

    # ── Broker API ───────────────────────────────────────────────────────────

    def place_order(self, request: OrderRequest) -> str:
        """Queue ``request`` for execution at the next session's open."""
        run_id = self._require_run()
        if request.product != "CNC":
            # MIS would need same-day exits, which this simulator refuses to
            # price (see IntradayRoundTripError).  Better to say so up front.
            raise BrokerError(
                f"paper broker trades delivery only; product={request.product!r} "
                "cannot be filled by a daily-bar simulator"
            )
        self._order_seq += 1
        order_id = f"P{run_id:d}-{self._order_seq:06d}"
        ts = self._submit_ts()
        working = _WorkingOrder(
            request=request, order_id=order_id, row_id=0, ts_submitted=ts, ts_updated=ts
        )
        working.row_id = self._recorder.record_order(run_id, working.snapshot())
        self._orders[order_id] = working
        self._queue.append(order_id)
        return order_id

    def cancel_order(self, order_id: str) -> None:
        working = self._orders.get(order_id)
        if working is None:
            raise OrderNotFound(order_id)
        if not working.snapshot().is_open:
            raise InvalidOrderState(
                f"order {order_id} is {working.status} and cannot be cancelled"
            )
        working.status = "CANCELLED"
        working.ts_updated = self._submit_ts()
        if order_id in self._queue:
            self._queue.remove(order_id)
        self._recorder.update_order(working.row_id, working.snapshot())

    def get_orders(self) -> tuple[BrokerOrder, ...]:
        return tuple(w.snapshot() for w in self._orders.values())

    def get_positions(self) -> tuple[Position, ...]:
        """Net positions, one row per held symbol.

        For a CNC-only daily simulator the session book and the demat book hold
        the same rows, so this and :meth:`get_holdings` agree on quantity; the
        holdings view adds the unsettled T+1 split.
        """
        return tuple(
            Position(
                symbol=symbol,
                quantity=qty,
                average_price=self._average_basis(symbol),
                last_price=self._marks.get(symbol, 0.0),
                product="CNC",
                exchange=self._cfg.exchange,
            )
            for symbol, qty in sorted(self._shares.items())
            if qty != 0
        )

    def get_holdings(self) -> tuple[Holding, ...]:
        return tuple(
            Holding(
                symbol=symbol,
                quantity=qty,
                average_price=self._average_basis(symbol),
                last_price=self._marks.get(symbol, 0.0),
                t1_quantity=self._bought_today.get(symbol, 0),
                exchange=self._cfg.exchange,
            )
            for symbol, qty in sorted(self._shares.items())
            if qty != 0
        )

    def get_margins(self) -> Margins:
        return Margins(
            available_cash=self._cash,
            unsettled_cash=self.unsettled_cash,
            utilised_debits=float(
                sum(self._average_basis(s) * q for s, q in self._shares.items() if q > 0)
            ),
            opening_balance=float(self._cfg.initial_cash),
        )

    # ── target weights ───────────────────────────────────────────────────────

    def submit_target_weights(
        self, weights: Mapping[str, float], *, ts: datetime | None = None
    ) -> None:
        """Register the portfolio the strategy wants, to be traded next open.

        Weights are *equity* weights of NAV; whatever is left over is cash.
        Leave a sliver of it: a basket summing to exactly 1.0 asks to spend
        every rupee on stock and leaves nothing for the ~12 bps of statutory
        charges, so its last buys are rejected for want of funds.
        Each is clipped to ``max_weight_per_name`` — clipped, not redistributed:
        deciding where the excess goes is a policy choice and the broker is not
        entitled to make it.  A basket summing to more than 1 is a caller bug
        (it would need leverage this account does not have) and raises.
        """
        cleaned: dict[str, float] = {}
        total = 0.0
        for symbol, raw in weights.items():
            w = float(raw)
            if not isfinite(w):
                raise ValueError(f"weight for {symbol} is not finite: {raw!r}")
            if w < 0.0:
                raise ValueError(
                    f"weight for {symbol} is negative ({w}); this account is long-only"
                )
            w = min(w, self._cfg.max_weight_per_name)
            total += w
            if w > 0.0:
                cleaned[symbol] = w
        if total > 1.0 + 1e-9:
            raise ValueError(
                f"target weights sum to {total:.6f} after capping; the account "
                "has no leverage"
            )
        self._targets = cleaned
        if ts is not None:
            self._target_ts = ts
        elif self._last_marked_day is not None:
            # The decision was made on the previous close, which is the
            # timestamp a live run would stamp on the order.
            self._target_ts = _close_ts(self._last_marked_day)

    # ── the trading session ──────────────────────────────────────────────────

    def open_session(
        self,
        day: date,
        opens: Mapping[str, float],
        *,
        impact: Mapping[str, ImpactInputs] | None = None,
    ) -> SessionResult:
        """Settle matured proceeds, then fill everything queued at ``opens``.

        Sells are executed before buys, and buys within the session in symbol
        order.  The ordering is fixed only so that two runs of the same inputs
        produce the same book — the simulator must not decide which buy
        survives a cash shortfall, because that would be an allocation choice
        the strategy did not make.  Note what selling first does *not* do:
        today's proceeds land in the settlement pipeline, so they cannot fund
        the buys that follow them in the very same session.
        """
        self._require_run()
        if self._day is not None and day <= self._day:
            raise BrokerError(
                f"sessions must advance: {day} is not after {self._day}"
            )
        self._day = day
        self._session_index += 1
        self._sold_today = set()
        self._bought_today = {}
        self._session_fees = 0.0
        self._session_turnover = 0.0

        cash_settled = self._release_settled()
        if self._targets is not None:
            self._materialise_targets(opens)
            self._targets = None
            self._target_ts = None

        fills: list[Fill] = []
        rejected: list[BrokerOrder] = []
        # Sells first, then buys in symbol order — deterministic, and
        # deliberately not a preference for which buy gets the scarce cash.
        queue = sorted(
            self._queue,
            key=lambda oid: (self._orders[oid].request.side != "SELL", oid),
        )
        self._queue = []
        for order_id in queue:
            working = self._orders[order_id]
            if working.status != "PENDING":
                continue
            fill = self._execute(working, opens, impact)
            if fill is not None:
                fills.append(fill)
            elif working.status in ("REJECTED", "CANCELLED"):
                rejected.append(working.snapshot())

        self._recorder.flush()
        self._assert_identity()
        return SessionResult(
            day=day,
            fills=tuple(fills),
            rejected=tuple(rejected),
            cash_settled=cash_settled,
            fees=self._session_fees,
            turnover=self._session_turnover,
        )

    def mark_to_market(self, day: date, closes: Mapping[str, float]) -> DaySnapshot:
        """Mark the book at ``day``'s close and write the daily ledger rows.

        A symbol missing from ``closes`` keeps its previous mark rather than
        going to zero: a data gap is not a 100% loss, and marking it as one
        would put a fictitious crater in the NAV series.
        """
        run_id = self._require_run()
        if self._day != day:
            raise BrokerError(
                f"mark_to_market({day}) does not match the open session {self._day}"
            )
        if self._last_marked_day == day:
            raise BrokerError(f"{day} has already been marked")

        for symbol, price in closes.items():
            if price > 0.0:
                self._marks[symbol] = float(price)

        ts = _close_ts(day)
        equity_value = self.equity_value
        unrealised = self._unrealised_pnl()
        nav = self._cash + self.unsettled_cash + equity_value

        snapshot = DaySnapshot(
            day=day,
            settled_cash=self._cash,
            unsettled_cash=self.unsettled_cash,
            equity_value=equity_value,
            nav=nav,
            realised_pnl=self._realised_pnl,
            unrealised_pnl=unrealised,
            fees_paid=self._fees_paid,
            turnover=self._session_turnover,
        )
        self._prev_nav = self._nav
        self._nav = nav
        self._last_marked_day = day
        self._snapshots.append(snapshot)

        for symbol, qty in self._shares.items():
            self._recorder.upsert_position(
                run_id,
                symbol,
                qty,
                self._average_basis(symbol),
                self._marks.get(symbol),
                ts,
            )
        self._recorder.record_snapshot(
            run_id,
            snapshot,
            ts,
            {
                "settled_cash": round(self._cash, 2),
                "unsettled_cash": round(self.unsettled_cash, 2),
                "n_positions": sum(1 for q in self._shares.values() if q > 0),
                "session_index": self._session_index,
                # The ledger has no tax table, and the liability is a property
                # of the financial year rather than of any row in it — so the
                # accrual rides along here, where `scripts/ledger.py` can read
                # it back without re-deriving every lot match.
                "tax_accrued": round(self._tax.total_tax(), 2),
                "tax_by_fy": {
                    financial_year_label(liability.financial_year): {
                        "short_term_gain": round(liability.short_term_gain, 2),
                        "long_term_gain": round(liability.long_term_gain, 2),
                        "tax": round(liability.total, 2),
                    }
                    for liability in self._tax.liabilities()
                },
            },
        )
        # Daily P&L decomposes the NAV change exactly:
        #   Δnav = Δrealised + Δunrealised - Δfees
        # which is the same identity _assert_identity checks at the level.
        self._recorder.record_pnl(
            run_id,
            day,
            self._realised_pnl - self._prev_realised,
            unrealised - self._prev_unrealised,
            self._fees_paid - self._prev_fees,
            nav - self._prev_nav,
        )
        self._prev_realised = self._realised_pnl
        self._prev_unrealised = unrealised
        self._prev_fees = self._fees_paid
        self._recorder.flush()
        self._assert_identity()
        return snapshot

    # ── execution internals ──────────────────────────────────────────────────

    def _materialise_targets(self, opens: Mapping[str, float]) -> None:
        """Turn target weights into integer share deltas at today's open.

        Deliberately identical to ``PanelTradingEnv.step``: NAV is the previous
        close's mark, the target rupee value is ``nav * weight``, and the share
        count is floored so no fractional share is ever created.  Deltas worth
        less than ``min_trade_value`` are dropped rather than rejected — they
        are dust the strategy did not really ask for, and filling the ledger
        with a hundred rejected orders a day would bury the real ones.
        """
        assert self._targets is not None
        nav = self._nav
        symbols = sorted(set(self._targets) | set(self._shares))
        for symbol in symbols:
            open_px = float(opens.get(symbol, 0.0))
            if open_px <= 0.0:
                continue
            weight = self._targets.get(symbol, 0.0)
            target_shares = int(floor(nav * weight / open_px))
            delta = target_shares - self._shares.get(symbol, 0)
            if delta == 0:
                continue
            if abs(delta) * open_px < self._cfg.min_trade_value:
                continue
            side: Side = "BUY" if delta > 0 else "SELL"
            self.place_order(
                OrderRequest(
                    symbol=symbol,
                    side=side,
                    quantity=abs(delta),
                    product="CNC",
                    order_type="MARKET",
                    exchange=self._cfg.exchange,
                    tag="target",
                )
            )

    def _execute(
        self,
        working: _WorkingOrder,
        opens: Mapping[str, float],
        impact: Mapping[str, ImpactInputs] | None,
    ) -> Fill | None:
        request = working.request
        symbol = request.symbol
        assert self._day is not None
        ts = _open_ts(self._day)
        working.ts_updated = ts

        reference = float(opens.get(symbol, 0.0))
        if reference <= 0.0:
            self._reject(working, REJECT_NO_PRICE)
            return None

        self._guard_intraday(symbol, request.side)

        slip = self._slippage_fraction(symbol, request.side, request.quantity, impact)
        price = reference * (1.0 + slip)
        if request.limit_price is not None:
            limit = request.limit_price
            marketable = reference <= limit if request.side == "BUY" else reference >= limit
            if not marketable:
                # A daily bar cannot say whether the order would have filled
                # later in the session, and assuming it did is look-ahead.
                self._cancel_unfilled(working, CANCEL_LIMIT_NOT_MARKETABLE)
                return None
            price = min(price, limit) if request.side == "BUY" else max(price, limit)

        if request.side == "BUY":
            return self._execute_buy(working, price, slip, ts)
        return self._execute_sell(working, price, slip, ts)

    def _execute_buy(
        self, working: _WorkingOrder, price: float, slip: float, ts: datetime
    ) -> Fill | None:
        request = working.request
        value = request.quantity * price
        cost = self._cost_model.cost(value, True)
        if value + cost > self._cash + 1e-9:
            # No partial fills: a real broker rejects the order outright on a
            # margin shortfall rather than buying what it can afford, and
            # scaling the order down here would be the simulator making an
            # allocation decision the strategy did not make.
            self._reject(working, REJECT_INSUFFICIENT_FUNDS)
            return None

        self._cash -= value + cost
        self._shares[request.symbol] = self._shares.get(request.symbol, 0) + request.quantity
        self._bought_today[request.symbol] = (
            self._bought_today.get(request.symbol, 0) + request.quantity
        )
        self._marks.setdefault(request.symbol, price)

        # Cost basis is the gross fill price.  STT is expressly not deductible
        # under the proviso to s.48, and the deductible remainder (exchange,
        # SEBI, GST, stamp — brokerage is nil on delivery) is ~2 bps here.
        # Excluding all of it overstates the taxable gain very slightly, which
        # is the conservative direction, and keeps the accounting identity in
        # _assert_identity exact.
        assert self._day is not None
        self._lots.buy(
            request.symbol,
            Lot(
                buy_date=self._day,
                quantity=float(request.quantity),
                cost_basis_per_share=price,
            ),
        )
        row_id = self._recorder.record_lot(
            self._require_run(), request.symbol, self._day, request.quantity, price
        )
        self._lot_rows.setdefault(request.symbol, []).append(row_id)
        return self._book_fill(working, price, cost, slip, ts, value)

    def _execute_sell(
        self, working: _WorkingOrder, price: float, slip: float, ts: datetime
    ) -> Fill | None:
        request = working.request
        symbol = request.symbol
        held = self._shares.get(symbol, 0)
        if request.quantity > held:
            # Long-only: there is no borrow in this account.
            self._reject(working, REJECT_INSUFFICIENT_HOLDING)
            return None

        value = request.quantity * price
        # The depository charges once per scrip per day however many orders it
        # took, so only the session's first sell of this symbol carries the DP
        # fee.  Getting this per-order instead would overcharge a position
        # exited in three clips by ₹30.68.
        n_scrips = 0 if symbol in self._sold_today else 1
        cost = self._cost_model.cost(value, False, n_scrips_sold=n_scrips)
        self._sold_today.add(symbol)

        self._shares[symbol] = held - request.quantity
        assert self._day is not None
        gain = self._lots.sell(symbol, float(request.quantity), self._day, price)
        self._consume_lot_rows(symbol, request.quantity)
        self._realised_pnl += gain.total
        self._tax.record(gain)

        # The contract note pays out net of charges, and the payout is what
        # enters the T+1 pipeline — so charges never push settled cash negative.
        self._credit_settlement(value - cost)
        return self._book_fill(working, price, cost, slip, ts, value, gain=gain)

    def _book_fill(
        self,
        working: _WorkingOrder,
        price: float,
        cost: float,
        slip: float,
        ts: datetime,
        value: float,
        *,
        gain: RealisedGain | None = None,
    ) -> Fill:
        working.status = "FILLED"
        working.filled_quantity = working.request.quantity
        working.average_price = price
        working.ts_updated = ts
        self._fees_paid += cost
        self._session_fees += cost
        self._session_turnover += value
        fill = Fill(
            order_id=working.order_id,
            symbol=working.request.symbol,
            side=working.request.side,
            quantity=working.request.quantity,
            price=price,
            ts=ts,
            fees=cost,
            slippage=slip,
        )
        self._recorder.update_order(working.row_id, working.snapshot())
        self._recorder.record_fill(working.row_id, fill)
        return fill

    def _reject(self, working: _WorkingOrder, reason: str) -> None:
        working.status = "REJECTED"
        working.reject_reason = reason
        self._recorder.update_order(working.row_id, working.snapshot())
        return None

    def _cancel_unfilled(self, working: _WorkingOrder, reason: str) -> None:
        working.status = "CANCELLED"
        working.reject_reason = reason
        self._recorder.update_order(working.row_id, working.snapshot())
        return None

    def _submit_ts(self) -> datetime:
        """When a live run would have stamped this order as submitted.

        Preference order matters for reproducibility: an explicit timestamp,
        else the close at which the strategy last saw the market, else this
        session's open.  ``now()`` is only reached by an order placed before
        the first session has opened, where there is genuinely nothing else
        to say.
        """
        if self._target_ts is not None:
            return self._target_ts
        if self._last_marked_day is not None:
            return _close_ts(self._last_marked_day)
        if self._day is not None:
            return _open_ts(self._day)
        return datetime.now(tz=IST)

    def _guard_intraday(self, symbol: str, side: Side) -> None:
        bought = symbol in self._bought_today
        sold = symbol in self._sold_today
        if (side == "SELL" and bought) or (side == "BUY" and sold):
            raise IntradayRoundTripError(
                f"{symbol} would be bought and sold in the same session "
                f"({self._day}); a same-day round trip is taxed and charged at "
                "intraday rates, not the CNC delivery rates this broker prices "
                "with, and a next-open daily simulator cannot produce one. "
                "Check the caller's order generation."
            )

    def _slippage_fraction(
        self,
        symbol: str,
        side: Side,
        quantity: int,
        impact: Mapping[str, ImpactInputs] | None,
    ) -> float:
        """Signed fraction added to the reference price; buys pay up."""
        sign = 1.0 if side == "BUY" else -1.0
        model = self._cfg.slippage_model
        if model == "none":
            return 0.0
        if model == "percentage":
            return sign * self._cfg.slippage_pct
        inputs = None if impact is None else impact.get(symbol)
        if inputs is None:
            # No impact inputs for this name — charge nothing rather than
            # guess, and let the caller notice the missing data.
            return 0.0
        adv = max(inputs.adv_shares, 1.0)
        return sign * SLIPPAGE_K * inputs.atr_fraction * float((quantity / adv) ** 0.5)

    # ── settlement ───────────────────────────────────────────────────────────

    def _credit_settlement(self, amount: float) -> None:
        if self._cfg.settlement_days <= 0:
            self._cash += amount
            return
        self._pending.append((self._session_index + self._cfg.settlement_days, amount))

    def _release_settled(self) -> float:
        """Move everything due into spendable cash; return how much moved."""
        if not self._pending:
            return 0.0
        due = sum(a for i, a in self._pending if i <= self._session_index)
        self._pending = [(i, a) for i, a in self._pending if i > self._session_index]
        self._cash += due
        return due

    @property
    def unsettled_cash(self) -> float:
        return float(sum(a for _, a in self._pending))

    # ── lots, marks and the invariant ────────────────────────────────────────

    def _consume_lot_rows(self, symbol: str, quantity: int) -> None:
        """Write the FIFO consumption back to ``ledger.lots``.

        The matching itself belongs to :class:`FifoLotBook` and is not redone
        here: the surviving lots it reports, compared against the row ids kept
        in the same order, say exactly which rows closed and how much is left
        in the new oldest one.
        """
        rows = self._lot_rows.get(symbol)
        if not rows:
            return
        surviving = self._lots.open_lots(symbol)
        n_closed = len(rows) - len(surviving)
        for row_id in rows[:n_closed]:
            self._recorder.update_lot(row_id, 0)
        del rows[:n_closed]
        if surviving and rows:
            self._recorder.update_lot(rows[0], int(round(surviving[0].quantity)))
        if not rows:
            self._lot_rows.pop(symbol, None)

    def _average_basis(self, symbol: str) -> float:
        lots = self._lots.open_lots(symbol)
        qty = sum(lot.quantity for lot in lots)
        if qty <= 0.0:
            return 0.0
        return sum(lot.cost for lot in lots) / qty

    def _unrealised_pnl(self) -> float:
        total = 0.0
        for symbol, qty in self._shares.items():
            if qty <= 0:
                continue
            mark = self._marks.get(symbol)
            if mark is None:
                continue
            total += sum(
                lot.quantity * (mark - lot.cost_basis_per_share)
                for lot in self._lots.open_lots(symbol)
            )
        return total

    def _assert_identity(self) -> None:
        """Two views of NAV must agree, or the run has lost money it cannot name.

        ``cash + equity`` is what the account is worth; ``initial + realised +
        unrealised - fees`` is what happened to it.  They are equal because a
        lot's basis is the gross fill price and every charge is expensed to
        cash, so no rupee is counted twice or dropped.  See the module
        docstring; the derivation is one line of algebra over the buy/sell sums.
        """
        nav = self._cash + self.unsettled_cash + self.equity_value
        explained = (
            float(self._cfg.initial_cash)
            + self._realised_pnl
            + self._unrealised_pnl()
            - self._fees_paid
        )
        if abs(nav - explained) > _IDENTITY_RTOL * max(1.0, abs(nav)):
            raise AccountingError(
                f"NAV identity broken on {self._day}: cash+equity={nav:.6f} but "
                f"initial+realised+unrealised-fees={explained:.6f} "
                f"(drift {nav - explained:+.6f})"
            )
        if self._cash < -1e-9:
            raise AccountingError(f"settled cash went negative: {self._cash:.6f}")

    # ── read-only views ──────────────────────────────────────────────────────

    @property
    def config(self) -> PaperBrokerConfig:
        return self._cfg

    @property
    def cash(self) -> float:
        """Settled cash — what a buy order can consume right now."""
        return self._cash

    @property
    def equity_value(self) -> float:
        return sum(q * self._marks.get(s, 0.0) for s, q in self._shares.items())

    @property
    def nav(self) -> float:
        return self._cash + self.unsettled_cash + self.equity_value

    @property
    def realised_pnl(self) -> float:
        return self._realised_pnl

    @property
    def unrealised_pnl(self) -> float:
        return self._unrealised_pnl()

    @property
    def fees_paid(self) -> float:
        return self._fees_paid

    @property
    def snapshots(self) -> tuple[DaySnapshot, ...]:
        return tuple(self._snapshots)

    @property
    def tax(self) -> TaxModel:
        return self._tax

    def tax_liabilities(self) -> tuple[TaxLiability, ...]:
        """Accrued capital-gains tax per financial year — reported, not deducted."""
        return self._tax.liabilities()

    def open_lots(self, symbol: str) -> tuple[Lot, ...]:
        return self._lots.open_lots(symbol)

    def nav_series(self) -> tuple[tuple[date, float], ...]:
        return tuple((s.day, s.nav) for s in self._snapshots)

    def daily_log_returns(self) -> list[float]:
        """Daily log NAV returns — the input QuantStats wants."""
        import math

        navs = [s.nav for s in self._snapshots]
        if len(navs) < 2:
            return []
        return [
            math.log(max(b, 1e-12) / max(a, 1e-12))
            for a, b in zip(navs[:-1], navs[1:], strict=True)
        ]


# ── module helpers ────────────────────────────────────────────────────────────


def _open_ts(day: date) -> datetime:
    return datetime.combine(day, MARKET_OPEN, tzinfo=IST)


def _close_ts(day: date) -> datetime:
    return datetime.combine(day, MARKET_CLOSE, tzinfo=IST)


def weights_from_logits(
    logits: Sequence[float] | Any,
    symbols: Sequence[str],
    tradeable: Iterable[bool] | Any,
    max_weight: float = DEFAULT_MAX_WEIGHT,
) -> dict[str, float]:
    """Map a policy's ``(N+1,)`` logit vector to per-symbol target weights.

    Reuses the environment's :func:`~trader.env.panel_env.masked_softmax` so
    that a paper run and a backtest turn the same action into the same
    portfolio.  Index 0 is cash and is dropped: the broker infers cash as the
    residual.
    """
    import numpy as np

    from trader.env.panel_env import masked_softmax

    mask = np.asarray(list(tradeable), dtype=bool)
    weights = masked_softmax(np.asarray(logits, dtype=np.float64), mask, max_weight)
    return {
        symbol: float(w)
        for symbol, w in zip(symbols, weights[1:], strict=True)
        if w > 0.0
    }
