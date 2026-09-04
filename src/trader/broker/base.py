"""Broker interface, deliberately shaped like Zerodha Kite Connect.

Only a paper simulator implements this today, so the obvious question is why
an interface exists at all.  The answer is that the day this system goes live,
the *only* thing that should change is which object the runner constructs.  If
the simulator grows an API of its own invention, that swap stops being a
one-line change and becomes a rewrite of every call site — and the differences
between simulated and real surface as live-money bugs instead of test failures.

So the vocabulary here is Kite's, not ours::

    place_order / cancel_order / get_orders / get_positions / get_holdings /
    get_margins

and so are the enum values: ``BUY``/``SELL``, ``CNC``/``MIS``,
``MARKET``/``LIMIT``, ``NSE``/``BSE``.  Three places deliberately diverge:

* ``place_order`` takes an :class:`OrderRequest` instead of Kite's fifteen
  loose keyword arguments.  The flat signature cannot be type-checked (every
  argument is ``str``), and half of it is meaningless for cash equity delivery.
  A Kite adapter unpacks the dataclass at the boundary; nothing is lost.
* :data:`OrderStatus` uses the ledger's five values rather than Kite's dozen,
  because ``ledger.orders.status`` is check-constrained to exactly those.  A
  Kite adapter maps ``COMPLETE``→``FILLED``, ``OPEN``/``TRIGGER PENDING``→
  ``PENDING``, ``CANCELLED``→``CANCELLED``, ``REJECTED``→``REJECTED``.
* Money is ``float``.  The ledger stores ``Numeric`` and Kite returns strings;
  both are converted at their own boundary.  Rupee arithmetic in this system
  runs through NumPy in the environment, and a float/Decimal split down the
  middle of the accounting would be worse than a consistent float.

The interface is intentionally small.  Kite exposes modify_order, GTTs, MF
orders, historical data and a websocket; none of them are implemented here and
none are declared, because a declared-but-unimplemented method is a live
trading incident waiting to happen.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

Side = Literal["BUY", "SELL"]
Product = Literal["CNC", "MIS"]
OrderType = Literal["MARKET", "LIMIT"]
Exchange = Literal["NSE", "BSE"]

# The ledger's vocabulary — see the module docstring on why it is not Kite's.
OrderStatus = Literal["PENDING", "FILLED", "PARTIAL", "CANCELLED", "REJECTED"]

# Statuses from which an order can no longer change.
TERMINAL_STATUSES: frozenset[str] = frozenset({"FILLED", "CANCELLED", "REJECTED"})


class BrokerError(Exception):
    """Base class for every error raised by a :class:`Broker` implementation."""


class OrderNotFound(BrokerError):
    """The order id is unknown to this broker."""


class InvalidOrderState(BrokerError):
    """The order exists but the requested transition is not legal for it.

    Cancelling an order that has already filled is the motivating case: Kite
    returns an error rather than silently unwinding the trade, and so do we.
    """


@dataclass(frozen=True, slots=True)
class OrderRequest:
    """One order to place.  Validated at construction, not at execution.

    ``quantity`` is an ``int`` and is type-checked at runtime as well as by
    mypy: Indian cash equity has no fractional shares, and a float quantity
    that silently truncates somewhere downstream is the kind of bug that only
    shows up as a rupee or two of unexplained cash drift per fill.
    """

    symbol: str
    side: Side
    quantity: int
    product: Product = "CNC"
    order_type: OrderType = "MARKET"
    limit_price: float | None = None
    exchange: Exchange = "NSE"
    tag: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.quantity, int) or isinstance(self.quantity, bool):
            raise TypeError(
                f"quantity must be an int (no fractional shares), got "
                f"{type(self.quantity).__name__}"
            )
        if self.quantity <= 0:
            raise ValueError(f"quantity must be positive, got {self.quantity}")
        if self.order_type == "LIMIT" and self.limit_price is None:
            raise ValueError("a LIMIT order needs a limit_price")
        if self.order_type == "MARKET" and self.limit_price is not None:
            raise ValueError("a MARKET order must not carry a limit_price")
        if self.limit_price is not None and self.limit_price <= 0.0:
            raise ValueError(f"limit_price must be positive, got {self.limit_price}")


@dataclass(frozen=True, slots=True)
class BrokerOrder:
    """An order as the broker currently sees it — Kite's order book row."""

    order_id: str
    symbol: str
    side: Side
    quantity: int
    product: Product
    order_type: OrderType
    status: OrderStatus
    ts_submitted: datetime
    ts_updated: datetime
    filled_quantity: int = 0
    average_price: float = 0.0
    limit_price: float | None = None
    exchange: Exchange = "NSE"
    tag: str | None = None
    reject_reason: str | None = None

    @property
    def is_open(self) -> bool:
        return self.status not in TERMINAL_STATUSES


@dataclass(frozen=True, slots=True)
class Fill:
    """One execution against an order.

    ``fees`` is the whole Indian charge stack for this leg (see
    :mod:`trader.env.costs`), not brokerage alone.  ``slippage`` is the signed
    *fraction* by which the fill price differed from the reference price, which
    is what ``ledger.fills.slippage`` stores — keeping it dimensionless makes
    fills comparable across a ₹100 stock and a ₹4,000 one.
    """

    order_id: str
    symbol: str
    side: Side
    quantity: int
    price: float
    ts: datetime
    fees: float = 0.0
    slippage: float = 0.0

    @property
    def value(self) -> float:
        """Gross turnover of this fill, before charges."""
        return self.quantity * self.price


@dataclass(frozen=True, slots=True)
class Position:
    """A net position for the current trading session (Kite's ``day`` book).

    Distinct from :class:`Holding`: a position is what the *session* did, a
    holding is what sits in the demat account.  For a CNC-only daily strategy
    the two converge, but keeping them apart is what lets an MIS adapter drop
    in later without redefining either.
    """

    symbol: str
    quantity: int
    average_price: float
    last_price: float
    product: Product = "CNC"
    exchange: Exchange = "NSE"

    @property
    def value(self) -> float:
        return self.quantity * self.last_price

    @property
    def pnl(self) -> float:
        """Unrealised P&L at ``last_price``, excluding charges."""
        return self.quantity * (self.last_price - self.average_price)


@dataclass(frozen=True, slots=True)
class Holding:
    """A demat holding.  ``t1_quantity`` is the part that has not settled yet.

    Indian equity settles T+1, so shares bought in the current session are
    legally yours but are not in the demat account until the next one.  Kite
    reports that split and so do we, because it is the same mechanism that
    makes sale proceeds unusable on the day of the sale.
    """

    symbol: str
    quantity: int
    average_price: float
    last_price: float
    t1_quantity: int = 0
    exchange: Exchange = "NSE"

    @property
    def value(self) -> float:
        return self.quantity * self.last_price

    @property
    def pnl(self) -> float:
        return self.quantity * (self.last_price - self.average_price)


@dataclass(frozen=True, slots=True)
class Margins:
    """The funds view for the equity segment.

    ``available_cash`` is what a buy order can actually consume right now;
    ``unsettled_cash`` is money that is already ours but is still in the
    settlement pipeline and therefore cannot fund a purchase today.  Kite
    reports the same distinction as ``available.live_balance`` versus the
    payout awaiting settlement.
    """

    available_cash: float
    unsettled_cash: float
    utilised_debits: float
    opening_balance: float

    @property
    def total_cash(self) -> float:
        """Settled plus unsettled — the cash leg of NAV."""
        return self.available_cash + self.unsettled_cash


class Broker(ABC):
    """The six calls a strategy runner is allowed to make on a broker."""

    @abstractmethod
    def place_order(self, request: OrderRequest) -> str:
        """Submit ``request`` and return the broker's order id.

        Raises :class:`~trader.broker.base.BrokerError` on a rejection the
        broker can detect at submission time (an unknown symbol, say).  A
        rejection that only becomes knowable at execution — insufficient funds
        against the fill price — comes back as a ``REJECTED`` order in
        :meth:`get_orders` instead, which is how Kite behaves too.
        """

    @abstractmethod
    def cancel_order(self, order_id: str) -> None:
        """Cancel a still-open order.

        Raises :class:`OrderNotFound` for an unknown id and
        :class:`InvalidOrderState` for one that has already reached a terminal
        status.
        """

    @abstractmethod
    def get_orders(self) -> tuple[BrokerOrder, ...]:
        """The full order book for the current run, oldest first."""

    @abstractmethod
    def get_positions(self) -> tuple[Position, ...]:
        """Net positions for the current session, one row per symbol."""

    @abstractmethod
    def get_holdings(self) -> tuple[Holding, ...]:
        """Demat holdings, one row per symbol with a non-zero quantity."""

    @abstractmethod
    def get_margins(self) -> Margins:
        """The equity-segment funds view."""
