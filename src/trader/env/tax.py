"""Indian capital-gains tax on listed equity (post Budget 2024).

A backtest that ignores tax overstates a high-turnover strategy badly: every
sale inside twelve months is taxed at 20% under section 111A, whereas the same
position held past the twelve-month line is taxed at 12.5% under section 112A
*and* enjoys a ₹1,25,000 per-financial-year exemption.  The gap is large enough
to flip the ranking of two strategies with similar gross returns, so the
holding period has to be tracked per purchase tranche rather than inferred from
a single ``avg_price``.

That is why this module is built around FIFO lots.  Indian tax law matches
sales against purchases first-in-first-out for listed equity held in demat, so
:class:`FifoLotBook` consumes the oldest open lot first and splits the realised
gain into a short-term and a long-term bucket.  :class:`TaxModel` then
accumulates those buckets per financial year (1 April – 31 March) and applies
the rates, the LTCG exemption and the 4% health & education cess.

Deliberate omissions, all of which depend on facts this module cannot see:

* **Surcharge** (10–15% on the tax, capped at 15% for 111A/112A gains) is a
  function of the taxpayer's *total* income slab, not of trading activity.
* **Loss carry-forward** (eight assessment years) and **cross-head set-off**
  (a short-term capital loss may be set off against long-term gains, but not
  the reverse) need a multi-year filing view.  Losses here net only against
  gains of the same bucket within the same financial year, and a net loss is
  floored at zero tax rather than carried forward.
* **Grandfathering** of pre-1-Feb-2018 acquisition costs under 112A — no
  backtest here starts that early.

All figures are rupees.  Nothing touches the database: the broker persists
lots in ``ledger.lots`` and rehydrates this book from them.
"""
from __future__ import annotations

import calendar
from collections import deque
from dataclasses import dataclass, replace
from datetime import date

# Section 111A — short-term capital gains on STT-paid listed equity.
STCG_RATE = 0.20
# Section 112A — long-term capital gains on STT-paid listed equity.
LTCG_RATE = 0.125
# Health & education cess, levied on the tax itself.
CESS_RATE = 0.04
# Section 112A exemption: the first ₹1,25,000 of long-term gains each FY.
LTCG_EXEMPTION_PER_FY = 125_000.0
# Listed equity turns long-term after *more than* twelve months of holding.
LONG_TERM_MONTHS = 12


def add_months(anchor: date, months: int) -> date:
    """``anchor`` shifted forward by whole calendar months, clamping the day.

    29 Feb + 12 months is 28 Feb, not 1 March: the holding-period test is a
    calendar-month test, so the day is clamped to the target month's length.
    """
    total = anchor.month - 1 + months
    year = anchor.year + total // 12
    month = total % 12 + 1
    day = min(anchor.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


def financial_year(day: date) -> int:
    """Return the starting calendar year of the Indian FY containing ``day``.

    FY 2025-26 runs 1 Apr 2025 – 31 Mar 2026 and is identified as ``2025``.
    """
    return day.year if day.month >= 4 else day.year - 1


def financial_year_label(fy: int) -> str:
    """``2025`` → ``"FY2025-26"`` — for logging and report tables."""
    return f"FY{fy}-{(fy + 1) % 100:02d}"


def is_long_term(buy_date: date, sell_date: date) -> bool:
    """True when the holding period exceeds twelve months.

    The statute says "held for more than twelve months", so a sale on the
    twelve-month anniversary itself is still short-term.
    """
    return sell_date > add_months(buy_date, LONG_TERM_MONTHS)


@dataclass(frozen=True, slots=True)
class Lot:
    """One open purchase tranche of a single scrip.

    ``quantity`` is the quantity still *open*.  Partial consumption produces a
    new ``Lot`` with a smaller quantity rather than mutating this one, which
    keeps a lot safely shareable with the caller that reported the gain.
    """

    buy_date: date
    quantity: float
    cost_basis_per_share: float

    def __post_init__(self) -> None:
        if self.quantity <= 0.0:
            raise ValueError(f"lot quantity must be positive, got {self.quantity}")
        if self.cost_basis_per_share < 0.0:
            raise ValueError(
                f"cost basis must be non-negative, got {self.cost_basis_per_share}"
            )

    @property
    def cost(self) -> float:
        return self.quantity * self.cost_basis_per_share


@dataclass(frozen=True, slots=True)
class RealisedGain:
    """The outcome of one sale, split by holding period.

    Both buckets are signed: a loss is a negative gain, and netting it against
    other gains in the same bucket is the caller's (``TaxModel``'s) job.
    """

    sell_date: date
    quantity: float
    proceeds: float
    cost: float
    short_term: float
    long_term: float

    @property
    def total(self) -> float:
        return self.short_term + self.long_term


class FifoLotBook:
    """Open lots per symbol, consumed oldest-first on a sale.

    Lots must be added in chronological order — which is how fills arrive from
    a broker and how ``ledger.lots`` is read back (ordered by ``buy_date``) —
    because FIFO here is insertion order, not a re-sort on every sale.
    """

    __slots__ = ("_lots",)

    def __init__(self) -> None:
        self._lots: dict[str, deque[Lot]] = {}

    def buy(self, symbol: str, lot: Lot) -> None:
        self._lots.setdefault(symbol, deque()).append(lot)

    def open_lots(self, symbol: str) -> tuple[Lot, ...]:
        return tuple(self._lots.get(symbol, ()))

    def open_quantity(self, symbol: str) -> float:
        return sum(lot.quantity for lot in self._lots.get(symbol, ()))

    def sell(
        self,
        symbol: str,
        quantity: float,
        sell_date: date,
        price_per_share: float,
    ) -> RealisedGain:
        """Consume ``quantity`` shares oldest-first and realise the gain.

        ``price_per_share`` should be net of transaction costs if the caller
        wants a cost-inclusive basis; this module does not know about fees.
        """
        if quantity <= 0.0:
            raise ValueError(f"sell quantity must be positive, got {quantity}")
        queue = self._lots.get(symbol)
        available = self.open_quantity(symbol)
        if queue is None or quantity > available + 1e-9:
            raise ValueError(
                f"cannot sell {quantity} of {symbol}: only {available} open"
            )

        remaining = quantity
        short_term = 0.0
        long_term = 0.0
        cost = 0.0
        # Float quantities can leave a sliver behind; treat anything under this
        # as fully consumed so the queue does not accumulate dust lots.
        eps = 1e-9

        while remaining > eps:
            lot = queue[0]
            taken = min(lot.quantity, remaining)
            lot_cost = taken * lot.cost_basis_per_share
            gain = taken * price_per_share - lot_cost
            if is_long_term(lot.buy_date, sell_date):
                long_term += gain
            else:
                short_term += gain
            cost += lot_cost
            remaining -= taken
            if lot.quantity - taken <= eps:
                queue.popleft()
            else:
                queue[0] = replace(lot, quantity=lot.quantity - taken)

        if not queue:
            del self._lots[symbol]

        return RealisedGain(
            sell_date=sell_date,
            quantity=quantity,
            proceeds=quantity * price_per_share,
            cost=cost,
            short_term=short_term,
            long_term=long_term,
        )


@dataclass(frozen=True, slots=True)
class TaxLiability:
    """Tax computed for one financial year."""

    financial_year: int
    short_term_gain: float  # net, signed
    long_term_gain: float  # net, signed
    taxable_long_term: float  # after the §112A exemption, floored at zero
    stcg_tax: float
    ltcg_tax: float
    cess: float
    total: float


class TaxModel:
    """Accumulates realised gains per financial year and prices the tax.

    Cheap enough to feed on every step: recording a sale is a dict lookup and
    two additions, and the tax itself is only computed when asked for.
    """

    __slots__ = ("_cess_rate", "_exemption", "_gains", "_ltcg_rate", "_stcg_rate")

    def __init__(
        self,
        *,
        stcg_rate: float = STCG_RATE,
        ltcg_rate: float = LTCG_RATE,
        cess_rate: float = CESS_RATE,
        ltcg_exemption: float = LTCG_EXEMPTION_PER_FY,
    ) -> None:
        self._stcg_rate = stcg_rate
        self._ltcg_rate = ltcg_rate
        self._cess_rate = cess_rate
        self._exemption = ltcg_exemption
        self._gains: dict[int, list[float]] = {}  # fy -> [short_term, long_term]

    def record(self, gain: RealisedGain) -> None:
        """Book a realised gain into the FY that contains its sale date."""
        bucket = self._gains.setdefault(financial_year(gain.sell_date), [0.0, 0.0])
        bucket[0] += gain.short_term
        bucket[1] += gain.long_term

    def record_gains(
        self, sell_date: date, *, short_term: float = 0.0, long_term: float = 0.0
    ) -> None:
        """Book raw bucket amounts — for callers that match lots themselves."""
        bucket = self._gains.setdefault(financial_year(sell_date), [0.0, 0.0])
        bucket[0] += short_term
        bucket[1] += long_term

    @property
    def financial_years(self) -> tuple[int, ...]:
        return tuple(sorted(self._gains))

    def realised(self, fy: int) -> tuple[float, float]:
        """Net (short-term, long-term) gain booked in ``fy``."""
        short_term, long_term = self._gains.get(fy, [0.0, 0.0])
        return short_term, long_term

    def liability(self, fy: int) -> TaxLiability:
        """Tax owed for one financial year.

        The exemption and the zero-floor are applied per year and per bucket:
        a long-term loss does not shelter short-term gains here (see the module
        docstring on cross-head set-off).
        """
        short_term, long_term = self.realised(fy)
        taxable_st = max(0.0, short_term)
        taxable_lt = max(0.0, max(0.0, long_term) - self._exemption)
        stcg_tax = self._stcg_rate * taxable_st
        ltcg_tax = self._ltcg_rate * taxable_lt
        cess = self._cess_rate * (stcg_tax + ltcg_tax)
        return TaxLiability(
            financial_year=fy,
            short_term_gain=short_term,
            long_term_gain=long_term,
            taxable_long_term=taxable_lt,
            stcg_tax=stcg_tax,
            ltcg_tax=ltcg_tax,
            cess=cess,
            total=stcg_tax + ltcg_tax + cess,
        )

    def liabilities(self) -> tuple[TaxLiability, ...]:
        return tuple(self.liability(fy) for fy in self.financial_years)

    def total_tax(self) -> float:
        """Tax across every financial year seen so far."""
        return sum(liability.total for liability in self.liabilities())
