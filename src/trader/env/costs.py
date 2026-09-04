"""Transaction cost models for the trading environment.

Indian equity trading costs are *not* a single bps number: they are a stack of
statutory levies (STT, stamp duty, SEBI turnover fee, GST) plus broker and
depository charges, each with its own base, its own side (buy / sell / both)
and its own product-dependent rate.  Getting them wrong biases every backtest
in the same direction, so the schedule below is transcribed literally from
zerodha.com/charges (verified September 2026) rather than approximated.

Two subtleties that are easy to get wrong and were wrong before:

1. **STT is charged on both legs of a delivery trade** (0.1% each way).  The
   earlier model charged the sell side only, understating round-trip cost by
   roughly ₹100 per ₹1L of turnover — about 40% of the true cost.
2. **SEBI turnover fees sit inside the GST base.**  GST is 18% of
   (brokerage + SEBI fee + exchange transaction charge), not of
   (brokerage + exchange) alone.

Delivery vs intraday is a *product* distinction, not a timing one, but the
tax authority treats a same-day buy-and-sell as intraday (speculative) even
when the order was tagged CNC.  The daily panel environment fills at the next
session's open and therefore can never round-trip within a day, so no attempt
is made to infer the product here: callers that can round-trip intraday (the
paper/live broker) pass ``intraday=True`` explicitly.

Demat AMC is deliberately *not* part of :meth:`CostModel.cost`.  It is an
annual account-level charge that depends on the value of the holdings, not on
any individual order, so charging it per trade would double-count it wildly.
See :func:`demat_amc_annual`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Final, Literal

import numpy as np

Exchange = Literal["NSE", "BSE"]

# Exchange transaction charge, as a fraction of turnover, charged on both legs.
_EXCHANGE_TXN_RATE: Final[dict[str, float]] = {
    "NSE": 0.0000307,  # 0.00307%
    "BSE": 0.0000375,  # 0.00375%
}

# SEBI turnover fee: ₹10 per crore = 0.0001% of turnover, both legs.
_SEBI_RATE: Final = 0.000001

# GST on (brokerage + SEBI fee + exchange transaction charge).
_GST_RATE: Final = 0.18

# Depository (CDSL) debit on every scrip sold, once per scrip per day:
# ₹3.50 CDSL + ₹9.50 Zerodha + ₹2.34 GST.
_DP_CHARGE: Final = 15.34

# Demat AMC — BSDA slabs, charged annually on the value of the holdings.
_BSDA_TIER1_LIMIT: Final = 400_000.0
_BSDA_TIER2_LIMIT: Final = 1_000_000.0
_BSDA_TIER2_FEE: Final = 100.0
_BSDA_TIER3_FEE: Final = 300.0


@dataclass(frozen=True, slots=True)
class _ProductSchedule:
    """The rates that differ between the delivery and intraday products."""

    brokerage_rate: float
    brokerage_cap: float  # INR per executed order
    stt_buy_rate: float
    stt_sell_rate: float
    stamp_rate: float  # buy side only
    dp_charge: float  # per distinct scrip per sell day


# Delivery (CNC): brokerage-free, STT both ways, stamp duty 0.015% on buys,
# and the depository charges a DP fee for moving stock out of the demat.
_DELIVERY: Final = _ProductSchedule(
    brokerage_rate=0.0,
    brokerage_cap=float("inf"),
    stt_buy_rate=0.001,
    stt_sell_rate=0.001,
    stamp_rate=0.00015,
    dp_charge=_DP_CHARGE,
)

# Intraday (MIS): 0.03% or ₹20 per order (whichever is lower), STT only on the
# sell leg at a quarter of the delivery rate, cheaper stamp duty, and no DP
# charge because nothing ever leaves the demat account.
_INTRADAY: Final = _ProductSchedule(
    brokerage_rate=0.0003,
    brokerage_cap=20.0,
    stt_buy_rate=0.0,
    stt_sell_rate=0.00025,
    stamp_rate=0.00003,
    dp_charge=0.0,
)


def demat_amc_annual(holdings_value: float) -> float:
    """Annual demat maintenance charge for a Basic Services Demat Account.

    BSDA slabs are assessed on the value of the holdings, not on turnover, so
    this is an account-level annual accrual and must never be folded into a
    per-order cost.  Only the top slab is quoted exclusive of GST on Zerodha's
    published schedule, so GST is added there and nowhere else.
    """
    if holdings_value < _BSDA_TIER1_LIMIT:
        return 0.0
    if holdings_value <= _BSDA_TIER2_LIMIT:
        return _BSDA_TIER2_FEE
    return _BSDA_TIER3_FEE * (1.0 + _GST_RATE)


class CostModel(ABC):
    @abstractmethod
    def cost(
        self,
        trade_value: float,
        is_buy: bool,
        *,
        n_scrips_sold: int = 0,
    ) -> float:
        """Return total cost in INR for a single order leg."""

    def cost_vec(
        self,
        trade_values: np.ndarray,   # [N] non-negative
        is_buy: np.ndarray,         # [N] bool
        n_scrips_sold: np.ndarray,  # [N] int — 1 per name being sold (0 for buys / no-trade)
    ) -> np.ndarray:
        """Vectorised cost over N order legs simultaneously.

        Default falls back to a Python loop calling :meth:`cost`; subclasses
        with closed-form formulas should override for ~10× speedup.
        """
        out = np.zeros_like(trade_values, dtype=np.float64)
        for i in range(trade_values.shape[0]):
            out[i] = self.cost(
                float(trade_values[i]),
                bool(is_buy[i]),
                n_scrips_sold=int(n_scrips_sold[i]),
            )
        return out


class ZerodhaEquityDeliveryCostModel(CostModel):
    """Zerodha cost model for Indian cash equities.

    Defaults to NSE equity *delivery*, which is what the daily panel
    environment trades.  ``exchange`` and ``intraday`` widen it to the other
    cells of the published schedule without changing any existing call site.
    """

    __slots__ = ("_exchange", "_exchange_rate", "_schedule")

    def __init__(self, *, exchange: Exchange = "NSE", intraday: bool = False) -> None:
        if exchange not in _EXCHANGE_TXN_RATE:
            raise ValueError(f"unknown exchange {exchange!r}; expected one of NSE, BSE")
        self._exchange = exchange
        self._exchange_rate = _EXCHANGE_TXN_RATE[exchange]
        self._schedule = _INTRADAY if intraday else _DELIVERY

    @property
    def exchange(self) -> Exchange:
        return self._exchange

    @property
    def intraday(self) -> bool:
        return self._schedule is _INTRADAY

    def _resolve(self, intraday: bool | None) -> _ProductSchedule:
        if intraday is None:
            return self._schedule
        return _INTRADAY if intraday else _DELIVERY

    def cost(
        self,
        trade_value: float,
        is_buy: bool,
        *,
        n_scrips_sold: int = 0,
        intraday: bool | None = None,
    ) -> float:
        """Total cost in INR for one order leg of ``trade_value`` turnover.

        ``n_scrips_sold`` is a count of *distinct scrips* leaving the demat
        account today, not a count of orders — the depository charges the DP
        fee once per scrip per day however many sell orders it took.
        """
        if trade_value <= 0.0:
            return 0.0
        sched = self._resolve(intraday)

        brokerage = min(sched.brokerage_cap, sched.brokerage_rate * trade_value)
        exchange = self._exchange_rate * trade_value
        sebi = _SEBI_RATE * trade_value
        # SEBI fees are inside the GST base alongside brokerage and exchange.
        gst = _GST_RATE * (brokerage + sebi + exchange)

        stt_rate = sched.stt_buy_rate if is_buy else sched.stt_sell_rate
        stt = stt_rate * trade_value
        stamp = sched.stamp_rate * trade_value if is_buy else 0.0
        dp = sched.dp_charge * n_scrips_sold if not is_buy else 0.0

        return brokerage + exchange + gst + sebi + stt + stamp + dp

    def cost_vec(
        self,
        trade_values: np.ndarray,
        is_buy: np.ndarray,
        n_scrips_sold: np.ndarray,
        *,
        intraday: bool | None = None,
    ) -> np.ndarray:
        """Closed-form vectorised cost over the ticker axis.

        This is on the environment's step() hot path — N≈163 names × 252 days ×
        16 parallel envs — so it must stay a handful of NumPy ops with no
        per-name Python iteration.  ``intraday`` applies to the whole batch;
        a mixed batch needs two calls (the daily env never has one).
        """
        tv = trade_values.astype(np.float64, copy=False)
        positive = tv > 0.0
        sched = self._resolve(intraday)

        brokerage = np.minimum(sched.brokerage_cap, sched.brokerage_rate * tv)
        exchange = self._exchange_rate * tv
        sebi = _SEBI_RATE * tv
        gst = _GST_RATE * (brokerage + sebi + exchange)

        buy_mask = is_buy.astype(bool)
        sell_mask = (~buy_mask) & positive
        buy_mask = buy_mask & positive

        stt = np.where(buy_mask, sched.stt_buy_rate * tv, 0.0) + np.where(
            sell_mask, sched.stt_sell_rate * tv, 0.0
        )
        stamp = np.where(buy_mask, sched.stamp_rate * tv, 0.0)
        dp = np.where(sell_mask, sched.dp_charge * n_scrips_sold.astype(np.float64), 0.0)

        out = brokerage + exchange + gst + sebi + stt + stamp + dp
        return np.asarray(np.where(positive, out, 0.0), dtype=np.float64)


class ZeroCostModel(CostModel):
    """No-cost model for sanity / ablation runs."""

    def cost(
        self,
        trade_value: float,
        is_buy: bool,
        *,
        n_scrips_sold: int = 0,
    ) -> float:
        return 0.0

    def cost_vec(
        self,
        trade_values: np.ndarray,
        is_buy: np.ndarray,
        n_scrips_sold: np.ndarray,
    ) -> np.ndarray:
        # Override the ABC's Python loop — ablation runs share the same hot path.
        return np.zeros(trade_values.shape, dtype=np.float64)
