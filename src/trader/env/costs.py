"""Transaction cost models for the trading environment."""
from __future__ import annotations

from abc import ABC, abstractmethod

# Zerodha delivery equity fee schedule (as of 2024)
_BROKERAGE_RATE = 0.0003
_BROKERAGE_CAP = 20.0        # INR per order
_STT_RATE = 0.001            # sell side only
_EXCHANGE_CHARGE = 0.0000322 # NSE
_GST_RATE = 0.18             # on brokerage + exchange charge
_SEBI_RATE = 0.000001
_STAMP_RATE = 0.00015        # buy side only
_DP_CHARGE = 15.93           # per scrip per sell day (CDSL/NSDL)


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


class ZerodhaEquityDeliveryCostModel(CostModel):
    """Zerodha equity delivery cost model for Indian cash equities."""

    def cost(
        self,
        trade_value: float,
        is_buy: bool,
        *,
        n_scrips_sold: int = 0,
    ) -> float:
        if trade_value <= 0.0:
            return 0.0

        brokerage = min(_BROKERAGE_CAP, _BROKERAGE_RATE * trade_value)
        exchange = _EXCHANGE_CHARGE * trade_value
        gst = _GST_RATE * (brokerage + exchange)
        sebi = _SEBI_RATE * trade_value

        stt = _STT_RATE * trade_value if not is_buy else 0.0
        stamp = _STAMP_RATE * trade_value if is_buy else 0.0
        dp = _DP_CHARGE * n_scrips_sold if not is_buy else 0.0

        return brokerage + exchange + gst + sebi + stt + stamp + dp


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
