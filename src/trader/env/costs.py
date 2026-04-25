"""Transaction cost models for the trading environment."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

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

    def cost_vec(
        self,
        trade_values: np.ndarray,
        is_buy: np.ndarray,
        n_scrips_sold: np.ndarray,
    ) -> np.ndarray:
        """Closed-form vectorised cost. Avoids per-name Python overhead."""
        tv = trade_values.astype(np.float64, copy=False)
        positive = tv > 0.0

        brokerage = np.minimum(_BROKERAGE_CAP, _BROKERAGE_RATE * tv)
        exchange = _EXCHANGE_CHARGE * tv
        gst = _GST_RATE * (brokerage + exchange)
        sebi = _SEBI_RATE * tv

        sell_mask = (~is_buy.astype(bool)) & positive
        buy_mask = is_buy.astype(bool) & positive

        stt = np.where(sell_mask, _STT_RATE * tv, 0.0)
        stamp = np.where(buy_mask, _STAMP_RATE * tv, 0.0)
        dp = np.where(sell_mask, _DP_CHARGE * n_scrips_sold.astype(np.float64), 0.0)

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
