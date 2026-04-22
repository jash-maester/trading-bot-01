from __future__ import annotations

from datetime import datetime
from typing import Literal, Protocol, runtime_checkable

import polars as pl


@runtime_checkable
class MarketDataSource(Protocol):
    def fetch_ohlcv(
        self,
        tickers: list[str],
        start: datetime,
        end: datetime,
        interval: Literal["1d", "1h", "15m"] = "1d",
    ) -> pl.DataFrame: ...

    def fetch_corporate_actions(self, tickers: list[str]) -> pl.DataFrame: ...
