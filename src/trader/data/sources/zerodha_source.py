from __future__ import annotations

from datetime import datetime
from typing import Literal

import polars as pl


class ZerodhaSource:
    """Zerodha Kite Connect data source — interface only, not yet implemented.

    TODO: Implement using kite-connect Python SDK historical data API.
    Reference: https://kite.trade/docs/pykiteconnect/v4/#historical-data
    Do not add kiteconnect as a required dependency; import it lazily inside methods.
    """

    def fetch_ohlcv(
        self,
        tickers: list[str],
        start: datetime,
        end: datetime,
        interval: Literal["1d", "1h", "15m"] = "1d",
    ) -> pl.DataFrame:
        raise NotImplementedError(
            "ZerodhaSource.fetch_ohlcv is not implemented. "
            "Use YFinanceSource or KaggleSource for historical data."
        )

    def fetch_corporate_actions(self, tickers: list[str]) -> pl.DataFrame:
        raise NotImplementedError(
            "ZerodhaSource.fetch_corporate_actions is not implemented."
        )
