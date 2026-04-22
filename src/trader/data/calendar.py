from __future__ import annotations

from datetime import date

import polars as pl


def build_calendar(ohlcv: pl.DataFrame) -> list[date]:
    """Derive NSE trading days from the union of all dates present in the OHLCV data.

    Using observed dates is more accurate than pandas business-day generation,
    which misses Indian market holidays.
    """
    return sorted(ohlcv["date"].unique().to_list())
