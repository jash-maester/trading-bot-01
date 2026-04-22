from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal

import polars as pl
from loguru import logger


class KaggleSource:
    """Fallback source loading pre-downloaded Kaggle parquet files.

    Expected layout: <data_root>/<TICKER>.parquet
    Columns (after normalisation): date, open, high, low, close, volume, adj_close
    """

    def __init__(self, data_root: str | Path = "data/raw/kaggle") -> None:
        self._root = Path(data_root)

    def fetch_ohlcv(
        self,
        tickers: list[str],
        start: datetime,
        end: datetime,
        interval: Literal["1d", "1h", "15m"] = "1d",
    ) -> pl.DataFrame:
        if interval != "1d":
            logger.warning(f"KaggleSource only supports '1d'; ignoring '{interval}'")
            return pl.DataFrame()

        frames: list[pl.DataFrame] = []
        for ticker in tickers:
            path = self._root / f"{ticker}.parquet"
            if not path.exists():
                logger.debug(f"No Kaggle file for {ticker}: {path}")
                continue
            try:
                df = pl.read_parquet(path)
                df = self._normalize(df, ticker)
                df = df.filter(
                    (pl.col("date") >= start.date()) & (pl.col("date") <= end.date())
                )
                if not df.is_empty():
                    frames.append(df)
            except Exception as exc:
                logger.warning(f"KaggleSource failed for {ticker}: {exc}")

        return pl.concat(frames) if frames else pl.DataFrame()

    def fetch_corporate_actions(self, tickers: list[str]) -> pl.DataFrame:
        return pl.DataFrame(
            schema={
                "ticker": pl.Utf8,
                "action_date": pl.Date,
                "action_type": pl.Utf8,
                "ratio": pl.Float64,
                "value": pl.Float64,
            }
        )

    def _normalize(self, df: pl.DataFrame, ticker: str) -> pl.DataFrame:
        rename: dict[str, str] = {}
        for col in df.columns:
            normalized = col.lower().replace(" ", "_")
            if normalized != col:
                rename[col] = normalized
        if rename:
            df = df.rename(rename)

        if "date" in df.columns and df["date"].dtype != pl.Date:
            df = df.with_columns(pl.col("date").cast(pl.Date))

        if "adj_close" not in df.columns and "close" in df.columns:
            df = df.with_columns(pl.col("close").alias("adj_close"))

        df = df.with_columns(
            pl.lit(ticker).alias("ticker"),
            pl.lit("kaggle").alias("source"),
        )
        return df
