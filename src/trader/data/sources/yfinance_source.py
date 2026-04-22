from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import polars as pl
from loguru import logger


class YFinanceSource:
    """Primary data source using Yahoo Finance via yfinance.

    Caches raw responses to avoid redundant API calls.
    Cache layout: <cache_root>/<ticker>/<start>_<end>.parquet
    """

    def __init__(self, cache_root: str | Path = "data/raw/yfinance") -> None:
        self._cache = Path(cache_root)

    # ------------------------------------------------------------------ public

    def fetch_ohlcv(
        self,
        tickers: list[str],
        start: datetime,
        end: datetime,
        interval: Literal["1d", "1h", "15m"] = "1d",
    ) -> pl.DataFrame:
        frames: list[pl.DataFrame] = []
        for ticker in tickers:
            frame = self._fetch_cached(ticker, start, end, interval)
            if frame is not None:
                frames.append(frame)
        return pl.concat(frames) if frames else pl.DataFrame()

    def fetch_corporate_actions(self, tickers: list[str]) -> pl.DataFrame:
        rows: list[dict[str, Any]] = []
        for ticker in tickers:
            try:
                rows.extend(self._fetch_actions_one(ticker))
            except Exception as exc:
                logger.warning(f"Corporate actions failed for {ticker}: {exc}")
        if not rows:
            return pl.DataFrame(
                schema={
                    "ticker": pl.Utf8,
                    "action_date": pl.Date,
                    "action_type": pl.Utf8,
                    "ratio": pl.Float64,
                    "value": pl.Float64,
                }
            )
        return pl.DataFrame(rows)

    # ------------------------------------------------------------------ cache

    def _fetch_cached(
        self,
        ticker: str,
        start: datetime,
        end: datetime,
        interval: str,
    ) -> pl.DataFrame | None:
        path = self._cache_path(ticker, start, end)
        if path.exists():
            logger.debug(f"Cache hit: {path}")
            return pl.read_parquet(path)
        frame = self._fetch_one(ticker, start, end, interval)
        if frame is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.write_parquet(path)
            logger.debug(f"Cached: {path}")
        return frame

    def _cache_path(self, ticker: str, start: datetime, end: datetime) -> Path:
        safe = ticker.replace("/", "_").replace("\\", "_")
        slug = f"{start.date()}_{end.date()}"
        return self._cache / safe / f"{slug}.parquet"

    # ------------------------------------------------------------------ fetch

    def _fetch_one(
        self,
        ticker: str,
        start: datetime,
        end: datetime,
        interval: str,
        max_retries: int = 3,
    ) -> pl.DataFrame | None:
        import yfinance as yf  # lazy import — keeps module loadable without yfinance

        for attempt in range(max_retries):
            try:
                raw: Any = yf.download(
                    ticker,
                    start=start.date(),
                    end=end.date(),
                    interval=interval,
                    auto_adjust=True,
                    progress=False,
                )
                if len(raw) == 0:
                    logger.warning(f"No data for {ticker}")
                    return None
                return self._to_polars(raw, ticker)
            except Exception as exc:
                wait = 2**attempt
                logger.warning(
                    f"yfinance {ticker} attempt {attempt + 1}/{max_retries} failed: {exc}"
                    f"; retrying in {wait}s"
                )
                if attempt < max_retries - 1:
                    time.sleep(wait)
        logger.error(f"yfinance {ticker} exhausted {max_retries} retries")
        return None

    def _to_polars(self, raw: Any, ticker: str) -> pl.DataFrame:
        import pandas as pd

        df: Any = raw.copy()

        # Flatten MultiIndex columns (newer yfinance always returns MultiIndex)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = pd.Index(
                [str(c[0]).lower().replace(" ", "_") for c in df.columns]
            )
        else:
            df.columns = pd.Index(
                [str(c).lower().replace(" ", "_") for c in df.columns]
            )

        df = df.reset_index()

        # Normalize date/datetime column → "date"
        for col in list(df.columns):
            if str(col).lower() in ("date", "datetime"):
                if str(col) != "date":
                    df = df.rename(columns={str(col): "date"})
                break

        # With auto_adjust=True close == adj_close; add adj_close alias
        col_names: list[str] = [str(c) for c in df.columns]
        if "adj_close" not in col_names:
            if "adj close" in col_names:
                df = df.rename(columns={"adj close": "adj_close"})
            else:
                df["adj_close"] = df["close"]

        df["ticker"] = ticker
        df["source"] = "yfinance"

        want = [
            "date",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "adj_close",
            "ticker",
            "source",
        ]
        present = [c for c in want if c in df.columns]
        pdf = df[present].dropna(subset=["date"])
        return pl.from_pandas(pdf)

    # ------------------------------------------------------------------ actions

    def _fetch_actions_one(self, ticker: str) -> list[dict[str, Any]]:
        import yfinance as yf

        yft: Any = yf.Ticker(ticker)
        rows: list[dict[str, Any]] = []

        splits: Any = yft.splits
        if splits is not None and len(splits) > 0:
            for dt, ratio in splits.items():
                rows.append(
                    {
                        "ticker": ticker,
                        "action_date": dt.date(),
                        "action_type": "SPLIT",
                        "ratio": float(ratio),
                        "value": None,
                    }
                )

        dividends: Any = yft.dividends
        if dividends is not None and len(dividends) > 0:
            for dt, value in dividends.items():
                rows.append(
                    {
                        "ticker": ticker,
                        "action_date": dt.date(),
                        "action_type": "DIVIDEND",
                        "ratio": None,
                        "value": float(value),
                    }
                )

        return rows
