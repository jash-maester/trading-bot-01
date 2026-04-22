from __future__ import annotations

from datetime import datetime
from pathlib import Path

import polars as pl
from loguru import logger


class OhlcvStore:
    """Partitioned Parquet store for raw OHLCV data.

    Layout: <root>/year=YYYY/ticker=TICKER.parquet
    Each file covers one calendar year for one ticker.
    On duplicate date rows, the existing record is kept (idempotent saves).
    """

    def __init__(self, root: str | Path = "data/ohlcv") -> None:
        self._root = Path(root)

    def save(self, df: pl.DataFrame) -> None:
        """Persist OHLCV rows, partitioned by year × ticker."""
        if df.is_empty():
            return

        df_aug = df.with_columns(pl.col("date").dt.year().alias("_year"))
        pairs = df_aug.select(["ticker", "_year"]).unique()

        for row in pairs.iter_rows():
            ticker: str = str(row[0])
            year: int = int(row[1])
            group = df_aug.filter(
                (pl.col("ticker") == ticker) & (pl.col("_year") == year)
            ).drop("_year")

            out = self._partition_path(ticker, year)
            out.parent.mkdir(parents=True, exist_ok=True)

            if out.exists():
                existing = pl.read_parquet(out)
                combined = (
                    pl.concat([existing, group])
                    .unique(subset=["date"])
                    .sort("date")
                )
                combined.write_parquet(out)
                logger.debug(f"Updated {out} ({len(combined)} rows)")
            else:
                group.sort("date").write_parquet(out)
                logger.debug(f"Created {out} ({len(group)} rows)")

    def load(
        self,
        tickers: list[str] | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pl.DataFrame:
        """Load and concatenate OHLCV rows from partitioned store."""
        start_year = start.year if start is not None else 2010
        end_year = end.year if end is not None else datetime.now().year

        frames: list[pl.DataFrame] = []
        for year in range(start_year, end_year + 1):
            year_dir = self._root / f"year={year}"
            if not year_dir.exists():
                continue
            if tickers is not None:
                for ticker in tickers:
                    p = self._partition_path(ticker, year)
                    if p.exists():
                        frames.append(pl.read_parquet(p))
            else:
                for p in sorted(year_dir.glob("ticker=*.parquet")):
                    frames.append(pl.read_parquet(p))

        if not frames:
            return pl.DataFrame()

        result = pl.concat(frames)
        if start is not None:
            result = result.filter(pl.col("date") >= start.date())
        if end is not None:
            result = result.filter(pl.col("date") <= end.date())
        return result.sort(["ticker", "date"])

    def _partition_path(self, ticker: str, year: int) -> Path:
        safe = ticker.replace("/", "_").replace("\\", "_")
        return self._root / f"year={year}" / f"ticker={safe}.parquet"
