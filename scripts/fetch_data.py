#!/usr/bin/env python
"""Fetch OHLCV for the full universe and cache to partitioned Parquet.

Usage:
    python scripts/fetch_data.py data=universe_v1
    python scripts/fetch_data.py data=universe_v1 data.start_date=2020-01-01
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import hydra
from loguru import logger
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    import hydra.utils

    from trader.data.sources.yfinance_source import YFinanceSource
    from trader.data.storage import OhlcvStore
    from trader.data.universe import all_tickers

    orig_cwd = Path(hydra.utils.get_original_cwd())

    start = datetime.fromisoformat(str(cfg.data.start_date))
    end = datetime.fromisoformat(str(cfg.data.end_date))
    tickers = all_tickers()

    logger.info(
        f"Fetching {len(tickers)} tickers"
        f"  {start.date()} → {end.date()}"
    )

    cache_root = orig_cwd / str(cfg.data.raw_cache) / "yfinance"
    parquet_root = orig_cwd / str(cfg.data.parquet_root)

    source = YFinanceSource(cache_root=cache_root)
    store = OhlcvStore(root=parquet_root)

    failed: list[str] = []
    for i, ticker in enumerate(tickers, 1):
        logger.info(f"[{i:3d}/{len(tickers)}] {ticker}")
        try:
            df = source.fetch_ohlcv([ticker], start, end)
            if not df.is_empty():
                store.save(df)
            else:
                logger.warning(f"No data for {ticker}")
                failed.append(ticker)
        except Exception as exc:
            logger.error(f"Failed for {ticker}: {exc}")
            failed.append(ticker)

    logger.info(f"Done. {len(tickers) - len(failed)}/{len(tickers)} succeeded.")
    if failed:
        logger.warning(f"Failed tickers: {failed}")


if __name__ == "__main__":
    main()
