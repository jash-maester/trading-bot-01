#!/usr/bin/env python
"""Fetch the full universe's daily bars from Zerodha Kite into a partitioned store.

Mirrors ``scripts/fetch_data.py`` (yfinance) but points at the broker's own history
API, which reaches back to 2005-01-03 for established names — roughly nine extra
years compared to the 2014 start the yfinance dataset was built with.

Usage:
    set -a && . ./.env && set +a          # KITE_API_KEY / KITE_ACCESS_TOKEN
    uv run python scripts/fetch_kite_data.py data=kite_v1

Writes to ``data/kite_ohlcv/`` (config: ``data.parquet_root``). It never touches
``data/ohlcv/`` — the existing validated yfinance dataset is a separate artefact
that the new one is meant to be compared against, not overwritten by.

Being a good API citizen
------------------------
* Chunked at ``data.max_days_per_request`` calendar days. Kite's cap is 2000
  *calendar* days per request — not 2000 candles — and a longer range is a hard
  ``InputException``, so the 21-year span becomes 6 requests per ticker.
* Throttled to 3 requests/second (``ZerodhaSource``'s default) with exponential
  backoff on transient failures.
* Cached per ticker under ``data/raw/zerodha/bars/``. A re-run after a partial
  failure costs one instrument-dump read and nothing else.

Auth failures abort the whole run rather than being retried per ticker: a dead
access token fails identically for all 163 symbols, and 163 login-loop retries
against an expired token produce an empty store plus 163 identical errors.
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
    import polars as pl

    from trader.data.sources.kite_symbols import ResolutionStatus
    from trader.data.sources.zerodha_source import (
        ZerodhaAuthError,
        ZerodhaPermissionError,
        ZerodhaSource,
        ZerodhaUnavailableError,
    )
    from trader.data.storage import OhlcvStore
    from trader.data.universe import BLACKLISTED_TICKERS, all_tickers

    orig_cwd = Path(hydra.utils.get_original_cwd())

    start = datetime.fromisoformat(str(cfg.data.start_date))
    end = datetime.fromisoformat(str(cfg.data.end_date))
    parquet_root = orig_cwd / str(cfg.data.parquet_root)
    cache_root = orig_cwd / str(cfg.data.raw_cache) / "zerodha"
    bar_cache = cache_root / "bars"
    force = bool(cfg.data.get("force_refetch", False))
    max_days = int(cfg.data.get("max_days_per_request", 1400))
    index_ticker = str(cfg.data.get("index_ticker", "^NSEI"))
    want_index = bool(cfg.data.get("fetch_index", True))

    tickers = all_tickers()

    # Blacklisted names are fetched but never enter the panel: `all_tickers()` still
    # excludes them, so nothing downstream picks them up. Having the bars on disk is
    # what makes "can TATAMOTORS come off the blacklist?" answerable with evidence
    # instead of another round of guessing.
    extra = sorted(BLACKLISTED_TICKERS)
    if want_index:
        extra.append(index_ticker)

    source = ZerodhaSource(
        cache_root=cache_root,
        include_indices=want_index,
        max_days_per_request={"day": max_days},
    )
    store = OhlcvStore(root=parquet_root)

    # ── resolve first, fetch second ──────────────────────────────────────────
    # One instrument-dump read answers "which of these names does the broker still
    # know about" for the whole universe. Finding out per ticker, a thousand
    # requests into the run, is how a rename turns into a silently smaller panel.
    report = source.resolve(tickers + extra)
    for line in report.format_lines():
        logger.info(line)

    for res in report.by_status(ResolutionStatus.DELISTED):
        logger.warning(f"DELISTED — {res.ticker} will have no bars: {res.note}")
    unresolved = report.by_status(ResolutionStatus.UNRESOLVED)
    for res in unresolved:
        logger.error(f"UNRESOLVED — {res.ticker}: {res.note}")
    if unresolved:
        logger.error(
            f"{len(unresolved)} ticker(s) could not be resolved and will be missing from "
            "the panel. Add each to KITE_RENAMES or DELISTED_SYMBOLS in "
            "trader.data.sources.kite_symbols before trusting this dataset."
        )

    fetchable = [r.ticker for r in report.fetchable]
    logger.info(
        f"Fetching {len(fetchable)} instruments  {start.date()} → {end.date()}  "
        f"(chunked at {max_days}d, ~{_n_chunks(start, end, max_days)} requests each)"
    )

    total_bars = 0
    cached = 0
    failed: list[str] = []

    for i, ticker in enumerate(fetchable, 1):
        path = _cache_path(bar_cache, ticker, start, end)
        try:
            if path.exists() and not force:
                df = pl.read_parquet(path)
                cached += 1
            else:
                df = source.fetch_ohlcv([ticker], start, end)
                if not df.is_empty():
                    path.parent.mkdir(parents=True, exist_ok=True)
                    df.write_parquet(path)
        except (ZerodhaAuthError, ZerodhaPermissionError, ZerodhaUnavailableError):
            # Global, not per-ticker: stop now so the failure stays one loud line.
            logger.error("Aborting: the Kite session/plan cannot serve history.")
            raise
        except Exception as exc:
            logger.error(f"[{i:3d}/{len(fetchable)}] {ticker} failed: {exc}")
            failed.append(ticker)
            continue

        if df.is_empty():
            logger.warning(f"[{i:3d}/{len(fetchable)}] {ticker}: no bars returned")
            failed.append(ticker)
            continue

        store.save(df)
        total_bars += len(df)
        first, last = df["date"].min(), df["date"].max()
        logger.info(
            f"[{i:3d}/{len(fetchable)}] {ticker:18s} {len(df):>5,} bars  "
            f"{_day(first)} → {_day(last)}"
        )

    logger.info(
        f"Done. {len(fetchable) - len(failed)}/{len(fetchable)} instruments, "
        f"{total_bars:,} bars → {parquet_root}  ({cached} served from cache)"
    )
    if failed:
        logger.warning(f"No data for: {failed}")


# ── helpers ────────────────────────────────────────────────────────────────────


def _cache_path(root: Path, ticker: str, start: datetime, end: datetime) -> Path:
    """One parquet per (ticker, range) — same layout as the yfinance raw cache."""
    safe = ticker.replace("/", "_").replace("\\", "_").replace("^", "IDX_")
    return root / safe / f"{start.date()}_{end.date()}.parquet"


def _n_chunks(start: datetime, end: datetime, max_days: int) -> int:
    span = (end.date() - start.date()).days
    return max(1, -(-span // max_days))


def _day(value: object) -> str:
    return str(value)[:10]


if __name__ == "__main__":
    main()
