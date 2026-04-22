from __future__ import annotations

from datetime import date

import polars as pl
from loguru import logger

_PRICE_COLS = ["open", "high", "low", "close", "adj_close"]


def align_panel(
    ohlcv: pl.DataFrame,
    tickers: list[str],
    calendar: list[date],
    sector_ids: dict[str, int] | None = None,
) -> pl.DataFrame:
    """Produce a dense (date × ticker) panel with an is_tradeable mask.

    Rules (from 02_data_pipeline.md):
    - Dates before a ticker's first trade → is_tradeable=False, prices=0.
    - Dates after a ticker's last trade → is_tradeable=False, prices=0.
    - 1-day OHLC gap within tradeable span → forward-fill, volume=0, is_tradeable=True.
    - 2+ consecutive day gap → is_tradeable=False (beyond forward-fill limit).
    - Volume gaps are always filled with 0 (honest: no trading occurred).
    - Sentinel zeros must never be treated as signal — the mask is the source of truth.
    """
    # ── skeleton: full cross product of (date × ticker) ──────────────────────
    dates_df = pl.DataFrame({"date": pl.Series(calendar, dtype=pl.Date)})
    tickers_df = pl.DataFrame({"ticker": pl.Series(tickers, dtype=pl.Utf8)})
    skeleton = dates_df.join(tickers_df, how="cross")

    # ── join OHLCV (left: every skeleton row is kept) ────────────────────────
    want = ["date", "ticker"] + _PRICE_COLS + ["volume"]
    available = [c for c in want if c in ohlcv.columns]
    merged = skeleton.join(ohlcv.select(available), on=["date", "ticker"], how="left")

    # ── per-ticker first / last trade dates ───────────────────────────────────
    bounds = ohlcv.group_by("ticker").agg(
        pl.col("date").min().alias("first_date"),
        pl.col("date").max().alias("last_date"),
    )
    merged = merged.join(bounds, on="ticker", how="left")

    # ── sort so forward-fill runs in chronological order within each ticker ──
    merged = merged.sort(["ticker", "date"])

    # ── forward-fill price cols: max 1-day gap allowed ───────────────────────
    merged = merged.with_columns(
        [pl.col(c).forward_fill(limit=1).over("ticker") for c in _PRICE_COLS]
    )

    # ── mask out rows outside each ticker's [first_date, last_date] ──────────
    in_range = (pl.col("date") >= pl.col("first_date")) & (
        pl.col("date") <= pl.col("last_date")
    )
    merged = merged.with_columns(
        [
            pl.when(in_range)
            .then(pl.col(c))
            .otherwise(pl.lit(None, dtype=pl.Float64))
            .alias(c)
            for c in _PRICE_COLS
        ]
        + [
            pl.when(in_range)
            .then(pl.col("volume").fill_null(0).cast(pl.Int64))
            .otherwise(pl.lit(0, dtype=pl.Int64))
            .alias("volume")
        ]
    )

    # ── is_tradeable: row has data (real or 1-gap filled) within range ────────
    merged = merged.with_columns(
        pl.col("adj_close").is_not_null().alias("is_tradeable")
    )

    # ── fill sentinel zeros for non-tradeable price rows ─────────────────────
    merged = merged.with_columns(
        [pl.col(c).fill_null(0.0) for c in _PRICE_COLS]
    )

    # ── sector_id column ─────────────────────────────────────────────────────
    if sector_ids is not None:
        _sids = sector_ids
        merged = merged.with_columns(
            pl.col("ticker")
            .map_elements(lambda t: _sids.get(str(t), 0), return_dtype=pl.Int32)
            .alias("sector_id")
        )
    else:
        merged = merged.with_columns(pl.lit(0, dtype=pl.Int32).alias("sector_id"))

    # ── drop alignment helpers ────────────────────────────────────────────────
    merged = merged.drop(["first_date", "last_date"])

    n_tradeable = int(merged["is_tradeable"].sum())
    logger.info(
        f"Aligned panel: {merged['date'].n_unique()} days"
        f" × {merged['ticker'].n_unique()} tickers"
        f" | tradeable rows: {n_tradeable:,}"
    )

    return merged.sort(["date", "ticker"])
