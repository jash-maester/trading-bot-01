#!/usr/bin/env python
"""Turn the full-market bhavcopy into an OhlcvStore the panel builder can read.

`scripts/fetch_bhavcopy.py` produces one long frame of every security NSE
traded. `scripts/build_features.py` reads an `OhlcvStore` and knows nothing
about bhavcopy. Writing the bars into that layout means the entire downstream
pipeline — `align_panel`, `compute_features`, the splits, the env — works
unchanged on a point-in-time universe, rather than being forked.

    uv run python scripts/bhavcopy_to_store.py
    uv run python scripts/bhavcopy_to_store.py --root data/bhav_ohlcv

THREE THINGS THIS HAS TO GET RIGHT
----------------------------------
**Adjustment.** Bhavcopy prices are raw; Kite's are `auto_adjust=True`. A panel
mixing the two conventions would price every split as a crash. Every ticker is
back-adjusted onto its latest scale from NSE's own ``prev_close``
(`nse_bhavcopy.back_adjust`), which also catches demergers — the case
`auto_adjust` does not handle and that produced the fake NIITLTD −76.13% loss in
`CLAUDE.md`.

**EQ/BE collisions.** `nse_symbol_to_ticker` strips the series suffix, so
`STLTECH` and `STLTECH-BE` both map to `STLTECH.NS`. A name moved to
trade-to-trade mid-history would otherwise contribute two rows for one date and
silently double its volume. EQ wins where both exist on a day; BE is kept where
it is the only listing, because a name in trade-to-trade is still tradeable.

**The index.** `compute_features` requires NIFTY returns for `beta_nifty_60d`
and refuses to fabricate them — `CLAUDE.md` records that channel sitting at a
constant 1.0 across 276,005 training rows because `^NSEI` was never fetched.
Bhavcopy carries securities, not indices, so `^NSEI` is copied from the existing
Kite store. If it is absent the script says so and stops rather than writing a
panel whose beta column is dead.
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import polars as pl
from loguru import logger


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bars", type=Path, default=Path("data/ext/bhavcopy.parquet"))
    ap.add_argument("--root", type=Path, default=Path("data/bhav_ohlcv"))
    ap.add_argument("--index-from", type=Path, default=Path("data/kite_ohlcv"),
                    help="store to copy ^NSEI out of")
    ap.add_argument("--series", default="EQ,BE")
    ap.add_argument("--min-sessions", type=int, default=60,
                    help="drop a ticker with fewer sessions than this in total")
    ap.add_argument("--ever-eligible", action="store_true", default=True,
                    help="keep only tickers that clear the liquidity bar at "
                         "least once (see --min-turnover-cr)")
    ap.add_argument("--keep-all", dest="ever_eligible", action="store_false")
    ap.add_argument("--min-turnover-cr", type=float, default=5.0)
    ap.add_argument("--min-eligible-sessions", type=int, default=100)
    args = ap.parse_args()

    from trader.data.sources.nse_bhavcopy import back_adjust
    from trader.data.storage import OhlcvStore

    keep = [s.strip() for s in args.series.split(",") if s.strip()]
    bars = pl.read_parquet(args.bars)
    logger.info(f"{bars.height:,} raw rows, {bars['ticker'].n_unique():,} tickers")

    bars = bars.filter(pl.col("series").is_in(keep))
    logger.info(f"{bars.height:,} rows in series {keep}")

    # ── EQ/BE collision ──────────────────────────────────────────────────────
    # Rank so EQ sorts first, then keep one row per (ticker, date).
    before = bars.height
    bars = (
        bars.with_columns(
            pl.when(pl.col("series") == "EQ").then(0).otherwise(1).alias("_pref")
        )
        .sort(["ticker", "date", "_pref"])
        .unique(subset=["ticker", "date"], keep="first", maintain_order=True)
        .drop("_pref")
    )
    if before != bars.height:
        logger.info(
            f"{before - bars.height:,} duplicate (ticker, date) row(s) collapsed — "
            "a name listed in both EQ and BE on the same day"
        )

    # ── drop names with too little history to model ──────────────────────────
    counts = bars.group_by("ticker").agg(pl.len().alias("n"))
    thin = counts.filter(pl.col("n") < args.min_sessions)
    if thin.height:
        logger.info(f"dropping {thin.height:,} ticker(s) with < {args.min_sessions} sessions")
        bars = bars.join(thin.select("ticker"), on="ticker", how="anti")

    # ── ever-eligible prefilter ──────────────────────────────────────────────
    # NSE lists ~2,200 EQ/BE securities. Writing all of them would make a panel
    # of 2,200 x 4,138 = 9.1M rows, and `build_features` takes its universe from
    # the store, so the action space would follow. Almost all of that width is
    # names the liquidity rule never admits on any date.
    #
    # So the store holds exactly the modelling universe: every ticker that
    # clears the turnover bar in at least ONE year-long window. That is a
    # strictly wider set than any single date's universe -- it is the union over
    # all dates -- so nothing the point-in-time rule would ever select is
    # dropped here, and the per-date decision still happens later against
    # `is_tradeable`.
    if args.ever_eligible:
        per_year = (
            bars.with_columns(pl.col("date").dt.year().alias("_y"))
            .group_by(["ticker", "_y"])
            .agg(
                pl.col("turnover").median().alias("med"),
                pl.len().alias("n"),
            )
            .filter(
                (pl.col("med") >= args.min_turnover_cr * 1e7)
                & (pl.col("n") >= args.min_eligible_sessions)
            )
        )
        keep_t = per_year.select("ticker").unique()
        before_t = bars["ticker"].n_unique()
        bars = bars.join(keep_t, on="ticker", how="inner")
        logger.info(
            f"ever-eligible prefilter: {before_t:,} -> {bars['ticker'].n_unique():,} "
            f"tickers (>= Rs {args.min_turnover_cr}cr median in some year, "
            f">= {args.min_eligible_sessions} sessions that year)"
        )

    # ── corporate actions ────────────────────────────────────────────────────
    from trader.data.sources.nse_bhavcopy import adjustment_ratios

    acts = adjustment_ratios(bars)
    logger.info(
        f"{acts.height:,} corporate action(s) detected from prev_close across "
        f"{acts['ticker'].n_unique():,} ticker(s)"
    )
    adj = back_adjust(bars)

    # `compute_features` wants adj_close alongside close. These bars ARE
    # adjusted, so the two are the same series; writing both keeps the
    # downstream contract unchanged rather than special-casing it.
    out = adj.select(
        pl.col("date"),
        pl.col("ticker"),
        pl.col("open"),
        pl.col("high"),
        pl.col("low"),
        pl.col("close"),
        pl.col("close").alias("adj_close"),
        pl.col("volume").cast(pl.Int64),
        pl.col("turnover"),
    ).sort(["ticker", "date"])

    store = OhlcvStore(root=args.root)
    n_t = out["ticker"].n_unique()
    logger.info(f"writing {out.height:,} rows for {n_t:,} tickers to {args.root}")
    store.save(out)

    # ── the index ────────────────────────────────────────────────────────────
    src = OhlcvStore(root=args.index_from)
    nsei = src.load(tickers=["^NSEI"], start=datetime(2005, 1, 1))
    if not nsei.is_empty():
        # `OhlcvStore.load` concatenates partitions vertically, which needs
        # identical column names AND order. The Kite store's ^NSEI has no
        # `turnover` — bhavcopy publishes it and Kite bars do not — so a plain
        # copy makes the store unreadable with
        # "unable to vstack, column names don't match". Align it to the bhav
        # schema with a null turnover: an index has no traded value, and null
        # says that, where 0.0 would claim it traded nothing and quietly fail
        # the liquidity rule for a reason that is not true.
        cols = out.columns
        for c in cols:
            if c not in nsei.columns:
                nsei = nsei.with_columns(pl.lit(None).cast(out.schema[c]).alias(c))
        nsei = nsei.select(cols)
    if nsei.is_empty():
        raise SystemExit(
            f"^NSEI not found in {args.index_from}. compute_features needs NIFTY "
            "returns for beta_nifty_60d and refuses to fabricate them; writing a "
            "store without it would produce the dead constant-1.0 beta channel "
            "CLAUDE.md records against 276,005 training rows."
        )
    store.save(nsei)
    logger.info(f"copied ^NSEI: {nsei.height:,} rows, "
                f"{nsei['date'].min()}..{nsei['date'].max()}")

    print(f"\n{'metric':<44}{'value':>22}")
    print("-" * 66)
    print(f"{'tickers written':<44}{n_t:>22,}")
    print(f"{'rows':<44}{out.height:>22,}")
    print(f"{'date range':<44}"
          f"{str(out['date'].min()) + '..' + str(out['date'].max()):>22}")
    print(f"{'corporate actions applied':<44}{acts.height:>22,}")
    print("-" * 66)
    print("\nBuild a panel on it with:\n"
          "  uv run python scripts/build_features.py data=bhav_v1")


if __name__ == "__main__":
    main()
