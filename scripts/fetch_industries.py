#!/usr/bin/env python
"""Fetch NSE's industry classification for the point-in-time universe.

`src/trader/data/nse_industry.py` explains why this is needed rather than
reusing `sector_id_of`: that returns 0 for anything outside the 504 curated
names, and 0 is the phantom sector `CLAUDE.md` records. A point-in-time universe
admits 600–750 names, most of them outside `SECTOR_MAP`.

    uv run python scripts/fetch_industries.py

One request. ``ind_niftytotalmarket_list.csv`` is a superset of the NIFTY 500,
smallcap 250 and microcap 250 constituent lists — verified 2026-09-08 that their
union is exactly its 755 names — so the broader lists add nothing and are not
fetched.

COVERAGE IS NOT COMPLETE, BY CONSTRUCTION. This is the *current* constituent
list, so a name that delisted before today is absent and lands in the explicit
unknown bucket. That is the residual limit of classifying a historical universe
from a present-day source, and the printed report says how many names it hits
rather than leaving it to be discovered downstream.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl
from loguru import logger


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/ext/industries.parquet"))
    ap.add_argument("--cache-root", type=Path, default=Path("data/raw/nse/industries"))
    ap.add_argument("--offline", action="store_true", help="parse the cache only")
    ap.add_argument("--bars", type=Path, default=Path("data/ext/bhavcopy.parquet"),
                    help="optional: report coverage against these tickers")
    args = ap.parse_args()

    from trader.data.nse_industry import (
        INDUSTRY_URL,
        NSE_INDUSTRY_IDS,
        parse_industry_list,
    )
    from trader.data.sources.nse_flows import NSEClient, _FileCache

    cache = _FileCache(args.cache_root)
    name = "ind_niftytotalmarket_list.csv"
    text = cache.read(name)
    if text is None:
        if args.offline:
            raise SystemExit(f"--offline and {name} is not cached")
        text = NSEClient().get_text(INDUSTRY_URL)
        cache.write(name, text, url=INDUSTRY_URL)
        logger.info(f"fetched {len(text):,} bytes")
    else:
        logger.info("read from cache")

    df = parse_industry_list(text, name=name)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(args.out)

    print(f"\n{'metric':<44}{'value':>22}")
    print("-" * 66)
    print(f"{'constituents':<44}{df.height:>22,}")
    print(f"{'industries':<44}{df['industry'].n_unique():>22,}")
    print(f"{'known industries in the table':<44}{len(NSE_INDUSTRY_IDS):>22,}")
    print("-" * 66)
    print(df.group_by("industry").len().sort("len", descending=True).head(25))

    if args.bars.exists():
        from trader.data.nse_industry import UNKNOWN_INDUSTRY_ID, industry_ids

        bars = pl.read_parquet(args.bars, columns=["ticker", "series"])
        traded = sorted(
            bars.filter(pl.col("series").is_in(["EQ", "BE"]))["ticker"].unique().to_list()
        )
        ids = industry_ids(df, traded)
        unknown = [t for t, v in ids.items() if v == UNKNOWN_INDUSTRY_ID]
        print(f"\n{'tickers ever traded (EQ/BE)':<44}{len(traded):>22,}")
        print(f"{'  classified':<44}{len(traded) - len(unknown):>22,}")
        print(f"{'  unknown bucket':<44}{len(unknown):>22,}")
        print("\nThe unknown bucket is dominated by names that delisted before the "
              "current\nconstituent list was drawn, plus the illiquid tail a "
              "liquidity rule excludes\nanyway. What matters is how many survive "
              "that rule — check with\nscripts/pit_universe_report.py once the "
              "bars are complete.")

    logger.info(f"wrote {args.out}")


if __name__ == "__main__":
    main()
