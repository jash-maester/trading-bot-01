#!/usr/bin/env python
"""How much of the investable universe does the fixed 504-name list actually hold?

The number that motivated the whole rebuild, made reproducible rather than
quoted (`CLAUDE.md` rule 3):

> of the names clearing ₹5 crore of median daily turnover, the 645 panelled
> names cover 60.0% on average over 2011–2020, and 27 of the 99 missing at
> 2016-01-01 had stopped trading altogether

    uv run python scripts/pit_universe_report.py
    uv run python scripts/pit_universe_report.py --max-names 504

Reports coverage at every month-end across the bars available, and splits the
missing names into the two mechanisms — `vanished` (stopped trading, and so
absent from any list drawn today) and `omitted` (still trades, simply not
selected) — because only the second is fixable by widening a ticker list, and
reporting them together overstates how much widening would help.
"""
from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import polars as pl
from loguru import logger


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bars", type=Path, default=Path("data/ext/bhavcopy.parquet"))
    ap.add_argument("--min-turnover-cr", type=float, default=5.0)
    ap.add_argument("--lookback-days", type=int, default=365)
    ap.add_argument("--min-sessions", type=int, default=100)
    ap.add_argument("--max-names", type=int, default=0, help="0 = no cap")
    ap.add_argument("--every-months", type=int, default=6)
    args = ap.parse_args()

    from trader.data.pit_universe import (
        LiquidityRule,
        coverage_report,
        survivorship_gap,
    )
    from trader.data.universe import active_tickers, all_tickers

    rule = LiquidityRule(
        min_median_turnover=args.min_turnover_cr * 1e7,
        lookback_days=args.lookback_days,
        min_sessions=args.min_sessions,
        max_names=args.max_names or None,
    )
    logger.info(f"rule: {rule.describe()}")

    bars = pl.read_parquet(
        args.bars, columns=["date", "ticker", "series", "close", "turnover"]
    )
    lo, hi = bars["date"].min(), bars["date"].max()
    logger.info(f"{bars.height:,} rows, {bars['ticker'].n_unique():,} tickers, {lo}..{hi}")

    # Month-ends, every N months, from one lookback after the data starts.
    assert isinstance(lo, date) and isinstance(hi, date)
    start_year = lo.year + 1
    dates = [
        date(y, m, 1)
        for y in range(start_year, hi.year + 1)
        for m in range(1, 13, max(args.every_months, 1))
        if date(y, m, 1) <= hi
    ]
    if not dates:
        raise SystemExit("no evaluation dates inside the bar range")

    # TWO universes, and conflating them overstates the problem.
    #
    # `active_tickers()` is 504 and `all_tickers()` is 645. The difference is
    # not survivorship: `INACTIVE_SECTORS` holds out five whole sectors --
    # chemicals, services, construction materials, consumer durables, telecom --
    # explicitly "to keep the observation width, and therefore the wall-clock
    # cost per run, at 504 rather than 645" (universe.py). It costs eight NIFTY
    # 50 names including ASIANPAINT and BHARTIARTL, and emptying that set is a
    # config change, not a re-download.
    #
    # So coverage against 504 measures a compute decision PLUS selection, and
    # only coverage against 645 -- everything actually fetched and panelled --
    # isolates the selection bias this rebuild exists to remove.
    active, everything = active_tickers(), all_tickers()
    rep = coverage_report(bars, dates, everything, rule)
    rep_active = coverage_report(bars, dates, active, rule)

    print(f"\n{'date':<12}{'eligible':>10}{'in 645':>9}{'cover':>8}"
          f"{'in 504':>9}{'cover':>8}{'missing/645':>13}")
    print("-" * 69)
    for a, b in zip(rep.iter_rows(named=True), rep_active.iter_rows(named=True)):
        print(f"{str(a['date']):<12}{a['n_eligible']:>10,}{a['n_covered']:>9,}"
              f"{a['coverage']:>8.1%}{b['n_covered']:>9,}{b['coverage']:>8.1%}"
              f"{a['n_missing']:>13,}")
    print("-" * 69)
    fin = rep.filter(pl.col("n_eligible") > 0)
    fa = rep_active.filter(pl.col("n_eligible") > 0)
    if fin.height:
        print(f"{'mean':<12}{fin['n_eligible'].mean():>10,.0f}"
              f"{fin['n_covered'].mean():>9,.0f}{fin['coverage'].mean():>8.1%}"
              f"{fa['n_covered'].mean():>9,.0f}{fa['coverage'].mean():>8.1%}"
              f"{fin['n_missing'].mean():>13,.0f}")
        print("\n  in 645 = everything fetched and panelled; the honest measure "
              "of selection bias.")
        print("  in 504 = the traded subset, which also excludes five sectors "
              "for compute\n           reasons (universe.py INACTIVE_SECTORS) "
              "-- not survivorship.")
    universe = everything

    # The split that decides what widening a list can and cannot fix.
    probe = dates[len(dates) // 2]
    gap = survivorship_gap(bars, probe, universe, rule, still_trading_after=hi)
    if gap.is_empty():
        print(f"\nNo gap at {probe}.")
    else:
        counts = gap.group_by("reason").len().sort("reason")
        print(f"\nAt {probe}, the {gap.height} eligible names our universe misses "
              f"split as:")
        for r in counts.iter_rows(named=True):
            print(f"  {r['reason']:<12}{r['len']:>5}")
        print("\n  vanished = stopped trading, so absent from any list drawn today;")
        print("             widening a 2026 ticker list cannot recover these.")
        print("  omitted  = still trades and was simply not selected;")
        print("             widening the list fixes exactly these.")
        for reason in ("vanished", "omitted"):
            sub = gap.filter(pl.col("reason") == reason)
            if sub.height:
                names = ", ".join(t.removesuffix(".NS") for t in sub["ticker"].to_list()[:12])
                print(f"\n  {reason} e.g.: {names}")

    n_elig = int(rep["n_eligible"].max() or 0)
    print(f"\nPanelled universe {len(everything)} names ({len(active)} traded). "
          f"The point-in-time universe peaks at {n_elig:,}.")
    print("A coverage below 100% is names that were liquid and tradeable on the "
          "date and\nare simply absent — the selection half of survivorship, "
          "which `listing.py`\nstates is not prevented.")


if __name__ == "__main__":
    main()
