#!/usr/bin/env python
"""Fetch NSE quarterly-results metadata for the universe, with visibility dates.

`13_fundamentals_and_news.md` section 3. This fetches the results INDEX — the
period each filing covers, when NSE broadcast it, and where its XBRL document
lives. It does not yet parse the figures themselves.

    # dry run: 10 symbols, nothing written
    uv run python scripts/fetch_fundamentals.py --limit 10 --dry-run

    # the real thing
    uv run python scripts/fetch_fundamentals.py --out data/ext/results.parquet

Every figure carries `visible_from`, snapped onto the panel's own trading
calendar. Nothing downstream may join on `period_to`: that is the lookahead this
whole module exists to prevent, and it is worth restating because the wrong
column is right there next to the right one.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl
from loguru import logger


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/ext/results.parquet"))
    ap.add_argument("--panel", type=Path, default=Path("data/panels_kite/full.parquet"))
    ap.add_argument("--cache-root", type=Path, default=Path("data/raw/nse"))
    ap.add_argument("--period", default="Quarterly")
    ap.add_argument("--limit", type=int, default=0, help="first N symbols only")
    ap.add_argument("--symbols", default="", help="comma-separated, overrides --limit")
    ap.add_argument("--offline", action="store_true", help="parse the cache only")
    ap.add_argument("--dry-run", action="store_true", help="report, write nothing")
    args = ap.parse_args()

    from trader.data.sources.nse_fundamentals import (
        FinancialResultsSource,
        ResultsFetchPlan,
        attach_visibility,
        latest_per_period,
    )
    from trader.data.universe import active_tickers

    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    else:
        # active_tickers() is ".NS"-suffixed; NSE's API wants the bare symbol.
        symbols = [t.removesuffix(".NS") for t in active_tickers()]
        if args.limit:
            symbols = symbols[: args.limit]

    plan = ResultsFetchPlan(tuple(symbols), args.period)
    logger.info(f"plan: {plan.describe()}")

    src = FinancialResultsSource(
        cache_root=args.cache_root, offline=args.offline, period=args.period
    )
    raw = src.fetch(symbols)
    if raw.is_empty():
        raise SystemExit("no filings returned; nothing to do")
    logger.info(f"{raw.height} filings for {raw['ticker'].n_unique()} ticker(s)")

    one = latest_per_period(raw)
    logger.info(
        f"{one.height} after one-per-(ticker, period): "
        f"{raw.height - one.height} duplicate/standalone/revised rows collapsed"
    )

    sessions = sorted(
        pl.read_parquet(args.panel, columns=["date"])["date"].unique().to_list()
    )
    out = attach_visibility(one, sessions)
    n_null = int(out["visible_from"].is_null().sum())
    usable = out.filter(pl.col("visible_from").is_not_null())

    lag = (usable["visible_from"] - usable["period_to"]).dt.total_days()
    print(f"\n{'metric':<44}{'value':>22}")
    print("-" * 66)
    print(f"{'filings fetched':<44}{raw.height:>22,}")
    print(f"{'after one-per-(ticker, period)':<44}{one.height:>22,}")
    print(f"{'usable (visible_from on a session)':<44}{usable.height:>22,}")
    print(f"{'dropped: visible after last session':<44}{n_null:>22,}")
    print(f"{'tickers covered':<44}{usable['ticker'].n_unique():>22,}")
    print(f"{'period range':<44}"
          f"{str(usable['period_to'].min()) + '..' + str(usable['period_to'].max()):>22}")
    print(f"{'announcement lag, median days':<44}{lag.median():>22.0f}")
    print(f"{'announcement lag, p10 / p90':<44}"
          f"{f'{lag.quantile(0.1):.0f} / {lag.quantile(0.9):.0f}':>22}")
    print(f"{'filings carrying an XBRL document':<44}"
          f"{int(usable['xbrl_url'].is_not_null().sum()):>22,}")
    print(f"{'consolidated basis':<44}"
          f"{int((usable['consolidated'] == 'Consolidated').sum()):>22,}")
    print("-" * 66)

    # The invariant that makes any of this usable. Asserted, not assumed.
    bad = usable.filter(pl.col("visible_from") <= pl.col("period_to"))
    if bad.height:
        raise SystemExit(
            f"{bad.height} filing(s) are visible on or before their period end — "
            "that is lookahead, refusing to write"
        )
    print("  OK: every filing becomes visible strictly after its period ends.")

    if args.dry_run:
        logger.info("dry run: nothing written")
        return
    args.out.parent.mkdir(parents=True, exist_ok=True)
    usable.write_parquet(args.out)
    logger.info(f"wrote {args.out}: {usable.height:,} rows")


if __name__ == "__main__":
    main()
