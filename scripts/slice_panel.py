#!/usr/bin/env python
"""Cut a panel to a date range, so two panels can be compared on the same span.

Comparing a rebuilt panel against the one it replaces only means anything if
both cover identical dates. `scripts/make_oos_split.py` cuts a slice to a
SIGNAL's prediction span, which is the right tool when there is a signal; this
is the right one when there is not — the baselines need a span, not a model.

    uv run python scripts/slice_panel.py \
        --in data/panels_bhav/full.parquet \
        --out data/panels_bhav/oos_pit.parquet \
        --from 2016-09-28 --to 2024-06-28

Refuses to write a slice that does not actually cover the range asked for,
rather than silently producing a shorter one — a backtest quietly run on four
years instead of eight still prints a CAGR.
"""
from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import polars as pl
from loguru import logger


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", type=Path, required=True)
    ap.add_argument("--out", dest="dst", type=Path, required=True)
    ap.add_argument("--from", dest="start", required=True)
    ap.add_argument("--to", dest="end", required=True)
    ap.add_argument("--tolerance-days", type=int, default=7,
                    help="how far the actual edges may sit inside the request")
    args = ap.parse_args()

    lo, hi = date.fromisoformat(args.start), date.fromisoformat(args.end)
    if lo >= hi:
        raise SystemExit(f"--from {lo} is not before --to {hi}")

    panel = pl.read_parquet(args.src)
    have_lo, have_hi = panel["date"].min(), panel["date"].max()
    logger.info(f"{args.src}: {panel.height:,} rows, {have_lo}..{have_hi}")

    out = panel.filter((pl.col("date") >= lo) & (pl.col("date") <= hi))
    if out.is_empty():
        raise SystemExit(f"no rows in {lo}..{hi}; the panel covers {have_lo}..{have_hi}")

    got_lo, got_hi = out["date"].min(), out["date"].max()
    assert isinstance(got_lo, date) and isinstance(got_hi, date)
    # A trading calendar rarely has a session on the exact boundary, so a few
    # days of slack is expected; a month is not.
    if (got_lo - lo).days > args.tolerance_days or (hi - got_hi).days > args.tolerance_days:
        raise SystemExit(
            f"asked for {lo}..{hi} but the slice covers {got_lo}..{got_hi} — "
            f"more than {args.tolerance_days} days short at an edge. The source "
            "panel does not span the range; refusing to write a slice that would "
            "quietly measure a different period."
        )

    args.dst.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(args.dst)
    logger.info(
        f"wrote {args.dst}: {out.height:,} rows, {out['ticker'].n_unique():,} "
        f"tickers, {got_lo}..{got_hi} ({out['date'].n_unique():,} sessions)"
    )
    if "is_tradeable" in out.columns:
        per_day = out.filter(pl.col("is_tradeable")).group_by("date").len()
        logger.info(
            f"tradeable names per session: min {per_day['len'].min()}, "
            f"median {per_day['len'].median():.0f}, max {per_day['len'].max()}"
        )


if __name__ == "__main__":
    main()
