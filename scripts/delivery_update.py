"""Append recent NSE delivery data to data/ext/delivery.parquet without losing history.

    uv run python scripts/delivery_update.py            # nightly: last 10 calendar days
    uv run python scripts/delivery_update.py --days 45  # catch-up

`scripts/fetch_nse_flows.py` writes `<out_root>/delivery.parquet` for exactly the
range it was given, so running it on a short range would replace six years of
history with a few days (found 2026-10-19; its raw cache under data/raw/nse was
empty, so nothing could be rebuilt from cache). This fetches the recent range
into a scratch folder with the same fetcher and backend (`bhavdata`, which is
what the existing file holds), then merges: rows for the refetched dates are
replaced, every other date is kept. Written atomically.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from datetime import date, timedelta
from pathlib import Path

import polars as pl

STORE = Path("data/ext/delivery.parquet")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=10)
    a = ap.parse_args()
    end = date.today() - timedelta(days=1)
    start = end - timedelta(days=a.days)
    with tempfile.TemporaryDirectory() as tmp:
        r = subprocess.run(
            [sys.executable, "scripts/fetch_nse_flows.py", "--start", start.isoformat(),
             "--end", end.isoformat(), "--sources", "delivery", "--backend", "bhavdata",
             "--out-root", tmp],
            capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stderr[-2000:])
            raise SystemExit(f"fetch_nse_flows failed ({r.returncode})")
        new = pl.read_parquet(Path(tmp) / "delivery.parquet")
    old = pl.read_parquet(STORE)
    if new.height == 0:
        print(f"delivery_update: no new rows for {start}..{end}; store ends {old['date'].max()}")
        return
    new = new.select(old.columns).cast(old.schema)
    dates = new["date"].unique()
    merged = (pl.concat([old.filter(~pl.col("date").is_in(dates)), new])
              .sort("date", "ticker", "series"))
    tmpf = STORE.with_suffix(".tmp.parquet")
    merged.write_parquet(tmpf)
    tmpf.replace(STORE)
    print(f"delivery_update: +{new.height:,} rows over {len(dates)} dates "
          f"({new['date'].min()}..{new['date'].max()}); store {merged.height:,} rows, "
          f"{merged['date'].min()}..{merged['date'].max()}")


if __name__ == "__main__":
    main()
