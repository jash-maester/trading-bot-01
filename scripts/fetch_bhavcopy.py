#!/usr/bin/env python
"""Fetch NSE's full-market daily bars, so the universe can be point-in-time.

Measured 2026-09-08: of the names clearing ₹5 crore of median daily turnover,
the 645 panelled names cover only **60.0%** on average over 2011–2020, and 27 of
the 99 missing at 2016-01-01 had stopped trading altogether. Every number in
this repository is measured on a list drawn in 2026, and this is the data that
fixes it.

    uv run python scripts/fetch_bhavcopy.py --from 2010-01-01     # everything
    uv run python scripts/fetch_bhavcopy.py --limit 20            # a probe

Two archives, and the script prefers the older one on purpose:

* **classic** ``cm<DDMMMYYYY>bhav.csv.zip`` — verified to serve 2010-01-04 …
  2024-06-03, and it carries **ISIN**. ISIN is the identity that survives a
  rename, and `CLAUDE.md` records renames as having already cost this project
  history (`AMARAJABAT`→`ARE&M`, `CADILAHC`→`ZYDUSLIFE`). It spans the whole
  2016-07 … 2024-06 walk-forward.
* **sec** ``sec_bhavdata_full_<DDMMYYYY>.csv`` — starts between 2019-09-27 and
  2019-09-30 and runs to today, but publishes no ISIN. Used only where classic
  is unavailable, which is the 2024-07+ tail.

Built to be interrupted, like `fetch_xbrl_figures.py`: every body is cached on
disk, rows flush every ``--flush-every`` days with a progress marker, and a day
that will not parse gets a ``.bad`` marker rather than being retried forever or
replayed from cache as if it were data.

Trading days come from the panel's own calendar, so weekends and holidays are
never requested — roughly a third of the calls saved against iterating dates.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import date
from pathlib import Path

import polars as pl
from loguru import logger


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/ext/bhavcopy.parquet"))
    ap.add_argument("--panel", type=Path, default=Path("data/panels_kite/full.parquet"),
                    help="source of the trading calendar")
    ap.add_argument("--cache-root", type=Path, default=Path("data/raw/nse/bhavcopy"))
    ap.add_argument("--from", dest="start", default="2010-01-01")
    ap.add_argument("--to", dest="end", default="")
    ap.add_argument("--limit", type=int, default=0, help="first N sessions only")
    ap.add_argument("--flush-every", type=int, default=100)
    ap.add_argument("--sleep", type=float, default=0.4)
    ap.add_argument("--restart", action="store_true")
    args = ap.parse_args()

    from trader.data.sources.nse_bhavcopy import (
        CLASSIC_VERIFIED_FROM,
        BhavcopyParseError,
        classic_url,
        parse_classic_bhavcopy,
        parse_sec_bhavdata_ohlcv,
        unzip_bhavcopy,
    )
    from trader.data.sources.nse_flows import (
        BHAVDATA_URL,
        NSEClient,
        NSENotFound,
        _FileCache,
    )

    lo = date.fromisoformat(args.start)
    hi = date.fromisoformat(args.end) if args.end else date(2100, 1, 1)
    sessions = [
        d for d in sorted(
            pl.read_parquet(args.panel, columns=["date"])["date"].unique().to_list()
        )
        if lo <= d <= hi
    ]
    if args.limit:
        sessions = sessions[: args.limit]
    total = len(sessions)
    logger.info(f"{total:,} trading sessions from {sessions[0]} to {sessions[-1]}")

    cache = _FileCache(args.cache_root)
    client = NSEClient()
    prog = args.out.with_suffix(".progress.json")
    done: set[str] = set()
    frames: list[pl.DataFrame] = []

    if args.out.exists() and prog.exists() and not args.restart:
        frames = [pl.read_parquet(args.out)]
        done = set(json.loads(prog.read_text()).get("done", []))
        logger.info(f"resuming: {frames[0].height:,} rows, {len(done):,} sessions done")

    args.out.parent.mkdir(parents=True, exist_ok=True)

    def flush() -> None:
        if frames:
            pl.concat(frames, how="vertical").write_parquet(args.out)
        prog.write_text(json.dumps({"done": sorted(done)}) + "\n")

    t0 = time.time()
    fetched = cached = failed = 0
    for i, day in enumerate(sessions, start=1):
        key = day.isoformat()
        if key in done:
            continue

        frame: pl.DataFrame | None = None
        # Classic first, for the ISIN. Only attempted where it is plausible, so
        # the 2024-07+ tail does not pay a 404 per session.
        if day >= CLASSIC_VERIFIED_FROM:
            name = f"cm{day.strftime('%d%b%Y').upper()}bhav.csv"
            if not cache.is_bad(name):
                body = cache.read(name)
                if body is None:
                    try:
                        raw = client.get_bytes(classic_url(day))
                        body = unzip_bhavcopy(raw)
                        cache.write(name, body, url=classic_url(day))
                        fetched += 1
                        time.sleep(args.sleep)
                    except (NSENotFound, BhavcopyParseError):
                        body = None
                    except Exception as exc:  # noqa: BLE001
                        logger.debug(f"{name}: classic fetch failed, {exc}")
                        body = None
                else:
                    cached += 1
                if body is not None:
                    try:
                        frame = parse_classic_bhavcopy(body, name=name)
                    except ValueError as exc:
                        # ValueError, not BhavcopyParseError: the latter is a
                        # SUBCLASS of it, so anything raised deeper in the parse
                        # walked straight past this handler and killed the run.
                        # A two-digit TIMESTAMP did exactly that at session
                        # 2,600 of 4,138. One malformed day must never abort a
                        # multi-thousand-day backfill -- the same lesson the
                        # bhavdata backfill already paid for.
                        cache.mark_bad(name, str(exc))
                        logger.warning(f"{name}: unparseable classic, marked bad — {exc}")

        if frame is None:
            name = f"sec_bhavdata_full_{day.strftime('%d%m%Y')}.csv"
            if not cache.is_bad(name):
                body = cache.read(name)
                if body is None:
                    try:
                        body = client.get_text(BHAVDATA_URL.format(
                            ddmmyyyy=day.strftime("%d%m%Y")))
                        cache.write(name, body, url="sec")
                        fetched += 1
                        time.sleep(args.sleep)
                    except Exception as exc:  # noqa: BLE001
                        logger.debug(f"{name}: sec fetch failed, {exc}")
                        body = None
                else:
                    cached += 1
                if body is not None:
                    try:
                        frame = parse_sec_bhavdata_ohlcv(body, name=name)
                    except ValueError as exc:
                        cache.mark_bad(name, str(exc))
                        logger.warning(f"{name}: unparseable sec, marked bad — {exc}")

        if frame is None:
            failed += 1
            logger.warning(f"{day}: NEITHER archive yielded a bhavcopy")
        else:
            frames.append(frame)
        done.add(key)

        if i % args.flush_every == 0:
            flush()
            el = time.time() - t0
            rate = max(fetched, 1) / max(el, 1e-9)
            rows = sum(f.height for f in frames)
            logger.info(
                f"  {i}/{total}  rows={rows:,}  fetched={fetched} cached={cached} "
                f"failed={failed}  eta {(total - i) / max(rate, 1e-9) / 3600:.1f} h"
            )

    flush()
    df = pl.concat(frames, how="vertical") if frames else pl.DataFrame()
    logger.info(
        f"done: {df.height:,} rows over {total:,} sessions "
        f"({fetched} fetched, {cached} cached, {failed} with no data) "
        f"in {(time.time() - t0) / 60:.1f} min"
    )
    if df.is_empty():
        raise SystemExit("no bhavcopy rows fetched")

    eq = df.filter(pl.col("series").is_in(["EQ", "BE"]))
    print(f"\n{'metric':<44}{'value':>22}")
    print("-" * 66)
    print(f"{'rows':<44}{df.height:>22,}")
    print(f"{'EQ/BE rows':<44}{eq.height:>22,}")
    print(f"{'distinct symbols (EQ/BE)':<44}{eq['symbol'].n_unique():>22,}")
    print(f"{'distinct ISINs':<44}{eq['isin'].n_unique():>22,}")
    print(f"{'sessions':<44}{df['date'].n_unique():>22,}")
    print(f"{'date range':<44}"
          f"{str(df['date'].min()) + '..' + str(df['date'].max()):>22}")
    for src in ("classic", "sec"):
        n = int((df["source"] == src).sum())
        print(f"{'  from ' + src:<44}{n:>22,}")
    print("-" * 66)
    print("\nFor scale: the current universe is 504 names, and this is every "
          "security\nthat traded — including the ones that later stopped.")


if __name__ == "__main__":
    main()
