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
from datetime import date, timedelta
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
    ap.add_argument("--calendar", choices=("panel", "weekdays"), default="panel",
                    help="where sessions come from. 'panel': the reference panel's "
                         "dates -- cannot see past its last date, so a forward loop "
                         "finds nothing to fetch. 'weekdays': every weekday in "
                         "[from, to]; a 404 is a holiday or an unpublished day and "
                         "is never marked done, so the next run retries it.")
    args = ap.parse_args()

    from trader.data.sources.nse_bhavcopy import (
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
    if args.calendar == "weekdays":
        hi = date.fromisoformat(args.end) if args.end else date.today()
        sessions, d = [], lo
        while d <= hi:
            if d.weekday() < 5:
                sessions.append(d)
            d += timedelta(days=1)
    else:
        hi = date.fromisoformat(args.end) if args.end else date(2100, 1, 1)
        sessions = [
            d for d in sorted(
                pl.read_parquet(args.panel, columns=["date"])["date"].unique().to_list()
            )
            if lo <= d <= hi
        ]
    if not sessions:
        raise SystemExit(
            f"no sessions in [{lo}, {hi}] from calendar={args.calendar!r}; "
            f"for a span past the reference panel use --calendar weekdays"
        )
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
        # Attempted for EVERY session, not gated on CLASSIC_VERIFIED_FROM. That
        # constant records what has been verified, and using it as a floor made
        # a span extension a silent no-op: 1,239 sessions from 2005-2010 skipped
        # the archive entirely and were logged as "no data". A doomed request
        # for a genuinely absent date costs one 404; a missing five years costs
        # the experiment.
        if True:
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
            # NOT marked done. A day that yielded nothing may have failed for a
            # reason that is fixed later -- a bad guard, a transient 5xx, a
            # layout the parser did not yet handle -- and marking it done makes
            # that failure PERMANENT across every future run. Exactly that
            # turned a span extension into a no-op twice: the first run skipped
            # 2005-2010 because of a date floor, recorded all 1,239 days as
            # done, and the second run then skipped them again in 0.0 minutes
            # having fixed the floor.
            #
            # A genuinely absent date costs two requests per run to rediscover.
            # A permanently-lost year costs the experiment. Bodies that are
            # present but unparseable are handled by `.bad` markers, which ARE
            # permanent and are the right tool for that case.
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
    # Did the run actually deliver the span it was asked for? `failed` alone
    # cannot answer that: a resume where every date is already marked done
    # reports 0 fetched, 0 cached and 0 failed, and looks like a success.
    got_lo = df["date"].min()
    if got_lo is not None and (got_lo - lo).days > 30:
        logger.error(
            f"asked for {lo} onward and the output starts {got_lo} — "
            f"{(got_lo - lo).days} days short. The archive did not serve the "
            "earlier span, or the checkpoint skipped it. Refusing to report a "
            "wider span than was fetched, because everything downstream would "
            "then be rebuilt on the same data and look fine."
        )
        raise SystemExit(1)
    if args.calendar == "weekdays" and failed:
        # Every weekday was requested, so a 404 is a holiday or a session NSE
        # has not published yet. The forward loop asks for today at 21:00 IST
        # and finds nothing on a day the archive is late: expected, not a
        # defect. The caller reads the data's end date and decides.
        logger.info(f"{failed:,} of {total:,} requested weekday(s) had no file "
                    "(holiday or not yet published)")
    elif failed > total * 0.25:
        logger.error(
            f"{failed:,} of {total:,} sessions ({failed / max(total, 1):.0%}) "
            "yielded no data. A span extension that silently adds nothing looks "
            "exactly like this: the run reports success, the date range does not "
            "move, and everything downstream is rebuilt on the same data. Check "
            "the archive actually serves the requested dates before trusting a "
            "wider span."
        )
        raise SystemExit(1)

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
