#!/usr/bin/env python
"""Fetch and parse every XBRL results document named by the results index.

`scripts/fetch_fundamentals.py` produced the index — one row per company-quarter
with the date it became public and a link to its XBRL. This turns those links
into figures.

    uv run python scripts/fetch_xbrl_figures.py            # everything
    uv run python scripts/fetch_xbrl_figures.py --limit 20 # a probe

~21,800 documents at one request a second is roughly six hours, so this is
built to be interrupted:

* every document is cached on disk by URL, so a re-run makes no network call for
  anything already fetched — the network is the expensive part, not the parse;
* parsed rows are flushed to the output parquet every ``--flush-every``
  documents, together with a progress marker, so a kill loses at most that many;
* a document that will not parse is marked ``.bad`` and skipped, never retried
  in a loop and never replayed from cache as if it were data. That is the same
  failure the bhavdata backfill hit when NSE served a ZIP at a .csv URL and one
  bad day aborted 1,613.

The output carries ``visible_from`` from the index, so a consumer joining these
figures onto a panel cannot accidentally use ``period_to`` — the lookahead this
whole line of work exists to prevent.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import polars as pl
from loguru import logger


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path, default=Path("data/ext/results.parquet"))
    ap.add_argument("--out", type=Path, default=Path("data/ext/fundamentals.parquet"))
    ap.add_argument("--cache-root", type=Path, default=Path("data/raw/nse/xbrl"))
    ap.add_argument("--limit", type=int, default=0, help="first N documents only")
    ap.add_argument("--flush-every", type=int, default=200)
    ap.add_argument("--sleep", type=float, default=1.0, help="seconds between requests")
    ap.add_argument("--restart", action="store_true", help="ignore any checkpoint")
    ap.add_argument("--quarterly-only", action="store_true", default=True)
    args = ap.parse_args()

    from trader.data.sources.nse_flows import NSEClient, _FileCache
    from trader.data.sources.nse_fundamentals import (
        XBRLParseError,
        parse_results_xbrl,
        quarterly_only,
    )

    index = pl.read_parquet(args.index).filter(pl.col("xbrl_url").is_not_null())
    index = index.sort(["ticker", "period_to"])
    if args.limit:
        index = index.head(args.limit)
    total = index.height

    cache = _FileCache(args.cache_root)
    client = NSEClient()
    prog_path = args.out.with_suffix(".progress.json")
    done_urls: set[str] = set()
    rows: list[dict[str, object]] = []

    if args.out.exists() and prog_path.exists() and not args.restart:
        prev = pl.read_parquet(args.out)
        rows = prev.to_dicts()
        done_urls = set(json.loads(prog_path.read_text()).get("done_urls", []))
        logger.info(
            f"resuming: {len(rows):,} rows already parsed, "
            f"{len(done_urls):,} documents accounted for"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)

    def flush() -> None:
        if rows:
            pl.DataFrame(rows).write_parquet(args.out)
        prog_path.write_text(
            json.dumps({"done_urls": sorted(done_urls), "rows": len(rows)}) + "\n"
        )

    t0 = time.time()
    fetched = skipped_cached = failed = 0
    for i, rec in enumerate(index.iter_rows(named=True), start=1):
        url = str(rec["xbrl_url"])
        if url in done_urls:
            continue
        name = url.rsplit("/", 1)[-1]
        if cache.is_bad(name):
            done_urls.add(url)
            failed += 1
            continue

        text = cache.read(name)
        if text is None:
            try:
                text = client.get_text(url)
            except Exception as exc:
                logger.warning(f"{name}: fetch failed, {type(exc).__name__}: {exc}")
                done_urls.add(url)
                failed += 1
                continue
            cache.write(name, text, url=url)
            fetched += 1
            time.sleep(args.sleep)
        else:
            skipped_cached += 1

        try:
            parsed = parse_results_xbrl(text, name=name)
        except XBRLParseError as exc:
            cache.mark_bad(name, str(exc))
            logger.warning(f"{name}: unparseable, marked bad — {exc}")
            done_urls.add(url)
            failed += 1
            continue

        if args.quarterly_only:
            parsed = quarterly_only(parsed)
        for row in parsed.iter_rows(named=True):
            # Carry the index's visibility date onto every figure, so a
            # downstream join literally cannot reach for `period_to`.
            row["visible_from"] = rec["visible_from"]
            row["broadcast_ts"] = rec["broadcast_ts"]
            row["xbrl_url"] = url
            rows.append(row)
        done_urls.add(url)

        if i % args.flush_every == 0:
            flush()
            el = time.time() - t0
            rate = max(fetched, 1) / el
            left = total - i
            logger.info(
                f"  {i}/{total}  rows={len(rows):,}  fetched={fetched} "
                f"cached={skipped_cached} failed={failed}  "
                f"eta {left / max(rate, 1e-9) / 3600:.1f} h"
            )

    flush()
    logger.info(
        f"done: {len(rows):,} figure rows from {total:,} documents "
        f"({fetched} fetched, {skipped_cached} from cache, {failed} failed) "
        f"in {(time.time() - t0) / 60:.1f} min"
    )
    if rows:
        df = pl.DataFrame(rows)
        bad = df.filter(pl.col("visible_from") <= pl.col("period_to"))
        if bad.height:
            raise SystemExit(
                f"{bad.height} figure row(s) visible on or before their period "
                "end — that is lookahead, refusing to stand behind this output"
            )
        logger.info(
            f"{df['ticker'].n_unique()} tickers, "
            f"{df['period_to'].min()}..{df['period_to'].max()}; "
            "every row visible strictly after its period ends"
        )


if __name__ == "__main__":
    main()
