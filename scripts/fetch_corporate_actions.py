#!/usr/bin/env python
"""Fetch NSE's corporate-actions feed, so prices can be adjusted from a source.

`src/trader/data/sources/nse_corporate_actions.py` explains why this replaces
the `prev_close` detector it supersedes: PREVCLOSE is the RAW previous close, so
the ratio built on it was 1.0 by construction and found nothing.

    uv run python scripts/fetch_corporate_actions.py --from 2010 --to 2026

One request per year, cached. The API is date-ranged and refuses very wide
spans, which is why it is not fetched in a single call.

The report at the end is the point: it prints the parsed factor for three
splits whose dates and ratios are public record. An adjustment layer that
cannot reproduce those is not to be trusted, and the last one could not.
"""
from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import polars as pl
from loguru import logger


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/ext/corporate_actions.parquet"))
    ap.add_argument("--cache-root", type=Path, default=Path("data/raw/nse/corpactions"))
    ap.add_argument("--from", dest="lo", type=int, default=2010)
    ap.add_argument("--to", dest="hi", type=int, default=2026)
    ap.add_argument("--sleep", type=float, default=0.5)
    args = ap.parse_args()

    import time

    from trader.data.sources.nse_corporate_actions import (
        CA_URL,
        CorporateActionParseError,
        parse_corporate_actions,
    )
    from trader.data.sources.nse_flows import NSEClient, _FileCache

    cache = _FileCache(args.cache_root)
    client = NSEClient()
    frames: list[pl.DataFrame] = []
    for year in range(args.lo, args.hi + 1):
        name = f"ca_{year}.json"
        text = cache.read(name)
        if text is None:
            url = CA_URL.format(from_date=f"01-01-{year}", to_date=f"31-12-{year}")
            try:
                text = client.get_text(url)
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"{year}: fetch failed, {type(exc).__name__}: {exc}")
                continue
            cache.write(name, text, url=url)
            time.sleep(args.sleep)
        try:
            df = parse_corporate_actions(text, name=name)
        except CorporateActionParseError as exc:
            cache.mark_bad(name, str(exc))
            logger.warning(f"{year}: unparseable, marked bad — {exc}")
            continue
        logger.info(f"{year}: {df.height:,} split/bonus action(s)")
        frames.append(df)

    if not frames:
        raise SystemExit("no corporate actions fetched")
    actions = (
        pl.concat(frames, how="vertical")
        .unique(subset=["ticker", "ex_date", "kind", "price_factor"], keep="first")
        .sort(["ticker", "ex_date"])
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    actions.write_parquet(args.out)

    print(f"\n{'metric':<44}{'value':>22}")
    print("-" * 66)
    print(f"{'actions':<44}{actions.height:>22,}")
    print(f"{'tickers affected':<44}{actions['ticker'].n_unique():>22,}")
    print(f"{'date range':<44}"
          f"{str(actions['ex_date'].min()) + '..' + str(actions['ex_date'].max()):>22}")
    for kind in ("split", "bonus", "consolidation"):
        n = int((actions["kind"] == kind).sum())
        print(f"{'  ' + kind:<44}{n:>22,}")
    print("-" * 66)

    # ── the check the previous implementation never had ──────────────────────
    known = [
        ("NESTLEIND.NS", date(2024, 1, 5), 0.1),
        ("HDFCBANK.NS", date(2019, 9, 19), 0.5),
        ("IRCTC.NS", date(2021, 10, 28), 0.2),
    ]
    print("\nKnown splits, as parsed from the feed:")
    ok = True
    for tk, ex, want in known:
        hit = actions.filter(
            (pl.col("ticker") == tk)
            & (pl.col("ex_date") >= ex - __import__("datetime").timedelta(days=4))
            & (pl.col("ex_date") <= ex + __import__("datetime").timedelta(days=4))
        )
        if hit.is_empty():
            print(f"  {tk:<14} {ex}  NOT FOUND  (expected factor {want})")
            ok = False
            continue
        r = hit.row(0, named=True)
        good = abs(r["price_factor"] - want) < 1e-6
        ok &= good
        print(f"  {tk:<14} {r['ex_date']}  factor {r['price_factor']:.4f}  "
              f"expected {want:.4f}  {'OK' if good else 'MISMATCH'}")
    if not ok:
        raise SystemExit(
            "a known split is missing or wrong. The last adjustment layer failed "
            "silently for want of exactly this check; refusing to write a feed "
            "that cannot reproduce a public record."
        )
    logger.info(f"wrote {args.out}")


if __name__ == "__main__":
    main()
