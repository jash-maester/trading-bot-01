#!/usr/bin/env python
"""What the fundamentals actually cover, before and after a re-parse.

Two coverage holes were fixed on 2026-09-08 and both were silent — an
ampersand symbol returned an empty body, and a banking filing returned a full
row of nulls. Neither showed up in a row count, which is exactly why this
prints per-FIELD coverage and a before/after diff rather than a total.

    uv run python scripts/fundamentals_coverage.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

FIELDS = ("revenue", "net_profit", "pbt", "eps_basic", "paid_up_equity", "face_value")


def _report(name: str, f: pl.DataFrame, universe: set[str]) -> dict[str, float]:
    inu = f.filter(pl.col("ticker").is_in(list(universe)))
    print(f"\n{name}")
    print(f"  rows {f.height:,}  in-universe {inu.height:,}  "
          f"tickers {inu['ticker'].n_unique()} of {len(universe)}")
    out: dict[str, float] = {"rows": float(inu.height),
                             "tickers": float(inu["ticker"].n_unique())}
    for c in FIELDS:
        if c not in inu.columns:
            continue
        n = int(inu[c].is_not_null().sum())
        out[c] = float(n)
        print(f"    {c:<16}{n:>7,} non-null  {n / max(inu.height, 1):>7.1%}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--figures", type=Path, default=Path("data/ext/fundamentals.parquet"))
    ap.add_argument("--before", type=Path,
                    default=Path("data/ext/fundamentals.pre_banking.parquet"))
    args = ap.parse_args()

    from trader.data.universe import active_tickers

    universe = set(active_tickers())
    now = pl.read_parquet(args.figures)
    a = _report(f"AFTER  {args.figures}", now, universe)

    if args.before.exists():
        b = _report(f"BEFORE {args.before}", pl.read_parquet(args.before), universe)
        print(f"\n{'metric':<20}{'before':>12}{'after':>12}{'delta':>12}")
        print("-" * 56)
        for k in ("rows", "tickers", *FIELDS):
            if k in a and k in b:
                print(f"{k:<20}{b[k]:>12,.0f}{a[k]:>12,.0f}{a[k] - b[k]:>+12,.0f}")
        print("-" * 56)

    # The bank check, named explicitly: these were 233 rows of pure null.
    banks = {
        "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "KOTAKBANK.NS", "AXISBANK.NS",
        "AUBANK.NS", "INDUSINDBK.NS", "FEDERALBNK.NS", "BANKBARODA.NS", "PNB.NS",
    }
    fb = now.filter(pl.col("ticker").is_in(list(banks)))
    print(f"\nTen named banks: {fb.height:,} rows, "
          f"{fb['ticker'].n_unique()} tickers")
    for c in ("revenue", "net_profit", "eps_basic"):
        if c in fb.columns:
            n = int(fb[c].is_not_null().sum())
            print(f"    {c:<16}{n:>7,} non-null  {n / max(fb.height, 1):>7.1%}")
    if fb.height and int(fb["net_profit"].is_not_null().sum()) == 0:
        raise SystemExit("banks still carry no figures — the alias map is not firing")

    amp = now.filter(pl.col("ticker").str.contains("&"))
    print(f"\nAmpersand symbols: {amp.height:,} rows, "
          f"{sorted(amp['ticker'].unique().to_list())}")


if __name__ == "__main__":
    main()
