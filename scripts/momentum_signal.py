"""12-1 momentum signal for the P6 live book (audit/P6_MOMENTUM_15.md).

    uv run python scripts/momentum_signal.py            # nightly, after P2's panel build

Score for ticker i on date d = log(adj_close[d-21] / adj_close[d-252]), in
trading sessions of the panel calendar: the return over the past year, skipping
the most recent month (Jegadeesh & Titman 1993; the IIMA Indian momentum factor
uses the same 12-1 construction). No model, no fitting, no parameters beyond
the two lags, which are fixed by the pre-registration.

A score is emitted only where both prices are positive and the row is
tradeable on d. Written in the same shape the live executor reads
(date, ticker, r_hat_5d, r_hat_20d), with the score in both columns: only the
ranking matters to the allocator, and it reads r_hat_20d.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

LONG, SKIP = 252, 21


def build(panel: Path, start: str | None) -> pl.DataFrame:
    df = (pl.scan_parquet(panel)
          .select("date", "ticker", "adj_close", "is_tradeable")
          .collect()
          .sort("ticker", "date"))
    df = df.with_columns(
        pl.col("adj_close").shift(SKIP).over("ticker").alias("p_skip"),
        pl.col("adj_close").shift(LONG).over("ticker").alias("p_long"),
    )
    ok = ((pl.col("p_skip") > 0) & (pl.col("p_long") > 0)
          & pl.col("is_tradeable").cast(pl.Boolean))
    out = (df.filter(ok)
           .select("date", "ticker",
                   (pl.col("p_skip") / pl.col("p_long")).log().cast(pl.Float32).alias("r_hat_20d")))
    if start:
        out = out.filter(pl.col("date") >= pl.lit(start).str.to_date())
    return out.with_columns(pl.col("r_hat_20d").alias("r_hat_5d")).select(
        "date", "ticker", "r_hat_5d", "r_hat_20d").sort("date", "ticker")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="data/panels_forward/full.parquet")
    ap.add_argument("--out", default="data/signal/momentum_12_1/predictions.parquet")
    ap.add_argument("--start", default="2026-09-01")
    a = ap.parse_args()
    out = build(Path(a.panel), a.start)
    p = Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp.parquet")
    out.write_parquet(tmp)
    tmp.replace(p)
    last = out["date"].max()
    n_last = out.filter(pl.col("date") == last).height
    print(f"momentum_12_1: {out.height:,} rows, {out['date'].min()}..{last}, "
          f"{n_last} names on {last}")


if __name__ == "__main__":
    main()
