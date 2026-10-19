"""Book 5 (P8) signal: rising delivery %, z-scored within sector.

    uv run python scripts/delivery_signal.py                         # live: nightly, forward panel
    uv run python scripts/delivery_signal.py --panel data/panels_bhav/oos_deliv.parquet \
        --start 2020-10-15 --end 2024-07-02 --out-tag delivery_oos    # historical test

Pre-registered in audit/P8_DELIVERY_15.md before any result was seen. For
ticker i on date d, EQ-series rows of data/ext/delivery.parquet only:

  rise = mean(delivery_pct, 20 sessions ending d) - mean(delivery_pct, 120 sessions ending d)

needing >= 16 of 20 and >= 100 of 120 non-null values. Sessions are the
stock's own delivery rows. Delivery for d is published after d's close, so it
is known when the d-dated signal trades at d+1, like every other book's
close-based signal. z-scored across tradeable panel names on d within sector
(>= MIN_SECTOR names), otherwise across all names. Higher rise ranks higher.

Output is the shape the allocator reads (date, ticker, r_hat_5d, r_hat_20d,
score in both): only the ranking matters.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

SHORT, LONG, MIN_SHORT, MIN_LONG, MIN_SECTOR = 20, 120, 16, 100, 5
DELIVERY = Path("data/ext/delivery.parquet")


def build(panel: Path, start: str | None, end: str | None) -> pl.DataFrame:
    dv = (pl.scan_parquet(DELIVERY)
          .filter(pl.col("series") == "EQ")
          .select("date", "ticker", "delivery_pct")
          .collect()
          .unique(["date", "ticker"])
          .sort("ticker", "date"))
    x = pl.col("delivery_pct")
    dv = dv.with_columns(
        x.rolling_mean(SHORT, min_samples=MIN_SHORT).over("ticker").alias("m_s"),
        x.rolling_mean(LONG, min_samples=MIN_LONG).over("ticker").alias("m_l"),
    ).with_columns((pl.col("m_s") - pl.col("m_l")).alias("rise"))
    pn = (pl.scan_parquet(panel)
          .select("date", "ticker", "is_tradeable", "sector_id")
          .collect())
    df = (pn.filter(pl.col("is_tradeable").cast(pl.Boolean))
          .join(dv.select("date", "ticker", "rise"), on=["date", "ticker"], how="inner")
          .filter(pl.col("rise").is_finite()))
    if start:
        df = df.filter(pl.col("date") >= pl.lit(start).str.to_date())
    if end:
        df = df.filter(pl.col("date") <= pl.lit(end).str.to_date())
    df = df.with_columns(pl.len().over("date", "sector_id").alias("n_sec"))

    def z(by: list[str]) -> pl.Expr:
        return (pl.col("rise") - pl.col("rise").mean().over(by)) / pl.col("rise").std().over(by)

    score = (pl.when(pl.col("n_sec") >= MIN_SECTOR).then(z(["date", "sector_id"]))
             .otherwise(z(["date"])).fill_nan(0.0).fill_null(0.0).cast(pl.Float32))
    return (df.select("date", "ticker", score.alias("r_hat_20d"))
              .with_columns(pl.col("r_hat_20d").alias("r_hat_5d"))
              .select("date", "ticker", "r_hat_5d", "r_hat_20d")
              .sort("date", "ticker"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="data/panels_forward/full.parquet")
    ap.add_argument("--start", default="2026-09-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--out-tag", default="delivery_rise")
    a = ap.parse_args()
    out = build(Path(a.panel), a.start, a.end)
    d = Path("data/signal") / a.out_tag
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / "predictions.tmp.parquet"
    out.write_parquet(tmp)
    tmp.replace(d / "predictions.parquet")
    (d / "gate.json").write_text(json.dumps({
        "verdict": "NONE",
        "note": "Rule-based delivery signal (audit/P8_DELIVERY_15.md); no model, no IC gate.",
    }, indent=2))
    last = out["date"].max()
    print(f"{a.out_tag}: {out.height:,} rows, {out['date'].min()}..{last}, "
          f"{out.filter(pl.col('date') == last).height} names on {last}")


if __name__ == "__main__":
    main()
