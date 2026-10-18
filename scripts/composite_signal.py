"""Book 4 (P7) signal: three price signals, z-scored within sector, averaged.

    uv run python scripts/composite_signal.py                       # live: nightly, forward panel
    uv run python scripts/composite_signal.py --panel data/panels_bhav/full.parquet \
        --start 2011-07-04 --end 2024-07-02 --out-tag composite3_oos     # historical test

Pre-registered in audit/P7_COMPOSITE_15.md before any result was seen. For
ticker i on date d, in panel sessions:

  momentum  log(adj_close[d-21] / adj_close[d-252])            (12-1, as Book 3)
  low vol   -std(log(adj_close[t]/adj_close[t-1]), t in d-251..d)   (steadier = higher)
  52w high  adj_close[d] / max(adj_close[d-251..d])             (nearer the high = higher)

Each is z-scored across tradeable names on d within the name's sector when the
sector has at least MIN_SECTOR tradeable names with all three signals, else
across all such names; the score is the plain mean of the three z-scores.
A name needs a positive price on every one of the 253 sessions d-252..d. No
fitting, no weights other than 1/3 each.

Output is the shape the allocator reads (date, ticker, r_hat_5d, r_hat_20d,
score in both): only the ranking matters.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

LONG, SKIP, MIN_SECTOR = 252, 21, 5


def build(panel: Path, start: str | None, end: str | None) -> pl.DataFrame:
    df = (pl.scan_parquet(panel)
          .select("date", "ticker", "adj_close", "is_tradeable", "sector_id")
          .collect()
          .sort("ticker", "date"))
    p = pl.col("adj_close")
    pos = (p > 0).cast(pl.Int32)
    lr = (p / p.shift(1)).log()
    df = df.with_columns(
        (p.shift(SKIP) / p.shift(LONG)).log().over("ticker").alias("mom"),
        (-lr.rolling_std(LONG)).over("ticker").alias("lowvol"),
        (p / p.rolling_max(LONG)).over("ticker").alias("hi52"),
        pos.rolling_sum(LONG + 1).over("ticker").alias("n_pos"),
    )
    ok = (pl.col("is_tradeable").cast(pl.Boolean) & (pl.col("n_pos") == LONG + 1)
          & pl.all_horizontal(pl.col(c).is_finite() for c in ("mom", "lowvol", "hi52")))
    df = df.filter(ok)
    if start:
        df = df.filter(pl.col("date") >= pl.lit(start).str.to_date())
    if end:
        df = df.filter(pl.col("date") <= pl.lit(end).str.to_date())
    df = df.with_columns(
        pl.len().over("date", "sector_id").alias("n_sec"))

    def z(c: str, by: list[str]) -> pl.Expr:
        return ((pl.col(c) - pl.col(c).mean().over(by)) / pl.col(c).std().over(by))

    zs = []
    for c in ("mom", "lowvol", "hi52"):
        zs.append(pl.when(pl.col("n_sec") >= MIN_SECTOR)
                  .then(z(c, ["date", "sector_id"]))
                  .otherwise(z(c, ["date"]))
                  .fill_nan(0.0).fill_null(0.0)        # a constant column carries no rank
                  .alias(f"z_{c}"))
    df = df.with_columns(zs)
    score = ((pl.col("z_mom") + pl.col("z_lowvol") + pl.col("z_hi52")) / 3.0).cast(pl.Float32)
    return (df.select("date", "ticker", score.alias("r_hat_20d"))
              .with_columns(pl.col("r_hat_20d").alias("r_hat_5d"))
              .select("date", "ticker", "r_hat_5d", "r_hat_20d")
              .sort("date", "ticker"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="data/panels_forward/full.parquet")
    ap.add_argument("--start", default="2026-09-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--out-tag", default="composite3")
    a = ap.parse_args()
    out = build(Path(a.panel), a.start, a.end)
    d = Path("data/signal") / a.out_tag
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / "predictions.tmp.parquet"
    out.write_parquet(tmp)
    tmp.replace(d / "predictions.parquet")
    (d / "gate.json").write_text(json.dumps({
        "verdict": "NONE",
        "note": "Rule-based composite (audit/P7_COMPOSITE_15.md); no model, no IC gate.",
    }, indent=2))
    last = out["date"].max()
    print(f"{a.out_tag}: {out.height:,} rows, {out['date'].min()}..{last}, "
          f"{out.filter(pl.col('date') == last).height} names on {last}")


if __name__ == "__main__":
    main()
