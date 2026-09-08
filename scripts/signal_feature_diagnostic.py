"""What is each input feature worth on its own, next to the trained model?

    uv run python scripts/signal_feature_diagnostic.py --tag r4_pit_long

Three things, in order of how much they have cost this project when skipped:

1. **Liveness.** `beta_nifty_60d` was once constant 1.0 across all 276,005
   tradeable training rows, and the stored stats recorded mean 0.0 for it, so a
   dead channel acted as a fixed bias into every convolution (`CLAUDE.md`).
   Null fraction, spread and distinct-value count, per feature.

2. **Univariate rank IC.** Each raw feature, ranked within each date's
   tradeable cross-section, against the forward return — the same statistic the
   R4 gate scores the model with.  A feature that beats the model on its own is
   not a feature the model is using.

3. **The same comparison on the model's own OOS rows**, so neither side is
   helped by covering different names or dates.

This is what found the cross-sectional-input defect: on `r4_pit_long`'s OOS
rows at 20d, blocked by year, a plain rank of `realized_vol_60d` scored +0.0535
against the trained 15-feature model's +0.0275, winning 12 of 14 years.  See
`supervised.cross_sectional_normalise`.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import polars as pl

from trader.data.features import FEATURE_COLS

HORIZONS = (5, 20)


def _t(x: np.ndarray) -> float:
    sd = float(np.std(x, ddof=1))
    return float("nan") if sd == 0 or len(x) < 2 else float(np.mean(x)) / (sd / math.sqrt(len(x)))


def _forward(df: pl.DataFrame) -> pl.DataFrame:
    df = df.sort(["ticker", "date"])
    for h in HORIZONS:
        df = df.with_columns(
            ((pl.col("adj_close").shift(-h) / pl.col("adj_close")).log())
            .over("ticker").alias(f"fwd_{h}")
        )
    return df


def liveness(df: pl.DataFrame, feats: list[str]) -> None:
    print(f"\n{'=' * 70}\nFEATURE LIVENESS (tradeable rows)\n{'=' * 70}")
    tr = df.filter(pl.col("is_tradeable"))
    print(f"{'feature':<22}{'null%':>8}{'std':>14}{'n_uniq':>10}  status")
    for f in feats:
        c = tr[f]
        nullpct = 100.0 * c.null_count() / max(len(c), 1)
        s, nu = c.std(), c.n_unique()
        dead = s is None or s == 0 or nu <= 2 or nullpct > 50
        print(f"{f:<22}{nullpct:>7.2f}%"
              f"{(s if s is not None else float('nan')):>14.6g}{nu:>10,}"
              f"  {'*** DEAD — do not trust a run using it' if dead else 'ok'}")


def rank_ic(df: pl.DataFrame, cols: list[str], h: int, block: str) -> pl.DataFrame:
    d = df.filter(pl.col("is_tradeable") & pl.col(f"fwd_{h}").is_not_null())
    aggs = [pl.corr(c, f"fwd_{h}", method="spearman").alias(c) for c in cols]
    per = (d.group_by("date").agg(*aggs, pl.len().alias("n"))
             .filter(pl.col("n") >= 10).sort("date"))
    unit = pl.col("date").dt.year() if block == "year" else pl.col("date").dt.strftime("%Y-%m")
    return per.with_columns(unit.alias("blk"))


def report(per: pl.DataFrame, cols: list[str], h: int, block: str, ref: str | None) -> None:
    print(f"\n  ── horizon {h}d — {per.height:,} dates, blocked by {block} ──")
    print(f"    {'signal':<24}{'mean IC':>10}{'t(blk)':>9}{'blocks up':>11}")
    agg = per.group_by("blk").agg([pl.col(c).mean().alias(c) for c in cols]).sort("blk")
    vals = {}
    for c in cols:
        v = agg[c].drop_nulls().to_numpy()
        if len(v) < 2:
            continue
        vals[c] = v
        # A negative IC is a usable signal with the sign flipped; show that.
        flip = " (short side)" if float(np.mean(v)) < 0 else ""
        print(f"    {c:<24}{float(np.mean(v)):>+10.4f}{_t(v):>9.2f}"
              f"{f'{int((v > 0).sum())}/{len(v)}':>11}{flip}")
    if ref and ref in vals:
        best = max((c for c in vals if c != ref), key=lambda c: abs(float(np.mean(vals[c]))))
        m_ref, m_best = float(np.mean(vals[ref])), abs(float(np.mean(vals[best])))
        gap = m_best - m_ref
        verdict = f"FEATURE AHEAD by {gap:+.4f}" if gap > 0 else "model ahead"
        print(f"\n    strongest single feature: |{best}| = {m_best:+.4f}")
        print(f"    model ({ref}) = {m_ref:+.4f}  ->  {verdict}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="r4_pit_long", help="signal artefact to compare against")
    ap.add_argument("--panel", default="data/panels_bhav/oos_r4_pit_long.parquet")
    ap.add_argument("--block", default="year", choices=("year", "month"))
    a = ap.parse_args()

    feats = [c for c in FEATURE_COLS]
    df = _forward(
        pl.scan_parquet(a.panel)
        .select(["date", "ticker", "adj_close", "is_tradeable", *feats])
        .collect()
    )
    print(f"panel {a.panel}: {df.height:,} rows, {df['date'].n_unique():,} dates, "
          f"{df['ticker'].n_unique():,} tickers")
    liveness(df, feats)

    print(f"\n{'=' * 70}\nUNIVARIATE RANK IC — whole panel\n{'=' * 70}")
    for h in HORIZONS:
        report(rank_ic(df, feats, h, a.block), feats, h, a.block, ref=None)

    pred_path = Path("data/signal") / a.tag / "predictions.parquet"
    if not pred_path.exists():
        print(f"\nno predictions at {pred_path} — skipping the head-to-head")
        return
    pred = pl.read_parquet(pred_path)
    joined = pred.join(df, on=["date", "ticker"], how="inner")
    cols = [f"r_hat_{h}d" for h in HORIZONS] + feats
    print(f"\n{'=' * 70}\nSAME ROWS AS THE MODEL — {a.tag}, {joined.height:,} OOS rows\n{'=' * 70}")
    for h in HORIZONS:
        report(rank_ic(joined, cols, h, a.block), cols, h, a.block, ref=f"r_hat_{h}d")


if __name__ == "__main__":
    main()
