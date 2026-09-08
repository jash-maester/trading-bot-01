"""Compare two signal artefacts on identical rows, in-sample OOS and holdout.

    SIGNAL=r4_pit_xs BASE=r4_pit_long uv run python scripts/xs_head_to_head.py

Both models are scored on exactly the rows where BOTH have a prediction, so
neither is helped by covering a different set of names or dates.  Rank IC is
Spearman within a date's tradeable cross-section — the same statistic the R4
gate uses — and the paired difference is aggregated to calendar years, because
a per-date t over ~3,200 autocorrelated days is not a test of anything.
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import polars as pl

HORIZONS = (5, 20)
PANEL = Path("data/panels_bhav")
SIGNALS = Path("data/signal")


def _with_forward_returns(panel_path: Path) -> pl.DataFrame:
    df = (
        pl.scan_parquet(panel_path)
        .select(["date", "ticker", "adj_close", "is_tradeable"])
        .collect()
        .sort(["ticker", "date"])
    )
    for h in HORIZONS:
        df = df.with_columns(
            ((pl.col("adj_close").shift(-h) / pl.col("adj_close")).log())
            .over("ticker")
            .alias(f"fwd_{h}")
        )
    return df.filter(pl.col("is_tradeable"))


def _preds(tag: str) -> pl.DataFrame | None:
    p = SIGNALS / tag / "predictions.parquet"
    return pl.read_parquet(p) if p.exists() else None


def _t(x: np.ndarray) -> float:
    sd = float(np.std(x, ddof=1))
    return float("nan") if sd == 0 or len(x) < 2 else float(np.mean(x)) / (sd / math.sqrt(len(x)))


def compare(panel_path: Path, new_tag: str, base_tag: str, label: str,
            block: str = "year") -> None:
    new, base = _preds(new_tag), _preds(base_tag)
    print(f"\n{'=' * 74}\n{label}\n  new  = {new_tag}\n  base = {base_tag}\n{'=' * 74}")
    if new is None or base is None:
        print(f"  SKIPPED — missing predictions ({new_tag if new is None else base_tag})")
        return

    pan = _with_forward_returns(panel_path)
    d = (
        new.rename({f"r_hat_{h}d": f"new_{h}" for h in HORIZONS})
        .join(base.rename({f"r_hat_{h}d": f"base_{h}" for h in HORIZONS}),
              on=["date", "ticker"], how="inner")
        .join(pan, on=["date", "ticker"], how="inner")
    )
    if d.height == 0:
        print("  SKIPPED — no overlapping rows")
        return
    print(f"  {d.height:,} shared rows, {d['date'].n_unique():,} dates, "
          f"{d['date'].min()} .. {d['date'].max()}")

    for h in HORIZONS:
        dd = d.filter(pl.col(f"fwd_{h}").is_not_null())
        per = (
            dd.group_by("date")
            .agg(
                new=pl.corr(f"new_{h}", f"fwd_{h}", method="spearman"),
                base=pl.corr(f"base_{h}", f"fwd_{h}", method="spearman"),
                n=pl.len(),
            )
            .filter((pl.col("n") >= 10) & pl.col("new").is_not_null()
                    & pl.col("base").is_not_null())
            .sort("date")
            .with_columns(
                (pl.col("date").dt.year().cast(pl.Utf8) if block == "year"
                 else pl.col("date").dt.strftime("%Y-%m")).alias("y")
            )
        )
        if per.height == 0:
            print(f"\n  horizon {h}d: no scorable dates")
            continue
        yr = per.group_by("y").agg(
            pl.col("new").mean().alias("new"), pl.col("base").mean().alias("base")
        ).sort("y")
        nv, bv = yr["new"].to_numpy(), yr["base"].to_numpy()
        diff = nv - bv
        print(f"\n  ── horizon {h}d — {per.height:,} dates, "
              f"{len(nv)} {block} block(s) ──")
        print(f"    {'':<12}{'mean IC':>10}{f't({block})':>10}"
              f"{block + 's up':>11}")
        print(f"    {'new':<12}{nv.mean():>+10.4f}{_t(nv):>10.2f}"
              f"{f'{(nv > 0).sum()}/{len(nv)}':>11}")
        print(f"    {'base':<12}{bv.mean():>+10.4f}{_t(bv):>10.2f}"
              f"{f'{(bv > 0).sum()}/{len(bv)}':>11}")
        verdict = "new ahead" if diff.mean() > 0 else "base ahead"
        if len(diff) < 4:
            verdict += "  (too few blocks for a t — read the point estimate only)"
        print(f"    {'DIFF':<12}{diff.mean():>+10.4f}{_t(diff):>10.2f}"
              f"{f'{(diff > 0).sum()}/{len(diff)}':>11}   {verdict}")
        print(f"    per {block}: " + "  ".join(
            f"{y}:{v:+.3f}" for y, v in zip(yr["y"].to_list(), diff, strict=True)))


def main() -> None:
    new_tag = os.environ.get("SIGNAL", "r4_pit_xs")
    base_tag = os.environ.get("BASE", "r4_pit_long")
    compare(PANEL / "oos_r4_pit_long.parquet", new_tag, base_tag,
            "IN-SAMPLE WALK-FORWARD OOS (2005-2024, 13 windows)")
    # The base's holdout artefact may be named for either the long or the
    # 8-window fit; take whichever exists rather than guessing.
    base_hold = next(
        (t for t in (f"{base_tag}_holdout", "r4_pit_holdout")
         if (SIGNALS / t / "predictions.parquet").exists()),
        f"{base_tag}_holdout",
    )
    # The holdout spans ~15 months. Blocking it by year would compute a t from
    # two numbers, which is not a statistic; months are the only honest unit
    # there, and they are more autocorrelated, so read that t as generous.
    compare(PANEL / "holdout.parquet", f"{new_tag}_holdout", base_hold,
            "UNSEEN HOLDOUT (2025-04 .. 2026-09)", block="month")
    print("\nRank IC is Spearman within each date's tradeable cross-section,")
    print("identical rows for both models. In-sample blocks by year; the holdout")
    print("is ~15 months long and blocks by month of necessity.")


if __name__ == "__main__":
    main()
