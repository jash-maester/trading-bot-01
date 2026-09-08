#!/usr/bin/env python
"""Does the fundamental score say anything the R4 signal does not already know?

`scripts/fundamental_ic.py` establishes that earnings CHANGE carries rank IC on
its own. That is necessary but not sufficient. A company whose profit jumped
usually had a price jump too, and R4 is a momentum-shaped model trained on price
— so the fundamental score could be a noisier restatement of something R4
already reads off the tape. Adding it would then cost coverage and add nothing.

This measures the incremental part directly:

* **corr** — cross-sectional Spearman between the fundamental score and R4's
  own prediction. Near zero means they are looking at different things.
* **residual IC** — regress the fundamental score on R4 each day, cross
  sectionally, and score the residual. This is the part of the fundamental
  score that R4 demonstrably does not contain, and it is the only part worth
  paying coverage for.
* **combined IC** — the equal-weighted rank blend, against R4 alone, on the
  SAME days and the SAME names. Restricting R4 to the covered subset matters:
  scoring a blend on 20% of the panel against R4 on 100% compares two different
  universes and would flatter whichever has the easier one.

    uv run python scripts/fundamental_orthogonality.py
"""
from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
from loguru import logger
from scipy.stats import rankdata


def _rank01(x: np.ndarray) -> np.ndarray:
    """Cross-sectional rank in [0, 1], NaN-preserving."""
    out = np.full_like(x, np.nan, dtype=np.float64)
    v = np.isfinite(x)
    if v.sum() < 2:
        return out
    out[v] = (rankdata(x[v]) - 1.0) / (v.sum() - 1.0)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--figures", type=Path, default=Path("data/ext/fundamentals.parquet"))
    ap.add_argument("--panel", type=Path, default=Path("data/panels_kite/full.parquet"))
    ap.add_argument("--signal", type=Path,
                    default=Path("data/signal/r4_v2/predictions.parquet"))
    ap.add_argument("--horizons", default="5,20")
    ap.add_argument("--min-names", type=int, default=50)
    args = ap.parse_args()

    from trader.data.fundamental_features import (
        as_of_panel,
        attach_earnings_yield,
        build_quarterly_features,
    )
    from trader.data.universe import active_tickers
    from trader.training.supervised import build_panel_tensors, daily_scores, t_critical_95
    from trader.training.walk_forward import compute_windows

    # The three features that survived the standalone IC pass, 6/6 windows
    # positive at both horizons. The dead levels are deliberately not here.
    SCORE_COLS = ("f_profit_growth_yoy", "f_eps_growth_yoy", "f_net_margin_change_yoy")

    horizons = tuple(int(h) for h in args.horizons.split(","))
    tickers = active_tickers()
    panel = pl.read_parquet(args.panel)
    feats = attach_earnings_yield(
        build_quarterly_features(pl.read_parquet(args.figures)),
        panel.select(["ticker", "date", "close"]),
    )

    tens = build_panel_tensors(panel, tickers, ["log_return_1d"], horizons)
    dates, T, N = tens.dates, len(tens.dates), len(tickers)
    grids = as_of_panel(feats, list(dates), tickers, SCORE_COLS)

    # The fundamental score: mean of the cross-sectional ranks of the three
    # surviving features. Ranks, not raw values, because growth rates have
    # unbounded tails and one recovering-from-a-loss name would otherwise
    # dominate the average.
    parts = np.stack([np.stack([_rank01(grids[c][t]) for t in range(T)]) for c in SCORE_COLS])
    with np.errstate(invalid="ignore"):
        # All-NaN columns are the norm here — most names have not filed on
        # most days — so an empty-slice mean is the expected case, not a fault.
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            fscore = np.nanmean(parts, axis=0)
    logger.info(f"fundamental score: {np.isfinite(fscore).mean():.1%} of the grid populated")

    sig = pl.read_parquet(args.signal)
    d_idx = {d: i for i, d in enumerate(dates)}
    t_idx = {t: i for i, t in enumerate(tickers)}
    r4 = np.full((T, N), np.nan, dtype=np.float64)
    col = "prediction" if "prediction" in sig.columns else sig.columns[-1]
    for row in sig.iter_rows(named=True):
        i, j = d_idx.get(row["date"]), t_idx.get(row["ticker"])
        if i is not None and j is not None:
            r4[i, j] = float(row[col])
    logger.info(f"R4 signal '{col}': {np.isfinite(r4).mean():.1%} of the grid populated")

    # Both must exist for a fair comparison, so everything below runs on the
    # intersection: days and names where the fundamental score AND R4 AND a
    # label all exist.
    windows = compute_windows(
        data_start=date(2010, 1, 1), data_end=date(2024, 12, 31),
        train_years=5, val_months=12, test_months=12, purge_months=3,
        n_windows=12, step_months=12,
    )

    resid = np.full((T, N), np.nan, dtype=np.float64)
    combo = np.full((T, N), np.nan, dtype=np.float64)
    r4_on_covered = np.full((T, N), np.nan, dtype=np.float64)
    corrs: list[float] = []
    for t in range(T):
        both = np.isfinite(fscore[t]) & np.isfinite(r4[t]) & tens.mask[t]
        if both.sum() < args.min_names:
            continue
        f = _rank01(np.where(both, fscore[t], np.nan))
        s = _rank01(np.where(both, r4[t], np.nan))
        v = np.isfinite(f) & np.isfinite(s)
        fv, sv = f[v], s[v]
        corrs.append(float(np.corrcoef(fv, sv)[0, 1]))
        # Residualise the fundamental rank on the R4 rank: least squares, one
        # slope per day. What is left is orthogonal to R4 by construction.
        beta = np.polyfit(sv, fv, 1)
        resid[t, v] = fv - np.polyval(beta, sv)
        combo[t, v] = 0.5 * fv + 0.5 * sv
        r4_on_covered[t, v] = sv

    corr = np.array(corrs)
    print(f"\nDays with >= {args.min_names} names carrying both scores: {corr.size:,}")
    print(f"Cross-sectional corr(fundamental, R4): mean {corr.mean():+.4f}  "
          f"sd {corr.std():.4f}  |mean| < 0.1 means they are near-independent")

    def score(name: str, grid: np.ndarray) -> None:
        print(f"\n{name}")
        for h in horizons:
            per: list[float] = []
            marks: list[str] = []
            for w in windows:
                te = np.array([i for d, i in d_idx.items()
                               if w.test_start <= d <= w.test_end], dtype=np.int64)
                if te.size < 30:
                    continue
                d = daily_scores(grid[te].astype(np.float32),
                                 tens.targets[h][te], tens.fwd_raw[h][te])
                if d.ic.size >= 30:
                    per.append(float(np.nanmean(d.ic)))
                    marks.append(f"{w.name}:{np.nanmean(d.ic):+.3f}")
            if len(per) < 3:
                print(f"  h={h:<3} too few scorable windows")
                continue
            a = np.array(per)
            t = a.mean() / (a.std(ddof=1) / np.sqrt(a.size))
            crit = t_critical_95(a.size - 1)
            flag = "SIGNIFICANT" if abs(t) > crit else f"ns (need |t|>{crit:.2f})"
            print(f"  h={h:<3} IC {a.mean():+.4f}  t {t:+.2f}  {int((a>0).sum())}/{a.size} "
                  f"positive  {flag}")
            print(f"       {' '.join(marks)}")

    score("R4 alone, restricted to names with a filing (the fair baseline)", r4_on_covered)
    score("Fundamental score alone, same names", np.where(np.isfinite(r4_on_covered),
                                                          fscore, np.nan))
    score("Fundamental score RESIDUALISED on R4 (the incremental part)", resid)
    score("50/50 rank blend of R4 and fundamentals", combo)

    print("\nRead it this way: if the residual IC is flat, the fundamental score is")
    print("a restatement of what R4 already reads off price, and adding it buys")
    print("nothing but a 4x cut in coverage. If the residual holds up, the blend")
    print("should beat R4 alone on these same names — and if it does not, the")
    print("blend weight is wrong, not the feature.")


if __name__ == "__main__":
    main()
