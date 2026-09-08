#!/usr/bin/env python
"""F2/F5/availability — what do fundamental features carry, on their own?

`13_fundamentals_and_news.md` section 3b.5 insists on two checks before any
fundamental feature is added to a model, because a lift measured after adding it
is easy to misread:

1. **standalone rank IC** — what the feature predicts by itself, per horizon;
2. **per-window stability** — a genuine factor shows up broadly across the 8
   walk-forward windows; a memorised set of tickers shows up only in the windows
   containing those tickers, and the mean alone cannot tell them apart.

A third check is specific to this dataset. Fundamental COVERAGE grows over time
here — 20 tickers reported in 2017 against 410 in 2023 — so a feature's
availability is itself correlated with the calendar. The availability arm scores
a pure "do I have a filing for this stock" indicator: if that alone predicts
returns, any lift from the real features is suspect.

    uv run python scripts/fundamental_ic.py

Scores are computed with R4's own `daily_scores`, on the same targets, so these
numbers sit on the same scale as the signal model's 0.039.
"""
from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
from loguru import logger


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--figures", type=Path, default=Path("data/ext/fundamentals.parquet"))
    ap.add_argument("--panel", type=Path, default=Path("data/panels_kite/full.parquet"))
    ap.add_argument("--horizons", default="5,20")
    ap.add_argument("--min-cross-section", type=int, default=10)
    ap.add_argument("--min-names", type=int, default=50,
                    help="drop a window whose median covered cross-section is thinner")
    args = ap.parse_args()

    from trader.data.fundamental_features import (
        CHANGE_COLS,
        LEVEL_COLS,
        as_of_panel,
        attach_earnings_yield,
        build_quarterly_features,
    )
    from trader.data.universe import active_tickers
    from trader.training.supervised import build_panel_tensors, daily_scores
    from trader.training.walk_forward import compute_windows

    horizons = tuple(int(h) for h in args.horizons.split(","))
    tickers = active_tickers()
    panel = pl.read_parquet(args.panel)

    figures = pl.read_parquet(args.figures)
    feats = build_quarterly_features(figures)
    feats = attach_earnings_yield(feats, panel.select(["ticker", "date", "close"]))
    logger.info(f"{feats.height:,} company-quarters, {feats['ticker'].n_unique()} tickers")

    tens = build_panel_tensors(panel, tickers, ["log_return_1d"], horizons)
    dates = tens.dates
    cols = tuple(LEVEL_COLS) + tuple(CHANGE_COLS)
    grids = as_of_panel(feats, list(dates), tickers, cols)

    # The availability control: 1.0 where ANY fundamental is known, else NaN.
    avail = np.zeros((len(dates), len(tickers)), dtype=np.float64)
    for c in cols:
        avail += np.isfinite(grids[c]).astype(np.float64)
    grids["f_availability"] = np.where(avail > 0, 1.0, 0.0)

    windows = compute_windows(
        data_start=date(2010, 1, 1), data_end=date(2024, 12, 31),
        train_years=5, val_months=12, test_months=12, purge_months=3,
        n_windows=12, step_months=12,
    )
    d_idx = {d: i for i, d in enumerate(dates)}

    def span(a: date, b: date) -> np.ndarray:
        return np.array([i for d, i in d_idx.items() if a <= d <= b], dtype=np.int64)

    # How many names does each window actually rank? An IC over 20 names is not
    # the same measurement as an IC over 400, and the early windows here are
    # thin enough that their contribution has to be read with that in mind.
    print(f"\n{'window':<10}{'test span':<26}{'days':>6}{'median names w/ a filing':>26}")
    print("-" * 68)
    usable: list[object] = []
    for w in windows:
        te = span(w.test_start, w.test_end)
        if te.size < 30:
            continue
        n_cov = np.isfinite(grids["f_net_margin"][te]) & tens.mask[te]
        med = int(np.median(n_cov.sum(axis=1)))
        keep = med >= args.min_names
        if keep:
            usable.append(w)
        print(f"{w.name:<10}{str(w.test_start) + '..' + str(w.test_end):<26}"
              f"{te.size:>6}{med:>26}   {'' if keep else 'DROPPED — too thin to rank'}")
    print("-" * 68)
    print(f"Scoring on {len(usable)} of {len(windows)} windows "
          f"(median cross-section >= {args.min_names}).")
    windows = usable

    all_cols = [*cols, "f_availability"]
    print(f"\n{'feature':<26}{'h':>4}{'mean IC':>10}{'t':>8}{'pos':>7}{'cover':>8}  per-window")
    print("-" * 108)
    summary: list[tuple[str, int, float, float, int, float]] = []
    for col in all_cols:
        grid = grids[col]
        for h in horizons:
            per_window: list[float] = []
            names: list[str] = []
            for w in windows:
                te = span(w.test_start, w.test_end)
                if te.size < 30:
                    continue
                # Standardise cross-sectionally each day, exactly as the signal
                # model's targets are, so IC is scale-free.
                pred = grid[te].copy()
                pred[~tens.mask[te]] = np.nan
                d = daily_scores(
                    pred.astype(np.float32), tens.targets[h][te], tens.fwd_raw[h][te],
                    min_cross_section=args.min_cross_section,
                )
                if d.ic.size:
                    per_window.append(float(np.nanmean(d.ic)))
                    names.append(w.name)
            if len(per_window) < 3:
                continue
            a = np.array(per_window)
            t = a.mean() / (a.std(ddof=1) / np.sqrt(a.size)) if a.std(ddof=1) > 0 else np.nan
            cover = float(np.isfinite(grid).mean())
            summary.append((col, h, float(a.mean()), float(t), int((a > 0).sum()), a.size))
            marks = " ".join(f"{n}:{v:+.3f}" for n, v in zip(names, per_window))
            print(f"{col:<26}{h:>4}{a.mean():>10.4f}{t:>8.2f}"
                  f"{int((a > 0).sum()):>4}/{a.size}{cover:>8.1%}  {marks}")
    print("-" * 108)

    from trader.training.supervised import t_critical_95

    strong = [s for s in summary if abs(s[3]) > t_critical_95(7) and abs(s[2]) > 0.005]
    print(f"\nFeatures with |mean IC| > 0.005 AND |t| > {t_critical_95(7):.2f}: "
          f"{len(strong)} of {len(summary)}")
    for col, h, ic, t, pos, nw in sorted(strong, key=lambda s: -abs(s[2])):
        print(f"  {col:<26} h={h:<3} IC {ic:+.4f}  t {t:+.2f}  positive {pos}/{nw}")

    av = [s for s in summary if s[0] == "f_availability"]
    if av:
        worst = max(av, key=lambda s: abs(s[2]))
        print(f"\nAVAILABILITY CONTROL: |IC| {abs(worst[2]):.4f} at h={worst[1]}, "
              f"t {worst[3]:+.2f}")
        print("  If this is comparable to a real feature's IC, that feature is "
              "measuring coverage growth, not fundamentals.")
    print("\nFor scale: the R4 signal model's pooled 5d IC is +0.0392 "
          "(audit/r4_v2/gate.json).")


if __name__ == "__main__":
    main()
