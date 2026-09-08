#!/usr/bin/env python
"""Where does a rank-IC lift go when the allocator only buys the top K?

The fundamental blend raises pooled 20d rank IC from +0.0439 to +0.0517 (+18%)
and yet every arm of the allocator grid came out flat or slightly worse. Those
two facts are not in conflict, and this measures why.

Rank IC scores agreement over the WHOLE cross-section — 504 names. A long-only
allocator taking K=20 sees only the extreme top of it, and is completely blind
to how the other 484 are ordered. A signal can therefore get materially better
at ranking the middle, lift IC, and leave the top 20 almost unchanged.

So this reports, per window and per K, the plain mean forward return of the K
names each signal actually selects — no costs, no tax, no rebalancing rules,
nothing between the ranking and the outcome:

    uv run python scripts/topk_diagnostic.py --signals r4_v2,r4_v2_fund
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
    ap.add_argument("--signals", default="r4_v2,r4_v2_fund,r4_v2_fundall")
    ap.add_argument("--signal-root", type=Path, default=Path("data/signal"))
    ap.add_argument("--panel", type=Path, default=Path("data/panels_kite/full.parquet"))
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--k-grid", default="20,30,50,100")
    ap.add_argument("--figures", type=Path,
                    default=Path("data/ext/fundamentals.parquet"))
    args = ap.parse_args()

    from trader.data.universe import active_tickers
    from trader.training.supervised import build_panel_tensors, t_critical_95
    from trader.training.walk_forward import compute_windows

    tickers = active_tickers()
    h = args.horizon
    ks = [int(k) for k in args.k_grid.split(",")]
    tens = build_panel_tensors(pl.read_parquet(args.panel), tickers,
                               ["log_return_1d"], (h,))
    p_idx = {d: i for i, d in enumerate(tens.dates)}
    t_idx = {t: i for i, t in enumerate(tickers)}
    windows = compute_windows(
        data_start=date(2010, 1, 1), data_end=date(2024, 12, 31),
        train_years=5, val_months=12, test_months=12, purge_months=3,
        n_windows=12, step_months=12,
    )

    grids: dict[str, np.ndarray] = {}
    for tag in args.signals.split(","):
        g = np.full((len(tens.dates), len(tickers)), np.nan)
        df = pl.read_parquet(args.signal_root / tag / "predictions.parquet")
        col = f"r_hat_{h}d"
        for row in df.iter_rows(named=True):
            i, j = p_idx.get(row["date"]), t_idx.get(row["ticker"])
            if i is not None and j is not None and row[col] is not None:
                g[i, j] = float(row[col])
        grids[tag] = g
        logger.info(f"{tag}: {np.isfinite(g).mean():.1%} of the grid populated")

    fwd = tens.fwd_raw[h]
    base = args.signals.split(",")[0]
    for k in ks:
        print(f"\nMean {h}-day forward return of the top {k} names, by window")
        print(f"{'window':<8}" + "".join(f"{t:>16}" for t in grids)
              + f"{'delta vs ' + base:>22}")
        print("-" * (8 + 16 * len(grids) + 22))
        per: dict[str, list[float]] = {t: [] for t in grids}
        for w in windows:
            te = [i for d, i in p_idx.items() if w.test_start <= d <= w.test_end]
            if len(te) < 30:
                continue
            got: dict[str, float] = {}
            for tag, g in grids.items():
                vals: list[float] = []
                for i in te:
                    v = np.isfinite(g[i]) & np.isfinite(fwd[i]) & tens.mask[i]
                    n = int(v.sum())
                    if n < k:
                        continue
                    idx = np.flatnonzero(v)
                    top = idx[np.argsort(g[i][idx])[-k:]]
                    vals.append(float(fwd[i][top].mean()))
                if vals:
                    got[tag] = float(np.mean(vals))
                    per[tag].append(got[tag])
            if len(got) == len(grids):
                d = got[list(grids)[-1]] - got[base]
                print(f"{w.name:<8}" + "".join(f"{got[t]:>16.5f}" for t in grids)
                      + f"{d:>+22.5f}")
        print("-" * (8 + 16 * len(grids) + 22))
        means = {t: float(np.mean(v)) for t, v in per.items() if v}
        print(f"{'mean':<8}" + "".join(f"{means.get(t, float('nan')):>16.5f}" for t in grids))
        # A paired test over windows: the arms see the same days, so the
        # difference per window is the right unit, not the levels.
        for tag in grids:
            if tag == base or len(per[tag]) != len(per[base]) or len(per[tag]) < 3:
                continue
            d = np.array(per[tag]) - np.array(per[base])
            t = d.mean() / (d.std(ddof=1) / np.sqrt(d.size)) if d.std(ddof=1) > 0 else np.nan
            crit = t_critical_95(d.size - 1)
            verdict = "SIGNIFICANT" if abs(t) > crit else f"ns (need |t|>{crit:.2f})"
            print(f"  {tag} - {base}: mean {d.mean():+.5f} per {h}d, "
                  f"t {t:+.2f}, {int((d > 0).sum())}/{d.size} windows up, {verdict}")

    print(f"\nRank IC scores all {len(tickers)} names. A long-only book of K "
          f"sees only the top {ks[0]}.")
    print("If the deltas above are ~zero while IC rose, the extra information "
          "is real and lands outside the part of the ranking that gets bought.")

    # ── the filter shape ────────────────────────────────────────────────────
    # Re-ranking the whole covered subset spreads the fundamental information
    # over 500 names, most of which are never bought. A FILTER concentrates it
    # exactly where the allocator looks: take R4's top N, then keep the K of
    # those with the best fundamentals. If earnings change carries anything
    # about which good-momentum names do best, this is the shape that would
    # show it.
    from trader.data.fundamental_features import (
        attach_earnings_yield,
        build_quarterly_features,
        company_score,
    )

    feats = attach_earnings_yield(
        build_quarterly_features(pl.read_parquet(args.figures)),
        pl.read_parquet(args.panel).select(["ticker", "date", "close"]),
    )
    fscore = company_score(feats, list(tens.dates), tickers)

    g = grids[base]
    for k in ks[:2]:
        for pool in (2 * k, 3 * k):
            print(f"\nFILTER: R4 top {pool} -> best {k} by fundamentals "
                  f"(vs R4 top {k})")
            print(f"{'window':<8}{'R4 top' + str(k):>16}{'filtered':>16}"
                  f"{'delta':>14}{'names swapped':>16}")
            print("-" * 70)
            d_all, r_all, f_all = [], [], []
            for w in windows:
                te = [i for d, i in p_idx.items() if w.test_start <= d <= w.test_end]
                if len(te) < 30:
                    continue
                rr, ff, sw = [], [], []
                for i in te:
                    v = np.isfinite(g[i]) & np.isfinite(fwd[i]) & tens.mask[i]
                    if int(v.sum()) < pool:
                        continue
                    idx = np.flatnonzero(v)
                    order = idx[np.argsort(g[i][idx])]
                    plain, cand = order[-k:], order[-pool:]
                    fs = fscore[i][cand]
                    if np.isfinite(fs).sum() < k:
                        continue      # not enough filings to filter on
                    ranked = cand[np.argsort(np.where(np.isfinite(fs), fs, -np.inf))]
                    picked = ranked[-k:]
                    rr.append(float(fwd[i][plain].mean()))
                    ff.append(float(fwd[i][picked].mean()))
                    sw.append(k - len(set(plain.tolist()) & set(picked.tolist())))
                if rr:
                    r_all.append(float(np.mean(rr)))
                    f_all.append(float(np.mean(ff)))
                    d_all.append(f_all[-1] - r_all[-1])
                    print(f"{w.name:<8}{r_all[-1]:>16.5f}{f_all[-1]:>16.5f}"
                          f"{d_all[-1]:>+14.5f}{np.mean(sw):>16.1f}")
            print("-" * 70)
            if len(d_all) >= 3:
                d = np.array(d_all)
                t = (d.mean() / (d.std(ddof=1) / np.sqrt(d.size))
                     if d.std(ddof=1) > 0 else float("nan"))
                crit = t_critical_95(d.size - 1)
                verdict = ("SIGNIFICANT" if abs(t) > crit
                           else f"ns (need |t|>{crit:.2f})")
                print(f"  mean {d.mean():+.5f} per {h}d, t {t:+.2f}, "
                      f"{int((d > 0).sum())}/{d.size} windows up, {verdict}")


if __name__ == "__main__":
    main()
