#!/usr/bin/env python
"""Where the allocator's drawdown actually comes from, and whether it recovered.

`audit/P3_P4_P5_RESULTS.md` reports a single number, -53.0% maximum drawdown,
and a single number cannot distinguish two very different worlds:

  * one market-wide crash that recovered fully, which is beta and which a risk
    overlay would most likely make WORSE by de-risking at the bottom; from
  * a strategy that bleeds repeatedly, which is a design flaw worth fixing.

This prints the drawdown timeline: peak date, trough date, recovery date, time
under water, the worst drawdown per calendar year, and the worst drawdown with
a chosen year excluded. It runs the allocator exactly as `run_allocator.py`
does — same information set, same costs — and writes no MLflow run.

    uv run python scripts/drawdown_profile.py --split oos_r4_v2 --signal-tag r4_v2
"""
from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
from loguru import logger


def _episodes(navs: np.ndarray, dates: list[date]) -> None:
    peak = np.maximum.accumulate(navs)
    dd = 1.0 - navs / np.maximum(peak, 1e-12)
    i_trough = int(np.argmax(dd))
    worst = float(dd[i_trough])
    # The peak that preceded the trough, and the first date NAV regained it.
    i_peak = int(np.argmax(navs[: i_trough + 1]))
    target = navs[i_peak]
    after = np.nonzero(navs[i_trough:] >= target)[0]
    i_rec = int(i_trough + after[0]) if after.size else -1

    print("\n── worst drawdown ───────────────────────────────────────────────")
    print(f"  peak    {dates[i_peak]}   NAV {navs[i_peak]:>14,.0f}")
    print(f"  trough  {dates[i_trough]}   NAV {navs[i_trough]:>14,.0f}   "
          f"drawdown {worst:.1%}")
    if i_rec >= 0:
        print(f"  recovered {dates[i_rec]}  after {i_rec - i_peak} trading days "
              f"({(dates[i_rec] - dates[i_peak]).days / 365.25:.1f} years from peak)")
        print(f"  time under water from the trough: {i_rec - i_trough} trading days")
    else:
        print("  NEVER RECOVERED within the backtest span")

    print("\n── worst drawdown within each calendar year ─────────────────────")
    years = sorted({d.year for d in dates})
    for y in years:
        idx = [i for i, d in enumerate(dates) if d.year == y]
        if len(idx) < 2:
            continue
        seg = navs[idx]
        p = np.maximum.accumulate(seg)
        print(f"  {y}: {float(np.max(1.0 - seg / np.maximum(p, 1e-12))):>7.1%}")

    print("\n── worst drawdown EXCLUDING one year ────────────────────────────")
    for drop in years:
        keep = [i for i, d in enumerate(dates) if d.year != drop]
        if len(keep) < 3:
            continue
        seg = navs[keep]
        p = np.maximum.accumulate(seg)
        print(f"  without {drop}: {float(np.max(1.0 - seg / np.maximum(p, 1e-12))):>7.1%}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="oos_r4_v2")
    ap.add_argument("--signal-tag", default="r4_v2")
    ap.add_argument("--panels-root", default="data/panels_kite")
    ap.add_argument("--k", type=int, default=30)
    ap.add_argument("--horizon", default="20d")
    ap.add_argument("--freq", default="monthly")
    ap.add_argument("--capital", type=float, default=1_000_000.0)
    ap.add_argument("--band", type=float, default=0.0)
    ap.add_argument("--equal-weight", action="store_true",
                    help="profile the equal-weight baseline instead")
    args = ap.parse_args()

    # scripts/ is not a package; load run_allocator.py by path so its loaders
    # are reused verbatim rather than reimplemented and allowed to drift.
    import importlib.util

    from trader.allocator import AllocatorParams, RebalanceSchedule
    from trader.data.features import FEATURE_COLS
    from trader.data.universe import active_tickers
    from trader.env.allocator_env import vol_column_for
    from trader.env.panel_env import PanelTradingEnv
    _spec = importlib.util.spec_from_file_location(
        "_run_allocator_mod", Path(__file__).with_name("run_allocator.py")
    )
    assert _spec and _spec.loader
    ra = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(ra)

    root = Path(args.panels_root)
    panel_path = root / f"{args.split}.parquet"
    pred_path = Path("data/signal") / args.signal_tag / "predictions.parquet"
    universe = active_tickers()
    dates = sorted(pl.read_parquet(panel_path, columns=["date"])["date"].unique().to_list())
    lookback = 60
    episode_length = len(dates) - lookback - 1

    r_hat_all = ra._dense_signal(pred_path, dates, universe, (args.horizon,))
    vol_all = ra._dense_column(panel_path, vol_column_for(20), dates, universe)

    env = PanelTradingEnv(
        panel_path=panel_path, universe=universe, feature_columns=FEATURE_COLS,
        lookback=lookback, episode_length=episode_length,
        initial_cash=args.capital, seed=0,
        rebalance_schedule=RebalanceSchedule(args.freq),
    )
    params = AllocatorParams(k=args.k, no_trade_band=args.band)

    if args.equal_weight:
        navs, turns, _ = ra._run_baseline(env, 0)
        label = "equal_weight"
    else:
        navs, turns, _ = ra._run_allocator(env, r_hat_all[args.horizon], vol_all, params, 0)
        label = f"allocator k={args.k} {args.freq} {args.horizon} band={args.band}"

    nav = np.asarray(navs, dtype=np.float64)
    # navs carries one entry per step plus the opening NAV; align to the dates
    # the episode actually stepped through.
    step_dates = dates[lookback : lookback + len(nav)]
    logger.info(f"{label}: {len(nav)} NAV points, {step_dates[0]}..{step_dates[-1]}, "
                f"capital Rs {args.capital:,.0f}")
    _episodes(nav, step_dates)


if __name__ == "__main__":
    main()
