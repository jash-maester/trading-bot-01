#!/usr/bin/env python
"""Alpha, beta and capture of the strategy against NIFTY 50.

Every benchmark in this repo so far has been `equal_weight` over our own
universe. That is the right *internal control* — it isolates selection skill
from market exposure — but it is not the benchmark a real investor uses. The
question "did this beat just buying the index?" has never been answered.

    uv run python scripts/benchmark_vs_nifty.py --split test --signal-tag r4_v2_holdout

Reports, over the traded span:

  * NIFTY 50 buy-and-hold, on the same calendar, as a floor to clear;
  * **beta** — the strategy's sensitivity to the index, from an OLS regression
    of daily strategy returns on daily index returns;
  * **alpha** — the intercept of that regression, annualised. This is return
    NOT explained by index exposure, which is what the strategy has to produce
    to be worth running over an index fund;
  * tracking error, information ratio, correlation;
  * up/down capture — what fraction of the index's up days and down days the
    strategy participated in.

The index is `^NSEI`, fetched into `data/kite_ohlcv` alongside the equities
because `beta_nifty_60d` needs it (its absence is what made that feature
constant 1.0 for months). It is NOT part of the tradeable universe, so nothing
here can leak into a position.

Caveat carried in the output: NIFTY 50 is a large-cap index while this universe
is 504 names skewed to mid- and small-cap, so a positive alpha here is partly
compensation for size and liquidity risk, not purely selection skill. A
same-universe control is what `null_signal` is for; this is the other half of
the picture, not a replacement.
"""
from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
from loguru import logger

_ANNUALISE = 252.0


def load_index(start: date, end: date, root: Path, ticker: str) -> pl.DataFrame:
    """Daily closes for the index over ``[start, end]``, sorted, deduplicated."""
    files = sorted(root.rglob(f"ticker={ticker}.parquet"))
    if not files:
        raise SystemExit(
            f"No {ticker} bars under {root}. `configs/data/kite_v1.yaml` sets "
            "`fetch_index: true`; re-run scripts/fetch_kite_data.py."
        )
    df = pl.concat([pl.read_parquet(f) for f in files], how="diagonal_relaxed")
    df = (
        df.with_columns(pl.col("date").cast(pl.Date))
        .filter((pl.col("date") >= start) & (pl.col("date") <= end))
        .unique(subset=["date"])
        .sort("date")
        .select(["date", "close"])
    )
    if df.height < 3:
        raise SystemExit(f"Only {df.height} {ticker} bars in {start}..{end}.")
    return df


def stats(strat: np.ndarray, bench: np.ndarray) -> dict[str, float]:
    """OLS of strategy daily log returns on benchmark daily log returns."""
    x = bench - bench.mean()
    y = strat - strat.mean()
    var = float(np.dot(x, x))
    beta = float(np.dot(x, y) / var) if var > 0 else float("nan")
    alpha_daily = float(strat.mean() - beta * bench.mean())
    resid = strat - (alpha_daily + beta * bench)
    te = float(np.std(resid, ddof=1) * np.sqrt(_ANNUALISE))
    active = strat - bench
    ir = (
        float(active.mean() / np.std(active, ddof=1) * np.sqrt(_ANNUALISE))
        if np.std(active, ddof=1) > 0
        else float("nan")
    )
    up, dn = bench > 0, bench < 0
    return {
        "beta": beta,
        # Annualised by compounding the daily intercept, not by scaling it.
        "alpha_ann": float(np.expm1(alpha_daily * _ANNUALISE)),
        "corr": float(np.corrcoef(strat, bench)[0, 1]),
        "tracking_error_ann": te,
        "information_ratio": ir,
        "up_capture": (
            float(strat[up].mean() / bench[up].mean()) if up.sum() > 1 else float("nan")
        ),
        "down_capture": (
            float(strat[dn].mean() / bench[dn].mean()) if dn.sum() > 1 else float("nan")
        ),
        "n_days": float(strat.size),
    }


def summarise(nav: np.ndarray, label: str, years: float) -> dict[str, float]:
    r = np.diff(np.log(np.maximum(nav, 1e-12)))
    sd = float(np.std(r, ddof=1))
    peak = np.maximum.accumulate(nav)
    return {
        "label": label,  # type: ignore[dict-item]
        "cagr": float((nav[-1] / nav[0]) ** (1.0 / max(years, 1e-9)) - 1.0),
        "sharpe": float(r.mean() / sd * np.sqrt(_ANNUALISE)) if sd > 0 else float("nan"),
        "vol_ann": sd * np.sqrt(_ANNUALISE),
        "mdd": -float(np.max(1.0 - nav / np.maximum(peak, 1e-12))),
        "total": float(nav[-1] / nav[0] - 1.0),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="oos_r4_v2")
    ap.add_argument("--signal-tag", default="r4_v2")
    ap.add_argument("--panels-root", default="data/panels_kite")
    ap.add_argument("--ohlcv-root", default="data/kite_ohlcv")
    ap.add_argument("--index-ticker", default="^NSEI")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--horizon", default="20d")
    ap.add_argument("--freq", default="monthly")
    ap.add_argument("--capital", type=float, default=1_000_000.0)
    ap.add_argument("--band", type=float, default=0.0)
    ap.add_argument("--stop-loss", type=float, default=None,
                    help="enable the per-name stop overlay at this loss fraction")
    ap.add_argument("--equal-weight", action="store_true",
                    help="benchmark the equal-weight baseline instead of the allocator")
    args = ap.parse_args()

    import importlib.util

    from trader.allocator import AllocatorParams, RebalanceSchedule
    from trader.allocator.risk import RiskOverlay, RiskParams
    from trader.data.features import FEATURE_COLS
    from trader.data.universe import active_tickers
    from trader.env.allocator_env import vol_column_for
    from trader.env.panel_env import PanelTradingEnv

    spec = importlib.util.spec_from_file_location(
        "_ra", Path(__file__).with_name("run_allocator.py")
    )
    assert spec and spec.loader
    ra = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ra)

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
    # `navs[0]` is the NAV BEFORE the first traded day, and `navs[i]` for i>=1 is
    # the mark at the close of day `start_idx + i - 1`. Labelling navs[0] with
    # dates[lookback] shifts the whole series one day against the index, which
    # does not merely blur the correlation — daily equity returns are close to
    # independent day to day, so a one-day shift collapses a ~0.9 correlation to
    # ~0. The equal-weight arm reporting beta -0.107 against its own market is
    # what caught it.
    env.reset(seed=0)
    start_idx = env.day_index

    overlay = (
        RiskOverlay(RiskParams(stop_loss=args.stop_loss), len(universe))
        if args.stop_loss
        else None
    )
    if args.equal_weight:
        navs, turns, diag = ra._run_baseline(env, 0)
    else:
        navs, turns, diag = ra._run_allocator(
            env, r_hat_all[args.horizon],
            vol_all, AllocatorParams(k=args.k, no_trade_band=args.band), 0, risk=overlay,
        )
    nav = np.asarray(navs, dtype=np.float64)
    step_dates = dates[start_idx - 1 : start_idx - 1 + len(nav)]
    if len(step_dates) != len(nav):
        raise SystemExit(
            f"{len(nav)} NAV points but only {len(step_dates)} dates from "
            f"index {start_idx - 1}; alignment is not sound, refusing to regress."
        )

    idx = load_index(step_dates[0], step_dates[-1], Path(args.ohlcv_root), args.index_ticker)
    # Align on the strategy's calendar; an index holiday the panel traded (or
    # the reverse) would otherwise silently shift one series against the other.
    joined = (
        pl.DataFrame({"date": step_dates, "nav": nav})
        .join(idx, on="date", how="inner")
        .sort("date")
    )
    if joined.height < 30:
        raise SystemExit(f"Only {joined.height} overlapping days; nothing to regress.")
    s_nav = joined["nav"].to_numpy()
    b_nav = joined["close"].to_numpy()
    years = (joined["date"][-1] - joined["date"][0]).days / 365.25

    logger.info(
        f"{args.split} | K={args.k} {args.freq} {args.horizon} band={args.band} "
        f"stop={args.stop_loss} | {joined.height} overlapping days "
        f"{joined['date'][0]}..{joined['date'][-1]} ({years:.2f}y)"
    )
    if joined.height < len(nav):
        logger.warning(
            f"{len(nav) - joined.height} strategy day(s) had no index bar and were "
            "dropped from the regression."
        )

    st = summarise(s_nav, "equal_weight" if args.equal_weight else "strategy", years)
    bm = summarise(b_nav, args.index_ticker, years)
    print(f"\n{'arm':<22}{'CAGR':>9}{'Sharpe':>9}{'Vol':>9}{'MaxDD':>9}{'Total':>10}")
    print("-" * 68)
    for m in (bm, st):
        print(f"{str(m['label']):<22}{m['cagr']:>9.3f}{m['sharpe']:>9.3f}"
              f"{m['vol_ann']:>9.3f}{m['mdd']:>9.3f}{m['total']:>10.3f}")
    print("-" * 68)

    sr = np.diff(np.log(np.maximum(s_nav, 1e-12)))
    br = np.diff(np.log(np.maximum(b_nav, 1e-12)))
    k = stats(sr, br)
    print(f"\nvs {args.index_ticker} over {int(k['n_days'])} days")
    print(f"  beta                {k['beta']:>8.3f}")
    print(f"  alpha (annualised)  {k['alpha_ann']:>8.3%}")
    print("      ^ return NOT explained by index exposure")
    print(f"  correlation         {k['corr']:>8.3f}")
    print(f"  tracking error      {k['tracking_error_ann']:>8.3f}")
    print(f"  information ratio   {k['information_ratio']:>8.3f}")
    print(f"  up capture          {k['up_capture']:>8.3f}")
    print(f"  down capture        {k['down_capture']:>8.3f}")
    print(
        "\nNIFTY 50 is large-cap; this universe is 504 names skewed mid/small-cap,\n"
        "so part of any positive alpha is compensation for size and liquidity risk,\n"
        "not selection skill. `null_signal` is the same-universe control."
    )


if __name__ == "__main__":
    main()
