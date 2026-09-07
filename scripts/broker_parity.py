#!/usr/bin/env python
"""Do the backtest env and the paper broker agree on the same weights?

They have diverged twice, and both times it was expensive:

  * `min_trade_value` was honoured by the broker and ignored by the env, worth a
    measured 11% NAV gap (`11_cost_defect_and_fix_plan.md`);
  * `floor()` decided *whether* a trade happened rather than its size, so
    float32 weights produced one-share phantom sells in the env that the broker
    never made.

Both are fixed and neither fix was ever checked end to end. This drives the
IDENTICAL target weights through both paths over the same calendar and compares
NAV day by day.

    uv run python scripts/broker_parity.py --split test --signal-tag r4_v2_holdout

An exact match is NOT expected and is not the bar. The broker settles cash T+1,
fills sells before buys, and clips rather than redistributes a name above the
weight cap; the env does none of those. What matters is that the gap stays
small and does not TREND — a widening gap means a rule is applied on one side
and not the other, which is exactly what both past divergences looked like.
"""
from __future__ import annotations

import argparse
from datetime import UTC, datetime, time
from pathlib import Path

import numpy as np
import polars as pl
from loguru import logger


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--signal-tag", default="r4_v2_holdout")
    ap.add_argument("--panels-root", default="data/panels_kite")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--horizon", default="20d")
    ap.add_argument("--freq", default="monthly")
    ap.add_argument("--capital", type=float, default=1_000_000.0)
    ap.add_argument("--days", type=int, default=250, help="cap the comparison length")
    ap.add_argument("--tolerance", type=float, default=0.05,
                    help="max |NAV gap| as a fraction, before this exits non-zero")
    args = ap.parse_args()

    import importlib.util

    from trader.allocator import AllocatorParams, RebalanceSchedule, allocate
    from trader.broker.paper_broker import PaperBroker, PaperBrokerConfig
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
    panel = pl.read_parquet(panel_path)
    dates = sorted(panel["date"].unique().to_list())
    lookback = 60
    episode = min(args.days, len(dates) - lookback - 1)

    r_hat = ra._dense_signal(pred_path, dates, universe, (args.horizon,))[args.horizon]
    vol = ra._dense_column(panel_path, vol_column_for(20), dates, universe)

    # Dense open/close by (date, ticker) so both paths read the SAME prices.
    opens = ra._dense_column(panel_path, "open", dates, universe)
    closes = ra._dense_column(panel_path, "close", dates, universe)

    env = PanelTradingEnv(
        panel_path=panel_path, universe=universe, feature_columns=FEATURE_COLS,
        lookback=lookback, episode_length=episode, initial_cash=args.capital, seed=0,
        rebalance_schedule=RebalanceSchedule(args.freq),
    )
    obs, _ = env.reset(seed=0)
    start_idx = env.day_index

    broker = PaperBroker(
        PaperBrokerConfig(initial_cash=args.capital, slippage_model="none"),
    )
    broker.start_run(strategy_id="parity", started_at=datetime.now(UTC))

    params = AllocatorParams(k=args.k)
    rows: list[dict[str, object]] = []
    prev_day = None
    done = False
    while not done:
        day_idx = env.day_index
        day = dates[day_idx]
        rebal = env.is_rebalance_step()

        if rebal:
            sig = day_idx - 1
            target = allocate(
                r_hat[sig], vol[sig], obs["mask"].astype(bool),
                obs["sector_ids"].astype(np.int64),
                obs["portfolio"].astype(np.float64), params,
            )
            # Same vector to both. The broker wants a mapping and a sliver of
            # cash left for charges, which the env does not require.
            w = {
                universe[i]: float(target[i + 1])
                for i in range(len(universe))
                if target[i + 1] > 0.0
            }
            broker.submit_target_weights(w, ts=datetime.combine(day, time(9, 15)))
            obs, _, term, trunc, info = env.step_weights(target)
        else:
            obs, _, term, trunc, info = env.step(
                np.zeros(len(universe) + 1, dtype=np.float32)
            )
        done = bool(term or trunc)

        px_open = {universe[i]: float(opens[day_idx, i])
                   for i in range(len(universe)) if opens[day_idx, i] > 0.0}
        px_close = {universe[i]: float(closes[day_idx, i])
                    for i in range(len(universe)) if closes[day_idx, i] > 0.0}
        broker.open_session(day, px_open)
        snap = broker.mark_to_market(day, px_close)

        rows.append({
            "date": day,
            "env_nav": float(info["nav"]),
            "broker_nav": float(snap.nav),
            "rebalance": rebal,
        })
        prev_day = day

    broker.end_run(datetime.now(UTC))
    df = pl.DataFrame(rows).with_columns(
        ((pl.col("env_nav") - pl.col("broker_nav")) / pl.col("broker_nav")).alias("gap")
    )
    gap = df["gap"].to_numpy()

    logger.info(
        f"{args.split} K={args.k} {args.freq} {args.horizon}: {df.height} days "
        f"{dates[start_idx]}..{prev_day}, {int(df['rebalance'].sum())} rebalances"
    )
    print(f"\n{'':<14}{'env NAV':>14}{'broker NAV':>14}{'gap':>10}")
    print("-" * 52)
    for i in (0, df.height // 4, df.height // 2, 3 * df.height // 4, df.height - 1):
        r = df.row(i, named=True)
        print(f"{str(r['date']):<14}{r['env_nav']:>14,.0f}{r['broker_nav']:>14,.0f}"
              f"{r['gap']:>9.2%}")
    print("-" * 52)
    print(f"  final gap        {gap[-1]:>+8.2%}")
    print(f"  max |gap|        {np.abs(gap).max():>8.2%}")
    print(f"  mean gap         {gap.mean():>+8.2%}")
    # A gap that TRENDS is the signature of a rule applied on one side only; a
    # gap that wanders is settlement and fill-ordering noise.
    x = np.arange(gap.size, dtype=np.float64)
    slope = float(np.polyfit(x, gap, 1)[0]) if gap.size > 2 else 0.0
    print(f"  drift per day    {slope:>+8.4%}   <- trend is the dangerous shape")

    ok = bool(np.abs(gap).max() <= args.tolerance)
    print(f"\n  VERDICT: {'PASS' if ok else 'FAIL'} "
          f"(max |gap| {np.abs(gap).max():.2%} vs tolerance {args.tolerance:.2%})")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
