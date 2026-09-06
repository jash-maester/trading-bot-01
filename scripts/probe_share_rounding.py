#!/usr/bin/env python
"""What the half-share snap in ``PanelTradingEnv._step_target`` costs or saves.

The env is handed **weights** and holds **shares**.  ``floor(target_value/open)``
used to decide both the size of a trade and whether there was one, which turned
a float32 weight round-trip into whole-share orders nobody asked for
(`tests/unit/test_weight_to_share_orders.py`).  The fix snaps a request back to
the held position when it rounds to it.  That changes the executed order set, so
it changes every backtest number measured before it — this script is how the
size of that change is quoted rather than guessed (CLAUDE.md rule 3).

**This is a diagnostic, not the R5 grid**: one cell (monthly, K=30, 20d), no
MLflow run, no GPU, and a bounded number of steps.  Nothing it prints is
quotable as a result; it exists to size a code change.

    uv run python scripts/probe_share_rounding.py                  # 500 steps
    uv run python scripts/probe_share_rounding.py --steps 900 --split oos

Run it once on each side of the change (stash the snap, run, restore, run) and
compare the two JSON blobs.  Measured 2026-09-06 on `oos.parquet` + `r4_v1`,
500 steps, ₹10 lakh, band 0.0:

    metric              floor decides      snap decides       delta
    final NAV           983,453.50         987,037.46         +3,583.96  (+0.36%)
    executed legs       2,160              2,144              -16
    scrip-sell-days     1,546              1,531              -15
    DP fees (₹)         23,715.64          23,485.54          -230.10
    sum daily turnover  7.4158             7.2366             -0.1791
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

from trader.allocator import AllocatorParams, RebalanceSchedule, allocate
from trader.data.features import FEATURE_COLS
from trader.data.universe import active_tickers
from trader.env.costs import _DP_CHARGE, ZerodhaEquityDeliveryCostModel
from trader.env.panel_env import PanelTradingEnv


class _LegProbe(ZerodhaEquityDeliveryCostModel):
    """Counts the legs and the per-scrip demat debits the env actually pays."""

    def __init__(self) -> None:
        super().__init__()
        self.legs: list[float] = []
        self.n_sold = 0

    def cost_vec(self, trade_values, is_buy, n_scrips_sold, *, intraday=None):  # type: ignore[no-untyped-def]
        self.legs.extend(float(v) for v in trade_values[trade_values > 0.0])
        self.n_sold += int(np.asarray(n_scrips_sold).sum())
        return super().cost_vec(trade_values, is_buy, n_scrips_sold, intraday=intraday)


def _dense(frame: pl.DataFrame, column: str, dates: list[object], tickers: list[str]) -> np.ndarray:
    di = {d: i for i, d in enumerate(dates)}
    ti = {t: i for i, t in enumerate(tickers)}
    rows = frame.filter(pl.col("date").is_in(list(di)) & pl.col("ticker").is_in(list(ti)))
    out = np.full((len(dates), len(tickers)), np.nan)
    out[
        [di[d] for d in rows["date"].to_list()],
        [ti[t] for t in rows["ticker"].to_list()],
    ] = rows[column].to_numpy()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--panels-root", default="data/panels_kite")
    ap.add_argument("--split", default="oos")
    ap.add_argument("--signal-tag", default="r4_v1")
    ap.add_argument("--horizon", default="r_hat_20d")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--k", type=int, default=30)
    ap.add_argument("--band", type=float, default=0.0)
    ap.add_argument("--capital", type=float, default=1_000_000.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    root = Path(args.panels_root)
    panel_path = root / f"{args.split}.parquet"
    universe = active_tickers()
    dates = sorted(pl.read_parquet(panel_path, columns=["date"])["date"].unique().to_list())

    preds = pl.read_parquet(Path("data/signal") / args.signal_tag / "predictions.parquet")
    r_hat = _dense(preds, args.horizon, dates, universe)
    vol = _dense(
        pl.read_parquet(panel_path, columns=["date", "ticker", "realized_vol_20d"]),
        "realized_vol_20d",
        dates,
        universe,
    )

    probe = _LegProbe()
    env = PanelTradingEnv(
        panel_path=panel_path,
        universe=universe,
        feature_columns=FEATURE_COLS,
        lookback=60,
        episode_length=args.steps,
        initial_cash=args.capital,
        cost_model=probe,
        rebalance_schedule=RebalanceSchedule("monthly"),
        seed=42,
    )
    params = AllocatorParams(k=args.k, no_trade_band=args.band)
    obs, _ = env.reset(seed=42)
    navs = [float(obs["nav"])]
    turns: list[float] = []
    done = False
    while not done:
        if env.is_rebalance_step():
            sig = env.day_index - 1
            obs, _, term, trunc, info = env.step_weights(
                allocate(
                    r_hat[sig],
                    vol[sig],
                    obs["mask"].astype(bool),
                    obs["sector_ids"].astype(np.int64),
                    obs["portfolio"].astype(np.float64),
                    params,
                )
            )
        else:
            obs, _, term, trunc, info = env.step(
                np.zeros(len(env.universe) + 1, dtype=np.float32)
            )
        navs.append(float(info["nav"]))
        turns.append(float(info["turnover"]))
        done = bool(term or trunc)

    payload = {
        "split": args.split,
        "signal_tag": args.signal_tag,
        "steps": args.steps,
        "k": args.k,
        "band": args.band,
        "capital": args.capital,
        "final_nav": navs[-1],
        "n_legs": len(probe.legs),
        "n_scrip_sell_days": probe.n_sold,
        "dp_fees": probe.n_sold * _DP_CHARGE,
        "sum_daily_turnover": float(np.sum(turns)),
        "median_leg": float(np.median(probe.legs)) if probe.legs else 0.0,
    }
    text = json.dumps(payload, indent=2)
    if args.out:
        Path(args.out).write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
