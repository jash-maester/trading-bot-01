#!/usr/bin/env python
"""Is a news veto on the stop-loss worth building? Measured without any news.

`13_fundamentals_and_news.md` §4 designs the news arm as a **veto on the stop**,
not a ranking signal: a held name that breaches its stop is sold if the fall
carries adverse news and held if it looks like noise. That is the IndiGo case —
a sound company falling on sentiment and recovering, against one whose fall is
structural.

The design has a precondition nobody has measured, and it can be measured with
no news at all. A veto is only worth building if **stopping is destroying value,
on a subset that something could separate.** Three questions, in order:

1. **Is there anything to recover?** When a name is stopped, its proceeds sit in
   cash for the cooldown. So stopping cost us that name's forward return over
   the cooldown, or saved us it. If stopped names keep falling on average, a
   perfect veto has nothing to win and the arm is dead here, for free.

2. **How much could a PERFECT veto win?** An oracle that sold only the names
   that went on to fall, and held every one that recovered. That is the ceiling
   on any veto, news-driven or otherwise — no real signal beats it.

3. **Is the difference separable by anything we already have?** If R4's own
   score at the moment of the stop already predicts which stopped names recover,
   the answer is not news, it is to read the signal we already pay for. If
   nothing available separates them, news has a job — but the ceiling from (2)
   is what it is competing for.

    uv run python scripts/stop_veto_headroom.py

PREDICTION, recorded before the numbers exist so it can be wrong: stopped names
average between -2% and +1% over the next 21 days (a stop fires on a name that
has already fallen, and weak momentum continuation should keep it soft), the
distribution is wide with roughly 40-45% recovering, the oracle ceiling is worth
several points of gross CAGR, and R4's score at stop time separates the two
groups hardly at all -- |Spearman| < 0.05.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl
from loguru import logger
from scipy.stats import spearmanr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--events-dir", type=Path, default=Path("audit/stop_events"))
    ap.add_argument("--panel", type=Path, default=Path("data/panels_kite/oos_r4_v2.parquet"))
    ap.add_argument("--signal", type=Path,
                    default=Path("data/signal/r4_v2/predictions.parquet"))
    ap.add_argument("--horizons", default="5,10,21,60")
    ap.add_argument("--freq", default="monthly",
                    help="the cadence the events were generated at")
    args = ap.parse_args()

    horizons = [int(h) for h in args.horizons.split(",")]
    dates = sorted(pl.read_parquet(args.panel, columns=["date"])["date"].unique().to_list())

    closes = (
        pl.read_parquet(args.panel, columns=["date", "ticker", "close"])
        .pivot(index="date", on="ticker", values="close")
        .sort("date")
    )
    px_tickers = [c for c in closes.columns if c != "date"]
    px = closes.select(px_tickers).to_numpy()
    px_idx = {t: i for i, t in enumerate(px_tickers)}
    market = np.nanmean(np.where(px > 0, px, np.nan), axis=1)

    sig = pl.read_parquet(args.signal)
    sig_map: dict[tuple[object, str], float] = {}
    if "r_hat_20d" in sig.columns:
        for r in sig.iter_rows(named=True):
            if r["r_hat_20d"] is not None:
                sig_map[(r["date"], r["ticker"])] = float(r["r_hat_20d"])

    from trader.allocator.rebalance import RebalanceSchedule

    reb = np.flatnonzero(
        np.asarray(RebalanceSchedule(args.freq).mask(dates), dtype=bool)
    )
    logger.info(f"{len(dates)} sessions, {reb.size} {args.freq} rebalance days")

    files = sorted(args.events_dir.glob("stop_events_*.parquet"))
    if not files:
        raise SystemExit(f"no stop-event files under {args.events_dir}")

    for f in files:
        ev = pl.read_parquet(f)
        arm = f.stem.replace("stop_events_", "")
        print(f"\n{'=' * 78}\n{arm}   {ev.height:,} stop event(s)\n{'=' * 78}")
        if ev.is_empty():
            print("  no stops fired — nothing to veto")
            continue

        # Self-check the day_index -> date mapping before trusting anything
        # built on it. A silent off-by-one here would move every forward
        # return by a day and no downstream number would look wrong.
        #
        # The recorded `close` is `env.closes_today()`, which is the close at
        # `day_index - 1` (panel_env.py:439) -- the price the stop was EVALUATED
        # against. The sale itself happens on the step at `day_index`, so that
        # is where the money leaves the name and where a forward return has to
        # be anchored. Getting these two confused is exactly the off-by-one
        # this check exists to catch.
        checked = mismatch = 0
        for r in ev.head(200).iter_rows(named=True):
            i, j = int(r["day_index"]) - 1, px_idx.get(r["ticker"])
            if j is None or not 0 <= i < len(dates):
                continue
            checked += 1
            if not np.isclose(px[i, j], r["close"], rtol=1e-6):
                mismatch += 1
        if checked and mismatch > checked * 0.02:
            raise SystemExit(
                f"{mismatch}/{checked} recorded closes disagree with the panel at "
                "dates[day_index] — the index mapping is wrong, refusing to "
                "report forward returns off it"
            )
        logger.info(f"{arm}: date mapping verified on {checked} event(s), "
                    f"{mismatch} mismatch")

        # How long the proceeds ACTUALLY sit in cash. The whole verdict turns
        # on this: the cooldown bars re-buying the stopped NAME for 21 steps,
        # but the cash is redeployed at the next scheduled rebalance. If that
        # is ~10 days, the 10d column is the honest one; if it were ~21, the
        # 21d column would be, and the sign of the result flips between them.
        # Measured rather than assumed for exactly that reason.
        if reb.size:
            gaps = np.array([
                int(reb[reb > d][0]) - d
                for d in ev["day_index"].to_list() if (reb > d).any()
            ])
            if gaps.size:
                print(f"  cash idle until the next rebalance: mean "
                      f"{gaps.mean():.1f}d, median {np.median(gaps):.0f}d, "
                      f"p10 {np.percentile(gaps, 10):.0f}, "
                      f"p90 {np.percentile(gaps, 90):.0f}")

        span = float(len(dates)) / 252.0
        print(f"  {ev.height / max(span, 1e-9):.1f} stops per year over "
              f"{span:.1f} years")
        print(f"  loss at stop: median {ev['loss'].median():.1%}, "
              f"p10 {ev['loss'].quantile(0.1):.1%}, "
              f"p90 {ev['loss'].quantile(0.9):.1%}")
        print(f"  weight when stopped: median {ev['weight'].median():.1%}")

        for h in horizons:
            fwd, mkt, sg, wt, loss = [], [], [], [], []
            for r in ev.iter_rows(named=True):
                # Anchored at the EXECUTION day, not the evaluation day: the
                # money is in cash from `day_index` onward.
                i, j = int(r["day_index"]), px_idx.get(r["ticker"])
                if j is None or i + h >= len(px):
                    continue
                p0, p1 = px[i, j], px[i + h, j]
                if not (np.isfinite(p0) and np.isfinite(p1) and p0 > 0):
                    continue
                fwd.append(p1 / p0 - 1.0)
                mkt.append(market[i + h] / market[i] - 1.0
                           if market[i] > 0 else np.nan)
                sg.append(sig_map.get((dates[i], r["ticker"]), np.nan))
                wt.append(float(r["weight"]))
                loss.append(float(r["loss"]))
            if not fwd:
                continue
            a, m, w = np.array(fwd), np.array(mkt), np.array(wt)
            excess = a - m

            print(f"\n  ── {h} trading days after the stop "
                  f"({len(a)} events with a full window) ──")
            print(f"  the sold name returned          mean {a.mean():+.2%}   "
                  f"median {np.median(a):+.2%}")
            print(f"  the market returned             mean "
                  f"{np.nanmean(m):+.2%}")
            print(f"  excess of the sold name         mean {excess.mean():+.2%}")
            print(f"  recovered (return > 0)          {(a > 0).mean():.1%} of events")

            # (1) what stopping actually did. Proceeds sit in cash for the
            #     cooldown, so the cost is the return forgone, position-weighted.
            cost = float((a * w).sum())
            print(f"\n  cost of stopping, position-weighted, summed over all "
                  f"events: {cost:+.2%} of NAV")
            print("    (positive = stopping COST return; the money sat in cash "
                  "while the name did this)")

            # (2) the ceiling. An oracle sells only the names that fall.
            ceiling = float((np.maximum(a, 0.0) * w).sum())
            print(f"  PERFECT veto would recover: {ceiling:+.2%} of NAV over "
                  f"{span:.1f} years  = {ceiling / span:+.2%}/yr gross")
            print("    (it holds every name that recovered and sells every one "
                  "that fell; no real veto beats this)")

            # (3) can anything we already have separate them?
            s = np.array(sg)
            v = np.isfinite(s)
            if v.sum() > 10:
                rho, p = spearmanr(s[v], a[v])
                print("\n  R4's own 20d score at the moment of the stop vs what "
                      "followed:")
                print(f"    Spearman {rho:+.4f} (p {p:.3f}, n {int(v.sum())})")
                print("    A strong number here would mean the answer is to read "
                      "the signal we\n    already pay for, not to buy news.")
            rho_l, p_l = spearmanr(np.array(loss), a)
            print("  depth of the fall that triggered the stop vs what followed:")
            print(f"    Spearman {rho_l:+.4f} (p {p_l:.3f})")

    print("\nRead it this way. If the cost of stopping is negative, stops are "
          "already\nadding return and a veto would remove it. If the ceiling is "
          "small relative to\nthe strategy's CAGR, no veto is worth building at "
          "any accuracy. Only a large\nceiling that nothing available separates "
          "leaves a job for news.")
    print("\nWHICH HORIZON IS THE RIGHT ONE. The cooldown bars RE-BUYING the "
          "stopped name\nfor 21 steps, but the cash itself is redeployed at the "
          "next monthly rebalance --\non average about 10 trading days later, "
          "not 21. So the 21d and 60d columns\nOVERSTATE both the cost and the "
          "ceiling, and the 5d/10d columns bracket the\nhonest number from "
          "below. Read 10d as the central case.")


if __name__ == "__main__":
    main()
