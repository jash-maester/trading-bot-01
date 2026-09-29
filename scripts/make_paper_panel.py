"""Trim the forward panel so the paper replay STARTS TRADING on a given date.

    uv run python scripts/make_paper_panel.py --src data/panels_forward/full.parquet \
        --out data/panels_forward/paper.parquet --trade-start 2026-09-30 --lookback 60

Restart 3 (user, 2026-09-29): Rs 1,00,000 of cash deployed from 2026-09-30,
instead of scoring books that carry positions from a 2024 replay start.
PanelTradingEnv's first step always trades and needs `lookback` rows of
history before it, so the panel is cut to begin exactly `lookback` sessions
before the first session on/after --trade-start. Every book -- signal, random,
equal-weight -- then holds cash until that session and deploys on it. Features
are unaffected: they were computed on the full forward panel before the cut.
"""
from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import polars as pl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--trade-start", type=date.fromisoformat, required=True)
    ap.add_argument("--lookback", type=int, default=60)
    a = ap.parse_args()
    df = pl.read_parquet(a.src)
    dates = sorted(df["date"].unique().to_list())
    first = next((i for i, d in enumerate(dates) if d >= a.trade_start), None)
    if first is None:
        raise SystemExit(f"no session on/after {a.trade_start} in {a.src} (ends {dates[-1]})")
    if first < a.lookback:
        raise SystemExit(f"only {first} sessions before {a.trade_start}; need {a.lookback}")
    cut = dates[first - a.lookback]
    out = df.filter(pl.col("date") >= cut)
    out.write_parquet(a.out)
    print(f"{a.out}: {out['date'].n_unique()} sessions {cut}..{dates[-1]}; "
          f"first trading session {dates[first]} (lookback {a.lookback})")


if __name__ == "__main__":
    main()
