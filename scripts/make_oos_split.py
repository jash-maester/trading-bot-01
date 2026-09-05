#!/usr/bin/env python
"""Cut a panel split covering exactly R4's out-of-sample prediction span.

R4's walk-forward OOS calendar (the union of its per-window test segments) does
not line up with the panel's train/val/test boundaries: with the shipped
windows it runs 2016-07..2024-06, which sits mostly inside `train.parquet` by
date while being genuinely out-of-sample by construction — every prediction
comes from a model fitted on data strictly before it.

`scripts/run_allocator.py` backtests a whole split end to end, so pointing it at
`train` would hand it eleven years with no predictions at all, where the
allocator has no candidates and holds cash. That is not a bad result, it is a
meaningless one. This writes `<panels_root>/oos.parquet` bounded to the dates
the predictions actually cover.

    uv run python scripts/make_oos_split.py --tag r4_v1
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import polars as pl
from loguru import logger


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="signal tag under data/signal/")
    ap.add_argument("--panels-root", default="data/panels_kite")
    ap.add_argument("--signal-root", default="data/signal")
    ap.add_argument("--out", default="oos", help="split name to write")
    args = ap.parse_args()

    root = Path(args.panels_root)
    preds = pl.read_parquet(Path(args.signal_root) / args.tag / "predictions.parquet",
                            columns=["date"])
    lo, hi = preds["date"].min(), preds["date"].max()
    logger.info(f"{args.tag} predictions span {lo}..{hi}  ({preds.height:,} rows)")

    src = root / "full.parquet"
    if not src.exists():
        raise SystemExit(f"{src} not found — rebuild panels first.")
    panel = pl.read_parquet(src).filter(
        (pl.col("date") >= lo) & (pl.col("date") <= hi)
    ).sort(["date", "ticker"])
    if panel.height == 0:
        raise SystemExit("No panel rows in the prediction span.")

    out = root / f"{args.out}.parquet"
    panel.write_parquet(out)
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    (root / f"{args.out}.sha256").write_text(digest + "\n")
    logger.info(
        f"{out}: {panel.height:,} rows, {panel['ticker'].n_unique()} tickers, "
        f"{panel['date'].n_unique()} dates, "
        f"tradeable {panel.filter(pl.col('is_tradeable')).height:,}"
    )
    logger.info(f"SHA256={digest[:16]}...")


if __name__ == "__main__":
    main()
