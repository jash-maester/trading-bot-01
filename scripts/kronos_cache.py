#!/usr/bin/env python
"""K1a — cache frozen Kronos embeddings for every (date, stock) in a panel.

`13_fundamentals_and_news.md` section 2, arm 1. Kronos is used as a FROZEN
encoder: its weights never move, so its output is a pure function of the panel
and can be computed once and reused. That is the same argument that makes our
own encoder cache worth 47x, and it is what makes K1 a cheap experiment rather
than a training run.

    uv run python scripts/kronos_cache.py \\
        --panel data/panels_kite/oos_r4_v2.parquet --out data/kronos/oos_r4_v2

Writes, mirroring R4's pinned artefact shape so downstream code reads one
convention:

    embeddings.npy   float16 [T, N, d_model]
    index.json       {"dates", "tickers", "embed_dim", "model", "lookback", ...}

The first ``lookback - 1`` dates have no full window and are written as NaN, the
same convention `predict_panel` uses, so a consumer cannot mistake "not
computable" for "zero".
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
from loguru import logger

_REPO = Path(__file__).resolve().parents[1]
_VENDOR = _REPO / "vendor" / "Kronos"

_CHANNELS = ("open", "high", "low", "close", "volume")


def _dense(panel: pl.DataFrame, col: str, dates: list, tickers: list) -> np.ndarray:
    """``[T, N]`` of ``col``, NaN where a (date, ticker) has no row."""
    di = {d: i for i, d in enumerate(dates)}
    ti = {t: i for i, t in enumerate(tickers)}
    out = np.full((len(dates), len(tickers)), np.nan, dtype=np.float64)
    sub = panel.filter(pl.col("date").is_in(list(di)) & pl.col("ticker").is_in(list(ti)))
    r = np.array([di[d] for d in sub["date"].to_list()], dtype=np.int64)
    c = np.array([ti[t] for t in sub["ticker"].to_list()], dtype=np.int64)
    out[r, c] = sub[col].to_numpy().astype(np.float64)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="NeoQuasar/Kronos-small")
    ap.add_argument("--tokenizer", default="NeoQuasar/Kronos-Tokenizer-base")
    ap.add_argument("--lookback", type=int, default=60)
    ap.add_argument("--batch-stocks", type=int, default=252,
                    help="stocks per forward pass; K0 measured 504 at ~1.5 GiB")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit-dates", type=int, default=0, help="debug: cap dates")
    ap.add_argument(
        "--amount-mode", choices=("close_volume", "zero"), default="close_volume",
        help=(
            "How to fill Kronos's optional 6th channel. `close_volume` is "
            "close*volume, which is REDUNDANT -- a deterministic function of two "
            "channels the model already sees. `zero` is Kronos's own default for "
            "a missing column. Neither is real turnover; NSE's bhavcopy carries "
            "TURNOVER_LACS and AVG_PRICE and R8 already parses that file but "
            "discards both. This flag exists to measure whether the channel "
            "moves rank IC at all, before paying for the fetch that would supply "
            "the real thing."
        ),
    )
    args = ap.parse_args()

    if not _VENDOR.exists():
        raise SystemExit(f"{_VENDOR} not found — see scripts/kronos_smoke.py.")
    sys.path.insert(0, str(_VENDOR))
    from model import Kronos, KronosTokenizer  # type: ignore[import-not-found]

    from trader.data.universe import active_tickers

    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    tok = KronosTokenizer.from_pretrained(args.tokenizer).to(dev).eval()
    mdl = Kronos.from_pretrained(args.model).to(dev).eval()

    panel = pl.read_parquet(args.panel, columns=["date", "ticker", *_CHANNELS])
    tickers = active_tickers()
    dates = sorted(panel["date"].unique().to_list())
    if args.limit_dates:
        dates = dates[: args.limit_dates]
    T, N, L = len(dates), len(tickers), args.lookback

    cols = {c: _dense(panel, c, dates, tickers) for c in _CHANNELS}
    # `amount` is turnover. Kronos accepts it as an optional 6th channel and
    # zero-fills when missing; close*volume is the definition NSE uses and is
    # strictly better than handing the model a dead channel.
    if args.amount_mode == "zero":
        cols["amount"] = np.zeros_like(cols["close"])
    else:
        cols["amount"] = cols["close"] * cols["volume"]
    stack = np.stack([cols[c] for c in (*_CHANNELS, "amount")], axis=2)  # [T, N, 6]

    d_model = int(mdl.d_model) if hasattr(mdl, "d_model") else 512
    out = np.full((T, N, d_model), np.nan, dtype=np.float16)

    logger.info(
        f"{args.panel.name}: {T} dates x {N} tickers, lookback {L}, "
        f"d_model {d_model}, device {dev}"
    )
    t_start = time.time()
    done = 0
    for t in range(L - 1, T):
        window = stack[t - L + 1 : t + 1]            # [L, N, 6]
        x = torch.tensor(window, dtype=torch.float32, device=dev).permute(1, 0, 2)
        # A stock with any gap in its window cannot be embedded; leave it NaN
        # rather than imputing, so a consumer sees "unknown" not "flat".
        usable = torch.isfinite(x).all(dim=(1, 2)) & (x[:, :, 3] > 0).all(dim=1)
        idx = torch.nonzero(usable, as_tuple=True)[0]
        if idx.numel() == 0:
            continue
        for s in range(0, idx.numel(), args.batch_stocks):
            sel = idx[s : s + args.batch_stocks]
            xb = x[sel]
            mu = xb.mean(dim=1, keepdim=True)
            sd = xb.std(dim=1, keepdim=True).clamp_min(1e-8)
            xn = ((xb - mu) / sd).clamp(-5, 5)
            with torch.no_grad():
                z = tok.encode(xn, half=True)
                s1, s2 = (z[0], z[1]) if isinstance(z, (list, tuple)) else (z[..., 0], z[..., 1])
                _, ctx = mdl.decode_s1(s1, s2)
            out[t, sel.cpu().numpy()] = ctx[:, -1, :].to(torch.float16).cpu().numpy()
        done += 1
        if done % 100 == 0:
            el = time.time() - t_start
            rate = done / el
            logger.info(
                f"  {done}/{T - L + 1} dates  {rate:.1f} dates/s  "
                f"eta {(T - L + 1 - done) / max(rate, 1e-9) / 60:.1f} min"
            )

    args.out.mkdir(parents=True, exist_ok=True)
    np.save(args.out / "embeddings.npy", out)
    covered = float(np.isfinite(out[:, :, 0]).mean())
    (args.out / "index.json").write_text(
        json.dumps(
            {
                "dates": [d.isoformat() for d in dates],
                "tickers": tickers,
                "embed_dim": d_model,
                "model": args.model,
                "tokenizer": args.tokenizer,
                "lookback": L,
                "source_panel": str(args.panel),
                "coverage": round(covered, 4),
                "frozen": True,
                "amount_mode": args.amount_mode,
            },
            indent=2,
        )
        + "\n"
    )
    logger.info(
        f"wrote {args.out}/embeddings.npy {out.shape} float16, "
        f"coverage {covered:.1%}, {(time.time()-t_start)/60:.1f} min"
    )
    if covered < 0.10:
        raise SystemExit(f"coverage {covered:.1%} is too low to be usable.")


if __name__ == "__main__":
    main()
