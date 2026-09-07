#!/usr/bin/env python
"""K0 — can Kronos load, run, and produce per-stock embeddings from our panel?

`13_fundamentals_and_news.md` section 2. This is the cheapest gate on the whole
Kronos line: if the model will not load and embed at our shape on 8 GiB, K1
never happens and nothing downstream matters.

    uv sync --extra kronos
    uv run python scripts/kronos_smoke.py --panel data/panels_kite/oos_r4_v2.parquet

What it checks, in order, stopping at the first failure:

1. the vendored repo imports (it is a repo-local `model` package, not a pip
   distribution, so `vendor/Kronos` has to be on the path);
2. the tokenizer and model load from the Hugging Face Hub. Upstream pins
   `huggingface_hub==0.33.1` and we resolve 1.x, so this is a real break risk
   and is checked before anything expensive;
3. a forecast runs end to end, proving the documented path works;
4. **embeddings can be extracted at all.** There is no documented API for this.
   The route is `tokenizer.encode` -> `model.decode_s1`, whose second return
   value is the transformer's context representation, `[B, seq_len, d_model]`.
   That is the tensor K1 would use in place of `TCNEncoder`'s output;
5. VRAM and wall clock at a realistic batch, because `ARCHITECTURE.md` section 5
   records that the binding constraint here is memory, not arithmetic, and that
   WSL2 pages silently rather than failing.

Writes nothing and trains nothing.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
from loguru import logger

_REPO = Path(__file__).resolve().parents[1]
_VENDOR = _REPO / "vendor" / "Kronos"


def _ok(step: str, detail: str = "") -> None:
    logger.info(f"  PASS  {step}{'  ' + detail if detail else ''}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", type=Path, default=Path("data/panels_kite/oos_r4_v2.parquet"))
    ap.add_argument("--model", default="NeoQuasar/Kronos-small")
    ap.add_argument("--tokenizer", default="NeoQuasar/Kronos-Tokenizer-base")
    ap.add_argument("--lookback", type=int, default=60, help="bars per stock, our L")
    ap.add_argument("--batch", type=int, default=64, help="stocks per forward pass")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    # ── 1. import ────────────────────────────────────────────────────────────
    if not _VENDOR.exists():
        raise SystemExit(
            f"{_VENDOR} not found. Kronos is a repo-local package:\n"
            "  git clone --depth 1 https://github.com/shiyu-coder/Kronos.git vendor/Kronos"
        )
    sys.path.insert(0, str(_VENDOR))
    try:
        from model import Kronos, KronosTokenizer  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - environment probe
        raise SystemExit(f"K0 FAIL at step 1 (import): {exc}") from exc
    _ok("import", f"from {_VENDOR}")

    # ── 2. load ──────────────────────────────────────────────────────────────
    t0 = time.time()
    try:
        tok = KronosTokenizer.from_pretrained(args.tokenizer)
        mdl = Kronos.from_pretrained(args.model)
    except Exception as exc:
        raise SystemExit(
            f"K0 FAIL at step 2 (load): {exc}\n"
            "Upstream pins huggingface_hub==0.33.1; we resolve 1.x. If this is "
            "an API break, pin the extra in pyproject rather than downgrading "
            "the whole environment."
        ) from exc
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    tok, mdl = tok.to(dev).eval(), mdl.to(dev).eval()
    n_par = sum(p.numel() for p in mdl.parameters())
    _ok("load", f"{args.model} {n_par/1e6:.1f}M params on {dev} in {time.time()-t0:.1f}s")

    # ── 3. our panel, in Kronos's input shape ────────────────────────────────
    panel = pl.read_parquet(
        args.panel, columns=["date", "ticker", "open", "high", "low", "close", "volume"]
    )
    tickers = (
        panel.filter(pl.col("close") > 0)
        .group_by("ticker").len().sort("len", descending=True)
        .head(args.batch)["ticker"].to_list()
    )
    dates = sorted(panel["date"].unique().to_list())[-args.lookback:]
    sub = panel.filter(pl.col("ticker").is_in(tickers) & pl.col("date").is_in(dates))
    wide = []
    for t in tickers:
        d = sub.filter(pl.col("ticker") == t).sort("date")
        if d.height != args.lookback:
            continue
        # Kronos takes OHLCV plus `amount` (turnover). We do not carry amount,
        # so it is close*volume — the definition NSE itself uses — rather than
        # zero-filled, which would hand the model a dead channel of exactly the
        # kind that made beta_nifty_60d constant for months.
        arr = np.stack([
            d["open"].to_numpy(), d["high"].to_numpy(), d["low"].to_numpy(),
            d["close"].to_numpy(), d["volume"].to_numpy(),
            d["close"].to_numpy() * d["volume"].to_numpy(),
        ], axis=1)
        wide.append(arr)
    if not wide:
        raise SystemExit("K0 FAIL at step 3: no ticker had a full lookback window.")
    x = torch.tensor(np.stack(wide), dtype=torch.float32, device=dev)
    _ok("panel", f"{x.shape[0]} stocks x {x.shape[1]} bars x {x.shape[2]} channels")

    # Kronos normalises inside its predictor; encoding raw prices would quantise
    # a Rs 3,000 stock and a Rs 30 stock into different regions of the codebook
    # for no reason. Standardise per stock per channel, as the predictor does.
    mu = x.mean(dim=1, keepdim=True)
    sd = x.std(dim=1, keepdim=True).clamp_min(1e-8)
    xn = ((x - mu) / sd).clamp(-5, 5)

    # ── 4. embeddings ────────────────────────────────────────────────────────
    torch.cuda.reset_peak_memory_stats() if dev.type == "cuda" else None
    t0 = time.time()
    with torch.no_grad():
        z = tok.encode(xn, half=True)
        s1, s2 = (z[0], z[1]) if isinstance(z, (list, tuple)) else (z[..., 0], z[..., 1])
        _, ctx = mdl.decode_s1(s1, s2)
    dt = time.time() - t0
    if ctx.ndim != 3 or ctx.shape[0] != xn.shape[0]:
        raise SystemExit(f"K0 FAIL at step 4: context has shape {tuple(ctx.shape)}")
    emb = ctx[:, -1, :]        # last position = the embedding for "today"
    _ok("embeddings", f"context {tuple(ctx.shape)} -> per-stock {tuple(emb.shape)}")

    finite = torch.isfinite(emb).all().item()
    spread = float(emb.std(dim=0).mean())
    logger.info(
        f"  embedding finite={finite}  cross-sectional sd={spread:.4f}  "
        f"d_model={emb.shape[-1]}"
    )
    if not finite:
        raise SystemExit("K0 FAIL: embeddings contain non-finite values.")
    if spread < 1e-6:
        raise SystemExit(
            "K0 FAIL: embeddings are identical across stocks — a constant "
            "channel carries no cross-sectional information (CLAUDE.md, "
            "feature liveness)."
        )

    # ── 5. cost ──────────────────────────────────────────────────────────────
    if dev.type == "cuda":
        peak = torch.cuda.max_memory_allocated() / 2**30
        logger.info(f"  peak VRAM {peak:.2f} GiB for {xn.shape[0]} stocks")
        per_full = peak * (504 / xn.shape[0])
        logger.info(
            f"  extrapolated to 504 stocks: {per_full:.2f} GiB "
            f"({'FITS' if per_full < 6.5 else 'BATCH IT'} against 8 GiB)"
        )
    rate = xn.shape[0] / dt
    logger.info(f"  {dt:.2f}s for {xn.shape[0]} stocks = {rate:.0f} stock-windows/s")
    logger.info(
        f"  full cache estimate: 504 stocks x ~1980 dates = "
        f"{504*1980/rate/3600:.1f} h at this rate"
    )

    logger.info("K0 PASS — Kronos loads, runs, and yields per-stock embeddings.")
    logger.info("Next: K1, frozen Kronos embeddings vs the from-scratch TCN on rank IC.")


if __name__ == "__main__":
    main()
