#!/usr/bin/env python
"""K1 — do frozen Kronos embeddings beat our from-scratch TCN on rank IC?

`13_fundamentals_and_news.md` section 2, arm 1, and the gate on the whole
Kronos line: if pretraining on 45 exchanges does not transfer to NSE at a 5–20
day horizon, nothing downstream rescues it.

    uv run python scripts/kronos_k1.py \\
        --cache data/kronos/full --panel data/panels_kite/full.parquet

Everything except the encoder is held identical to R4: the same walk-forward
windows, the same cross-sectionally z-scored forward-return targets, the same
`daily_scores` / `summarise_daily` scoring, and the same window-level gate from
`12_gate_decision.md`. Only the representation changes — `TCNEncoder` output
becomes a cached Kronos context vector — so a difference in rank IC is
attributable to the encoder and not to the plumbing around it.

Because the encoder is frozen, this trains **only the heads**: a small MLP on a
512-dim input. That is a far smaller fit than R4's, which is the point — §1 of
the plan argues our binding constraint is sample size, and a pretrained encoder
is the one intervention that reduces the parameter count instead of adding to
it.

The comparison baseline is r4_v2's committed per-window ICs
(`audit/r4_v2/summary.json`), so the two are read off the same windows.
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import torch
from loguru import logger
from torch import nn


def _as_date(v: object) -> date:
    return v if isinstance(v, date) else date.fromisoformat(str(v))


class _Heads(nn.Module):
    """One small MLP per horizon over a frozen embedding. Nothing else is fitted."""

    def __init__(self, d_in: int, hidden: int, horizons: tuple[int, ...]) -> None:
        super().__init__()
        self.horizons = horizons
        self.nets = nn.ModuleDict(
            {
                str(h): nn.Sequential(
                    nn.LayerNorm(d_in), nn.Linear(d_in, hidden), nn.GELU(),
                    nn.Dropout(0.1), nn.Linear(hidden, 1),
                )
                for h in horizons
            }
        )

    def forward(self, z: torch.Tensor) -> dict[int, torch.Tensor]:
        return {h: self.nets[str(h)](z).squeeze(-1) for h in self.horizons}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True, help="data/kronos/<tag>")
    ap.add_argument("--panel", type=Path, default=Path("data/panels_kite/full.parquet"))
    ap.add_argument("--baseline", type=Path, default=Path("audit/r4_v2/summary.json"))
    ap.add_argument("--horizons", default="5,20")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-days", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--label", default="kronos")
    ap.add_argument(
        "--with-features", action="store_true",
        help=(
            "concatenate the panel's 15 engineered features at date t onto the "
            "Kronos embedding. K1 showed the two representations win DIFFERENT "
            "windows -- Kronos took W3/W4/W8 and the TCN took W1/W2/W7 -- which "
            "says they are not redundant. This tests whether a head seeing both "
            "beats a head seeing either."
        ),
    )
    args = ap.parse_args()

    from trader.data.universe import active_tickers
    from trader.training.supervised import (
        build_panel_tensors,
        daily_scores,
        gate_verdict,
        summarise_daily,
    )
    from trader.training.walk_forward import compute_windows
    from trader.utils.seeding import seed_everything

    horizons = tuple(int(h) for h in args.horizons.split(","))
    seed_everything(args.seed)
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")

    index = json.loads((args.cache / "index.json").read_text())
    dates = [_as_date(d) for d in index["dates"]]
    tickers = list(index["tickers"])
    emb = np.load(args.cache / "embeddings.npy")            # [T, N, D] float16
    logger.info(
        f"cache {args.cache.name}: {emb.shape} amount_mode="
        f"{index.get('amount_mode')} coverage={index.get('coverage')}"
    )
    if tickers != active_tickers():
        logger.warning("cache ticker order differs from active_tickers(); using the cache's.")

    # Targets from the SAME panel, built exactly as R4 builds them.
    panel = pl.read_parquet(args.panel)
    # One feature column, because build_panel_tensors stacks them and an empty
    # list has nothing to stack. Its VALUES are never read here — only the
    # targets, mask, dates and fwd_raw it derives — but it must be a live column
    # or the liveness assertion inside will (correctly) refuse the panel.
    from trader.data.features import FEATURE_COLS

    feat_cols = list(FEATURE_COLS) if args.with_features else ["log_return_1d"]
    tens = build_panel_tensors(panel, tickers, feat_cols, horizons)
    if [str(d) for d in tens.dates] != [str(d) for d in dates]:
        # Align the cache onto the panel's calendar rather than assuming.
        pos = {d: i for i, d in enumerate(dates)}
        keep = [pos[d] for d in tens.dates if d in pos]
        if len(keep) != tens.n_days:
            raise SystemExit(
                f"cache covers {len(keep)} of the panel's {tens.n_days} dates; "
                "rebuild the cache from this panel."
            )
        emb = emb[keep]
    mask = tens.mask
    targets = tens.targets
    if args.with_features:
        # [T, N, D] embeddings ++ [T, N, F] features on the last axis. Features
        # are already the panel's own columns at date t, so no new lookahead is
        # introduced: the embedding's window also ends at t.
        emb = np.concatenate(
            [emb.astype(np.float32), tens.features.astype(np.float32)], axis=2
        )
        logger.info(f"combined representation: {emb.shape[2]} dims "
                    f"({emb.shape[2] - len(feat_cols)} kronos + {len(feat_cols)} features)")

    windows = compute_windows(
        data_start=date(2010, 1, 1), data_end=date(2024, 12, 31),
        train_years=5, val_months=12, test_months=12, purge_months=3,
        n_windows=12, step_months=12,
    )
    logger.info(f"{len(windows)} windows, {dates[0]}..{dates[-1]}, horizons {horizons}")

    d_idx = {d: i for i, d in enumerate(tens.dates)}

    def span(a: date, b: date) -> np.ndarray:
        return np.array([i for d, i in d_idx.items() if a <= d <= b], dtype=np.int64)

    per_window: dict[str, dict[int, object]] = {}
    for w in windows:
        tr = span(w.train_start, w.train_end)
        va = span(w.val_start, w.val_end)
        te = span(w.test_start, w.test_end)
        if min(tr.size, va.size, te.size) < 30:
            logger.warning(f"{w.name}: too few dates ({tr.size}/{va.size}/{te.size}); skipped")
            continue

        heads = _Heads(emb.shape[2], args.hidden, horizons).to(dev)
        opt = torch.optim.Adam(heads.parameters(), lr=args.lr)
        # Standardise the frozen embedding using TRAIN rows only — the same
        # discipline R4 applies to its features, and the reason a leak cannot
        # enter through the normaliser.
        tr_emb = emb[tr].astype(np.float32)
        fin = np.isfinite(tr_emb) & mask[tr][:, :, None]
        mu = np.where(fin, tr_emb, np.nan)
        e_mu = np.nanmean(mu, axis=(0, 1))
        e_sd = np.nanstd(mu, axis=(0, 1))
        e_sd = np.where(np.isfinite(e_sd) & (e_sd > 1e-8), e_sd, 1.0)

        def batch(idx: np.ndarray, i: int, j: int) -> tuple[torch.Tensor, torch.Tensor]:
            sl = idx[i:j]
            z = (emb[sl].astype(np.float32) - e_mu) / e_sd
            z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
            m = mask[sl] & np.isfinite(emb[sl][:, :, 0])
            return (
                torch.tensor(z, dtype=torch.float32, device=dev),
                torch.tensor(m, dtype=torch.bool, device=dev),
            )

        best, best_state, bad = -np.inf, None, 0
        for _ep in range(args.epochs):
            heads.train()
            perm = np.random.default_rng(args.seed + _ep).permutation(tr.size)
            for i in range(0, tr.size, args.batch_days):
                sel = tr[perm[i : i + args.batch_days]]
                z, m = batch(sel, 0, sel.size)
                out = heads(z)
                loss = torch.zeros((), device=dev)
                for h in horizons:
                    y = torch.tensor(targets[h][sel], dtype=torch.float32, device=dev)
                    ok = m & torch.isfinite(y)
                    if ok.any():
                        loss = loss + ((out[h][ok] - y[ok]) ** 2).mean()
                if loss.requires_grad:
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(heads.parameters(), 1.0)
                    opt.step()
            # early stop on mean validation rank IC, exactly as R4 does
            heads.eval()
            with torch.no_grad():
                z, m = batch(va, 0, va.size)
                out = {h: v.cpu().numpy() for h, v in heads(z).items()}
            ics = []
            for h in horizons:
                p = np.where(m.cpu().numpy(), out[h], np.nan)
                d = daily_scores(p, targets[h][va], tens.fwd_raw[h][va], min_cross_section=10)
                if d.ic.size:
                    ics.append(float(np.nanmean(d.ic)))
            v = float(np.mean(ics)) if ics else -np.inf
            if v > best:
                best, bad = v, 0
                best_state = {k: t.detach().clone() for k, t in heads.state_dict().items()}
            else:
                bad += 1
                if bad >= args.patience:
                    break
        if best_state is not None:
            heads.load_state_dict(best_state)

        heads.eval()
        with torch.no_grad():
            z, m = batch(te, 0, te.size)
            out = {h: v.cpu().numpy() for h, v in heads(z).items()}
        per_window[w.name] = {}
        for h in horizons:
            p = np.where(m.cpu().numpy(), out[h], np.nan)
            d = daily_scores(p, targets[h][te], tens.fwd_raw[h][te], min_cross_section=10)
            per_window[w.name][h] = summarise_daily(h, d, n_boot=500)
        line = "  ".join(
            f"{h}d IC {per_window[w.name][h].mean_ic:+.4f} (n={per_window[w.name][h].n_days})"
            for h in horizons
        )
        logger.info(f"  {w.name} val_ic {best:+.4f} | test {line}")

    gate = gate_verdict(per_window)

    base = json.loads(args.baseline.read_text()) if args.baseline.exists() else None
    print(f"\n{'window':<8}", end="")
    for h in horizons:
        print(f"{'kronos ' + str(h) + 'd':>14}{'tcn ' + str(h) + 'd':>12}{'delta':>9}", end="")
    print()
    print("-" * (8 + 35 * len(horizons)))
    deltas: dict[int, list[float]] = {h: [] for h in horizons}
    for name in sorted(per_window):
        print(f"{name:<8}", end="")
        for h in horizons:
            k = per_window[name][h].mean_ic
            b = np.nan
            if base:
                for wrow in base["windows"]:
                    ww = wrow["window"]
                    wn = ww["name"] if isinstance(ww, dict) else ww
                    if wn == name and str(h) in wrow["test"]:
                        b = float(wrow["test"][str(h)]["mean_ic"])
            if np.isfinite(b):
                deltas[h].append(k - b)
            print(f"{k:>14.4f}{b:>12.4f}{k - b:>+9.4f}", end="")
        print()
    print("-" * (8 + 35 * len(horizons)))
    for h in horizons:
        if deltas[h]:
            d = np.array(deltas[h])
            wins = int((d > 0).sum())
            print(
                f"  h={h}: mean delta {d.mean():+.4f}  kronos wins {wins}/{d.size} windows"
            )
    print()
    from trader.training.supervised import format_verdict

    print(format_verdict(gate))
    print(
        "\nK1 passes only if Kronos BEATS the from-scratch TCN. Clearing the gate "
        "on its own is not the bar — the TCN already does that."
    )


if __name__ == "__main__":
    main()
