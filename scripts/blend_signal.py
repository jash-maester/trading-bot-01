#!/usr/bin/env python
"""Fold the fundamental score into an R4 signal, and gate the result.

`audit/F2_FUNDAMENTAL_IC.md` establishes that a three-feature earnings-change
score is near-independent of R4 (cross-sectional corr +0.022) and that a 50/50
rank blend lifts IC ~31% at both horizons. This writes that blend as a signal
artefact the allocator can consume, so the lift can be measured in CAGR after
costs and tax rather than in IC.

    uv run python scripts/blend_signal.py --signal-tag r4_v2 --out-tag r4_v2_fund

HOW THE BLEND IS APPLIED, AND WHY NOT THE OBVIOUS WAY
-----------------------------------------------------
Only ~20% of the panel has a filing on a given day, so a blended score has to
say something about names it knows nothing about. The obvious approach — score
uncovered names with R4's rank ``s`` and covered names with ``0.5f + 0.5s`` —
is quietly broken: averaging pulls every covered name toward 0.5, shrinking
their spread, so covered names become systematically less likely to reach an
extreme. With K=20 selected from 504 that is a real distortion, and it acts
against exactly the names we have extra information about.

Instead the fundamental score is applied as a **reordering within the covered
subset**. Covered names are re-sorted by the blended rank, and R4's own values
are re-assigned among them in that new order. Uncovered names are not touched
at all. Two properties follow, and both matter:

* the marginal distribution of the signal is **identical** to R4's, so band and
  threshold logic downstream behaves exactly as it did;
* on a day where nothing has been filed, the output is **bit-identical** to
  R4 — the blend cannot manufacture a difference out of absent data.

The gate is recomputed from scratch on the blended predictions with the same
`window-level-t/v2` rule R4 is judged by. A blend that does not clear it does
not ship, and `eval_allocator_rl.py` will refuse to quote its numbers.
"""
from __future__ import annotations

import argparse
import json
import warnings
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
from loguru import logger
from scipy.stats import rankdata

#: The three features that survived the standalone IC pass 6/6 windows positive
#: at both horizons (`audit/F2_FUNDAMENTAL_IC.md` §2). The level features are
#: deliberately excluded: not one of them was significant, and net margin flips
#: sign at W5 and stays flipped.
SCORE_COLS: tuple[str, ...] = (
    "f_profit_growth_yoy",
    "f_eps_growth_yoy",
    "f_net_margin_change_yoy",
)


def _rank01(x: np.ndarray) -> np.ndarray:
    """Cross-sectional rank in [0, 1], NaN-preserving."""
    out = np.full_like(x, np.nan, dtype=np.float64)
    v = np.isfinite(x)
    if v.sum() < 2:
        return out
    out[v] = (rankdata(x[v]) - 1.0) / (v.sum() - 1.0)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--signal-tag", default="r4_v2")
    ap.add_argument("--out-tag", default="r4_v2_fund")
    ap.add_argument("--signal-root", type=Path, default=Path("data/signal"))
    ap.add_argument("--figures", type=Path, default=Path("data/ext/fundamentals.parquet"))
    ap.add_argument("--panel", type=Path, default=Path("data/panels_kite/full.parquet"))
    ap.add_argument("--weight", type=float, default=0.5,
                    help="weight on the fundamental rank inside the covered subset")
    ap.add_argument("--min-names", type=int, default=50,
                    help="days with a thinner covered subset are passed through unchanged")
    ap.add_argument("--all-change-features", action="store_true",
                    help="use every CHANGE_COLS feature instead of the three that "
                         "survived the IC pass — the control for selection bias, "
                         "since those three were picked on the same windows the "
                         "blend is then scored over")
    ap.add_argument("--data-end", default="2024-12-31", help="gate window horizon end")
    args = ap.parse_args()

    from trader.data.fundamental_features import (
        CHANGE_COLS,
        as_of_panel,
        attach_earnings_yield,
        build_quarterly_features,
    )
    from trader.data.universe import active_tickers
    from trader.training.supervised import (
        build_panel_tensors,
        daily_scores,
        format_verdict,
        gate_json_payload,
        gate_verdict,
        summarise_daily,
    )
    from trader.training.walk_forward import compute_windows

    score_cols = tuple(CHANGE_COLS) if args.all_change_features else SCORE_COLS
    logger.info(f"blending on {len(score_cols)} feature(s): {', '.join(score_cols)}")

    src = args.signal_root / args.signal_tag / "predictions.parquet"
    sig = pl.read_parquet(src)
    hcols = [c for c in sig.columns if c.startswith("r_hat_")]
    horizons = [int(c.removeprefix("r_hat_").removesuffix("d")) for c in hcols]
    logger.info(f"{src}: {sig.height:,} rows, horizons {horizons}")

    tickers = active_tickers()
    panel = pl.read_parquet(args.panel)
    feats = attach_earnings_yield(
        build_quarterly_features(pl.read_parquet(args.figures)),
        panel.select(["ticker", "date", "close"]),
    )

    dates = sorted(sig["date"].unique().to_list())
    T, N = len(dates), len(tickers)
    d_idx = {d: i for i, d in enumerate(dates)}
    t_idx = {t: i for i, t in enumerate(tickers)}

    grids = as_of_panel(feats, list(dates), tickers, score_cols)
    with warnings.catch_warnings():
        # All-NaN rows are the norm: most names have not filed on most days.
        warnings.simplefilter("ignore", RuntimeWarning)
        fscore = np.nanmean(
            np.stack([np.stack([_rank01(grids[c][t]) for t in range(T)])
                      for c in score_cols]),
            axis=0,
        )

    r4 = {h: np.full((T, N), np.nan, dtype=np.float64) for h in horizons}
    for row in sig.iter_rows(named=True):
        i, j = d_idx.get(row["date"]), t_idx.get(row["ticker"])
        if i is None or j is None:
            continue
        for h, c in zip(horizons, hcols):
            v = row[c]
            if v is not None:
                r4[h][i, j] = float(v)

    out = {h: g.copy() for h, g in r4.items()}
    n_reordered = 0
    n_passthrough = 0
    for t in range(T):
        cov = np.isfinite(fscore[t])
        for h in horizons:
            idx = np.flatnonzero(cov & np.isfinite(r4[h][t]))
            if idx.size < args.min_names:
                continue
            f = _rank01(fscore[t][idx])
            s = _rank01(r4[h][t][idx])
            blended = args.weight * f + (1.0 - args.weight) * s
            # Re-assign R4's own values among the covered names in blended
            # order. `order[k]` is the name that should hold the k-th smallest
            # R4 value, so sorting the values and scattering them by that order
            # permutes without changing the multiset.
            order = np.argsort(blended, kind="stable")
            out[h][t, idx[order]] = np.sort(r4[h][t][idx])
        if np.flatnonzero(cov).size >= args.min_names:
            n_reordered += 1
        else:
            n_passthrough += 1

    logger.info(
        f"{n_reordered:,} days reordered, {n_passthrough:,} passed through "
        f"unchanged (covered subset thinner than {args.min_names})"
    )
    ident = all(
        np.array_equal(out[h][:1], r4[h][:1], equal_nan=True) or n_reordered == T
        for h in horizons
    )
    logger.info(f"first-day identity to R4 holds: {ident}")

    dst = args.signal_root / args.out_tag
    dst.mkdir(parents=True, exist_ok=True)
    recs = {"date": [], "ticker": [], **{c: [] for c in hcols}}
    for i, d in enumerate(dates):
        live = np.flatnonzero(np.isfinite(out[horizons[0]][i]))
        for j in live:
            recs["date"].append(d)
            recs["ticker"].append(tickers[j])
            for h, c in zip(horizons, hcols):
                recs[c].append(float(out[h][i, j]))
    blended_df = pl.DataFrame(recs)
    blended_df.write_parquet(dst / "predictions.parquet")
    logger.info(f"wrote {dst / 'predictions.parquet'}: {blended_df.height:,} rows")

    # ── gate the blend on its own merits, same rule R4 is judged by ──────────
    tens = build_panel_tensors(panel, tickers, ["log_return_1d"], tuple(horizons))
    p_idx = {d: i for i, d in enumerate(tens.dates)}
    windows = compute_windows(
        data_start=date(2010, 1, 1),
        data_end=date.fromisoformat(args.data_end),
        train_years=5, val_months=12, test_months=12, purge_months=3,
        n_windows=12, step_months=12,
    )
    aligned = {h: np.full((len(tens.dates), N), np.nan) for h in horizons}
    for i, d in enumerate(dates):
        k = p_idx.get(d)
        if k is not None:
            for h in horizons:
                aligned[h][k] = out[h][i]

    per_window: dict[str, dict[int, object]] = {}
    pooled_daily: dict[int, list[np.ndarray]] = {h: [] for h in horizons}
    for w in windows:
        te = np.array([i for d, i in p_idx.items() if w.test_start <= d <= w.test_end],
                      dtype=np.int64)
        if te.size < 30:
            continue
        hm: dict[int, object] = {}
        for h in horizons:
            ds = daily_scores(aligned[h][te].astype(np.float32),
                              tens.targets[h][te], tens.fwd_raw[h][te])
            if ds.ic.size == 0:
                continue
            hm[h] = summarise_daily(h, ds)
            pooled_daily[h].append(ds.ic)
        if hm:
            per_window[w.name] = hm

    from trader.training.supervised import DailyScores

    pooled: dict[int, object] = {}
    for h in horizons:
        if not pooled_daily[h]:
            continue
        ic = np.concatenate(pooled_daily[h])
        pooled[h] = summarise_daily(
            h, DailyScores(day_index=np.arange(ic.size), ic=ic,
                           decile_spread=np.zeros_like(ic),
                           n_skipped_thin=0, n_skipped_degenerate=0)
        )

    gate = gate_verdict(per_window)  # type: ignore[arg-type]
    payload = gate_json_payload(gate, pooled, len(per_window))  # type: ignore[arg-type]
    payload["blend"] = {
        "source_signal": str(src),
        "weight_on_fundamentals": args.weight,
        "score_cols": list(score_cols),
        "days_reordered": n_reordered,
        "days_passed_through": n_passthrough,
        "note": "reordering within the covered subset; marginal distribution "
                "identical to the source signal",
    }
    (dst / "gate.json").write_text(json.dumps(payload, indent=2) + "\n")
    print("\n" + format_verdict(gate))
    print(f"\ngate.json written to {dst / 'gate.json'}")

    src_gate = args.signal_root / args.signal_tag / "gate.json"
    if src_gate.exists():
        old = json.loads(src_gate.read_text())
        print(f"\n{'horizon':<10}{args.signal_tag:>16}{args.out_tag:>18}{'delta':>12}")
        print("-" * 56)
        for h in horizons:
            a = old.get(f"mean_ic_{h}d")
            b = payload.get(f"mean_ic_{h}d")
            if a is None or b is None:
                continue
            print(f"{h:<10}{a:>16.4f}{b:>18.4f}{b - a:>+12.4f}")
        print("-" * 56)
        print("Pooled OOS mean rank IC. The gate verdict above is the "
              "window-level test, which is the one that decides.")


if __name__ == "__main__":
    main()
