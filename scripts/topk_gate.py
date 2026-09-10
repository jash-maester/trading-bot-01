"""Gate a signal on what the allocator actually consumes: top-K forward return.

    uv run python scripts/topk_gate.py --signal r4_pit_long \
        --panel data/panels_bhav/oos_r4_pit_long.parquet --k 30 --horizon 20

WHY. R4's gate is full-cross-section rank IC. The allocator buys the top 30.
The two were measured to disagree: `r4_pit_long_holdout` has the HIGHER
holdout IC (+0.0459 against +0.0338) and produced the WORSE portfolio, and the
fundamentals work raised IC without moving top-K return
(`audit/S4_NULL_CONTROL.md`, `audit/F3_FUNDAMENTALS_VERDICT.md`). A gate that
scores the whole ranking cannot see that. This one scores the slice that gets
bought, net of a cost proxy, against a distribution of RANDOM top-K books
built on the identical support -- the same control the allocator now runs.

This is hygiene, not search: it re-scores artefacts that already exist so the
divergence is a measured column. Nothing is trained against it.

STATISTIC. On every h-th scorable date in each walk-forward test window
(non-overlapping horizons), take the K highest-scored tradeable names with a
prediction and a forward return; excess = mean forward h-day log return of
those K minus the mean over all eligible names that date; cost = turnover of
the K-set since the previous scoring date x a round-trip proxy (default 23
bps: STT 10 bps each leg, stamp 1.5 bps, a DP allowance). Window value = mean
of (excess - cost). Verdict at the window level, mirroring the IC gate:

    PASS  iff  mean over windows > 0, t > t_crit(95%), >= 75% windows positive,
          AND the signal's per-window value beats the mean of N random books
          with paired t > t_crit.

Windows come from the artefact's own summary.json so this scores exactly the
test spans the IC gate scored; tickers come from the panel, not a fixed list.
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
from loguru import logger

from trader.training.supervised import build_panel_tensors, t_critical_95


def _windows_from_summary(summary: dict) -> list[tuple[str, date, date]] | None:
    for key in ("windows", "window_results", "per_window", "results"):
        v = summary.get(key)
        if not v:
            continue
        entries = v if isinstance(v, list) else list(v.values())
        out = []
        for e in entries:
            if not isinstance(e, dict):
                continue
            # summary.json nests the spans: {"window": {"name","test_start",...}, ...}
            w = e.get("window") if isinstance(e.get("window"), dict) else e
            name = w.get("name") or f"W{len(out) + 1}"
            ts, te = w.get("test_start"), w.get("test_end")
            if ts and te:
                out.append((str(name), date.fromisoformat(str(ts)[:10]),
                            date.fromisoformat(str(te)[:10])))
        if out:
            return out
    return None


def _grid_from_predictions(
    path: Path, col: str, d_idx: dict, t_idx: dict, shape: tuple[int, int]
) -> np.ndarray:
    g = np.full(shape, np.nan)
    df = pl.read_parquet(path).select("date", "ticker", col).drop_nulls(col)
    di = df["date"].to_list()
    ti = df["ticker"].to_list()
    vi = df[col].to_numpy()
    for d, t, v in zip(di, ti, vi, strict=True):
        i, j = d_idx.get(d), t_idx.get(t)
        if i is not None and j is not None:
            g[i, j] = float(v)
    return g


def _window_values(g, fwd, mask, dates, d_idx, windows, k, h, cost_rt, min_days):
    """Per-window mean of (top-K excess - cost) on stride-h dates."""
    out: dict[str, float] = {}
    for name, ts, te in windows:
        te_idx = [d_idx[d] for d in dates if ts <= d <= te]
        if len(te_idx) < min_days:
            continue
        vals, prev = [], None
        for i in te_idx[::h]:
            v = np.isfinite(g[i]) & np.isfinite(fwd[i]) & mask[i]
            if int(v.sum()) < k + 5:
                continue
            idx = np.flatnonzero(v)
            top = set(idx[np.argsort(g[i][idx])[-k:]].tolist())
            excess = float(fwd[i][list(top)].mean() - fwd[i][idx].mean())
            turn = 1.0 if prev is None else len(top - prev) / k
            vals.append(excess - turn * cost_rt)
            prev = top
        if len(vals) >= 3:
            out[name] = float(np.mean(vals))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--signal", required=True)
    ap.add_argument("--panel", type=Path, required=True)
    ap.add_argument("--signal-root", type=Path, default=Path("data/signal"))
    ap.add_argument("--k", type=int, default=30)
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--n-null", type=int, default=20)
    ap.add_argument("--cost-bps", type=float, default=23.0, help="round-trip cost proxy, bps")
    ap.add_argument("--min-days", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    sdir = a.signal_root / a.signal
    summary = json.loads((sdir / "summary.json").read_text())
    windows = _windows_from_summary(summary)
    if windows is None:
        raise SystemExit(f"{sdir/'summary.json'} carries no window test spans; "
                         "cannot score the gate's windows")
    logger.info(f"{len(windows)} windows from summary.json: "
                f"{windows[0][0]} {windows[0][1]}..{windows[-1][2]}")

    panel = pl.read_parquet(a.panel)
    tickers = sorted(panel["ticker"].unique().to_list())
    h = a.horizon
    tens = build_panel_tensors(panel, tickers, ["log_return_1d"], (h,))
    d_idx = {d: i for i, d in enumerate(tens.dates)}
    t_idx = {t: j for j, t in enumerate(tickers)}
    fwd, mask = tens.fwd_raw[h], tens.mask

    g = _grid_from_predictions(sdir / "predictions.parquet", f"r_hat_{h}d", d_idx, t_idx, fwd.shape)
    logger.info(f"{a.signal}: {np.isfinite(g).mean():.1%} of the "
                f"[{len(tens.dates)}x{len(tickers)}] grid populated")
    cost = a.cost_bps / 1e4

    sig = _window_values(g, fwd, mask, tens.dates, d_idx, windows, a.k, h, cost, a.min_days)
    rng_grids = []
    for s in range(a.n_null):
        noise = np.random.default_rng(a.seed + s).normal(0.0, 0.02, g.shape)
        rng_grids.append(np.where(np.isfinite(g), noise, np.nan))
    nulls = [_window_values(n, fwd, mask, tens.dates, d_idx, windows, a.k, h, cost, a.min_days)
             for n in rng_grids]

    names = [w for w in sig if all(w in n for n in nulls)]
    if len(names) < 3:
        raise SystemExit(f"only {len(names)} scorable windows")
    sv = np.array([sig[w] for w in names])
    nv = np.array([[n[w] for w in names] for n in nulls])          # [n_null, n_win]
    nmean = nv.mean(axis=0)
    diff = sv - nmean
    def tstat(x):
        sd = x.std(ddof=1)
        return float(x.mean() / (sd / np.sqrt(x.size))) if sd > 0 else float("nan")
    crit = t_critical_95(len(names) - 1)
    t_sig, t_diff = tstat(sv), tstat(diff)
    pos = int((sv > 0).sum())
    rank = 1 + int((nv.mean(axis=1) > sv.mean()).sum())
    verdict = "PASS" if (sv.mean() > 0 and t_sig > crit and pos >= 0.75 * len(names)
                         and diff.mean() > 0 and t_diff > crit) else "FAIL"

    print(f"\nTop-{a.k} gate, {h}d, {a.signal}, cost proxy {a.cost_bps:.0f} bps round trip")
    print(f"{'window':<8}{'signal':>12}{'null mean':>12}{'diff':>12}"
          f"{'null min':>10}{'null max':>10}")
    print("-" * 64)
    for i, w in enumerate(names):
        print(f"{w:<8}{sv[i]:>+12.5f}{nmean[i]:>+12.5f}{diff[i]:>+12.5f}"
              f"{nv[:, i].min():>+10.4f}{nv[:, i].max():>+10.4f}")
    print("-" * 64)
    print(f"{'mean':<8}{sv.mean():>+12.5f}{nmean.mean():>+12.5f}{diff.mean():>+12.5f}")
    print(f"\n  signal: mean {sv.mean():+.5f} per {h}d, t {t_sig:.2f} vs crit {crit:.2f}, "
          f"{pos}/{len(names)} windows > 0")
    print(f"  vs {a.n_null} random books: paired diff {diff.mean():+.5f}, t {t_diff:.2f}, "
          f"rank {rank}/{a.n_null + 1} on the window mean")
    print(f"\n  TOP-K GATE: {verdict}")

    out = {"gate": "topk-forward-return/v1", "signal": a.signal, "k": a.k, "horizon": h,
           "cost_bps_roundtrip": a.cost_bps, "n_windows": len(names), "windows": names,
           "signal_per_window": [float(x) for x in sv],
           "null_mean_per_window": [float(x) for x in nmean],
           "mean": float(sv.mean()), "t": t_sig, "t_crit": crit, "positive": pos,
           "diff_vs_null_mean": float(diff.mean()), "t_diff": t_diff, "rank_vs_null": rank,
           "n_null": a.n_null, "verdict": verdict}
    (sdir / "topk_gate.json").write_text(json.dumps(out, indent=2))
    logger.info(f"wrote {sdir/'topk_gate.json'}")


if __name__ == "__main__":
    main()
