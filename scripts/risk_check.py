"""Do the RISK improvements survive out of sample, as the return ones did not?

    uv run python scripts/risk_check.py --in-sample audit/navs_long \
        --holdout audit/navs_holdout_pit

Every criterion this project has tested is a *return* criterion. R5 asks
whether the allocator beats the best baseline on return, and on that it fails
(`audit/R5_VERDICT_PIT.md`). But the in-sample tables also show the band-0.010
vol-stop arm at Sharpe 1.075 on a -0.285 drawdown against equal-weight's 0.617
and -0.589 -- nearly double the Sharpe on half the drawdown, discarded because
it gives up return.

That has never been checked out of sample, and this project's central finding
is that in-sample orderings REVERSE out of sample: R5's four arms reversed
perfectly, quarterly cadence vanished, and a factor scoring 2x the model in
sample lost to it on the holdout. So the prior here is that the risk
improvement does NOT survive either. This script measures it rather than
assuming it, on NAV series that already exist.

Reads `nav_*.parquet` written by `run_allocator.py ++allocator.nav_dir=...`.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl

TRADING_DAYS = 252.0


def _load(nav_dir: Path) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for f in sorted(nav_dir.glob("nav_*.parquet")):
        d = pl.read_parquet(f).drop_nulls("date").sort("date")
        nav = d["nav"].to_numpy().astype(np.float64)
        if nav.size >= 3 and np.all(nav > 0):
            out[f.stem.removeprefix("nav_")] = nav
    return out


def metrics(nav: np.ndarray) -> dict[str, float]:
    r = np.diff(np.log(nav))
    yrs = len(r) / TRADING_DAYS
    cagr = float(nav[-1] / nav[0]) ** (1.0 / yrs) - 1.0 if yrs > 0 else float("nan")
    sd = float(np.std(r, ddof=1))
    sharpe = float(np.mean(r)) / sd * np.sqrt(TRADING_DAYS) if sd > 0 else float("nan")
    peak = np.maximum.accumulate(nav)
    mdd = float((nav / peak - 1.0).min())
    # Downside deviation, because a drawdown claim should not rest on one path.
    dn = r[r < 0]
    sortino = (float(np.mean(r)) / float(np.std(dn, ddof=1)) * np.sqrt(TRADING_DAYS)
               if dn.size > 2 and float(np.std(dn, ddof=1)) > 0 else float("nan"))
    return {"cagr": cagr, "sharpe": sharpe, "sortino": sortino, "mdd": mdd,
            "days": float(len(r))}


def table(navs: dict[str, np.ndarray], label: str, focus: list[str]) -> dict[str, dict]:
    print(f"\n{'=' * 78}\n{label}   ({len(navs)} series)\n{'=' * 78}")
    m = {k: metrics(v) for k, v in navs.items()}
    base = next((k for k in m if k.startswith("equal_weight") and "frozen" not in k), None)
    print(f"{'arm':<46}{'Sharpe':>8}{'Sortino':>9}{'CAGR':>8}{'MDD':>8}")
    shown = [k for k in m if any(f in k for f in focus)] if focus else list(m)
    for k in sorted(shown, key=lambda k: -m[k]["sharpe"]):
        star = " <-" if base and k == base else ""
        print(f"{k[:44]:<46}{m[k]['sharpe']:>8.3f}{m[k]['sortino']:>9.3f}"
              f"{m[k]['cagr']:>+8.3f}{m[k]['mdd']:>+8.3f}{star}")
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-sample", type=Path, default=Path("audit/navs_long"))
    ap.add_argument("--holdout", type=Path, default=Path("audit/navs_holdout_pit"))
    ap.add_argument("--focus", nargs="*",
                    default=["equal_weight", "b0.01", "momentum"],
                    help="substrings of arm names to show; empty shows all")
    a = ap.parse_args()

    spans = {}
    for label, d in (("IN-SAMPLE 2005-2024", a.in_sample), ("HOLDOUT 2025-26", a.holdout)):
        if not d.exists():
            print(f"\n{label}: MISSING {d}")
            continue
        navs = _load(d)
        if not navs:
            print(f"\n{label}: no nav_*.parquet under {d}")
            continue
        spans[label] = table(navs, f"{label}  ({d})", a.focus)

    if len(spans) < 2:
        print("\nNeed both spans to answer the question. Stopping.")
        return

    (li, mi), (lh, mh) = list(spans.items())
    base_i = next((k for k in mi if k.startswith("equal_weight") and "frozen" not in k), None)
    base_h = next((k for k in mh if k.startswith("equal_weight") and "frozen" not in k), None)
    if not (base_i and base_h):
        print("\nNo equal_weight baseline in one of the spans; cannot compare.")
        return

    print(f"\n{'=' * 78}\nDOES THE RISK IMPROVEMENT SURVIVE?  (vs equal_weight, same span)"
          f"\n{'=' * 78}")
    print(f"{'arm':<38}{'dSharpe in':>12}{'dSharpe out':>13}{'dMDD in':>10}{'dMDD out':>10}")
    # Match arms across spans by the part of the name before the split suffix.
    def key(n: str) -> str:
        return n.split("_20d")[0]
    hmap = {key(k): k for k in mh}
    rows = []
    for k in sorted(mi, key=lambda k: -mi[k]["sharpe"]):
        hk = hmap.get(key(k))
        if hk is None or k == base_i:
            continue
        dsi = mi[k]["sharpe"] - mi[base_i]["sharpe"]
        dso = mh[hk]["sharpe"] - mh[base_h]["sharpe"]
        dmi = mi[k]["mdd"] - mi[base_i]["mdd"]
        dmo = mh[hk]["mdd"] - mh[base_h]["mdd"]
        rows.append((k, dsi, dso, dmi, dmo))
        print(f"{key(k)[:36]:<38}{dsi:>+12.3f}{dso:>+13.3f}{dmi:>+10.3f}{dmo:>+10.3f}")
    if rows:
        held = sum(1 for _, a_, b_, _, _ in rows if a_ > 0 and b_ > 0)
        print(f"\n  Sharpe improvement positive on BOTH spans: {held} of {len(rows)} arms")
        print("  A positive dMDD means a SHALLOWER drawdown than equal-weight.")
    print("\nLevels are not comparable across spans (different markets, lengths).")
    print("The question is only whether the SIGN and ORDERING hold, which is what")
    print("reversed for every return-based result this project has measured.")


if __name__ == "__main__":
    main()
