#!/usr/bin/env python
"""R5's gate: does the allocator beat its baseline by more than chance?

`PROGRESS.md` carries R5's acceptance criterion verbatim:

> Beats best R2 baseline net of cost+tax, **paired bootstrap CI excluding zero**

The margin has been measured many times — +0.159 CAGR over equal-weight in
sample, +0.304 on the 2025-26 holdout, after tax and every Zerodha charge — and
the interval never has been. Until it is, R5 is a measurement rather than a
passed gate, and `CLAUDE.md` rule 1 forbids building R6 or R7 on top of it.

    uv run python scripts/run_allocator.py ... ++allocator.nav_dir=audit/navs
    uv run python scripts/allocator_gate.py --nav-dir audit/navs

WHY PAIRED, AND WHY BLOCKS
--------------------------
Paired because both arms trade the same universe over the same days: their
returns share every market-wide move, and testing them as two independent
samples would drown a real difference in market volatility that cancels exactly.

Blocks because the daily difference is not i.i.d. Positions are held for a month
at monthly cadence, so consecutive daily excesses share the same book. The i.i.d.
bootstrap this reuses was measured against a strict null in
`bootstrap_mean_ci`'s own docstring and excluded zero in 33.7% of trials at a
nominal 5% for a persistent series. `--block` therefore defaults to the
rebalance period, and the lag-1 autocorrelation of the difference is reported so
a reader can judge whether that was enough.

PICK THE BASELINE DELIBERATELY. The criterion says "best R2 baseline", and the
default here is `EqualWeightRebalanced` only because it is the conventional bar —
it is **not** the strongest. `scripts/run_baselines.py` measured the grid on
2026-09-08 and `MomentumTopK` beats equal-weight on CAGR at monthly cadence
(0.273 against 0.257). Pass `--baseline momentum_topk` for the comparison R5's
criterion actually asks for; `audit/R5_GATE.md` reports both, and they do not
agree.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
from loguru import logger

_ANNUALISE = 252.0


def _returns(nav: np.ndarray) -> np.ndarray:
    """Daily log returns from a NAV path."""
    if nav.size < 3 or np.any(nav <= 0):
        raise ValueError("NAV path is too short or non-positive")
    return np.diff(np.log(nav))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nav-dir", type=Path, default=Path("audit/navs"))
    ap.add_argument("--baseline", default="", help="substring of the baseline file")
    ap.add_argument("--block", type=int, default=0,
                    help="moving-block length; 0 = infer from the cadence")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--out", type=Path, default=Path("audit/r5_gate.json"))
    args = ap.parse_args()

    from trader.training.supervised import bootstrap_mean_ci

    files = sorted(args.nav_dir.glob("nav_*.parquet"))
    if not files:
        raise SystemExit(f"no nav_*.parquet under {args.nav_dir}")
    _CADENCES = ("daily", "weekly", "monthly", "quarterly")

    def _arm_name(stem: str) -> str:
        """The strategy part of a nav filename, with cadence and split removed.

        `nav_equal_weight_frozen_monthly_oos_pit` -> `equal_weight_frozen`.
        The cadence token is the delimiter, because the strategy name itself
        contains underscores and nothing else marks where it ends.
        """
        parts = stem.removeprefix("nav_").split("_")
        for i, tok in enumerate(parts):
            if tok in _CADENCES:
                return "_".join(parts[:i])
        return "_".join(parts)

    def _match(name: str) -> list[Path]:
        """Files whose ARM NAME is exactly `name`.

        Substring matching is wrong here and quietly gave the wrong answer:
        `--baseline equal_weight` also matched `equal_weight_frozen`, so a
        Phase 3 run that asked for two different baselines tested the same one
        twice and printed both as though they differed. Prefix matching on `_`
        does not fix it either — `equal_weight_frozen_monthly` genuinely starts
        with `equal_weight_` — so the arm name has to be extracted first.
        """
        return [f for f in files if _arm_name(f.stem) == name]

    wanted = args.baseline or "equal_weight"
    base_files = _match(wanted)
    if not base_files:
        raise SystemExit(
            f"no baseline matching {wanted!r} among "
            f"{sorted(f.stem.removeprefix('nav_') for f in files)}"
        )
    if len({f.stem for f in base_files}) > 1:
        # Several arms share the prefix (different cadences, say). Take the
        # shortest — the least-qualified name — and say which, so a reader can
        # see what was compared instead of inferring it.
        base_files = sorted(base_files, key=lambda f: (len(f.stem), f.stem))
        logger.info(
            f"{wanted!r} matched {len(base_files)} arms; using "
            f"{base_files[0].stem.removeprefix('nav_')}"
        )
    base_path = base_files[0]
    bl = pl.read_parquet(base_path).drop_nulls("date").sort("date")
    logger.info(f"baseline {base_path.stem}: {bl.height:,} rows")

    cadence = "monthly" if "monthly" in base_path.stem else (
        "weekly" if "weekly" in base_path.stem else "daily")
    block = args.block or {"monthly": 21, "weekly": 5, "daily": 1}[cadence]
    logger.info(f"cadence {cadence} -> moving-block length {block}")

    results: list[dict[str, object]] = []
    arms = [f for f in files if f != base_path]
    def short(stem: str) -> str:
        """Trim the arm name to what varies, so the table stays readable."""
        n = stem.replace("nav_", "")
        for junk in (f"_{cadence}", f"_{split_tag}", "_b0.0", "allocator_"):
            n = n.replace(junk, "")
        return n

    split_tag = base_path.stem.split(f"_{cadence}_", 1)[-1]
    print(f"\n{'arm':<24}{'excess/yr':>12}{'95% CI':>24}{'t':>8}{'ACF1':>8}  verdict")
    print("-" * 84)
    for f in arms:
        arm = pl.read_parquet(f).drop_nulls("date").sort("date")
        j = bl.join(arm, on="date", how="inner", suffix="_arm")
        if j.height < 60:
            logger.warning(f"{f.stem}: only {j.height} shared days, skipped")
            continue
        rb = _returns(j["nav"].to_numpy().astype(np.float64))
        ra = _returns(j["nav_arm"].to_numpy().astype(np.float64))
        d = ra - rb                                   # the paired difference
        lo, hi = bootstrap_mean_ci(d, block=block, n_boot=args.n_boot, rng_seed=0)
        mean_ann = float(d.mean() * _ANNUALISE)
        lo_ann, hi_ann = lo * _ANNUALISE, hi * _ANNUALISE
        sd = float(d.std(ddof=1))
        t = float(d.mean() / (sd / np.sqrt(d.size))) if sd > 0 else float("nan")
        acf1 = (float(np.corrcoef(d[:-1], d[1:])[0, 1])
                if d.size > 2 and sd > 0 else float("nan"))
        excludes = bool(lo > 0.0 or hi < 0.0)
        verdict = "PASS" if excludes and mean_ann > 0 else "FAIL"
        print(f"{short(f.stem):<24}{mean_ann:>+12.4f}"
              f"{f'[{lo_ann:+.4f}, {hi_ann:+.4f}]':>24}{t:>8.2f}{acf1:>8.3f}  {verdict}")
        results.append({
            "arm": f.stem.replace("nav_", ""),
            "baseline": base_path.stem.replace("nav_", ""),
            "n_days": int(d.size),
            "mean_excess_log_return_daily": float(d.mean()),
            "mean_excess_annualised": mean_ann,
            "ci_low_annualised": float(lo_ann),
            "ci_high_annualised": float(hi_ann),
            "t_stat": t,
            "acf_lag1": acf1,
            "block": block,
            "n_boot": int(args.n_boot),
            "ci_excludes_zero": excludes,
            "verdict": verdict,
        })
    print("-" * 84)

    passed = [r for r in results if r["verdict"] == "PASS"]
    print(f"\n{len(passed)} of {len(results)} arm(s) clear a paired 95% CI "
          f"excluding zero against {base_path.stem.replace('nav_', '')}.")
    if not results:
        raise SystemExit("no arm had enough shared days to test")
    base_name = base_path.stem.replace("nav_", "")
    if "equal_weight" in base_name:
        print("\nBASELINE NOTE: which baseline is strongest depends on the "
              "UNIVERSE, and the\nanswer inverts between the two.\n"
              "  fixed 504 names   MomentumTopK 0.273 > equal_weight 0.257\n"
              "  point-in-time     equal_weight 0.119 > MomentumTopK -0.002\n"
              "Momentum's win on the fixed list was survivorship "
              "(audit/S1_SURVIVORSHIP.md).\nOn a point-in-time universe "
              "equal_weight IS the bar R5's criterion names.")

    payload = {
        "gate": "R5 — deterministic allocator",
        "criterion": "beats best R2 baseline net of cost+tax, paired bootstrap "
                     "CI excluding zero",
        "baseline_used": base_path.stem.replace("nav_", ""),
        "baseline_caveat": (
            "R5's criterion says 'best R2 baseline'. run_baselines.py measured "
            "the grid on 2026-09-08: MomentumTopK beats EqualWeightRebalanced on "
            "CAGR at monthly cadence (0.273 vs 0.257), so a run against "
            "equal_weight is NOT the criterion's comparison."
        ),
        "method": "paired daily log-return difference, moving-block bootstrap",
        "block": block,
        "n_boot": int(args.n_boot),
        "arms": results,
        "n_pass": len(passed),
        "n_arms": len(results),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
