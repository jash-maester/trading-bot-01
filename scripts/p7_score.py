"""Score a P7 historical run against its pre-registered criterion (audit/P7_COMPOSITE_15.md).

    uv run python scripts/p7_score.py audit/navs_p7/holdout_composite3 [--arm volstop]

Criterion, per period, for the volstop arm: cumulative log return ranks 1st of
21 against the same arm of the 20 random books, AND beats equal-weight.
Prints every arm for information; only the volstop line is the verdict.
"""
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import polars as pl


def cum(p: Path) -> tuple[float, float, str, str]:
    d = pl.read_parquet(p).sort("date")
    nav = d["nav"].to_numpy()
    peak = -math.inf
    mdd = 0.0
    for v in nav:
        peak = max(peak, v)
        mdd = min(mdd, v / peak - 1.0)
    return math.log(nav[-1] / nav[0]), mdd, str(d["date"][0]), str(d["date"][-1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("nav_dir")
    ap.add_argument("--arm", default="volstop")
    a = ap.parse_args()
    root = Path(a.nav_dir)
    ew = next(root.glob("nav_equal_weight_*.parquet"))
    ew_c, ew_mdd, d0, d1 = cum(ew)
    years = (pl.Series([d1]).str.to_date()[0] - pl.Series([d0]).str.to_date()[0]).days / 365.25
    print(f"{root}: {d0} .. {d1} ({years:.2f} y)")
    print(f"  equal_weight           cum log {ew_c:+.4f}  "
          f"CAGR {math.expm1(ew_c / years):+.2%}  MDD {ew_mdd:.1%}")
    for arm in ("none", "stop10", "stop15", "volstop"):
        sig = list(root.glob(f"nav_allocator_k*_r{arm}_*.parquet"))
        nulls = [p for p in root.glob(f"nav_null_signal_k*_r{arm}_s*_*.parquet")
                 if re.search(r"_s(\d+)_", p.name)]
        if not sig or not nulls:
            continue
        s_c, s_mdd, _, _ = cum(sig[0])
        n_c = sorted(cum(p)[0] for p in nulls)
        rank = 1 + sum(c > s_c for c in n_c)
        med = n_c[len(n_c) // 2]
        tag = "  <- verdict arm" if arm == a.arm else ""
        print(f"  {arm:8s} signal cum log {s_c:+.4f}  CAGR {math.expm1(s_c / years):+.2%}  "
              f"MDD {s_mdd:.1%}  rank {rank}/{len(n_c) + 1}  null median {med:+.4f}  "
              f"vs EW {s_c - ew_c:+.4f}{tag}")
        if arm == a.arm:
            ok = rank == 1 and s_c > ew_c
            print(f"  VERDICT ({arm}): {'PASS' if ok else 'FAIL'} "
                  f"(rank {rank}/{len(n_c) + 1}, vs EW {s_c - ew_c:+.4f})")


if __name__ == "__main__":
    main()
