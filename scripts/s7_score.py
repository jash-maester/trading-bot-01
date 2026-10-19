"""Score the S7 stop re-entry ban test (audit/S7_COOLDOWN_MOMENTUM.md).

    uv run python scripts/s7_score.py            # both periods under audit/navs_s7/

Per period and arm (vcd21, vcd10, vcd5): cumulative log return, CAGR, max
drawdown, and rank against that arm's own 20 random books.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl


def cum(p: Path) -> tuple[float, float, float]:
    d = pl.read_parquet(p)
    v = d[[c for c in d.columns if c != "date"][0]].to_numpy()
    yrs = (d["date"][-1] - d["date"][0]).days / 365.25
    return (float(np.log(v[-1] / v[0])), float((v / np.maximum.accumulate(v) - 1).min()),
            float((v[-1] / v[0]) ** (1 / yrs) - 1))


def main() -> None:
    for per in ("oos", "holdout"):
        root = Path(f"audit/navs_s7/{per}")
        ew = cum(next(root.glob("nav_equal_weight_*.parquet")))
        print(f"== {per}  EW cum {ew[0]:+.4f} CAGR {ew[2]:+.2%} MDD {ew[1]:.1%}")
        for arm in ("vcd21", "vcd10", "vcd5"):
            s = cum(next(root.glob(f"nav_allocator_k15_*_r{arm}_*.parquet")))
            nl = sorted(cum(p)[0] for p in root.glob(f"nav_null_signal_k15_*_r{arm}_s*_*.parquet"))
            print(f"  {arm}: cum {s[0]:+.4f}  CAGR {s[2]:+.2%}  MDD {s[1]:.1%}  "
                  f"rank {1 + sum(x > s[0] for x in nl)}/{len(nl) + 1}  "
                  f"null median {np.median(nl):+.4f}")


if __name__ == "__main__":
    main()
