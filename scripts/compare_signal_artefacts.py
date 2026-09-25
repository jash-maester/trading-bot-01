"""Statistical parity of two trained signal artefacts (e.g. CUDA vs MPS).

    uv run python scripts/compare_signal_artefacts.py --a r4_pit_long --b r4_pit_long_mps \
        --panel data/panels_bhav/oos_r4_pit_long.parquet

Bit-exact parity across GPU backends is not achievable, so this asks whether
the two models are the SAME MODEL statistically:

1. gate verdicts and window-level IC stats (data/signal/*/gate.json)
2. per-window 20d IC, side by side, and their correlation across windows
3. per-date Spearman correlation of the two models' r_hat_20d on identical
   rows -- how similarly they rank the same names on the same day
4. top-30 overlap per date -- whether they would buy the same book

Pass criteria, stated before the run: same gate verdict; |delta mean 20d IC|
below the window-level standard error; median per-date rank correlation
>= 0.8. Anything less is reported as NOT at parity, with the numbers.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path("data/signal")


def _gate(tag: str) -> dict:
    return json.loads((ROOT / tag / "gate.json").read_text())


def _wl(g: dict, h: str) -> dict:
    return g["window_level"].get(h) or g["window_level"].get(int(h))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--panel", type=Path, required=True)
    ap.add_argument("--k", type=int, default=30)
    a = ap.parse_args()

    ga, gb = _gate(a.a), _gate(a.b)
    print(f"{'':<24}{a.a:>22}{a.b:>22}")
    print(f"{'gate verdict':<24}{ga['verdict']:>22}{gb['verdict']:>22}")
    ok = ga["verdict"] == gb["verdict"]
    for h in ("5", "20"):
        wa, wb = _wl(ga, h), _wl(gb, h)
        print(f"{h + 'd mean IC':<24}{wa['mean_ic']:>+22.4f}{wb['mean_ic']:>+22.4f}")
        print(f"{h + 'd window t':<24}{wa['t_stat']:>22.2f}{wb['t_stat']:>22.2f}")
        pos_a = f"{wa['n_positive']}/{wa['n_windows']}"
        pos_b = f"{wb['n_positive']}/{wb['n_windows']}"
        print(f"{h + 'd windows positive':<24}{pos_a:>22}{pos_b:>22}")
    wa, wb = _wl(ga, "20"), _wl(gb, "20")
    se = wa["sd_ic"] / math.sqrt(wa["n_windows"])
    d_ic = wb["mean_ic"] - wa["mean_ic"]
    ok &= abs(d_ic) < se
    ia, ib = wa["window_ics"], wb["window_ics"]
    names = [w for w in ia if w in ib]
    va, vb = np.array([ia[w] for w in names]), np.array([ib[w] for w in names])
    print("\nper-window 20d IC:")
    for w, x, y in zip(names, va, vb, strict=True):
        print(f"  {w:<6}{x:>+10.4f}{y:>+10.4f}{y - x:>+10.4f}")
    print(f"  corr across windows {np.corrcoef(va, vb)[0, 1]:+.3f};  delta mean {d_ic:+.4f} "
          f"vs window SE {se:.4f}")

    def _pred(tag: str, name: str) -> pl.DataFrame:
        return pl.read_parquet(ROOT / tag / "predictions.parquet").select(
            "date", "ticker", pl.col("r_hat_20d").alias(name))

    pa, pb = _pred(a.a, "a"), _pred(a.b, "b")
    tr = pl.scan_parquet(a.panel).select("date", "ticker", "is_tradeable").collect()
    j = pa.join(pb, on=["date", "ticker"]).join(tr, on=["date", "ticker"]).filter(pl.col("is_tradeable"))
    rc = (j.group_by("date").agg(pl.corr("a", "b", method="spearman").alias("rc"), pl.len().alias("n"))
            .filter(pl.col("n") >= 10))["rc"].drop_nulls().to_numpy()
    k = a.k
    top = (j.with_columns(pl.col("a").rank("ordinal", descending=True).over("date").alias("ra"),
                          pl.col("b").rank("ordinal", descending=True).over("date").alias("rb"))
             .group_by("date").agg(((pl.col("ra") <= k) & (pl.col("rb") <= k)).sum().alias("both")))
    ov = (top["both"].to_numpy() / k)
    med = float(np.median(rc))
    ok &= med >= 0.8
    print(f"\nidentical rows: {j.height:,} over {j['date'].n_unique():,} dates")
    print(f"per-date rank corr of r_hat_20d: median {med:.3f}, p10 {np.percentile(rc, 10):.3f}, "
          f"p90 {np.percentile(rc, 90):.3f}")
    print(f"top-{k} overlap per date: median {np.median(ov):.0%}, mean {ov.mean():.0%}")
    print(f"\nPARITY: {'AT PARITY' if ok else 'NOT AT PARITY'} "
          "(criteria: same verdict; |d mean 20d IC| < window SE; median rank corr >= 0.8)")


if __name__ == "__main__":
    main()
