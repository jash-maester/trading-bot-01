"""P3 adaptive book: stitch every refit's predictions into one signal, no lookahead.

    uv run python scripts/adaptive_signal.py     # -> data/signal/adaptive_signal/

The signal dated d (it trades at d+1's open) comes from the NEWEST OK refit in
audit/paper/adaptive/refits.jsonl whose data ends on or before d
(audit/P3_ADAPTIVE_SHADOW.md). Each refit is scored with scripts/predict_signal.py
on the forward panel, from its own fit date on.

Cheap by construction: a refit that has been superseded covers a closed date
range, so its predictions are computed once and cached next to its artefact
(live_predictions.parquet). Only the newest refit is re-scored each night
(~10 s on CPU).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import date
from pathlib import Path

import polars as pl

REG = Path("audit/paper/adaptive/refits.jsonl")
PANEL = Path("data/panels_forward/full.parquet")
OUT = Path("data/signal/adaptive_signal")


def _score(sig_dir: Path, start: date, tag: str) -> pl.DataFrame:
    """predict_signal for one refit from `start` on; returns its predictions."""
    out_root = Path("data/signal/_adaptive_scratch")
    r = subprocess.run(
        [
            sys.executable,
            "scripts/predict_signal.py",
            "--signal-dir",
            str(sig_dir),
            "--panel",
            str(PANEL),
            "--out-tag",
            tag,
            "--out-root",
            str(out_root),
            "--device",
            "cpu",
            "--start-date",
            start.isoformat(),
            "--batch-days",
            "6",
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "TRADER_TCN_FULL": "1"},
    )
    if r.returncode != 0:
        raise SystemExit(f"predict_signal failed for {sig_dir}:\n{r.stderr[-1500:]}")
    return pl.read_parquet(out_root / tag / "predictions.parquet")


def main() -> None:
    refits = (
        [json.loads(x) for x in REG.read_text().splitlines() if x.strip()] if REG.exists() else []
    )
    refits = sorted((r for r in refits if r.get("status") == "OK"), key=lambda r: r["fit_date"])
    if not refits:
        raise SystemExit("no OK refit registered yet -- run scripts/refit_adaptive.py on the host")
    panel_last = pl.scan_parquet(PANEL).select(pl.col("date").max()).collect().item()
    parts = []
    for i, r in enumerate(refits):
        lo = date.fromisoformat(r["fit_date"])
        hi = date.fromisoformat(refits[i + 1]["fit_date"]) if i + 1 < len(refits) else None
        if lo > panel_last:
            continue
        sig_dir = Path(r["signal_dir"])
        cache = sig_dir / "live_predictions.parquet"
        if hi is not None and cache.exists():
            p = pl.read_parquet(cache)
        else:
            p = _score(sig_dir, lo, f"live_{r['tag']}")
            if hi is not None:  # superseded: its range is closed, cache it
                p = p.filter(pl.col("date") < hi)
                p.write_parquet(cache)
        p = p.filter(pl.col("date") >= lo)
        if hi is not None:
            p = p.filter(pl.col("date") < hi)
        parts.append(p.with_columns(pl.lit(r["tag"]).alias("model")))
    if not parts:
        raise SystemExit(f"no refit covers the forward panel yet (panel ends {panel_last})")
    stitched = pl.concat(parts, how="vertical").sort(["date", "ticker"])
    OUT.mkdir(parents=True, exist_ok=True)
    stitched.drop("model").write_parquet(OUT / "predictions.parquet")
    idx = json.loads((Path(refits[-1]["signal_dir"]) / "index.json").read_text())
    idx.update(
        {
            "inference_only": True,
            "stitched_from": [r["tag"] for r in refits],
            "dates": sorted({str(d) for d in stitched["date"].unique().to_list()}),
        }
    )
    (OUT / "index.json").write_text(json.dumps(idx, indent=2) + "\n")
    per = stitched.group_by("model").agg(
        pl.col("date").min().alias("from"), pl.col("date").max().alias("to"), pl.len().alias("rows")
    )
    print(
        f"adaptive_signal: {stitched.height:,} rows, "
        f"{stitched['date'].min()}..{stitched['date'].max()}"
    )
    for m, a, b, n in per.sort("from").iter_rows():
        print(f"  {m}: {a}..{b} ({n:,} rows)")


if __name__ == "__main__":
    main()
