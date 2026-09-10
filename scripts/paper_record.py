"""Append today's paper-loop result to the record, and check determinism.

    uv run python scripts/paper_record.py --nav-dir audit/paper/latest \
        --record audit/paper/record.jsonl --snapshots audit/paper/snapshots \
        --freeze-date 2026-09-09 --data-date 2026-09-10

THE RECORD IS THE DAILY RETURN, NOT THE REPLAYED NAV.  Every run replays the
whole forward span from scratch (deterministic, ~40 s), which carries position
state without persisting it.  But back-adjustment is retroactive: a split
landing today rescales past ``macd``, ``atr_14`` and ``dollar_volume_20`` for
that name, which can move a past score, a past top-30 membership, and so the
replay's own history.  A real book does not re-decide yesterday.  So the value
of record for date D is ``log(nav_D / nav_{D-1})`` taken from the FIRST run
that included D -- self-consistent within that run, and never rewritten.  The
permutation rank is computed on those recorded returns.  Replay divergence
from the previous snapshot is measured and logged, not silently absorbed.
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import polars as pl

VOLSTOP_SIG = re.compile(r"^allocator_k30_b0\.01_rvolstop_monthly_20d")
VOLSTOP_NULL = re.compile(r"^null_signal_k30_b0\.01_rvolstop_s(\d+)_monthly_20d")
EW = re.compile(r"^equal_weight_monthly")


def _book(f: Path) -> str:
    return re.sub(r"_(full|holdout|oos_[a-z0-9_]+)$", "", f.stem.removeprefix("nav_"))


def _load(nav_dir: Path) -> dict[str, pl.DataFrame]:
    out = {}
    for f in sorted(nav_dir.glob("nav_*.parquet")):
        d = pl.read_parquet(f).drop_nulls("date").sort("date").select("date", "nav")
        out[_book(f)] = d
    if not out:
        raise SystemExit(f"no nav_*.parquet under {nav_dir}")
    return out


def _ret(d: pl.DataFrame, day: date) -> tuple[float, float] | None:
    i = d["date"].to_list()
    if day not in i:
        return None
    k = i.index(day)
    if k == 0:
        return None
    return float(d["nav"][k]), float(np.log(d["nav"][k] / d["nav"][k - 1]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nav-dir", type=Path, required=True)
    ap.add_argument("--record", type=Path, required=True)
    ap.add_argument("--snapshots", type=Path, required=True)
    ap.add_argument("--freeze-date", type=date.fromisoformat, required=True)
    ap.add_argument("--data-date", type=date.fromisoformat, required=True)
    a = ap.parse_args()

    books = _load(a.nav_dir)
    sig = next(b for b in books if VOLSTOP_SIG.match(b))
    nulls = sorted((b for b in books if VOLSTOP_NULL.match(b)),
                   key=lambda b: int(VOLSTOP_NULL.match(b).group(1)))
    ew = next(b for b in books if EW.match(b))
    dates = books[sig]["date"].to_list()
    if a.data_date not in dates:
        raise SystemExit(f"{a.data_date} is not in the replay ({dates[-1]} is its last date)")

    # ── determinism vs the previous snapshot ────────────────────────────────
    prev_dirs = sorted(p for p in a.snapshots.glob("*") if p.is_dir())
    det = {"previous": None, "overlap_days": 0, "max_abs_rel_diff": 0.0, "books_diverged": 0}
    if prev_dirs:
        prev = _load(prev_dirs[-1])
        det["previous"] = prev_dirs[-1].name
        worst, diverged = 0.0, 0
        for b, d in books.items():
            if b not in prev:
                continue
            j = d.join(prev[b].rename({"nav": "nav_prev"}), on="date", how="inner")
            if j.is_empty():
                continue
            det["overlap_days"] = max(det["overlap_days"], j.height)
            m = float((abs(j["nav"] - j["nav_prev"]) / j["nav_prev"]).max())
            worst = max(worst, m)
            diverged += int(m > 1e-9)
        det["max_abs_rel_diff"], det["books_diverged"] = worst, diverged

    # ── new record lines: every session after the last recorded one ─────────
    recorded = []
    if a.record.exists():
        recorded = [json.loads(ln) for ln in a.record.read_text().splitlines() if ln.strip()]
    last_rec = date.fromisoformat(recorded[-1]["date"]) if recorded else None
    new_days = [d for d in dates if d <= a.data_date and (last_rec is None or d > last_rec)]
    if last_rec is None:
        new_days = [a.data_date]          # first run: record the freeze date only
    run_ts = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    lines = []
    for day in new_days:
        entry = {"run_ts": run_ts, "date": day.isoformat(), "books": {}}
        for b, d in books.items():
            r = _ret(d, day)
            if r is not None:
                entry["books"][b] = {"nav": round(r[0], 6), "ret": round(r[1], 10)}
        lines.append(entry)
    with a.record.open("a") as fh:
        for e in lines:
            fh.write(json.dumps(e, separators=(",", ":")) + "\n")
    recorded.extend(lines)

    # ── the statistic, on RECORDED returns strictly after the freeze date ───
    scored = [e for e in recorded if date.fromisoformat(e["date"]) > a.freeze_date]
    def cum(b: str) -> float:
        return float(sum(e["books"].get(b, {}).get("ret", 0.0) for e in scored))
    cs, cn, ce = cum(sig), [cum(b) for b in nulls], cum(ew)
    rank = 1 + sum(1 for c in cn if c > cs) if scored else None

    # warm-up context, from THIS replay (pre-clock; not the statistic)
    S = books[sig]
    i0, i1 = dates.index(dates[0]), dates.index(a.freeze_date) if a.freeze_date in dates else None
    warm = None
    if i1 is not None:
        wcum = lambda b: float(np.log(books[b]["nav"][i1] / books[b]["nav"][i0]))  # noqa: E731
        ws, wn = wcum(sig), [wcum(b) for b in nulls]
        warm = {"span": [dates[0].isoformat(), a.freeze_date.isoformat()],
                "signal_cum": round(ws, 6), "null_mean": round(float(np.mean(wn)), 6),
                "rank": 1 + sum(1 for c in wn if c > ws), "n": len(wn) + 1}

    summary = {
        "run_ts": run_ts, "data_date": a.data_date.isoformat(),
        "freeze_date": a.freeze_date.isoformat(), "new_record_lines": len(lines),
        "scored_sessions": len(scored), "n_null": len(nulls),
        "signal_book": sig, "signal_cum": round(cs, 6), "equal_weight_cum": round(ce, 6),
        "null_cum_mean": round(float(np.mean(cn)), 6) if cn else None,
        "rank": rank, "warmup": warm, "determinism": det,
        "signal_nav_last": round(float(S["nav"][-1]), 6),
    }
    (a.nav_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
