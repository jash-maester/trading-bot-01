"""Backfill Kite 1-minute candles for the tick recorder's universe. Data only; no trading.

    uv run python scripts/kite_minute_backfill.py              # resumable; skips finished years
    uv run python scripts/kite_minute_backfill.py --probe 3    # first 3 names only

Intraday H0 (14_weekend_scoping_notes.md A7; future_HF_extension.md section 6).
The 19 Oct probe found minute history from 2015 onwards for listed names and
none in 2012.

Raw, as served, keyed by instrument token. **Not cleaned.** Kite reuses
instrument tokens: IRCTC (listed Oct 2019) returns 2015 bars at ~Rs 33 that
belong to another company. Every name must be clipped to its own listing
before use; that is a separate, documented step. Delisted names are absent
(survivorship), as in the daily data.

Output: data/intraday/minute/<SYMBOL>/<YEAR>.parquet (zstd), one row per
minute: date (IST, naive), open, high, low, close, volume, instrument_token.
A year file is written once the year is complete (or, for the current year,
on every run), so a crash loses at most one name-year. Kite's historical API
allows 3 requests/second; this paces at about 2.5 (latency + 0.05 s).

Kite is READ-ONLY here: `instruments` and `historical_data`. No order
endpoint is imported or called (CLAUDE.md).
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tick_recorder import _kite, universe  # noqa: E402

ROOT = Path("data/intraday/minute")
START = date(2015, 1, 1)
WINDOW_DAYS = 60          # Kite's per-request limit for minute candles
PACE_S = 0.05          # request latency (~0.35 s) does most of the pacing


def _fetch_year(k, token: int, year: int, today: date) -> pl.DataFrame:
    a, end = date(year, 1, 1), min(date(year, 12, 31), today)
    rows: list[dict] = []
    while a <= end:
        b = min(a + timedelta(days=WINDOW_DAYS - 1), end)
        for attempt in range(4):
            try:
                rows += k.historical_data(token, datetime.combine(a, datetime.min.time()),
                                          datetime.combine(b, datetime.max.time()), "minute")
                break
            except Exception as e:  # noqa: BLE001 -- rate limit / network: back off
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt + 1)
                print(f"    retry {token} {a}: {type(e).__name__}", flush=True)
        time.sleep(PACE_S)
        a = b + timedelta(days=1)
    if not rows:
        return pl.DataFrame()
    # Kite occasionally serves a junk volume (seen 2026-10-19: exactly 2**63,
    # which overflowed int64 and crashed the run after 39 names). Null it and
    # say so; the price fields of that minute are kept.
    bad = 0
    for r in rows:
        v = r.get("volume")
        if not isinstance(v, int) or not 0 <= v < 2**62:
            r["volume"], bad = None, bad + 1
    if bad:
        print(f"    {token} {year}: {bad} junk volume value(s) nulled", flush=True)
    return (pl.DataFrame(rows, schema_overrides={"volume": pl.Int64})
            .with_columns(pl.col("date").dt.convert_time_zone("Asia/Kolkata")
                          .dt.replace_time_zone(None),
                          pl.lit(token).alias("instrument_token"))
            .select("date", "open", "high", "low", "close", "volume", "instrument_token")
            .unique("date").sort("date"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--probe", type=int, default=None, help="only the first N names")
    a = ap.parse_args()
    _key, _tok, k = _kite()
    inst = {i["tradingsymbol"]: i["instrument_token"] for i in k.instruments("NSE")}
    names = universe(a.n)
    if a.probe:
        names = names[: a.probe]
    today = date.today()
    t0, n_req0 = time.time(), 0
    print(f"backfill {len(names)} names, {START}..{today}", flush=True)
    for i, t in enumerate(names, 1):
        s = t.removesuffix(".NS")
        tok = inst.get(s) or inst.get(f"{s}-BE")
        if not tok:
            print(f"[{i}/{len(names)}] {s}: no instrument today -- skipped", flush=True)
            continue
        out = ROOT / s
        out.mkdir(parents=True, exist_ok=True)
        got = []
        for y in range(START.year, today.year + 1):
            f = out / f"{y}.parquet"
            if y < today.year and (f.exists() or f.with_suffix(".empty").exists()):
                continue
            df = _fetch_year(k, tok, y, today)
            n_req0 += (min(date(y, 12, 31), today) - date(y, 1, 1)).days // WINDOW_DAYS + 1
            if df.height:
                tmp = f.with_suffix(".tmp.parquet")
                df.write_parquet(tmp, compression="zstd")
                tmp.replace(f)
                got.append(f"{y}:{df.height}")
            elif y < today.year:
                f.with_suffix(".empty").touch()   # marks a finished empty year
        el = time.time() - t0
        print(f"[{i}/{len(names)}] {s} token {tok}: {' '.join(got) or 'no new data'}"
              f"  ({el / 60:.1f} min, {n_req0 / max(el, 1):.2f} req/s)", flush=True)


if __name__ == "__main__":
    main()
