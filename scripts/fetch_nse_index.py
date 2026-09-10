"""Fetch NIFTY 50 daily OHLC from NSE's free index archive and upsert ``^NSEI``.

    uv run python scripts/fetch_nse_index.py --from 2026-09-05 --to 2026-09-10

WHY. ``beta_nifty_60d`` needs the index series, and until now ``^NSEI`` came
only from Kite (``bhavcopy_to_store.py --index-from data/kite_ohlcv``). Kite
access tokens die at ~6 AM daily and need a browser login to mint -- which is
the one piece of daily friction a forward paper loop cannot carry. NSE
publishes every index close for free at

    https://archives.nseindia.com/content/indices/ind_close_all_DDMMYYYY.csv

no auth, same NIFTY 50 series, same closes to the paisa (checked 2026-09-04:
Kite 23897.7, NSE 23897.7). This makes the loop Kite-free end to end.

WEEKDAYS, NOT A CALENDAR. Sessions are enumerated as weekdays in [from, to] and
a 404 is a holiday or an unpublished day -- logged, never recorded as done, so
tomorrow's run retries it. A calendar derived from an existing panel cannot see
past the panel's last date, which is exactly the trap ``fetch_bhavcopy.py`` has.

BOTH STORES. ``bhavcopy_to_store.py`` rebuilds ``data/bhav_ohlcv`` from scratch
and copies ``^NSEI`` out of ``data/kite_ohlcv``, so the Kite store is the source
of truth and must be updated; the bhav store is updated too so a run that does
NOT rebuild the store still sees today's index. Schemas differ (Datetime vs
Date, column order, ``source`` vs ``turnover``) and ``OhlcvStore.save`` vstacks
onto the existing partition, so each is cast exactly.
"""
from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta
from pathlib import Path

import polars as pl
from loguru import logger

from trader.data.sources.nse_flows import NSEClient, NSENotFound
from trader.data.storage import OhlcvStore

INDEX_URL = "https://archives.nseindia.com/content/indices/ind_close_all_{ddmmyyyy}.csv"
INDEX_NAME = "Nifty 50"
TICKER = "^NSEI"


def _weekdays(lo: date, hi: date) -> list[date]:
    d, out = lo, []
    while d <= hi:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def _parse(text: str, day: date) -> pl.DataFrame | None:
    df = (pl.read_csv(text.encode(), infer_schema_length=0)
            .filter(pl.col("Index Name") == INDEX_NAME))
    if df.is_empty():
        return None
    row = df.row(0, named=True)
    got = datetime.strptime(row["Index Date"], "%d-%m-%Y").date()
    if got != day:
        raise ValueError(f"{day}: file carries {got}")
    return pl.DataFrame({
        "date": [day],
        "open": [float(row["Open Index Value"])], "high": [float(row["High Index Value"])],
        "low": [float(row["Low Index Value"])], "close": [float(row["Closing Index Value"])],
        "volume": [int(float(row["Volume"] or 0))],
    })


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", required=True)
    ap.add_argument("--to", dest="end", default=date.today().isoformat())
    ap.add_argument("--cache", type=Path, default=Path("data/raw/nse/indices"))
    ap.add_argument("--kite-store", type=Path, default=Path("data/kite_ohlcv"))
    ap.add_argument("--bhav-store", type=Path, default=Path("data/bhav_ohlcv"))
    ap.add_argument("--sleep", type=float, default=0.5)
    a = ap.parse_args()

    import time
    client = NSEClient()
    a.cache.mkdir(parents=True, exist_ok=True)
    frames: list[pl.DataFrame] = []
    missing: list[str] = []
    for day in _weekdays(date.fromisoformat(a.start), date.fromisoformat(a.end)):
        f = a.cache / f"ind_close_all_{day.strftime('%d%m%Y')}.csv"
        if f.exists():
            text = f.read_text(encoding="utf-8")
        else:
            try:
                text = client.get_text(INDEX_URL.format(ddmmyyyy=day.strftime("%d%m%Y")))
            except NSENotFound:
                missing.append(day.isoformat())
                continue
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"{day}: fetch failed, {exc}")
                missing.append(day.isoformat())
                continue
            f.write_text(text, encoding="utf-8")
            time.sleep(a.sleep)
        fr = _parse(text, day)
        if fr is not None:
            frames.append(fr)
    if missing:
        logger.info(f"no index file for {len(missing)} weekday(s) "
                    f"(holiday or not yet published): {missing}")
    if not frames:
        logger.info("nothing new")
        return

    idx = pl.concat(frames).sort("date")
    logger.info(f"NIFTY 50: {idx.height} session(s) {idx['date'].min()}..{idx['date'].max()}, "
                f"last close {idx['close'][-1]}")

    kite = idx.select(
        pl.col("date").cast(pl.Datetime("ns")), "open", "high", "low", "close",
        pl.col("volume").cast(pl.Int64), pl.col("close").alias("adj_close"),
        pl.lit(TICKER).alias("ticker"), pl.lit("nse").alias("source"),
    )
    bhav = idx.select(
        "date", pl.lit(TICKER).alias("ticker"), "open", "high", "low", "close",
        pl.col("close").alias("adj_close"), pl.col("volume").cast(pl.Int64),
        pl.lit(None, dtype=pl.Float64).alias("turnover"),
    )
    OhlcvStore(a.kite_store).save(kite)
    OhlcvStore(a.bhav_store).save(bhav)
    for root in (a.kite_store, a.bhav_store):
        last = OhlcvStore(root).load(tickers=[TICKER], start=datetime(2026, 1, 1))["date"].max()
        logger.info(f"{root}: ^NSEI now ends {last}")


if __name__ == "__main__":
    main()
