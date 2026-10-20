"""Record live NSE ticks with 5-level depth for liquid names. Data only; no trading.

    uv run python scripts/tick_recorder.py            # cron 09:00, retried every 10 min to 15:00

Intraday Phase 0 (14_weekend_scoping_notes.md §A7): the spread and depth data
this project has never had, and which every intraday test and the VWAP/TWAP
execution study need.

* Universe: the N most liquid tradeable names by 20-day traded value on the
  latest forward-panel date, plus every name any live book holds.
* Kite WebSocket (KiteTicker) in `full` mode: LTP, volume, OHLC and 5 levels of
  bid/ask with quantities and order counts. Ticks are about one-per-second
  snapshots, not every trade (Kite docs).
* Written as zstd parquet parts every FLUSH_S seconds to
  data/intraday/ticks/<YYYY-MM-DD>/part-<HHMMSS>.parquet, so a crash loses at
  most one interval.
* Stops at 15:35 IST, or after 09:45 if no tick has arrived (holiday).
* One instance at a time (pid lock); cron retries cover a late Kite login.

Kite is READ-ONLY here: `profile`, `instruments` and the market-data
WebSocket. No order endpoint is imported or called (CLAUDE.md).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from live_paper import _token_candidates  # noqa: E402

IST = timezone(timedelta(hours=5, minutes=30))
ROOT = Path("data/intraday/ticks")
FLUSH_S = 30
STOP_AT = (15, 35)
HOLIDAY_GIVE_UP = (9, 45)
BOOK_DIRS = ["audit/paper/live", "audit/paper/live_adaptive",
             "audit/paper/live_momentum", "audit/paper/live_composite",
             "audit/paper/live_delivery", "audit/paper/live_momentum10"]


def _now() -> datetime:
    return datetime.now(IST)


def universe(n: int) -> list[str]:
    p = pl.scan_parquet("data/panels_forward/full.parquet")
    last = p.select(pl.col("date").max()).collect().item()
    top = (p.filter((pl.col("date") == last) & pl.col("is_tradeable").cast(pl.Boolean))
           .select("ticker", "dollar_volume_20").collect()
           .sort("dollar_volume_20", descending=True).head(n)["ticker"].to_list())
    held = set()
    for d in BOOK_DIRS:
        try:
            held |= set(json.loads(Path(d, "state.json").read_text())["holdings"])
        except (OSError, ValueError, KeyError):
            pass
    return sorted(set(top) | held)


def _kite():
    import kiteconnect  # noqa: PLC0415

    key = os.environ.get("KITE_API_KEY", "")
    for _src, tok in _token_candidates():
        k = kiteconnect.KiteConnect(api_key=key)
        k.set_access_token(tok)
        try:
            k.profile()
            return key, tok, k
        except Exception:  # noqa: BLE001 -- try the next token source
            continue
    raise SystemExit("no valid Kite token -- log in on the dashboard's Kite page")


def _row(t: dict, recv: str, sym: dict[int, str]) -> dict:
    d = t.get("depth") or {}
    o = t.get("ohlc") or {}
    ts = t.get("exchange_timestamp")
    ltt = t.get("last_trade_time")
    r = {
        "recv_ts": recv,
        "exch_ts": ts.isoformat() if ts else None,
        "last_trade_ts": ltt.isoformat() if ltt else None,
        "token": t.get("instrument_token"),
        "symbol": sym.get(t.get("instrument_token")),
        "ltp": t.get("last_price"),
        "ltq": t.get("last_traded_quantity"),
        "atp": t.get("average_traded_price"),
        "volume": t.get("volume_traded"),
        "buy_qty": t.get("total_buy_quantity"),
        "sell_qty": t.get("total_sell_quantity"),
        "open": o.get("open"), "high": o.get("high"), "low": o.get("low"),
        "close": o.get("close"),
    }
    for side, key in (("bid", "buy"), ("ask", "sell")):
        lv = d.get(key) or []
        for i in range(5):
            x = lv[i] if i < len(lv) else {}
            r[f"{side}{i + 1}_px"] = x.get("price")
            r[f"{side}{i + 1}_qty"] = x.get("quantity")
            r[f"{side}{i + 1}_n"] = x.get("orders")
    return r


def _lock() -> Path | None:
    lock = ROOT / ".lock"
    if lock.exists():
        try:
            pid = int(lock.read_text().strip())
            os.kill(pid, 0)
            return None                                  # a live recorder already runs
        except (ValueError, OSError):
            pass                                         # stale lock
    lock.write_text(str(os.getpid()))
    return lock


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--max-seconds", type=float, default=None, help="probe: stop early")
    a = ap.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    now = _now()
    if now.weekday() >= 5 and a.max_seconds is None:
        print("weekend -- not recording")
        return
    from trader.data import market_calendar as mc  # noqa: PLC0415

    if a.max_seconds is None and mc.covered(now.date()) and not mc.is_trading_day(now.date()):
        print(f"NSE holiday ({mc.holiday_name(now.date())}) -- not recording")
        return
    if (now.hour, now.minute) >= STOP_AT:
        print("after the close -- not recording")
        return
    lock = _lock()
    if lock is None:
        print("recorder already running")
        return
    try:
        key, tok, k = _kite()
        tickers = universe(a.n)
        inst = {r["tradingsymbol"]: r["instrument_token"]
                for r in k.instruments("NSE") if r.get("segment") == "NSE"}
        sym: dict[int, str] = {}
        missing = []
        for t in tickers:
            s = t.removesuffix(".NS")
            tokn = inst.get(s) or inst.get(f"{s}-BE")
            if tokn:
                sym[int(tokn)] = t
            else:
                missing.append(t)
        print(f"{now.isoformat(timespec='seconds')} recording {len(sym)} names "
              f"(unresolved {len(missing)}: {missing[:10]})", flush=True)
        run(key, tok, sym, a.max_seconds)
    finally:
        lock.unlink(missing_ok=True)


def run(key: str, tok: str, sym: dict[int, str], max_s: float | None) -> None:
    from kiteconnect import KiteTicker  # noqa: PLC0415

    buf: list[dict] = []
    lock = threading.Lock()
    stats = {"ticks": 0, "parts": 0}
    tokens = list(sym)

    def on_ticks(ws, ticks):  # noqa: ARG001
        recv = datetime.now(UTC).isoformat(timespec="milliseconds")
        rows = [_row(t, recv, sym) for t in ticks]
        with lock:
            buf.extend(rows)
            stats["ticks"] += len(rows)

    def on_connect(ws, resp):  # noqa: ARG001
        ws.subscribe(tokens)
        ws.set_mode(ws.MODE_FULL, tokens)
        print(f"{_now().isoformat(timespec='seconds')} connected, subscribed {len(tokens)}",
              flush=True)

    def on_close(ws, code, reason):  # noqa: ARG001
        print(f"{_now().isoformat(timespec='seconds')} closed {code} {reason}", flush=True)

    def on_error(ws, code, reason):  # noqa: ARG001
        print(f"{_now().isoformat(timespec='seconds')} error {code} {reason}", flush=True)

    kws = KiteTicker(key, tok)
    kws.on_ticks, kws.on_connect, kws.on_close, kws.on_error = (
        on_ticks, on_connect, on_close, on_error)
    kws.connect(threaded=True)

    def flush() -> None:
        with lock:
            rows = buf[:]
            buf.clear()
        if not rows:
            return
        n = _now()
        day = ROOT / n.date().isoformat()
        day.mkdir(parents=True, exist_ok=True)
        part = day / f"part-{n.strftime('%H%M%S')}.parquet"
        pl.DataFrame(rows, infer_schema_length=None).write_parquet(part, compression="zstd")
        stats["parts"] += 1

    t0 = time.monotonic()
    try:
        while True:
            time.sleep(FLUSH_S)
            flush()
            n = _now()
            if (n.hour, n.minute) >= STOP_AT:
                break
            if max_s is not None and time.monotonic() - t0 >= max_s:
                break
            if stats["ticks"] == 0 and (n.hour, n.minute) >= HOLIDAY_GIVE_UP:
                print("no ticks by 09:45 -- holiday? stopping", flush=True)
                break
    finally:
        try:
            kws.close()
        except Exception:  # noqa: BLE001
            pass
        flush()
        print(f"{_now().isoformat(timespec='seconds')} stopped: {stats['ticks']:,} ticks, "
              f"{stats['parts']} parts", flush=True)


if __name__ == "__main__":
    main()
