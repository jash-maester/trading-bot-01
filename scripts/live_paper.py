"""Live paper execution: the algorithm's book, filled at LIVE prices. No real orders.

    uv run python scripts/live_paper.py deploy --at-open            # 09:16 IST, first funding
    uv run python scripts/live_paper.py mark                         # mark the book at live LTP
    uv run python scripts/live_paper.py deploy --prices panel --dry  # rehearsal on closing prices

User instruction (2026-09-29): a new investor with Rs 1,00,000 in cash, nothing
invested, following the algorithm; orders right after the 09:15 open.

This SHADOWS the end-of-day paper record (audit/P2, restart 3), it does not
replace it. The record fills at NSE's official open, known only after the
close; this fills at the Kite quote seen at 09:16. The book is decided by the
SAME code: `trader.allocator.allocate` with the replay's inputs (signal and
volatility dated the previous session, first rebalance allowed to deploy in
full) and sized by `PanelTradingEnv._topup` (whole shares, leftover cash spent
within the 10% cap). The difference between the two fills is the execution gap
a real account would face, and the dashboard reports it.

Kite is READ-ONLY here: `profile` (token check) and `quote`. No order-placing
endpoint is imported or called (CLAUDE.md). The access token expires daily at
~06:00 IST and needs a browser login; without it this refuses to run.

State (append-only where it matters), under audit/paper/live/:
  state.json      cash and holdings after the last execution
  ledger.jsonl    one line per simulated fill
  marks.csv       timestamped marks: cash, holdings value, NAV
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import polars as pl

IST = timezone(timedelta(hours=5, minutes=30))
# Overridable so a rehearsal never touches the real live book.
LIVE = Path(os.environ.get("TRADER_LIVE_DIR", "audit/paper/live"))
PANEL = Path("data/panels_forward/full.parquet")
SIGNAL = Path("data/signal/paper_signal/predictions.parquet")
CAPITAL = 100_000.0
MIN_TRADE = 500.0
K, BAND, MAX_W = 30, 0.010, 0.10


def _sector_names() -> dict[int, str]:
    from trader.data import nse_industry as ni  # noqa: PLC0415

    out = {i: n for n, i in ni.NSE_INDUSTRY_IDS.items()}
    out[ni.UNKNOWN_INDUSTRY_ID] = "Unclassified (ETF / other)"
    return out


def _sectors() -> dict[str, str]:
    names = _sector_names()
    p = pl.read_parquet(PANEL, columns=["ticker", "sector_id"]).unique("ticker", keep="last")
    return {t: names.get(int(s or 0), "Unknown") for t, s in p.iter_rows()}


def _now() -> datetime:
    return datetime.now(IST)


def _sym(ticker: str) -> str:
    return ticker.removesuffix(".NS")


def _kite():
    import kiteconnect  # noqa: PLC0415

    key, tok = os.environ.get("KITE_API_KEY", ""), os.environ.get("KITE_ACCESS_TOKEN", "")
    if not key or not tok:
        raise SystemExit("KITE_API_KEY / KITE_ACCESS_TOKEN missing -- log in to Kite first")
    k = kiteconnect.KiteConnect(api_key=key)
    k.set_access_token(tok)
    try:
        k.profile()
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"Kite token rejected ({type(e).__name__}) -- the daily login is needed")
    return k


def _live_prices(tickers: list[str]) -> dict[str, dict]:
    k = _kite()
    out: dict[str, dict] = {}
    keys = [f"NSE:{_sym(t)}" for t in tickers]
    for i in range(0, len(keys), 200):
        q = k.quote(keys[i : i + 200])
        for t in tickers[i : i + 200]:
            v = q.get(f"NSE:{_sym(t)}")
            if v and v.get("last_price"):
                ohlc = v.get("ohlc") or {}
                out[t] = {"price": float(v["last_price"]),
                          "open": float(ohlc.get("open") or 0.0),
                          "prev_close": float(ohlc.get("close") or 0.0),
                          "quote_ts": str(v.get("timestamp") or v.get("last_trade_time") or "")}
    return out


def _panel_prices(tickers: list[str], day: date) -> dict[str, dict]:
    """Rehearsal prices: the session's close as 'live', the prior close as prev_close."""
    p = pl.read_parquet(PANEL, columns=["date", "ticker", "close"])
    ds = sorted(p["date"].unique().to_list())
    prev = ds[ds.index(day) - 1]
    now = dict(p.filter(pl.col("date") == day).select("ticker", "close").iter_rows())
    was = dict(p.filter(pl.col("date") == prev).select("ticker", "close").iter_rows())
    return {t: {"price": float(now[t]), "open": float(now[t]),
                "prev_close": float(was.get(t) or now[t]),
                "quote_ts": f"{day} close (rehearsal)"} for t in tickers if now.get(t)}


def _target(signal_day: date) -> tuple[list[str], np.ndarray]:
    """Equity target weights for the next session, exactly as the replay computes them."""
    from trader.allocator import AllocatorParams, allocate  # noqa: PLC0415

    panel = pl.read_parquet(PANEL, columns=["date", "ticker", "is_tradeable", "sector_id",
                                            "realized_vol_20d"])
    uni = sorted(panel["ticker"].unique().to_list())
    day = panel.filter(pl.col("date") == signal_day)
    if day.is_empty():
        raise SystemExit(f"no panel rows for {signal_day}")
    ix = {t: i for i, t in enumerate(uni)}
    n = len(uni)
    mask, sids, vol, r = (np.zeros(n, bool), np.zeros(n, np.int64),
                          np.full(n, np.nan), np.full(n, np.nan))
    for t, tr, sec, v in day.select("ticker", "is_tradeable", "sector_id",
                                    "realized_vol_20d").iter_rows():
        i = ix[t]
        mask[i], sids[i], vol[i] = bool(tr), int(sec or 0), float(v) if v is not None else np.nan
    preds = pl.read_parquet(SIGNAL).filter(pl.col("date") == signal_day)
    if preds.is_empty():
        raise SystemExit(f"no predictions dated {signal_day} in {SIGNAL}")
    for t, v in preds.select("ticker", "r_hat_20d").iter_rows():
        if t in ix and v is not None:
            r[ix[t]] = float(v)
    cur = np.zeros(n + 1)
    cur[0] = 1.0                                          # all cash: a new investor
    # First funding: turnover budget lifted, as `initial_full_deploy` does in the replay.
    w = allocate(r, vol, mask, sids, cur,
                 AllocatorParams(k=K, no_trade_band=BAND, max_name_weight=MAX_W,
                                 turnover_budget=2.0))
    return uni, w[1:]


def deploy(a: argparse.Namespace) -> None:
    from trader.env.costs import ZerodhaEquityDeliveryCostModel  # noqa: PLC0415
    from trader.env.panel_env import PanelTradingEnv  # noqa: PLC0415

    LIVE.mkdir(parents=True, exist_ok=True)
    state_p = LIVE / "state.json"
    if state_p.exists() and not a.dry:
        raise SystemExit(f"{state_p} exists: the first deployment has already run. "
                         "Rebalances are a separate step.")
    dates = sorted(pl.read_parquet(PANEL, columns=["date"])["date"].unique().to_list())
    signal_day = dates[-1]                                 # last published session
    trade_day = _now().date() if not a.dry else signal_day + timedelta(days=1)
    uni, w = _target(signal_day)
    book = [uni[i] for i in np.flatnonzero(w > 0)]
    px = (_panel_prices(book, signal_day) if a.prices == "panel" else _live_prices(book))
    missing = [t for t in book if t not in px]
    prices = np.array([px[t]["price"] if t in px else 0.0 for t in book])
    wb = np.array([w[uni.index(t)] for t in book])
    floored = np.floor(np.where(prices > 0, CAPITAL * wb / np.maximum(prices, 1e-9), 0.0))
    env = PanelTradingEnv.__new__(PanelTradingEnv)
    env._max_weight = MAX_W
    shares = env._topup(floored, wb, prices, CAPITAL)
    cm = ZerodhaEquityDeliveryCostModel()
    sec = _sectors()
    fills, cash = [], CAPITAL
    for t, q, p, wt in zip(book, shares, prices, wb, strict=True):
        value = float(q * p)
        if q <= 0 or value < MIN_TRADE:
            continue
        charges = float(cm.cost(value, is_buy=True))
        cash -= value + charges
        fills.append({"ts": _now().isoformat(timespec="seconds"), "trade_date": str(trade_day),
                      "signal_date": str(signal_day), "ticker": t, "side": "BUY",
                      "qty": int(q), "price": round(float(p), 2), "value": round(value, 2),
                      "charges": round(charges, 2), "target_weight": round(float(wt), 5),
                      "sector": sec.get(t, "Unknown"),
                      "quote_ts": px[t]["quote_ts"], "official_open_at_quote": px[t]["open"]})
    invested = sum(f["value"] for f in fills)
    summary = {"trade_date": str(trade_day), "signal_date": str(signal_day),
               "prices": a.prices, "capital": CAPITAL, "names": len(fills),
               "invested": round(invested, 2),
               "charges": round(sum(f["charges"] for f in fills), 2),
               "cash": round(cash, 2), "invested_pct": round(invested / CAPITAL, 4),
               "unpriced": missing}
    print(json.dumps(summary, indent=2))
    for f in sorted(fills, key=lambda f: -f["value"]):
        print(f"  BUY {f['qty']:>5} {f['ticker']:<16} @ Rs {f['price']:>9,.2f} "
              f"= Rs {f['value']:>8,.0f}"
              f"  (target {f['target_weight']:.1%})")
    if a.dry:
        print("DRY: nothing written")
        return
    with (LIVE / "ledger.jsonl").open("a") as fh:
        for f in fills:
            fh.write(json.dumps(f) + "\n")
    state = {"as_of": _now().isoformat(timespec="seconds"), "cash": round(cash, 2),
             "holdings": {f["ticker"]: {"qty": f["qty"], "avg_price": f["price"],
                                        "cost": round(f["value"] + f["charges"], 2),
                                        "sector": f["sector"]} for f in fills},
             "deployed": summary}
    state_p.write_text(json.dumps(state, indent=2))
    print(f"wrote {state_p} and {LIVE / 'ledger.jsonl'}")


def mark(a: argparse.Namespace) -> None:
    """Mark the live book: per-stock and portfolio, day move and total P&L."""
    state_p = LIVE / "state.json"
    if not state_p.exists():
        raise SystemExit("no live book yet (state.json missing) -- nothing to mark")
    st = json.loads(state_p.read_text())
    hold = st["holdings"]
    if a.prices == "live":
        px = _live_prices(list(hold))
    else:
        last = sorted(pl.read_parquet(PANEL, columns=["date"])["date"].unique().to_list())[-1]
        px = _panel_prices(list(hold), last)
    ts = _now().isoformat(timespec="seconds")
    rows, value, day_rs, cost = [], 0.0, 0.0, 0.0
    for t, h in sorted(hold.items()):
        q = h["qty"]
        p = px.get(t, {})
        ltp = float(p.get("price") or h["avg_price"])
        pc = float(p.get("prev_close") or 0.0)
        c = float(h.get("cost", q * h["avg_price"]))
        v = q * ltp
        # Day move: vs yesterday's close, except on the day a name was bought,
        # when the position only existed since the fill.
        bought_today = st.get("deployed", {}).get("trade_date") == _now().date().isoformat()
        base = pc if pc > 0 and not bought_today else h["avg_price"]
        d_rs = q * (ltp - base)
        rows.append({"ticker": t, "symbol": _sym(t), "sector": h.get("sector", "Unknown"),
                     "qty": q, "avg_price": h["avg_price"], "cost": round(c, 2),
                     "ltp": round(ltp, 2), "prev_close": round(pc, 2),
                     "value": round(v, 2), "day_chg_rs": round(d_rs, 2),
                     "day_chg_pct": round(ltp / base - 1, 5) if base else 0.0,
                     "pnl_rs": round(v - c, 2), "pnl_pct": round(v / c - 1, 5) if c else 0.0,
                     "priced": t in px})
        value, day_rs, cost = value + v, day_rs + d_rs, cost + c
    nav = st["cash"] + value
    snap = {"ts": ts, "prices": a.prices, "cash": round(st["cash"], 2),
            "holdings_value": round(value, 2), "nav": round(nav, 2),
            "capital": CAPITAL, "pnl_rs": round(nav - CAPITAL, 2),
            "pnl_pct": round(nav / CAPITAL - 1, 5), "day_chg_rs": round(day_rs, 2),
            "day_chg_pct": round(day_rs / (nav - day_rs), 5) if nav - day_rs else 0.0,
            "unpriced": [r["ticker"] for r in rows if not r["priced"]], "holdings": rows}
    (LIVE / "holdings_latest.json").write_text(json.dumps(snap, indent=2))
    with (LIVE / "marks_holdings.jsonl").open("a") as fh:
        fh.write(json.dumps(snap) + "\n")
    out = LIVE / "marks.csv"
    new = not out.exists()
    with out.open("a") as fh:
        if new:
            fh.write("ts,cash,holdings_value,nav,pnl_rs,pnl_pct,day_chg_rs\n")
        fh.write(f"{ts},{snap['cash']},{snap['holdings_value']},{snap['nav']},"
                 f"{snap['pnl_rs']},{snap['pnl_pct']},{snap['day_chg_rs']}\n")
    print(json.dumps({k: v for k, v in snap.items() if k != "holdings"}))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("deploy")
    d.add_argument("--at-open", action="store_true", help="documentation only: run at 09:16 IST")
    d.add_argument("--prices", choices=("live", "panel"), default="live")
    d.add_argument("--dry", action="store_true")
    m = sub.add_parser("mark")
    m.add_argument("--prices", choices=("live", "panel"), default="live")
    a = ap.parse_args()
    {"deploy": deploy, "mark": mark}[a.cmd](a)


if __name__ == "__main__":
    main()
