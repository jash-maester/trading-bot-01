"""Live paper execution: the algorithm's book, filled at LIVE prices. No real orders.

    uv run python scripts/live_paper.py trade                        # 09:16 IST, every weekday
    uv run python scripts/live_paper.py mark [--close]               # mark at live LTP
    uv run python scripts/live_paper.py trade --prices panel --dry   # rehearsal on closing prices

`trade` decides what the day calls for, exactly as the replay's loop does
(`scripts/run_allocator.py::_run_allocator`):
  * no book yet             -> first funding, turnover budget lifted (initial_full_deploy)
  * first session of a month -> monthly rebalance, turnover budget 0.30, band 0.010,
                               quarantined names zeroed, leftovers topped up
  * stops pending           -> sell the stopped names in full (volstop)
  * otherwise               -> hold; nothing is traded
The volstop (configs/allocator/default.yaml `volstop`): a held name whose close
is below its ENTRY fill by more than 1.0 x daily vol x sqrt(20), clipped to
[5%, 30%], is sold at the next 09:16 and barred from re-entry for 21 sessions.
`mark --close` (15:35) is the close that check runs on.

Known differences from the replay, all small and reported rather than hidden:
fills are the 09:16 quote, not the official open; the stop's volatility is
the previous session's (the replay uses the day it stepped); the stop's
"close" is the 15:35 last traded price, not NSE's official closing price.

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
import math
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
TURNOVER_BUDGET = 0.30                    # every rebalance after the first funding
STOP_MULT, STOP_H, STOP_MIN, STOP_MAX, COOLDOWN = 1.0, 20, 0.05, 0.30, 21
MAX_SIGNAL_AGE_DAYS = 4                   # a Friday signal still serves a Monday open


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


def _token_candidates() -> list[tuple[str, str]]:
    """Access tokens to try, newest source first: the dashboard's token file
    (secrets/kite/access_token.json, written by the Kite login page), then .env."""
    out = []
    p = Path(os.environ.get("TRADER_KITE_TOKEN_FILE", "secrets/kite/access_token.json"))
    try:
        t = json.loads(p.read_text()).get("access_token", "")
        if t:
            out.append(("dashboard login", t))
    except (OSError, ValueError):
        pass
    t = os.environ.get("KITE_ACCESS_TOKEN", "")
    if t and all(t != x for _, x in out):
        out.append((".env", t))
    return out


def _kite():
    import kiteconnect  # noqa: PLC0415

    key = os.environ.get("KITE_API_KEY", "")
    toks = _token_candidates()
    if not key or not toks:
        raise SystemExit("KITE_API_KEY or an access token is missing -- log in to Kite "
                         "on the dashboard's Kite login page")
    errors = []
    for src, tok in toks:
        k = kiteconnect.KiteConnect(api_key=key)
        k.set_access_token(tok)
        try:
            k.profile()
            return k
        except Exception as e:  # noqa: BLE001
            errors.append(f"{src}: {type(e).__name__}")
    raise SystemExit(f"Kite token rejected ({'; '.join(errors)}) -- the daily login is "
                     "needed (dashboard → Kite login)")


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
                          "quote_ts": str(v.get("timestamp") or v.get("last_trade_time") or ""),
                          "last_trade": str(v.get("last_trade_time") or "")}
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


def _inputs(signal_day: date) -> dict:
    """The allocator's inputs dated `signal_day`, over the panel's full universe."""
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
    return {"uni": uni, "ix": ix, "r": r, "vol": vol, "mask": mask, "sids": sids}


def _target(inp: dict, cur: np.ndarray, budget: float) -> np.ndarray:
    """Equity target weights [N], exactly as the replay's `allocate` call computes them."""
    from trader.allocator import AllocatorParams, allocate  # noqa: PLC0415

    w = allocate(inp["r"], inp["vol"], inp["mask"], inp["sids"], cur,
                 AllocatorParams(k=K, no_trade_band=BAND, max_name_weight=MAX_W,
                                 turnover_budget=budget))
    return w[1:]


def stop_threshold(vol_ann: float) -> float:
    """RiskOverlay.stop_thresholds for one name (volstop arm)."""
    if not (math.isfinite(vol_ann) and vol_ann > 0):
        return STOP_MAX                        # unknown vol gets the loosest threshold
    raw = STOP_MULT * vol_ann / math.sqrt(252.0) * math.sqrt(float(STOP_H))
    return min(max(raw, STOP_MIN), STOP_MAX)


def _signal_day(trade_day: date) -> date:
    dates = sorted(pl.read_parquet(PANEL, columns=["date"])["date"].unique().to_list())
    sd = dates[-1]
    if sd >= trade_day:
        raise SystemExit(f"panel already has {sd} >= trade day {trade_day}: wrong clock?")
    if (trade_day - sd).days > MAX_SIGNAL_AGE_DAYS:
        raise SystemExit(f"latest signal is {sd}, {(trade_day - sd).days} days before "
                         f"{trade_day}: the nightly run has not refreshed it -- not trading "
                         "on a stale signal")
    return sd


def _session_open(px: dict[str, dict], today: date) -> bool:
    """True if the quotes show trades today. False on a holiday or before the open."""
    days = {p.get("last_trade", "")[:10] for p in px.values()}
    return today.isoformat() in days


def deploy(a: argparse.Namespace) -> None:
    from trader.env.costs import ZerodhaEquityDeliveryCostModel  # noqa: PLC0415
    from trader.env.panel_env import PanelTradingEnv  # noqa: PLC0415

    LIVE.mkdir(parents=True, exist_ok=True)
    state_p = LIVE / "state.json"
    if state_p.exists() and not a.dry:
        raise SystemExit(f"{state_p} exists: the first deployment has already run.")
    if a.dry:
        dates = sorted(pl.read_parquet(PANEL, columns=["date"])["date"].unique().to_list())
        signal_day = dates[-1]
        trade_day = signal_day + timedelta(days=1)
    else:
        trade_day = _now().date()
        signal_day = _signal_day(trade_day)
    inp = _inputs(signal_day)
    uni = inp["uni"]
    cur = np.zeros(len(uni) + 1)
    cur[0] = 1.0                                          # all cash: a new investor
    # First funding: turnover budget lifted, as `initial_full_deploy` does in the replay.
    w = _target(inp, cur, 2.0)
    book = [uni[i] for i in np.flatnonzero(w > 0)]
    px = (_panel_prices(book, signal_day) if a.prices == "panel" else _live_prices(book))
    if a.prices == "live" and not _session_open(px, trade_day):
        raise SystemExit(f"no trades on NSE yet today ({trade_day}): holiday or pre-open -- "
                         "not deploying")
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
                                        "entry_price": f["price"], "opened": str(trade_day),
                                        "cost": round(f["value"] + f["charges"], 2),
                                        "sector": f["sector"]} for f in fills},
             "deployed": summary, "last_rebalance": str(trade_day),
             "realised_rs": 0.0, "charges_rs": summary["charges"],
             "pending_stops": [], "cooldown": {}}
    state_p.write_text(json.dumps(state, indent=2))
    print(f"wrote {state_p} and {LIVE / 'ledger.jsonl'}")


def plan_shares(held: np.ndarray, w: np.ndarray, px: np.ndarray, nav: float) -> np.ndarray:
    """Target whole shares for equity weights `w` [N], as `PanelTradingEnv._step_target`.

    floor for size; a request within half a share of the position is no order;
    leftover cash topped up within the 10% cap, 0.5% reserve.
    """
    from trader.env.panel_env import PanelTradingEnv  # noqa: PLC0415

    requested = np.where(px > 0, nav * w / np.maximum(px, 1e-9), 0.0)
    target = np.floor(requested)
    target = np.where(np.abs(requested - held) < 0.5, held, target)
    target = np.where(px > 0, target, held)            # never trade an unpriced name
    env = PanelTradingEnv.__new__(PanelTradingEnv)
    env._max_weight = MAX_W
    return env._topup(target, w, px, nav)


def _apply_fills(st: dict, orders: list[tuple[str, int, float]], px: dict[str, dict],
                 trade_day: date, signal_day: date, kind: str,
                 w_of: dict[str, float] | None = None) -> list[dict]:
    """Execute (ticker, signed qty, price) orders on `st` in place: sells, then buys.

    Orders under Rs 500 are dropped (min_trade_value). Buys that would overdraw
    cash are trimmed a share at a time, largest first.
    """
    from trader.env.costs import ZerodhaEquityDeliveryCostModel  # noqa: PLC0415

    cm = ZerodhaEquityDeliveryCostModel()
    sec = _sectors()
    ts = _now().isoformat(timespec="seconds")
    hold = st["holdings"]
    fills: list[dict] = []

    def fill(t: str, q: int, p: float, side: str, charges: float) -> dict:
        return {"ts": ts, "trade_date": str(trade_day), "signal_date": str(signal_day),
                "kind": kind, "ticker": t, "side": side, "qty": int(q),
                "price": round(p, 2), "value": round(q * p, 2), "charges": round(charges, 2),
                "target_weight": round(float((w_of or {}).get(t, 0.0)), 5),
                "sector": hold.get(t, {}).get("sector") or sec.get(t, "Unknown"),
                "quote_ts": px.get(t, {}).get("quote_ts", ""),
                "official_open_at_quote": px.get(t, {}).get("open", 0.0)}

    orders = [(t, q, p) for t, q, p in orders if q != 0 and p > 0 and abs(q) * p >= MIN_TRADE]
    for t, q, p in [o for o in orders if o[1] < 0]:
        h = hold[t]
        q = min(-q, h["qty"])
        value = q * p
        charges = float(cm.cost(value, is_buy=False, n_scrips_sold=1))
        cost_part = h["cost"] * q / h["qty"]
        st["cash"] += value - charges
        st["realised_rs"] = st.get("realised_rs", 0.0) + value - charges - cost_part
        st["charges_rs"] = st.get("charges_rs", 0.0) + charges
        h["qty"] -= q
        h["cost"] -= cost_part
        f = fill(t, q, p, "SELL", charges)
        f["realised_rs"] = round(value - charges - cost_part, 2)
        fills.append(f)
        if h["qty"] <= 0:
            del hold[t]
    buys = sorted([[t, q, p] for t, q, p in orders if q > 0], key=lambda o: -o[1] * o[2])
    def need(b):  # noqa: E306
        return sum(q * p + float(cm.cost(q * p, is_buy=True)) for _, q, p in b if q > 0)
    while buys and need(buys) > st["cash"]:
        buys[0][1] -= 1                                    # trim the largest buy
        buys = sorted([b for b in buys if b[1] > 0 and b[1] * b[2] >= MIN_TRADE],
                      key=lambda o: -o[1] * o[2])
    for t, q, p in buys:
        value = q * p
        charges = float(cm.cost(value, is_buy=True))
        st["cash"] -= value + charges
        st["charges_rs"] = st.get("charges_rs", 0.0) + charges
        f = fill(t, q, p, "BUY", charges)
        fills.append(f)
        h = hold.get(t)
        if h is None:
            hold[t] = {"qty": q, "avg_price": round(p, 2), "entry_price": round(p, 2),
                       "opened": str(trade_day), "cost": round(value + charges, 2),
                       "sector": f["sector"]}
            h = hold[t]
        else:
            h["avg_price"] = round((h["avg_price"] * h["qty"] + value) / (h["qty"] + q), 4)
            h["qty"] += q
            h["cost"] = round(h["cost"] + value + charges, 2)
        h["buy_today"] = {"date": str(trade_day), "qty": q, "px": round(p, 2)}
    st["cash"] = round(st["cash"], 2)
    return fills


def trade(a: argparse.Namespace) -> None:
    """09:16 every weekday: first funding, monthly rebalance, pending stops, or hold."""
    state_p = LIVE / "state.json"
    if not state_p.exists():
        return deploy(a)
    st = json.loads(state_p.read_text())
    hold = st["holdings"]
    if a.dry:
        dates = sorted(pl.read_parquet(PANEL, columns=["date"])["date"].unique().to_list())
        signal_day = dates[-1]
        trade_day = date.fromisoformat(a.as_of) if a.as_of else signal_day + timedelta(days=1)
    else:
        trade_day = _now().date()
        signal_day = _signal_day(trade_day)
    if st.get("last_trade_day") == str(trade_day) or st["deployed"]["trade_date"] == str(trade_day):
        print(f"already traded {trade_day} -- nothing to do")
        return None
    rebalance = st.get("last_rebalance", "")[:7] != trade_day.isoformat()[:7]
    pending = list(st.get("pending_stops", []))
    if not rebalance and not pending:
        print(f"{trade_day}: hold day, no stops pending -- nothing traded")
        return None
    inp = _inputs(signal_day) if rebalance else None
    names = sorted(set(hold) | set(pending)
                   | ({inp["uni"][i] for i in np.flatnonzero(inp["mask"])} if inp else set()))
    px = (_panel_prices(names, signal_day) if a.prices == "panel" else _live_prices(names))
    if a.prices == "live" and not _session_open(px, trade_day):
        print(f"no trades on NSE yet today ({trade_day}): holiday or pre-open -- nothing traded")
        return None
    cool = {t: int(n) for t, n in st.get("cooldown", {}).items() if int(n) > 0}
    if rebalance:
        uni, ix = inp["uni"], inp["ix"]
        n = len(uni)
        held = np.zeros(n)
        pc = np.zeros(n)
        live = np.zeros(n)
        for t, h in hold.items():
            if t in ix:
                held[ix[t]] = h["qty"]
        for t, v in px.items():
            if t in ix:
                live[ix[t]] = v["price"]
                pc[ix[t]] = v["prev_close"] or v["price"]
        # NAV and weights marked at the previous close, as the env marks them.
        nav = st["cash"] + float(np.sum(held * pc))
        cur = np.empty(n + 1)
        cur[1:] = held * pc / nav
        cur[0] = st["cash"] / nav
        w = _target(inp, cur, TURNOVER_BUDGET)
        # RiskOverlay.apply: quarantined names zeroed, no renormalisation.
        for t in cool:
            if t in ix:
                w[ix[t]] = 0.0
        w = np.where(inp["mask"], w, 0.0)
        tgt = plan_shares(held, w, live, nav)
        orders = [(uni[i], int(tgt[i] - held[i]), float(live[i]))
                  for i in np.flatnonzero(tgt != held)]
        # Replay order: apply() THEN register_stops(): a stop pending on a
        # rebalance day starts its cooldown but is not force-sold here.
        w_of = {uni[i]: float(w[i]) for i in np.flatnonzero(w > 0)}
        kind = "rebalance"
    else:
        orders = [(t, -hold[t]["qty"], float(px[t]["price"]))
                  for t in pending if t in hold and t in px]
        w_of, kind, nav = {}, "volstop", None
    for t in pending:
        cool[t] = COOLDOWN
    fills = _apply_fills(st, orders, px, trade_day, signal_day, kind, w_of)
    st["pending_stops"] = []
    st["cooldown"] = cool
    st["last_trade_day"] = str(trade_day)
    if rebalance:
        st["last_rebalance"] = str(trade_day)
    st["as_of"] = _now().isoformat(timespec="seconds")
    summary = {"trade_date": str(trade_day), "signal_date": str(signal_day), "kind": kind,
               "prices": a.prices, "buys": sum(f["side"] == "BUY" for f in fills),
               "sells": sum(f["side"] == "SELL" for f in fills),
               "bought_rs": round(sum(f["value"] for f in fills if f["side"] == "BUY"), 2),
               "sold_rs": round(sum(f["value"] for f in fills if f["side"] == "SELL"), 2),
               "charges": round(sum(f["charges"] for f in fills), 2),
               "cash": st["cash"], "names_after": len(hold),
               "nav_prev_close": round(nav, 2) if nav else None,
               "stops_executed": [t for t in pending if kind == "volstop"],
               "stops_registered": pending}
    print(json.dumps(summary, indent=2))
    for f in fills:
        print(f"  {f['side']:<4} {f['qty']:>5} {f['ticker']:<16} @ Rs {f['price']:>9,.2f} "
              f"= Rs {f['value']:>8,.0f}")
    if a.dry:
        print("DRY: nothing written")
        return None
    with (LIVE / "ledger.jsonl").open("a") as fh:
        for f in fills:
            fh.write(json.dumps(f) + "\n")
    with (LIVE / "trades.jsonl").open("a") as fh:
        fh.write(json.dumps(summary) + "\n")
    state_p.write_text(json.dumps(st, indent=2))
    print(f"wrote {state_p}")
    return None


def mark(a: argparse.Namespace) -> None:
    """Mark the live book: per-stock and portfolio, day move and total P&L."""
    state_p = LIVE / "state.json"
    if not state_p.exists():
        raise SystemExit("no live book yet (state.json missing) -- nothing to mark")
    st = json.loads(state_p.read_text())
    hold = st["holdings"]
    if a.prices == "live":
        px = _live_prices(list(hold))
        if hold and not _session_open(px, _now().date()):
            print(f"no NSE trades today ({_now().date()}): holiday or pre-open -- not marking")
            return
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
        # Day move: vs yesterday's close for shares held overnight, vs the fill
        # for shares bought today.
        today = _now().date().isoformat()
        bt = h.get("buy_today") or {}
        qb = int(bt.get("qty", 0)) if bt.get("date") == today else 0
        if h.get("opened") == today or pc <= 0:
            qb = q
        fpx = float(bt.get("px") or h["avg_price"])
        d_rs = (q - qb) * (ltp - pc) + qb * (ltp - fpx)
        base = (q * ltp - d_rs) / q if q else ltp
        rows.append({"ticker": t, "symbol": _sym(t), "sector": h.get("sector", "Unknown"),
                     "qty": q, "avg_price": h["avg_price"], "cost": round(c, 2),
                     "ltp": round(ltp, 2), "prev_close": round(pc, 2),
                     "value": round(v, 2), "day_chg_rs": round(d_rs, 2),
                     "day_chg_pct": round(ltp / base - 1, 5) if base else 0.0,
                     "pnl_rs": round(v - c, 2), "pnl_pct": round(v / c - 1, 5) if c else 0.0,
                     "entry_price": h.get("entry_price", h["avg_price"]),
                     "stop_level": h.get("stop_level"), "opened": h.get("opened"),
                     "priced": t in px})
        value, day_rs, cost = value + v, day_rs + d_rs, cost + c
    nav = st["cash"] + value
    stops = []
    if a.close:
        stops = _close_stop_check(st, px)
    snap = {"ts": ts, "prices": a.prices, "cash": round(st["cash"], 2),
            "holdings_value": round(value, 2), "nav": round(nav, 2),
            "capital": CAPITAL, "pnl_rs": round(nav - CAPITAL, 2),
            "pnl_pct": round(nav / CAPITAL - 1, 5), "day_chg_rs": round(day_rs, 2),
            "day_chg_pct": round(day_rs / (nav - day_rs), 5) if nav - day_rs else 0.0,
            "unpriced": [r["ticker"] for r in rows if not r["priced"]],
            "realised_rs": round(st.get("realised_rs", 0.0), 2),
            "charges_rs": round(st.get("charges_rs", 0.0), 2),
            "close_mark": bool(a.close), "stops_pending": stops,
            "cooldown": st.get("cooldown", {}), "holdings": rows}
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
    if a.close:
        state_p.write_text(json.dumps(st, indent=2))
    print(json.dumps({k: v for k, v in snap.items() if k != "holdings"}))


def _close_stop_check(st: dict, px: dict[str, dict]) -> list[str]:
    """RiskOverlay.update at the close: tick cooldowns once a day, flag breached stops.

    Mutates `st` (pending_stops, cooldown, rows' thresholds); the caller saves it.
    """
    today = _now().date().isoformat()
    if st.get("last_close_mark") != today:
        st["cooldown"] = {t: n - 1 for t, n in st.get("cooldown", {}).items() if n - 1 > 0}
        st["last_close_mark"] = today
    last = sorted(pl.read_parquet(PANEL, columns=["date"])["date"].unique().to_list())[-1]
    vol = dict(pl.read_parquet(PANEL, columns=["date", "ticker", "realized_vol_20d"])
               .filter(pl.col("date") == last).select("ticker", "realized_vol_20d").iter_rows())
    stops = []
    for t, h in st["holdings"].items():
        if t not in px:
            continue
        entry = float(h.get("entry_price") or h["avg_price"])
        thr = stop_threshold(float(vol.get(t) or float("nan")))
        h["stop_threshold"] = round(thr, 4)
        h["stop_level"] = round(entry * (1 - thr), 2)
        if px[t]["price"] / entry - 1.0 <= -thr:
            stops.append(t)
    st["pending_stops"] = stops
    return stops


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("trade", "deploy"):
        d = sub.add_parser(name)
        d.add_argument("--at-open", action="store_true", help="documentation only: 09:16 IST")
        d.add_argument("--prices", choices=("live", "panel"), default="live")
        d.add_argument("--dry", action="store_true")
        d.add_argument("--as-of", default="", help="dry runs only: pretend today is this date")
    m = sub.add_parser("mark")
    m.add_argument("--prices", choices=("live", "panel"), default="live")
    m.add_argument("--close", action="store_true",
                   help="15:35 close mark: tick cooldowns and run the volstop check")
    a = ap.parse_args()
    {"trade": trade, "deploy": trade, "mark": mark}[a.cmd](a)


if __name__ == "__main__":
    main()
