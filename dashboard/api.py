"""JSON payloads for the web dashboard (dashboard/server.py). No HTTP here.

Every function reads files the paper loop writes and returns plain data. Results
are memoised on the mtimes of the files they read, so a poll that finds nothing
new costs a handful of ``stat`` calls. The portfolio and status payloads use the
standard library only; polars is imported (lazily, via data.py) only by the
experiment, health, research and warm-up payloads.
"""

from __future__ import annotations

import csv
import json
import math
import os
import subprocess
import sys
import threading
import time
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import kite_auth as KA

import data as D

IST = KA.IST
ROOT = Path(os.environ.get("DASHBOARD_ROOT", Path(__file__).resolve().parent.parent))
PAPER = ROOT / "audit/paper"
LIVE = PAPER / "live"
LOGS = ROOT / "logs/paper"
DASH_HEALTH = ROOT / os.environ.get("DASHBOARD_HEALTH_OUT", "logs/dashboard/health.json")
PROBE_STAMP = DASH_HEALTH.parent / "probe_stamp"
PROBE_LOCK = DASH_HEALTH.parent / "probe.lock"
PROBE_MIN_INTERVAL_S = 600
PROBE_TIMEOUT_S = 330
PROBE_LOCK_TTL_S = PROBE_TIMEOUT_S + 60
KITE_RECHECK_S = 300  # a live profile call at most every 5 min
STALE_MARK_MIN = 20
MAX_POINTS = 400  # charts are downsampled to this many points

_lock = threading.Lock()
_memo: dict[str, tuple[tuple, object]] = {}


def _mtime(p: Path) -> float:
    try:
        return p.stat().st_mtime
    except OSError:
        return 0.0


def _dir_mtimes(d: Path, pattern: str) -> tuple:
    try:
        return tuple(sorted((f.name, f.stat().st_mtime) for f in d.glob(pattern)))
    except OSError:
        return ()


def _cached(name: str, key: tuple, build):
    with _lock:
        hit = _memo.get(name)
        if hit and hit[0] == key:
            return hit[1]
    val = build()
    with _lock:
        _memo[name] = (key, val)
    return val


def _jsonl(p: Path) -> list[dict]:
    out = []
    try:
        for ln in p.read_text().splitlines():
            if ln.strip():
                try:
                    out.append(json.loads(ln))
                except ValueError:
                    pass
    except OSError:
        pass
    return out


def _json(p: Path):
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def _rel(p: Path) -> str:
    """Path relative to the repo for display; the absolute path if it is outside it."""
    try:
        return str(Path(p).relative_to(ROOT))
    except ValueError:
        return str(p)


def _thin(xs: list, n: int = MAX_POINTS) -> list:
    if len(xs) <= n:
        return xs
    step = len(xs) / n
    out = [xs[int(i * step)] for i in range(n)]
    out[-1] = xs[-1]
    return out


# ── market clock ─────────────────────────────────────────────────────────────


def phase(now: datetime) -> str:
    """weekend | pre_open (09:00-09:15) | open (09:15-15:30) | closed. Holidays unknown."""
    if now.weekday() >= 5:
        return "weekend"
    t = now.hour * 60 + now.minute
    if 9 * 60 <= t < 9 * 60 + 15:
        return "pre_open"
    if 9 * 60 + 15 <= t < 15 * 60 + 30:
        return "open"
    return "closed"


def next_weekday(d: date) -> date:
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d


def next_rebalance(last: str | None, today: date) -> date:
    """First weekday of the month after the last rebalance (NSE holidays not known)."""
    base = date.fromisoformat(last) if last else today
    first = date(base.year + (base.month == 12), base.month % 12 + 1, 1)
    return next_weekday(first)


def next_live_event(now: datetime) -> dict:
    """The next job in docker/paper.crontab's live-book section."""
    for add in range(8):
        d = (now + timedelta(days=add)).date()
        if d.weekday() >= 5:
            continue
        slots = [(9, 16, "trade"), (15, 35, "close mark and stop check")]
        slots += [(h, m, "price update") for h in range(9, 16) for m in (0, 15, 30, 45)]
        for h, m, what in sorted(slots):
            when = datetime(d.year, d.month, d.day, h, m, tzinfo=IST)
            if when > now:
                return {"at": when.isoformat(), "what": what}
    return {"at": None, "what": None}


# ── portfolio (stdlib only) ──────────────────────────────────────────────────


def _marks() -> list[dict]:
    p = LIVE / "marks.csv"
    try:
        with p.open() as fh:
            rows = list(csv.DictReader(fh))
    except OSError:
        return []
    out = []
    for r in rows:
        try:
            out.append(
                {
                    "ts": r["ts"],
                    "nav": float(r["nav"]),
                    "pnl_rs": float(r["pnl_rs"]),
                    "day_chg_rs": float(r["day_chg_rs"]),
                }
            )
        except (KeyError, ValueError):
            continue
    return out


def _trade_days(ledger: list[dict], trades: list[dict]) -> list[dict]:
    days: dict[str, dict] = {}
    for f in ledger:
        d = days.setdefault(
            f.get("trade_date", "?"),
            {
                "date": f.get("trade_date"),
                "ts": f.get("ts"),
                "kind": f.get("kind") or "first purchase",
                "buys": 0,
                "sells": 0,
                "bought": 0.0,
                "sold": 0.0,
                "charges": 0.0,
                "realised": 0.0,
                "fills": [],
            },
        )
        side = f.get("side", "BUY")
        d["buys" if side == "BUY" else "sells"] += 1
        d["bought" if side == "BUY" else "sold"] += float(f.get("value") or 0)
        d["charges"] += float(f.get("charges") or 0)
        d["realised"] += float(f.get("realised_rs") or 0)
        d["ts"] = min(d["ts"] or f.get("ts"), f.get("ts") or d["ts"])
        d["fills"].append(
            {
                k: f.get(k)
                for k in (
                    "ts",
                    "side",
                    "ticker",
                    "sector",
                    "qty",
                    "price",
                    "value",
                    "charges",
                    "realised_rs",
                    "target_weight",
                    "quote_ts",
                )
            }
        )
    extra = {t.get("trade_date"): t for t in trades}
    for k, d in days.items():
        if k in extra:
            d["stops_executed"] = extra[k].get("stops_executed") or []
    return sorted(days.values(), key=lambda d: d["date"] or "", reverse=True)


def portfolio() -> dict:
    files = [
        LIVE / n
        for n in ("holdings_latest.json", "state.json", "marks.csv", "ledger.jsonl", "trades.jsonl")
    ]
    key = tuple(_mtime(p) for p in files)

    def build() -> dict:
        snap, state = _json(files[0]), _json(files[1])
        if snap is None or state is None:
            return {"deployed": False}
        marks = _marks()
        days = sorted({m["ts"][:10] for m in marks})
        dep = state.get("deployed") or {}
        for h in snap.get("holdings", []):
            h["weight"] = h["value"] / snap["nav"] if snap.get("nav") else 0.0
        return {
            "deployed": True,
            "snapshot": snap,
            "state": {
                k: state.get(k)
                for k in (
                    "last_rebalance",
                    "pending_stops",
                    "cooldown",
                    "realised_rs",
                    "charges_rs",
                    "last_trade_day",
                )
            },
            "deployed_on": dep.get("trade_date"),
            "deployed_at": dep.get("trade_date"),
            "marks": _thin(marks, 2000),
            "sessions": len(days),
            "trade_days": _trade_days(_jsonl(files[3]), _jsonl(files[4])),
            "next_rebalance": next_rebalance(
                state.get("last_rebalance"), datetime.now(IST).date()
            ).isoformat(),
            "evidence_months": D.EVIDENCE_MONTHS,
        }

    return _cached("portfolio", key, build)


# ── status: chips, attention banner, change detection ────────────────────────

_kite_cache: dict = {"at": 0.0, "mtime": -1.0, "val": {"state": "missing"}}


def kite_status(force: bool = False) -> dict:
    m = _mtime(KA.token_path(ROOT))
    now = time.time()
    c = _kite_cache
    if force or m != c["mtime"] or now - c["at"] > KITE_RECHECK_S:
        c.update(at=now, mtime=m, val=KA.check(ROOT))
    return c["val"]


def pipeline() -> dict:
    key = (
        _mtime(PAPER / "record.jsonl"),
        _mtime(PAPER / "health.json"),
        _mtime(DASH_HEALTH),
        _dir_mtimes(LOGS, "*.status"),
        int(time.time() // 600),
    )

    def build() -> dict:
        rec = D.load_record(PAPER / "record.jsonl").value or []
        h = D.merge_health(PAPER / "health.json", DASH_HEALTH)
        live = D.live_checks(LOGS, rec)
        reasons, sts = [], []
        if h.ok:
            sts.append(h.value["overall"])
            if h.value["overall"] != "OK":
                bad = [c["check"] for c in h.value["checks"] if c.get("status") != "OK"]
                reasons.append(f"health check {h.value['overall']}: {', '.join(bad)}")
            if h.value["stale"]:
                sts.append("WARN")
                reasons.append(f"health not re-checked for {h.value['age_hours']:.0f} h")
        else:
            sts.append("WARN")
            reasons.append(str(h.error))
        for c in live:
            sts.append(c["status"])
            if c["status"] != "OK":
                reasons.append(f"{c['check']} {c['status']}: {c['detail']}")
        return {
            "status": D.worst(sts),
            "reasons": reasons,
            "checked_at": h.value["checked_at"] if h.ok else None,
        }

    return _cached("pipeline", key, build)


def attention(now: datetime, kite: dict, pipe: dict, pf: dict) -> list[dict]:
    """Things that need the user, most urgent first. level: urgent | warn | info."""
    out = []
    ph = phase(now)
    trading_day = now.weekday() < 5
    t = now.hour * 60 + now.minute
    if kite.get("state") != "valid":
        before_trade = trading_day and 6 * 60 <= t < 9 * 60 + 16
        during = ph == "open"
        what = {
            "missing": "No Kite token is stored",
            "expired": "The Kite token has expired",
            "error": "The Kite token could not be verified",
        }.get(kite.get("state"), "Kite")
        if before_trade:
            msg = f"{what}. Log in before 09:16 IST or today's trade cannot read prices."
        elif during:
            msg = f"{what}. Live price updates are failing until you log in."
        else:
            msg = f"{what}. Log in before 09:16 IST on the next trading day."
        out.append(
            {
                "id": f"kite-{kite.get('state')}-{now.date()}",
                "level": "urgent" if (before_trade or during) else "warn",
                "title": "Kite login needed",
                "message": msg,
                "action": "kite",
            }
        )
    if pf.get("deployed") and ph == "open":
        last = pf["snapshot"]["ts"]
        age = (now - datetime.fromisoformat(last)).total_seconds() / 60
        if age > STALE_MARK_MIN and t >= 9 * 60 + 35:
            out.append(
                {
                    "id": f"stale-{last}",
                    "level": "warn",
                    "title": "Prices are stale",
                    "message": f"No price update for {age:.0f} minutes during market hours.",
                    "action": "health",
                }
            )
    if pf.get("deployed") and pf["snapshot"].get("prices") != "live":
        out.append(
            {
                "id": "rehearsal",
                "level": "warn",
                "title": "Rehearsal prices",
                "message": "The portfolio is marked on closing prices, not live quotes.",
                "action": None,
            }
        )
    if pipe.get("status") == "FAIL":
        out.append(
            {
                "id": f"pipe-{pipe.get('checked_at')}",
                "level": "urgent",
                "title": "Data pipeline failing",
                "message": "; ".join(pipe["reasons"])[:300],
                "action": "health",
            }
        )
    return out


def status() -> dict:
    now = datetime.now(IST)
    kite = kite_status()
    pipe = pipeline()
    pf = portfolio()
    version = "|".join(
        str(x)
        for x in (
            _mtime(LIVE / "holdings_latest.json"),
            _mtime(LIVE / "state.json"),
            _mtime(PAPER / "record.jsonl"),
            _mtime(PAPER / "health.json"),
            _mtime(DASH_HEALTH),
            _mtime(ADAPT / "record.jsonl"),
            _mtime(ADAPT / "refits.jsonl"),
            kite.get("minted_at"),
            kite.get("state"),
        )
    )
    return {
        "now": now.isoformat(timespec="seconds"),
        "phase": phase(now),
        "kite": kite,
        "pipeline": pipe,
        "last_mark": pf["snapshot"]["ts"] if pf.get("deployed") else None,
        "next_event": next_live_event(now),
        "attention": attention(now, kite, pipe, pf),
        # 30 s from 06:00 (when the Kite token expires and a login is due) through
        # the close, and whenever something urgent is showing; 5 min otherwise.
        "poll_s": 30
        if (now.weekday() < 5 and 6 * 60 <= now.hour * 60 + now.minute <= 15 * 60 + 50)
        or any(a["level"] == "urgent" for a in attention(now, kite, pipe, pf))
        else 300,
        "version": version,
    }


# ── experiment (polars) ──────────────────────────────────────────────────────


def experiment() -> dict:
    key = (
        _mtime(PAPER / "record.jsonl"),
        _mtime(ADAPT / "record.jsonl"),
        _mtime(ADAPT / "refits.jsonl"),
    )

    def build() -> dict:
        r = D.load_record(PAPER / "record.jsonl")
        if r.error:
            return {
                "error": r.error,
                "evidence_months": D.EVIDENCE_MONTHS,
                "restart_note": D.RESTART_NOTE,
                "adaptive": adaptive(),
            }
        rec = r.value or []
        hd = D.headline(rec)
        ser = D.cumulative_series(rec)
        lines: dict[str, dict] = {}
        band: list = []
        if ser.height:
            for row in ser.iter_rows(named=True):
                if row["kind"] in ("signal", "equal_weight", "overlay"):
                    b = lines.setdefault(
                        row["book"], {"book": row["book"], "kind": row["kind"], "points": []}
                    )
                    b["points"].append([str(row["date"]), row["rupees"]])
            nb = D.null_band(ser)
            band = (
                [
                    [str(a), p10, p50, p90]
                    for a, p10, p50, p90 in nb.select("date", "p10", "p50", "p90").iter_rows()
                ]
                if nb.height
                else []
            )
        tab = D.session_table(rec)
        return {
            "headline": hd,
            "lines": list(lines.values()),
            "band": band,
            "sessions": [
                {k: (str(v) if isinstance(v, date) else v) for k, v in row.items()}
                for row in tab.iter_rows(named=True)
            ]
            if tab.height
            else [],
            "missing_pairs": D.missing_pairs(rec)[:10],
            "bad_lines": (r.extra or {}).get("bad_lines", 0),
            "evidence_months": D.EVIDENCE_MONTHS,
            "restart_note": D.RESTART_NOTE,
            "freeze_date": D.FREEZE_DATE.isoformat(),
            "adaptive": adaptive(),
        }

    return _cached("experiment", key, build)


# ── P3 adaptive shadow book (stdlib) ─────────────────────────────────────────

ADAPT = PAPER / "adaptive"
ADAPT_FROM = "2026-10-01"
ADAPT_READ = "2028-10-01"
ADAPT_SIG = "allocator_k30_b0.01_rvolstop_semimonthly_20d"
ADAPT_EW = "equal_weight_semimonthly"
P2_SIG = D.SIGNAL_BOOK


def _q(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    if not xs:
        return 0.0
    k = (len(xs) - 1) * q
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def adaptive() -> dict:
    files = (ADAPT / "record.jsonl", ADAPT / "refits.jsonl", PAPER / "record.jsonl")
    key = tuple(_mtime(f) for f in files)

    def build() -> dict:
        refits = _jsonl(files[1])
        rec = sorted(
            (e for e in _jsonl(files[0]) if e.get("date", "") >= ADAPT_FROM),
            key=lambda e: e["date"],
        )
        p2 = {e["date"]: e.get("books", {}) for e in _jsonl(files[2])}
        out = {
            "refits": refits,
            "start": ADAPT_FROM,
            "read_at": ADAPT_READ,
            "sessions": len(rec),
            "lines": [],
            "band": [],
            "headline": None,
        }
        if not rec:
            return out
        cum: dict[str, float] = {}
        p2cum = 0.0
        lines = {"adaptive": [], "frozen": [], "equal_weight": []}
        band = []
        for e in rec:
            for b, v in e["books"].items():
                cum[b] = cum.get(b, 0.0) + float(v.get("ret") or 0.0)
            fr = p2.get(e["date"], {}).get(P2_SIG)
            if fr is not None:
                p2cum += float(fr.get("ret") or 0.0)
            nulls = [c for b, c in cum.items() if b.startswith("null_signal") and "rvolstop" in b]
            rup = lambda c: (math.exp(c) - 1.0) * D.NOTIONAL  # noqa: E731
            lines["adaptive"].append([e["date"], rup(cum.get(ADAPT_SIG, 0.0))])
            lines["equal_weight"].append([e["date"], rup(cum.get(ADAPT_EW, 0.0))])
            if fr is not None:
                lines["frozen"].append([e["date"], rup(p2cum)])
            if nulls:
                band.append(
                    [e["date"], rup(_q(nulls, 0.1)), rup(_q(nulls, 0.5)), rup(_q(nulls, 0.9))]
                )
        nulls = [c for b, c in cum.items() if b.startswith("null_signal") and "rvolstop" in b]
        sig = cum.get(ADAPT_SIG, 0.0)
        pack = lambda c: {"rupees": (math.exp(c) - 1.0) * D.NOTIONAL, "pct": math.exp(c) - 1.0}  # noqa: E731
        out.update(
            lines=[{"kind": k, "points": v} for k, v in lines.items() if v],
            band=band,
            headline={
                "adaptive": pack(sig),
                "frozen": pack(p2cum),
                "equal_weight": pack(cum.get(ADAPT_EW, 0.0)),
                "null_median": pack(_q(nulls, 0.5)),
                "n_null": len(nulls),
                "rank": 1 + sum(1 for c in nulls if c > sig) if nulls else None,
                "first_date": rec[0]["date"],
                "last_date": rec[-1]["date"],
            },
        )
        return out

    return _cached("adaptive", key, build)


# ── health (polars only for the memory chart) ────────────────────────────────


def health() -> dict:
    key = (
        _mtime(PAPER / "health.json"),
        _mtime(DASH_HEALTH),
        _dir_mtimes(LOGS, "*.status"),
        _dir_mtimes(LOGS, "mem_*.csv"),
        _mtime(PAPER / "record.jsonl"),
        int(time.time() // 60),
    )

    def build() -> dict:
        rec = D.load_record(PAPER / "record.jsonl").value or []
        h = D.merge_health(PAPER / "health.json", DASH_HEALTH)
        hv = None
        if h.ok:
            hv = dict(h.value)
            hv["checks"] = [
                {**c, "_source": str(c.get("_source", "")).removeprefix(str(ROOT) + "/")}
                for c in hv["checks"]
            ]
        mem = D.memory_samples(LOGS)
        memory = None
        if mem.ok:
            m = mem.value
            cols = [c for c in ("used_mb", "avail_mb") if c in m.columns]
            memory = {
                "source": mem.source,
                "columns": cols,
                "rows": _thin([[str(r[0]), *r[1:]] for r in m.select("ts", *cols).iter_rows()]),
            }
        ld = D.latest_dir(PAPER)
        det = None
        if ld.ok:
            s = D.read_json(ld.value / "summary.json")
            if s.ok and s.value.get("determinism"):
                det = {**s.value["determinism"], "source": _rel(ld.value)}
        can, age, why = D.probe_gate(
            PROBE_STAMP, PROBE_LOCK, datetime.now(UTC), PROBE_MIN_INTERVAL_S, PROBE_LOCK_TTL_S
        )
        return {
            "pipeline": pipeline(),
            "live": D.live_checks(LOGS, rec),
            "report": hv,
            "report_error": None if h.ok else h.error,
            "runs": D.status_blocks(LOGS, n=5),
            "memory": memory,
            "determinism": det,
            "probe": {"can": can, "age_s": age, "why": why, "min_interval_s": PROBE_MIN_INTERVAL_S},
            "schedule": SCHEDULE,
        }

    return _cached("health", key, build)


SCHEDULE = [
    ["07:30", "daily", "Nightly catch-up run (paper record)"],
    ["07:50", "daily", "Health check"],
    ["09:16, 09:20, 09:30", "weekdays", "Live book trade (idempotent per day)"],
    ["every 15 min 09:00-15:45", "weekdays", "Price update (mark)"],
    ["15:35", "weekdays", "Close mark and volatility-stop check"],
    ["21:00", "daily", "Nightly run: data, prediction, 85 replay books, record"],
    ["21:20", "daily", "Health check"],
]


def rerun_health() -> dict:
    now = datetime.now(UTC)
    can, _, why = D.probe_gate(PROBE_STAMP, PROBE_LOCK, now, PROBE_MIN_INTERVAL_S, PROBE_LOCK_TTL_S)
    if not can:
        return {"ok": False, "error": why}
    if not D.acquire_probe(PROBE_STAMP, PROBE_LOCK, now, PROBE_LOCK_TTL_S):
        return {"ok": False, "error": "another session is already running the health check"}
    try:
        r = subprocess.run(
            [sys.executable, "scripts/paper_healthcheck.py"],
            cwd=ROOT,
            env={**os.environ, "PAPER_HEALTH_OUT": str(DASH_HEALTH)},
            capture_output=True,
            text=True,
            timeout=PROBE_TIMEOUT_S,
        )
        if r.returncode != 0:
            return {"ok": False, "error": f"exited {r.returncode}: {r.stderr[-600:]}"}
        return {"ok": True}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"could not run: {e}"}
    finally:
        D.release_probe(PROBE_LOCK)


# ── research and warm-up ─────────────────────────────────────────────────────


def research() -> dict:
    key = (_dir_mtimes(ROOT / "audit", "*.md"), _dir_mtimes(ROOT / "audit/topk_gate", "*.json"))

    def build() -> dict:
        g = D.gate_jsons(ROOT)
        return {
            "experiments": D.experiments(ROOT),
            "gates": g.to_dicts() if g.height else [],
            "docs": [d for d in D.DOCS if (ROOT / d).exists()],
        }

    return _cached("research", key, build)


def doc(name: str) -> str | None:
    """Raw markdown of a whitelisted document; None if not allowed or missing."""
    if name not in D.DOCS:
        return None
    try:
        return (ROOT / name).read_text()
    except OSError:
        return None


def warmup() -> dict:
    ld = D.latest_dir(PAPER)
    if not ld.ok:
        return {"error": ld.error}
    path = ld.value
    key = (str(path), _dir_mtimes(path, "nav_*.parquet"), _mtime(path / "summary.json"))

    def build() -> dict:
        w = D.warmup_navs(path)
        summ = D.read_json(path / "summary.json")
        out = {
            "source": _rel(path),
            "note": ld.extra.get("note"),
            "warmup": (summ.value or {}).get("warmup") if summ.ok else None,
            "notional": D.NOTIONAL,
            "lines": [],
            "band": [],
        }
        if not w.height:
            return out
        scale = D.NOTIONAL
        for kind in ("signal", "equal_weight"):
            g = w.filter(D.pl.col("kind") == kind).sort("date")
            if g.height:
                out["lines"].append(
                    {
                        "kind": kind,
                        "book": g["book"][0],
                        "points": _thin(
                            [
                                [str(d), (x - 1) * scale]
                                for d, x in g.select("date", "growth").iter_rows()
                            ]
                        ),
                    }
                )
        nulls = w.filter(D.pl.col("kind") == "null")
        if nulls.height:
            q = (
                nulls.group_by("date")
                .agg(
                    D.pl.col("growth").quantile(0.1, interpolation="linear").alias("p10"),
                    D.pl.col("growth").quantile(0.5, interpolation="linear").alias("p50"),
                    D.pl.col("growth").quantile(0.9, interpolation="linear").alias("p90"),
                )
                .sort("date")
            )
            out["band"] = _thin(
                [
                    [str(d), (a - 1) * scale, (b - 1) * scale, (c - 1) * scale]
                    for d, a, b, c in q.iter_rows()
                ]
            )
        return out

    return _cached("warmup", key, build)
