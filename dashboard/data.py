"""Read-only loaders for the paper-trading dashboard. No Streamlit here: every
function is pure (paths in, plain data out) so tests/unit/test_dashboard_data.py
can exercise it on synthetic fixtures.

Sources, all written by the paper loop or by hand, never by this module:
  audit/paper/record.jsonl        append-only record of truth for forward P&L
  audit/paper/latest/             the latest replay (deleted and rebuilt during a
                                  run -- we fall back to the newest snapshot)
  audit/paper/health.json         scripts/paper_healthcheck.py, run on the host
  logs/dashboard/health.json      the same script, run from the dashboard button
                                  (merged per check by merge_health)
  logs/paper/*.status, mem_*.csv  per-run stage lines and memory samples

Forward P&L convention (audit/P2, scripts/paper_record.py): each record line's
``ret`` is a daily LOG return, frozen by the first run that recorded that date.
A book's forward cumulative log return is the sum of its recorded ``ret`` over
sessions strictly after the freeze date; rupee P&L on the notional is
``NOTIONAL * (exp(cum) - 1)``. The ``nav`` field is never used across runs.
Rank is paper_record.py's definition: 1 + number of random books whose
cumulative return is strictly greater than the signal book's.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import polars as pl

# Restart 3 (2026-09-29): Rs 1,00,000 of cash deployed at the 2026-09-30 open.
NOTIONAL = 100_000.0
FREEZE_DATE = date(2026, 9, 9)  # scoring filter; cross-checked vs summary.json freeze_date
EVIDENCE_MONTHS = 48  # audit/P2: no rank reading before this
# audit/P2_PAPER_PERMUTATION_TEST.md "Restart 2": the record restarted on the
# Docker platform; line 1 of record.jsonl is 2026-09-25 and the 11 host sessions
# 09-10..09-25 were discarded (archived as record.jsonl.restart2).
RESTART_NOTE = (
    "Restart 3 (user decision 2026-09-29): every book starts with Rs 1,00,000 in CASH and "
    "deploys it at the 2026-09-30 open. A session's close is marked by the NEXT run, so "
    "each session appears one run late, under its correct date. Earlier records are "
    "archived as audit/paper/record.jsonl.restart1..3."
)
SESSIONS_PER_MONTH = 21.0
STALE_HOURS = 26.0  # both crontab slots fire daily; a gap > 26 h = a missed day
SIGNAL_BOOK = "allocator_k30_b0.01_rvolstop_monthly_20d"
EW_BOOK = "equal_weight_monthly"
OVERLAY_BOOKS = [f"allocator_k30_b0.01_r{r}_monthly_20d" for r in ("none", "stop10", "stop15")]
NULL_RE = re.compile(r"^null_signal_k30_b0\.01_rvolstop_s(\d+)_monthly_20d$")


# ── generic, failure-tolerant readers ──────────────────────────────────────
@dataclass
class Loaded:
    """A value plus a human-readable reason when it could not be loaded."""

    value: object = None
    error: str | None = None
    source: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.error is None


def read_json(path: Path) -> Loaded:
    if not path.exists():
        return Loaded(error=f"{path} does not exist", source=str(path))
    try:
        txt = path.read_text()
        if not txt.strip():
            return Loaded(error=f"{path} is empty", source=str(path))
        return Loaded(value=json.loads(txt), source=str(path))
    except (OSError, json.JSONDecodeError) as e:
        return Loaded(error=f"{path}: {type(e).__name__}: {e}", source=str(path))


# ── the record ──────────────────────────────────────────────────────────────
def load_record(path: Path) -> Loaded:
    """Parse record.jsonl. Bad lines are skipped and counted, not fatal."""
    if not path.exists():
        return Loaded(value=[], error=f"{path} does not exist", source=str(path))
    lines, bad = [], 0
    try:
        raw = path.read_text().splitlines()
    except OSError as e:
        return Loaded(value=[], error=f"{path}: {e}", source=str(path))
    for ln in raw:
        if not ln.strip():
            continue
        try:
            e = json.loads(ln)
            e["_date"] = date.fromisoformat(e["date"])
            if not isinstance(e.get("books"), dict):
                raise ValueError("no books")
            lines.append(e)
        except (ValueError, KeyError, TypeError):
            bad += 1
    err = None if lines else f"{path} has no valid sessions yet"
    return Loaded(value=lines, error=err, source=str(path), extra={"bad_lines": bad})


def null_books(books: set[str] | list[str]) -> list[str]:
    """The pre-registered null distribution: rvolstop random books, by seed."""
    hits = [(int(m.group(1)), b) for b in books if (m := NULL_RE.match(b))]
    return [b for _, b in sorted(hits)]


def scored(record: list[dict], freeze: date = FREEZE_DATE) -> list[dict]:
    return [e for e in record if e["_date"] > freeze]


def cum_log(record: list[dict], book: str, freeze: date = FREEZE_DATE) -> float:
    return float(sum(e["books"].get(book, {}).get("ret", 0.0) for e in scored(record, freeze)))


def missing_pairs(record: list[dict], freeze: date = FREEZE_DATE) -> list[tuple[str, str]]:
    """(date, book) pairs absent from a scored session among the books that feed
    the headline: signal, equal-weight and the rvolstop random books. cum_log
    counts a missing pair as 0 (as paper_record.py does), so the page must say so."""
    sc = scored(record, freeze)
    seen: set[str] = set()
    for e in sc:
        seen |= set(e["books"])
    want = [SIGNAL_BOOK, EW_BOOK, *null_books(seen)]
    return [(e["date"], b) for e in sc for b in want if b not in e["books"]]


def freeze_mismatch(summary: dict | None, freeze: date = FREEZE_DATE) -> str | None:
    """None when summary.json's freeze_date agrees with the hard-coded one."""
    if not summary or "freeze_date" not in summary:
        return "summary.json has no freeze_date; the scoring filter could not be cross-checked"
    try:
        f = date.fromisoformat(str(summary["freeze_date"]))
    except ValueError:
        return f"summary.json freeze_date {summary['freeze_date']!r} is not a date"
    if f != freeze:
        return (
            f"summary.json freeze_date {f} differs from the dashboard's {freeze}: "
            "forward P&L and rank below are scored against the WRONG date"
        )
    return None


def to_rupees(cum: float, notional: float = NOTIONAL) -> float:
    return notional * (math.exp(cum) - 1.0)


def rank_among(signal_cum: float, null_cums: list[float]) -> int:
    return 1 + sum(1 for c in null_cums if c > signal_cum)


def quantile(xs: list[float], q: float) -> float:
    """Linear-interpolated quantile (numpy's default), without numpy."""
    s = sorted(xs)
    if not s:
        return float("nan")
    pos = (len(s) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def headline(record: list[dict], freeze: date = FREEZE_DATE, notional: float = NOTIONAL) -> dict:
    """Forward numbers for the overview. Every value derives from record ``ret``."""
    sc = scored(record, freeze)
    all_books: set[str] = set()
    for e in sc:
        all_books |= set(e["books"])
    nulls = null_books(all_books)
    s = cum_log(record, SIGNAL_BOOK, freeze)
    ew = cum_log(record, EW_BOOK, freeze)
    nc = [cum_log(record, b, freeze) for b in nulls]

    def pack(c: float) -> dict:
        return {"cum_log": c, "pct": math.exp(c) - 1.0, "rupees": to_rupees(c, notional)}

    out = {
        "sessions": len(sc),
        "first_date": sc[0]["date"] if sc else None,
        "last_date": sc[-1]["date"] if sc else None,
        "last_run_ts": max((e.get("run_ts", "") for e in record), default=None) or None,
        "n_null": len(nulls),
        "signal": pack(s),
        "equal_weight": pack(ew),
        "overlays": {b: pack(cum_log(record, b, freeze)) for b in OVERLAY_BOOKS if b in all_books},
        "rank": rank_among(s, nc) if sc and nc else None,
        # Length of the evidence = the record actually on file, NOT time since the
        # freeze: P2 restart 2 discarded 09-10..09-25. Calendar months from the
        # first recorded session to the last, plus sessions / 21.
        "months_elapsed": months_between(sc[0]["_date"], sc[-1]["_date"]) if sc else 0.0,
        "trading_months": len(sc) / SESSIONS_PER_MONTH,
    }
    if nc:
        out["null_median"] = pack(quantile(nc, 0.5))
        out["null_p10"] = pack(quantile(nc, 0.1))
        out["null_p90"] = pack(quantile(nc, 0.9))
        out["null_mean"] = pack(sum(nc) / len(nc))
    return out


def months_between(a: date, b: date) -> float:
    return (b - a).days / 30.4375


def cumulative_series(
    record: list[dict], freeze: date = FREEZE_DATE, notional: float = NOTIONAL
) -> pl.DataFrame:
    """Long frame: date, book, kind (signal/overlay/equal_weight/null), cum_log, rupees."""
    sc = scored(record, freeze)
    if not sc:
        return pl.DataFrame(
            schema={
                "date": pl.Date,
                "book": pl.Utf8,
                "kind": pl.Utf8,
                "ret": pl.Float64,
                "cum_log": pl.Float64,
                "rupees": pl.Float64,
            }
        )
    books: set[str] = set()
    for e in sc:
        books |= set(e["books"])
    nulls = set(null_books(books))
    rows = []
    for b in sorted(books):
        kind = (
            "signal"
            if b == SIGNAL_BOOK
            else "overlay"
            if b in OVERLAY_BOOKS
            else "equal_weight"
            if b == EW_BOOK
            else "null"
            if b in nulls
            else None
        )
        if kind is None:
            continue  # non-volstop randoms etc.: not shown
        c = 0.0
        for e in sc:
            r = e["books"].get(b, {}).get("ret", 0.0)
            c += r
            rows.append(
                {
                    "date": e["_date"],
                    "book": b,
                    "kind": kind,
                    "ret": r,
                    "cum_log": c,
                    "rupees": to_rupees(c, notional),
                }
            )
    return pl.DataFrame(rows)


def null_band(series: pl.DataFrame, notional: float = NOTIONAL) -> pl.DataFrame:
    """Per date: 10th / 50th / 90th percentile of the random books' rupee P&L."""
    n = series.filter(pl.col("kind") == "null") if series.height else series
    if not n.height:
        return pl.DataFrame(
            schema={"date": pl.Date, "p10": pl.Float64, "p50": pl.Float64, "p90": pl.Float64}
        )
    rows = []
    for d, grp in n.group_by("date", maintain_order=True):
        # quantile of the log return, then to rupees: identical to headline()
        v = grp["cum_log"].to_list()
        rows.append(
            {
                "date": d[0],
                "p10": to_rupees(quantile(v, 0.1), notional),
                "p50": to_rupees(quantile(v, 0.5), notional),
                "p90": to_rupees(quantile(v, 0.9), notional),
            }
        )
    return pl.DataFrame(rows).sort("date")


def session_table(record: list[dict], freeze: date = FREEZE_DATE) -> pl.DataFrame:
    """One row per recorded session: daily log return of each headline book."""
    rows = []
    for e in scored(record, freeze):
        nr = [v["ret"] for b, v in e["books"].items() if NULL_RE.match(b)]
        rows.append(
            {
                "date": e["_date"],
                "run_ts": e.get("run_ts"),
                "signal_ret": e["books"].get(SIGNAL_BOOK, {}).get("ret"),
                "equal_weight_ret": e["books"].get(EW_BOOK, {}).get("ret"),
                "null_median_ret": quantile(nr, 0.5) if nr else None,
                "n_books": len(e["books"]),
            }
        )
    return pl.DataFrame(rows) if rows else pl.DataFrame()


# ── the replay (warm-up context) ────────────────────────────────────────────
def latest_dir(paper: Path) -> Loaded:
    """audit/paper/latest if complete; else the newest snapshot. The loop deletes
    latest/ at the start of the allocator stage, so mid-run it is partial."""
    lat = paper / "latest"
    if (lat / "summary.json").exists():
        return Loaded(value=lat, source=str(lat))
    snaps = (
        sorted(p for p in (paper / "snapshots").glob("20*") if (p / "summary.json").exists())
        if (paper / "snapshots").exists()
        else []
    )
    if snaps:
        return Loaded(
            value=snaps[-1],
            source=str(snaps[-1]),
            extra={
                "note": f"latest/ incomplete (run in progress?); using snapshot {snaps[-1].name}"
            },
        )
    return Loaded(error="no complete replay: latest/ has no summary.json and no snapshot exists")


def book_from_nav_file(p: Path) -> str:
    return re.sub(r"_(full|paper)\.parquet$", "", p.name.removeprefix("nav_"))


def warmup_navs(nav_dir: Path, freeze: date = FREEZE_DATE) -> pl.DataFrame:
    """Long frame date, book, kind, nav, growth (nav / first nav) up to the freeze."""
    frames = []
    for f in sorted(nav_dir.glob("nav_*.parquet")):
        b = book_from_nav_file(f)
        kind = (
            "signal"
            if b == SIGNAL_BOOK
            else "overlay"
            if b in OVERLAY_BOOKS
            else "equal_weight"
            if b == EW_BOOK
            else "null"
            if NULL_RE.match(b)
            else None
        )
        if kind is None:
            continue
        try:
            d = pl.read_parquet(f, columns=["date", "nav"]).drop_nulls().sort("date")
        except Exception:  # noqa: BLE001 -- a torn file mid-run must not crash the page
            continue
        d = d.filter(pl.col("date") <= freeze)
        if not d.height:
            continue
        frames.append(
            d.with_columns(
                pl.lit(b).alias("book"),
                pl.lit(kind).alias("kind"),
                (pl.col("nav") / d["nav"][0]).alias("growth"),
            )
        )
    return pl.concat(frames) if frames else pl.DataFrame()


# ── health, status, memory ──────────────────────────────────────────────────
HEALTH_CHECKS = [
    "nse_access",
    "data_fresh",
    "feeds_aligned",
    "record_current",
    "last_run",
    "scheduler_fired",
    "determinism",
    "scheduler",
]
_ORDER = {"OK": 0, "WARN": 1, "FAIL": 2}


def worst(statuses) -> str:
    st = [s for s in statuses if s in _ORDER]
    return max(st, key=_ORDER.__getitem__) if st else "FAIL"


def merge_health(*paths: Path, now: datetime | None = None) -> Loaded:
    """Merge health files per check instead of picking one file.

    Each check comes from whichever file ran it most recently, tagged with that
    file and its checked_at, so the host-only ``scheduler`` check is never dropped
    by a newer in-container report that could not run it. ``overall`` is the worst
    status across the merged set; ``stale`` is set when the newest report is more
    than STALE_HOURS old (nothing re-runs the check on a schedule), and ``missing``
    lists expected checks no file reported."""
    now = now or datetime.now(UTC)
    got = [(read_json(p), p) for p in paths]
    good = [
        (h.value, p)
        for h, p in got
        if h.ok and isinstance(h.value, dict) and "checked_at" in h.value
    ]
    if not good:
        errs = "; ".join(h.error or f"{p}: no checked_at" for h, p in got)
        return Loaded(error=f"no health report yet ({errs})")
    merged: dict[str, dict] = {}
    for v, p in sorted(good, key=lambda vp: vp[0]["checked_at"]):  # oldest first
        for c in v.get("checks", []):
            name = c.get("check")
            if name:
                merged[name] = {**c, "_source": str(p), "_checked_at": v["checked_at"]}
    checks = [merged[n] for n in HEALTH_CHECKS if n in merged] + [
        c for n, c in merged.items() if n not in HEALTH_CHECKS
    ]
    newest = max(v["checked_at"] for v, _ in good)
    age_h = health_age_seconds(newest, now) / 3600
    stale = age_h > STALE_HOURS
    missing = [n for n in HEALTH_CHECKS if n not in merged]
    overall = worst(c.get("status") for c in checks)
    return Loaded(
        value={
            "overall": overall,
            "checks": checks,
            "checked_at": newest,
            "age_hours": age_h,
            "stale": stale,
            "missing": missing,
            "sources": sorted({c["_source"] for c in checks}),
        },
        source=", ".join(str(p) for _, p in good),
    )


def health_age_seconds(checked_at: str, now: datetime) -> float:
    return (now - datetime.fromisoformat(checked_at.replace("Z", "+00:00"))).total_seconds()


def live_checks(log_dir: Path, record: list[dict], now: datetime | None = None) -> list[dict]:
    """Probe-free liveness, recomputed on every page load from files only (no NSE).

    run_started: newest '=== run <ts> ===' header in logs/paper/*.status must be
                 <= STALE_HOURS old (same rule as paper_healthcheck scheduler_fired).
    last_run:    the newest status block ended DONE / no-new-session with no FAIL.
    record_age:  the last recorded session is at most 4 calendar days old (a long
                 weekend is 3; an NSE holiday next to a weekend can make it 4). WARN
                 only: without asking NSE a holiday cannot be told from a miss."""
    now = now or datetime.now(UTC)
    out: list[dict] = []
    blocks = status_blocks(log_dir, n=1_000_000)
    if not blocks:
        out.append(
            {"check": "run_started", "status": "FAIL", "detail": f"no status files in {log_dir}"}
        )
    else:
        ts = blocks[-1]["started"]
        try:
            age_h = health_age_seconds(ts, now) / 3600
            out.append(
                {
                    "check": "run_started",
                    "status": "OK" if age_h <= STALE_HOURS else "FAIL",
                    "detail": f"last run started {ts} ({age_h:.1f} h ago; "
                    f"limit {STALE_HOURS:.0f} h)",
                    "age_hours": age_h,
                }
            )
        except ValueError:
            out.append(
                {"check": "run_started", "status": "WARN", "detail": f"unparseable header {ts!r}"}
            )
        b = blocks[-1]
        st = {"DONE": "OK", "NO-NEW": "OK", "DRY": "WARN", "FAIL": "FAIL"}.get(b["outcome"], "WARN")
        if b["outcome"] == "INCOMPLETE":
            detail = f"{b['started']}: in progress, or died without a FAIL line"
        else:
            detail = f"{b['started']}: {b['outcome']}"
        out.append({"check": "last_run", "status": st, "detail": detail})
    if record:
        last = max(e["_date"] for e in record)
        today = (now + timedelta(hours=5, minutes=30)).date()  # IST
        gap = (today - last).days
        out.append(
            {
                "check": "record_age",
                "status": "OK" if gap <= 4 else "WARN",
                "detail": f"last recorded session {last} ({gap} day(s) before today IST)",
            }
        )
    else:
        out.append({"check": "record_age", "status": "WARN", "detail": "record is empty"})
    return out


# ── on-demand probe rate limit ──────────────────────────────────────────────
def probe_gate(stamp: Path, lock: Path, now: datetime, min_interval_s: float, lock_ttl_s: float):
    """(allowed, seconds_since_last_attempt | None, reason). The stamp records the
    ATTEMPT (written before probing), so a check that dies mid-probe still counts."""
    if lock.exists():
        try:
            held = now.timestamp() - lock.stat().st_mtime
        except OSError:
            held = 0.0
        if held < lock_ttl_s:
            return False, None, "a health check is already running"
    age = None
    if stamp.exists():
        try:
            age = health_age_seconds(stamp.read_text().strip(), now)
        except (OSError, ValueError):
            age = None
    if age is not None and age < min_interval_s:
        return False, age, "rate limit"
    return True, age, ""


def acquire_probe(stamp: Path, lock: Path, now: datetime, lock_ttl_s: float) -> bool:
    """Take the O_EXCL lock (clearing one older than lock_ttl_s) and write the stamp.
    False if another session holds a live lock."""
    import os

    lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        if lock.exists() and now.timestamp() - lock.stat().st_mtime >= lock_ttl_s:
            lock.unlink(missing_ok=True)
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w") as f:
        f.write(now.isoformat(timespec="seconds"))
    stamp.write_text(now.isoformat(timespec="seconds"))
    return True


def release_probe(lock: Path) -> None:
    lock.unlink(missing_ok=True)


def status_blocks(log_dir: Path, n: int = 5) -> list[dict]:
    """Last n '=== run <ts> ===' blocks across logs/paper/*.status, newest last."""
    blocks: list[dict] = []
    for f in sorted(log_dir.glob("20*.status")):
        try:
            txt = f.read_text()
        except OSError:
            continue
        for part in re.split(r"(?m)^=== run ", txt):
            part = part.strip()
            if not part:
                continue
            head, *body = part.splitlines()
            ts = head.rstrip("= ").strip()
            outcome = (
                "FAIL"
                if any(" FAIL " in f" {x} " for x in body)
                else "DONE"
                if any(" DONE " in f" {x} " for x in body)
                else "NO-NEW"
                if any("no new session" in x or "nothing to record yet" in x for x in body)
                else "DRY"
                if any("DRY:" in x for x in body)
                else "INCOMPLETE"
            )
            blocks.append({"file": f.name, "started": ts, "outcome": outcome, "lines": body})
    return blocks[-n:]


def memory_samples(log_dir: Path) -> Loaded:
    files = sorted(log_dir.glob("mem_20*.csv"))
    if not files:
        return Loaded(error=f"no mem_*.csv in {log_dir}")
    f = files[-1]
    try:
        d = pl.read_csv(f, try_parse_dates=True)
    except Exception as e:  # noqa: BLE001
        return Loaded(error=f"{f}: {e}", source=str(f))
    if not d.height:
        return Loaded(error=f"{f} is empty", source=str(f))
    return Loaded(value=d, source=str(f))


# ── research ledger ─────────────────────────────────────────────────────────
DOCS = [
    "audit/P2_PAPER_PERMUTATION_TEST.md",
    "audit/R_2026-09-29.md",
    "audit/R_OVERNIGHT_2026-09-25.md",
    "audit/S6_TOPK_GATE.md",
    "audit/S5_TURNOVER_SELECTION.md",
    "audit/S4_NULL_CONTROL.md",
    "audit/R5_VERDICT_PIT.md",
    "PROGRESS.md",
]

# id, file, question, how the verdict is found in the doc. Question text is a
# paraphrase; result and verdict are EXTRACTED from the file at load time, so
# the table cannot drift from the ledger. Criteria: extracted from the E-rows'
# pre-registration blocks; for R5/S4/S5/S6, whose docs state the bar in prose
# rather than a labelled block, see CRITERIA below (regex where the doc has one
# sentence to extract, otherwise a hand-written paraphrase with file:line).
EXPERIMENTS = [
    (
        "R5",
        "audit/R5_VERDICT_PIT.md",
        "Does the allocator beat the best baseline on a point-in-time universe?",
        r"^## Verdict:\s*(.+)$",
    ),
    (
        "S4",
        "audit/S4_NULL_CONTROL.md",
        "Does the signal beat a random signal running identical machinery?",
        r"^# (.+)$",
    ),
    (
        "S5",
        "audit/S5_TURNOVER_SELECTION.md",
        "Can top-30 by turnover replace the random picker?",
        r"\*\*(It is much worse than random[^*]*)\*\*",
    ),
    (
        "S6",
        "audit/S6_TOPK_GATE.md",
        "Is R4's information in the top 30 a long-only book buys?",
        r"^\| `r4_pit_long`, 13 win \|.*\| \*?\*?(?:\d+)\*?\*? \| (\w+) \|$",
    ),
    (
        "E0",
        "audit/R_OVERNIGHT_2026-09-25.md",
        "MPS parity retrain matches the CUDA artefact?",
        None,
    ),
    (
        "E0b",
        "audit/R_OVERNIGHT_2026-09-25.md",
        "Is platform the cause of E0's rank-corr gap?",
        None,
    ),
    (
        "E1",
        "audit/R_OVERNIGHT_2026-09-25.md",
        "Bottom-30 screen (monthly) beats universe and random screens?",
        None,
    ),
    (
        "E2",
        "audit/R_OVERNIGHT_2026-09-25.md",
        "ListNet loss moves information to the top 30?",
        None,
    ),
    (
        "E3",
        "audit/R_2026-09-29.md",
        "Volatility-neutral training target frees top-30 information?",
        None,
    ),
    ("E4", "audit/R_2026-09-29.md", "Bottom-30 screen re-formed quarterly clears α/2?", None),
]


# ("rx", pattern) extracts from the experiment's own file; ("text", s) is a
# paraphrase that carries its own file:line citation.
CRITERIA = {
    "R5": ("rx", r"^(The criterion is .+)$"),
    "S4": (
        "text",
        "Beat a random signal running identical machinery (same universe, K, band, stop, "
        "costs, tax) in matched pairs, in sample and on the 2025-26 holdout "
        "[PROGRESS.md:37; audit/S4_NULL_CONTROL.md:3-6]",
    ),
    "S5": (
        "text",
        "Top-30 by 20d median turnover, pre-committed (monthly, K=30, band 0.010, four "
        "overlays), must beat the random picker under the matched null control "
        "[audit/S5_TURNOVER_SELECTION.md:8-10; PROGRESS.md:38]",
    ),
    "S6": (
        "text",
        "Top-30 mean 20d forward return, net of a 23 bps cost proxy, in excess of the "
        "eligible-universe mean, per walk-forward window: window t above t_crit (2.18 at "
        "13 windows), and compared with 20 random books on identical support "
        "[audit/S6_TOPK_GATE.md:3-7, 26-28]",
    ),
}

# Rows that are measurements with pre-registered readings, not pass/fail gates,
# and cross-row notes that stop one row being read as reversing another.
MEASUREMENTS = {"E0b"}
NOTES = {
    "E0": "NOT AT PARITY stands; E0b does not change it [audit/R_OVERNIGHT_2026-09-25.md:128, 191]",
    "E0b": "Measurement, not a gate: seed floor vs cross-platform gap. Does not reverse E0.",
}


def _section(text: str, eid: str) -> str:
    m = re.search(rf"(?ms)^## {re.escape(eid)} — .*?(?=^## |\Z)", text)
    if m:
        return m.group(0)
    # E0b and similar: pre-registered as a ### block under ## Results
    m = re.search(rf"(?ms)^### {re.escape(eid)} — pre-registered.*?(?=^##+ |\Z)", text)
    return m.group(0) if m else ""


def _criterion(section: str) -> str:
    for para in re.split(r"\n\s*\n", section):
        if re.search(r"\*\*Pass|Criteria|\*\*Reading, fixed", para):
            return re.sub(r"\s+", " ", para).strip()[:600]
    return ""


def _classify(v: str) -> str:
    u = v.upper()
    if "PENDING" in u or not v:
        return "PENDING"
    if (
        "NOT PASSED" in u
        or "FAIL" in u
        or "NOT AT PARITY" in u
        or "DOES NOT" in u
        or "WORSE" in u
        or "NO SIGNAL" in u
    ):
        return "FAIL"
    if "PASS" in u:
        return "PASS"
    return "SEE DOC"


def experiments(root: Path) -> list[dict]:
    rows = []
    for eid, rel, question, rx in EXPERIMENTS:
        p = root / rel
        if not p.exists():
            rows.append(
                {
                    "id": eid,
                    "question": question,
                    "criterion": "",
                    "result": "",
                    "verdict": "MISSING",
                    "note": NOTES.get(eid, ""),
                    "file": rel,
                }
            )
            continue
        text = p.read_text()
        crit, result = "", ""
        if rx is not None:
            m = re.search(rx, text, re.M)
            result = m.group(1).strip() if m else ""
            kind, spec = CRITERIA.get(eid, ("text", ""))
            if kind == "rx":
                cm = re.search(spec, text, re.M)
                crit = f"{cm.group(1).strip()} [{rel}]" if cm else ""
            else:
                crit = spec
        else:
            crit = _criterion(_section(text, eid))
            heads = [
                h
                for h in re.findall(rf"(?m)^### {re.escape(eid)} — (.+)$", text)
                if "pre-registered" not in h and "follow-up" not in h
            ]
            result = heads[0].strip() if heads else "PENDING — no result heading in the doc"
        rows.append(
            {
                "id": eid,
                "question": question,
                "criterion": crit,
                "result": result,
                "verdict": "MEASUREMENT" if eid in MEASUREMENTS and result else _classify(result),
                "note": NOTES.get(eid, ""),
                "file": rel,
            }
        )
    return rows


def gate_jsons(root: Path) -> pl.DataFrame:
    """audit/topk_gate/*.json as a table (numbers straight from the files)."""
    rows = []
    for f in sorted((root / "audit/topk_gate").glob("*.json")):
        j = read_json(f)
        if not j.ok or not isinstance(j.value, dict):
            continue
        v = j.value
        rows.append(
            {
                "file": f"audit/topk_gate/{f.name}",
                "gate": v.get("gate"),
                "signal": v.get("signal"),
                "k": v.get("k"),
                "horizon": v.get("horizon"),
                "n_windows": v.get("n_windows"),
                "mean": v.get("mean"),
                "t": v.get("t"),
                "t_crit": v.get("t_crit"),
                "diff_vs_null": v.get("diff_vs_null_mean"),
                "t_diff": v.get("t_diff"),
                "rank": v.get("rank_vs_null"),
                "n_null": v.get("n_null"),
                "verdict": str(v.get("verdict")),
            }
        )
    return pl.DataFrame(rows) if rows else pl.DataFrame()
