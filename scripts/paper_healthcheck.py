"""Is the forward paper loop actually getting data, and running? One answer, one file.

    uv run python scripts/paper_healthcheck.py          # prints + writes audit/paper/health.json

Checks, each OK / WARN / FAIL with the evidence:
  nse_access      live HTTP probes of the two NSE archives the loop depends on.
                  200 = served, 404 = not published yet (normal before ~18:30 IST
                  or on a holiday), 401/403/429/5xx or a network error = blocked.
                  No credentials exist to expire: both archives are public.
  data_fresh      last bhavcopy date and last ^NSEI date vs the latest weekday
                  NSE has actually published.
  feeds_aligned   bhavcopy and index end on the same date (else beta is dead).
  record_current  every published session after the record start is recorded.
  last_run        the most recent status block ended (DONE / no new session / nothing to
                  record yet) with no FAIL.
  scheduler_fired some run started within the last 26 h (cron fires twice a day).
  determinism     the last run's replay matched the previous snapshot.
  scheduler       the `paper` container is running (skipped inside it).
Kite is not checked: the loop does not use it (NSE archives only).
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import urllib.error
import urllib.request
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import polars as pl

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
BHAV = "https://archives.nseindia.com/products/content/sec_bhavdata_full_{d}.csv"
INDEX = "https://archives.nseindia.com/content/indices/ind_close_all_{d}.csv"
# Overridable so a read-only consumer (the dashboard container) can write elsewhere.
OUT = Path(os.environ.get("PAPER_HEALTH_OUT", "audit/paper/health.json"))


def _probe(url: str) -> int | str:
    # A one-byte ranged GET, not HEAD: NSE's archive CDN answers every HEAD with
    # 503 while serving GETs normally (2026-09-29: HEAD 503 / GET 206 on the same
    # sec_bhavdata_full_28092026.csv), which made this check cry "blocked" on a
    # feed the loop had just fetched. 206 (or 200 if Range is ignored) = served.
    req = urllib.request.Request(url, method="GET", headers={
        "User-Agent": UA, "Referer": "https://www.nseindia.com/", "Accept": "*/*",
        "Range": "bytes=0-0"})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return 200 if int(r.status) in (200, 206) else int(r.status)
    except urllib.error.HTTPError as e:
        return int(e.code)
    except Exception as e:  # noqa: BLE001
        return f"{type(e).__name__}"


def _weekdays_back(n: int) -> list[date]:
    d, out = date.today(), []
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d -= timedelta(days=1)
    return out


def _check(name: str, status: str, detail: str, **ev: object) -> dict:
    return {"check": name, "status": status, "detail": detail, **ev}


def main() -> None:
    checks: list[dict] = []

    # nse_access + latest published weekday (walk back until both feeds answer 200)
    probes, latest_pub = [], None
    for d in _weekdays_back(6):
        k = d.strftime("%d%m%Y")
        b, i = _probe(BHAV.format(d=k)), _probe(INDEX.format(d=k))
        probes.append({"date": d.isoformat(), "bhavcopy": b, "index": i})
        if latest_pub is None and b == 200 and i == 200:
            latest_pub = d
    bad = [p for p in probes for c in (p["bhavcopy"], p["index"])
           if c not in (200, 404)]
    if bad:
        checks.append(_check("nse_access", "FAIL",
                             "NSE returned blocking/error codes (not 200/404)", probes=probes))
    elif latest_pub is None:
        checks.append(_check("nse_access", "WARN",
                             "no weekday in the last 6 answered 200 on both feeds", probes=probes))
    else:
        checks.append(_check("nse_access", "OK",
                             f"both archives serving; latest published {latest_pub}",
                             probes=probes))

    # data_fresh + feeds_aligned
    bars = pl.scan_parquet("data/ext/bhavcopy.parquet")
    bh = bars.select(pl.col("date").max()).collect().item()
    try:
        from trader.data.storage import OhlcvStore
        ix = OhlcvStore("data/kite_ohlcv").load(tickers=["^NSEI"],
                                               start=datetime(2026, 1, 1))["date"].max()
        ix = ix.date() if isinstance(ix, datetime) else ix
    except Exception as e:  # noqa: BLE001
        ix = None
        checks.append(_check("feeds_aligned", "FAIL", f"cannot read ^NSEI: {e}"))
    if latest_pub is not None:
        st = "OK" if bh >= latest_pub else "WARN"
        checks.append(_check("data_fresh", st,
                             f"bhavcopy ends {bh}; NSE latest published {latest_pub}",
                             bhavcopy_last=str(bh), index_last=str(ix),
                             nse_latest=str(latest_pub)))
    if ix is not None:
        checks.append(_check("feeds_aligned", "OK" if ix == bh else "FAIL",
                             f"bhavcopy {bh}, ^NSEI {ix}"))

    # record_current
    rec_path = Path("audit/paper/record.jsonl")
    rec = [json.loads(x) for x in rec_path.read_text().splitlines() if x.strip()] \
        if rec_path.exists() else []
    rec_dates = [date.fromisoformat(r["date"]) for r in rec]
    mrf = re.search(r'^RECORD_FROM="(\d{4}-\d{2}-\d{2})"', Path("scripts/paper_daily.sh")
                    .read_text(), re.M) if Path("scripts/paper_daily.sh").exists() else None
    record_from = date.fromisoformat(mrf.group(1)) if mrf else None
    # Restart 3: a session's close is marked by the NEXT run, so the latest data
    # date is never expected in the record yet.
    if not rec_dates and record_from is not None and bh <= record_from:
        # Restart 3: an empty record is the expected state until the first
        # session on/after RECORD_FROM has been published.
        checks.append(_check("record_current", "OK",
                             f"record empty, awaiting first session on/after {record_from} "
                             f"(data currently ends {bh})"))
    elif not rec_dates:
        checks.append(_check("record_current", "FAIL",
                             f"record is empty but data reaches {bh}"
                             + (f" (RECORD_FROM {record_from})" if record_from else "")))
    else:
        panel_dates = set(bars.select("date").unique().collect()["date"].to_list())
        missing = sorted(d for d in panel_dates if rec_dates[0] < d < bh and d not in rec_dates)
        st = "OK" if not missing else ("WARN" if len(missing) <= 5 else "FAIL")
        checks.append(_check("record_current", st,
                             f"{len(rec_dates)} sessions recorded, "
                             f"{rec_dates[0]}..{rec_dates[-1]}; "
                             f"{len(missing)} published-but-unrecorded",
                             missing=[str(d) for d in missing], last_recorded=str(rec_dates[-1])))

    # last_run
    st_files = sorted(Path("logs/paper").glob("20*.status"))
    if st_files:
        blocks = re.split(r"(?m)^=== run ", st_files[-1].read_text())
        last = blocks[-1].strip().splitlines()
        failed = [x for x in last if " FAIL " in f" {x} "]
        ended = any(k in x for x in last
                    for k in (" DONE ", "no new session", "nothing to record yet", "DRY:"))
        st = "FAIL" if failed else ("OK" if ended else "WARN")
        detail = failed[0] if failed else (last[-1] if last else "empty")
        if not failed and not ended:
            detail = f"in progress or died without a FAIL line: {detail}"
        checks.append(_check("last_run", st, detail, file=str(st_files[-1])))
        # scheduler_fired: both crontab slots (07:30, 21:00 IST) run every day and
        # every run writes an "=== run" header first, so a header older than ~26h
        # means the scheduler did not fire. 2026-09-25 15:03Z -> 09-29 14:40Z had
        # none: the container was up in name only or not up at all.
        starts = re.findall(r"(?m)^=== run (\S+) ===", "\n".join(
            f.read_text() for f in st_files[-3:]))
        if starts:
            age_h = (datetime.now(UTC) - datetime.fromisoformat(
                starts[-1].replace("Z", "+00:00"))).total_seconds() / 3600
            checks.append(_check("scheduler_fired", "OK" if age_h <= 26 else "FAIL",
                                 f"last run started {starts[-1]} ({age_h:.1f} h ago)",
                                 last_start=starts[-1], age_hours=round(age_h, 1)))

    # determinism
    sp = Path("audit/paper/latest/summary.json")
    if sp.exists():
        d = json.loads(sp.read_text())["determinism"]
        st = "OK" if d["books_diverged"] == 0 else "WARN"
        checks.append(_check("determinism", st,
                             f"{d['books_diverged']} of 85 books diverged vs {d['previous']} "
                             f"(max rel {d['max_abs_rel_diff']:.1e})"))

    # scheduler (host only)
    dk = shutil.which("docker") or str(Path.home() / ".docker/bin/docker")
    if Path(dk).exists() and not Path("/.dockerenv").exists():
        r = subprocess.run([dk, "inspect", "-f", "{{.State.Status}} {{.RestartCount}}",
                            "trading-bot-paper-1"], capture_output=True, text=True)
        state = r.stdout.strip() or r.stderr.strip()
        checks.append(_check("scheduler", "OK" if state.startswith("running") else "FAIL",
                             f"paper container: {state}"))

    order = {"OK": 0, "WARN": 1, "FAIL": 2}
    overall = max((c["status"] for c in checks), key=order.__getitem__, default="FAIL")
    res = {"checked_at": datetime.now(UTC).isoformat(timespec="seconds"),
           "overall": overall, "checks": checks}
    OUT.write_text(json.dumps(res, indent=2))
    print(f"OVERALL: {overall}   ({res['checked_at']})")
    for c in checks:
        print(f"  {c['status']:<5} {c['check']:<15} {c['detail']}")


if __name__ == "__main__":
    main()
