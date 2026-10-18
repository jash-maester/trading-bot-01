"""Forward-only log of NSE corporate announcements, with our own first-seen time.

    uv run python scripts/nse_announcements.py            # cron, every 15 min

Intraday Phase 0 (14_weekend_scoping_notes.md §A7). Polls NSE's public
corporate-announcements API for today (and yesterday, to catch late-night
filings) and appends every announcement not seen before to
data/news/nse_announcements/<YYYY-MM-DD>.jsonl, keyed by NSE's `seq_id`.

Each line is NSE's record verbatim plus `first_seen_utc`: the time this logger
first saw it. That timestamp is the point. It is the only honest
"when could we have known" for an announcement, and it is collected live, so
no model trained before today can have seen it (the leakage that sinks the
LLM-agent studies). Read-only; nothing here trades.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

from trader.data.sources.nse_flows import NSEClient

IST = timezone(timedelta(hours=5, minutes=30))
URL = ("https://www.nseindia.com/api/corporate-announcements?index=equities"
       "&from_date={d}&to_date={d}")
OUT = Path("data/news/nse_announcements")


def _seen(path: Path) -> set[str]:
    if not path.exists():
        return set()
    out = set()
    for line in path.read_text().splitlines():
        try:
            out.add(str(json.loads(line).get("seq_id")))
        except ValueError:
            continue
    return out


def main() -> None:
    now = datetime.now(IST)
    OUT.mkdir(parents=True, exist_ok=True)
    client = NSEClient(min_interval_s=2.0)
    stamp = datetime.now(UTC).isoformat(timespec="seconds")
    total_new = 0
    for day in (now.date() - timedelta(days=1), now.date()):
        rows = json.loads(client.get_text(URL.format(d=day.strftime("%d-%m-%Y"))))
        path = OUT / f"{day.isoformat()}.jsonl"
        seen = _seen(path)
        new = [r for r in rows if str(r.get("seq_id")) not in seen]
        if new:
            with path.open("a") as f:
                for r in new:
                    f.write(json.dumps({**r, "first_seen_utc": stamp}) + "\n")
        total_new += len(new)
        print(f"{day}: {len(rows)} listed, {len(new)} new")
    print(f"{stamp} done, {total_new} new")


if __name__ == "__main__":
    main()
