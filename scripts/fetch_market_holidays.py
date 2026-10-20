"""Fetch NSE's official trading-holiday list into data/ext/market_holidays.json.

    uv run python scripts/fetch_market_holidays.py     # cron: weekly, plus on demand

Source: https://www.nseindia.com/api/holiday-master?type=trading, segment CM
(capital market, the equity cash segment every book trades). Fetched through
the same browser-session client as the other NSE data (`NSEClient`). Verified
2026-10-19 against Kite: 20-Oct-2026 is on the list, and Kite showed no
trades that day.

The file keeps every year ever fetched (NSE serves only the current year, and
the next once announced in December), so past holidays stay known. Read by
`trader.data.market_calendar`. BSE is not fetched: no stable endpoint was
found (2026-10-20). Its equity holidays normally match NSE's (UNVERIFIED), and
every book trades NSE.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from trader.data.sources.nse_flows import NSEClient

URL = "https://www.nseindia.com/api/holiday-master?type=trading"
OUT = Path("data/ext/market_holidays.json")


def main() -> None:
    raw = json.loads(NSEClient().get_text(URL))
    cm = raw.get("CM") or []
    if not cm:
        raise SystemExit(f"NSE holiday list has no CM segment (keys: {sorted(raw)})")
    new = {datetime.strptime(x["tradingDate"], "%d-%b-%Y").date().isoformat(): x["description"]
           for x in cm}
    old = json.loads(OUT.read_text()) if OUT.exists() else {}
    hol = {**old.get("NSE_CM", {}), **new}
    doc = {
        "source": URL + " (segment CM)",
        "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "years": sorted({d[:4] for d in hol}),
        "NSE_CM": dict(sorted(hol.items())),
    }
    tmp = OUT.with_suffix(".tmp.json")
    tmp.write_text(json.dumps(doc, indent=1) + "\n")
    tmp.replace(OUT)
    yrs = ", ".join(doc["years"])
    print(f"market_holidays: {len(new)} fetched, {len(hol)} kept ({yrs}) -> {OUT}")


if __name__ == "__main__":
    main()
