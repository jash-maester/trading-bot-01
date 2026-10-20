"""NSE trading days for the live books: weekends and NSE's official holidays.

Reads `data/ext/market_holidays.json`, written by
`scripts/fetch_market_holidays.py` from NSE's holiday master (segment CM).

A year the file does not cover is reported as unknown (`covered(d)` is
False). Callers then fall back to what they did before this module existed:
asking Kite whether anything traded. Special sessions held on a weekend or
holiday, such as Diwali Muhurat trading, are NOT trading days here. The books
trade only the regular 09:15 session.
"""
from __future__ import annotations

import json
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path

DEFAULT = Path("data/ext/market_holidays.json")


@lru_cache(maxsize=4)
def _load(path: str, mtime: float) -> tuple[dict[str, str], frozenset[str]]:
    doc = json.loads(Path(path).read_text())
    return doc.get("NSE_CM", {}), frozenset(doc.get("years", []))


def _data(path: Path = DEFAULT) -> tuple[dict[str, str], frozenset[str]]:
    try:
        return _load(str(path), path.stat().st_mtime)
    except (OSError, ValueError):
        return {}, frozenset()


def covered(d: date, path: Path = DEFAULT) -> bool:
    """True if the holiday list includes d's year, so `is_trading_day` is reliable."""
    return str(d.year) in _data(path)[1]


def holiday_name(d: date, path: Path = DEFAULT) -> str | None:
    """NSE's description if d is a listed trading holiday, else None."""
    return _data(path)[0].get(d.isoformat())


def is_trading_day(d: date, path: Path = DEFAULT) -> bool:
    """Weekday and not an NSE holiday. In an uncovered year, weekday only."""
    return d.weekday() < 5 and holiday_name(d, path) is None


def next_trading_day(d: date, path: Path = DEFAULT, *, inclusive: bool = True) -> date:
    """First trading day on or after d (or strictly after, if inclusive=False)."""
    x = d if inclusive else d + timedelta(days=1)
    for _ in range(30):
        if is_trading_day(x, path):
            return x
        x += timedelta(days=1)
    raise ValueError(f"no trading day within 30 days of {d}")
