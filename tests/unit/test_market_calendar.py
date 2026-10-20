"""NSE holiday calendar (src/trader/data/market_calendar.py)."""
from __future__ import annotations

import json
from datetime import date

from trader.data import market_calendar as mc


def _write(tmp_path, hol, years):
    p = tmp_path / "h.json"
    p.write_text(json.dumps({"NSE_CM": hol, "years": years}))
    return p


def test_listed_holiday_weekend_and_next_session(tmp_path):
    p = _write(tmp_path, {"2026-10-20": "Dussehra"}, ["2026"])
    assert mc.holiday_name(date(2026, 10, 20), p) == "Dussehra"
    assert not mc.is_trading_day(date(2026, 10, 20), p)
    assert not mc.is_trading_day(date(2026, 10, 24), p)          # Saturday
    assert mc.is_trading_day(date(2026, 10, 21), p)
    assert mc.next_trading_day(date(2026, 10, 20), p) == date(2026, 10, 21)
    assert mc.next_trading_day(date(2026, 10, 23), p, inclusive=False) == date(2026, 10, 26)


def test_uncovered_year_and_missing_file_fall_back_to_weekdays(tmp_path):
    p = _write(tmp_path, {"2026-10-20": "Dussehra"}, ["2026"])
    assert not mc.covered(date(2027, 1, 26), p)
    assert mc.is_trading_day(date(2027, 1, 26), p)
    missing = tmp_path / "none.json"
    assert not mc.covered(date(2026, 10, 20), missing)
    assert mc.is_trading_day(date(2026, 10, 20), missing)
