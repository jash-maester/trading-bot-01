"""dashboard/api.py: market clock, next rebalance, attention banner. No network."""
from __future__ import annotations

import sys
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "dashboard"))
import api  # noqa: E402

IST = api.IST


def at(s: str) -> datetime:
    return datetime.fromisoformat(s).replace(tzinfo=IST)


def test_phase():
    assert api.phase(at("2026-10-01T09:05")) == "pre_open"
    assert api.phase(at("2026-10-01T09:15")) == "open"
    assert api.phase(at("2026-10-01T15:29")) == "open"
    assert api.phase(at("2026-10-01T15:30")) == "closed"
    assert api.phase(at("2026-10-03T11:00")) == "weekend"


def test_next_rebalance_is_first_weekday_of_next_month():
    assert api.next_rebalance("2026-09-30", date(2026, 9, 30)) == date(2026, 10, 1)
    # 1 Nov 2026 is a Sunday
    assert api.next_rebalance("2026-10-01", date(2026, 10, 5)) == date(2026, 11, 2)
    assert api.next_rebalance("2026-12-01", date(2026, 12, 5)) == date(2027, 1, 1)


def test_next_live_event_skips_weekend():
    e = api.next_live_event(at("2026-10-02T16:00"))       # Friday after the close
    assert e["at"].startswith("2026-10-05T09:00")


def _pf(ts="2026-10-01T10:00:00+05:30", prices="live"):
    return {"deployed": True, "snapshot": {"ts": ts, "prices": prices}}


def test_kite_banner_is_urgent_before_the_trade_and_warn_in_the_evening():
    pipe = {"status": "OK", "reasons": []}
    morning = api.attention(at("2026-10-01T07:30"), {"state": "expired"}, pipe, _pf())
    assert morning[0]["action"] == "kite" and morning[0]["level"] == "urgent"
    assert "09:16" in morning[0]["message"]
    evening = api.attention(at("2026-10-01T20:00"), {"state": "expired"}, pipe, _pf())
    assert evening[0]["level"] == "warn"
    saturday = api.attention(at("2026-10-03T08:00"), {"state": "missing"}, pipe, _pf())
    assert saturday[0]["level"] == "warn"
    assert api.attention(at("2026-10-01T07:30"), {"state": "valid"}, pipe, _pf()) == []


def test_stale_marks_and_pipeline_fail():
    ok = {"state": "valid"}
    stale = api.attention(at("2026-10-01T11:00"), ok, {"status": "OK", "reasons": []},
                          _pf("2026-10-01T10:15:00+05:30"))
    assert [a["title"] for a in stale] == ["Prices are stale"]
    fail = api.attention(at("2026-10-01T20:00"), ok, {"status": "FAIL", "reasons": ["x"]}, _pf())
    assert fail[0]["level"] == "urgent" and fail[0]["action"] == "health"
