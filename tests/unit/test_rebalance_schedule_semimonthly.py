"""Semimonthly schedule: first session on/after the 1st and on/after the 16th."""
from datetime import date, timedelta

from trader.allocator.rebalance import RebalanceSchedule


def _weekdays(a: date, b: date) -> list[date]:
    out, d = [], a
    while d <= b:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def test_semimonthly_trades_on_first_session_after_1st_and_16th():
    days = _weekdays(date(2026, 9, 28), date(2026, 11, 20))
    m = RebalanceSchedule("semimonthly").mask(days)
    got = [d for d, x in zip(days, m, strict=True) if x]
    # 28 Sep is the calendar edge; then 1 Oct, 16 Oct (Fri), 2 Nov (1 Nov is Sunday), 16 Nov
    assert got == [date(2026, 9, 28), date(2026, 10, 1), date(2026, 10, 16),
                   date(2026, 11, 2), date(2026, 11, 16)]


def test_semimonthly_includes_every_monthly_rebalance():
    days = _weekdays(date(2026, 1, 1), date(2026, 12, 31))
    semi = RebalanceSchedule("semimonthly").mask(days)
    mon = RebalanceSchedule("monthly").mask(days)
    assert all(s for s, m in zip(semi, mon, strict=True) if m)
    assert semi.sum() == 2 * mon.sum()
