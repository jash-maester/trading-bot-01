"""Tests for the rebalance calendar."""
from __future__ import annotations


def test_quarterly_trades_four_times_a_year() -> None:
    """Calendar quarters, so the boundary is the same every year.

    Added because monthly turnover realises nearly every gain inside twelve
    months -- the STCG boundary, 20% + cess against LTCG's 12.5% -- so a longer
    hold cuts turnover AND moves lots toward the lower rate.
    """
    from datetime import date, timedelta

    from trader.allocator.rebalance import RebalanceSchedule

    days = []
    d = date(2020, 1, 1)
    while d <= date(2021, 12, 31):
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    mask = list(RebalanceSchedule("quarterly").mask(days))
    hits = [d for d, m in zip(days, mask) if m]
    assert len(hits) == 8, f"two years should give eight quarters, got {len(hits)}"
    assert [h.month for h in hits] == [1, 4, 7, 10, 1, 4, 7, 10]
    assert [h.year for h in hits] == [2020] * 4 + [2021] * 4


def test_quarterly_is_a_strict_subset_of_monthly() -> None:
    """Every quarter start is a month start; the reverse is not true."""
    from datetime import date, timedelta

    from trader.allocator.rebalance import RebalanceSchedule

    days = []
    d = date(2020, 1, 1)
    while d <= date(2020, 12, 31):
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    q = {d for d, m in zip(days, RebalanceSchedule("quarterly").mask(days)) if m}
    m = {d for d, m2 in zip(days, RebalanceSchedule("monthly").mask(days)) if m2}
    assert q < m, "quarterly should be a proper subset of monthly"
    assert len(q) == 4 and len(m) == 12


def test_rebalances_per_year_knows_quarterly() -> None:
    """The fee estimator would silently KeyError without it."""
    from trader.allocator.sizing import REBALANCES_PER_YEAR

    assert REBALANCES_PER_YEAR["quarterly"] == 4.0
