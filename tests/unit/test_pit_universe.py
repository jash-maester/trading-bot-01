"""Tests for point-in-time universe selection.

The failure mode this guards is the one CLAUDE.md calls survivorship: a rule
that quietly uses tomorrow's data to decide what to buy today looks completely
normal and inflates every backtest downstream of it. So the tests here are
mostly about the boundary — what counts as "before".
"""
from __future__ import annotations

from datetime import date, timedelta

import polars as pl
import pytest

from trader.data.pit_universe import (
    LiquidityRule,
    coverage_report,
    eligible_on,
    survivorship_gap,
    universe_schedule,
)


def _bars(
    tickers: dict[str, float],
    start: date = date(2020, 1, 1),
    days: int = 150,
    close: float = 100.0,
    series: str = "EQ",
) -> pl.DataFrame:
    """One row per (ticker, day) at a constant turnover per ticker."""
    rows: list[dict[str, object]] = []
    for t, turn in tickers.items():
        for i in range(days):
            rows.append({
                "date": start + timedelta(days=i),
                "ticker": t,
                "series": series,
                "close": close,
                "turnover": turn,
            })
    return pl.DataFrame(rows)


_LOOSE = LiquidityRule(min_median_turnover=1e7, min_sessions=10, lookback_days=365)


def test_the_bar_admits_and_excludes() -> None:
    bars = _bars({"RICH.NS": 5e7, "THIN.NS": 1e6})
    got = eligible_on(bars, date(2020, 6, 1), _LOOSE)
    assert got == ["RICH.NS"]


def test_asof_itself_is_excluded() -> None:
    """A decision on the morning of `asof` cannot use that day's turnover.

    The name trades on exactly one day. Asking on that day must return nothing;
    asking the next day must return it. Anything else is reading the tape it is
    about to trade into.
    """
    one = pl.DataFrame({
        "date": [date(2020, 3, 2)], "ticker": ["X.NS"], "series": ["EQ"],
        "close": [100.0], "turnover": [1e9],
    })
    rule = LiquidityRule(min_median_turnover=1e7, min_sessions=1)
    assert eligible_on(one, date(2020, 3, 2), rule) == []
    assert eligible_on(one, date(2020, 3, 3), rule) == ["X.NS"]


def test_only_the_lookback_window_counts() -> None:
    """Liquidity long ago does not qualify a name that has since gone quiet."""
    old = _bars({"WAS.NS": 5e8}, start=date(2018, 1, 1), days=200)
    rule = LiquidityRule(min_median_turnover=1e7, min_sessions=10, lookback_days=365)
    assert eligible_on(old, date(2018, 8, 1), rule) == ["WAS.NS"]
    assert eligible_on(old, date(2021, 1, 1), rule) == [], "stale liquidity admitted"


def test_min_sessions_blocks_a_thin_name_with_a_huge_median() -> None:
    """Two enormous prints must not read as a liquid stock.

    The median of a two-element sample is as large as its elements, so a
    turnover bar alone is satisfied by a name that traded twice.
    """
    spiky = pl.DataFrame({
        "date": [date(2020, 1, 2), date(2020, 1, 3)],
        "ticker": ["SPIKE.NS"] * 2, "series": ["EQ"] * 2,
        "close": [100.0] * 2, "turnover": [1e10, 1e10],
    })
    assert eligible_on(spiky, date(2020, 6, 1),
                       LiquidityRule(min_median_turnover=1e7, min_sessions=1)) == ["SPIKE.NS"]
    assert eligible_on(spiky, date(2020, 6, 1),
                       LiquidityRule(min_median_turnover=1e7, min_sessions=20)) == []


def test_price_floor_uses_the_latest_close_in_the_window() -> None:
    penny = _bars({"P.NS": 5e8}, close=2.0)
    assert eligible_on(penny, date(2020, 6, 1), _LOOSE) == []
    rule = LiquidityRule(min_median_turnover=1e7, min_sessions=10, min_price=1.0)
    assert eligible_on(penny, date(2020, 6, 1), rule) == ["P.NS"]


def test_non_cash_series_are_excluded() -> None:
    """`N1`/`BZ` and friends are different instruments, not the equity."""
    odd = _bars({"ODD.NS": 5e8}, series="N1")
    assert eligible_on(odd, date(2020, 6, 1), _LOOSE) == []
    eq = _bars({"ODD.NS": 5e8}, series="BE")
    assert eligible_on(eq, date(2020, 6, 1), _LOOSE) == ["ODD.NS"]


def test_null_turnover_rows_do_not_qualify() -> None:
    """The MTO delivery backend cannot supply turnover and leaves it null."""
    nul = pl.DataFrame({
        "date": [date(2020, 1, 1) + timedelta(days=i) for i in range(50)],
        "ticker": ["N.NS"] * 50, "series": ["EQ"] * 50,
        "close": [100.0] * 50, "turnover": [None] * 50,
    }, schema_overrides={"turnover": pl.Float64})
    assert eligible_on(nul, date(2020, 6, 1), _LOOSE) == []


def test_schedule_is_one_entry_per_rebalance() -> None:
    bars = _bars({"A.NS": 5e8, "B.NS": 1e6})
    dates = [date(2020, 3, 1), date(2020, 4, 1), date(2020, 5, 1)]
    sched = universe_schedule(bars, dates, _LOOSE)
    assert list(sched) == dates
    assert all(v == ["A.NS"] for v in sched.values())


def test_coverage_report_counts_what_a_fixed_list_misses() -> None:
    bars = _bars({"HAVE.NS": 5e8, "MISS.NS": 5e8})
    rep = coverage_report(bars, [date(2020, 6, 1)], ["HAVE.NS"], _LOOSE)
    r = rep.row(0, named=True)
    assert r["n_eligible"] == 2
    assert r["n_covered"] == 1
    assert r["n_missing"] == 1
    assert r["coverage"] == pytest.approx(0.5)


def test_survivorship_gap_separates_vanished_from_omitted() -> None:
    """The two mechanisms have different remedies and must not be conflated.

    A name that stopped trading cannot be recovered by widening a ticker list
    drawn today; a name that still trades can. Reporting them together
    overstates how much a bigger list would help.
    """
    alive = _bars({"ALIVE.NS": 5e8}, start=date(2020, 1, 1), days=400)
    dead = _bars({"DEAD.NS": 5e8}, start=date(2020, 1, 1), days=100)
    bars = pl.concat([alive, dead])
    gap = survivorship_gap(
        bars, date(2020, 5, 1), current_universe=[], rule=_LOOSE,
        still_trading_after=date(2020, 12, 1),
    )
    reasons = dict(zip(gap["ticker"].to_list(), gap["reason"].to_list()))
    assert reasons["DEAD.NS"] == "vanished"
    assert reasons["ALIVE.NS"] == "omitted"


def test_survivorship_gap_is_empty_when_the_list_covers_everything() -> None:
    bars = _bars({"A.NS": 5e8})
    gap = survivorship_gap(bars, date(2020, 6, 1), ["A.NS"], _LOOSE)
    assert gap.is_empty()


def test_missing_columns_are_refused() -> None:
    with pytest.raises(ValueError, match="missing column"):
        eligible_on(pl.DataFrame({"date": [date(2020, 1, 1)]}), date(2020, 2, 1))


def test_zero_min_sessions_is_refused() -> None:
    """"At least 0 sessions" is satisfied by a name that never traded."""
    with pytest.raises(ValueError, match="min_sessions"):
        LiquidityRule(min_sessions=0)


def test_rule_describes_itself() -> None:
    """The bar must be quotable in an artefact, not a hidden constant."""
    d = LiquidityRule().describe()
    assert "50,000,000" in d and "365d" in d and "EQ/BE" in d
