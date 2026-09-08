"""Tests for the corporate-actions feed, written after the previous one failed.

The adjustment layer was first built on the belief that NSE publishes
``PREVCLOSE`` already adjusted for overnight actions. It does not. The detector
that belief produced was INERT, and the measurement that appeared to prove it
precise — 23,693 of 23,693 ratios exactly 1.0 — was the symptom.

Those two states are indistinguishable unless you test against a KNOWN action,
so that is what this file does, with three splits whose dates and ratios are
matters of public record.
"""
from __future__ import annotations

import json
from datetime import date

import polars as pl
import pytest

from trader.data.sources.nse_corporate_actions import (
    CA_SCHEMA,
    CorporateActionParseError,
    back_adjust_with_actions,
    parse_corporate_actions,
    price_factor_for,
)


@pytest.mark.parametrize(
    ("subject", "expected"),
    [
        ("Face Value Split (Sub-Division) - From Rs 10/- Per Share To Re 1/- Per Share",
         ("split", 0.1)),
        ("Face Value Split (Sub-Division) - From Rs 5/- Per Share To Re 1/- Per Share",
         ("split", 0.2)),
        ("Face Value Split (Sub-Division) - From Rs 10/- Per Share To Rs 2/- Per Share",
         ("split", 0.2)),
        ("Face Value Split(Sub-Division) - From Rs.2/- To Re.1/-", ("split", 0.5)),
        # A reverse split moves the other way and must not be dropped: ten
        # shares becoming one is a tenfold price change.
        ("Face Value Consolidation - From Re 1/- Per Share To Rs 10/- Per Share",
         ("consolidation", 10.0)),
        ("Bonus 1:1", ("bonus", 0.5)),
        ("Bonus 3:1", ("bonus", 0.25)),
        ("Bonus 1:2", ("bonus", 2 / 3)),
    ],
)
def test_the_ratio_is_read_from_the_subject(subject: str, expected: tuple) -> None:
    kind, factor = price_factor_for(subject)  # type: ignore[misc]
    assert kind == expected[0]
    assert factor == pytest.approx(expected[1])


@pytest.mark.parametrize(
    "subject",
    [
        "Dividend - Rs 6.00 Per Share",
        "Interest Payment",
        "Annual General Meeting",
        "Rights 1:4",
        "",
    ],
)
def test_non_price_actions_are_ignored(subject: str) -> None:
    """Dividends especially.

    Kite's bars are adjusted for splits and bonuses but NOT ordinary dividends,
    so adjusting for them here would make the two price sources disagree in a
    way no downstream comparison could detect.
    """
    assert price_factor_for(subject) is None


def _feed(rows: list[dict[str, str]]) -> str:
    return json.dumps(rows)


def test_parses_the_api_shape() -> None:
    text = _feed([
        {"symbol": "NESTLEIND", "series": "EQ", "exDate": "05-Jan-2024",
         "isin": "INE239A01024",
         "subject": "Face Value Split (Sub-Division) - From Rs 10/- Per Share "
                    "To Re 1/- Per Share"},
        {"symbol": "GLOBUS", "series": "EQ", "exDate": "08-Sep-2026",
         "isin": "INE615I01010", "subject": "Dividend - Rs 6.00 Per Share"},
        {"symbol": "GOI", "series": "GS", "exDate": "01-Jan-2024",
         "isin": "IN0020100031", "subject": "Interest Payment"},
    ])
    df = parse_corporate_actions(text)
    assert list(df.columns) == list(CA_SCHEMA)
    assert df.height == 1, "only the split is an adjustment on a cash series"
    r = df.row(0, named=True)
    assert r["ticker"] == "NESTLEIND.NS"
    assert r["ex_date"] == date(2024, 1, 5)
    assert r["price_factor"] == pytest.approx(0.1)


def test_a_non_json_body_raises() -> None:
    with pytest.raises(CorporateActionParseError, match="not JSON"):
        parse_corporate_actions("<html>404</html>")


# ── the three known splits, end to end ───────────────────────────────────────
#
# Real closes from data/ext/bhavcopy.parquet. Each pair is the last raw close
# before the ex-date and the first raw close on it.
_KNOWN = [
    ("NESTLEIND.NS", date(2024, 1, 5), 27116.40, 2666.40, 0.1,
     "Face Value Split (Sub-Division) - From Rs 10/- Per Share To Re 1/- Per Share"),
    ("HDFCBANK.NS", date(2019, 9, 19), 2187.75, 1101.05, 0.5,
     "Face Value Split (Sub-Division) - From Rs 2/- Per Share To Re 1/- Per Share"),
    ("IRCTC.NS", date(2021, 10, 28), 4130.15, 913.50, 0.2,
     "Face Value Split (Sub-Division) - From Rs 10/- Per Share To Rs 2/- Per Share"),
]


@pytest.mark.parametrize(("ticker", "ex", "before", "after", "factor", "subject"), _KNOWN)
def test_a_known_split_is_adjusted_into_a_continuous_series(
    ticker: str, ex: date, before: float, after: float, factor: float, subject: str
) -> None:
    """The test the previous implementation never had.

    Raw, the ex-date shows a 50-90% "crash". Adjusted, the step across it must
    be an ordinary daily move — which is the only thing that distinguishes a
    working adjustment from an inert one.
    """
    bars = pl.DataFrame({
        "date": [date(ex.year, ex.month, ex.day - 1), ex],
        "ticker": [ticker] * 2,
        "close": [before, after],
        "volume": [1000, 1000],
    })
    raw_step = after / before - 1.0
    assert raw_step < -0.45, "fixture is not actually a split"

    actions = parse_corporate_actions(_feed([{
        "symbol": ticker.removesuffix(".NS"), "series": "EQ",
        "exDate": ex.strftime("%d-%b-%Y"), "isin": "X", "subject": subject,
    }]))
    out = back_adjust_with_actions(bars, actions).sort("date")
    c = out["close"].to_list()
    adj_step = c[1] / c[0] - 1.0
    assert abs(adj_step) < 0.15, (
        f"{ticker}: adjusted step {adj_step:+.1%} is still a crash — the "
        "adjustment did not fire"
    )
    # The pre-split bar is scaled and the post-split bar is untouched, because
    # adjustment is on the LATEST scale to match Kite's convention.
    assert c[0] == pytest.approx(before * factor)
    assert c[1] == pytest.approx(after)


def test_volume_moves_the_other_way_so_turnover_is_unchanged() -> None:
    bars = pl.DataFrame({
        "date": [date(2024, 1, 4), date(2024, 1, 5)],
        "ticker": ["NESTLEIND.NS"] * 2,
        "close": [27116.40, 2666.40],
        "volume": [1000, 10000],
    })
    actions = parse_corporate_actions(_feed([{
        "symbol": "NESTLEIND", "series": "EQ", "exDate": "05-Jan-2024", "isin": "X",
        "subject": "Face Value Split (Sub-Division) - From Rs 10/- Per Share "
                   "To Re 1/- Per Share"}]))
    out = back_adjust_with_actions(bars, actions).sort("date")
    for i in range(bars.height):
        assert out["close"][i] * out["volume"][i] == pytest.approx(
            bars["close"][i] * bars["volume"][i]
        )


def test_two_actions_compound() -> None:
    bars = pl.DataFrame({
        "date": [date(2020, 1, 1), date(2020, 6, 1), date(2021, 6, 1)],
        "ticker": ["A.NS"] * 3,
        "close": [100.0, 50.0, 25.0],
        "volume": [100, 200, 400],
    })
    actions = parse_corporate_actions(_feed([
        {"symbol": "A", "series": "EQ", "exDate": "01-Jun-2020", "isin": "X",
         "subject": "Bonus 1:1"},
        {"symbol": "A", "series": "EQ", "exDate": "01-Jun-2021", "isin": "X",
         "subject": "Bonus 1:1"},
    ]))
    out = back_adjust_with_actions(bars, actions).sort("date")
    # The first bar sits behind both halvings.
    assert out["close"].to_list() == pytest.approx([25.0, 25.0, 25.0])


def test_actions_do_not_leak_across_tickers() -> None:
    bars = pl.DataFrame({
        "date": [date(2024, 1, 4), date(2024, 1, 5)] * 2,
        "ticker": ["A.NS", "A.NS", "B.NS", "B.NS"],
        "close": [100.0, 10.0, 200.0, 201.0],
        "volume": [10, 100, 10, 10],
    })
    actions = parse_corporate_actions(_feed([{
        "symbol": "A", "series": "EQ", "exDate": "05-Jan-2024", "isin": "X",
        "subject": "Face Value Split (Sub-Division) - From Rs 10/- Per Share "
                   "To Re 1/- Per Share"}]))
    out = back_adjust_with_actions(bars, actions)
    b = out.filter(pl.col("ticker") == "B.NS").sort("date")
    assert b["close"].to_list() == pytest.approx([200.0, 201.0])


def test_no_actions_leaves_prices_raw_and_says_so() -> None:
    bars = pl.DataFrame({
        "date": [date(2024, 1, 4)], "ticker": ["A.NS"],
        "close": [100.0], "volume": [10],
    })
    out = back_adjust_with_actions(bars, pl.DataFrame(schema=CA_SCHEMA))
    assert out["close"].to_list() == [100.0]
