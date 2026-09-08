"""Tests for NSE's industry classification.

The failure this guards is CLAUDE.md's phantom sector: an id that is not in the
id table becoming a graph node nobody intended. Every path here either produces
a real id, produces the explicit unknown bucket, or raises — never a silent 0.
"""
from __future__ import annotations

import polars as pl
import pytest

from trader.data.nse_industry import (
    INDUSTRY_SCHEMA,
    NSE_INDUSTRY_IDS,
    UNKNOWN_INDUSTRY_ID,
    IndustryParseError,
    industry_ids,
    parse_industry_list,
)

_CSV = (
    "Company Name,Industry,Symbol,Series,ISIN Code\n"
    "360 ONE WAM Ltd.,Financial Services,360ONE,EQ,INE466L01038\n"
    "Tata Consultancy Services Ltd.,Information Technology,TCS,EQ,INE467B01029\n"
    "Some Realty Ltd.,Realty,SOMEREAL,EQ,INE000A01001\n"
)


def test_constituents_parse_to_ids() -> None:
    df = parse_industry_list(_CSV)
    assert list(df.columns) == list(INDUSTRY_SCHEMA)
    by = dict(zip(df["ticker"].to_list(), df["industry_id"].to_list()))
    assert by["360ONE.NS"] == NSE_INDUSTRY_IDS["Financial Services"]
    assert by["TCS.NS"] == NSE_INDUSTRY_IDS["Information Technology"]
    assert by["SOMEREAL.NS"] == NSE_INDUSTRY_IDS["Realty"]


def test_zero_is_never_a_valid_id() -> None:
    """0 is what `sector_id_of` returns by accident; nothing here may produce it."""
    assert 0 not in NSE_INDUSTRY_IDS.values()
    assert UNKNOWN_INDUSTRY_ID != 0
    df = parse_industry_list(_CSV)
    assert 0 not in df["industry_id"].to_list()


def test_ids_are_unique() -> None:
    """A duplicated id silently merges two industries into one graph node."""
    ids = list(NSE_INDUSTRY_IDS.values())
    assert len(ids) == len(set(ids))
    assert UNKNOWN_INDUSTRY_ID not in ids


def test_an_unrecognised_industry_raises_rather_than_becoming_unknown() -> None:
    """A new NSE industry is a taxonomy change, not a reclassification.

    Falling through to `unknown` would move every name in a newly-published
    industry into the unclassified bucket without anyone noticing.
    """
    bad = (
        "Company Name,Industry,Symbol,Series,ISIN Code\n"
        "New Thing Ltd.,Quantum Widgets,NEWTHING,EQ,INE111A01011\n"
    )
    with pytest.raises(IndustryParseError, match="Quantum Widgets"):
        parse_industry_list(bad)


def test_a_non_constituent_body_raises() -> None:
    with pytest.raises(IndustryParseError, match="Symbol/Industry"):
        parse_industry_list("<html>404</html>")


def test_every_requested_ticker_gets_an_id() -> None:
    """A missing name must map to the explicit bucket, never to None or 0.

    A None reaching `align_panel` becomes a 0, and a 0 is the phantom sector.
    """
    df = parse_industry_list(_CSV)
    got = industry_ids(df, ["TCS.NS", "DELISTED.NS", "360ONE.NS"])
    assert set(got) == {"TCS.NS", "DELISTED.NS", "360ONE.NS"}
    assert got["DELISTED.NS"] == UNKNOWN_INDUSTRY_ID
    assert all(v != 0 for v in got.values())
    assert all(isinstance(v, int) for v in got.values())


def test_industry_ids_needs_the_columns() -> None:
    with pytest.raises(ValueError, match="ticker"):
        industry_ids(pl.DataFrame({"x": [1]}), ["A.NS"])
