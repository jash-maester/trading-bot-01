"""Unit tests for project-ticker → Kite-tradingsymbol resolution.

The load-bearing property is negative: a symbol that cannot be resolved must never
come back looking like a success, and a symbol that CAN be resolved must never be
dropped just because it was renamed. Everything else here defends the ordering of
the four lookup strategies, which is where a plausible-looking wrong answer would
come from.
"""

from __future__ import annotations

import pytest

from trader.data.sources.kite_symbols import (
    DELISTED_SYMBOLS,
    INDEX_ALIASES,
    KITE_RENAMES,
    ResolutionStatus,
    resolve_symbol,
    resolve_universe,
    to_kite_symbol,
    to_project_ticker,
)

# A stand-in instrument dump. Mirrors the real one's shape: bare symbols, one
# series-suffixed name, one renamed name, and an index in its own segment.
DUMP = frozenset(
    {
        "RELIANCE",
        "TCS",
        "LTM",
        "STLTECH-BE",
        "TMPV",
        "TMCV",
        "NIFTY 50",
    }
)


# ---------------------------------------------------------------------------
# Symbol string conversion
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("ticker", "expected"),
    [
        ("RELIANCE.NS", "RELIANCE"),
        ("reliance.ns", "RELIANCE"),
        ("  TCS.NS  ", "TCS"),
        ("RELIANCE", "RELIANCE"),
        ("^NSEI", "^NSEI"),
    ],
)
def test_to_kite_symbol(ticker: str, expected: str) -> None:
    assert to_kite_symbol(ticker) == expected


def test_to_project_ticker_is_idempotent() -> None:
    assert to_project_ticker("RELIANCE") == "RELIANCE.NS"
    assert to_project_ticker("RELIANCE.NS") == "RELIANCE.NS"


# ---------------------------------------------------------------------------
# The four resolution strategies
# ---------------------------------------------------------------------------


def test_direct_hit() -> None:
    res = resolve_symbol("RELIANCE.NS", DUMP)
    assert res.status is ResolutionStatus.DIRECT
    assert res.kite_symbol == "RELIANCE"
    assert res.is_fetchable


def test_known_rename_resolves_instead_of_dropping() -> None:
    """The whole point: LTIM disappeared from NSE but the company did not."""
    res = resolve_symbol("LTIM.NS", DUMP)
    assert res.status is ResolutionStatus.ALIAS
    assert res.kite_symbol == "LTM"


def test_series_suffix_resolves() -> None:
    res = resolve_symbol("STLTECH.NS", DUMP)
    assert res.status is ResolutionStatus.SERIES
    assert res.kite_symbol == "STLTECH-BE"
    assert "BE" in res.note


def test_delisted_is_reported_not_silently_skipped() -> None:
    res = resolve_symbol("MCDHOLDING.NS", DUMP)
    assert res.status is ResolutionStatus.DELISTED
    assert res.kite_symbol is None
    assert not res.is_fetchable
    # A reason, not a shrug: this is what distinguishes "known gone" from "bug".
    assert "suspended" in res.note.lower()


def test_unknown_symbol_is_unresolved_with_actionable_note() -> None:
    res = resolve_symbol("NOSUCHCO.NS", DUMP)
    assert res.status is ResolutionStatus.UNRESOLVED
    assert res.kite_symbol is None
    assert "KITE_RENAMES" in res.note and "DELISTED_SYMBOLS" in res.note


def test_delisted_wins_over_a_coincidental_series_match() -> None:
    """A dead name must not be resurrected by an unrelated SME listing.

    ``MCDHOLDING`` is delisted; if some ``MCDHOLDING-SM`` ever appears in the dump,
    resolving to it would silently splice a different company's prices into the
    panel. The delisted check therefore runs first.
    """
    dump = DUMP | {"MCDHOLDING-SM"}
    assert resolve_symbol("MCDHOLDING.NS", dump).status is ResolutionStatus.DELISTED


def test_stale_alias_is_unresolved_not_a_suffix_guess() -> None:
    """A rename map pointing at a symbol that is also gone must say so.

    Falling through to the suffix search here would turn "our alias map is out of
    date" into a confident wrong answer.
    """
    res = resolve_symbol("LTIM.NS", DUMP - {"LTM"})
    assert res.status is ResolutionStatus.UNRESOLVED
    assert "stale" in res.note


def test_index_alias_only_resolves_when_indices_are_in_the_dump() -> None:
    assert resolve_symbol("^NSEI", DUMP).kite_symbol == "NIFTY 50"
    # Equity-only dump: an index must not resolve, because it is not tradable.
    assert resolve_symbol("^NSEI", DUMP - {"NIFTY 50"}).kite_symbol is None


def test_tatamotors_maps_to_the_surviving_listing_not_the_spin_off() -> None:
    """TMPV keeps the 2005 history; TMCV is a 2025 listing that merely reuses the name."""
    assert resolve_symbol("TATAMOTORS.NS", DUMP).kite_symbol == "TMPV"


# ---------------------------------------------------------------------------
# Universe-level report
# ---------------------------------------------------------------------------


def test_report_counts_and_symbol_map() -> None:
    report = resolve_universe(
        ["RELIANCE.NS", "LTIM.NS", "STLTECH.NS", "MCDHOLDING.NS", "NOSUCHCO.NS"], DUMP
    )
    assert report.counts == {
        "direct": 1,
        "alias": 1,
        "series": 1,
        "delisted": 1,
        "unresolved": 1,
    }
    assert report.symbol_map == {
        "RELIANCE.NS": "RELIANCE",
        "LTIM.NS": "LTM",
        "STLTECH.NS": "STLTECH-BE",
    }
    assert len(report.fetchable) == 3


def test_report_lists_every_non_direct_ticker() -> None:
    """Direct hits are counted only — the exceptions must not be buried under them."""
    report = resolve_universe(["RELIANCE.NS", "TCS.NS", "MCDHOLDING.NS"], DUMP)
    body = "\n".join(report.format_lines()[1:])
    assert "MCDHOLDING.NS" in body
    assert "RELIANCE.NS" not in body


def test_report_preserves_input_order() -> None:
    tickers = ["TCS.NS", "RELIANCE.NS", "LTIM.NS"]
    report = resolve_universe(tickers, DUMP)
    assert [r.ticker for r in report.resolutions] == tickers


# ---------------------------------------------------------------------------
# The maps themselves
# ---------------------------------------------------------------------------


def test_maps_are_disjoint() -> None:
    """A symbol cannot be both renamed and delisted; that would be a silent conflict."""
    assert not set(KITE_RENAMES) & set(DELISTED_SYMBOLS)
    assert not set(INDEX_ALIASES) & set(KITE_RENAMES)


def test_map_keys_are_bare_upper_case_symbols() -> None:
    """Keys are looked up post-``to_kite_symbol``, so a ``.NS`` key would never hit."""
    for key in (*KITE_RENAMES, *DELISTED_SYMBOLS):
        assert key == key.upper()
        assert not key.endswith(".NS")
