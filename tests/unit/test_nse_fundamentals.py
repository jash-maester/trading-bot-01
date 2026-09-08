"""Quarterly results, and the visibility rule that makes them usable.

The whole point of this module is one property: a figure must not reach the
model before the day it was announced. Every test here exists to pin some part
of that, because the failure mode is silent and produces a spectacular backtest
rather than an error.
"""
from __future__ import annotations

import json
import tempfile
from datetime import date, datetime
from pathlib import Path

import polars as pl
import pytest

from trader.data.sources.nse_fundamentals import (
    RESULTS_SCHEMA,
    ResultsParseError,
    attach_visibility,
    latest_per_period,
    parse_results,
    visible_from,
)


def _filing(**kw: object) -> dict[str, object]:
    base = {
        "symbol": "INFY",
        "companyName": "Infosys Limited",
        "fromDate": "01-Oct-2024",
        "toDate": "31-Dec-2024",
        "relatingTo": "Third Quarter",
        "financialYear": "01-Apr-2024 To 31-Mar-2025",
        "audited": "Audited",
        "consolidated": "Consolidated",
        "cumulative": "Non-cumulative",
        "indAs": "Ind-AS New",
        "broadCastDate": "16-Jan-2025 19:42:10",
        "filingDate": "16-Jan-2025 19:42",
        "xbrl": "https://nsearchives.nseindia.com/corporate/xbrl/INDAS_1.xml",
    }
    base.update(kw)
    return base


# ── the visibility rule ──────────────────────────────────────────────────────


def test_a_result_is_not_visible_on_the_day_it_was_broadcast() -> None:
    """Verified against the live feed: INFY Q3 FY25 broadcast 16-Jan-2025 19:42.

    Broadcast at 19:42 is after the 15:30 close, and this project fills at the
    NEXT session's open regardless, so the earliest usable date is the 17th.
    """
    assert visible_from(datetime(2025, 1, 16, 19, 42, 10)) == date(2025, 1, 17)


def test_an_intraday_broadcast_is_still_not_visible_same_day() -> None:
    """We trade at the next open, so even a 10am release cannot be acted on."""
    assert visible_from(datetime(2025, 1, 16, 10, 0, 0)) == date(2025, 1, 17)


def test_period_end_is_never_the_visibility_date() -> None:
    """The bug this module exists to prevent, stated as a test.

    Joining by period end would make Q3 visible on 31-Dec-2024, sixteen days
    before anyone could know it.
    """
    df = parse_results(json.dumps([_filing()]), symbol="INFY")
    row = df.row(0, named=True)
    assert row["period_to"] == date(2024, 12, 31)
    assert row["visible_from"] == date(2025, 1, 17)
    assert row["visible_from"] > row["period_to"]


def test_the_measured_lag_is_weeks_not_days() -> None:
    df = parse_results(json.dumps([_filing()]), symbol="INFY")
    r = df.row(0, named=True)
    assert (r["visible_from"] - r["period_to"]).days == 17


# ── refusing to guess ────────────────────────────────────────────────────────


def test_a_filing_with_no_knowable_date_is_dropped_not_lagged() -> None:
    """A guessed lag is exactly the lookahead this module prevents.

    Dropped when NEITHER broadcast nor filing date exists. A filing date IS
    knowable and is used instead (see the fallback test below); what is refused
    is inventing a lag from the period end.
    """
    rows = [
        _filing(),
        _filing(broadCastDate=None, filingDate=None,
                toDate="30-Sep-2024", fromDate="01-Jul-2024"),
    ]
    df = parse_results(json.dumps(rows), symbol="INFY")
    assert df.height == 1
    assert df.row(0, named=True)["period_to"] == date(2024, 12, 31)


def test_a_filing_with_no_period_is_dropped() -> None:
    df = parse_results(json.dumps([_filing(toDate=None)]), symbol="INFY")
    assert df.height == 0


def test_a_non_json_payload_raises_rather_than_returning_empty() -> None:
    """Empty and broken must be distinguishable, or a fetch bug looks like
    'this company files nothing'."""
    with pytest.raises(ResultsParseError, match="not JSON"):
        parse_results("<html>rate limited</html>", symbol="INFY")


# ── one filing per period ────────────────────────────────────────────────────


def test_consolidated_is_preferred_over_standalone() -> None:
    """An equity holder is exposed to the consolidated entity."""
    rows = [
        _filing(consolidated="Standalone", broadCastDate="16-Jan-2025 19:40:00"),
        _filing(consolidated="Consolidated", broadCastDate="16-Jan-2025 19:42:10"),
    ]
    one = latest_per_period(parse_results(json.dumps(rows), symbol="INFY"))
    assert one.height == 1
    assert one.row(0, named=True)["consolidated"] == "Consolidated"


def test_a_revision_wins_but_only_from_its_own_broadcast_date() -> None:
    """A restatement must not retro-correct the date the first figure was known.

    This is the subtle half of the point-in-time problem: the LATEST figure is
    what an investor eventually sees, but it becomes visible when the REVISION
    was published, not when the original was.
    """
    rows = [
        _filing(broadCastDate="16-Jan-2025 19:42:10"),
        _filing(broadCastDate="05-Mar-2025 11:00:00"),   # restatement
    ]
    one = latest_per_period(parse_results(json.dumps(rows), symbol="INFY"))
    assert one.height == 1
    r = one.row(0, named=True)
    assert r["broadcast_ts"] == datetime(2025, 3, 5, 11, 0, 0)
    assert r["visible_from"] == date(2025, 3, 6)


def test_different_periods_are_kept_separately() -> None:
    rows = [
        _filing(),
        _filing(fromDate="01-Jul-2024", toDate="30-Sep-2024",
                relatingTo="Second Quarter", broadCastDate="17-Oct-2024 19:45:51"),
    ]
    one = latest_per_period(parse_results(json.dumps(rows), symbol="INFY"))
    assert one.height == 2
    assert one["period_to"].to_list() == [date(2024, 9, 30), date(2024, 12, 31)]


# ── snapping onto real trading days ──────────────────────────────────────────


def test_visibility_snaps_forward_to_the_next_trading_session() -> None:
    """NSE broadcasts most results in the evening and many on a Friday.

    Without this the join attaches a figure to a date the market was shut, and
    the panel then carries it a day early on the following Monday.
    """
    rows = [_filing(broadCastDate="17-Jan-2025 19:42:10")]     # a Friday
    df = parse_results(json.dumps(rows), symbol="INFY")
    assert df.row(0, named=True)["visible_from"] == date(2025, 1, 18)   # Saturday
    sessions = [date(2025, 1, 16), date(2025, 1, 17), date(2025, 1, 20)]
    snapped = attach_visibility(df, sessions)
    assert snapped.row(0, named=True)["visible_from"] == date(2025, 1, 20)


def test_a_filing_after_the_last_session_is_null_not_clamped() -> None:
    """Clamping would make a future filing visible on the last known day."""
    df = parse_results(json.dumps([_filing()]), symbol="INFY")
    snapped = attach_visibility(df, [date(2024, 1, 2), date(2024, 1, 3)])
    assert snapped.row(0, named=True)["visible_from"] is None


def test_snapping_an_empty_frame_is_a_no_op() -> None:
    empty = pl.DataFrame(schema=RESULTS_SCHEMA)
    assert attach_visibility(empty, [date(2025, 1, 1)]).is_empty()


def test_empty_trading_calendar_is_refused() -> None:
    df = parse_results(json.dumps([_filing()]), symbol="INFY")
    with pytest.raises(ValueError, match="trading_days is empty"):
        attach_visibility(df, [])


# ── plumbing ─────────────────────────────────────────────────────────────────


def test_symbols_are_mapped_to_the_projects_ticker_convention() -> None:
    df = parse_results(json.dumps([_filing()]), symbol="INFY")
    assert df.row(0, named=True)["ticker"] == "INFY.NS"


def test_schema_is_exactly_the_pinned_contract() -> None:
    df = parse_results(json.dumps([_filing()]), symbol="INFY")
    assert list(df.columns) == list(RESULTS_SCHEMA)


# ── XBRL: the figures ────────────────────────────────────────────────────────

_XBRL_FIXTURE = "tests/fixtures/nse/results_xbrl_INFY_Q3FY25.xml"


def _xbrl() -> str:
    from pathlib import Path

    return Path(_XBRL_FIXTURE).read_text()


def test_xbrl_period_comes_from_the_facts_not_the_context_header() -> None:
    """The trap this parser exists to avoid, on the real document.

    Contexts OneD and FourD BOTH declare 2024-10-01..2024-12-31, but FourD's own
    DateOfStartOfReportingPeriod says 2024-04-01 — it is the nine-month
    cumulative wearing the quarter's header. Trusting the header trebles revenue
    and does it silently.
    """
    from trader.data.sources.nse_fundamentals import parse_results_xbrl

    df = parse_results_xbrl(_xbrl(), name="INFY Q3FY25")
    spans = {
        (r["period_from"], r["period_to"]): r["revenue"] for r in df.iter_rows(named=True)
    }
    assert (date(2024, 10, 1), date(2024, 12, 31)) in spans      # the quarter
    assert (date(2024, 4, 1), date(2024, 12, 31)) in spans       # nine months
    q = spans[(date(2024, 10, 1), date(2024, 12, 31))]
    ytd = spans[(date(2024, 4, 1), date(2024, 12, 31))]
    assert q == pytest.approx(417_640_000_000.0)
    assert ytd == pytest.approx(1_220_640_000_000.0)
    assert ytd > 2.5 * q, "the cumulative must not be mistaken for the quarter"


def test_quarterly_only_drops_the_cumulative() -> None:
    from trader.data.sources.nse_fundamentals import parse_results_xbrl, quarterly_only

    q = quarterly_only(parse_results_xbrl(_xbrl()))
    assert q.height == 1
    r = q.row(0, named=True)
    assert r["period_from"] == date(2024, 10, 1)
    assert r["period_to"] == date(2024, 12, 31)


def test_xbrl_headline_figures_are_extracted_and_internally_consistent() -> None:
    """Cross-checks that catch a mis-mapped tag, which a shape test would not."""
    from trader.data.sources.nse_fundamentals import parse_results_xbrl, quarterly_only

    r = quarterly_only(parse_results_xbrl(_xbrl())).row(0, named=True)
    assert r["ticker"] == "INFY.NS"
    assert r["consolidated"] == "Consolidated"
    assert r["audited"] == "Audited"
    assert r["reporting_quarter"] == "Third quarter"
    assert r["board_meeting_date"] == date(2025, 1, 16)
    # income = revenue + other income
    assert r["total_income"] == pytest.approx(r["revenue"] + r["other_income"], rel=1e-6)
    # pbt = income - expenses
    assert r["pbt"] == pytest.approx(r["total_income"] - r["total_expenses"], rel=1e-6)
    # net profit = pbt - tax
    assert r["net_profit"] == pytest.approx(r["pbt"] - r["tax_expense"], rel=1e-6)
    assert r["eps_basic"] == pytest.approx(16.43)
    assert r["eps_diluted"] <= r["eps_basic"]


def test_segment_breakdowns_do_not_leak_into_the_headline() -> None:
    """Dimensioned contexts are segment splits; including them would double-count."""
    from trader.data.sources.nse_fundamentals import parse_results_xbrl

    df = parse_results_xbrl(_xbrl())
    assert df.height == 2, "only the quarter and the year-to-date are headline rows"


def test_a_non_results_xbrl_raises() -> None:
    from trader.data.sources.nse_fundamentals import XBRLParseError, parse_results_xbrl

    # No Symbol anywhere: rejected before any period logic runs.
    with pytest.raises(XBRLParseError, match="no Symbol fact"):
        parse_results_xbrl('<?xml version="1.0"?><xbrl xmlns="x"><a>1</a></xbrl>')


def test_a_document_with_a_symbol_but_no_period_raises() -> None:
    """A filing that is not a results statement must not yield an empty frame."""
    from trader.data.sources.nse_fundamentals import XBRLParseError, parse_results_xbrl

    doc = (
        '<?xml version="1.0"?><xbrl xmlns="x">'
        '<Symbol contextRef="C1">INFY</Symbol></xbrl>'
    )
    with pytest.raises(XBRLParseError, match="not a quarterly results filing"):
        parse_results_xbrl(doc)


def test_malformed_xml_raises_rather_than_returning_empty() -> None:
    from trader.data.sources.nse_fundamentals import XBRLParseError, parse_results_xbrl

    with pytest.raises(XBRLParseError, match="not well-formed"):
        parse_results_xbrl("<xbrl><unclosed>")


def test_filing_date_is_a_legitimate_fallback_for_a_missing_broadcast_date() -> None:
    """Pre-2008 filings often carry filingDate but no broadCastDate.

    filingDate is KNOWABLE -- the exchange timestamped it -- so falling back to
    it is not the same as guessing a fixed lag from the period end, which is
    what this module refuses to do.
    """
    rows = [_filing(broadCastDate=None, filingDate="24-May-2007 17:27",
                    fromDate="01-Jan-2007", toDate="31-Mar-2007")]
    df = parse_results(json.dumps(rows), symbol="BPCL")
    assert df.height == 1
    r = df.row(0, named=True)
    assert r["visible_from"] == date(2007, 5, 25)
    assert r["visible_from"] > r["period_to"]


def test_a_filing_with_neither_date_is_still_dropped() -> None:
    rows = [_filing(broadCastDate=None, filingDate="-")]
    assert parse_results(json.dumps(rows), symbol="BPCL").height == 0


def test_a_dash_xbrl_link_is_null_not_a_url() -> None:
    df = parse_results(json.dumps([_filing(xbrl="-")]), symbol="INFY")
    assert df.row(0, named=True)["xbrl_url"] is None


def test_a_placeholder_url_ending_in_a_dash_is_rejected() -> None:
    """The real shape of the placeholder, and 54% of the index carries it.

    NSE does not write a bare "-": it writes a well-formed URL whose last path
    segment is "-". A startswith("http") check passes every one, which would
    have sent a six-hour fetch to collect 11,781 404s.
    """
    url = "https://nsearchives.nseindia.com/corporate/xbrl/-"
    df = parse_results(json.dumps([_filing(xbrl=url)]), symbol="INFY")
    assert df.row(0, named=True)["xbrl_url"] is None


def test_a_real_xbrl_url_survives() -> None:
    url = "https://nsearchives.nseindia.com/corporate/xbrl/INDAS_1_2_3.xml"
    df = parse_results(json.dumps([_filing(xbrl=url)]), symbol="INFY")
    assert df.row(0, named=True)["xbrl_url"] == url


def test_a_non_http_xbrl_value_is_rejected() -> None:
    df = parse_results(json.dumps([_filing(xbrl="NA")]), symbol="INFY")
    assert df.row(0, named=True)["xbrl_url"] is None


def test_symbol_is_percent_encoded_in_the_url() -> None:
    """`&` in a symbol is a query separator, not a character.

    Measured against the live endpoint on 2026-09-08: `symbol=M&M` reaches NSE
    as `symbol=M` and comes back as a two-byte empty list -- no error, no
    warning, no filings. Percent-encoded it returns 81,896 bytes and 99
    filings. Six universe members carry an ampersand (M&M, M&MFIN, J&KBANK,
    ARE&M, GVT&D, GMRP&UI) and every one of them was silently missing from the
    fundamentals because of this.
    """
    from trader.data.sources.nse_fundamentals import RESULTS_URL, FinancialResultsSource

    class _Recorder:
        def __init__(self) -> None:
            self.urls: list[str] = []

        def get_text(self, url: str) -> str:
            self.urls.append(url)
            return "[]"

    rec = _Recorder()
    src = FinancialResultsSource(
        cache_root=Path(tempfile.mkdtemp()), client=rec, period="Quarterly"
    )
    src.fetch_symbol("M&M")
    assert rec.urls, "no request was made"
    url = rec.urls[0]
    assert "symbol=M%26M" in url, f"symbol not encoded: {url}"
    # The decisive property: exactly one `&` before `symbol`, and none inside
    # its value, so the server cannot read a second parameter out of the name.
    tail = url.split("symbol=", 1)[1]
    assert not tail.split("&", 1)[0].endswith("M&M")
    assert RESULTS_URL.count("{symbol}") == 1


def test_ampersand_symbol_cache_file_is_distinct() -> None:
    """Two symbols must not collide on one cache file.

    Encoding the symbol for the URL and NOT for the filename would be worse
    than the bug it fixes: `M&M` and `M%26M` would fetch correctly but share a
    path with whatever else normalised to it.
    """
    from trader.data.sources.nse_fundamentals import FinancialResultsSource

    class _Client:
        def get_text(self, url: str) -> str:
            return "[]"

    root = Path(tempfile.mkdtemp())
    src = FinancialResultsSource(cache_root=root, client=_Client(), period="Quarterly")
    src.fetch_symbol("M&M")
    src.fetch_symbol("MM")
    names = {p.name for p in (root / "results").rglob("*.json")}
    assert len(names) >= 2, f"cache collided: {names}"


# ── XBRL: the banking taxonomy ───────────────────────────────────────────────

_BANK_FIXTURE = "tests/fixtures/nse/results_xbrl_AUBANK_Q3FY25.xml"


def _bank_xbrl() -> str:
    return Path(_BANK_FIXTURE).read_text()


def test_banking_filings_yield_figures_not_a_row_of_nulls() -> None:
    """Banks file under a different element set, and the failure is silent.

    There is no `RevenueFromOperations` or `ProfitLossForPeriod` anywhere in a
    banking document. Before the alias map the parser read one without error
    and returned a row per period with EVERY financial column null — which is
    worse than raising, because an empty row joins fine and simply carries no
    information into whatever consumes it. 33 universe members file this way,
    including several of the largest index weights.
    """
    from trader.data.sources.nse_fundamentals import parse_results_xbrl, quarterly_only

    r = quarterly_only(parse_results_xbrl(_bank_xbrl(), name="AUBANK Q3FY25")).row(
        0, named=True
    )
    assert r["symbol"] == "AUBANK"
    assert r["period_from"] == date(2024, 10, 1)
    assert r["period_to"] == date(2024, 12, 31)
    for col in ("revenue", "pbt", "net_profit", "eps_basic", "employee_cost",
                "finance_costs", "paid_up_equity", "face_value"):
        assert r[col] is not None, f"{col} is null on a banking filing"


def test_banking_figures_are_internally_consistent() -> None:
    """Cross-checks that catch a mis-mapped bank tag, which a null check cannot.

    `InterestExpended` and `EmployeesCost` are both plausible landing spots for
    a careless mapping, and swapping them would still pass a not-null test.
    """
    from trader.data.sources.nse_fundamentals import parse_results_xbrl, quarterly_only

    r = quarterly_only(parse_results_xbrl(_bank_xbrl())).row(0, named=True)
    assert r["revenue"] == pytest.approx(41_134_753_000.0)     # InterestEarned
    assert r["finance_costs"] == pytest.approx(20_907_687_000.0)  # InterestExpended
    assert r["employee_cost"] == pytest.approx(7_546_621_000.0)
    assert r["pbt"] == pytest.approx(7_032_298_000.0)
    assert r["net_profit"] == pytest.approx(5_284_463_000.0)
    assert r["net_profit"] < r["pbt"], "post-tax profit exceeds pre-tax"
    assert r["finance_costs"] < r["revenue"], "a bank paying out more than it earns"

    # Shares from paid-up equity / face value must reproduce EPS, which is the
    # end-to-end check: it ties the balance-sheet-ish pair to the P&L.
    shares = r["paid_up_equity"] / r["face_value"]
    assert shares == pytest.approx(744_230_000, rel=0.01)
    assert r["net_profit"] / shares == pytest.approx(r["eps_basic"], rel=0.02)


def test_commercial_tags_win_over_banking_aliases() -> None:
    """The alias map is a fallback, never an override.

    A filing carrying both taxonomies must keep the commercial value, so adding
    bank support cannot silently change a number that already parsed.
    """
    from trader.data.sources.nse_fundamentals import parse_results_xbrl

    doc = (
        '<?xml version="1.0"?><xbrl xmlns="x">'
        '<context id="OneD"><period><startDate>2024-10-01</startDate>'
        "<endDate>2024-12-31</endDate></period></context>"
        '<Symbol contextRef="OneD">TESTCO</Symbol>'
        '<DateOfStartOfReportingPeriod contextRef="OneD">2024-10-01'
        "</DateOfStartOfReportingPeriod>"
        '<DateOfEndOfReportingPeriod contextRef="OneD">2024-12-31'
        "</DateOfEndOfReportingPeriod>"
        '<RevenueFromOperations contextRef="OneD">100</RevenueFromOperations>'
        '<InterestEarned contextRef="OneD">999</InterestEarned>'
        "</xbrl>"
    )
    r = parse_results_xbrl(doc).row(0, named=True)
    assert r["revenue"] == pytest.approx(100.0), "the banking alias overrode a real tag"
