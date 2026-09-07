"""Quarterly financial results from NSE, with the date each became public.

`13_fundamentals_and_news.md` section 3 identified one requirement that decides
whether fundamentals are usable at all: **a figure must not be visible to the
model before the day it was announced.** Quarterly results are published weeks
after the quarter ends and are restated afterwards, so joining them to a panel
by quarter-end hands the model a number nobody could have had — and, worse, one
that was later corrected to be right. That is the most reliable way to
manufacture a spectacular backtest, and this repo has already been burned by its
cousin: a universe of 645 tickers with zero delistings.

The F0 audit found that commercial fundamentals APIs document income statements
and balance sheets and say nothing about announcement dates. NSE publishes them
itself, because SEBI's Listing Obligations regulations require a company to tell
the exchange when its board will consider results. `corporates-financial-results`
returns, per filing:

    fromDate / toDate          the period the figures cover
    broadCastDate              when NSE released it — the field that matters
    filingDate                 when the company filed it
    audited / consolidated     Audited|Unaudited, Consolidated|Standalone
    xbrl                       a structured XBRL document, not a PDF

Verified 2026-09-07 against INFY: Q3 FY25 covers 01-Oct-2024..31-Dec-2024 and
was broadcast 16-Jan-2025 19:42 IST — a 16-day lag, and after the 15:30 close.

That last detail is not a footnote. A result broadcast after the close cannot be
traded until the **next** session, so `visible_from` is derived from the
broadcast timestamp and the market clock rather than from its date alone.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any, Final

import polars as pl
from loguru import logger

from trader.data.sources.nse_flows import nse_symbol_to_ticker

NSE_HOME: Final[str] = "https://www.nseindia.com"
RESULTS_URL: Final[str] = (
    NSE_HOME + "/api/corporates-financial-results?index=equities&symbol={symbol}"
    "&period={period}"
)

#: NSE's normal close. A filing broadcast at or after this is next-session news.
MARKET_CLOSE: Final[time] = time(15, 30)

#: NSE stamps these as ``16-Jan-2025 19:42:10``.
_TS_FORMATS: Final[tuple[str, ...]] = ("%d-%b-%Y %H:%M:%S", "%d-%b-%Y %H:%M", "%d-%b-%Y")

RESULTS_SCHEMA: Final[dict[str, pl.DataType]] = {
    "symbol": pl.Utf8(),
    "ticker": pl.Utf8(),
    "period_from": pl.Date(),
    "period_to": pl.Date(),
    "relating_to": pl.Utf8(),
    "financial_year": pl.Utf8(),
    "audited": pl.Utf8(),
    "consolidated": pl.Utf8(),
    "cumulative": pl.Utf8(),
    "ind_as": pl.Utf8(),
    # When NSE released it. NULL is not tolerated downstream: a figure with no
    # broadcast date has no knowable visibility and must be dropped, never
    # lagged by a guess.
    "broadcast_ts": pl.Datetime(),
    "filing_ts": pl.Datetime(),
    #: First DATE the figure may be used. Derived, not reported by NSE.
    "visible_from": pl.Date(),
    "xbrl_url": pl.Utf8(),
    "source": pl.Utf8(),
}


class ResultsParseError(ValueError):
    """The results payload was not the JSON list this endpoint promises."""


def _parse_ts(raw: Any) -> datetime | None:
    text = str(raw or "").strip()
    if not text or text.lower() in {"none", "null", "-"}:
        return None
    for fmt in _TS_FORMATS:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def _parse_day(raw: Any) -> date | None:
    ts = _parse_ts(raw)
    return ts.date() if ts is not None else None


def visible_from(broadcast: datetime, *, close: time = MARKET_CLOSE) -> date:
    """First date a figure broadcast at ``broadcast`` may be used.

    A filing released before the close is public intraday, but this project
    trades at the NEXT session's open (`panel_env` fills at ``open``), so the
    earliest a same-day release can be acted on is still the following day.
    A filing released after the close is a further day away only if it lands on
    the last session of a week — which the trading calendar, not this function,
    decides. Being deliberately conservative here: one day for an intraday
    release, one day for an after-hours release, and the caller intersects with
    the real trading calendar.

    Returning the *calendar* day after keeps this function pure and testable;
    `attach_visibility` maps it onto trading days.
    """
    return broadcast.date() + timedelta(days=1)


def parse_results(payload: str, *, symbol: str) -> pl.DataFrame:
    """Parse one ``corporates-financial-results`` response into RESULTS_SCHEMA."""
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ResultsParseError(f"{symbol}: results payload is not JSON: {exc}") from exc
    rows = data if isinstance(data, list) else (data.get("data") or [])
    if not isinstance(rows, list):
        raise ResultsParseError(f"{symbol}: expected a list of filings, got {type(rows)}")

    out: dict[str, list[Any]] = {k: [] for k in RESULTS_SCHEMA}
    dropped_no_date = 0
    for r in rows:
        bcast = _parse_ts(r.get("broadCastDate"))
        p_from, p_to = _parse_day(r.get("fromDate")), _parse_day(r.get("toDate"))
        if bcast is None or p_from is None or p_to is None:
            # No broadcast date means no knowable visibility. Dropped, never
            # lagged by a guess — a guessed lag is exactly the lookahead this
            # module exists to prevent.
            dropped_no_date += 1
            continue
        sym = str(r.get("symbol") or symbol).strip()
        out["symbol"].append(sym)
        out["ticker"].append(nse_symbol_to_ticker(sym))
        out["period_from"].append(p_from)
        out["period_to"].append(p_to)
        out["relating_to"].append(str(r.get("relatingTo") or "").strip() or None)
        out["financial_year"].append(str(r.get("financialYear") or "").strip() or None)
        out["audited"].append(str(r.get("audited") or "").strip() or None)
        out["consolidated"].append(str(r.get("consolidated") or "").strip() or None)
        out["cumulative"].append(str(r.get("cumulative") or "").strip() or None)
        out["ind_as"].append(str(r.get("indAs") or "").strip() or None)
        out["broadcast_ts"].append(bcast)
        out["filing_ts"].append(_parse_ts(r.get("filingDate")))
        out["visible_from"].append(visible_from(bcast))
        out["xbrl_url"].append(str(r.get("xbrl") or "").strip() or None)
        out["source"].append("nse_results")

    if dropped_no_date:
        logger.warning(
            f"{symbol}: dropped {dropped_no_date} filing(s) with no usable "
            "broadcast/period date — never lagged by a guess"
        )
    return pl.DataFrame(out, schema=RESULTS_SCHEMA)


def latest_per_period(frame: pl.DataFrame, *, consolidated: bool = True) -> pl.DataFrame:
    """One filing per (ticker, period), the LAST broadcast for it.

    A company files the same quarter more than once: standalone and consolidated
    on the same evening, and revisions later. Taking the last broadcast is what
    an investor would see, and keeping ``visible_from`` from THAT filing means a
    revision only becomes visible on the day it was actually published — an
    earlier row is not retro-corrected.

    Prefers consolidated where a company files both, because that is the basis
    an equity holder is exposed to.
    """
    if frame.is_empty():
        return frame
    pref = (
        pl.when(pl.col("consolidated").str.to_lowercase() == "consolidated")
        .then(0 if consolidated else 1)
        .otherwise(1 if consolidated else 0)
    )
    return (
        frame.with_columns(pref.alias("_pref"))
        .sort(
            ["ticker", "period_to", "_pref", "broadcast_ts"],
            descending=[False, False, False, True],
        )
        .unique(subset=["ticker", "period_to"], keep="first", maintain_order=True)
        .drop("_pref")
        .sort(["ticker", "period_to"])
    )


def attach_visibility(
    frame: pl.DataFrame, trading_days: list[date]
) -> pl.DataFrame:
    """Snap ``visible_from`` forward to the next real trading session.

    NSE broadcasts most results in the evening and many on a Friday, so the
    calendar day after a broadcast is frequently not a session. Without this the
    join would silently attach a figure to a date the market was shut, and the
    panel would carry it a day early on the following Monday.
    """
    if frame.is_empty():
        return frame
    days = sorted(trading_days)
    if not days:
        raise ValueError("trading_days is empty; cannot snap visibility")
    import bisect

    snapped: list[date | None] = []
    for d in frame["visible_from"].to_list():
        i = bisect.bisect_left(days, d)
        snapped.append(days[i] if i < len(days) else None)
    n_past = sum(1 for s in snapped if s is None)
    if n_past:
        logger.warning(
            f"{n_past} filing(s) become visible after the last trading day "
            "supplied; they are null and must not be joined"
        )
    return frame.with_columns(pl.Series("visible_from", snapped, dtype=pl.Date()))


class FinancialResultsSource:
    """Fetch quarterly results per symbol, cached on disk, one file per symbol.

    Cached per SYMBOL rather than per day: NSE serves a company's whole filing
    history in one response, so a symbol is the natural unit and a re-run costs
    nothing. Reuses `nse_flows`'s client for the browser-like session warm-up
    NSE requires — a plain request to this host times out.
    """

    def __init__(
        self,
        *,
        cache_root: str | Path = "data/raw/nse",
        client: Any | None = None,
        offline: bool = False,
        period: str = "Quarterly",
    ) -> None:
        from trader.data.sources.nse_flows import NSEClient, _FileCache

        self.client = client or NSEClient()
        self.offline = offline
        self.period = period
        self.cache = _FileCache(Path(cache_root) / "results")

    def fetch_symbol(self, symbol: str) -> pl.DataFrame:
        name = f"results_{symbol}_{self.period}.json"
        if self.cache.is_bad(name):
            logger.debug(f"{name}: known-bad, skipped ({self.cache.bad_reason(name)})")
            return pl.DataFrame(schema=RESULTS_SCHEMA)
        text = self.cache.read(name)
        if text is None:
            if self.offline:
                return pl.DataFrame(schema=RESULTS_SCHEMA)
            url = RESULTS_URL.format(symbol=symbol, period=self.period)
            text = self.client.get_text(url)
            self.cache.write(name, text, url=url)
        try:
            return parse_results(text, symbol=symbol)
        except ResultsParseError as exc:
            # Mark and skip, exactly as the bhavdata backfill learned to: one
            # malformed symbol must not abort a 504-symbol run, and a corrupt
            # body must not be replayed from cache forever.
            self.cache.mark_bad(name, str(exc))
            logger.warning(f"{symbol}: skipped and marked bad, {exc}")
            return pl.DataFrame(schema=RESULTS_SCHEMA)

    def fetch(self, symbols: list[str]) -> pl.DataFrame:
        frames, skipped = [], 0
        for sym in symbols:
            frame = self.fetch_symbol(sym)
            if frame.is_empty():
                skipped += 1
                continue
            frames.append(frame)
        if skipped:
            logger.warning(f"{skipped}/{len(symbols)} symbol(s) returned no filings")
        if not frames:
            return pl.DataFrame(schema=RESULTS_SCHEMA)
        return pl.concat(frames).sort(["ticker", "period_to"])


# ── XBRL: the figures themselves ─────────────────────────────────────────────

_XBRLI: Final[str] = "{http://www.xbrl.org/2003/instance}"

#: Headline figures worth extracting, mapped to the column they become. All are
#: reported in rupees except the per-share and ratio items.
XBRL_FIELDS: Final[dict[str, str]] = {
    "RevenueFromOperations": "revenue",
    "OtherIncome": "other_income",
    "Income": "total_income",
    "Expenses": "total_expenses",
    "EmployeeBenefitExpense": "employee_cost",
    "FinanceCosts": "finance_costs",
    "DepreciationDepletionAndAmortisationExpense": "depreciation",
    "OtherExpenses": "other_expenses",
    "ProfitBeforeExceptionalItemsAndTax": "pbt_before_exceptional",
    "ExceptionalItemsBeforeTax": "exceptional_items",
    "ProfitBeforeTax": "pbt",
    "TaxExpense": "tax_expense",
    "ProfitLossForPeriod": "net_profit",
    "ProfitOrLossAttributableToOwnersOfParent": "net_profit_owners",
    "ComprehensiveIncomeForThePeriod": "comprehensive_income",
    "BasicEarningsLossPerShareFromContinuingAndDiscontinuedOperations": "eps_basic",
    "DilutedEarningsLossPerShareFromContinuingAndDiscontinuedOperations": "eps_diluted",
    "PaidUpValueOfEquityShareCapital": "paid_up_equity",
    "FaceValueOfEquityShareCapital": "face_value",
}

XBRL_SCHEMA: Final[dict[str, pl.DataType]] = {
    "symbol": pl.Utf8(),
    "ticker": pl.Utf8(),
    "period_from": pl.Date(),
    "period_to": pl.Date(),
    "consolidated": pl.Utf8(),
    "audited": pl.Utf8(),
    "reporting_quarter": pl.Utf8(),
    "board_meeting_date": pl.Date(),
    **{c: pl.Float64() for c in XBRL_FIELDS.values()},
}


class XBRLParseError(ValueError):
    """The XBRL document could not be read as a quarterly results filing."""


def _num(text: str | None) -> float | None:
    if text is None:
        return None
    t = text.strip().replace(",", "")
    if not t or t in {"-", "NA", "N/A"}:
        return None
    try:
        return float(t)
    except ValueError:
        return None


def parse_results_xbrl(xml_text: str, *, name: str = "<xbrl>") -> pl.DataFrame:
    """Extract headline figures from one NSE results XBRL document.

    **The period comes from the FACTS, never from the context header.** Verified
    on INFY Q3 FY25: contexts ``OneD`` and ``FourD`` both DECLARE
    2024-10-01..2024-12-31, but ``FourD``'s own
    ``DateOfStartOfReportingPeriod`` fact says 2024-04-01 — it is the nine-month
    cumulative wearing the quarter's header. Its revenue is Rs 1,220.6bn against
    the quarter's Rs 417.6bn, so trusting the header would treble the figure and
    do it silently.

    Only undimensioned contexts are considered; dimensioned ones are segment and
    other-comprehensive-income breakdowns, not the headline statement.

    Returns one row per (context) reporting period found, so a filing that
    carries both the quarter and the year-to-date yields both, correctly
    labelled, and the caller picks.
    """
    import xml.etree.ElementTree as ET

    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        raise XBRLParseError(f"{name}: not well-formed XML: {exc}") from exc

    dimensioned: set[str] = set()
    for el in root:
        if el.tag.endswith("}context"):
            cid = el.get("id") or ""
            if any(m.tag.endswith("explicitMember") or m.tag.endswith("typedMember")
                   for m in el.iter()):
                dimensioned.add(cid)

    by_ctx: dict[str, dict[str, str]] = {}
    for el in root:
        ctx = el.get("contextRef")
        if not ctx or ctx in dimensioned or not (el.text and el.text.strip()):
            continue
        by_ctx.setdefault(ctx, {})[el.tag.split("}")[-1]] = el.text.strip()

    # Symbol and company name are FILING-level metadata: NSE stamps them on one
    # context, not on every one. Requiring them per context silently dropped the
    # year-to-date row here, and would drop the QUARTER on any filing that
    # happens to carry them on the cumulative context instead.
    doc_symbol = next(
        (f["Symbol"] for f in by_ctx.values() if f.get("Symbol")), None
    )
    if not doc_symbol:
        raise XBRLParseError(f"{name}: no Symbol fact anywhere in the document")

    rows: list[dict[str, Any]] = []
    for _ctx, facts in by_ctx.items():
        start = _parse_iso(facts.get("DateOfStartOfReportingPeriod"))
        end = _parse_iso(facts.get("DateOfEndOfReportingPeriod"))
        sym = facts.get("Symbol") or doc_symbol
        if start is None or end is None:
            continue          # not a reporting-period context
        row: dict[str, Any] = {
            "symbol": sym,
            "ticker": nse_symbol_to_ticker(sym),
            "period_from": start,
            "period_to": end,
            "consolidated": facts.get("NatureOfReportStandaloneConsolidated"),
            "audited": facts.get("WhetherResultsAreAuditedOrUnaudited"),
            "reporting_quarter": facts.get("ReportingQuarter"),
            "board_meeting_date": _parse_iso(
                facts.get("DateOfBoardMeetingWhenFinancialResultsWereApproved")
            ),
        }
        for tag, col in XBRL_FIELDS.items():
            row[col] = _num(facts.get(tag))
        rows.append(row)

    if not rows:
        raise XBRLParseError(
            f"{name}: no undimensioned context carried a reporting period; "
            "this is not a quarterly results filing"
        )
    return pl.DataFrame(rows, schema=XBRL_SCHEMA).sort(["period_from", "period_to"])


def _parse_iso(raw: str | None) -> date | None:
    if not raw:
        return None
    try:
        return date.fromisoformat(raw.strip()[:10])
    except ValueError:
        return None


def quarterly_only(frame: pl.DataFrame, *, max_days: int = 100) -> pl.DataFrame:
    """Keep the single-quarter rows, dropping year-to-date cumulatives.

    A quarter spans ~90 days; a nine-month cumulative spans ~275. Filtering on
    the span computed from the FACT dates is what separates them, because their
    context headers do not.
    """
    if frame.is_empty():
        return frame
    span = (pl.col("period_to") - pl.col("period_from")).dt.total_days()
    return frame.filter(span <= max_days)


@dataclass(frozen=True)
class ResultsFetchPlan:
    """What a fetch would do, so a caller can size it before running it."""

    symbols: tuple[str, ...]
    period: str

    @property
    def n_requests(self) -> int:
        return len(self.symbols)

    def describe(self) -> str:
        return (
            f"{self.n_requests} symbol(s), period={self.period}; at ~1 req/s "
            f"that is ~{self.n_requests / 60:.1f} min"
        )
