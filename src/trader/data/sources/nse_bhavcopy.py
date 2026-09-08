"""Full-market daily bars from NSE's own bhavcopy archives, with ISIN identity.

Why this exists
---------------
`CLAUDE.md` lists survivorship under *Known-dangerous ground* and
`audit/P4_survivorship.md` called delisting an unrecoverable limit. Measured
2026-09-08 by `scripts/pit_universe_report.py`, the cost is real: of the names
clearing ₹5 crore of median daily turnover, the **645 panelled names cover only
60.0%** on average over 2011–2020, and at 2016-01-01 twenty-seven of the
ninety-nine missing had stopped trading altogether. Every headline number in
this repository is measured on a list drawn in 2026.

It is recoverable after all, because NSE archives the whole market daily.

Two layouts, and the boundary matters
-------------------------------------
**Classic** (``cm<DDMMMYYYY>bhav.csv.zip``), verified 2026-09-08 to serve
2010-01-04 … 2024-07-01, with 404 from 2025-01-01::

    SYMBOL,SERIES,OPEN,HIGH,LOW,CLOSE,LAST,PREVCLOSE,TOTTRDQTY,TOTTRDVAL,
    TIMESTAMP,TOTALTRADES,ISIN

**Security-wise** (``sec_bhavdata_full_<DDMMYYYY>.csv``), which bisects to a
start between 2019-09-27 and 2019-09-30::

    SYMBOL, SERIES, DATE1, PREV_CLOSE, OPEN_PRICE, HIGH_PRICE, LOW_PRICE,
    LAST_PRICE, CLOSE_PRICE, AVG_PRICE, TTL_TRD_QNTY, TURNOVER_LACS,
    NO_OF_TRADES, DELIV_QTY, DELIV_PER

The classic layout is preferred wherever it exists, for one reason: **it carries
ISIN and the newer one does not.** ISIN is the identity that survives a rename,
and renames are their own entry in `CLAUDE.md` — `AMARAJABAT` and `ARE&M` are
one company, as are `CADILAHC` and `ZYDUSLIFE`, and a symbol-keyed join silently
drops the history on both sides of the change. The classic archive spans the
entire 2016-07 … 2024-06 walk-forward, so the windows that matter get ISIN; the
2025+ holdout falls back to the newer layout and inherits the symbol→ISIN map
built over the 2019-09 … 2024-06 overlap.

TWO THINGS THIS DATA IS NOT
---------------------------
**It is not adjusted.** Kite's bars come back with ``auto_adjust=True``; these do
not. Mixing the two silently would put a raw price series next to an adjusted
one. ``PREVCLOSE`` is the repair: NSE states the previous close *as adjusted for
anything that happened overnight*, so comparing it to the prior session's
``CLOSE`` recovers the split/bonus ratio for every name, on the day it happened,
from the file itself. See :func:`adjustment_ratios`.

**It is not a universe.** It is every security that traded, including illiquid
and non-equity series. ``CASH_SERIES`` filtering and a liquidity bar are the
caller's job — `trader.data.listing` owns eligibility.
"""
from __future__ import annotations

import csv
import io
import zipfile
from datetime import date, datetime
from typing import Final

import polars as pl
from loguru import logger

from trader.data.sources.nse_flows import NSE_ARCHIVES, nse_symbol_to_ticker

#: Classic archive: zipped CSV, OHLC + PREVCLOSE + ISIN. Verified reachable for
#: 2010-01-04 through 2024-06-03 on 2026-09-08; 404 from 2024-08-01.
CLASSIC_BHAV_URL: Final[str] = (
    NSE_ARCHIVES + "/content/historical/EQUITIES/{year}/{mon}/cm{stamp}bhav.csv.zip"
)

#: Last date the classic archive was verified to serve, re-measured 2026-09-08:
#: 2024-07-01 returns a normal file and 2025-01-01 returns 404, so the changeover
#: sits between them. INFORMATIONAL ONLY — the fetcher tries classic on every
#: session and falls back on a 404 rather than trusting this date, because the
#: first value recorded here (2024-06-03) was already a month early and a hard
#: cutoff would have silently dropped ISIN for every session after it.
CLASSIC_VERIFIED_TO: Final[date] = date(2024, 7, 1)

#: First date the classic archive was verified to serve. Re-measured 2026-09-08:
#: 2005-01-03 returns a normal file (816 rows, 814 EQ/BE), so the archive is at
#: least five years deeper than the 2010-01-04 first recorded here.
#:
#: INFORMATIONAL ONLY, and that matters. Used as a hard floor, the earlier value
#: silently blocked a span extension: `fetch_bhavcopy.py` gated the classic
#: attempt on `day >= CLASSIC_VERIFIED_FROM`, so 1,239 sessions from 2005-2010
#: never tried the archive at all, fell through to a layout that starts in 2019,
#: and were logged as "no data for this date". The run reported success having
#: added nothing.
CLASSIC_VERIFIED_FROM: Final[date] = date(2005, 1, 3)

BHAVCOPY_SCHEMA: Final[dict[str, pl.DataType]] = {
    "date": pl.Date(),
    "ticker": pl.Utf8(),
    "symbol": pl.Utf8(),
    # Null for rows sourced from the newer layout, which does not publish it.
    "isin": pl.Utf8(),
    "series": pl.Utf8(),
    "open": pl.Float64(),
    "high": pl.Float64(),
    "low": pl.Float64(),
    "close": pl.Float64(),
    "last": pl.Float64(),
    # NSE's own previous close, ALREADY adjusted for overnight corporate
    # actions. The whole point of keeping it: see `adjustment_ratios`.
    "prev_close": pl.Float64(),
    "volume": pl.Int64(),
    "turnover": pl.Float64(),
    "n_trades": pl.Int64(),
    "source": pl.Utf8(),
}

_LACS_TO_RUPEES: Final[float] = 1e5


class BhavcopyParseError(ValueError):
    """The payload is not a bhavcopy this module can read."""


def _f(v: str | None) -> float | None:
    s = (v or "").strip()
    if not s or s == "-":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _i(v: str | None) -> int | None:
    f = _f(v)
    return int(f) if f is not None else None


def classic_url(day: date) -> str:
    return CLASSIC_BHAV_URL.format(
        year=day.year,
        mon=day.strftime("%b").upper(),
        stamp=day.strftime("%d%b%Y").upper(),
    )


def unzip_bhavcopy(payload: bytes) -> str:
    """Return the single CSV inside a classic bhavcopy zip."""
    if payload[:2] != b"PK":
        raise BhavcopyParseError(
            f"not a zip: first bytes {payload[:8]!r} (NSE serves an HTML error "
            "page with HTTP 200 for some missing dates)"
        )
    try:
        z = zipfile.ZipFile(io.BytesIO(payload))
        names = z.namelist()
        if not names:
            raise BhavcopyParseError("zip is empty")
        return z.read(names[0]).decode("utf-8", errors="replace")
    except zipfile.BadZipFile as exc:
        raise BhavcopyParseError(f"corrupt zip: {exc}") from exc


#: TIMESTAMP formats seen in the classic archive. NSE switched to a two-digit
#: year around mid-2020 ("13-Jul-20"), which killed a 4,138-day backfill at
#: session 2,600. Tried in order; the four-digit form is the common one.
_TIMESTAMP_FORMATS: Final[tuple[str, ...]] = ("%d-%b-%Y", "%d-%b-%y")


def _parse_timestamp(text: str, *, name: str) -> date:
    """Parse a classic bhavcopy TIMESTAMP, whichever year form it uses."""
    for fmt in _TIMESTAMP_FORMATS:
        try:
            d = datetime.strptime(text, fmt).date()
        except ValueError:
            continue
        # `%y` maps 00-68 to 2000-2068 and 69-99 to 1969-1999. A bhavcopy from
        # 1970 does not exist, so a date outside the plausible archive range
        # means the format guess was wrong rather than the data being odd.
        if 2000 <= d.year <= 2100:
            return d
    raise BhavcopyParseError(
        f"{name}: TIMESTAMP {text!r} matches none of {list(_TIMESTAMP_FORMATS)}"
    )


def parse_classic_bhavcopy(text: str, *, name: str = "<classic>") -> pl.DataFrame:
    """Parse a classic ``cm*bhav.csv`` body into :data:`BHAVCOPY_SCHEMA`.

    **The layout changed inside the classic era, and ISIN is the part that
    moved.** Files from 2010-2011 carry::

        SYMBOL,SERIES,OPEN,HIGH,LOW,CLOSE,LAST,PREVCLOSE,TOTTRDQTY,TOTTRDVAL,
        TIMESTAMP,

    with no ``TOTALTRADES`` and no ``ISIN`` at all; both were added later.
    Requiring ISIN rejected every pre-ISIN session — 369 of the first 500 in the
    2010 backfill — and, because the rejection path marks the body ``.bad`` and
    deletes it, each one then fell through to a newer archive that does not
    reach back that far and was recorded as "no data for this date".

    So ISIN is OPTIONAL here and null where the file predates it. That is a real
    limitation, not a cosmetic one: rename resolution needs ISIN, and it is
    unavailable for the earliest years. It does not bite for this project's
    purpose — the walk-forward starts 2016-07 and only pre-2016 *feature
    history* comes from those years — but a caller doing identity work before
    ~2011 must know the column is empty rather than assume a join failed.
    """
    head = text.lstrip()[:200]
    if "SYMBOL" not in head:
        raise BhavcopyParseError(
            f"{name}: no SYMBOL header in the first 200 chars; got {head[:80]!r}"
        )
    rows = [
        {(k or "").strip(): (v or "").strip() for k, v in r.items()}
        for r in csv.DictReader(io.StringIO(text))
    ]
    out: dict[str, list[object]] = {c: [] for c in BHAVCOPY_SCHEMA}
    for r in rows:
        ts = r.get("TIMESTAMP", "")
        if not ts or not r.get("SYMBOL"):
            continue
        out["date"].append(_parse_timestamp(ts, name=name))
        out["symbol"].append(r["SYMBOL"])
        out["ticker"].append(nse_symbol_to_ticker(r["SYMBOL"]))
        out["isin"].append(r.get("ISIN") or None)
        out["series"].append(r.get("SERIES", ""))
        out["open"].append(_f(r.get("OPEN")))
        out["high"].append(_f(r.get("HIGH")))
        out["low"].append(_f(r.get("LOW")))
        out["close"].append(_f(r.get("CLOSE")))
        out["last"].append(_f(r.get("LAST")))
        out["prev_close"].append(_f(r.get("PREVCLOSE")))
        out["volume"].append(_i(r.get("TOTTRDQTY")))
        # TOTTRDVAL is already in rupees: 20MICRONS on 2016-01-04 shows
        # 258,669 shares and 9,428,181.1, against an average price near 36.
        out["turnover"].append(_f(r.get("TOTTRDVAL")))
        out["n_trades"].append(_i(r.get("TOTALTRADES")))
        out["source"].append("classic")
    if not out["date"]:
        raise BhavcopyParseError(f"{name}: parsed zero rows")
    return pl.DataFrame(out, schema=BHAVCOPY_SCHEMA)


def parse_sec_bhavdata_ohlcv(text: str, *, name: str = "<sec>") -> pl.DataFrame:
    """Parse a ``sec_bhavdata_full_*.csv`` body into :data:`BHAVCOPY_SCHEMA`.

    The same file `nse_flows.parse_sec_bhavdata` reads for delivery, taken for
    its price columns instead. Kept separate rather than widening that parser,
    so the delivery path and its tests are untouched.

    ``isin`` is null throughout: this layout does not publish it.
    """
    head = text.lstrip()[:400]
    if text.startswith("PK\x03\x04") or "[Content_Types].xml" in head:
        raise BhavcopyParseError(f"{name}: ZIP/XLSX body served at the .csv URL")
    if "SYMBOL" not in head:
        raise BhavcopyParseError(f"{name}: no SYMBOL header; got {head[:80]!r}")
    rows = [
        {(k or "").strip(): (v or "").strip() for k, v in r.items()}
        for r in csv.DictReader(io.StringIO(text))
    ]
    out: dict[str, list[object]] = {c: [] for c in BHAVCOPY_SCHEMA}
    for r in rows:
        d1 = r.get("DATE1", "")
        if not d1 or not r.get("SYMBOL"):
            continue
        out["date"].append(_parse_timestamp(d1, name=name))
        out["symbol"].append(r["SYMBOL"])
        out["ticker"].append(nse_symbol_to_ticker(r["SYMBOL"]))
        out["isin"].append(None)
        out["series"].append(r.get("SERIES", ""))
        out["open"].append(_f(r.get("OPEN_PRICE")))
        out["high"].append(_f(r.get("HIGH_PRICE")))
        out["low"].append(_f(r.get("LOW_PRICE")))
        out["close"].append(_f(r.get("CLOSE_PRICE")))
        out["last"].append(_f(r.get("LAST_PRICE")))
        out["prev_close"].append(_f(r.get("PREV_CLOSE")))
        out["volume"].append(_i(r.get("TTL_TRD_QNTY")))
        lacs = _f(r.get("TURNOVER_LACS"))
        out["turnover"].append(lacs * _LACS_TO_RUPEES if lacs is not None else None)
        out["n_trades"].append(_i(r.get("NO_OF_TRADES")))
        out["source"].append("sec")
    if not out["date"]:
        raise BhavcopyParseError(f"{name}: parsed zero rows")
    return pl.DataFrame(out, schema=BHAVCOPY_SCHEMA)


def price_discontinuities(
    bars: pl.DataFrame, *, tolerance: float = 0.005, max_gap_days: int = 7
) -> pl.DataFrame:
    """Rows where ``prev_close`` disagrees with the previous session's close.

    **This is a data-integrity check, NOT a corporate-action detector.** It was
    written as one, on the belief that NSE publishes ``prev_close`` already
    adjusted for overnight actions. That belief was wrong — ``prev_close`` is
    the RAW previous close, so on an ex-date it carries the pre-split level
    unchanged and this ratio stays 1.0. Verified against NESTLEIND 2024-01-05,
    HDFCBANK 2019-09-19 and IRCTC 2021-10-28, all of which show a 50-90% price
    step and a ratio of exactly 1.0. Real adjustments come from
    :mod:`trader.data.sources.nse_corporate_actions`.

    What it still catches, and what it is kept for: a break in the chain. NSE's
    ``prev_close`` refers to the security's previous SESSION, so a departure
    from our previous ROW means our history is missing something — a suspension,
    an archive gap, or an EQ/BE session filtered out. Over 23,693 contiguous
    ticker-days the two agree in 23,693 of them, so a disagreement is a genuine
    signal about the data rather than about the company.

    ``max_gap_days`` suppresses rows whose previous entry is further back than a
    weekend plus a holiday run, where the comparison spans a hole by definition.
    """
    need = {"date", "ticker", "close", "prev_close"}
    missing = need - set(bars.columns)
    if missing:
        raise ValueError(f"bars is missing {sorted(missing)}")
    df = bars.sort(["ticker", "date"]).with_columns(
        (pl.col("prev_close") / pl.col("close").shift(1).over("ticker")).alias("ratio"),
        (pl.col("date") - pl.col("date").shift(1).over("ticker"))
        .dt.total_days()
        .alias("_gap"),
    )
    hits = df.filter(
        pl.col("ratio").is_not_null()
        & ((pl.col("ratio") - 1.0).abs() > tolerance)
        & pl.col("close").shift(1).over("ticker").is_not_null()
        & (pl.col("_gap") <= max_gap_days)
    )
    return hits.select(["date", "ticker", "close", "prev_close", "ratio"])


def isin_map(bars: pl.DataFrame) -> pl.DataFrame:
    """Symbol → ISIN, from rows that carry one.

    The identity that survives a rename. `AMARAJABAT` and `ARE&M` are one
    company; a symbol-keyed join drops the history on both sides of the change
    and `CLAUDE.md` records that as already having cost this project data.

    Only the classic layout publishes ISIN, so this is built over
    2010 … 2024-06 and carried forward onto the newer rows by symbol.
    """
    if "isin" not in bars.columns:
        raise ValueError("bars carries no isin column")
    m = (
        bars.filter(pl.col("isin").is_not_null() & (pl.col("isin") != ""))
        .group_by(["symbol", "isin"])
        .agg(
            pl.col("date").min().alias("first_seen"),
            pl.col("date").max().alias("last_seen"),
            pl.len().alias("sessions"),
        )
        .sort(["symbol", "first_seen"])
    )
    dupes = (
        m.group_by("isin").agg(pl.col("symbol").n_unique().alias("n"))
        .filter(pl.col("n") > 1)
    )
    if dupes.height:
        logger.info(
            f"{dupes.height} ISIN(s) carry more than one symbol — these are the "
            "renames a symbol-keyed join would have dropped"
        )
    return m
