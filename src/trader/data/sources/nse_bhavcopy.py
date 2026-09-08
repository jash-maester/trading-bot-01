"""Full-market daily bars from NSE's own bhavcopy archives, with ISIN identity.

Why this exists
---------------
`CLAUDE.md` lists survivorship under *Known-dangerous ground* and
`audit/P4_survivorship.md` called delisting an unrecoverable limit. Measured
2026-09-08 against `data/ext/delivery.parquet`, the cost of that limit is
larger than the note implies: of the 605 names carrying at least ₹5 crore of
median daily turnover in 2021, **our 504-name universe contains 292 — 48.3%** —
and 55 of those 605 had stopped trading by 2026. Every headline number in this
repository is measured on a list chosen in 2026.

It is recoverable after all, because NSE archives the whole market daily.

Two layouts, and the boundary matters
-------------------------------------
**Classic** (``cm<DDMMMYYYY>bhav.csv.zip``), verified 2026-09-08 to serve
2010-01-04 … 2024-06-03 and 404 from 2024-08-01::

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

#: Last date the classic archive was verified to serve. Not a hard cutoff — the
#: fetcher tries classic first and falls back — but it stops the backfill from
#: making a doomed request for every session after the changeover.
CLASSIC_VERIFIED_TO: Final[date] = date(2024, 6, 3)

#: First date the classic archive was verified to serve.
CLASSIC_VERIFIED_FROM: Final[date] = date(2010, 1, 4)

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


def adjustment_ratios(
    bars: pl.DataFrame, *, tolerance: float = 0.005, max_gap_days: int = 7
) -> pl.DataFrame:
    """Detect split/bonus adjustments from NSE's own ``prev_close``.

    NSE publishes ``prev_close`` **already adjusted** for anything that happened
    overnight. So for each ticker, on each session, ``prev_close / close.shift(1)``
    is 1.0 on an ordinary day and the adjustment factor on the day a split,
    bonus or consolidation took effect. That is a corporate-action feed for
    free, stated by the exchange, on the day it applied.

    This matters more than it sounds. `CLAUDE.md` records that ``auto_adjust``
    handles splits and dividends but **not demergers**, which showed up as fake
    catastrophic losses (NIITLTD −76.13%, MASTEK −66.00%). A demerger moves value
    to a separate listed entity and NSE reflects it in ``prev_close`` exactly as
    it does a split, so this detector catches the case that broke the old path.

    ``tolerance`` is the band around 1.0 treated as an ordinary day; 0.5% is
    wide enough to absorb rounding in NSE's published prices and far narrower
    than any real action.

    ``max_gap_days`` is the load-bearing one. The ratio compares NSE's stated
    previous close against **our previous row**, and those are the same session
    only when the history is contiguous. Across a gap — a suspension, a missing
    archive day, a name filtered out and back — the comparison spans however
    long the gap was and reports ordinary price movement as a corporate action.
    Measured on a deliberately gappy sample: 2,070 of 2,072 detections landed on
    the single date spanning a one-year hole, with ratios from 0.46 to 1.86,
    against 2 detections on every other date combined. Feeding those into
    :func:`back_adjust` would compound a year of returns into the adjustment
    factor and silently rescale every price before the gap. Seven days covers a
    weekend plus a holiday run; anything longer is not a next session.

    Returns the rows where an adjustment fired, with the ratio.
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


def back_adjust(
    bars: pl.DataFrame, *, tolerance: float = 0.005, max_gap_days: int = 7
) -> pl.DataFrame:
    """Return ``bars`` with prices back-adjusted onto the latest scale.

    Prices before an action are multiplied by the cumulative product of every
    ratio at or after them, so the series is continuous and comparable with
    Kite's ``auto_adjust=True`` bars. Volume is divided by the same factor, so
    price × volume stays equal to turnover.

    The adjustment is applied on the LATEST scale — the most recent bar keeps
    its published price — because that is the convention Kite uses and mixing
    conventions is exactly the failure this module exists to avoid.

    ``max_gap_days`` is passed through to the same guard
    :func:`adjustment_ratios` documents: a ratio measured across a hole in a
    ticker's history is ordinary price movement, not an action, and compounding
    one into the factor would rescale everything before it.
    """
    need = {"date", "ticker", "close", "prev_close"}
    missing = need - set(bars.columns)
    if missing:
        raise ValueError(f"bars is missing {sorted(missing)}")

    df = bars.sort(["ticker", "date"])
    prior = pl.col("close").shift(1).over("ticker")
    ratio = pl.col("prev_close") / prior
    gap = (pl.col("date") - pl.col("date").shift(1).over("ticker")).dt.total_days()
    clean = (
        pl.when(
            prior.is_null()
            | ratio.is_null()
            | ((ratio - 1.0).abs() <= tolerance)
            | (gap > max_gap_days)
        )
        .then(1.0)
        .otherwise(ratio)
        .alias("_r")
    )
    df = df.with_columns(clean)
    # Cumulative product of every ratio STRICTLY AFTER each row, per ticker.
    # Reversing, taking a cumulative product, then reversing puts the factor for
    # row i equal to the product over rows > i.
    df = df.with_columns(
        pl.col("_r")
        .shift(-1)
        .fill_null(1.0)
        .reverse()
        .cum_prod()
        .reverse()
        .over("ticker")
        .alias("_factor")
    )
    price_cols = [c for c in ("open", "high", "low", "close", "last", "prev_close")
                  if c in df.columns]
    out = df.with_columns(
        [(pl.col(c) * pl.col("_factor")).alias(c) for c in price_cols]
        + ([(pl.col("volume") / pl.col("_factor")).alias("volume")]
           if "volume" in df.columns else [])
    )
    return out.drop(["_r", "_factor"])


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
