"""Tests for the full-market bhavcopy source and its corporate-action repair.

The interesting cases here are all silent ones. An unadjusted price series sat
next to an adjusted one looks fine and prices a split as a 50% crash; a
symbol-keyed join across a rename looks fine and simply loses half a company's
history; and a back-adjustment applied on the wrong side of the action looks
fine and shifts every historical price by a constant factor.
"""
from __future__ import annotations

import io
import zipfile
from datetime import date

import polars as pl
import pytest

from trader.data.sources.nse_bhavcopy import (
    BHAVCOPY_SCHEMA,
    BhavcopyParseError,
    classic_url,
    isin_map,
    parse_classic_bhavcopy,
    parse_sec_bhavdata_ohlcv,
    price_discontinuities,
    unzip_bhavcopy,
)

_CLASSIC = (
    "SYMBOL,SERIES,OPEN,HIGH,LOW,CLOSE,LAST,PREVCLOSE,TOTTRDQTY,TOTTRDVAL,"
    "TIMESTAMP,TOTALTRADES,ISIN,\n"
    "20MICRONS,EQ,35.55,37.5,34.75,35.9,35.95,35.2,258669,9428181.1,"
    "04-JAN-2016,823,INE144J01027,\n"
    "3IINFOTECH,EQ,5.2,5.45,5.05,5.2,5.2,5.15,3927327,20506575.05,"
    "04-JAN-2016,2440,INE748C01020,\n"
)

_SEC = (
    "SYMBOL, SERIES, DATE1, PREV_CLOSE, OPEN_PRICE, HIGH_PRICE, LOW_PRICE, "
    "LAST_PRICE, CLOSE_PRICE, AVG_PRICE, TTL_TRD_QNTY, TURNOVER_LACS, "
    "NO_OF_TRADES, DELIV_QTY, DELIV_PER\n"
    "20MICRONS, EQ, 02-Jan-2024, 176.50, 176.50, 177.70, 168.50, 171.50, "
    "171.00, 171.50, 321759, 551.80, 12071, 153090, 47.58\n"
)


def test_classic_layout_parses_with_isin() -> None:
    df = parse_classic_bhavcopy(_CLASSIC)
    assert list(df.columns) == list(BHAVCOPY_SCHEMA)
    r = df.row(0, named=True)
    assert r["date"] == date(2016, 1, 4)
    assert r["ticker"] == "20MICRONS.NS"
    assert r["isin"] == "INE144J01027"
    assert r["close"] == pytest.approx(35.9)
    assert r["prev_close"] == pytest.approx(35.2)
    assert r["volume"] == 258669
    assert r["source"] == "classic"


def test_classic_turnover_is_already_rupees() -> None:
    """TOTTRDVAL is rupees, not lakhs.

    258,669 shares near ₹36 is ~₹9.4 million, and the file says 9,428,181.1. A
    lakhs interpretation would inflate every turnover by 100,000 and put a
    ₹5 crore liquidity bar in the wrong place entirely.
    """
    r = parse_classic_bhavcopy(_CLASSIC).row(0, named=True)
    assert r["turnover"] == pytest.approx(9_428_181.1)
    implied = r["turnover"] / r["volume"]
    assert r["low"] <= implied <= r["high"], "implied VWAP outside the day's range"


def test_sec_layout_parses_and_converts_lakhs() -> None:
    r = parse_sec_bhavdata_ohlcv(_SEC).row(0, named=True)
    assert r["date"] == date(2024, 1, 2)
    assert r["ticker"] == "20MICRONS.NS"
    assert r["isin"] is None, "this layout does not publish ISIN"
    assert r["close"] == pytest.approx(171.00)
    # 551.80 lakhs = 55,180,000 rupees
    assert r["turnover"] == pytest.approx(55_180_000.0)
    implied = r["turnover"] / r["volume"]
    assert r["low"] <= implied <= r["high"]
    assert r["source"] == "sec"


def test_the_two_layouts_agree_on_schema() -> None:
    """Rows from either era must stack without a cast."""
    a, b = parse_classic_bhavcopy(_CLASSIC), parse_sec_bhavdata_ohlcv(_SEC)
    assert a.schema == b.schema
    assert pl.concat([a, b]).height == a.height + b.height


def test_a_non_bhavcopy_body_raises() -> None:
    with pytest.raises(BhavcopyParseError, match="SYMBOL"):
        parse_classic_bhavcopy("<html>404 not found</html>")
    with pytest.raises(BhavcopyParseError, match="SYMBOL"):
        parse_sec_bhavdata_ohlcv("<html>nope</html>")


def test_a_zip_served_at_the_csv_url_is_rejected() -> None:
    """The bhavdata backfill already lost 1,613 days to exactly this."""
    with pytest.raises(BhavcopyParseError, match="ZIP/XLSX"):
        parse_sec_bhavdata_ohlcv("PK\x03\x04 binary rubbish")


def test_unzip_rejects_an_html_error_page() -> None:
    """NSE serves an HTML error page with HTTP 200 for some missing dates."""
    with pytest.raises(BhavcopyParseError, match="not a zip"):
        unzip_bhavcopy(b"<html><body>File not found</body></html>")


def test_unzip_reads_the_member() -> None:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("cm04JAN2016bhav.csv", _CLASSIC)
    assert "20MICRONS" in unzip_bhavcopy(buf.getvalue())


def test_classic_url_shape() -> None:
    u = classic_url(date(2016, 1, 4))
    assert u.endswith("/2016/JAN/cm04JAN2016bhav.csv.zip")


# ── data continuity (NOT corporate actions — see the module docstring) ───────


def _gapped_frame() -> pl.DataFrame:
    """One ticker with a year-long hole and no corporate action at all."""
    return pl.DataFrame({
        "date": [date(2024, 6, 3), date(2024, 6, 4), date(2025, 6, 2), date(2025, 6, 3)],
        "ticker": ["G.NS"] * 4,
        "close": [100.0, 101.0, 160.0, 161.0],
        "prev_close": [99.0, 100.0, 159.0, 160.0],
        "volume": [100, 100, 100, 100],
    })


def test_a_gap_in_history_is_not_reported_as_a_discontinuity() -> None:
    """Ordinary price movement across a hole must not read as a break.

    On a deliberately gappy sample, 2,070 of 2,072 detections landed on the one
    date spanning a one-year hole, ratios 0.46 to 1.86.
    """
    assert price_discontinuities(_gapped_frame()).is_empty()


def test_a_single_skipped_session_IS_reported() -> None:
    """The case the gap guard cannot see, and the reason the EQ/BE dedupe matters.

    Real case, AMBICAAGAR 2024-05: EQ, then BE, then EQ. Keeping only the EQ
    rows skips one BE session, leaving a two-day hole that is well inside
    `max_gap_days` — so only the dedupe stands between that and a stale
    comparison. Verified on the real bars: EQ+BE deduped, all 23,693 ratios in
    that window are exactly 1.0.
    """
    rows = [
        (date(2024, 5, 20), "EQ", 100.0, 99.0),
        (date(2024, 5, 21), "BE", 90.0, 100.0),
        (date(2024, 5, 22), "EQ", 91.0, 90.0),
    ]
    full = pl.DataFrame({
        "date": [r[0] for r in rows],
        "ticker": ["X.NS"] * len(rows),
        "series": [r[1] for r in rows],
        "close": [r[2] for r in rows],
        "prev_close": [r[3] for r in rows],
    })
    assert price_discontinuities(full).is_empty(), "the contiguous chain fired"

    eq_only = full.filter(pl.col("series") == "EQ")
    hit = price_discontinuities(eq_only)
    assert hit.height == 1
    assert hit.row(0, named=True)["ratio"] == pytest.approx(0.9)


def test_a_known_split_produces_NO_discontinuity() -> None:
    """The measurement that exposed the original error.

    NESTLEIND fell from 27,116.40 to 2,666.40 on its 1:10 ex-date, and
    `prev_close` on that row is 27,116.40 — the pre-split level, carried
    forward untouched. So the ratio is exactly 1.0 and this function is blind
    to it BY DESIGN, which is why adjustment reads NSE's corporate-actions feed
    instead.
    """
    df = pl.DataFrame({
        "date": [date(2024, 1, 4), date(2024, 1, 5)],
        "ticker": ["NESTLEIND.NS"] * 2,
        "close": [27116.40, 2666.40],
        "prev_close": [26635.20, 27116.40],
    })
    assert price_discontinuities(df).is_empty(), (
        "if this ever fires, prev_close semantics have changed and the "
        "corporate-actions path should be re-examined"
    )


# ── identity ─────────────────────────────────────────────────────────────────


def test_isin_map_exposes_a_rename() -> None:
    """One ISIN under two symbols is a rename, and is the point of keeping ISIN."""
    df = pl.DataFrame(
        {
            "date": [date(2022, 1, 3), date(2024, 1, 3), date(2022, 1, 3)],
            "symbol": ["AMARAJABAT", "ARE&M", "TCS"],
            "isin": ["INE885A01032", "INE885A01032", "INE467B01029"],
        }
    )
    m = isin_map(df)
    shared = m.filter(pl.col("isin") == "INE885A01032")
    assert set(shared["symbol"].to_list()) == {"AMARAJABAT", "ARE&M"}
    assert m.filter(pl.col("isin") == "INE467B01029").height == 1


def test_isin_map_needs_the_column() -> None:
    with pytest.raises(ValueError, match="isin"):
        isin_map(pl.DataFrame({"symbol": ["X"], "date": [date(2020, 1, 1)]}))


_CLASSIC_2010 = (
    "SYMBOL,SERIES,OPEN,HIGH,LOW,CLOSE,LAST,PREVCLOSE,TOTTRDQTY,TOTTRDVAL,"
    "TIMESTAMP,\n"
    "20MICRONS,EQ,58.5,60.0,57.1,58.0,58.2,58.9,100000,5800000.0,04-JAN-2010,\n"
)


def test_the_pre_isin_layout_still_parses() -> None:
    """2010-2011 files carry neither ISIN nor TOTALTRADES.

    Requiring ISIN rejected 369 of the first 500 sessions of the 2010 backfill,
    and because the rejection path marks the body `.bad` and deletes it, each
    one then fell through to a newer archive that does not reach back that far
    and was logged as "no data for this date". The header was the whole cause.
    """
    df = parse_classic_bhavcopy(_CLASSIC_2010, name="cm04JAN2010bhav.csv")
    assert df.height == 1
    r = df.row(0, named=True)
    assert r["date"] == date(2010, 1, 4)
    assert r["ticker"] == "20MICRONS.NS"
    assert r["close"] == pytest.approx(58.0)
    assert r["prev_close"] == pytest.approx(58.9)
    assert r["turnover"] == pytest.approx(5_800_000.0)
    assert r["isin"] is None, "this vintage has no ISIN and must not invent one"
    assert r["n_trades"] is None, "this vintage has no TOTALTRADES either"
    assert list(df.columns) == list(BHAVCOPY_SCHEMA)


def test_both_classic_vintages_stack() -> None:
    """A 2010 file and a 2016 file must concatenate without a cast."""
    a = parse_classic_bhavcopy(_CLASSIC_2010)
    b = parse_classic_bhavcopy(_CLASSIC)
    assert a.schema == b.schema
    both = pl.concat([a, b])
    assert both.height == a.height + b.height
    assert both["isin"].null_count() == a.height, "only the 2010 rows lack ISIN"


def _gapped_frame() -> pl.DataFrame:
    """One ticker with a year-long hole and no corporate action at all."""
    return pl.DataFrame({
        "date": [date(2024, 6, 3), date(2024, 6, 4), date(2025, 6, 2), date(2025, 6, 3)],
        "ticker": ["G.NS"] * 4,
        "close": [100.0, 101.0, 160.0, 161.0],
        # NSE's prev_close on the resumption day refers to the previous SESSION,
        # which is a year earlier here only because our history has a hole.
        "prev_close": [99.0, 100.0, 159.0, 160.0],
        "volume": [100, 100, 100, 100],
    })



def test_a_two_digit_year_timestamp_parses() -> None:
    """NSE switched TIMESTAMP to a two-digit year around mid-2020.

    "13-Jul-20" against a hardcoded %d-%b-%Y raised a plain ValueError, which
    is NOT caught by `except BhavcopyParseError` because that class is a
    SUBCLASS of ValueError — so it propagated and killed a 4,138-day backfill at
    session 2,600.
    """
    doc = _CLASSIC.replace("04-JAN-2016", "13-Jul-20")
    df = parse_classic_bhavcopy(doc)
    assert df["date"].to_list() == [date(2020, 7, 13)] * df.height


def test_both_year_forms_give_the_same_date() -> None:
    a = parse_classic_bhavcopy(_CLASSIC.replace("04-JAN-2016", "13-JUL-2020"))
    b = parse_classic_bhavcopy(_CLASSIC.replace("04-JAN-2016", "13-Jul-20"))
    assert a["date"].to_list() == b["date"].to_list()


def test_an_implausible_year_is_refused_rather_than_guessed() -> None:
    """`%y` maps 69-99 to the 1900s; a 1970 bhavcopy does not exist.

    Accepting it would mean the format guess was wrong and the row silently
    lands two-thousand-odd years from where it belongs.
    """
    with pytest.raises(BhavcopyParseError, match="TIMESTAMP"):
        parse_classic_bhavcopy(_CLASSIC.replace("04-JAN-2016", "13-Jul-70"))


def test_an_unparseable_timestamp_raises_the_typed_error() -> None:
    """It must be a BhavcopyParseError so the fetcher marks the day bad."""
    with pytest.raises(BhavcopyParseError, match="matches none of"):
        parse_classic_bhavcopy(_CLASSIC.replace("04-JAN-2016", "not-a-date"))


def test_the_sec_layout_accepts_both_year_forms_too() -> None:
    a = parse_sec_bhavdata_ohlcv(_SEC)
    b = parse_sec_bhavdata_ohlcv(_SEC.replace("02-Jan-2024", "02-Jan-24"))
    assert a["date"].to_list() == b["date"].to_list() == [date(2024, 1, 2)]
