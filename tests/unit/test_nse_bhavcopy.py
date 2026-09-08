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
    adjustment_ratios,
    back_adjust,
    classic_url,
    isin_map,
    parse_classic_bhavcopy,
    parse_sec_bhavdata_ohlcv,
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


# ── corporate actions ────────────────────────────────────────────────────────


def _split_frame() -> pl.DataFrame:
    """AAA does a 2:1 split overnight into day 3; BBB never acts."""
    return pl.DataFrame(
        {
            "date": [date(2020, 1, d) for d in (1, 2, 3, 6)] * 2,
            "ticker": ["AAA.NS"] * 4 + ["BBB.NS"] * 4,
            #                       split here ↓
            "close": [100.0, 102.0, 51.0, 52.0, 200.0, 201.0, 202.0, 203.0],
            "prev_close": [99.0, 100.0, 51.0, 51.0, 199.0, 200.0, 201.0, 202.0],
            "volume": [1000, 1000, 2000, 2000, 500, 500, 500, 500],
            "open": [100.0, 102.0, 51.0, 52.0, 200.0, 201.0, 202.0, 203.0],
        }
    )


def test_adjustment_ratios_finds_the_split_and_only_the_split() -> None:
    hits = adjustment_ratios(_split_frame())
    assert hits.height == 1, f"expected one action, got {hits.to_dicts()}"
    r = hits.row(0, named=True)
    assert r["ticker"] == "AAA.NS"
    assert r["date"] == date(2020, 1, 3)
    assert r["ratio"] == pytest.approx(0.5)


def test_back_adjust_puts_history_on_the_current_scale() -> None:
    """Pre-split prices are halved; the latest bar is untouched.

    Adjusting on the LATEST scale is Kite's convention (`auto_adjust=True`), and
    mixing conventions between two price sources is the failure this module
    exists to prevent — so the most recent bar must keep its published price.
    """
    out = back_adjust(_split_frame()).sort(["ticker", "date"])
    aaa = out.filter(pl.col("ticker") == "AAA.NS")
    assert aaa["close"].to_list() == pytest.approx([50.0, 51.0, 51.0, 52.0])
    assert aaa["open"].to_list() == pytest.approx([50.0, 51.0, 51.0, 52.0])
    # The return across the split is now +2.0%, not the -50% a raw series shows.
    c = aaa["close"].to_list()
    assert c[2] / c[1] - 1.0 == pytest.approx(0.0, abs=1e-9)
    assert c[1] / c[0] - 1.0 == pytest.approx(0.02)


def test_back_adjust_preserves_turnover_through_volume() -> None:
    """price × volume must not move: halve the price, double the shares."""
    raw = _split_frame()
    out = back_adjust(raw).sort(["ticker", "date"])
    a_raw = raw.filter(pl.col("ticker") == "AAA.NS").sort("date")
    a_new = out.filter(pl.col("ticker") == "AAA.NS")
    for i in range(a_raw.height):
        assert (a_new["close"][i] * a_new["volume"][i]) == pytest.approx(
            a_raw["close"][i] * a_raw["volume"][i]
        )


def test_back_adjust_does_not_leak_across_tickers() -> None:
    """AAA's split must not touch BBB — the classic `.over()` mistake."""
    out = back_adjust(_split_frame())
    bbb = out.filter(pl.col("ticker") == "BBB.NS").sort("date")
    assert bbb["close"].to_list() == pytest.approx([200.0, 201.0, 202.0, 203.0])
    assert bbb["volume"].to_list() == pytest.approx([500, 500, 500, 500])


def test_back_adjust_is_a_no_op_without_actions() -> None:
    df = _split_frame().filter(pl.col("ticker") == "BBB.NS")
    out = back_adjust(df).sort("date")
    assert out["close"].to_list() == pytest.approx(df.sort("date")["close"].to_list())


def test_back_adjust_compounds_two_actions() -> None:
    """Two splits must multiply, not overwrite each other."""
    df = pl.DataFrame(
        {
            "date": [date(2020, 1, d) for d in (1, 2, 3, 6)],
            "ticker": ["AAA.NS"] * 4,
            "close": [100.0, 50.0, 25.0, 26.0],
            "prev_close": [99.0, 50.0, 25.0, 25.0],
            "volume": [100, 200, 400, 400],
        }
    )
    out = back_adjust(df).sort("date")
    # Day 1 sits behind both halvings: 100 * 0.5 * 0.5 = 25.
    assert out["close"].to_list() == pytest.approx([25.0, 25.0, 25.0, 26.0])
    assert out["volume"].to_list() == pytest.approx([400, 400, 400, 400])


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
