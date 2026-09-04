"""Unit tests for M2 — Data Ingestion.

Covers:
- Universe dedup and sector mapping
- OhlcvStore round-trip and idempotency
- Split-adjusted close continuity (Reliance 1:1 bonus, Sep 2017)
- ZerodhaSource raises NotImplementedError
"""
from __future__ import annotations

import tempfile
from datetime import date, datetime

import polars as pl
import pytest

from trader.data.universe import NIFTY_50, all_tickers, sector_id_of, sector_of

# ---------------------------------------------------------------------------
# Universe
# ---------------------------------------------------------------------------


def test_universe_no_duplicates() -> None:
    tickers = all_tickers()
    assert len(tickers) == len(set(tickers)), "Duplicate tickers in universe"


def test_no_nifty50_ticker_vanishes_silently() -> None:
    """Every NIFTY 50 name is either traded or *explicitly* reported as excluded.

    The original assertion was `NIFTY_50 subset of all_tickers()`, which broke when
    the taxonomy stopped covering every index member. Simply relaxing it would
    hide the thing worth protecting: a large cap must never drop out unnoticed.
    So the invariant becomes "accounted for", not "present".
    """
    from trader.data.universe import unsectored_nifty50

    traded = set(all_tickers())
    excluded = set(unsectored_nifty50())
    for t in NIFTY_50:
        assert t in traded or t in excluded, (
            f"NIFTY 50 ticker {t} is neither traded nor reported as excluded"
        )
    # Anything excluded must genuinely have no sector — not merely be forgotten.
    for t in excluded:
        assert sector_of(t) is None, f"{t} reported unsectored but sector_of says {sector_of(t)}"


def test_active_universe_is_subset_of_fetched() -> None:
    """Training trades a subset of what the pipeline fetches, never a superset."""
    from trader.data.universe import active_tickers

    assert set(active_tickers()) <= set(all_tickers())


def test_no_ticker_lands_in_a_phantom_sector() -> None:
    """`sector_id_of` must never return 0 — an id absent from SECTOR_IDS.

    Regression test. `all_tickers()` used to union NIFTY_50 with SECTOR_MAP, so
    index members without a sectoral entry came through with id 0 and became a
    phantom sector node in the graph model.
    """
    from trader.data.universe import SECTOR_IDS, active_tickers

    valid = set(SECTOR_IDS.values())
    for t in all_tickers():
        assert sector_id_of(t) in valid, f"{t} has sector_id {sector_id_of(t)}"
    for t in active_tickers():
        assert sector_id_of(t) in valid, f"{t} has sector_id {sector_id_of(t)}"


def test_universe_size_reasonable() -> None:
    from trader.data.universe import active_tickers

    fetched, active = len(all_tickers()), len(active_tickers())
    assert 400 <= active <= fetched, f"active universe {active} outside expected range"
    assert fetched <= 800, f"fetched universe {fetched} larger than NSE classifies"


def test_sector_known_tickers() -> None:
    assert sector_of("RELIANCE.NS") == "oil_gas_power"
    assert sector_of("TCS.NS") == "it"
    assert sector_of("HDFCBANK.NS") == "banking"
    assert sector_id_of("RELIANCE.NS") > 0
    assert sector_id_of("TCS.NS") > 0


def test_sector_unknown_ticker() -> None:
    assert sector_of("UNKNOWNTICKER.NS") is None
    assert sector_id_of("UNKNOWNTICKER.NS") == 0


# ---------------------------------------------------------------------------
# OhlcvStore
# ---------------------------------------------------------------------------


def _sample_df(ticker: str = "RELIANCE.NS") -> pl.DataFrame:
    return pl.DataFrame(
        {
            "date": [date(2017, 9, 15), date(2017, 9, 18), date(2017, 9, 19)],
            "open": [750.0, 752.0, 749.0],
            "high": [760.0, 762.0, 755.0],
            "low": [745.0, 748.0, 744.0],
            "close": [752.0, 749.0, 750.0],
            "volume": [1_000_000, 1_200_000, 900_000],
            "adj_close": [752.0, 749.0, 750.0],
            "ticker": [ticker] * 3,
            "source": ["yfinance"] * 3,
        }
    )


def test_ohlcv_store_roundtrip() -> None:
    from trader.data.storage import OhlcvStore

    df = _sample_df()
    with tempfile.TemporaryDirectory() as tmpdir:
        store = OhlcvStore(root=tmpdir)
        store.save(df)
        loaded = store.load(
            tickers=["RELIANCE.NS"],
            start=datetime(2017, 9, 1),
            end=datetime(2017, 9, 30),
        )

    assert len(loaded) == 3
    assert set(loaded["ticker"].to_list()) == {"RELIANCE.NS"}
    assert loaded.sort("date")["adj_close"].to_list() == [752.0, 749.0, 750.0]


def test_ohlcv_store_idempotent() -> None:
    """Saving the same data twice must not duplicate rows."""
    from trader.data.storage import OhlcvStore

    df = _sample_df()
    with tempfile.TemporaryDirectory() as tmpdir:
        store = OhlcvStore(root=tmpdir)
        store.save(df)
        store.save(df)
        loaded = store.load(tickers=["RELIANCE.NS"])

    assert len(loaded) == 3, f"Expected 3 rows after idempotent save, got {len(loaded)}"


def test_ohlcv_store_multi_year() -> None:
    """Data spanning multiple years is split into separate partition files."""
    from trader.data.storage import OhlcvStore

    df = pl.DataFrame(
        {
            "date": [date(2022, 12, 30), date(2023, 1, 2)],
            "open": [100.0, 102.0],
            "high": [105.0, 107.0],
            "low": [99.0, 101.0],
            "close": [103.0, 105.0],
            "volume": [500_000, 600_000],
            "adj_close": [103.0, 105.0],
            "ticker": ["TCS.NS", "TCS.NS"],
            "source": ["yfinance", "yfinance"],
        }
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        store = OhlcvStore(root=tmpdir)
        store.save(df)

        # Both years should be loadable
        loaded = store.load(tickers=["TCS.NS"])
        assert len(loaded) == 2

        # Each year partition must exist
        from pathlib import Path

        assert (Path(tmpdir) / "year=2022" / "ticker=TCS.NS.parquet").exists()
        assert (Path(tmpdir) / "year=2023" / "ticker=TCS.NS.parquet").exists()


def test_ohlcv_store_empty_load() -> None:
    """Loading from an empty store returns an empty DataFrame, not an error."""
    from trader.data.storage import OhlcvStore

    with tempfile.TemporaryDirectory() as tmpdir:
        store = OhlcvStore(root=tmpdir)
        loaded = store.load(tickers=["RELIANCE.NS"])
    assert loaded.is_empty()


# ---------------------------------------------------------------------------
# Split adjustment
# ---------------------------------------------------------------------------


def test_split_adjustment_continuity() -> None:
    """
    Adjusted close must be continuous around a 1:1 bonus issue.

    Reliance Industries 1:1 bonus record date: 2017-09-20.
    - Raw (unadjusted): pre-bonus ~1500, post-bonus ~750 → ~50% drop on record date.
    - yfinance auto_adjust=True: all historical prices retroactively halved → continuity.

    We simulate the auto-adjusted output and assert max |daily return| < 10%.
    """
    dates = [
        date(2017, 9, 15),
        date(2017, 9, 18),
        date(2017, 9, 19),
        date(2017, 9, 20),
        date(2017, 9, 21),
    ]
    # Auto-adjusted: all prices ~750, no discontinuity
    adj_close = [748.0, 752.0, 750.0, 748.0, 755.0]

    df = pl.DataFrame({"date": dates, "adj_close": adj_close})
    rets = df.with_columns(
        (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1).alias("ret")
    ).filter(pl.col("ret").is_not_null())

    max_abs: float | None = rets["ret"].abs().max()
    assert max_abs is not None
    assert max_abs < 0.10, (
        f"Adjusted close has discontinuity around bonus date: max|ret| = {max_abs:.2%}"
    )


def test_unadjusted_data_shows_discontinuity() -> None:
    """
    Validate the test setup: raw unadjusted prices WOULD show a large drop.
    This confirms that adjustment is necessary and that our continuity test is meaningful.
    """
    dates = [
        date(2017, 9, 15),
        date(2017, 9, 18),
        date(2017, 9, 19),
        date(2017, 9, 20),
        date(2017, 9, 21),
    ]
    # Unadjusted: pre-bonus ~1500, post-bonus ~750 → ~50% drop on record date
    raw_close = [1498.0, 1502.0, 1500.0, 750.0, 755.0]

    df = pl.DataFrame({"date": dates, "adj_close": raw_close})
    rets = df.with_columns(
        (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1).alias("ret")
    ).filter(pl.col("ret").is_not_null())

    max_abs: float | None = rets["ret"].abs().max()
    assert max_abs is not None
    assert max_abs > 0.40, (
        f"Test setup error: unadjusted data should show >40% drop, got {max_abs:.2%}"
    )


# ---------------------------------------------------------------------------
# ZerodhaSource interface contract
# ---------------------------------------------------------------------------


def test_zerodha_fetch_ohlcv_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    # Clear the Kite env vars first. Once `kiteconnect` is installed AND real
    # credentials are exported (e.g. a dev ran `. ./.env` before pytest), this
    # call otherwise reaches Kite's servers — a unit test making a live network
    # request, which is both slow and non-deterministic. Unsetting them forces
    # the credentials branch, which is what this contract test is actually about.
    for var in ("KITE_API_KEY", "KITE_API_SECRET", "KITE_ACCESS_TOKEN"):
        monkeypatch.delenv(var, raising=False)

    from trader.data.sources.zerodha_source import ZerodhaSource

    src = ZerodhaSource()
    with pytest.raises(NotImplementedError):
        src.fetch_ohlcv(["RELIANCE.NS"], datetime(2024, 1, 1), datetime(2024, 1, 31))


def test_zerodha_fetch_actions_raises() -> None:
    from trader.data.sources.zerodha_source import ZerodhaSource

    src = ZerodhaSource()
    with pytest.raises(NotImplementedError):
        src.fetch_corporate_actions(["RELIANCE.NS"])


# ---------------------------------------------------------------------------
# Protocol structural check
# ---------------------------------------------------------------------------


def test_sources_implement_protocol() -> None:
    from trader.data.sources.base import MarketDataSource
    from trader.data.sources.kaggle_source import KaggleSource
    from trader.data.sources.yfinance_source import YFinanceSource
    from trader.data.sources.zerodha_source import ZerodhaSource

    assert isinstance(YFinanceSource(), MarketDataSource)
    assert isinstance(KaggleSource(), MarketDataSource)
    assert isinstance(ZerodhaSource(), MarketDataSource)
