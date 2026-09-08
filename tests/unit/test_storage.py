"""Tests for the partitioned OHLCV store."""
from __future__ import annotations


def test_tickers_lists_the_store_deterministically(tmp_path) -> None:  # noqa: ANN001
    """The universe can be taken FROM the store, which is the point of a rebuild.

    Sorted because a panel built from this list gets its column order from it,
    and an unstable order means an unstable panel hash.
    """
    from datetime import date

    import polars as pl

    from trader.data.storage import OhlcvStore

    store = OhlcvStore(root=tmp_path / "s")
    store.save(pl.DataFrame({
        "date": [date(2020, 1, 1), date(2021, 1, 4), date(2020, 1, 1)],
        "ticker": ["ZED.NS", "ALPHA.NS", "ALPHA.NS"],
        "open": [1.0, 2.0, 3.0], "high": [1.0, 2.0, 3.0],
        "low": [1.0, 2.0, 3.0], "close": [1.0, 2.0, 3.0],
        "volume": [10, 20, 30],
    }))
    got = store.tickers()
    assert got == ["ALPHA.NS", "ZED.NS"], "not sorted, or a year partition missed"
    # A ticker spanning two years must appear once, not once per partition.
    assert got.count("ALPHA.NS") == 1


def test_tickers_is_empty_for_a_missing_root(tmp_path) -> None:  # noqa: ANN001
    from trader.data.storage import OhlcvStore

    assert OhlcvStore(root=tmp_path / "nope").tickers() == []


def test_tickers_includes_the_index_so_callers_must_filter(tmp_path) -> None:  # noqa: ANN001
    """`tickers()` lists what is stored, benchmark included.

    `^NSEI` lives in the same store because `compute_features` needs it for
    `beta_nifty_60d`, but it is a benchmark and not something to hold. A caller
    building a universe from the store has to drop it, or the action space
    gains a column that can never be traded — and one that would take an
    "unknown industry" id on the way in.
    """
    from datetime import date

    import polars as pl

    from trader.data.storage import OhlcvStore

    store = OhlcvStore(root=tmp_path / "s")
    store.save(pl.DataFrame({
        "date": [date(2020, 1, 1), date(2020, 1, 1)],
        "ticker": ["TCS.NS", "^NSEI"],
        "open": [1.0, 2.0], "high": [1.0, 2.0],
        "low": [1.0, 2.0], "close": [1.0, 2.0], "volume": [10, 20],
    }))
    assert store.tickers() == ["TCS.NS", "^NSEI"]
    tradeable = [t for t in store.tickers() if not t.startswith("^")]
    assert tradeable == ["TCS.NS"]
