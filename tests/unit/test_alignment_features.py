"""Unit tests for M3 — Alignment and Features.

Acceptance criteria from 07_roadmap.md:
1. Mask correctness: ticker listed in 2017 has is_tradeable=False for 2014–2016.
2. No-lookahead static scan: features.py must not contain .shift(-k) patterns.
3. No NaN in numeric feature columns on is_tradeable=True rows.
4. SHA256 determinism: rerun produces identical checksum.

Additional coverage:
- Forward-fill 1-day gap → is_tradeable=True, volume=0.
- 2+-day gap → is_tradeable=False (beyond forward-fill limit).
- After last trade date → is_tradeable=False.
"""
from __future__ import annotations

import hashlib
import re
import tempfile
from datetime import date, timedelta
from pathlib import Path

import polars as pl
import pytest

# ── fixtures / helpers ─────────────────────────────────────────────────────────


def _make_ohlcv(
    n: int = 100,
    ticker: str = "TEST.NS",
    start: date = date(2020, 1, 2),
    seed: int = 42,
) -> pl.DataFrame:
    """Synthetic OHLCV data: n sequential calendar days starting from start."""
    import random

    rng = random.Random(seed)
    price = 100.0
    rows = []
    for i in range(n):
        d = start + timedelta(days=i)
        price = max(1.0, price * (1.0 + rng.gauss(0.0, 0.015)))
        rows.append(
            {
                "date": d,
                "ticker": ticker,
                "open": round(price * 0.998, 4),
                "high": round(price * 1.006, 4),
                "low": round(price * 0.994, 4),
                "close": round(price, 4),
                "adj_close": round(price, 4),
                "volume": rng.randint(100_000, 1_000_000),
                "source": "test",
            }
        )
    return pl.DataFrame(rows)


# ── 1. Mask correctness ────────────────────────────────────────────────────────


def test_mask_ticker_listed_2017_is_false_for_2014_2016() -> None:
    """A ticker whose first trade is 2017-01-02 must be non-tradeable for all earlier dates."""
    from trader.data.alignment import align_panel

    ohlcv = pl.DataFrame(
        {
            "date": [date(2017, 1, 2), date(2017, 1, 3), date(2017, 1, 4)],
            "ticker": ["NEW.NS"] * 3,
            "open": [100.0, 101.0, 102.0],
            "high": [102.0, 103.0, 104.0],
            "low": [99.0, 100.0, 101.0],
            "close": [101.0, 102.0, 103.0],
            "adj_close": [101.0, 102.0, 103.0],
            "volume": [100_000, 120_000, 110_000],
            "source": ["test"] * 3,
        }
    )
    # Calendar spans 2014–2017
    calendar = (
        [date(2014, 1, 2), date(2015, 1, 2), date(2016, 1, 4), date(2016, 12, 30)]
        + [date(2017, 1, 2), date(2017, 1, 3), date(2017, 1, 4)]
    )

    panel = align_panel(ohlcv, ["NEW.NS"], calendar)

    pre_2017 = panel.filter(pl.col("date") < date(2017, 1, 1))
    assert len(pre_2017) == 4, "Expected 4 pre-2017 rows"
    assert pre_2017["is_tradeable"].to_list() == [False] * 4

    in_2017 = panel.filter(pl.col("date") >= date(2017, 1, 2))
    assert in_2017["is_tradeable"].to_list() == [True, True, True]


def test_mask_sentinel_prices_are_zero_on_non_tradeable() -> None:
    """Non-tradeable rows must have price columns filled with 0, not left null."""
    from trader.data.alignment import align_panel

    ohlcv = _make_ohlcv(n=5, ticker="X.NS", start=date(2020, 3, 1))
    calendar = [date(2020, 1, 15)] + sorted(ohlcv["date"].to_list())

    panel = align_panel(ohlcv, ["X.NS"], calendar)
    nt = panel.filter(~pl.col("is_tradeable"))

    for col in ["open", "high", "low", "close", "adj_close"]:
        assert nt[col].to_list() == [0.0], f"{col} should be 0.0 on non-tradeable row"
    assert nt["volume"].to_list() == [0]


# ── 2. Forward-fill gap rules ──────────────────────────────────────────────────


def test_one_day_gap_is_tradeable_volume_zero() -> None:
    """A single missing bar within the tradeable span is forward-filled and tradeable."""
    from trader.data.alignment import align_panel

    # Jan 2 and Jan 6 with a gap on Jan 3–5; but calendar only has Jan 3 (1-day gap)
    ohlcv = pl.DataFrame(
        {
            "date": [date(2020, 1, 2), date(2020, 1, 6)],
            "ticker": ["X.NS"] * 2,
            "open": [100.0, 106.0],
            "high": [102.0, 108.0],
            "low": [99.0, 105.0],
            "close": [101.0, 107.0],
            "adj_close": [101.0, 107.0],
            "volume": [500_000, 600_000],
            "source": ["test", "test"],
        }
    )
    # Calendar has only Jan 3 as the single gap between Jan 2 and Jan 6
    calendar = [date(2020, 1, 2), date(2020, 1, 3), date(2020, 1, 6)]
    panel = align_panel(ohlcv, ["X.NS"], calendar)

    jan3 = panel.filter(pl.col("date") == date(2020, 1, 3))
    assert jan3["is_tradeable"].to_list() == [True], "1-day gap should be tradeable"
    assert jan3["adj_close"].to_list() == [101.0], "Should be forward-filled from Jan 2"
    assert jan3["volume"].to_list() == [0], "Volume on a gap day should be 0"


def test_two_day_gap_second_day_is_not_tradeable() -> None:
    """A 2-day consecutive gap: first gap fills (limit=1), second does not."""
    from trader.data.alignment import align_panel

    ohlcv = pl.DataFrame(
        {
            "date": [date(2020, 1, 2), date(2020, 1, 7)],
            "ticker": ["X.NS"] * 2,
            "open": [100.0, 106.0],
            "high": [102.0, 108.0],
            "low": [99.0, 105.0],
            "close": [101.0, 107.0],
            "adj_close": [101.0, 107.0],
            "volume": [500_000, 600_000],
            "source": ["test", "test"],
        }
    )
    # Calendar: Jan 2, then Jan 3 (gap 1), Jan 6 (gap 2), then Jan 7
    calendar = [date(2020, 1, 2), date(2020, 1, 3), date(2020, 1, 6), date(2020, 1, 7)]
    panel = align_panel(ohlcv, ["X.NS"], calendar)

    jan3 = panel.filter(pl.col("date") == date(2020, 1, 3))
    assert jan3["is_tradeable"].to_list() == [True], "First gap day: should fill"

    jan6 = panel.filter(pl.col("date") == date(2020, 1, 6))
    assert jan6["is_tradeable"].to_list() == [False], "Second consecutive gap: not filled"


def test_after_last_trade_is_not_tradeable() -> None:
    """Dates after the last trade date are marked non-tradeable."""
    from trader.data.alignment import align_panel

    ohlcv = _make_ohlcv(n=3, ticker="X.NS", start=date(2020, 1, 2))
    last = date(2020, 1, 4)  # 3 days
    future_dates = [date(2020, 1, 5), date(2020, 1, 6)]
    calendar = sorted(ohlcv["date"].to_list()) + future_dates

    panel = align_panel(ohlcv, ["X.NS"], calendar)
    post = panel.filter(pl.col("date") > last)
    assert post["is_tradeable"].to_list() == [False, False]


# ── 3. No-lookahead static scan ────────────────────────────────────────────────


def test_features_no_negative_shift() -> None:
    """features.py must not contain .shift(-k) — that would be a lookahead."""
    features_path = (
        Path(__file__).parent.parent.parent / "src" / "trader" / "data" / "features.py"
    )
    source = features_path.read_text()

    assert not re.search(r"\.shift\s*\(\s*-", source), (
        "features.py contains a negative shift — potential lookahead bias detected"
    )
    assert "x[-" not in source, (
        "features.py contains reverse-indexing in a lambda — potential lookahead bias"
    )


# ── 4. No NaN on tradeable rows ────────────────────────────────────────────────


def test_no_nan_on_tradeable_rows() -> None:
    """All float feature columns must be NaN-free on is_tradeable=True rows."""
    from trader.data.alignment import align_panel
    from trader.data.features import FEATURE_COLS, compute_features

    # 100 consecutive days: enough history for all 60-day features
    ohlcv = _make_ohlcv(n=100, ticker="TEST.NS")
    calendar = sorted(ohlcv["date"].to_list())
    panel = align_panel(ohlcv, ["TEST.NS"], calendar)
    panel = compute_features(panel)

    tradeable = panel.filter(pl.col("is_tradeable"))
    assert len(tradeable) > 0, "Expected at least some tradeable rows"

    for col in FEATURE_COLS:
        if col not in tradeable.columns:
            continue
        null_count = tradeable[col].null_count()
        nan_count = tradeable[col].is_nan().sum()
        assert null_count == 0, f"Feature {col!r} has {null_count} nulls on tradeable rows"
        assert nan_count == 0, f"Feature {col!r} has {nan_count} NaNs on tradeable rows"


def test_warm_up_rows_are_not_tradeable() -> None:
    """Rows without enough history for 60-day features must not be tradeable."""
    from trader.data.alignment import align_panel
    from trader.data.features import compute_features

    ohlcv = _make_ohlcv(n=100, ticker="TEST.NS")
    calendar = sorted(ohlcv["date"].to_list())
    panel = compute_features(align_panel(ohlcv, ["TEST.NS"], calendar))

    # The first 60 rows can't have a valid realized_vol_60d; they must be non-tradeable
    first_60 = panel.head(60)
    # At least some of those first 60 rows should be non-tradeable (warm-up period)
    assert first_60["is_tradeable"].sum() < 60, (
        "Expected warm-up rows to be non-tradeable due to missing 60-day history"
    )


# ── 5. SHA256 determinism ──────────────────────────────────────────────────────


def test_sha256_deterministic() -> None:
    """Writing the same panel twice must produce identical SHA256 checksums."""
    from trader.data.alignment import align_panel
    from trader.data.features import compute_features

    ohlcv = _make_ohlcv(n=100, ticker="TEST.NS")
    calendar = sorted(ohlcv["date"].to_list())
    panel = compute_features(align_panel(ohlcv, ["TEST.NS"], calendar))
    panel = panel.sort(["date", "ticker"])

    with tempfile.TemporaryDirectory() as tmpdir:
        p = Path(tmpdir) / "panel.parquet"
        panel.write_parquet(p)
        sha1 = hashlib.sha256(p.read_bytes()).hexdigest()

        panel.write_parquet(p)
        sha2 = hashlib.sha256(p.read_bytes()).hexdigest()

    assert sha1 == sha2, "Parquet output is non-deterministic — SHA256 mismatch"


# ── 6. Multi-ticker alignment ──────────────────────────────────────────────────


def test_two_tickers_independent_masks() -> None:
    """Two tickers with different listing dates get independent is_tradeable masks."""
    from trader.data.alignment import align_panel

    # Ticker A: starts 2016-01-04
    ohlcv_a = _make_ohlcv(n=5, ticker="A.NS", start=date(2016, 1, 4))
    # Ticker B: starts 2018-01-02 (later)
    ohlcv_b = _make_ohlcv(n=5, ticker="B.NS", start=date(2018, 1, 2))
    ohlcv = pl.concat([ohlcv_a, ohlcv_b])

    calendar = sorted(
        ohlcv_a["date"].to_list()
        + ohlcv_b["date"].to_list()
        + [date(2015, 1, 2), date(2017, 1, 3)]
    )

    panel = align_panel(ohlcv, ["A.NS", "B.NS"], calendar)

    # A.NS must be non-tradeable before 2016-01-04
    a_pre = panel.filter(
        (pl.col("ticker") == "A.NS") & (pl.col("date") < date(2016, 1, 4))
    )
    assert a_pre["is_tradeable"].to_list() == [False]

    # B.NS must be non-tradeable during A.NS's tradeable period
    b_during_a = panel.filter(
        (pl.col("ticker") == "B.NS") & (pl.col("date") < date(2018, 1, 2))
    )
    assert all(not v for v in b_during_a["is_tradeable"].to_list())


# ── 7. Sector id propagation ───────────────────────────────────────────────────


def test_sector_id_propagated() -> None:
    """sector_id from the provided mapping appears in the aligned panel."""
    from trader.data.alignment import align_panel

    ohlcv = _make_ohlcv(n=3, ticker="REL.NS")
    calendar = sorted(ohlcv["date"].to_list())

    panel = align_panel(ohlcv, ["REL.NS"], calendar, sector_ids={"REL.NS": 7})
    assert panel["sector_id"].to_list() == [7, 7, 7]


# ── 8. Feature sanity checks ───────────────────────────────────────────────────


def test_log_return_1d_value() -> None:
    """log_return_1d must equal log(adj_close_t / adj_close_{t-1}).

    Tested via _add_returns directly: with only 3 rows all downstream 60-day
    features are null, so compute_features() forces is_tradeable=False and
    sentinel-fills log_return_1d to 0.  The formula itself is still correct —
    we verify it before the masking step.
    """
    import math

    from trader.data.features import _add_returns

    df = pl.DataFrame(
        {
            "date": [date(2020, 1, 2), date(2020, 1, 3), date(2020, 1, 6)],
            "ticker": ["X.NS"] * 3,
            "adj_close": [100.0, 110.0, 105.0],
        }
    ).sort(["ticker", "date"])

    df = _add_returns(df)

    jan3 = df.filter(pl.col("date") == date(2020, 1, 3))
    assert abs(float(jan3["log_return_1d"][0]) - math.log(110.0 / 100.0)) < 1e-6

    jan6 = df.filter(pl.col("date") == date(2020, 1, 6))
    assert abs(float(jan6["log_return_1d"][0]) - math.log(105.0 / 110.0)) < 1e-6


def test_rsi_bounded() -> None:
    """RSI must be in [0, 100] on all tradeable rows."""
    from trader.data.alignment import align_panel
    from trader.data.features import compute_features

    ohlcv = _make_ohlcv(n=100, ticker="X.NS")
    calendar = sorted(ohlcv["date"].to_list())
    panel = compute_features(align_panel(ohlcv, ["X.NS"], calendar))

    tradeable = panel.filter(pl.col("is_tradeable"))
    if len(tradeable) == 0:
        pytest.skip("No tradeable rows in synthetic data")

    rsi = tradeable["rsi_14"]
    assert float(rsi.min()) >= 0.0, f"RSI below 0: {rsi.min()}"
    assert float(rsi.max()) <= 100.0, f"RSI above 100: {rsi.max()}"
