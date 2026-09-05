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


def _make_index_rets(
    ohlcv: pl.DataFrame,
    seed: int = 7,
) -> pl.DataFrame:
    """Synthetic benchmark log returns on the same calendar as `ohlcv`.

    `compute_features` requires these — a missing benchmark is a hard error, not
    a beta of 1.0.  Returns must have real variance: a flat index gives a
    degenerate covariance window and beta comes back null (correctly), which
    would make every row non-tradeable.
    """
    import random

    rng = random.Random(seed)
    dates = sorted(set(ohlcv["date"].to_list()))
    return pl.DataFrame(
        {
            "date": dates,
            "index_return": [rng.gauss(0.0002, 0.01) for _ in dates],
        }
    )


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
    panel = compute_features(panel, index_rets=_make_index_rets(ohlcv))

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
    panel = compute_features(
        align_panel(ohlcv, ["TEST.NS"], calendar), index_rets=_make_index_rets(ohlcv)
    )

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
    panel = compute_features(
        align_panel(ohlcv, ["TEST.NS"], calendar), index_rets=_make_index_rets(ohlcv)
    )
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
    panel = compute_features(
        align_panel(ohlcv, ["X.NS"], calendar), index_rets=_make_index_rets(ohlcv)
    )

    tradeable = panel.filter(pl.col("is_tradeable"))
    if len(tradeable) == 0:
        pytest.skip("No tradeable rows in synthetic data")

    rsi = tradeable["rsi_14"]
    assert float(rsi.min()) >= 0.0, f"RSI below 0: {rsi.min()}"
    assert float(rsi.max()) <= 100.0, f"RSI above 100: {rsi.max()}"


# ── 9. beta_nifty_60d must not fake a benchmark (B7) ──────────────────────────


def test_beta_without_index_raises_instead_of_returning_one() -> None:
    """The regression this whole guard exists for.

    `_add_beta` used to return a constant 1.0 when no index was supplied, which
    is how beta_nifty_60d came to be constant across all 276,005 tradeable
    training rows: data/ohlcv has no ^NSEI, so index_rets was None and nothing
    said so.  A missing benchmark must be an error, never a plausible number.
    """
    from trader.data.alignment import align_panel
    from trader.data.features import MissingBenchmarkError, compute_features

    ohlcv = _make_ohlcv(n=100, ticker="X.NS")
    calendar = sorted(ohlcv["date"].to_list())
    panel = align_panel(ohlcv, ["X.NS"], calendar)

    with pytest.raises(MissingBenchmarkError, match="beta_nifty_60d"):
        compute_features(panel, index_rets=None)

    # Same for the default: forgetting the argument must not be the quiet path.
    with pytest.raises(MissingBenchmarkError):
        compute_features(panel)


def test_beta_is_not_constant_when_index_is_supplied() -> None:
    """With a real benchmark the channel has to actually vary.

    `compute_feature_stats` maps a zero-variance feature to (0.0, 1.0), which
    normalises a dead channel to a fixed +1 into every convolution — so
    "constant" is not a harmless outcome and is worth asserting against
    directly, not just inferring from the absence of an exception.
    """
    from trader.data.alignment import align_panel
    from trader.data.features import compute_features

    ohlcv = _make_ohlcv(n=140, ticker="X.NS")
    calendar = sorted(ohlcv["date"].to_list())
    panel = compute_features(
        align_panel(ohlcv, ["X.NS"], calendar), index_rets=_make_index_rets(ohlcv)
    )

    beta = panel.filter(pl.col("is_tradeable"))["beta_nifty_60d"]
    assert len(beta) > 0, "expected tradeable rows"
    assert float(beta.std() or 0.0) > 1e-8, (
        f"beta_nifty_60d is constant at {beta[0]} — a dead channel, not a feature"
    )


def test_beta_is_null_not_one_on_dates_the_index_does_not_cover() -> None:
    """A short benchmark is the same bug in a quieter costume.

    Uncovered dates used to fall through to 1.0.  They must go null, which
    `compute_features` turns into is_tradeable=False for exactly those rows —
    local and visible, rather than a fabricated beta indistinguishable from a
    measured one.
    """
    from trader.data.alignment import align_panel
    from trader.data.features import compute_features

    ohlcv = _make_ohlcv(n=140, ticker="X.NS")
    calendar = sorted(ohlcv["date"].to_list())
    index_rets = _make_index_rets(ohlcv)
    cutoff = calendar[100]
    short_index = index_rets.filter(pl.col("date") <= cutoff)

    panel = compute_features(
        align_panel(ohlcv, ["X.NS"], calendar), index_rets=short_index
    )

    uncovered = panel.filter(pl.col("date") > cutoff)
    assert len(uncovered) > 0
    assert uncovered["is_tradeable"].sum() == 0, (
        "rows the benchmark does not cover must not be tradeable"
    )
    # Sentinel-filled to 0.0 after the mask — the point is that it is not 1.0.
    assert 1.0 not in uncovered["beta_nifty_60d"].to_list()


# ── 10. Feature lookback table drives the walk-forward purge guard (B3) ───────


def test_feature_lookback_table_covers_every_feature() -> None:
    """The purge guard is only as good as this table's coverage."""
    from trader.data.features import FEATURE_COLS, FEATURE_LOOKBACK_DAYS

    assert set(FEATURE_LOOKBACK_DAYS) == set(FEATURE_COLS), (
        "FEATURE_LOOKBACK_DAYS and FEATURE_COLS have diverged: "
        f"missing={set(FEATURE_COLS) - set(FEATURE_LOOKBACK_DAYS)}, "
        f"extra={set(FEATURE_LOOKBACK_DAYS) - set(FEATURE_COLS)}"
    )


def test_no_window_literal_exceeds_the_declared_max_lookback() -> None:
    """Adding a longer-window feature must tighten the purge guard, not slip past it.

    Scans the module source for the window sizes actually used.  If someone adds
    a 120-day rolling feature without updating FEATURE_LOOKBACK_DAYS, the purge
    guard in `compute_windows` would keep passing a gap that no longer clears the
    features — the exact silent regression B3 was.
    """
    from trader.data import features as features_mod

    source = Path(features_mod.__file__).read_text()
    literals = [
        int(m)
        for m in re.findall(r"window_size=(\d+)|\bspan=(\d+)|\.shift\((\d+)\)", source)
        for m in (m if isinstance(m, tuple) else (m,))
        if m
    ]
    assert literals, "expected to find rolling-window literals in features.py"
    assert max(literals) <= features_mod.MAX_FEATURE_LOOKBACK_DAYS, (
        f"features.py uses a {max(literals)}-day window but "
        f"MAX_FEATURE_LOOKBACK_DAYS is {features_mod.MAX_FEATURE_LOOKBACK_DAYS} — "
        "update FEATURE_LOOKBACK_DAYS so the walk-forward purge guard follows."
    )


# ── 11. data.panels_root actually selects the panel (B2) ─────────────────────


def _write_stub_panel(path: Path, tickers: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "date": [date(2020, 1, 2)] * len(tickers),
            "ticker": tickers,
            "is_tradeable": [True] * len(tickers),
        }
    ).write_parquet(path)


def test_resolve_panels_root_honours_a_non_default_config_path() -> None:
    """`data.panels_root` must select the dataset, not be declared and ignored.

    data/panels_kite was built — live beta, 2005-2026, corporate-action masking
    — and then orphaned: build_features.py read `data.panels_root`, every other
    entrypoint hardcoded data/panels, and no override could reach the new panel.
    A smoke run silently used the stale yfinance data.
    """
    from omegaconf import OmegaConf

    from trader.data.features import resolve_panels_root

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write_stub_panel(root / "data" / "panels" / "train.parquet", ["A.NS"])
        _write_stub_panel(
            root / "data" / "panels_kite" / "train.parquet", ["A.NS", "B.NS", "C.NS"]
        )

        cfg = OmegaConf.create({"data": {"panels_root": "data/panels_kite"}})
        assert resolve_panels_root(cfg, root) == root / "data" / "panels_kite"

        # Absent key → the historical default, so existing invocations are unchanged.
        assert resolve_panels_root(OmegaConf.create({"data": {}}), root) == (
            root / "data" / "panels"
        )


def test_resolve_panels_root_logs_the_resolved_path_and_ticker_count() -> None:
    """Silence is the bug class.  A run must say which panel it opened."""
    from loguru import logger
    from omegaconf import OmegaConf

    from trader.data.features import resolve_panels_root

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write_stub_panel(
            root / "data" / "panels_kite" / "train.parquet", ["A.NS", "B.NS", "C.NS"]
        )
        cfg = OmegaConf.create({"data": {"panels_root": "data/panels_kite"}})

        lines: list[str] = []
        sink_id = logger.add(lines.append, level="INFO", format="{message}")
        try:
            resolve_panels_root(cfg, root)
        finally:
            logger.remove(sink_id)

        blob = "".join(lines)
        assert "panels_kite" in blob, f"resolved path not logged: {blob!r}"
        assert "3 tickers" in blob, f"ticker count not logged: {blob!r}"


def test_every_entrypoint_resolves_the_panel_root_from_config() -> None:
    """Static guard against re-hardcoding the path in any driver script.

    Four of the five entrypoints had `orig_cwd / "data" / "panels"` inlined.
    That is what made `data.panels_root` unreachable, and nothing would have
    caught a fifth copy appearing.
    """
    scripts = [
        "scripts/train.py",
        "scripts/walk_forward.py",
        "scripts/paper_run.py",
        "scripts/evaluate.py",
        "scripts/build_features.py",
    ]
    for script in scripts:
        source = Path(script).read_text()
        assert "resolve_panels_root(cfg" in source, (
            f"{script} does not resolve the panel root from config"
        )
        assert '"data" / "panels"' not in source, (
            f"{script} hardcodes the panel path again — use resolve_panels_root"
        )
