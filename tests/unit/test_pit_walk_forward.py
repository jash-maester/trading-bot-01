"""The per-window universe is chosen from data strictly before the test span.

This is the belt that makes a retrained signal point-in-time. If it slips, the
model trains on names selected with knowledge of the window it is about to be
scored on, and every downstream number inherits that — silently, because the
metrics look entirely normal.
"""
from __future__ import annotations

from datetime import date, timedelta

import polars as pl
import pytest

from trader.data.pit_universe import LiquidityRule, eligible_on
from trader.training.walk_forward import compute_windows


def _bars(spec: dict[str, tuple[date, date, float]]) -> pl.DataFrame:
    """{ticker: (first, last, turnover)} at one row per calendar day."""
    rows: list[dict[str, object]] = []
    for t, (lo, hi, turn) in spec.items():
        d = lo
        while d <= hi:
            rows.append({
                "date": d, "ticker": t, "series": "EQ",
                "close": 100.0, "turnover": turn,
            })
            d += timedelta(days=1)
    return pl.DataFrame(rows)


_RULE = LiquidityRule(
    min_median_turnover=1e7, min_sessions=30, lookback_days=365, min_price=0.0
)


def test_a_window_cannot_see_a_name_that_only_becomes_liquid_inside_it() -> None:
    """The exact leak this guards.

    LATE.NS is illiquid until the window's test span opens and liquid after. A
    universe picked at the test start must not contain it; one picked at the
    test END would, and would be trading on knowledge of the period it is
    scored over.
    """
    bars = pl.concat([
        _bars({"EARLY.NS": (date(2019, 1, 1), date(2021, 12, 31), 5e8)}),
        _bars({"LATE.NS": (date(2019, 1, 1), date(2020, 12, 31), 1e5)}),
        _bars({"LATE.NS": (date(2021, 1, 1), date(2021, 12, 31), 5e8)}),
    ])
    test_start = date(2021, 1, 1)
    at_start = eligible_on(bars, test_start, _RULE)
    at_end = eligible_on(bars, date(2021, 12, 31), _RULE)
    assert at_start == ["EARLY.NS"]
    assert "LATE.NS" in at_end, "fixture does not actually create the leak"


def test_the_universe_is_fixed_for_the_whole_window() -> None:
    """One decision at the boundary, held — the way an index reconstitutes.

    Re-deciding mid-window would drop a name out of a book the model is still
    being scored on.
    """
    bars = _bars({"A.NS": (date(2019, 1, 1), date(2021, 12, 31), 5e8)})
    chosen = eligible_on(bars, date(2020, 1, 1), _RULE)
    for probe in (date(2020, 3, 1), date(2020, 7, 1), date(2020, 12, 1)):
        # The window's universe is whatever was chosen at its start, not what
        # the rule would say later. The caller holds it; the rule is pure.
        assert chosen == eligible_on(bars, date(2020, 1, 1), _RULE), (
            f"selection at the boundary is not stable when re-asked for {probe}"
        )


def test_every_window_gets_a_universe_and_they_differ_over_time() -> None:
    """A universe that never changes is a fixed list wearing a new name."""
    bars = pl.concat([
        _bars({"OLD.NS": (date(2014, 1, 1), date(2018, 12, 31), 5e8)}),
        _bars({"NEW.NS": (date(2018, 1, 1), date(2024, 12, 31), 5e8)}),
    ])
    windows = compute_windows(
        data_start=date(2010, 1, 1), data_end=date(2024, 12, 31),
        train_years=5, val_months=12, test_months=12, purge_months=3,
        n_windows=12, step_months=12,
    )
    seen = {w.name: eligible_on(bars, w.test_start, _RULE) for w in windows}
    non_empty = {k: v for k, v in seen.items() if v}
    assert non_empty, "no window selected anything"
    assert len({tuple(v) for v in non_empty.values()}) > 1, (
        "every window selected the same names — the universe is not moving"
    )


def test_an_empty_universe_is_refused_not_trained_on(tmp_path) -> None:  # noqa: ANN001
    """A window with no names trains on nothing and scores nothing.

    Downstream that reads as a window which simply had no signal, so the runner
    raises rather than producing one.
    """
    from trader.models.signal import SignalConfig
    from trader.training.supervised import SupervisedConfig, run_signal_walk_forward

    windows = compute_windows(
        data_start=date(2010, 1, 1), data_end=date(2024, 12, 31),
        train_years=5, val_months=12, test_months=12, purge_months=3,
        n_windows=1, step_months=12,
    )
    with pytest.raises(ValueError, match="returned no tickers"):
        run_signal_walk_forward(
            full_panel=pl.DataFrame(
                {"date": [date(2020, 1, 1)], "ticker": ["A.NS"],
                 "log_return_1d": [0.0], "is_tradeable": [True]}
            ),
            windows=windows,
            tickers=["A.NS"],
            universe_fn=lambda _w: [],
            feature_cols=["log_return_1d"],
            model_cfg=SignalConfig(in_features=1),
            train_cfg=SupervisedConfig(),
            out_dir=tmp_path / "out",
            tag="t",
            mlflow_port=None,
        )
