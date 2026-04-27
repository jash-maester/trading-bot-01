"""Unit tests for the M7 walk-forward window logic."""
from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl
import pytest


def test_compute_windows_canonical() -> None:
    """The 4-window protocol from 05_training.md §Walk-forward."""
    from trader.training.walk_forward import compute_windows

    windows = compute_windows(
        data_start=date(2014, 1, 1),
        data_end=date(2024, 12, 31),
        train_years=5,
        val_months=12,
        test_months=12,
        purge_months=1,
        n_windows=4,
        step_months=12,
    )
    assert len(windows) == 4
    assert [w.name for w in windows] == ["W1", "W2", "W3", "W4"]
    # W1: train 2014-01..2018-12, val 2020-02..2021-01, test 2022-03..2023-02
    assert windows[0].train_start == date(2014, 1, 1)
    assert windows[0].train_end == date(2018, 12, 31)
    # W2..W4 must each advance the train_start by 12 months
    for prev, nxt in zip(windows[:-1], windows[1:]):
        assert (nxt.train_start.year, nxt.train_start.month) == (
            prev.train_start.year + 1,
            prev.train_start.month,
        )


def test_compute_windows_skips_when_test_overflows() -> None:
    """Last window must not extend past data_end."""
    from trader.training.walk_forward import compute_windows

    # data ends well before any 4th window's test segment can fit
    windows = compute_windows(
        data_start=date(2014, 1, 1),
        data_end=date(2021, 12, 31),
        train_years=5,
        val_months=12,
        test_months=12,
        n_windows=10,
    )
    # No window should have test_end > data_end
    assert all(w.test_end <= date(2021, 12, 31) for w in windows)


def test_compute_windows_purge_gaps() -> None:
    """Each segment is separated by `purge_months` from the next."""
    from trader.training.walk_forward import compute_windows

    windows = compute_windows(
        data_start=date(2014, 1, 1),
        data_end=date(2024, 12, 31),
        train_years=5, val_months=12, test_months=12,
        purge_months=1,
        n_windows=1,
    )
    w = windows[0]
    # train_end → val_start gap ≥ ~1 month
    gap_train_val = (w.val_start - w.train_end).days
    gap_val_test = (w.test_start - w.val_end).days
    assert 28 <= gap_train_val <= 32
    assert 28 <= gap_val_test <= 32


def test_slice_panel_inclusive_bounds() -> None:
    """Boundary dates are included; rows outside [start, end] are excluded."""
    from trader.training.walk_forward import slice_panel

    df = pl.DataFrame(
        {
            "date":   [date(2020, 1, 1), date(2020, 6, 15), date(2020, 12, 31)],
            "ticker": ["A", "A", "A"],
            "value":  [1.0, 2.0, 3.0],
        }
    )
    out = slice_panel(df, date(2020, 1, 1), date(2020, 12, 31))
    assert out.shape[0] == 3
    out2 = slice_panel(df, date(2020, 6, 1), date(2020, 6, 30))
    assert out2.shape[0] == 1
    assert out2["value"][0] == 2.0


def test_paired_bootstrap_ci_significance() -> None:
    """Reliably positive RL > baseline → CI strictly above zero."""
    from trader.training.walk_forward import paired_bootstrap_ci

    rl = [1.0, 1.1, 1.2, 0.9, 1.05, 1.15, 0.95, 1.0]
    bl = [0.0] * len(rl)
    res = paired_bootstrap_ci(rl, bl, n_boot=2000, rng_seed=0)
    assert res["ci_lo"] > 0.5
    assert res["ci_hi"] < 1.5
    assert res["p_above_zero"] > 0.99


def test_paired_bootstrap_ci_no_signal() -> None:
    """No signal → CI straddles zero."""
    from trader.training.walk_forward import paired_bootstrap_ci

    rng = np.random.default_rng(0)
    rl = rng.normal(0, 1, size=20).tolist()
    bl = rng.normal(0, 1, size=20).tolist()
    res = paired_bootstrap_ci(rl, bl, n_boot=2000, rng_seed=0)
    assert res["ci_lo"] < 0 < res["ci_hi"]


def test_aggregate_walk_forward_mean_std() -> None:
    """Aggregator pulls per-run metrics correctly."""
    from trader.training.walk_forward import aggregate_walk_forward

    results = [
        {"window": "W1", "seed": 42, "test": {"mean_sharpe": 0.5}},
        {"window": "W1", "seed": 43, "test": {"mean_sharpe": 0.7}},
        {"window": "W2", "seed": 42, "test": {"mean_sharpe": 0.3}},
        {"window": "W2", "seed": 43, "test": {"mean_sharpe": 0.9}},
    ]
    agg = aggregate_walk_forward(results, metric="mean_sharpe", split="test")
    assert agg["n"] == 4
    assert agg["mean"] == pytest.approx(0.6)
    assert agg["min"] == pytest.approx(0.3)
    assert agg["max"] == pytest.approx(0.9)


def test_materialise_window_writes_three_parquets(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """End-to-end: slicing + parquet I/O for a small synthetic panel."""
    from trader.training.walk_forward import (
        WindowConfig,
        materialise_window,
    )

    # Build a tiny panel that covers all three segments
    dates = [date(2014, 1, 1) + __import__("datetime").timedelta(days=i)
             for i in range(0, 3650, 30)]
    df = pl.DataFrame(
        {
            "date":   dates * 2,
            "ticker": ["A"] * len(dates) + ["B"] * len(dates),
            "value":  list(range(2 * len(dates))),
        }
    ).sort("date")

    win = WindowConfig(
        name="W1",
        train_start=date(2014, 1, 1), train_end=date(2018, 12, 31),
        val_start=date(2019, 2, 1), val_end=date(2019, 12, 31),
        test_start=date(2020, 2, 1), test_end=date(2020, 12, 31),
    )
    paths = materialise_window(df, win, tmp_path / "W1")
    assert set(paths.keys()) == {"train", "val", "test"}
    for p in paths.values():
        assert p.exists()
        sub = pl.read_parquet(p)
        assert sub.shape[0] > 0
