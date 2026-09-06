"""The test segment's feature-context prefix: used, but never scored.

`materialise_window` prepends `lookback - 1` trading days to the test segment so
the first prediction lands on the first real OOS date. Without it every OOS
segment lost its opening 59 days — 472 of 1980 days (24%) on the shipped
8-window configuration.

The rows come from the purge gap, carry no labels, and are never scored. The
danger is entirely one-directional: if a warm-up row ever reached a label or a
score, this would stop being lost power and become contamination. Every test
here exists to pin that down.
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fixtures.synthetic_panel import make_signal_panel  # noqa: E402

from trader.data.features import FEATURE_COLS  # noqa: E402
from trader.training.supervised import build_panel_tensors  # noqa: E402
from trader.training.walk_forward import WindowConfig, materialise_window  # noqa: E402

_LOOKBACK = 20
_WARMUP = _LOOKBACK - 1


def _panel() -> pl.DataFrame:
    return make_signal_panel(n_dates=400, n_tickers=12, seed=3)


def _window(panel: pl.DataFrame) -> WindowConfig:
    d = sorted(panel["date"].unique().to_list())
    return WindowConfig(
        name="W1",
        train_start=d[0],
        train_end=d[199],
        val_start=d[210],
        val_end=d[299],
        test_start=d[340],      # 40 trading days after val_end: room for warm-up
        test_end=d[-1],
    )


def _materialise(tmp_path: Path, warmup: int) -> tuple[pl.DataFrame, WindowConfig]:
    panel = _panel()
    w = _window(panel)
    paths = materialise_window(panel, w, tmp_path, warmup_days=warmup)
    return pl.read_parquet(paths["test"]), w


def test_warmup_days_are_prepended_and_marked(tmp_path: Path) -> None:
    test_df, w = _materialise(tmp_path, _WARMUP)
    warm = test_df.filter(pl.col("is_warmup"))
    assert warm["date"].n_unique() == _WARMUP
    assert warm["date"].max() < w.test_start
    real = test_df.filter(~pl.col("is_warmup"))
    assert real["date"].min() == w.test_start


def test_warmup_rows_carry_no_label(tmp_path: Path) -> None:
    test_df, _ = _materialise(tmp_path, _WARMUP)
    tickers = sorted(test_df["ticker"].unique().to_list())
    t = build_panel_tensors(test_df, tickers, list(FEATURE_COLS), horizons=(5,))
    assert t.n_warmup == _WARMUP
    assert not np.isfinite(t.targets[5][:_WARMUP]).any(), "a warm-up row was labelled"
    assert not np.isfinite(t.fwd_raw[5][:_WARMUP]).any()
    assert np.isfinite(t.targets[5][_WARMUP:]).any(), "no real labels — test is vacuous"


def test_no_prediction_lands_on_a_warmup_date(tmp_path: Path) -> None:
    from trader.training.supervised import _predictable_days, _scorable_days

    test_df, _ = _materialise(tmp_path, _WARMUP)
    tickers = sorted(test_df["ticker"].unique().to_list())
    t = build_panel_tensors(test_df, tickers, list(FEATURE_COLS), horizons=(5,))
    for idx in (_predictable_days(t, _LOOKBACK), _scorable_days(t, _LOOKBACK)):
        assert idx.size > 0
        assert idx.min() >= t.n_warmup, "a warm-up day is predictable or scorable"


def test_the_warmup_actually_buys_predictable_days(tmp_path: Path) -> None:
    """The whole point: without it, the first `lookback - 1` OOS days are lost."""
    from trader.training.supervised import _predictable_days

    with_warm, w = _materialise(tmp_path / "with", _WARMUP)
    without, _ = _materialise(tmp_path / "without", 0)
    tickers = sorted(with_warm["ticker"].unique().to_list())

    a = build_panel_tensors(with_warm, tickers, list(FEATURE_COLS), horizons=(5,))
    b = build_panel_tensors(without, tickers, list(FEATURE_COLS), horizons=(5,))
    pa = _predictable_days(a, _LOOKBACK)
    pb = _predictable_days(b, _LOOKBACK)

    first_real_a = a.dates[int(pa.min())]
    first_real_b = b.dates[int(pb.min())]
    assert first_real_a == w.test_start, "warm-up did not reach the first OOS date"
    assert first_real_b > w.test_start
    assert pa.size == pb.size + _WARMUP


def test_warmup_never_reaches_into_validation(tmp_path: Path) -> None:
    """The one-directional danger, asserted rather than trusted.

    A purge too short for the full warm-up is clamped, not failed: losing
    predictable days is the problem the warm-up exists to solve, so it is no
    reason to abort a run. What may never happen is context reaching back past
    `val_end`, because that is contamination rather than lost power.
    """
    panel = _panel()
    d = sorted(panel["date"].unique().to_list())
    tight = WindowConfig(
        name="W1",
        train_start=d[0],
        train_end=d[199],
        val_start=d[210],
        val_end=d[299],
        test_start=d[305],      # only 5 trading days of gap
        test_end=d[-1],
    )
    paths = materialise_window(panel, tight, tmp_path, warmup_days=_WARMUP)
    test_df = pl.read_parquet(paths["test"])
    warm = test_df.filter(pl.col("is_warmup"))
    assert warm["date"].min() > tight.val_end, "context reached into validation"
    assert warm["date"].max() < tight.test_start
    assert warm["date"].n_unique() < _WARMUP, "clamp did not bind"


def test_zero_warmup_is_the_old_behaviour(tmp_path: Path) -> None:
    test_df, w = _materialise(tmp_path, 0)
    assert not test_df["is_warmup"].any()
    assert test_df["date"].min() == w.test_start


def test_non_contiguous_warmup_marks_are_refused() -> None:
    panel = _panel()
    dates = sorted(panel["date"].unique().to_list())
    scattered = panel.with_columns(
        pl.col("date").is_in([dates[0], dates[5]]).alias("is_warmup")
    )
    tickers = sorted(panel["ticker"].unique().to_list())
    with pytest.raises(ValueError, match="contiguous prefix"):
        build_panel_tensors(scattered, tickers, list(FEATURE_COLS), horizons=(5,))


def _predict_first_oos(test_df: pl.DataFrame, tickers: list[str]) -> np.ndarray:
    """Prediction vector on the first predictable date, from a fixed model."""
    import torch

    from trader.models.signal import SignalConfig, SignalModel
    from trader.training.supervised import _predictable_days, predict_panel

    t = build_panel_tensors(test_df, tickers, list(FEATURE_COLS), horizons=(5,))
    torch.manual_seed(0)
    model = SignalModel(
        SignalConfig(
            in_features=len(FEATURE_COLS), embed_dim=8, num_channels=[8, 8],
            kernel_size=3, dropout=0.0, head_hidden=4, horizons=(5,),
        )
    )
    model.eval()
    preds = predict_panel(
        model, t, lookback=_LOOKBACK, device=torch.device("cpu"), batch_days=8
    )
    first = int(_predictable_days(t, _LOOKBACK).min())
    return np.asarray(preds[5][first], dtype=np.float64)


def _bump(df: pl.DataFrame, on: date, col: str = "z_close_20") -> pl.DataFrame:
    return df.with_columns(
        pl.when(pl.col("date") == on)
        .then(pl.col(col) + 5.0)
        .otherwise(pl.col(col))
        .alias(col)
    )


def test_a_warmup_row_actually_feeds_the_first_prediction(tmp_path: Path) -> None:
    """If perturbing warm-up changed nothing, the extra days would be noise."""
    test_df, _ = _materialise(tmp_path, _WARMUP)
    tickers = sorted(test_df["ticker"].unique().to_list())
    warm_dates = sorted(
        test_df.filter(pl.col("is_warmup"))["date"].unique().to_list()
    )

    base = _predict_first_oos(test_df, tickers)
    bumped = _predict_first_oos(_bump(test_df, warm_dates[-1]), tickers)
    # Exact, not allclose: the shift from one perturbed day through an untrained
    # encoder is small, and a tolerance would let a genuinely inert context pass.
    assert not np.array_equal(base, bumped, equal_nan=True), (
        "perturbing the last warm-up day left the first OOS prediction "
        "unchanged — the context is not reaching the encoder"
    )


def test_the_future_still_cannot_change_a_past_prediction(tmp_path: Path) -> None:
    """The leakage guard the warm-up must not have weakened."""
    from trader.training.supervised import _predictable_days

    test_df, _ = _materialise(tmp_path, _WARMUP)
    tickers = sorted(test_df["ticker"].unique().to_list())
    t = build_panel_tensors(test_df, tickers, list(FEATURE_COLS), horizons=(5,))
    dates = t.dates
    first = int(_predictable_days(t, _LOOKBACK).min())

    base = _predict_first_oos(test_df, tickers)
    for offset in (1, 5, 20):
        future = dates[first + offset]
        moved = _predict_first_oos(_bump(test_df, future), tickers)
        # equal_nan: untradeable names carry NaN, and NaN != NaN would make this
        # assertion fail on an unchanged vector.
        assert np.array_equal(base, moved, equal_nan=True), (
            f"perturbing {future}, {offset} day(s) after the prediction date "
            f"{dates[first]}, changed that prediction"
        )
