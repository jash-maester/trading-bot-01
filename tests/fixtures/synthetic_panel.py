"""A synthetic feature panel that the real R4/R5/R6 code paths accept.

Committed on purpose.  The R4 smoke evidence used to live in an ephemeral
session scratchpad (``scratchpad/make_panel.py``), so every number reported
from it — the IC, the gate CI — cited nothing that survived the session.  The
project's verification standard requires a durable artefact behind a claim, so
the generator lives in the repo and the smoke run's ``gate.json`` is committed
under ``audit/``.

The panel is deliberately *small* and *fast*: it exists to prove that the
cross-stream seams execute end to end on the real code, not to say anything
about the model's skill.  Nothing here touches a GPU or a real panel.

Signal content is controllable.  ``signal_strength=0`` gives a pure-noise panel
whose forward returns are unpredictable; a positive value leaks a scaled copy
of the forward return into ``z_close_20``, which lets a seam test produce a
non-degenerate IC without training for long.  Neither is a claim about the
model — a test that asserts skill on a fixture it planted would be measuring
its own arithmetic.
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl

from trader.data.features import FEATURE_COLS

__all__ = ["make_signal_panel", "trading_dates"]


def trading_dates(n: int, start: date = date(2015, 1, 1)) -> list[date]:
    """`n` weekday dates from `start` — no holiday calendar, none needed."""
    out: list[date] = []
    day = start
    while len(out) < n:
        if day.weekday() < 5:
            out.append(day)
        day += timedelta(days=1)
    return out


def make_signal_panel(
    n_dates: int = 400,
    n_tickers: int = 24,
    seed: int = 0,
    *,
    signal_strength: float = 0.0,
    tickers: list[str] | None = None,
    start: date = date(2015, 1, 1),
    untradeable_fraction: float = 0.05,
) -> pl.DataFrame:
    """A panel carrying every column in ``FEATURE_COLS`` plus the env's needs.

    Every feature has real cross-sectional variance: a dead channel would trip
    R4's own feature-liveness gate, which is the point of that gate (see the
    ``beta_nifty_60d`` incident in CLAUDE.md).
    """
    rng = np.random.default_rng(seed)
    dates = trading_dates(n_dates, start)
    names = tickers if tickers is not None else [f"SYN{i:03d}.NS" for i in range(n_tickers)]
    n = len(names)
    T = len(dates)

    rets = rng.normal(0.0004, 0.015, size=(T, n))
    close = 100.0 * np.exp(np.cumsum(rets, axis=0))
    open_ = np.vstack([close[:1], close[:-1]])

    # Forward 5-day return, used only to plant a controllable signal.
    fwd = np.zeros((T, n))
    if signal_strength:
        for t in range(T - 5):
            fwd[t] = rets[t + 1 : t + 6].sum(axis=0)

    def noisy(scale: float) -> np.ndarray:
        return rng.normal(0.0, scale, size=(T, n))

    columns: dict[str, np.ndarray] = {
        "log_return_1d": rets,
        "log_return_5d": noisy(0.03),
        "log_return_20d": noisy(0.06),
        "realized_vol_20d": np.abs(noisy(0.05)) + 0.15,
        "realized_vol_60d": np.abs(noisy(0.05)) + 0.18,
        "rsi_14": rng.uniform(20.0, 80.0, size=(T, n)),
        "macd": noisy(0.5),
        "macd_signal": noisy(0.5),
        "macd_hist": noisy(0.3),
        "bbw_20": np.abs(noisy(0.02)) + 0.05,
        # The planted channel: pure noise at signal_strength=0.
        "z_close_20": noisy(1.0) + signal_strength * fwd / 0.03,
        "volume_z_20": noisy(1.0),
        "dollar_volume_20": np.abs(rng.normal(5e8, 1e8, size=(T, n))) + 1e7,
        "atr_14": np.abs(noisy(0.01)) + 0.5,
        "beta_nifty_60d": rng.normal(1.0, 0.3, size=(T, n)),
    }
    assert set(columns) == set(FEATURE_COLS), (
        f"fixture columns drifted from FEATURE_COLS: "
        f"missing {set(FEATURE_COLS) - set(columns)}, extra {set(columns) - set(FEATURE_COLS)}"
    )

    tradeable = rng.random((T, n)) >= untradeable_fraction

    rows: dict[str, list[object]] = {
        "date": [d for d in dates for _ in names],
        "ticker": [t for _ in dates for t in names],
        "open": open_.reshape(-1).tolist(),
        "close": close.reshape(-1).tolist(),
        "is_tradeable": tradeable.reshape(-1).tolist(),
        "sector_id": [1 + (i % 5) for _ in dates for i in range(n)],
    }
    for name, arr in columns.items():
        rows[name] = arr.reshape(-1).tolist()
    return pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Date))
