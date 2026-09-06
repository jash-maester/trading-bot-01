"""The environment must not execute trades the paper broker would drop.

`PanelTradingEnv` gated execution on a 0.5-*share* delta while
`PaperBroker._emit_target_orders` gated on `abs(delta) * open_px <
min_trade_value`. The two disagreed for months, which is the `min_trade_value: 500`
dead-key trap in CLAUDE.md and the 11% backtest/paper NAV divergence beside it.

It mattered because the demat debit fee is FLAT — Rs 15.34 per scrip per selling
day — so on a Rs 102 trade it is a 15% charge. Measured over the R4 OOS slice at
daily cadence, 94.5% of sell trades were under Rs 500 and the flat fee came to
129% of initial capital. See `11_cost_defect_and_fix_plan.md`.
"""
from __future__ import annotations

import tempfile
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from trader.data.features import FEATURE_COLS
from trader.env.costs import DEFAULT_MIN_TRADE_VALUE, ZerodhaEquityDeliveryCostModel
from trader.env.panel_env import PanelTradingEnv

_TICKERS = [f"T{i}.NS" for i in range(12)]
_N = len(_TICKERS)
_DAYS = 130


class _TradeProbe(ZerodhaEquityDeliveryCostModel):
    """Records the rupee value of every leg the env actually executes."""

    def __init__(self) -> None:
        super().__init__()
        self.values: list[float] = []

    def cost_vec(self, trade_values, is_buy, n_scrips_sold, *, intraday=None):  # type: ignore[no-untyped-def]
        self.values.extend(float(v) for v in trade_values[trade_values > 0.0])
        return super().cost_vec(trade_values, is_buy, n_scrips_sold, intraday=intraday)


def _panel() -> pl.DataFrame:
    rng = np.random.default_rng(7)
    rows: list[dict[str, object]] = []
    start = date(2015, 1, 1)
    for t_i, ticker in enumerate(_TICKERS):
        # A wide price spread is the point: cheap names produce the sub-Rs-500
        # dust trades, expensive ones clear the bar on a single share.
        price = float(10.0 * (t_i + 1))
        for i in range(_DAYS):
            price = max(1.0, price * (1.0 + rng.normal(0.0, 0.012)))
            row: dict[str, object] = {
                "date": start + timedelta(days=i),
                "ticker": ticker,
                "open": round(price * 0.999, 4),
                "high": round(price * 1.005, 4),
                "low": round(price * 0.995, 4),
                "close": round(price, 4),
                "adj_close": round(price, 4),
                "volume": int(rng.integers(500_000, 2_000_000)),
                "is_tradeable": True,
                "sector_id": 1,
                "atr_14": round(price * 0.015, 4),
                "dollar_volume_20": round(price * 800_000, 2),
            }
            for fc in FEATURE_COLS:
                row[fc] = float(rng.normal(0.0, 0.01))
            rows.append(row)
    return pl.DataFrame(rows)


@pytest.fixture(scope="module")
def panel_file() -> Path:
    path = Path(tempfile.mkdtemp()) / "train.parquet"
    _panel().write_parquet(path)
    return path


def _run(panel_file: Path, min_trade_value: float) -> _TradeProbe:
    """Drive an equal-weight book so drift forces many small rebalances."""
    probe = _TradeProbe()
    env = PanelTradingEnv(
        panel_path=panel_file,
        universe=_TICKERS,
        feature_columns=list(FEATURE_COLS),
        lookback=60,
        episode_length=60,
        initial_cash=1_000_000.0,
        cost_model=probe,
        min_trade_value=min_trade_value,
        seed=0,
    )
    env.reset(seed=0)
    w = np.full(_N + 1, 1.0 / _N, dtype=np.float64)
    w[0] = 0.0
    w = w / w.sum()
    done = False
    while not done:
        _, _, term, trunc, _ = env.step(w.copy())
        done = bool(term or trunc)
    return probe


def test_env_executes_no_trade_below_the_threshold(panel_file: Path) -> None:
    probe = _run(panel_file, DEFAULT_MIN_TRADE_VALUE)
    assert probe.values, "no trades at all — the test would be vacuous"
    below = [v for v in probe.values if v < DEFAULT_MIN_TRADE_VALUE]
    assert not below, (
        f"{len(below)} of {len(probe.values)} executed legs are under "
        f"Rs {DEFAULT_MIN_TRADE_VALUE}; smallest Rs {min(below):.2f}"
    )


def test_the_guard_is_live_and_this_test_is_not_inert(panel_file: Path) -> None:
    """With the guard off, sub-threshold trades MUST reappear.

    Without this, `test_env_executes_no_trade_below_the_threshold` would still
    pass if the panel simply never produced a small trade, and would keep
    passing if the guard were deleted.
    """
    off = _run(panel_file, 0.0)
    below = [v for v in off.values if v < DEFAULT_MIN_TRADE_VALUE]
    assert below, (
        "min_trade_value=0.0 produced no sub-threshold trade, so the guarded "
        "test proves nothing about the guard"
    )
    on = _run(panel_file, DEFAULT_MIN_TRADE_VALUE)
    assert len(on.values) < len(off.values)


def test_env_and_broker_gate_on_the_same_constant() -> None:
    """One constant, imported by both, so they cannot silently drift again."""
    from trader.broker import paper_broker

    assert paper_broker.DEFAULT_MIN_TRADE_VALUE is DEFAULT_MIN_TRADE_VALUE


@pytest.mark.parametrize("threshold", [0.0, 100.0, 500.0, 5_000.0])
def test_threshold_is_monotone_in_executed_trade_count(
    panel_file: Path, threshold: float
) -> None:
    """Raising the bar can only remove trades, never add them."""
    probe = _run(panel_file, threshold)
    assert all(v >= threshold for v in probe.values)


def test_negative_threshold_is_rejected(panel_file: Path) -> None:
    with pytest.raises(ValueError, match="min_trade_value"):
        PanelTradingEnv(
            panel_path=panel_file,
            universe=_TICKERS,
            feature_columns=list(FEATURE_COLS),
            lookback=60,
            episode_length=10,
            min_trade_value=-1.0,
        )


def test_suppressed_trade_leaves_the_position_and_the_cash_untouched(
    panel_file: Path,
) -> None:
    """A dropped order must move neither shares nor cash.

    `self._shares = target_shares` ran unconditionally while the cash leg was
    masked by `traded`, so a suppressed trade moved the position for free. The
    bug was dormant while the only guard was `>= 0.5` shares — true for every
    nonzero integer delta — and went live with the value guard, inverting the
    whole R5 allocator grid before it was caught.

    With the threshold above any reachable trade value, NOTHING may execute: the
    book stays empty, cash stays exactly at its opening balance, and NAV cannot
    drift, because an empty book has nothing to mark.
    """
    probe = _TradeProbe()
    env = PanelTradingEnv(
        panel_path=panel_file,
        universe=_TICKERS,
        feature_columns=list(FEATURE_COLS),
        lookback=60,
        episode_length=40,
        initial_cash=1_000_000.0,
        cost_model=probe,
        min_trade_value=1e12,
        seed=0,
    )
    env.reset(seed=0)
    w = np.full(_N + 1, 1.0 / _N, dtype=np.float64)
    w[0] = 0.0
    w = w / w.sum()
    done = False
    while not done:
        _, _, term, trunc, info = env.step(w.copy())
        done = bool(term or trunc)
        assert not probe.values, "a trade executed above an unreachable threshold"
        assert np.all(env._shares == 0.0), "shares moved with no trade"
        assert env._cash == pytest.approx(1_000_000.0), "cash moved with no trade"
        assert info["nav"] == pytest.approx(1_000_000.0), "NAV drifted on an empty book"
        assert info["turnover"] == pytest.approx(0.0)
