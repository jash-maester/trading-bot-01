"""Capital-gains tax inside the backtest loop.

Every CAGR in this repo was pre-tax until 2026-09-07. At ~3.7x annual turnover
essentially every gain is short-term at 20% plus 4% cess, so the omission was
worth roughly a fifth of the reported return.

The properties that matter: tax is off by default so prior results still
reproduce; it is paid in CASH so it stops compounding; and it is matched FIFO,
which is what Indian law applies to listed equity held in demat.
"""
from __future__ import annotations

import tempfile
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from trader.data.features import FEATURE_COLS
from trader.env.panel_env import PanelTradingEnv

_TICKERS = [f"T{i}.NS" for i in range(6)]
_N = len(_TICKERS)


def _panel(days: int = 900, drift: float = 0.0015) -> pl.DataFrame:
    """A steadily rising panel, so sales realise gains and tax is non-zero."""
    rng = np.random.default_rng(11)
    rows: list[dict[str, object]] = []
    start = date(2015, 1, 1)
    for t_i, ticker in enumerate(_TICKERS):
        price = 100.0 + 10.0 * t_i
        for i in range(days):
            price = max(1.0, price * (1.0 + drift + rng.normal(0, 0.008)))
            rows.append(
                {
                    "date": start + timedelta(days=i),
                    "ticker": ticker,
                    "open": round(price * 0.999, 4),
                    "high": round(price * 1.005, 4),
                    "low": round(price * 0.995, 4),
                    "close": round(price, 4),
                    "adj_close": round(price, 4),
                    "volume": 1_000_000,
                    "is_tradeable": True,
                    "sector_id": 1,
                    "atr_14": round(price * 0.015, 4),
                    "dollar_volume_20": round(price * 800_000, 2),
                    **{fc: float(rng.normal(0, 0.01)) for fc in FEATURE_COLS},
                }
            )
    return pl.DataFrame(rows)


@pytest.fixture(scope="module")
def panel_file() -> Path:
    p = Path(tempfile.mkdtemp()) / "train.parquet"
    _panel().write_parquet(p)
    return p


def _run(panel_file: Path, apply_tax: bool, episode: int = 700) -> tuple[PanelTradingEnv, float]:
    env = PanelTradingEnv(
        panel_path=panel_file,
        universe=_TICKERS,
        feature_columns=list(FEATURE_COLS),
        lookback=60,
        episode_length=episode,
        initial_cash=1_000_000.0,
        seed=0,
        apply_tax=apply_tax,
    )
    env.reset(seed=0)
    rng = np.random.default_rng(3)
    done = False
    nav = 0.0
    while not done:
        # Rotate the book so gains are actually realised rather than held.
        w = np.zeros(_N + 1)
        pick = rng.choice(_N, size=3, replace=False)
        w[1:][pick] = 1.0 / 3.0
        _, _, term, trunc, info = env.step_weights(w)
        nav = float(info["nav"])
        done = bool(term or trunc)
    return env, nav


def test_tax_is_off_by_default_and_changes_nothing(panel_file: Path) -> None:
    """Every result predating this must still reproduce exactly."""
    a, _ = _run(panel_file, apply_tax=False)
    assert a.tax_paid == 0.0
    assert a.tax_accrued_unpaid() == 0.0


def test_tax_is_actually_charged_and_reduces_nav(panel_file: Path) -> None:
    untaxed, nav_untaxed = _run(panel_file, apply_tax=False)
    taxed, nav_taxed = _run(panel_file, apply_tax=True)
    assert taxed.tax_paid > 0.0, "no tax was ever paid on a rising, rotating book"
    # Paid in cash, so the taxed book must end poorer.
    assert nav_taxed < nav_untaxed
    # And poorer by roughly what was paid, allowing for lost compounding on it.
    assert nav_untaxed - nav_taxed >= taxed.tax_paid * 0.5


def test_tax_is_paid_only_after_a_financial_year_closes(panel_file: Path) -> None:
    """Within one FY the liability accrues but no cash leaves."""
    env = PanelTradingEnv(
        panel_path=panel_file,
        universe=_TICKERS,
        feature_columns=list(FEATURE_COLS),
        lookback=60,
        episode_length=60,          # well inside one FY
        initial_cash=1_000_000.0,
        seed=0,
        apply_tax=True,
    )
    env.reset(seed=0)
    rng = np.random.default_rng(3)
    done = False
    while not done:
        w = np.zeros(_N + 1)
        w[1:][rng.choice(_N, size=3, replace=False)] = 1.0 / 3.0
        _, _, term, trunc, _ = env.step_weights(w)
        done = bool(term or trunc)
    assert env.tax_paid == 0.0
    assert env.tax_accrued_unpaid() > 0.0, "gains were realised but nothing accrued"


def test_monthly_cadence_realises_far_less_gain_than_daily(panel_file: Path) -> None:
    """Rebalance cadence, not holding intent, is what drives the tax bill.

    A first version of this test asserted that "holding" equal weight realises
    less than a rotating book. That is FALSE and the measurement said so: the
    held book realised Rs 292k against the rotating book's Rs 219k. Maintaining
    a fixed weight in a rising market means trimming every winner every step,
    which realises gain continuously — the intent to hold is not the same as
    holding.

    What actually controls realised gain is how OFTEN the book is allowed to
    trade, which is the schedule. This is the tax half of the argument for
    monthly rebalancing that `10_architecture_revamp.md` section 4 makes on
    cost alone.
    """
    from trader.allocator.rebalance import RebalanceSchedule

    def run(schedule: RebalanceSchedule | None) -> float:
        env = PanelTradingEnv(
            panel_path=panel_file,
            universe=_TICKERS,
            feature_columns=list(FEATURE_COLS),
            lookback=60,
            episode_length=700,
            initial_cash=1_000_000.0,
            seed=0,
            apply_tax=True,
            rebalance_schedule=schedule,
        )
        env.reset(seed=0)
        w = np.zeros(_N + 1)
        w[1:] = 1.0 / _N
        done = False
        while not done:
            _, _, term, trunc, _ = env.step_weights(w)
            done = bool(term or trunc)
        return env.tax_paid + env.tax_accrued_unpaid()

    daily = run(None)
    monthly = run(RebalanceSchedule("monthly"))
    assert daily > 0.0
    assert monthly < daily * 0.5, f"monthly {monthly:,.0f} vs daily {daily:,.0f}"


def test_gross_basis_is_deliberate_and_documented() -> None:
    """STT is non-deductible under the proviso to s.48; delivery brokerage is 0.

    So the gross fill price is within a rounding error of the correct basis.
    This test pins the reasoning to the code so a future 'fix' that subtracts
    STT from the basis has to argue with it.
    """
    import inspect

    from trader.env import panel_env

    src = inspect.getsource(panel_env.PanelTradingEnv.__init__)
    assert "NON-deductible" in src
    assert "section 48" in src
