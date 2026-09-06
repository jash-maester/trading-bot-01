"""The env must not invent an order the allocator did not ask for.

`PanelTradingEnv` is handed **weights** and holds **shares**.  Turning one into
the other loses information twice:

* ``obs["portfolio"]`` is cast to **float32** (`panel_env.py:_build_obs`), so an
  allocator that pins a name to "its current weight" — which is what
  ``AllocatorParams.no_trade_band`` does — hands back a weight that reproduces
  the position only to ~6e-8 relative;
* ``target_shares = floor(target_value / open)`` (`panel_env.py:_step_target`)
  turns that dust into a **whole share**, roughly half the time, because
  ``floor`` of ``400 * (1 - 6e-8)`` is 399.

The resulting one-share SELL clears both existing guards — it is a full share,
so the 0.5-share integrality check passes, and it is worth the share price, so
``min_trade_value`` passes on any name above ₹500 (61.1% of
`data/panels_kite/test.parquet` by last close) — and it pays the flat ₹15.34
demat debit.  Measured before the fix: 25.6% of names the band had pinned still
traded, every one a sell, with the open exactly equal to the previous close.

The rule that stops it: **a request that rounds to the position already held is
not an order.**  ``floor`` still sets the *size* of a trade that does happen —
that is what stops a buy overspending its target value — but it no longer
decides *whether* one happens.  The rule only ever removes orders, never adds or
enlarges one, and it cannot block a liquidation: a full exit requests 0.0 shares
against a position of at least one.

Asserted here in the two regimes that generate a phantom order — float dust with
no price move at all, and an overnight gap against a weight marked at the
previous close — with a negative control that real trades still execute, and a
characterisation of what the rule does **not** fix: a gap worth more than half a
share is a genuine order, so the band's suppression does not survive into the
order book at every book geometry.  See `audit/P3_P4_P5_INTEGRATION.md`.
"""
from __future__ import annotations

import tempfile
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from trader.allocator import AllocatorParams, allocate
from trader.data.features import FEATURE_COLS
from trader.env.costs import ZerodhaEquityDeliveryCostModel
from trader.env.panel_env import PanelTradingEnv

# Prices span the range that matters: below ₹500 a one-share order is stopped by
# min_trade_value anyway, so every name here is above it and exposed.
_PRICES = (1342.0, 704.0, 2510.0, 998.0, 613.0, 1875.0)
_TICKERS = [f"P{i}.NS" for i in range(len(_PRICES))]
_N = len(_TICKERS)
_DAYS = 80
_LOOKBACK = 60


def _panel(gap: float = 0.0) -> pl.DataFrame:
    """A panel with **no price movement**, and an optional open-vs-close gap.

    ``gap == 0.0`` makes ``open`` identical to the previous close on every day,
    which removes every source of a genuine trade: a book that is already on its
    target weights has nothing to do.  Any order the env then emits is one it
    invented.  ``gap != 0.0`` opens every day that far off the previous close,
    which is the second regime (a weight marked at the previous close costs a
    real trade to hold across the gap).
    """
    rng = np.random.default_rng(11)
    rows: list[dict[str, object]] = []
    start = date(2015, 1, 1)
    for ticker, close in zip(_TICKERS, _PRICES, strict=True):
        for i in range(_DAYS):
            row: dict[str, object] = {
                "date": start + timedelta(days=i),
                "ticker": ticker,
                "open": close * (1.0 + gap),
                "high": close * 1.02,
                "low": close * 0.98,
                "close": close,
                "adj_close": close,
                "volume": 1_000_000,
                "is_tradeable": True,
                "sector_id": 1 + (_TICKERS.index(ticker) % 3),
                "atr_14": close * 0.015,
                "dollar_volume_20": close * 900_000.0,
            }
            for fc in FEATURE_COLS:
                row[fc] = float(rng.normal(0.0, 0.01))
            rows.append(row)
    return pl.DataFrame(rows)


class _LegProbe(ZerodhaEquityDeliveryCostModel):
    """Records every leg the env actually executes, and the flat fees paid."""

    def __init__(self) -> None:
        super().__init__()
        self.legs: list[float] = []
        self.n_sold: int = 0

    def cost_vec(self, trade_values, is_buy, n_scrips_sold, *, intraday=None):  # type: ignore[no-untyped-def]
        self.legs.extend(float(v) for v in trade_values[trade_values > 0.0])
        self.n_sold += int(np.asarray(n_scrips_sold).sum())
        return super().cost_vec(trade_values, is_buy, n_scrips_sold, intraday=intraday)


def _panel_file(gap: float) -> Path:
    path = Path(tempfile.mkdtemp()) / f"train_{gap!r}.parquet"
    _panel(gap).write_parquet(path)
    return path


@pytest.fixture(scope="module")
def flat_panel() -> Path:
    return _panel_file(0.0)


@pytest.fixture(scope="module")
def gapped_panel() -> Path:
    # +1.0% every open against the previous close: one daily standard deviation
    # for an Indian large cap, i.e. the ordinary case, not a corner.
    return _panel_file(0.01)


def _env(panel: Path, probe: _LegProbe, *, cash: float = 1_000_000.0) -> PanelTradingEnv:
    return PanelTradingEnv(
        panel_path=panel,
        universe=list(_TICKERS),
        feature_columns=list(FEATURE_COLS),
        lookback=_LOOKBACK,
        episode_length=_DAYS - _LOOKBACK - 1,
        initial_cash=cash,
        cost_model=probe,
        seed=0,
    )


def _pinned_run(panel: Path, *, band: float, cash: float = 1_000_000.0) -> _LegProbe:
    """Buy a book on day 0, then hand the allocator back its own book forever.

    With ``band`` above any deviation the allocator can produce, every name is
    pinned to its current weight from day 1 on, so days 1.. must be silent.
    """
    probe = _LegProbe()
    env = _env(panel, probe, cash=cash)
    obs, _ = env.reset(seed=0)

    r_hat = np.linspace(0.05, 0.01, _N)
    vol = np.full(_N, 0.25)
    sids = obs["sector_ids"].astype(np.int64)
    entry = AllocatorParams(k=_N, max_name_weight=0.30, max_sector_weight=1.0,
                            turnover_budget=2.0)
    obs, *_ = env.step_weights(
        allocate(r_hat, vol, obs["mask"].astype(bool), sids,
                 obs["portfolio"].astype(np.float64), entry)
    )
    probe.legs.clear()          # day 0 is the entry; it is meant to trade
    probe.n_sold = 0

    pinned = AllocatorParams(k=_N, max_name_weight=0.30, max_sector_weight=1.0,
                             turnover_budget=2.0, no_trade_band=band)
    done = False
    while not done:
        obs, _, term, trunc, _ = env.step_weights(
            allocate(r_hat, vol, obs["mask"].astype(bool), sids,
                     obs["portfolio"].astype(np.float64), pinned)
        )
        done = bool(term or trunc)
    return probe


def test_a_pinned_book_emits_no_order_when_nothing_moved(flat_panel: Path) -> None:
    """The claim P3's band rests on, asserted through the env rather than in weights.

    Open == previous close, no drift, every name pinned by a 0.5 band: the
    allocator's delta is the literal 0.0 for all six names.  Before the fix this
    still produced whole-share SELLs — the float32 ``obs["portfolio"]`` round
    trip through ``floor`` — and paid ₹15.34 for each.
    """
    probe = _pinned_run(flat_panel, band=0.5)
    assert not probe.legs, (
        f"{len(probe.legs)} orders on a book nobody asked to move: "
        f"{[round(v, 2) for v in probe.legs[:8]]}"
    )
    assert probe.n_sold == 0


def test_a_pinned_book_emits_no_order_across_a_sub_half_share_gap(
    gapped_panel: Path,
) -> None:
    """Same book, +1% open every day, at the R5 grid's position geometry.

    A weight is marked at the previous close and filled at the open, so pinning a
    name across a +1% gap asks the env to sell 1% of the position.  At ₹1 lakh
    over six names that is ₹166 against a cheapest share of ₹613 — under half a
    share, so it is not an executable order and must not become one.

    This is the geometry the standing R5 grid runs in: K=30 at ₹10 lakh holds
    ₹33k a name against a ₹704 median close (`data/panels_kite/test.parquet`),
    so a 1% gap is 0.47 of a share.
    """
    probe = _pinned_run(gapped_panel, band=0.5, cash=100_000.0)
    assert not probe.legs, (
        f"{len(probe.legs)} gap-driven orders on a pinned book: "
        f"{[round(v, 2) for v in probe.legs[:8]]}"
    )


def test_the_band_does_not_survive_a_gap_worth_more_than_half_a_share(
    gapped_panel: Path,
) -> None:
    """What the env fix does NOT do, pinned so nobody assumes otherwise.

    At ₹10 lakh over six names a position is ₹166k, so the same +1% gap asks for
    ₹1,660 — 1.2 shares of the cheapest name.  That is a real order and the env
    executes it, on a name ``no_trade_band`` reports as suppressed.  The band
    lives in weight space; orders are decided in share space at the open, and no
    change inside `panel_env.py` can close that gap because the env is never
    told which names the band pinned.  Recorded as a measurement, not softened
    into a claim that the band suppresses orders.
    """
    probe = _pinned_run(gapped_panel, band=0.5, cash=1_000_000.0)
    assert probe.legs, "the residual this test characterises has disappeared"
    assert probe.n_sold > 0, "the residual is sell-side — that is what is billed"


def test_a_real_move_still_trades(flat_panel: Path) -> None:
    """Negative control: the guard must not be a blanket 'never trade'.

    Without this, deleting the whole trading path would make the two tests above
    pass.  A rotation out of three names and into three others is worth far more
    than a share of any of them and must execute in full.
    """
    probe = _LegProbe()
    env = _env(flat_panel, probe)
    obs, _ = env.reset(seed=0)
    params = AllocatorParams(k=3, max_name_weight=0.40, max_sector_weight=1.0,
                             turnover_budget=2.0)
    vol = np.full(_N, 0.25)
    sids = obs["sector_ids"].astype(np.int64)

    first = np.array([0.06, 0.05, 0.04, 0.01, 0.01, 0.01])
    obs, *_ = env.step_weights(
        allocate(first, vol, obs["mask"].astype(bool), sids,
                 obs["portfolio"].astype(np.float64), params)
    )
    n_entry = len(probe.legs)
    assert n_entry == 3, f"entry should buy exactly 3 names, got {n_entry}"

    flipped = first[::-1].copy()
    obs, _, _, _, info = env.step_weights(
        allocate(flipped, vol, obs["mask"].astype(bool), sids,
                 obs["portfolio"].astype(np.float64), params)
    )
    assert len(probe.legs) >= n_entry + 6, "the rotation did not execute"
    assert info["turnover"] > 0.5


def test_a_full_exit_of_a_one_share_position_is_never_suppressed() -> None:
    """The guard is on the size of the MOVE, not on the size of the position.

    A ₹2,510 name held as a single share at ₹1 lakh of capital is a full share of
    intent when it is sold, so a naive "is the move worth less than one share?"
    test would refuse to liquidate it whenever the stock opened up.  Asserted
    against a +1% gap, which is exactly that case.
    """
    probe = _LegProbe()
    env = _env(_panel_file(0.01), probe, cash=100_000.0)
    obs, _ = env.reset(seed=0)
    sids = obs["sector_ids"].astype(np.int64)
    vol = np.full(_N, 0.25)
    params = AllocatorParams(k=_N, max_name_weight=0.30, max_sector_weight=1.0,
                             turnover_budget=2.0)
    obs, *_ = env.step_weights(
        allocate(np.linspace(0.05, 0.01, _N), vol, obs["mask"].astype(bool), sids,
                 obs["portfolio"].astype(np.float64), params)
    )
    held = env._shares.copy()
    assert held.min() >= 1.0, f"every name must hold at least one share: {held}"

    all_cash = np.zeros(_N + 1)
    all_cash[0] = 1.0
    probe.legs.clear()
    obs, _, _, _, _ = env.step_weights(all_cash)
    assert np.all(env._shares == 0.0), f"liquidation left {env._shares} behind"
    assert len(probe.legs) == _N
