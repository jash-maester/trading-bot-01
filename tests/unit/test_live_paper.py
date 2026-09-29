"""scripts/live_paper.py: sizing, volstop threshold, and fill accounting."""
from __future__ import annotations

import importlib.util
import math
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pytest

_P = Path(__file__).resolve().parents[2] / "scripts" / "live_paper.py"
_spec = importlib.util.spec_from_file_location("live_paper", _P)
lp = importlib.util.module_from_spec(_spec)
sys.modules["live_paper"] = lp
_spec.loader.exec_module(lp)


def test_stop_threshold_matches_risk_overlay():
    from trader.allocator.risk import RiskOverlay, RiskParams

    vol = np.array([0.10, 0.30, 0.60, 2.0, np.nan])
    ro = RiskOverlay(RiskParams(stop_vol_mult=1.0, stop_vol_horizon_days=20,
                                stop_cooldown_steps=21), n_names=len(vol))
    ro.update(1.0, np.ones(len(vol)), np.zeros(len(vol)), vol_ann=vol)
    expect = ro.stop_thresholds()
    got = np.array([lp.stop_threshold(float(v)) for v in vol])
    np.testing.assert_allclose(got, expect)


def test_plan_shares_keeps_positions_within_half_a_share():
    held = np.array([10.0, 5.0, 0.0])
    px = np.array([100.0, 200.0, 50.0])
    nav = 3000.0
    # name 0 asks for 10.3 shares: within 0.5 of 10 -> no order
    w = np.array([10.3 * 100 / nav, 0.0, 0.0])
    out = lp.plan_shares(held, w, px, nav)
    assert out[0] == 10 and out[1] == 0 and out[2] == 0


def test_plan_shares_respects_cap_and_reserve():
    # The allocator caps weights at MAX_W; the top-up must not push past it.
    px = np.array([30.0, 70.0, 0.0])
    nav = 1_000.0
    out = lp.plan_shares(np.zeros(3), np.array([0.1, 0.1, 0.0]), px, nav)
    assert np.all(out * px <= lp.MAX_W * nav + 1e-9)
    assert out[0] == 3 and out[1] == 1


def test_plan_shares_never_trades_an_unpriced_name():
    held = np.array([4.0, 0.0])
    out = lp.plan_shares(held, np.array([0.0, 0.1]), np.array([0.0, 100.0]), 1000.0)
    assert out[0] == 4


def _state():
    return {"cash": 1_000.0, "realised_rs": 0.0, "charges_rs": 0.0,
            "holdings": {"A.NS": {"qty": 10, "avg_price": 100.0, "entry_price": 100.0,
                                  "opened": "2026-09-30", "cost": 1_001.0,
                                  "sector": "X"}}}


def test_apply_fills_sell_realises_against_average_cost(monkeypatch):
    monkeypatch.setattr(lp, "_sectors", lambda: {})
    st = _state()
    fills = lp._apply_fills(st, [("A.NS", -10, 110.0)], {}, date(2026, 10, 1),
                            date(2026, 9, 30), "volstop")
    assert "A.NS" not in st["holdings"]
    f = fills[0]
    assert f["side"] == "SELL" and f["qty"] == 10
    assert math.isclose(f["realised_rs"], 1_100.0 - f["charges"] - 1_001.0, abs_tol=0.01)
    assert math.isclose(st["cash"], 1_000.0 + 1_100.0 - f["charges"], abs_tol=0.01)


def test_apply_fills_drops_small_orders_and_never_overdraws(monkeypatch):
    monkeypatch.setattr(lp, "_sectors", lambda: {})
    st = _state()
    fills = lp._apply_fills(st, [("B.NS", 4, 100.0),        # Rs 400 < 500: dropped
                                 ("C.NS", 50, 100.0)],      # Rs 5,000 > cash: trimmed
                            {}, date(2026, 10, 1), date(2026, 9, 30), "rebalance")
    assert "B.NS" not in st["holdings"]
    assert st["cash"] >= 0.0
    assert fills and all(f["ticker"] == "C.NS" for f in fills)
    assert st["holdings"]["C.NS"]["entry_price"] == 100.0


def test_apply_fills_add_keeps_entry_price(monkeypatch):
    monkeypatch.setattr(lp, "_sectors", lambda: {})
    st = _state()
    lp._apply_fills(st, [("A.NS", 5, 120.0)], {}, date(2026, 10, 1), date(2026, 9, 30),
                    "rebalance")
    h = st["holdings"]["A.NS"]
    assert h["qty"] == 15 and h["entry_price"] == 100.0
    assert math.isclose(h["avg_price"], (1000 + 600) / 15, rel_tol=1e-3)


@pytest.mark.parametrize("today,days,expect", [
    ("2026-09-30", {"2026-09-30"}, True), ("2026-10-02", {"2026-10-01"}, False)])
def test_session_open(today, days, expect):
    px = {f"T{i}": {"last_trade": f"{d} 09:15:03"} for i, d in enumerate(days)}
    assert lp._session_open(px, date.fromisoformat(today)) is expect
