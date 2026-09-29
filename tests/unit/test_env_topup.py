"""PanelTradingEnv._topup: spend whole-share rounding leftovers, safely."""
from __future__ import annotations

import numpy as np

from trader.env.panel_env import PanelTradingEnv


def _env(cap: float = 0.10) -> PanelTradingEnv:
    e = PanelTradingEnv.__new__(PanelTradingEnv)      # bypass panel loading
    e._max_weight = cap
    return e


def test_topup_spends_leftover_within_caps_and_budget() -> None:
    e = _env()
    nav = 100_000.0
    w = np.array([0.30, 0.30, 0.30, 0.0])            # three names in book, one not
    opens = np.array([900.0, 4_800.0, 21_900.0, 10.0])
    floored = np.floor(nav * w / opens)               # 33, 6, 1 -> but cap applies next
    floored = np.minimum(floored, np.floor(0.10 * nav / opens))  # 11, 2, 0
    out = e._topup(floored, w, opens, nav)
    value = out * opens
    assert (value <= 0.10 * nav + 1e-9).all(), "10% per-name cap respected"
    assert value.sum() <= nav * (1 - PanelTradingEnv._TOPUP_RESERVE) + 1e-9, "cash reserve kept"
    assert out[3] == 0, "names outside the target are never bought"
    assert out[2] == 0, "a share above the cap (Rs 21,900 > Rs 10,000) is never bought"
    assert (out >= floored).all(), "top-up only adds"


def test_topup_is_deterministic() -> None:
    e = _env()
    w = np.full(30, 1 / 30)
    opens = np.linspace(100, 5_000, 30)
    a = e._topup(np.floor(1e5 * w / opens), w, opens, 1e5)
    b = e._topup(np.floor(1e5 * w / opens), w, opens, 1e5)
    np.testing.assert_array_equal(a, b)


def test_unpriced_names_are_skipped_without_nan() -> None:
    import warnings

    e = _env()
    w = np.array([0.5, 0.5])
    opens = np.array([0.0, 1_000.0])                  # first name has no open today
    with warnings.catch_warnings():
        warnings.simplefilter("error")                # any NaN/inf RuntimeWarning fails
        out = e._topup(np.zeros(2), w, opens, 100_000.0)
    assert out[0] == 0 and out[1] == 10               # capped at 10% = 10 shares
