"""The random books must not change when the grid does."""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np

from trader.allocator.null_signal import stable_null_signal

D0 = date(2024, 6, 3)
DATES = [D0 + timedelta(days=i) for i in range(30)]
UNI = ["AAA.NS", "BBB.NS", "CCC.NS"]


def test_appending_a_session_keeps_history() -> None:
    a = stable_null_signal(DATES[:-1], UNI, seed=7)
    b = stable_null_signal(DATES, UNI, seed=7)
    np.testing.assert_array_equal(a, b[:-1])


def test_adding_a_ticker_keeps_the_others() -> None:
    a = stable_null_signal(DATES, UNI, seed=7)
    b = stable_null_signal(DATES, UNI + ["DDD.NS"], seed=7)
    np.testing.assert_array_equal(a, b[:, :3])


def test_moving_the_panel_start_keeps_the_cells() -> None:
    a = stable_null_signal(DATES, UNI, seed=7)
    b = stable_null_signal(DATES[10:], UNI, seed=7)
    np.testing.assert_array_equal(a[10:], b)


def test_seeds_and_tickers_differ() -> None:
    a = stable_null_signal(DATES, UNI, seed=1)
    b = stable_null_signal(DATES, UNI, seed=2)
    assert not np.allclose(a, b)
    assert not np.allclose(a[:, 0], a[:, 1])


def test_support_mask_applies() -> None:
    sup = np.ones((len(DATES), len(UNI)))
    sup[0, 0] = np.nan
    out = stable_null_signal(DATES, UNI, seed=3, support=sup)
    assert np.isnan(out[0, 0]) and np.isfinite(out[1, 0])


def test_pre_anchor_dates_are_stable_and_leave_post_anchor_cells_unchanged():
    """2011-24 tests (audit/P7) need pre-2020 cells; adding them must not move
    a single post-anchor cell, or every forward record would re-roll."""
    from datetime import date as _d

    pre = [_d(2011, 7, 4), _d(2015, 3, 2), _d(2019, 12, 31)]
    post = [_d(2020, 1, 1), _d(2024, 7, 2), _d(2026, 10, 16)]
    both = stable_null_signal(pre + post, UNI, seed=7)
    only_post = stable_null_signal(post, UNI, seed=7)
    np.testing.assert_array_equal(both[3:], only_post)
    np.testing.assert_array_equal(both[:3], stable_null_signal(pre, UNI, seed=7))
    assert np.isfinite(both).all()
    assert not np.array_equal(both[2], both[3])
