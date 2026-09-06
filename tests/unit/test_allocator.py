"""Unit tests for the R5 deterministic allocator and rebalance schedule.

Each test names the failure mode it exists to catch.  None of these touch a
panel or the env; `tests/integration/test_panel_env.py` covers the env side.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from trader.allocator import AllocatorParams, RebalanceSchedule, allocate
from trader.env.costs import DEFAULT_MIN_TRADE_VALUE

# ── fixtures ──────────────────────────────────────────────────────────────────

_N = 12


def _inputs(
    n: int = _N,
    seed: int = 0,
    n_sectors: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    r_hat = rng.normal(0.0, 1.0, n)
    vol = rng.uniform(0.15, 0.45, n)          # annualised, like realized_vol_20d
    mask = np.ones(n, dtype=bool)
    sector_ids = (np.arange(n) % n_sectors) + 1
    current_w = np.zeros(n + 1)
    current_w[0] = 1.0                        # all cash
    return r_hat, vol, mask, sector_ids, current_w


def _check_basic(w: np.ndarray, n: int = _N) -> None:
    assert w.shape == (n + 1,)
    assert np.all(w >= 0.0)
    assert w.sum() == pytest.approx(1.0, abs=1e-9)


# ── shape / sum / sign ────────────────────────────────────────────────────────


def test_weights_sum_to_one_nonnegative_cash_first() -> None:
    r_hat, vol, mask, sids, cur = _inputs()
    p = AllocatorParams(k=5, max_name_weight=0.5, max_sector_weight=1.0, turnover_budget=2.0)
    w = allocate(r_hat, vol, mask, sids, cur, p)
    _check_basic(w)
    # Fully investable with these loose caps: cash is exactly what the caps leave.
    assert w[0] == pytest.approx(0.0, abs=1e-9)


def test_exactly_k_names_when_k_candidates_exist() -> None:
    r_hat, vol, mask, sids, cur = _inputs()
    p = AllocatorParams(k=5, max_name_weight=1.0, max_sector_weight=1.0, turnover_budget=2.0)
    w = allocate(r_hat, vol, mask, sids, cur, p)
    assert int((w[1:] > 0).sum()) == 5
    # and they are the top-5 by r_hat
    top5 = set(np.argsort(-r_hat)[:5].tolist())
    assert set(np.flatnonzero(w[1:] > 0).tolist()) == top5


def test_fewer_than_k_when_fewer_tradeable() -> None:
    r_hat, vol, mask, sids, cur = _inputs()
    mask[:] = False
    mask[[1, 4, 7]] = True
    p = AllocatorParams(k=5, max_name_weight=1.0, max_sector_weight=1.0, turnover_budget=2.0)
    w = allocate(r_hat, vol, mask, sids, cur, p)
    _check_basic(w)
    assert set(np.flatnonzero(w[1:] > 0).tolist()) == {1, 4, 7}


def test_inverse_vol_sizing_before_caps() -> None:
    r_hat, vol, mask, sids, cur = _inputs()
    p = AllocatorParams(k=3, max_name_weight=1.0, max_sector_weight=1.0, turnover_budget=2.0)
    w = allocate(r_hat, vol, mask, sids, cur, p)
    chosen = np.flatnonzero(w[1:] > 0)
    inv = 1.0 / vol[chosen]
    np.testing.assert_allclose(w[1:][chosen], inv / inv.sum(), atol=1e-12)


# ── candidate filtering ───────────────────────────────────────────────────────


def test_nan_r_hat_and_zero_vol_never_selected() -> None:
    r_hat, vol, mask, sids, cur = _inputs()
    r_hat[0] = 10.0
    r_hat[1] = np.nan          # best score but no prediction
    vol[0] = 0.0               # best score but dead vol channel
    r_hat[2] = 9.0
    vol[2] = np.inf            # non-finite vol
    p = AllocatorParams(k=3, max_name_weight=1.0, max_sector_weight=1.0, turnover_budget=2.0)
    w = allocate(r_hat, vol, mask, sids, cur, p)
    _check_basic(w)
    assert w[1] == 0.0 and w[2] == 0.0 and w[3] == 0.0
    assert int((w[1:] > 0).sum()) == 3


def test_sector_zero_is_never_selected() -> None:
    r_hat, vol, mask, sids, cur = _inputs()
    r_hat[3] = 100.0
    sids[3] = 0
    p = AllocatorParams(k=3, max_name_weight=1.0, max_sector_weight=1.0, turnover_budget=2.0)
    w = allocate(r_hat, vol, mask, sids, cur, p)
    assert w[4] == 0.0


def test_no_candidates_goes_all_cash() -> None:
    r_hat, vol, mask, sids, cur = _inputs()
    mask[:] = False
    w = allocate(r_hat, vol, mask, sids, cur, AllocatorParams(turnover_budget=2.0))
    _check_basic(w)
    assert w[0] == pytest.approx(1.0)


# ── caps ──────────────────────────────────────────────────────────────────────


def test_name_cap_binds_and_excess_is_redistributed_not_dropped() -> None:
    r_hat, vol, mask, sids, cur = _inputs()
    vol[:] = 0.3
    vol[0] = 0.01              # would take ~75% of the book uncapped
    r_hat[:5] = 5.0            # force names 0..4 into the top-5
    p = AllocatorParams(k=5, max_name_weight=0.30, max_sector_weight=1.0, turnover_budget=2.0)
    w = allocate(r_hat, vol, mask, sids, cur, p)
    _check_basic(w)
    assert w[1] == pytest.approx(0.30, abs=1e-9)
    assert np.all(w[1:] <= 0.30 + 1e-9)
    # 5 names × 0.30 = 1.5 ≥ 1, so the excess fits: nothing dumped in cash.
    assert w[0] == pytest.approx(0.0, abs=1e-9)
    assert w[1:6].sum() == pytest.approx(1.0, abs=1e-9)


def test_name_cap_leaves_cash_only_when_everything_is_capped() -> None:
    r_hat, vol, mask, sids, cur = _inputs()
    # k * cap = 5 * 0.10 = 0.5 < 1: the book cannot be fully invested.
    # NOTE: max_sector_weight is the PINNED DEFAULT 0.25, not 1.0.  This test
    # drives the exact path that used to breach the sector cap (the name cap
    # freezes every chosen name), and the previous version switched the sector
    # cap off with max_sector_weight=1.0 — which is what hid the breach.
    # The 5 chosen names spread <= 2 per sector here, so 0.25 does not bind and
    # the assertions below are unchanged; test_sector_cap_binds_when_name_cap
    # _freezes_every_name covers the case where it does.
    p = AllocatorParams(k=5, max_name_weight=0.10, max_sector_weight=0.25, turnover_budget=2.0)
    w = allocate(r_hat, vol, mask, sids, cur, p)
    _check_basic(w)
    assert np.all(w[1:] <= 0.10 + 1e-9)
    assert np.all(np.bincount(sids, weights=w[1:], minlength=4) <= 0.25 + 1e-9)
    assert w[0] == pytest.approx(0.5, abs=1e-9)
    assert int(np.isclose(w[1:], 0.10).sum()) == 5


def test_sector_cap_binds_when_name_cap_freezes_every_name() -> None:
    """The regression test for the 3.2x sector breach.

    At the PINNED DEFAULTS the name cap froze the last free names and the loop
    then took its `b_sum == 0` break, so the sector cap was never evaluated for
    them.  Measured before the fix: sector 1 = 0.80 against a 0.25 cap.
    """
    n = 8
    sids = np.ones(n, dtype=np.int64)          # every candidate in one sector
    p = AllocatorParams(turnover_budget=10.0)  # pinned k=30 / 0.10 / 0.25
    w = allocate(
        np.linspace(1.0, 0.0, n), np.full(n, 0.2), np.ones(n, dtype=bool),
        sids, np.r_[1.0, np.zeros(n)], p,
    )
    _check_basic(w, n=n)
    sec_w = np.bincount(sids, weights=w[1:], minlength=2)
    assert sec_w[1] == pytest.approx(0.25, abs=1e-9), sec_w      # was 0.80
    assert w[0] == pytest.approx(0.75, abs=1e-9)


def test_sector_cap_binds_on_the_last_free_names_after_another_sector_froze() -> None:
    """Round 0 freezes sector 2; round 1's last 3 names hit the name cap.

    The verifier's exact repro: 23 candidates split 3 / 20 across two sectors
    returned sector 1 = 0.30 against a 0.25 cap before the fix.
    """
    n = 23
    sids = np.array([1, 1, 1] + [2] * 20, dtype=np.int64)
    p = AllocatorParams(turnover_budget=10.0)
    w = allocate(
        np.linspace(1.0, 0.0, n), np.full(n, 0.2), np.ones(n, dtype=bool),
        sids, np.r_[1.0, np.zeros(n)], p,
    )
    _check_basic(w, n=n)
    sec_w = np.bincount(sids, weights=w[1:], minlength=3)
    assert sec_w[1] == pytest.approx(0.25, abs=1e-9), sec_w      # was 0.30
    assert np.all(sec_w[1:] <= 0.25 + 1e-9)


def test_sector_cap_holds_when_allocate_is_iterated_on_its_own_output() -> None:
    """The breach compounded through the operating loop; the budget only delayed it.

    Feeding `allocate` its own output back as `current_w` for 30 rebalances at
    the pinned defaults (turnover_budget=0.30 included) converged to sector 1 =
    0.80 before the fix.
    """
    n = 8
    sids = np.ones(n, dtype=np.int64)
    p = AllocatorParams()                       # pinned defaults, budget 0.30
    r_hat, vol, mask = np.linspace(1.0, 0.0, n), np.full(n, 0.2), np.ones(n, dtype=bool)
    w = np.r_[1.0, np.zeros(n)]
    for step in range(30):
        w = allocate(r_hat, vol, mask, sids, w, p)
        _check_basic(w, n=n)
        sec_w = np.bincount(sids, weights=w[1:], minlength=2)
        assert sec_w[1] <= 0.25 + 1e-9, (step, sec_w)


def test_sector_cap_binds() -> None:
    r_hat, vol, mask, sids, cur = _inputs()
    vol[:] = 0.3
    # Put the whole top-6 in sector 1 except two names in sector 2.
    sids[:] = 3
    sids[:6] = 1
    sids[[6, 7]] = 2
    r_hat[:8] = 5.0
    r_hat[8:] = -5.0
    p = AllocatorParams(k=8, max_name_weight=0.5, max_sector_weight=0.40, turnover_budget=2.0)
    w = allocate(r_hat, vol, mask, sids, cur, p)
    _check_basic(w)
    sec_w = np.bincount(sids, weights=w[1:], minlength=4)
    assert sec_w[1] == pytest.approx(0.40, abs=1e-9)
    assert np.all(sec_w[1:] <= 0.40 + 1e-9)
    # Sector 2 (two names, cap 0.40 total) absorbs some; both caps hold;
    # the rest (1 − 0.40 − 0.40 = 0.20) is cash because every chosen name is
    # inside a capped sector.
    assert sec_w[2] == pytest.approx(0.40, abs=1e-9)
    assert w[0] == pytest.approx(0.20, abs=1e-9)


def test_name_and_sector_caps_hold_simultaneously() -> None:
    """Randomised cap fuzz.

    The previous version pinned k=30 over 40 names with near-uniform vol, a
    regime where 1/30 = 0.033 never trips the 0.10 name cap — so it exercised
    the sector branch only, and found 0 breaches where a wider draw found 455
    in 5000.  This version samples k, both caps and the sector sizes, and
    deliberately includes `k * name_cap < 1` and single-sector draws.
    """
    rng = np.random.default_rng(3)
    breaches = []
    for draw in range(2000):
        n = int(rng.integers(5, 60))
        n_sectors = int(rng.integers(1, 6))          # 1 => every name one sector
        sids = rng.integers(1, n_sectors + 1, size=n).astype(np.int64)
        mask = rng.random(n) > 0.2
        p = AllocatorParams(
            k=int(rng.integers(3, 40)),
            max_name_weight=float(rng.uniform(0.03, 0.5)),   # k*cap < 1 reachable
            max_sector_weight=float(rng.uniform(0.10, 0.60)),
            turnover_budget=1e9,                             # isolate the caps
        )
        w = allocate(
            rng.normal(size=n), rng.uniform(0.05, 0.6, n), mask,
            sids, np.r_[1.0, np.zeros(n)], p,
        )
        _check_basic(w, n=n)
        sec_w = np.bincount(sids, weights=w[1:], minlength=n_sectors + 1)
        if w[1:].max() > p.max_name_weight + 1e-9 or sec_w[1:].max() > p.max_sector_weight + 1e-9:
            breaches.append((draw, float(w[1:].max()), float(sec_w[1:].max()), p))
    assert not breaches, breaches[:5]


def test_caps_hold_on_a_sector_tilted_signal_over_the_real_universe_shape() -> None:
    """A cross-sectional model tilts toward sectors; white noise does not.

    On the real universe's sector-size distribution a flat signal never
    breached, but a sector-tilted one breached 55/500 days at up to 2x the cap.
    That asymmetry is why the old fuzz (uniform signal) saw nothing.
    """
    rng = np.random.default_rng(11)
    sizes = [121, 99, 70, 48, 45, 42, 36, 24, 19]          # real sector sizes
    sids = np.repeat(np.arange(1, len(sizes) + 1), sizes).astype(np.int64)
    n = int(sids.size)
    vol = rng.uniform(0.15, 0.60, n)
    p = AllocatorParams()                                   # pinned defaults
    worst = 0.0
    for _ in range(200):
        tilt = rng.integers(1, len(sizes) + 1)
        r_hat = rng.normal(size=n) + 3.0 * (sids == tilt)   # tilt toward one sector
        mask = rng.random(n) > 0.7                          # ~150 tradeable/day
        w = allocate(r_hat, vol, mask, sids, np.r_[1.0, np.zeros(n)], p)
        _check_basic(w, n=n)
        sec_w = np.bincount(sids, weights=w[1:], minlength=len(sizes) + 1)
        worst = max(worst, float(sec_w[1:].max()))
    assert worst <= 0.25 + 1e-9, worst


def test_cash_floor_is_reserved() -> None:
    r_hat, vol, mask, sids, cur = _inputs()
    p = AllocatorParams(
        k=6, max_name_weight=1.0, max_sector_weight=1.0, turnover_budget=2.0, cash_floor=0.15
    )
    w = allocate(r_hat, vol, mask, sids, cur, p)
    _check_basic(w)
    assert w[0] == pytest.approx(0.15, abs=1e-9)


# ── turnover budget ───────────────────────────────────────────────────────────


def _unconstrained(
    r_hat: np.ndarray, vol: np.ndarray, mask: np.ndarray, sids: np.ndarray, p: AllocatorParams
) -> np.ndarray:
    """The target the allocator would return with no turnover budget."""
    cur = np.zeros(len(r_hat) + 1)
    cur[0] = 1.0
    free = AllocatorParams(
        k=p.k, max_name_weight=p.max_name_weight, max_sector_weight=p.max_sector_weight,
        turnover_budget=2.0, cash_floor=p.cash_floor,
    )
    return allocate(r_hat, vol, mask, sids, cur, free)


def test_turnover_budget_binds_exactly_on_a_signal_flip() -> None:
    r_hat, vol, mask, sids, _ = _inputs()
    p = AllocatorParams(k=4, max_name_weight=0.5, max_sector_weight=1.0, turnover_budget=0.30)
    # Book currently holds the *bottom* 4; the signal now says hold the top 4.
    current_w = _unconstrained(-r_hat, vol, mask, sids, p)
    target = _unconstrained(r_hat, vol, mask, sids, p)
    assert set(np.flatnonzero(current_w[1:] > 0)).isdisjoint(np.flatnonzero(target[1:] > 0))
    full_gross = float(np.abs(target[1:] - current_w[1:]).sum())
    assert full_gross == pytest.approx(2.0, abs=1e-9)   # a full rotation

    w = allocate(r_hat, vol, mask, sids, current_w, p)
    _check_basic(w)
    gross = float(np.abs(w[1:] - current_w[1:]).sum())
    assert gross <= p.turnover_budget + 1e-9
    assert gross == pytest.approx(p.turnover_budget, abs=1e-9)   # binds exactly

    # w lies on the segment between current_w and the unconstrained target.
    lam = p.turnover_budget / full_gross
    expected = current_w + lam * (target - current_w)
    np.testing.assert_allclose(w, expected, atol=1e-12)
    # and every equity entry is between the two endpoints
    lo = np.minimum(current_w, target)
    hi = np.maximum(current_w, target)
    assert np.all(w >= lo - 1e-12) and np.all(w <= hi + 1e-12)


def test_turnover_budget_does_not_bind_when_move_is_small() -> None:
    r_hat, vol, mask, sids, _ = _inputs()
    p = AllocatorParams(k=4, max_name_weight=0.5, max_sector_weight=1.0, turnover_budget=0.30)
    target = _unconstrained(r_hat, vol, mask, sids, p)
    # Start 5% of the way from the target: gross move is 0.05 × 1.0 < budget.
    cash = np.zeros_like(target)
    cash[0] = 1.0
    current_w = 0.95 * target + 0.05 * cash
    w = allocate(r_hat, vol, mask, sids, current_w, p)
    np.testing.assert_allclose(w, target, atol=1e-12)


def test_turnover_budget_from_all_cash_entry() -> None:
    r_hat, vol, mask, sids, cur = _inputs()
    p = AllocatorParams(k=4, max_name_weight=0.5, max_sector_weight=1.0, turnover_budget=0.30)
    w = allocate(r_hat, vol, mask, sids, cur, p)
    _check_basic(w)
    # Entering from cash: gross = equity bought = 0.30, cash = 0.70.
    assert w[1:].sum() == pytest.approx(0.30, abs=1e-9)
    assert w[0] == pytest.approx(0.70, abs=1e-9)


def test_current_w_float32_does_not_leak_into_cash() -> None:
    r_hat, vol, mask, sids, _ = _inputs()
    p = AllocatorParams(k=4, max_name_weight=0.5, max_sector_weight=1.0, turnover_budget=0.30)
    target = _unconstrained(r_hat, vol, mask, sids, p)
    w = allocate(r_hat, vol, mask, sids, target.astype(np.float32), p)
    _check_basic(w)
    np.testing.assert_allclose(w, target, atol=1e-6)


# ── no-trade band (P3) ────────────────────────────────────────────────────────
#
# `AllocatorParams.turnover_budget` constrains gross RUPEES moved.  The demat
# debit bills ₹15.34 per distinct scrip per selling day, flat (`costs.py:55`),
# which is 83-98% of all cost in every backtest measured
# (`11_cost_defect_and_fix_plan.md`).  The band is the only control in the
# system on the quantity that fee is billed on: the number of names traded.


def _pre_band_finish(
    w_eq: np.ndarray, current_w: np.ndarray, budget: float
) -> np.ndarray:
    """`deterministic._finish` transcribed from the commit before the band.

    The reference for "band 0.0 changes nothing".  Deliberately a copy rather
    than a call into the module: a reference that imports the code under test
    proves nothing.
    """
    cur_eq = np.clip(current_w[1:], 0.0, None)
    cur_total = float(cur_eq.sum()) + max(float(current_w[0]), 0.0)
    if cur_total > 1e-12:
        cur_eq = cur_eq / cur_total
    delta = w_eq - cur_eq
    gross = float(np.abs(delta).sum())
    if gross > budget + 1e-12 and gross > 1e-12:
        w_eq = cur_eq + (budget / gross) * delta
    w_eq = np.maximum(w_eq, 0.0)
    eq_sum = float(w_eq.sum())
    if eq_sum > 1.0:
        w_eq = w_eq / eq_sum
        eq_sum = 1.0
    out = np.empty(w_eq.shape[0] + 1, dtype=np.float64)
    out[0] = 1.0 - eq_sum
    out[1:] = w_eq
    return out


def _target_equity(
    r_hat: np.ndarray, vol: np.ndarray, mask: np.ndarray, sids: np.ndarray,
    p: AllocatorParams,
) -> np.ndarray:
    """The pre-turnover, pre-band equity target: `allocate` from all cash with
    a budget nothing can exceed, which returns the capped weights untouched."""
    cur = np.zeros(len(r_hat) + 1)
    cur[0] = 1.0
    free = AllocatorParams(
        k=p.k, max_name_weight=p.max_name_weight, max_sector_weight=p.max_sector_weight,
        turnover_budget=1e9, cash_floor=p.cash_floor,
    )
    return allocate(r_hat, vol, mask, sids, cur, free)[1:]


def _norm_book(current_w: np.ndarray) -> np.ndarray:
    """The incoming book as `_current_equity` sees it.

    `allocate` renormalises `current_w` before doing anything with it — the env
    emits a `portfolio` that sums to 1.0098 when its cash goes negative and the
    Box(0, 1) clip hides it — so "pinned to its current weight exactly" is a
    statement about THIS vector.  The renormalisation predates the band and is
    shared with the band-0 path.
    """
    cur_eq = np.clip(current_w[1:], 0.0, None)
    total = float(cur_eq.sum()) + max(float(current_w[0]), 0.0)
    if total > 1e-12:
        cur_eq = cur_eq / total
    return np.asarray(cur_eq, dtype=np.float64)


def _deltas(w: np.ndarray, current_w: np.ndarray) -> np.ndarray:
    """Equity move the env would be asked to execute, against the renormalised book."""
    return np.asarray(w[1:] - _norm_book(current_w), dtype=np.float64)


def test_band_defaults_to_zero() -> None:
    assert AllocatorParams().no_trade_band == 0.0


def test_band_zero_is_bit_identical_to_the_pre_band_path() -> None:
    """Requirement 1: default 0.0 is TODAY'S behaviour exactly, not approximately.

    The band lives in `_finish`, and `_finish` is the only function this stream
    touched, so the claim is tested where it is made: the same equity target and
    the same book through both, with `==` rather than `approx`.  1000 draws,
    with the book anywhere from all cash to fully invested in the wrong names,
    and including the `portfolio`-sums-to-1.01 shape the env really emits.  One
    ulp of difference fails this.
    """
    from trader.allocator.deterministic import _finish

    rng = np.random.default_rng(17)
    for _ in range(1000):
        n = int(rng.integers(6, 40))
        sids = rng.integers(1, 5, size=n).astype(np.int64)
        vol = rng.uniform(0.1, 0.6, n)
        mask = rng.random(n) > 0.2
        p = AllocatorParams(
            k=int(rng.integers(3, 20)),
            max_name_weight=float(rng.uniform(0.05, 0.5)),
            max_sector_weight=float(rng.uniform(0.15, 0.6)),
            turnover_budget=float(rng.uniform(0.05, 1.0)),
        )
        assert p.no_trade_band == 0.0
        target = _target_equity(rng.normal(size=n), vol, mask, sids, p)
        cur = np.zeros(n + 1)
        cur[1:] = _target_equity(rng.normal(size=n), vol, mask, sids, p)
        cur[0] = max(0.0, 1.0 - float(cur[1:].sum()))
        cur *= float(rng.choice([1.0, 1.0, 1.01]))
        got = _finish(target.copy(), cur, p)
        want = _pre_band_finish(target.copy(), cur, p.turnover_budget)
        assert np.array_equal(got, want), float(np.abs(got - want).max())


def test_band_zero_leaves_the_whole_allocator_unchanged() -> None:
    """The same claim end to end, one step weaker than bit-identity.

    `allocate` is compared against the pre-band finishing step fed the same
    target.  The tolerance is 1e-15 rather than 0 because the harness has to
    reconstruct the target through a second `allocate` call, which renormalises
    on its own when the caps happen to sum to 1 + 1 ulp — a property of the
    reconstruction, not of the code path.  The exact claim is the test above.
    """
    rng = np.random.default_rng(29)
    for _ in range(300):
        n = int(rng.integers(6, 40))
        sids = rng.integers(1, 5, size=n).astype(np.int64)
        vol = rng.uniform(0.1, 0.6, n)
        r_hat = rng.normal(size=n)
        mask = rng.random(n) > 0.2
        p = AllocatorParams(
            k=int(rng.integers(3, 20)),
            max_name_weight=float(rng.uniform(0.05, 0.5)),
            max_sector_weight=float(rng.uniform(0.15, 0.6)),
            turnover_budget=float(rng.uniform(0.05, 1.0)),
        )
        cur = np.zeros(n + 1)
        cur[1:] = _target_equity(rng.normal(size=n), vol, mask, sids, p)
        cur[0] = max(0.0, 1.0 - float(cur[1:].sum()))
        w = allocate(r_hat, vol, mask, sids, cur, p)
        want = _pre_band_finish(_target_equity(r_hat, vol, mask, sids, p), cur,
                                p.turnover_budget)
        np.testing.assert_allclose(w, want, rtol=0.0, atol=1e-15)


def test_band_pins_a_suppressed_name_to_its_current_weight_exactly() -> None:
    """Requirement 3: the delta must be exactly 0.0, not merely small.

    A shrunken delta still produces an order, and an order still pays ₹15.34.
    """
    r_hat, vol, mask, sids, _ = _inputs()
    p0 = AllocatorParams(k=5, max_name_weight=0.5, max_sector_weight=1.0, turnover_budget=2.0)
    target = _target_equity(r_hat, vol, mask, sids, p0)
    held = np.flatnonzero(target > 0)
    # The book sits ON the target except for two names 0.002 either side of it
    # and two more 0.0005 either side.  Sign-balanced, so the book still sums
    # to 1 and `_current_equity` does not renormalise the deviations away.
    cur_eq = target.copy()
    cur_eq[held[0]] += 0.002
    cur_eq[held[1]] -= 0.002
    cur_eq[held[2]] += 0.0005
    cur_eq[held[3]] -= 0.0005
    cur = np.r_[max(0.0, 1.0 - cur_eq.sum()), cur_eq]

    banded = AllocatorParams(
        k=5, max_name_weight=0.5, max_sector_weight=1.0, turnover_budget=2.0,
        no_trade_band=0.005,
    )
    w = allocate(r_hat, vol, mask, sids, cur, banded)
    _check_basic(w)
    # Every deviation is inside the band, so NOTHING trades: every weight is
    # bit-identical to the book we came in with and every delta is exactly 0.
    assert np.array_equal(w[1:], _norm_book(cur))
    assert np.array_equal(_deltas(w, cur), np.zeros(len(cur) - 1))

    # Drop the band between the two deviation sizes: the 0.002 pair trades and
    # the 0.0005 pair does not.  The second pair is the one that matters here —
    # it has a real, nonzero deviation and must still come out EXACTLY at its
    # current weight.  A rule that shrank it instead of pinning it (say to
    # 1e-9 of the deviation) would pass every `approx` assertion in this file
    # and still send two orders and pay two ₹15.34 debits.
    narrow = AllocatorParams(
        k=5, max_name_weight=0.5, max_sector_weight=1.0, turnover_budget=2.0,
        no_trade_band=0.001,
    )
    w2 = allocate(r_hat, vol, mask, sids, cur, narrow)
    d2 = _deltas(w2, cur)
    assert set(np.flatnonzero(d2 != 0.0).tolist()) == {int(held[0]), int(held[1])}
    assert d2[held[0]] == pytest.approx(-0.002, abs=1e-9)
    assert d2[held[1]] == pytest.approx(+0.002, abs=1e-9)
    others = [int(i) for i in held.tolist() if i not in (held[0], held[1])]
    assert np.array_equal(d2[others], np.zeros(len(others)))
    assert np.array_equal(w2[1:][others], _norm_book(cur)[others])


def test_band_still_trades_a_name_that_has_drifted_past_it() -> None:
    """One name 10x the band above its target, one 10x below: both trade, alone.

    The perturbation is sign-balanced so the incoming book still sums to 1 and
    the sell funds the buy; a book summing to 1.05 would be renormalised by
    `_current_equity` and the deviation would no longer be the 0.05 asserted.
    """
    r_hat, vol, mask, sids, _ = _inputs()
    p = AllocatorParams(k=5, max_name_weight=0.5, max_sector_weight=1.0, turnover_budget=2.0,
                        no_trade_band=0.005)
    target = _target_equity(r_hat, vol, mask, sids, p)
    held = np.flatnonzero(target > 0)
    cur_eq = target.copy()
    cur_eq[held[0]] += 0.05                      # 10x the band, to be sold back
    cur_eq[held[1]] -= 0.05                      # 10x the band, to be bought back
    cur = np.r_[max(0.0, 1.0 - cur_eq.sum()), cur_eq]
    assert cur.sum() == pytest.approx(1.0, abs=1e-12)
    w = allocate(r_hat, vol, mask, sids, cur, p)
    _check_basic(w)
    d = _deltas(w, cur)
    assert d[held[0]] == pytest.approx(-0.05, abs=1e-9)
    assert d[held[1]] == pytest.approx(+0.05, abs=1e-9)
    assert int((d != 0.0).sum()) == 2


def test_every_sell_clears_the_band_even_when_the_turnover_budget_binds() -> None:
    """The postcondition that a single band-then-budget pass does NOT satisfy.

    A full rotation is gross 2.0 against a 0.30 budget, so lambda = 0.15 and a
    naive band-then-budget pass moves each 0.0333 name by 0.005 — well under
    the band it just cleared, which is the trade the band refused, reinstated
    by the budget one line later.  Here every executed sell must clear the band.

    The band is 0.008 rather than 0.005 on purpose: at 0.005 the naive rule
    lands exactly ON the band (0.0333 x 0.15 = 0.005) and whether the assert
    fires is decided by the last ulp, so the test would pass against the broken
    rule by luck.  Verified: with the band tested on the raw deviation instead
    of the executed move, this fails at 0.008 and passes at 0.005.
    """
    n, k = 60, 30
    sids = (np.arange(n) % 5 + 1).astype(np.int64)
    vol = np.full(n, 0.25)
    mask = np.ones(n, dtype=bool)
    r_hat = np.linspace(1.0, -1.0, n)
    p = AllocatorParams(k=k, max_name_weight=0.10, max_sector_weight=1.0,
                        turnover_budget=0.30, no_trade_band=0.008)
    cur_eq = _target_equity(-r_hat, vol, mask, sids, p)      # holds the bottom 30
    cur = np.r_[max(0.0, 1.0 - cur_eq.sum()), cur_eq]
    full = float(np.abs(_target_equity(r_hat, vol, mask, sids, p) - cur_eq).sum())
    assert full == pytest.approx(2.0, abs=1e-9)              # a full rotation

    w = allocate(r_hat, vol, mask, sids, cur, p)
    _check_basic(w, n=n)
    d = _deltas(w, cur)
    sells = d[d < 0.0]
    assert sells.size > 0, "the book must not freeze"
    assert float(np.abs(sells).min()) > p.no_trade_band
    assert float(np.abs(d).sum()) <= p.turnover_budget + 1e-9


def test_band_does_not_freeze_a_book_it_cannot_rotate_in_one_step() -> None:
    """The failure mode of iterating band-then-budget to a fixed point.

    On a symmetric rotation every leg moves the same distance, so a rule that
    drops every under-band name each round drops them all and the book never
    trades again.  The joint rule keeps ~budget/band legs instead.
    """
    n, k = 60, 30
    sids = (np.arange(n) % 5 + 1).astype(np.int64)
    vol, mask = np.full(n, 0.25), np.ones(n, dtype=bool)
    r_hat = np.linspace(1.0, -1.0, n)
    # 0.011, not 0.010: at 0.010 the cut falls exactly on 0.30 / 2 / 15, and a
    # test that turns on the last ulp of a tie is a flaky test, not a stricter one.
    p = AllocatorParams(k=k, max_name_weight=0.10, max_sector_weight=1.0,
                        turnover_budget=0.30, no_trade_band=0.011)
    cur_eq = _target_equity(-r_hat, vol, mask, sids, p)
    cur = np.r_[max(0.0, 1.0 - cur_eq.sum()), cur_eq]
    w = allocate(r_hat, vol, mask, sids, cur, p)
    d = _deltas(w, cur)
    n_traded = int((d != 0.0).sum())
    # 13 legs a side at 0.30 / 2 / 13 = 0.0115 each: the arithmetic in
    # `_apply_no_trade_band`'s docstring.  A batch-drop fixed point returns 0,
    # and so does a single-ranking rule (it selects 29 buys it cannot fund).
    assert n_traded == 26, n_traded
    assert int((d < 0).sum()) == 13 and int((d > 0).sum()) == 13
    assert float(np.abs(d[d != 0.0]).min()) == pytest.approx(0.30 / 2 / 13, rel=1e-6)
    assert float(np.abs(d).sum()) == pytest.approx(p.turnover_budget, abs=1e-9)


def test_band_never_overspends_cash_when_it_suppresses_the_sells() -> None:
    """Suppressing small sells while taking the buys would need cash never raised.

    Without the funding constraint this book returns an equity sum above 1,
    i.e. negative cash — and the `eq_sum > 1` guard in `_finish` would then
    rescale EVERY name, including the suppressed ones, handing the env an order
    for each.  The invariant is `sum == 1` with cash >= 0 and the suppressed
    names still pinned exactly.
    """
    n = 40
    sids = np.ones(n, dtype=np.int64)
    mask = np.ones(n, dtype=bool)
    p = AllocatorParams(k=25, max_name_weight=1.0, max_sector_weight=1.0,
                        turnover_budget=2.0, no_trade_band=0.004)
    # Inverse-vol weights are exactly 1/vol normalised, so the target is
    # constructed by choosing vols: 20 held names at 0.046 and 5 new at 0.016.
    vol = np.empty(n)
    vol[:20] = 1.0 / 0.046
    vol[20:25] = 1.0 / 0.016
    vol[25:] = 1.0
    r_hat = np.where(np.arange(n) < 25, 1.0, 0.0)
    target = _target_equity(r_hat, vol, mask, sids, p)
    np.testing.assert_allclose(target[:20], 0.046, atol=1e-12)
    np.testing.assert_allclose(target[20:25], 0.016, atol=1e-12)

    # The book holds those 20 names at 0.048 with 4% idle cash.  The target
    # trims each by 0.002 — inside the band — and buys 5 names at 0.016.  Taking
    # the buys without the trims needs 0.08 and only 0.04 is available.
    cur_eq = np.zeros(n)
    cur_eq[:20] = 0.048
    cur = np.r_[1.0 - cur_eq.sum(), cur_eq]
    assert cur.sum() == pytest.approx(1.0, abs=1e-12)

    w = allocate(r_hat, vol, mask, sids, cur, p)
    _check_basic(w, n=n)
    assert w[0] >= 0.0
    d = _deltas(w, cur)
    # The trims are inside the band and must be pinned exactly ...
    assert np.array_equal(w[1:][:20], cur_eq[:20])
    # ... and the 5 buys are funded by the 0.04 of cash, not by 0.08 of nothing:
    # half size each, and the equity sum still lands on 1.
    np.testing.assert_allclose(d[20:25], 0.008, atol=1e-9)
    assert float(d[d > 0].sum()) == pytest.approx(0.04, abs=1e-9)
    assert float(w[1:].sum()) == pytest.approx(1.0, abs=1e-9)
    # Without the funding cap the same book returns 0.96 + 0.08 = 1.04 of equity.
    assert float(cur_eq.sum() + 0.08) > 1.0


def test_band_invariants_hold_under_fuzz() -> None:
    """Requirement 4 under the band: sums to 1, non-negative, gross <= budget,
    every sell clears the band, suppressed names pinned exactly."""
    rng = np.random.default_rng(23)
    for draw in range(600):
        n = int(rng.integers(8, 60))
        sids = rng.integers(1, 6, size=n).astype(np.int64)
        vol = rng.uniform(0.1, 0.6, n)
        mask = rng.random(n) > 0.2
        p = AllocatorParams(
            k=int(rng.integers(3, 30)),
            max_name_weight=float(rng.uniform(0.05, 0.4)),
            max_sector_weight=float(rng.uniform(0.15, 0.6)),
            turnover_budget=float(rng.uniform(0.05, 2.0)),
            no_trade_band=float(rng.choice([0.001, 0.002, 0.005, 0.01, 0.02])),
        )
        cur = np.zeros(n + 1)
        cur[1:] = _target_equity(rng.normal(size=n), vol, mask, sids, p)
        cur[0] = max(0.0, 1.0 - float(cur[1:].sum()))
        w = allocate(rng.normal(size=n), vol, mask, sids, cur, p)
        _check_basic(w, n=n)
        d = _deltas(w, cur)
        assert float(np.abs(d).sum()) <= p.turnover_budget + 1e-9, draw
        sells = np.abs(d[d < 0.0])
        if sells.size:
            assert float(sells.min()) > p.no_trade_band, (draw, float(sells.min()))


def test_clearing_prefix_is_a_prefix_not_a_maximum_subset() -> None:
    """The docstring's own limitation, regenerated rather than asserted from memory.

    ``_clearing_prefix`` keeps the largest feasible PREFIX of the descending
    sort.  The docstring used to claim that was "the *most* legs the budget can
    move" and that it "never over-suppresses"; both are false, because dropping
    one large leg frees budget for several small ones.  This test regenerates
    the counterexample and the frequency now quoted there, so the claim and the
    code cannot drift apart again (CLAUDE.md rule 4 — spec and code disagreeing
    inside one docstring).

    Reproduce::

        uv run pytest -q tests/unit/test_allocator.py -k clearing_prefix_is_a_prefix
    """
    import itertools

    from trader.allocator.deterministic import _clearing_prefix

    def best_feasible(dev: np.ndarray, budget: float, band: float) -> int:
        best = 0
        for r in range(1, dev.size + 1):
            for combo in itertools.combinations(range(dev.size), r):
                idx = list(combo)
                gross = float(dev[idx].sum())
                s = min(1.0, budget / gross)
                if bool((s * dev[idx] > band).all()):
                    best = max(best, r)
        return best

    dev = np.array([0.0333, 0.0333, 0.0333, 0.010, 0.010, 0.010])
    keep, scale = _clearing_prefix(dev, 0.05, 0.005)
    assert int(keep.sum()) == 3
    assert scale == pytest.approx(0.500501, abs=1e-6)
    assert best_feasible(dev, 0.05, 0.005) == 5

    # The postconditions the caller actually relies on hold on every draw, and
    # the shortfall against the optimum is rare rather than systematic.
    rng = np.random.default_rng(0)
    worse = 0
    for _ in range(3000):
        m = int(rng.integers(2, 9))
        d = rng.uniform(0.001, 0.05, m)
        budget = float(rng.uniform(0.01, 0.2))
        band = float(rng.uniform(0.001, 0.02))
        kept, s = _clearing_prefix(d, budget, band)
        if kept.any():
            assert float((s * d[kept]).sum()) <= budget + 1e-12
            assert float((s * d[kept]).min()) > band
        worse += int(best_feasible(d, budget, band) > int(kept.sum()))
    assert worse == 59, worse            # 2.0%, the figure the docstring quotes


def test_band_suppression_helper_counts_the_orders_it_removed() -> None:
    """Requirement 6.  It must agree with the mechanism, not restate it."""
    from trader.allocator.deterministic import band_suppression

    r_hat, vol, mask, sids, _ = _inputs(n=40, seed=4, n_sectors=4)
    p0 = AllocatorParams(k=15, max_name_weight=0.15, max_sector_weight=0.5,
                         turnover_budget=0.5)
    cur = np.zeros(41)
    cur[1:] = _target_equity(np.roll(r_hat, 3), vol, mask, sids, p0)
    cur[0] = max(0.0, 1.0 - float(cur[1:].sum()))

    off = band_suppression(r_hat, vol, mask, sids, cur, p0)
    assert off.n_suppressed == 0
    assert off.n_traded == off.n_traded_unbanded
    assert off.gross == pytest.approx(off.gross_unbanded)

    on = AllocatorParams(k=15, max_name_weight=0.15, max_sector_weight=0.5,
                         turnover_budget=0.5, no_trade_band=0.01)
    rep = band_suppression(r_hat, vol, mask, sids, cur, on)
    assert rep.n_traded_unbanded == off.n_traded_unbanded
    assert rep.n_suppressed == rep.n_traded_unbanded - rep.n_traded
    assert rep.n_suppressed > 0
    assert rep.n_sold <= rep.n_sold_unbanded
    assert rep.gross <= rep.gross_unbanded
    # and the count is the count the allocator actually produced
    w = allocate(r_hat, vol, mask, sids, cur, on)
    assert rep.n_traded == int((_deltas(w, cur) != 0.0).sum())


# ── the measurement this stream exists to produce ────────────────────────────

_BANDS = (0.000, 0.002, 0.005, 0.010, 0.020)


_SWEEP_CAPITALS = (1_000_000.0, 10_000_000.0)      # ₹10 lakh, ₹1 crore

# Imported, never transcribed: the band's whole argument is about which orders
# the env will actually bill a flat fee on, and that threshold lives in
# `costs.py` (CLAUDE.md rule 5).
_MIN_TRADE = DEFAULT_MIN_TRADE_VALUE


def _band_sweep(
    band: float,
    *,
    rho: float,
    seeds: tuple[int, ...] = (0, 1, 2),
    periods: int = 60,
) -> tuple[float, float, float, tuple[float, ...]]:
    """Per rebalance: (names traded, names sold, gross turnover, names BILLED).

    Synthetic panel, no market data and no env: N=504 names on the real sector
    size shape, an AR(1) signal with persistence ``rho`` (so the top-K set
    churns without being resampled from scratch each period), period returns
    that drift the book between rebalances, and the pinned allocator defaults
    (K=30, 0.10 name cap, 0.25 sector cap, 0.30 turnover budget).  Deterministic
    in ``seeds``.

    The last entry is the count that matters and it is **not** ``n_sold``.  The
    flat ₹15.34 demat debit is billed only on a sell the env actually executes,
    and the env drops any order worth less than ``DEFAULT_MIN_TRADE_VALUE``
    (`costs.py`, `panel_env.py`), so the billed count is the sells with
    ``|Δw| · NAV >= ₹500`` — one entry per capital in :data:`_SWEEP_CAPITALS`,
    because a rupee floor is not scale-free and the band is.

    Quoting the band's effect against ``n_sold`` overstates it by ~3x: most of
    the band-0 sell tail is sub-₹500 dust the env was never going to execute, so
    the band gets credit for removing orders that did not exist.  The band's
    ratios are unaffected above 0.002 (a 0.002 band on ₹10 lakh is ₹2,000, well
    clear of the floor) — only the BASELINE is wrong, and the baseline sets
    every ratio.
    """
    p = AllocatorParams(no_trade_band=band)
    n_traded: list[int] = []
    n_sold: list[int] = []
    n_billed: list[list[int]] = [[] for _ in _SWEEP_CAPITALS]
    gross: list[float] = []
    sizes = [121, 99, 70, 48, 45, 42, 36, 24, 19]
    sids = np.resize(
        np.repeat(np.arange(1, len(sizes) + 1), sizes).astype(np.int64), 504
    )
    n = int(sids.size)
    for seed in seeds:
        rng = np.random.default_rng(seed)
        vol = rng.uniform(0.15, 0.60, n)
        mask = np.ones(n, dtype=bool)
        r_hat = rng.normal(size=n)
        cur = np.zeros(n + 1)
        cur[0] = 1.0
        for _ in range(periods):
            w = allocate(r_hat, vol, mask, sids, cur, p)
            d = _deltas(w, cur)
            n_traded.append(int((np.abs(d) > 1e-12).sum()))
            n_sold.append(int((d < -1e-12).sum()))
            gross.append(float(np.abs(d).sum()))
            for ci, capital in enumerate(_SWEEP_CAPITALS):
                n_billed[ci].append(
                    int(((d < -1e-12) & (np.abs(d) * capital >= _MIN_TRADE)).sum())
                )
            # Hold to the next rebalance: prices move, so weights drift.
            ret = np.exp(rng.normal(0.0, vol / np.sqrt(12.0), n)) - 1.0
            val = w[1:] * (1.0 + ret)
            cur = np.r_[w[0], val] / (w[0] + float(val.sum()))
            r_hat = rho * r_hat + np.sqrt(1.0 - rho * rho) * rng.normal(size=n)
    return (
        float(np.mean(n_traded)),
        float(np.mean(n_sold)),
        float(np.mean(gross)),
        tuple(float(np.mean(b)) for b in n_billed),
    )


def test_band_cuts_name_count_faster_than_turnover() -> None:
    """The headline claim of P3, measured rather than asserted.

    Reproduce, table included:

        uv run pytest -s -q tests/unit/test_allocator.py \
            -k band_cuts_name_count_faster_than_turnover

    The flat ₹15.34 demat debit is billed per distinct scrip SOLD; gross
    turnover is what `turnover_budget` constrains.  If a band cut both by the
    same proportion it would be a slower turnover budget and nothing more.
    Two signal persistences, because the answer depends on how fast the top-K
    set churns and one synthetic is not a measurement.

    **Quote the `billed` columns, not `sold`.**  `sold` counts weight-space
    sells, and at band 0 roughly two thirds of them are sub-₹500 dust that
    `min_trade_value` refuses to execute (`panel_env.py`, `paper_broker.py`,
    `DEFAULT_MIN_TRADE_VALUE` in `costs.py`), so the fee is never billed on
    them.  Measuring the band against that inflated baseline overstates its
    effect by about 3x — at ₹10 lakh a 0.005 band leaves 6.5% of `sold` but
    20.0% of `billed`.  Above a 0.002 band the two columns coincide (a 0.002
    band on ₹10 lakh is ₹2,000, far above the floor); only the baseline differs,
    and the baseline is what every ratio is divided by.  This is a CLAUDE.md
    rule 3 issue rather than an arithmetic one: the old number was reproducible
    and measured the wrong quantity.

    The `fee` column is indicative only: names BILLED x DP charge x 12
    rebalances a year on a ₹10 lakh book, as a percent of NAV.  It uses the
    constant from `costs.py` rather than a transcription of it (CLAUDE.md rule
    5), and it ignores every proportional levy and any effect on returns.  It is
    a cost arithmetic, not a backtest.
    """
    from trader.env.costs import _DP_CHARGE

    for rho in (0.90, 0.98):
        rows = {b: _band_sweep(b, rho=rho) for b in _BANDS}
        base_traded, base_sold, base_gross, base_billed = rows[0.0]
        print(f"\n  AR(1) rho={rho}, N=504, K=30, budget=0.30, 3 seeds x 60 periods")
        print(f"{'band':>7}{'traded':>9}{'sold':>8}{'gross':>8}"
              f"{'bill10L':>9}{'bill1Cr':>9}"
              f"{'traded%':>9}{'sold%':>8}{'gross%':>8}"
              f"{'bill10L%':>10}{'bill1Cr%':>10}{'fee %NAV/yr':>13}")
        for b in _BANDS:
            traded, sold, gross, billed = rows[b]
            fee = 100.0 * billed[0] * _DP_CHARGE * 12.0 / _SWEEP_CAPITALS[0]
            print(f"{b:>7.3f}{traded:>9.2f}{sold:>8.2f}{gross:>8.4f}"
                  f"{billed[0]:>9.2f}{billed[1]:>9.2f}"
                  f"{100 * traded / base_traded:>9.1f}{100 * sold / base_sold:>8.1f}"
                  f"{100 * gross / base_gross:>8.1f}"
                  f"{100 * billed[0] / base_billed[0]:>10.1f}"
                  f"{100 * billed[1] / base_billed[1]:>10.1f}{fee:>13.2f}")

        # A wider band never trades more names.
        counts = [rows[b][0] for b in _BANDS]
        assert counts == sorted(counts, reverse=True), (rho, counts)
        # The claim: at every non-zero band the name count falls by strictly
        # more, proportionally, than the gross turnover that bought it — and
        # the BILLED sell count, which is what the flat fee is actually charged
        # on, falls faster still.
        for b in _BANDS[1:]:
            traded, sold, gross, billed = rows[b]
            assert traded / base_traded < gross / base_gross, (rho, b)
            assert sold / base_sold < traded / base_traded, (rho, b)
            for ci in range(len(_SWEEP_CAPITALS)):
                assert billed[ci] / base_billed[ci] < gross / base_gross, (rho, b, ci)
        # And the correction is real, not cosmetic: at band 0 the weight-space
        # sell count is far above the count the env would ever bill for, so the
        # two baselines are not interchangeable.
        assert base_billed[0] < 0.5 * base_sold, (rho, base_billed[0], base_sold)
        # The floor is not scale-free and the band is, so a bigger account bills
        # more of the same weight-space tail.
        assert base_billed[1] > base_billed[0], (rho, base_billed)


# ── params validation ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "kwargs",
    [
        {"k": 0},
        {"max_name_weight": 0.0},
        {"max_sector_weight": 1.5},
        {"turnover_budget": -0.1},
        {"cash_floor": 1.0},
        {"no_trade_band": -0.001},
        {"no_trade_band": 1.0},
    ],
)
def test_params_rejects_out_of_range(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        AllocatorParams(**kwargs)  # type: ignore[arg-type]


def test_allocate_rejects_bad_shapes() -> None:
    r_hat, vol, mask, sids, cur = _inputs()
    with pytest.raises(ValueError):
        allocate(r_hat, vol, mask, sids, cur[:-1], AllocatorParams())
    with pytest.raises(ValueError):
        allocate(r_hat[:-1], vol, mask, sids, cur, AllocatorParams())


# ── RebalanceSchedule ─────────────────────────────────────────────────────────

# A trading calendar across a month boundary with a holiday on Monday 2 Feb
# (so the first trading day of Feb and of that ISO week is Tuesday 3 Feb) and
# a Friday-1st month (May 2026 starts on a Friday).
_CAL = [
    date(2026, 1, 28), date(2026, 1, 29), date(2026, 1, 30),   # Wed Thu Fri
    # Mon 2 Feb is a holiday
    date(2026, 2, 3), date(2026, 2, 4), date(2026, 2, 5), date(2026, 2, 6),   # Tue..Fri
    date(2026, 2, 9), date(2026, 2, 10),                                      # Mon Tue
]


def test_daily_schedule_always_rebalances() -> None:
    s = RebalanceSchedule("daily")
    assert s.mask(_CAL).all()
    assert s.is_rebalance_day(_CAL[1], _CAL[0])


def test_monthly_first_trading_day_across_holiday() -> None:
    s = RebalanceSchedule("monthly")
    m = s.mask(_CAL)
    expected = [True, False, False, True, False, False, False, False, False]
    assert m.tolist() == expected
    # pairwise form agrees with the vectorised mask
    pair = [s.is_rebalance_day(d, _CAL[i - 1] if i else None) for i, d in enumerate(_CAL)]
    assert pair == expected
    # Tue 3 Feb, not Mon 2 Feb (holiday) and not Sun 1 Feb, is the rebalance day
    assert _CAL[3] == date(2026, 2, 3)


def test_weekly_first_trading_day_of_week() -> None:
    s = RebalanceSchedule("weekly")
    m = s.mask(_CAL)
    # Wed 28 Jan starts the calendar (edge → True); Tue 3 Feb is the first
    # observed day of ISO week 6 because Monday was a holiday; Mon 9 Feb.
    assert m.tolist() == [True, False, False, True, False, False, False, True, False]


def test_last_anchor_uses_next_date() -> None:
    s = RebalanceSchedule("monthly", anchor="last")
    m = s.mask(_CAL)
    # Fri 30 Jan is the last trading day of Jan; the calendar's end counts too.
    assert m.tolist() == [False, False, True, False, False, False, False, False, True]
    assert s.is_rebalance_day(_CAL[2], _CAL[1], _CAL[3])
    assert not s.is_rebalance_day(_CAL[1], _CAL[0], _CAL[2])
    assert s.is_rebalance_day(_CAL[-1], _CAL[-2], None)


def test_year_boundary_is_a_month_boundary() -> None:
    s = RebalanceSchedule("monthly")
    assert s.is_rebalance_day(date(2027, 1, 1), date(2026, 12, 31))
    # ISO week 53 of 2026 → week 1 of 2027 is also a weekly boundary
    w = RebalanceSchedule("weekly")
    assert w.is_rebalance_day(date(2027, 1, 4), date(2026, 12, 31))


def test_schedule_rejects_bad_values() -> None:
    with pytest.raises(ValueError):
        RebalanceSchedule("fortnightly")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        RebalanceSchedule("monthly", anchor="middle")  # type: ignore[arg-type]


def test_current_w_summing_above_one_is_renormalised_not_trusted() -> None:
    """`obs["portfolio"]` can sum to >1 and the budget must still mean 30%.

    The env's `portfolio` space is `Box(0, 1)`, so when its cash goes slightly
    negative — it sizes against a NAV marked at the previous close and fills at
    the open, so a gap up overspends — the clipped cash reads 0.0 and the
    vector sums to about 1.01 (measured 1.009760, see
    `tests/integration/test_panel_env.py`).  Budgeting turnover against that
    unnormalised vector silently grants ~1% more turnover than asked for.
    """
    r_hat, vol, mask, sids, _ = _inputs()
    p = AllocatorParams(k=4, max_name_weight=0.5, max_sector_weight=1.0, turnover_budget=0.30)
    target = _unconstrained(r_hat, vol, mask, sids, p)

    # A book fully invested in the *bottom* names, reported 1% "over" with
    # zero cash — exactly the shape the env emits after a gap up.
    inflated = np.zeros_like(target)
    inflated[1:] = _unconstrained(-r_hat, vol, mask, sids, p)[1:] * 1.01
    assert inflated[0] == 0.0
    assert inflated.sum() == pytest.approx(1.01, abs=1e-9)

    w = allocate(r_hat, vol, mask, sids, inflated, p)
    _check_basic(w)
    norm = inflated / inflated.sum()
    gross = float(np.abs(w[1:] - norm[1:]).sum())
    assert gross == pytest.approx(p.turnover_budget, abs=1e-9)
    # And measured against the raw vector it would have read >0.30 — which is
    # the bug this test exists to keep out.
    assert float(np.abs(w[1:] - inflated[1:]).sum()) > p.turnover_budget
