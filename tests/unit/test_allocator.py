"""Unit tests for the R5 deterministic allocator and rebalance schedule.

Each test names the failure mode it exists to catch.  None of these touch a
panel or the env; `tests/integration/test_panel_env.py` covers the env side.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from trader.allocator import AllocatorParams, RebalanceSchedule, allocate

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


# ── params validation ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "kwargs",
    [
        {"k": 0},
        {"max_name_weight": 0.0},
        {"max_sector_weight": 1.5},
        {"turnover_budget": -0.1},
        {"cash_floor": 1.0},
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
