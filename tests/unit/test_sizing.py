"""The supportable name count must be *derived* from capital, not chosen by hand.

Every backtest in `audit/R4_R5_RESULTS.md` ran at ₹10 lakh
(`panel_env.py: initial_cash: float = 1_000_000.0`); the account being stood in
for holds ₹1 lakh (`paper_broker.py:111`).  Because the demat debit fee is FLAT
(₹15.34 per scrip per selling day, `costs.py:55`) that 10x is a 10x error in the
only cost that has a K in it.  At ₹1 lakh a 504-name equal-weight book holds ₹198
a position and pays 7.7% of it to sell.  P5 of `11_cost_defect_and_fix_plan.md`.

Every test here is a statement about the mechanism, not about the plumbing: each
one fails if either constraint in `sizing.max_supportable_k` is deleted, or if
the flat fee is replaced by a proportional one.
"""
from __future__ import annotations

import random

import pytest

from trader.allocator.sizing import (
    DEFAULT_CAPITALS,
    DEFAULT_KS,
    DEFAULT_MAX_FEE_FRACTION,
    DEFAULT_POSITION_SHARE,
    DEFAULT_REBALANCE_FRACTION,
    DEMAT_DEBIT_FEE,
    R5_MONTHLY_TURNOVER,
    REBALANCES_PER_YEAR,
    UNBOUNDED_K,
    capacity_table,
    fee_budget_crossover,
    fee_drag_estimate,
    fee_fraction_per_exit,
    format_capacity_table,
    max_supportable_k,
    names_sold_per_rebalance,
    position_value,
    supportable_k_detail,
)
from trader.env.costs import _DP_CHARGE, DEFAULT_MIN_TRADE_VALUE

_MONTHLY = REBALANCES_PER_YEAR["monthly"]


def _brute_force_k(
    capital: float,
    *,
    max_fee_fraction: float,
    min_trade_value: float,
    rebalance_fraction: float,
    smallest_position_share: float,
    dp_charge: float,
    ceiling: int = 6000,
) -> int:
    """Independent search, from first principles — no closed form anywhere.

    Scans every K and keeps the largest that satisfies BOTH stated conditions:
    a full exit's flat fee is within budget, and a routine trim clears the rupee
    floor.  Deliberately written from the English statement of the constraints
    rather than from the algebra it is checking.
    """
    best = 0
    for k in range(1, ceiling + 1):
        pos = smallest_position_share * capital / k
        exit_is_cheap_enough = dp_charge / pos <= max_fee_fraction
        trim_clears_the_floor = rebalance_fraction * pos >= min_trade_value
        if exit_is_cheap_enough and trim_clears_the_floor:
            best = k
    return best


# --------------------------------------------------------------------------
# The constants come from costs.py; they are never retyped here or there.
# --------------------------------------------------------------------------


def test_flat_fee_constant_is_sourced_from_costs_not_retyped() -> None:
    # CLAUDE.md rule 5: cost constants live in costs.py and are edited in place.
    # If someone hardcodes 15.34 into sizing.py this test does not notice the
    # value -- it notices that the identity to the source was broken.
    assert DEMAT_DEBIT_FEE == _DP_CHARGE
    assert DEMAT_DEBIT_FEE == pytest.approx(15.34)


def test_min_trade_value_default_is_the_shared_constant() -> None:
    detail = supportable_k_detail(1_000_000.0)
    explicit = supportable_k_detail(1_000_000.0, min_trade_value=DEFAULT_MIN_TRADE_VALUE)
    assert detail == explicit
    assert DEFAULT_MIN_TRADE_VALUE == pytest.approx(500.0)


# --------------------------------------------------------------------------
# The headline figure P5 is built on.
# --------------------------------------------------------------------------


def test_one_lakh_over_504_names_pays_7_7_percent_to_exit() -> None:
    """`11_cost_defect_and_fix_plan.md:97-99` — ₹198 a position, 7.7% per sale."""
    assert position_value(1e5, 504) == pytest.approx(198.41, abs=0.01)
    assert fee_fraction_per_exit(1e5, 504) == pytest.approx(0.0773, abs=5e-5)


def test_one_lakh_at_k30_is_workable() -> None:
    """The same source: ₹3,333 a position, 0.46% per sale."""
    assert position_value(1e5, 30) == pytest.approx(3333.33, abs=0.01)
    assert fee_fraction_per_exit(1e5, 30) == pytest.approx(0.0046, abs=5e-5)


def test_ten_lakh_hides_the_problem_by_exactly_ten_times() -> None:
    """Why the backtests never saw it: the same book, 10x the denominator."""
    at_1l = fee_fraction_per_exit(1e5, 504)
    at_10l = fee_fraction_per_exit(1e6, 504)
    assert at_1l / at_10l == pytest.approx(10.0)
    assert at_10l == pytest.approx(0.00773, abs=5e-6)


# --------------------------------------------------------------------------
# Closed form == brute force.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("capital", [2_500.0, 1e4, 1e5, 2.5e5, 5e5, 1e6])
def test_closed_form_matches_brute_force_at_defaults(capital: float) -> None:
    assert max_supportable_k(capital) == _brute_force_k(
        capital,
        max_fee_fraction=DEFAULT_MAX_FEE_FRACTION,
        min_trade_value=DEFAULT_MIN_TRADE_VALUE,
        rebalance_fraction=DEFAULT_REBALANCE_FRACTION,
        smallest_position_share=DEFAULT_POSITION_SHARE,
        dp_charge=DEMAT_DEBIT_FEE,
    )


def test_closed_form_matches_brute_force_over_random_parameters() -> None:
    """200 random parameter draws, both constraints live, both able to bind."""
    rng = random.Random(20260906)
    for _ in range(200):
        capital = rng.uniform(3_000.0, 8e5)
        f = rng.uniform(0.001, 0.02)          # straddles the 0.006136 crossover
        v = rng.uniform(100.0, 1_000.0)
        t = rng.uniform(0.05, 1.0)
        s = rng.uniform(0.5, 1.0)
        got = max_supportable_k(
            capital,
            max_fee_fraction=f,
            min_trade_value=v,
            rebalance_fraction=t,
            smallest_position_share=s,
        )
        want = _brute_force_k(
            capital,
            max_fee_fraction=f,
            min_trade_value=v,
            rebalance_fraction=t,
            smallest_position_share=s,
            dp_charge=DEMAT_DEBIT_FEE,
        )
        assert got == want, (capital, f, v, t, s, got, want)


def test_brute_force_would_catch_a_dropped_constraint() -> None:
    """Guards the guard: at these parameters the two bounds genuinely differ.

    If they coincided everywhere, the brute-force test above would pass with
    either constraint deleted -- which is this repository's recurring failure.
    """
    loose = supportable_k_detail(1e5, max_fee_fraction=0.05)   # min_trade binds
    tight = supportable_k_detail(1e5, max_fee_fraction=0.002)  # flat fee binds
    assert loose.k_flat_fee != loose.k_min_trade
    assert tight.k_flat_fee != tight.k_min_trade
    assert loose.k_max == loose.k_min_trade < loose.k_flat_fee
    assert tight.k_max == tight.k_flat_fee < tight.k_min_trade


# --------------------------------------------------------------------------
# Monotonicity in capital.
# --------------------------------------------------------------------------


def test_k_never_increases_as_capital_falls() -> None:
    capitals = [1e6 * (0.85**i) for i in range(60)]  # 10L down to ~₹6,000
    ks = [max_supportable_k(c) for c in capitals]
    assert ks == sorted(ks, reverse=True)


def test_k_strictly_falls_across_the_capital_grid() -> None:
    ks = [max_supportable_k(c) for c in DEFAULT_CAPITALS]
    assert ks == [40, 100, 200, 400, 2000]
    assert all(a < b for a, b in zip(ks[:-1], ks[1:], strict=True))


def test_k_is_linear_in_capital() -> None:
    """Both bounds are proportional to capital, so K is too (up to flooring)."""
    for c in (1e5, 2.5e5, 1e6):
        assert max_supportable_k(10.0 * c) == 10 * max_supportable_k(c)


def test_no_k_is_supportable_below_v_over_t() -> None:
    """Below ₹2,500 a 20% trim of a single whole position cannot clear ₹500."""
    threshold = DEFAULT_MIN_TRADE_VALUE / DEFAULT_REBALANCE_FRACTION
    assert threshold == pytest.approx(2_500.0)
    assert max_supportable_k(threshold) == 1
    assert max_supportable_k(threshold - 1.0) == 0
    assert supportable_k_detail(threshold - 1.0).binding == "none"


def test_capital_and_position_share_enter_only_as_a_product() -> None:
    """`s` is one knob folding cash floor and inverse-vol dispersion together."""
    assert max_supportable_k(1e6, smallest_position_share=0.5) == max_supportable_k(5e5)
    assert fee_fraction_per_exit(1e6, 30, smallest_position_share=0.5) == pytest.approx(
        fee_fraction_per_exit(5e5, 30)
    )


# --------------------------------------------------------------------------
# Which constraint binds, and where.
# --------------------------------------------------------------------------


def test_crossover_is_dp_times_trim_over_min_trade_value() -> None:
    assert fee_budget_crossover() == pytest.approx(
        DEMAT_DEBIT_FEE * DEFAULT_REBALANCE_FRACTION / DEFAULT_MIN_TRADE_VALUE
    )
    assert fee_budget_crossover() == pytest.approx(0.006136)


@pytest.mark.parametrize("capital", DEFAULT_CAPITALS)
def test_min_trade_value_binds_above_the_crossover_at_every_capital(capital: float) -> None:
    f_star = fee_budget_crossover()
    detail = supportable_k_detail(capital, max_fee_fraction=f_star * 1.5)
    assert detail.binding == "min_trade_value"
    assert detail.k_max == detail.k_min_trade < detail.k_flat_fee


@pytest.mark.parametrize("capital", DEFAULT_CAPITALS)
def test_flat_fee_binds_below_the_crossover_at_every_capital(capital: float) -> None:
    f_star = fee_budget_crossover()
    detail = supportable_k_detail(capital, max_fee_fraction=f_star * 0.5)
    assert detail.binding == "flat_fee"
    assert detail.k_max == detail.k_flat_fee < detail.k_min_trade


def test_the_default_budget_sits_above_the_crossover() -> None:
    """So min_trade_value is the active rule at ₹1 lakh AND at ₹50 lakh."""
    assert DEFAULT_MAX_FEE_FRACTION > fee_budget_crossover()
    assert {supportable_k_detail(c).binding for c in DEFAULT_CAPITALS} == {"min_trade_value"}


def test_which_constraint_binds_does_not_depend_on_capital() -> None:
    """The headline claim of the module docstring, asserted over 4 decades.

    Two things are checked, and the second is what makes the first non-vacuous:

    1. the label is the same at every capital;
    2. it is the label ``f`` vs ``f*`` predicts — so a bug that returned a
       *constant* label everywhere could not pass.

    Seed-swept on purpose.  The original body hard-coded ``random.Random(7)``
    and 50 draws; run over seeds 0-399 that exact body FAILS on 139 of them,
    because the label was read off the floored integers ``k_fee`` vs ``k_mtv``
    and those can tie at ₹1 lakh while separating at ₹10 lakh.  Seed 7 was one
    of the 261 survivors.  200 seeds x 4 draws x 4 capitals here, plus the
    counterexample budget below pinned explicitly.
    """
    f_star = fee_budget_crossover()
    caps = (1e5, 1e6, 1e7, 1e8)
    for seed in range(200):
        rng = random.Random(seed)
        for _ in range(4):
            f = rng.uniform(0.001, 0.02)
            bindings = {
                supportable_k_detail(c, max_fee_fraction=f).binding for c in caps
            }
            assert len(bindings) == 1, (seed, f, bindings)
            expected = "flat_fee" if f < f_star else "min_trade_value"
            assert bindings == {expected}, (seed, f, f_star, bindings)


def test_the_counterexample_budget_labels_the_same_rule_at_every_capital() -> None:
    """f = 0.00625697, the budget on which the floored-integer labelling broke.

    At this budget ``k_fee`` and ``k_mtv`` tie at ₹1 lakh (40 = 40) and separate
    at ₹10 lakh (407 vs 400), so the old code said "both" at one capital and
    "min_trade_value" at the next.  ``k_max`` is unaffected — it is the min of
    the two floors either way — which is why no capacity number ever moved.
    """
    f = 0.00625697
    assert f > fee_budget_crossover()
    for capital in (1e5, 1e6, 1e7, 1e8):
        detail = supportable_k_detail(capital, max_fee_fraction=f)
        assert detail.binding == "min_trade_value", (capital, detail)
        assert detail.k_max == min(detail.k_flat_fee, detail.k_min_trade)
    # The floors really do tie at ₹1 lakh — without this the test above could
    # pass on a budget that never exercised the defect.
    lakh = supportable_k_detail(1e5, max_fee_fraction=f)
    assert lakh.k_flat_fee == lakh.k_min_trade == 40
    crore = supportable_k_detail(1e7, max_fee_fraction=f)
    assert (crore.k_flat_fee, crore.k_min_trade) == (4078, 4000)


def test_min_trade_value_bound_pins_the_position_at_v_over_t() -> None:
    """When B binds, the smallest position lands at ₹2,500 whatever the capital.

    That is why the fee fraction at ``k_max`` is the crossover f* itself: the
    constraint fixes the position *size*, and the flat fee is a fixed rupee
    charge against it.
    """
    for capital in DEFAULT_CAPITALS:
        detail = supportable_k_detail(capital)
        assert detail.binding == "min_trade_value"
        assert detail.position_value == pytest.approx(
            DEFAULT_MIN_TRADE_VALUE / DEFAULT_REBALANCE_FRACTION
        )
        assert detail.fee_fraction_at_k_max == pytest.approx(fee_budget_crossover())


def test_the_trim_at_k_max_clears_the_floor_and_at_k_max_plus_one_does_not() -> None:
    """The constraint is tight: K+1 is genuinely un-rebalanceable."""
    for capital in DEFAULT_CAPITALS:
        k = max_supportable_k(capital)
        pos = position_value(capital, k)
        assert DEFAULT_REBALANCE_FRACTION * pos >= DEFAULT_MIN_TRADE_VALUE
        over = position_value(capital, k + 1)
        assert DEFAULT_REBALANCE_FRACTION * over < DEFAULT_MIN_TRADE_VALUE


def test_zero_min_trade_value_disables_constraint_b() -> None:
    detail = supportable_k_detail(1e5, min_trade_value=0.0)
    assert detail.k_min_trade == UNBOUNDED_K
    assert detail.binding == "flat_fee"
    assert detail.k_max == detail.k_flat_fee == 65  # floor(0.01 x 100000 / 15.34)


def test_max_k_clamps_and_is_reported() -> None:
    # ₹1 crore supports K=4000 on the economics; the investable universe today
    # is `active_tickers()` = 504, so the universe is what binds.
    detail = supportable_k_detail(1e7, max_k=504)
    assert detail.k_min_trade == 4000
    assert detail.k_max == 504
    assert detail.binding == "max_k"
    assert supportable_k_detail(1e5, max_k=504).binding == "min_trade_value"


# --------------------------------------------------------------------------
# Annual drag.
# --------------------------------------------------------------------------


def test_names_sold_uses_one_sided_turnover_and_caps_at_k() -> None:
    # Gross two-sided annual 3.73 at 12 rebalances => 0.15542 one-sided per
    # rebalance; at full exits that is 0.15542 x 30 names.
    n = names_sold_per_rebalance(30, _MONTHLY, R5_MONTHLY_TURNOVER)
    assert n == pytest.approx(30.0 * R5_MONTHLY_TURNOVER / 24.0)
    assert n == pytest.approx(4.6625)
    # Cannot sell more distinct names than are held, however large the turnover.
    assert names_sold_per_rebalance(30, _MONTHLY, 1000.0) == pytest.approx(30.0)


def test_partial_trims_cost_strictly_more_than_full_exits() -> None:
    """The same rupees spread over more scrips pay the flat fee more times."""
    full = fee_drag_estimate(1e5, 30, _MONTHLY, R5_MONTHLY_TURNOVER)
    trims = fee_drag_estimate(1e5, 30, _MONTHLY, R5_MONTHLY_TURNOVER, avg_sell_share=0.17)
    assert trims > full
    assert trims / full == pytest.approx(1.0 / 0.17, rel=1e-9)


def test_annual_drag_matches_a_hand_computation() -> None:
    # 15.34 x 12 x 30 x (3.73 / 24) = Rs 858.27/yr, whatever the account holds.
    expected_rupees = DEMAT_DEBIT_FEE * 12.0 * 30.0 * (R5_MONTHLY_TURNOVER / 24.0)
    assert expected_rupees == pytest.approx(858.27, abs=0.01)
    assert fee_drag_estimate(1e5, 30, _MONTHLY, R5_MONTHLY_TURNOVER) == pytest.approx(
        expected_rupees / 1e5
    )
    assert fee_drag_estimate(1e5, 30, _MONTHLY, R5_MONTHLY_TURNOVER) == pytest.approx(
        0.008583, abs=1e-6
    )


def test_drag_is_inversely_proportional_to_capital() -> None:
    """The flat bill does not shrink with the account; the drag does not either."""
    small = fee_drag_estimate(1e5, 30, _MONTHLY, R5_MONTHLY_TURNOVER)
    large = fee_drag_estimate(1e6, 30, _MONTHLY, R5_MONTHLY_TURNOVER)
    assert small / large == pytest.approx(10.0)


def test_drag_is_linear_in_k_below_saturation() -> None:
    a = fee_drag_estimate(1e5, 20, _MONTHLY, R5_MONTHLY_TURNOVER)
    b = fee_drag_estimate(1e5, 40, _MONTHLY, R5_MONTHLY_TURNOVER)
    assert b / a == pytest.approx(2.0)


def test_daily_cadence_drag_dwarfs_monthly() -> None:
    """`audit/R4_R5_RESULTS.md:54` — daily K=20 ran 72.3x turnover."""
    monthly = fee_drag_estimate(1e5, 20, _MONTHLY, 3.74)
    daily = fee_drag_estimate(1e5, 20, REBALANCES_PER_YEAR["daily"], 72.30)
    assert daily > 15.0 * monthly
    # 15.34 x 252 x 20 x (72.30 / 504) = Rs 11,091/yr in flat fees alone, which
    # is 11.1% of a ₹1 lakh account before a single proportional cost.
    assert daily == pytest.approx(0.11091, abs=1e-5)


# --------------------------------------------------------------------------
# The deliverable table.
# --------------------------------------------------------------------------


def test_capacity_table_covers_the_p5_grid() -> None:
    rows = capacity_table()
    assert len(rows) == len(DEFAULT_CAPITALS) * len(DEFAULT_KS)
    assert {r.capital for r in rows} == set(DEFAULT_CAPITALS)
    assert {r.k for r in rows} == set(DEFAULT_KS)


def test_capacity_table_headline_cells() -> None:
    rows = {(r.capital, r.k): r for r in capacity_table()}
    one_lakh_k30 = rows[(1e5, 30)]
    assert one_lakh_k30.position_value == pytest.approx(3333.33, abs=0.01)
    assert one_lakh_k30.fee_fraction_per_exit == pytest.approx(0.004602, abs=1e-6)
    assert one_lakh_k30.annual_drag == pytest.approx(0.008583, abs=1e-6)
    assert one_lakh_k30.rebalanceable and one_lakh_k30.within_fee_budget

    # K=40 at ₹1 lakh sits exactly on the min-trade-value boundary: ₹2,500 a
    # position, a 20% trim of which is exactly ₹500.
    edge = rows[(1e5, 40)]
    assert edge.position_value == pytest.approx(2500.0)
    assert edge.rebalanceable
    assert edge.fee_fraction_per_exit == pytest.approx(fee_budget_crossover())


def test_annual_fee_in_rupees_is_the_same_at_every_capital() -> None:
    """The flat bill is set by K and ANNUAL TURNOVER -- capital only divides it.

    Not "K and cadence": outside the saturation cap the R in ``D·R·n`` cancels
    the R inside ``n = k·turnover/(2R·a)``.  See
    ``test_cadence_cancels_at_fixed_annual_turnover`` below.
    """
    rows = capacity_table()
    for k in DEFAULT_KS:
        fees = {round(r.annual_fee_rupees, 6) for r in rows if r.k == k}
        assert len(fees) == 1


def test_cadence_cancels_at_fixed_annual_turnover() -> None:
    """``rebalances_per_year`` is algebraically absent below the saturation cap.

    drag = D·R·k·(turnover/(2R))/a / C = D·k·turnover/(2·a·C).  So quoting the
    daily arm's bill as "19.4x the monthly one because it rebalances 21x as
    often" is wrong; it is 19.4x because its annual turnover is 19.4x.
    """
    drags = {
        r: fee_drag_estimate(1e5, 30, r, R5_MONTHLY_TURNOVER)
        for r in REBALANCES_PER_YEAR.values()
    }
    assert len({round(d, 12) for d in drags.values()}) == 1, drags
    assert drags[12.0] == pytest.approx(0.008583, abs=1e-6)
    # The two rows the P5 table contrasts differ only by their turnover.
    monthly = fee_drag_estimate(1e5, 30, 12.0, 3.73)
    daily = fee_drag_estimate(1e5, 30, 252.0, 72.30)
    assert daily / monthly == pytest.approx(72.30 / 3.73, rel=1e-9)


def test_cadence_matters_only_inside_the_saturation_cap() -> None:
    """R's one real degree of freedom: it decides WHERE the cap bites.

    Above ``turnover > 2·R·a`` every rebalance sells the whole book, ``n`` pins
    at ``k`` and the bill grows linearly in R.  Without this the parameter could
    be deleted and only the tests that happen to straddle the cap would notice.
    """
    a, k, turnover = 1.0, 30, 800.0          # 800 > 2·252·1, saturated everywhere
    saturated = {
        r: fee_drag_estimate(1e5, k, r, turnover, avg_sell_share=a)
        for r in (12.0, 52.0, 252.0)
    }
    assert saturated[252.0] / saturated[12.0] == pytest.approx(252.0 / 12.0)
    assert names_sold_per_rebalance(k, 12.0, turnover, avg_sell_share=a) == float(k)


def test_a_504_name_book_at_one_lakh_is_not_rebalanceable() -> None:
    rows = capacity_table(capitals=[1e5], ks=[504])
    assert rows[0].rebalanceable is False
    assert rows[0].within_fee_budget is False
    assert rows[0].fee_fraction_per_exit == pytest.approx(0.0773, abs=5e-5)


def test_format_capacity_table_renders_every_row() -> None:
    text = format_capacity_table(capacity_table())
    assert text.count("\n") == len(DEFAULT_CAPITALS) * len(DEFAULT_KS) + 1
    assert "Fee/exit" in text and "Annual drag" in text


# --------------------------------------------------------------------------
# Argument validation.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"capital": 0.0}, "capital must be > 0"),
        ({"capital": -1.0}, "capital must be > 0"),
        ({"capital": 1e5, "max_fee_fraction": 0.0}, "max_fee_fraction must be > 0"),
        ({"capital": 1e5, "min_trade_value": -1.0}, "min_trade_value must be >= 0"),
        ({"capital": 1e5, "rebalance_fraction": 0.0}, "rebalance_fraction must be in"),
        ({"capital": 1e5, "rebalance_fraction": 1.5}, "rebalance_fraction must be in"),
        ({"capital": 1e5, "smallest_position_share": 0.0}, "smallest_position_share"),
        ({"capital": 1e5, "smallest_position_share": 1.5}, "smallest_position_share"),
        ({"capital": 1e5, "max_k": 0}, "max_k must be >= 1"),
    ],
)
def test_supportable_k_rejects_bad_arguments(kwargs: dict[str, float], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        supportable_k_detail(**kwargs)  # type: ignore[arg-type]


def test_position_value_rejects_k_below_one() -> None:
    with pytest.raises(ValueError, match="k must be >= 1"):
        position_value(1e5, 0)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"k": 0}, "k must be >= 1"),
        ({"rebalances_per_year": 0.0}, "rebalances_per_year must be > 0"),
        ({"turnover": -0.1}, "turnover must be >= 0"),
        ({"avg_sell_share": 0.0}, "avg_sell_share must be in"),
        ({"avg_sell_share": 1.5}, "avg_sell_share must be in"),
    ],
)
def test_names_sold_rejects_bad_arguments(kwargs: dict[str, float], match: str) -> None:
    args: dict[str, float] = {"k": 30, "rebalances_per_year": 12.0, "turnover": 1.0}
    args.update(kwargs)
    with pytest.raises(ValueError, match=match):
        names_sold_per_rebalance(**args)  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# The one calibration this module refuses to bake in.
# --------------------------------------------------------------------------


def test_implied_sell_share_is_cadence_dependent() -> None:
    """Regenerates the two `a` figures in `names_sold_per_rebalance`'s docstring.

    Backs the average sell share out of the equal-weight 504-name runs and shows
    it is NOT a constant: 0.167 monthly against 0.053 daily. That is why
    ``avg_sell_share`` is an argument with a full-exit default rather than a
    number frozen into the module -- a calibration fitted at one cadence is
    wrong by 3x at another.

    Sell name-days and span: `11_cost_defect_and_fix_plan.md:37-38` (pre-P1).
    Gross two-sided annual turnover: `audit/R4_R5_RESULTS.md:46,53` (post-P1).
    Two different runs under two different cost models -- an order-of-magnitude
    cross-check, never a measured constant.
    """
    k, steps, days_per_year = 504, 1860, 252.0
    years = steps / days_per_year
    implied: dict[str, float] = {}
    for cadence, sell_name_days, turnover in (("monthly", 10_250, 0.92), ("daily", 83_933, 2.39)):
        r = REBALANCES_PER_YEAR[cadence]
        names_per_rebalance = (sell_name_days / years) / r
        one_sided = turnover / (2.0 * r)
        implied[cadence] = one_sided * k / names_per_rebalance
        # The estimator inverts exactly: feed `a` back in and recover the count.
        assert names_sold_per_rebalance(
            k, r, turnover, avg_sell_share=implied[cadence]
        ) == pytest.approx(names_per_rebalance)

    assert implied["monthly"] == pytest.approx(0.167, abs=5e-4)
    assert implied["daily"] == pytest.approx(0.053, abs=5e-4)
    assert implied["monthly"] / implied["daily"] == pytest.approx(3.15, abs=0.01)


def test_full_exit_default_is_a_floor_on_equal_weight_drag() -> None:
    """`a = 1.0` understates what an equal-weight book actually paid, by a lot."""
    floor = fee_drag_estimate(1e6, 504, _MONTHLY, 0.92)
    calibrated = fee_drag_estimate(1e6, 504, _MONTHLY, 0.92, avg_sell_share=0.167)
    assert calibrated / floor == pytest.approx(1.0 / 0.167, rel=1e-9)
    assert calibrated / floor == pytest.approx(5.99, abs=0.01)   # "6x", rounded
    daily_calibrated = fee_drag_estimate(
        1e6, 504, REBALANCES_PER_YEAR["daily"], 2.39, avg_sell_share=0.053
    )
    daily_floor = fee_drag_estimate(1e6, 504, REBALANCES_PER_YEAR["daily"], 2.39)
    assert daily_calibrated / daily_floor == pytest.approx(18.87, abs=0.01)  # "19x"
