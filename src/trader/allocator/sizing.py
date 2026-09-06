"""How many names can this account actually hold?  (P5, `11_cost_defect_and_fix_plan.md`)

The supportable name count is a **function of capital** and has never been
derived here.  It was chosen by hand (`AllocatorParams.k = 30`) and then measured
at ₹10 lakh (`panel_env.py: initial_cash: float = 1_000_000.0`) while the account
this system is standing in for holds ₹1 lakh
(`paper_broker.py:111  DEFAULT_INITIAL_CASH = 100_000.0`).  That is a 10x error
in the denominator of the only cost that matters.

Why only one cost matters
-------------------------
Zerodha's demat debit fee is **flat**: ₹15.34 per distinct scrip per selling day,
independent of trade value (``costs.py:53-55``, ``zerodha.com/charges``).  It was
83-98% of all costs in every backtest measured (`11_cost_defect_and_fix_plan.md`:
97.7% at daily cadence, 89.8% at monthly).

**This module deliberately ignores every proportional cost** — STT (0.1% each
leg), stamp duty, SEBI turnover fee, exchange transaction charge, GST.  They are
scale-free: they take the same fraction of a ₹198 position and a ₹33,333 one, so
they shift the level of returns without shifting the *choice of K* at all.  Only
the flat fee has a K in it.  Anything here is therefore a statement about how to
pick K, never an estimate of total trading cost — for that, use
:class:`~trader.env.costs.ZerodhaEquityDeliveryCostModel`.

The algebra
-----------
Write

    C  = capital,
    K  = number of names held,
    s  = smallest position as a multiple of the equal weight ``1/K``
         (:data:`DEFAULT_POSITION_SHARE` = 1.0 for an equal-weighted, fully
         invested book; fold BOTH a cash floor and inverse-vol dispersion into
         this ONE number — see the note on ``smallest_position_share`` below),
    D  = the flat demat debit fee, :data:`DEMAT_DEBIT_FEE`,
    f  = ``max_fee_fraction``, the largest share of a position we will pay to
         liquidate it,
    V  = ``min_trade_value``, the rupee floor below which both the env and the
         broker drop a trade (:data:`~trader.env.costs.DEFAULT_MIN_TRADE_VALUE`),
    t  = ``rebalance_fraction``, the fraction of a position a routine trim moves.

The position being sized is

    P(K) = s·C / K.

**Constraint A — the flat fee must not eat the position on the way out.**
A full exit of one name pays exactly D, whatever it is worth, so

    D / P(K) = D·K / (s·C)  ≤  f     ⟺     K  ≤  f·s·C / D.

**Constraint B — a REBALANCE trade, not just an exit, has to clear V.**
This is the constraint that is easy to miss.  A position can be large enough to
exit cheaply and still be too small to *trim*: a trim of fraction t is worth
t·P(K), and if that is under V both `panel_env.py:410` and
`paper_broker.py:1038` drop the order.  Such a name is not rebalanceable — it can
only be held or fully exited, so the allocator's target weights silently stop
being reachable.  Requiring the trim to clear the floor:

    t·P(K) = t·s·C / K  ≥  V         ⟺     K  ≤  t·s·C / V.

(The *entry* is the whole position, P(K) ≥ V ⟺ K ≤ s·C/V, which is weaker than B
for any t ≤ 1 and so never binds on its own.)

Therefore

    K_max = floor( min( f·s·C/D , t·s·C/V ) ).

Which one binds
---------------
**Not the one you would guess.**  Both bounds are *linear in s·C*, so the ratio
between them does not contain capital at all:

    A ≤ B  ⟺  f/D ≤ t/V  ⟺  f ≤ D·t/V  ≜  f*.

At the defaults (D = ₹15.34, t = 0.20, V = ₹500) the crossover is

    f* = 15.34 × 0.20 / 500 = 0.006136  (0.61% of a position).

So: below a 0.61% fee budget the **flat fee binds at every capital**; above it
the **min-trade-value floor binds at every capital**; and no account size ever
changes the answer.  The intuition that "min_trade_value bites the small account"
is wrong — it bites the *loose fee budget*, at ₹1 lakh and ₹5 crore alike.  The
capital only sets the *level* of K, never which rule is active.
:func:`supportable_k_detail` reports the binding constraint per row so this is
never asserted from memory.

That invariant is a property of the **real-valued** bounds, and it is true only
because ``binding`` is derived from them.  It was first derived from the floored
integers ``k_fee`` vs ``k_mtv``, and that is *false*: at f = 0.00625697 (just
above f*) the two floors tie at ₹1 lakh (40 = 40, reported "both") and separate
at ₹10 lakh (407 vs 400, reported "min_trade_value"), so the label moved with the
account after all.  ``k_max = min(k_fee, k_mtv)`` was never affected — the
minimum of the two floors is the same either way — so no capacity number ever
moved; only the label did.  Regression:
``tests/unit/test_sizing.py::test_which_constraint_binds_does_not_depend_on_capital``
now sweeps 200 seeds and compares the label against ``fee_budget_crossover()``
rather than trusting one lucky ``random.Random(7)`` draw (the original body
failed on 139 of seeds 0-399).

Nothing in this module is imported by the env or the allocator; it is a sizing
calculator that answers "what K can this account run?" before a run is launched.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, Literal

from trader.env.costs import _DP_CHARGE, DEFAULT_MIN_TRADE_VALUE

# The flat demat debit fee, re-exported so nothing downstream retypes "15.34".
# Single source of truth is `costs.py:55` (₹3.50 CDSL + ₹9.50 Zerodha + ₹2.34
# GST), transcribed from zerodha.com/charges.  CLAUDE.md rule 5: cost constants
# are edited in place there, never regenerated, and never duplicated here.
DEMAT_DEBIT_FEE: Final[float] = _DP_CHARGE

# A fully invested, equal-weighted book: the smallest position is exactly 1/K.
DEFAULT_POSITION_SHARE: Final = 1.0

# A "routine trim" — the fraction of a position a rebalance typically moves.
# 20% is the figure `11_cost_defect_and_fix_plan.md` P5 reasons with; it is a
# modelling choice, not a measurement, and every caller may override it.
DEFAULT_REBALANCE_FRACTION: Final = 0.20

# The largest share of a position we are willing to pay to liquidate it.
# 1% is a choice, not a law.  Note it sits ABOVE the f* = 0.6136% crossover
# derived in the module docstring, so at this default the min-trade-value floor
# is the binding constraint at every capital.
DEFAULT_MAX_FEE_FRACTION: Final = 0.01

# Rebalances per year at each cadence in the R5 grid (`rebalance.py`).
REBALANCES_PER_YEAR: Final[dict[str, float]] = {
    "daily": 252.0,
    "weekly": 52.0,
    "monthly": 12.0,
}

# Gross two-sided ANNUAL turnover of the standing best configuration —
# monthly, K=30, 20d — from `audit/R4_R5_RESULTS.md:48` ("Turn" column, 3.73).
# Used only as the default for the capacity table; pass your own measurement.
R5_MONTHLY_TURNOVER: Final = 3.73

BindingConstraint = Literal["flat_fee", "min_trade_value", "both", "max_k", "none"]

# `min_trade_value = 0` disables constraint B entirely.  Reported as this
# sentinel rather than as a real K so it can never be mistaken for a bound that
# happens to coincide with the fee bound (which would read as binding="both").
UNBOUNDED_K: Final = 2**31 - 1

# The closed form floors a ratio of floats.  Without a relative nudge, a K that
# sits exactly on a constraint boundary (fee fraction == f to the last bit) is
# admitted or rejected by whichever way the last ULP fell.  1e-12 is far below
# any meaningful capital resolution and far above double-precision dust.
_FLOOR_TOL: Final = 1e-12


def _floor_bound(bound: float) -> int:
    """Largest integer K satisfying ``K <= bound``, tolerant of float dust."""
    if not math.isfinite(bound):
        return UNBOUNDED_K
    if bound < 0.0:
        return 0
    return min(UNBOUNDED_K, int(math.floor(bound + _FLOOR_TOL * max(1.0, bound))))


@dataclass(frozen=True, slots=True)
class KCapacity:
    """The two bounds on K at one capital, and which of them binds."""

    capital: float
    k_max: int
    k_flat_fee: int
    k_min_trade: int
    # Which RULE is active, read off the real-valued bounds (f vs f*), not off
    # the floored integers above.  The two can disagree: near the crossover the
    # floors may tie at one capital and separate at another, which would make
    # this label capital-dependent while the rule it names is not.  So
    # `binding == "both"` means the bounds genuinely coincide, NOT that the
    # floors happened to land on the same integer.
    binding: BindingConstraint
    position_value: float          # rupees in one position at k_max (0 if k_max == 0)
    fee_fraction_at_k_max: float   # flat fee as a fraction of that position


@dataclass(frozen=True, slots=True)
class CapacityRow:
    """One (capital, K) cell of the capacity table."""

    capital: float
    k: int
    position_value: float
    fee_fraction_per_exit: float
    annual_fee_rupees: float
    annual_drag: float
    rebalanceable: bool            # does a routine trim clear min_trade_value?
    within_fee_budget: bool        # is fee_fraction_per_exit <= max_fee_fraction?


def _check_capital(capital: float) -> float:
    if not capital > 0.0:
        raise ValueError(f"capital must be > 0, got {capital}")
    return float(capital)


def _check_share(smallest_position_share: float) -> float:
    if not 0.0 < smallest_position_share <= 1.0:
        raise ValueError(
            "smallest_position_share must be in (0, 1], got "
            f"{smallest_position_share}"
        )
    return float(smallest_position_share)


def _fee_fraction_of(pos_value: float, dp_charge: float) -> float:
    """``D / P`` — the one implementation of the flat fee as a share of a position.

    Three call sites need it (:func:`fee_fraction_per_exit`,
    :func:`supportable_k_detail`, :func:`capacity_table`) and three copies of one
    formula is how `min_trade_value` came to mean two different things in this
    codebase.  It stays a single function.
    """
    return dp_charge / pos_value


def position_value(
    capital: float,
    k: int,
    *,
    smallest_position_share: float = DEFAULT_POSITION_SHARE,
) -> float:
    """Rupees in the smallest position of a ``k``-name book.

    ``smallest_position_share`` folds a cash floor and inverse-vol dispersion
    into ONE number: a book with ``cash_floor=0.05`` whose lightest name sits at
    0.8x the equal weight has ``s = 0.95 * 0.8 = 0.76``.  It is one knob on
    purpose — two multiplicative knobs for one quantity is the ``vol_column`` /
    ``vol_lookback`` trap `deterministic.py` had to have removed.
    """
    capital = _check_capital(capital)
    smallest_position_share = _check_share(smallest_position_share)
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    return smallest_position_share * capital / float(k)


def fee_fraction_per_exit(
    capital: float,
    k: int,
    *,
    smallest_position_share: float = DEFAULT_POSITION_SHARE,
    dp_charge: float = DEMAT_DEBIT_FEE,
) -> float:
    """Flat demat fee as a fraction of one position, on a full exit.

    ``D·k / (s·C)``.  At ₹1 lakh over 504 equal-weighted names this is
    ``15.34 × 504 / 100000 = 0.0773`` — 7.7% of the position, paid to sell it,
    before any proportional cost.  At K=30 it is 0.46%.
    """
    return _fee_fraction_of(
        position_value(capital, k, smallest_position_share=smallest_position_share),
        dp_charge,
    )


def fee_budget_crossover(
    *,
    min_trade_value: float = DEFAULT_MIN_TRADE_VALUE,
    rebalance_fraction: float = DEFAULT_REBALANCE_FRACTION,
    dp_charge: float = DEMAT_DEBIT_FEE,
) -> float:
    """``f* = D·t/V`` — the fee budget at which the two constraints swap.

    Contains no capital term, which is the point: which constraint binds is a
    property of the *parameters*, not of the account size.  Below ``f*`` the flat
    fee binds; above it the min-trade-value floor does.
    """
    if not min_trade_value > 0.0:
        raise ValueError(f"min_trade_value must be > 0, got {min_trade_value}")
    if not 0.0 < rebalance_fraction <= 1.0:
        raise ValueError(f"rebalance_fraction must be in (0, 1], got {rebalance_fraction}")
    return dp_charge * rebalance_fraction / min_trade_value


def supportable_k_detail(
    capital: float,
    *,
    max_fee_fraction: float = DEFAULT_MAX_FEE_FRACTION,
    min_trade_value: float = DEFAULT_MIN_TRADE_VALUE,
    rebalance_fraction: float = DEFAULT_REBALANCE_FRACTION,
    smallest_position_share: float = DEFAULT_POSITION_SHARE,
    dp_charge: float = DEMAT_DEBIT_FEE,
    max_k: int | None = None,
) -> KCapacity:
    """Both bounds on K at ``capital``, plus which one binds.

    See the module docstring for the derivation.  ``max_k`` clamps the result to
    the investable universe (``active_tickers()`` = 504 today); it is reported as
    the binding constraint when it is the one that bites.

    ``k_max == 0`` is a real answer: it means the account cannot support even one
    name under these parameters.  At the defaults that happens below
    ``V/t = ₹2,500``.
    """
    capital = _check_capital(capital)
    smallest_position_share = _check_share(smallest_position_share)
    if not max_fee_fraction > 0.0:
        raise ValueError(f"max_fee_fraction must be > 0, got {max_fee_fraction}")
    if min_trade_value < 0.0:
        raise ValueError(f"min_trade_value must be >= 0, got {min_trade_value}")
    if not 0.0 < rebalance_fraction <= 1.0:
        raise ValueError(f"rebalance_fraction must be in (0, 1], got {rebalance_fraction}")
    if max_k is not None and max_k < 1:
        raise ValueError(f"max_k must be >= 1 or None, got {max_k}")

    budget = smallest_position_share * capital

    # A: D·K/(s·C) <= f   =>   K <= f·s·C/D
    bound_fee = max_fee_fraction * budget / dp_charge
    k_fee = _floor_bound(bound_fee)
    # B: t·s·C/K >= V     =>   K <= t·s·C/V     (V == 0 disables the constraint)
    bound_mtv = (
        math.inf
        if min_trade_value == 0.0
        else rebalance_fraction * budget / min_trade_value
    )
    k_mtv = UNBOUNDED_K if min_trade_value == 0.0 else _floor_bound(bound_mtv)

    k_max = min(k_fee, k_mtv)
    binding: BindingConstraint
    if k_max < 1:
        k_max = 0
        binding = "none"
    else:
        # Read the label off the REAL-VALUED bounds.  Both carry the same
        # `budget` factor, so this comparison is `f` against `f* = D·t/V` with
        # the capital cancelled — the module docstring's invariant, true by
        # construction instead of true for most inputs.  Comparing the floored
        # integers instead makes the label capital-dependent within a narrow
        # band of fee budgets around f*, where the floors tie at one capital and
        # separate at another (worked example in the module docstring).
        if math.isclose(bound_fee, bound_mtv, rel_tol=_FLOOR_TOL, abs_tol=0.0):
            binding = "both"
        elif bound_fee < bound_mtv:
            binding = "flat_fee"
        else:
            binding = "min_trade_value"
        if max_k is not None and max_k < k_max:
            k_max = max_k
            binding = "max_k"

    if k_max < 1:
        return KCapacity(capital, 0, k_fee, k_mtv, "none", 0.0, float("inf"))
    pv = position_value(capital, k_max, smallest_position_share=smallest_position_share)
    return KCapacity(
        capital, k_max, k_fee, k_mtv, binding, pv, _fee_fraction_of(pv, dp_charge)
    )


def max_supportable_k(
    capital: float,
    *,
    max_fee_fraction: float = DEFAULT_MAX_FEE_FRACTION,
    min_trade_value: float = DEFAULT_MIN_TRADE_VALUE,
    rebalance_fraction: float = DEFAULT_REBALANCE_FRACTION,
    smallest_position_share: float = DEFAULT_POSITION_SHARE,
    dp_charge: float = DEMAT_DEBIT_FEE,
    max_k: int | None = None,
) -> int:
    """Largest K this capital supports.  Closed form; see the module docstring.

        K_max = floor( min( f·s·C/D , t·s·C/V ) )

    Constraint A is the flat demat fee on a full exit (``D`` per scrip per selling
    day, from :data:`DEMAT_DEBIT_FEE` ← ``costs._DP_CHARGE``); constraint B is the
    rupee floor a routine trim must clear
    (``V`` from :data:`~trader.env.costs.DEFAULT_MIN_TRADE_VALUE`).  Neither
    number is retyped here.

    Returns 0 when no K is supportable.  Monotone non-decreasing in ``capital``.
    """
    return supportable_k_detail(
        capital,
        max_fee_fraction=max_fee_fraction,
        min_trade_value=min_trade_value,
        rebalance_fraction=rebalance_fraction,
        smallest_position_share=smallest_position_share,
        dp_charge=dp_charge,
        max_k=max_k,
    ).k_max


def names_sold_per_rebalance(
    k: int,
    rebalances_per_year: float,
    turnover: float,
    *,
    avg_sell_share: float = 1.0,
) -> float:
    """Distinct names leaving the demat on one rebalance day.

    This is the quantity the flat fee actually bills for, and no control in the
    system constrains it (`11_cost_defect_and_fix_plan.md`: "what drives it is
    the *number of distinct names sold per day*").

    ``turnover`` is the project's **gross two-sided ANNUAL** figure — the "Turn"
    column of `audit/R4_R5_RESULTS.md`, the ``sum |Δw|`` convention documented in
    `deterministic.py` and `panel_env.py` — so one-sided turnover per rebalance is
    ``u = turnover / (2·R)``.  If a name that is sold is sold in an average slice
    of ``a`` of its position, the count is

        n = k · min(1, u / a),

    capped because you cannot sell more distinct names than you hold.

    ``avg_sell_share = 1.0`` (the default) models sells as **full exits**, which
    is the right picture for a top-K rotation book and is a **lower bound** on the
    name count — partial trims spread the same rupees over more scrips and cost
    strictly more.

    ``a`` is NOT a constant, and the measurements say so.  Backing it out of the
    equal-weight 504-name runs — sell name-days from
    `11_cost_defect_and_fix_plan.md:37-38` (83,933 daily / 10,250 monthly over
    1,860 steps, a pre-P1 run) against gross two-sided annual turnover from
    `audit/R4_R5_RESULTS.md:46,53` (2.39 daily / 0.92 monthly, a *different*
    post-P1 run) — gives

        monthly  115.7 names/rebalance (23.0% of the book)  ->  a ≈ 0.167
        daily     45.1 names/rebalance ( 9.0% of the book)  ->  a ≈ 0.053

    A 3.1x spread across cadence.  Trading more often moves each name in smaller
    slices, so the fee is paid on more scrips per rupee of turnover — which is
    the daily arm's problem restated.  Regenerate both with
    ``tests/unit/test_sizing.py::test_implied_sell_share_is_cadence_dependent``.

    Two caveats bind on those figures: the counts and the turnovers come from two
    different runs under two different cost models, and neither is a calibration
    this module is entitled to bake in.  So ``a`` stays an explicit argument with
    a full-exit default, and any drag quoted at ``a = 1.0`` is a **floor**, 6x
    (monthly) to 19x (daily) below what an equal-weight book actually paid.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if not rebalances_per_year > 0.0:
        raise ValueError(f"rebalances_per_year must be > 0, got {rebalances_per_year}")
    if turnover < 0.0:
        raise ValueError(f"turnover must be >= 0, got {turnover}")
    if not 0.0 < avg_sell_share <= 1.0:
        raise ValueError(f"avg_sell_share must be in (0, 1], got {avg_sell_share}")
    one_sided = turnover / (2.0 * rebalances_per_year)
    return float(k) * min(1.0, one_sided / avg_sell_share)


def fee_drag_estimate(
    capital: float,
    k: int,
    rebalances_per_year: float,
    turnover: float,
    *,
    avg_sell_share: float = 1.0,
    dp_charge: float = DEMAT_DEBIT_FEE,
) -> float:
    """Expected annual flat-fee drag, as a fraction of capital.

        drag = D · R · k · min(1, turnover/(2·R) / a) / C

    This is what makes the K tradeoff legible: a larger K buys diversification
    and pays for it linearly in flat fees, while the proportional costs this
    module ignores do not care about K at all.  Nothing else in the expression
    depends on capital, so **drag is inversely proportional to the account** —
    the identical strategy costs 10x more at ₹1 lakh than at ₹10 lakh.

    ``rebalances_per_year`` CANCELS outside the saturation cap
    -----------------------------------------------------------
    Whenever ``turnover/(2·R) <= a`` the ``min`` is inactive and the R in the
    numerator meets the R inside it::

        drag = D · R · k · turnover/(2·R·a) / C = D · k · turnover / (2·a·C)

    so at a **fixed annual turnover** the rupee bill is the same at daily,
    weekly and monthly cadence: ``fee_drag_estimate(1e5, 30, R, 3.73)`` returns
    0.858273% for R = 12, 52 and 252 alike.  The bill is set by K and by ANNUAL
    TURNOVER, not by K and cadence — the 19.4x between the daily and monthly
    rows of the P5 table is entirely the turnover difference (72.30 vs 3.73).
    R is still load-bearing, but only through the cap: once
    ``turnover > 2·R·a`` every rebalance sells the whole book and the bill grows
    linearly in R (``test_cadence_matters_only_inside_the_saturation_cap``).

    Ignores STT, stamp, SEBI, exchange and GST by design; see the module
    docstring.  It is a drag *attributable to K*, not a total cost.
    """
    capital = _check_capital(capital)
    n = names_sold_per_rebalance(
        k, rebalances_per_year, turnover, avg_sell_share=avg_sell_share
    )
    return dp_charge * rebalances_per_year * n / capital


DEFAULT_CAPITALS: Final[tuple[float, ...]] = (1e5, 2.5e5, 5e5, 1e6, 5e6)
DEFAULT_KS: Final[tuple[int, ...]] = (20, 30, 40)


def capacity_table(
    capitals: Sequence[float] = DEFAULT_CAPITALS,
    ks: Sequence[int] = DEFAULT_KS,
    *,
    rebalances_per_year: float = REBALANCES_PER_YEAR["monthly"],
    turnover: float = R5_MONTHLY_TURNOVER,
    avg_sell_share: float = 1.0,
    max_fee_fraction: float = DEFAULT_MAX_FEE_FRACTION,
    min_trade_value: float = DEFAULT_MIN_TRADE_VALUE,
    rebalance_fraction: float = DEFAULT_REBALANCE_FRACTION,
    smallest_position_share: float = DEFAULT_POSITION_SHARE,
    dp_charge: float = DEMAT_DEBIT_FEE,
) -> list[CapacityRow]:
    """The (capital x K) grid: fee per exit, annual flat-fee drag, and viability.

    Defaults are the P5 grid — capital in {1L, 2.5L, 5L, 10L, 50L}, K in
    {20, 30, 40} — at monthly cadence on the standing best configuration's
    turnover (3.73, `audit/R4_R5_RESULTS.md:48`).
    """
    rows: list[CapacityRow] = []
    for c in capitals:
        for k in ks:
            pv = position_value(c, k, smallest_position_share=smallest_position_share)
            frac = _fee_fraction_of(pv, dp_charge)
            drag = fee_drag_estimate(
                c,
                k,
                rebalances_per_year,
                turnover,
                avg_sell_share=avg_sell_share,
                dp_charge=dp_charge,
            )
            rows.append(
                CapacityRow(
                    capital=float(c),
                    k=int(k),
                    position_value=pv,
                    fee_fraction_per_exit=frac,
                    annual_fee_rupees=drag * float(c),
                    annual_drag=drag,
                    rebalanceable=rebalance_fraction * pv >= min_trade_value,
                    within_fee_budget=frac <= max_fee_fraction,
                )
            )
    return rows


def format_capacity_table(rows: Sequence[CapacityRow]) -> str:
    """Markdown rendering of :func:`capacity_table`, for pasting into a report."""
    head = (
        "| Capital | K | Position ₹ | Fee/exit | Annual fee ₹ | Annual drag | "
        "Trim ≥ ₹min | Within budget |\n"
        "|---|---|---|---|---|---|---|---|"
    )
    body = "\n".join(
        f"| {r.capital:,.0f} | {r.k} | {r.position_value:,.0f} | "
        f"{r.fee_fraction_per_exit:.4%} | {r.annual_fee_rupees:,.0f} | "
        f"{r.annual_drag:.4%} | {'yes' if r.rebalanceable else 'NO'} | "
        f"{'yes' if r.within_fee_budget else 'NO'} |"
        for r in rows
    )
    return f"{head}\n{body}"
