"""Deterministic top-K / inverse-vol / capped / turnover-budgeted allocator.

This is the R5 allocator of `10_architecture_revamp.md` §5.  It has **no
learned parameters**: every knob is a number in :class:`AllocatorParams`, so
when it fails to beat equal-weight net of cost and tax the reason is readable
in hours rather than inferred from a training curve.

Pipeline (every step vectorised NumPy — this runs inside the env loop):

1. candidates  = tradeable ∧ finite ``r_hat`` ∧ ``vol > 0`` ∧ ``sector_id > 0``
2. rank by ``r_hat`` descending, take the top ``k``
3. raw weights ∝ ``1 / vol`` over the chosen names, scaled to ``1 − cash_floor``
4. per-name cap at ``max_name_weight`` and per-sector cap at
   ``max_sector_weight``, imposed together by water-filling: the excess a cap
   refuses is re-offered to the names still free of both caps, and cash
   absorbs it only when every chosen name is pinned (see :func:`_apply_caps`)
5. (folded into 4 — the two caps cannot be applied in sequence; alternating
   them does not converge)
6. ``cash_floor`` — whatever the caps could not place stays in cash, which is
   therefore ≥ ``cash_floor``
7. per-name **no-trade band** — a name whose target is within
   ``no_trade_band`` of its current weight is pinned *to its current weight*,
   so the delta this function returns for it is exactly ``0.0``
   (:func:`_apply_no_trade_band`).  That is a statement about weight space and
   **not** about the order book: the env re-creates a one-share order for some
   pinned names anyway, because ``obs["portfolio"]`` is float32 and
   ``panel_env`` floors ``target_value / open``.  Measured leak and the env-side
   fix in :func:`_apply_no_trade_band`
8. turnover budget — if the gross equity move from ``current_w`` exceeds
   ``turnover_budget``, move along the straight line towards the target so the
   budget binds exactly
9. return ``[cash, w_1..w_N]`` summing to 1, all ≥ 0

Units
-----
``vol`` is any positive volatility on a consistent scale across names —
inverse-vol weights are renormalised, so annualised vs daily makes no
difference to the output.  The project's panels carry ``realized_vol_20d``
**annualised** (``features.py`` multiplies the rolling std by √252); pass that.

``turnover_budget`` uses the env's *gross two-sided* convention
(`panel_env.py`, "Turnover"): the sum of |Δw| over equities, so a full
rotation of an all-equity book is 2.0 and moving from all-cash into a fully
invested book is 1.0.

``no_trade_band`` is a **one-sided** per-name deviation, in the same NAV
fraction units as a weight: 0.005 means "do not touch a name unless its
target is more than 0.5% of NAV away from where it already sits".  It is the
only control in this package on the *number of names traded*, which is the
quantity the flat ₹15.34 demat debit is billed on (`costs.py:55`, per distinct
scrip per selling day).  ``turnover_budget`` constrains rupees moved and is
the wrong instrument for a flat per-scrip fee — see
`11_cost_defect_and_fix_plan.md` §P3, and the measurement in
``tests/unit/test_allocator.py::test_band_cuts_name_count_faster_than_turnover``.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

_EPS = 1e-12


@dataclass(frozen=True)
class AllocatorParams:
    """Every free parameter of the allocator.  Nothing here is learned."""

    k: int = 30                      # names to hold
    max_name_weight: float = 0.10    # matches env's max_weight_per_name
    max_sector_weight: float = 0.25
    turnover_budget: float = 0.30    # max gross traded fraction of NAV per rebalance
    # Per-name no-trade band, as a fraction of NAV.  A name is traded only when
    # |target - current| exceeds it; otherwise its target IS its current weight,
    # so no order is produced for it.  0.0 (the default) switches the mechanism
    # off entirely and `_finish` takes the pre-band code path unchanged, so the
    # standing R5 result is reproduced bit for bit until the band is turned on
    # (test_band_zero_is_bit_identical_to_the_pre_band_path).
    no_trade_band: float = 0.0
    cash_floor: float = 0.0
    # Load-bearing: names the panel column the caller must feed, via
    # `trader.env.allocator_env.vol_column_for` -> f"realized_vol_{n}d".
    # It was previously "informational" and read by nothing, while a separate
    # `vol_column` key decided the actual column -- two knobs for one thing,
    # free to disagree silently. That is CLAUDE.md's `min_trade_value: 500`
    # trap, so the second knob is gone and this one names the column.
    vol_lookback: int = 20

    def __post_init__(self) -> None:
        if self.k < 1:
            raise ValueError(f"k must be >= 1, got {self.k}")
        if not 0.0 < self.max_name_weight <= 1.0:
            raise ValueError(f"max_name_weight must be in (0, 1], got {self.max_name_weight}")
        if not 0.0 < self.max_sector_weight <= 1.0:
            raise ValueError(
                f"max_sector_weight must be in (0, 1], got {self.max_sector_weight}"
            )
        if self.turnover_budget < 0.0:
            raise ValueError(f"turnover_budget must be >= 0, got {self.turnover_budget}")
        if not 0.0 <= self.no_trade_band < 1.0:
            raise ValueError(f"no_trade_band must be in [0, 1), got {self.no_trade_band}")
        if not 0.0 <= self.cash_floor < 1.0:
            raise ValueError(f"cash_floor must be in [0, 1), got {self.cash_floor}")


def allocate(
    r_hat: np.ndarray,
    vol: np.ndarray,
    mask: np.ndarray,
    sector_ids: np.ndarray,
    current_w: np.ndarray,
    params: AllocatorParams,
) -> np.ndarray:
    """Target weights ``[N+1]`` (index 0 = cash) from a return signal.

    Parameters
    ----------
    r_hat:
        ``[N]`` predicted forward return, any horizon; NaN = no prediction.
    vol:
        ``[N]`` realised vol on a consistent scale; ``<= 0`` or non-finite =
        never selected.
    mask:
        ``[N]`` bool, tradeable today.
    sector_ids:
        ``[N]`` int, 1-indexed.  ``0`` (unknown) is never selected — it has no
        sector to cap against.
    current_w:
        ``[N+1]`` current weights, index 0 = cash.  Only the turnover budget
        reads it, and it is renormalised first, so it need not sum to exactly
        1 — pass ``obs["portfolio"]`` straight through.  That vector routinely
        does not: its space is ``Box(0, 1)``, so when the env's cash goes
        slightly negative (it sizes against a NAV marked at the previous close
        and fills at the open, so a gap up overspends) the clipped cash reads
        0.0 and the vector sums to ~1.01.  Measured: 1.009760 on the synthetic
        panel of ``tests/integration/test_panel_env.py``.
    params:
        :class:`AllocatorParams`.
    """
    r_hat = np.asarray(r_hat, dtype=np.float64)
    vol = np.asarray(vol, dtype=np.float64)
    mask = np.asarray(mask, dtype=np.bool_)
    sector_ids = np.asarray(sector_ids, dtype=np.int64)
    current_w = np.asarray(current_w, dtype=np.float64)
    N = r_hat.shape[0]
    if vol.shape != (N,) or mask.shape != (N,) or sector_ids.shape != (N,):
        raise ValueError("r_hat, vol, mask and sector_ids must all be shape [N]")
    if current_w.shape != (N + 1,):
        raise ValueError(f"current_w must be shape [N+1]={N + 1}, got {current_w.shape}")

    investable = 1.0 - params.cash_floor

    # 1. candidates
    cand = mask & np.isfinite(r_hat) & np.isfinite(vol) & (vol > 0.0) & (sector_ids > 0)
    n_cand = int(cand.sum())
    if n_cand == 0:
        return _finish(np.zeros(N), current_w, params)

    # 2. top-k by r_hat (argsort on -r_hat, candidates only; ties by index)
    k = min(params.k, n_cand)
    cand_idx = np.flatnonzero(cand)
    order = np.argsort(-r_hat[cand_idx], kind="stable")
    chosen = cand_idx[order[:k]]

    # 3. inverse-vol raw weights over the chosen names
    base = np.zeros(N, dtype=np.float64)
    base[chosen] = 1.0 / vol[chosen]

    # 4/5. name cap and sector cap, applied as one water-filling pass
    w = _apply_caps(base, chosen, sector_ids, params, investable)

    # 6. cash floor is implicit: sum(w) <= investable, so cash >= cash_floor
    # 7/8/9. no-trade band, turnover budget, then assemble [cash, w]
    return _finish(w, current_w, params)


# ── helpers ───────────────────────────────────────────────────────────────────


def _apply_caps(
    base: np.ndarray,
    chosen: np.ndarray,
    sector_ids: np.ndarray,
    params: AllocatorParams,
    investable: float,
) -> np.ndarray:
    """Water-fill ``investable`` over ``chosen`` ∝ ``base``, under both caps.

    ``base`` is the *unnormalised* preference (here ``1 / vol``), zero off
    ``chosen``.  Each round distributes whatever is still unplaced over the
    names that are still **free** in proportion to ``base``, then freezes, in
    this order and **within the same round**:

    * any free name above ``max_name_weight`` — pinned at the cap;
    * then every chosen name of any sector above ``max_sector_weight`` — the
      whole sector (frozen names included) is scaled down to sit exactly on
      its cap, and frozen there.

    Freezing is monotone: a frozen name is never thawed, and the only later
    change to it is a downward sector rescale, so it can never re-breach the
    name cap.  The loop therefore terminates in at most one round per name.

    Why the two checks must run in the same round
    ---------------------------------------------
    They used to be `if name_cap_fires: ...; continue`, which jumped past the
    sector check.  Whenever the name cap froze the *last* free names, the next
    round found ``b_sum == 0`` and took the early ``break``, so the sector cap
    was never evaluated for them.  Measured breaches at the shipped defaults
    (k=30, name 0.10, sector 0.25): 23 candidates split 3/20 across two sectors
    returned sector 1 = 0.30, and 8 candidates in a single sector returned
    0.80 — a 3.2x breach.  Both are regression-tested in
    ``tests/unit/test_allocator.py``.

    ``_enforce_sector_caps`` runs once more after the loop.  With the
    fall-through above it is a no-op on every input we can construct; it exists
    so the postcondition holds unconditionally rather than by argument,
    including if the iteration bound is ever hit.

    Weight the caps refuse to place is simply not placed: the return sums to
    ``≤ investable`` and the caller turns the shortfall into cash.  That is the
    ``k · max_name_weight < 1`` case, and the "every chosen name sits in a
    capped sector" case.
    """
    name_cap = params.max_name_weight
    sec_cap = params.max_sector_weight
    n_sec = int(sector_ids.max()) + 1

    chosen_mask = np.zeros(base.shape[0], dtype=np.bool_)
    chosen_mask[chosen] = True
    free = chosen_mask.copy()
    w = np.zeros_like(base)

    # One round can freeze at least one name or one sector, never zero (it
    # breaks instead), so this bound is never the reason the loop ends.
    for _ in range(int(chosen.size) + n_sec + 2):
        placed = float(w[chosen_mask & ~free].sum())
        remaining = investable - placed
        b = np.where(free, base, 0.0)
        b_sum = float(b.sum())
        if remaining <= _EPS or b_sum <= _EPS:
            # Nothing left to place, or nobody left to place it on.
            w[free] = 0.0
            break
        w[free] = remaining * b[free] / b_sum

        froze = False

        over = free & (w > name_cap + _EPS)
        if np.any(over):
            w[over] = name_cap
            free &= ~over
            froze = True
            # NO `continue` here: falling through to the sector check in the
            # same round is what makes the sector cap hold when the name cap
            # freezes the last free names.  See the docstring.

        sec_w = np.bincount(sector_ids, weights=w, minlength=n_sec)
        sec_over = sec_w > sec_cap + _EPS
        sec_over[0] = False          # sector 0 is never chosen; never capped
        hit = sec_over[sector_ids] & chosen_mask
        if np.any(hit):
            scale = np.ones(n_sec, dtype=np.float64)
            scale[sec_over] = sec_cap / np.maximum(sec_w[sec_over], _EPS)
            # Scaling is downward only, so this cannot breach a name cap.
            w = np.where(hit, w * scale[sector_ids], w)
            free &= ~hit
            froze = True

        if not froze:
            break   # both caps hold and nothing was frozen: fixed point

    w = _enforce_sector_caps(w, chosen_mask, sector_ids, sec_cap, n_sec)
    out: np.ndarray = np.clip(w, 0.0, name_cap)
    return out


def _enforce_sector_caps(
    w: np.ndarray,
    chosen_mask: np.ndarray,
    sector_ids: np.ndarray,
    sec_cap: float,
    n_sec: int,
) -> np.ndarray:
    """Scale every over-cap sector's chosen names down onto the cap.

    Downward-only, so it cannot breach the name cap, and the weight it removes
    correctly becomes cash in :func:`_finish`.  This is a postcondition guard,
    not the mechanism: :func:`_apply_caps` should already have converged.
    """
    sec_w = np.bincount(sector_ids, weights=w, minlength=n_sec)
    sec_over = sec_w > sec_cap + _EPS
    sec_over[0] = False
    if not np.any(sec_over):
        return w
    scale = np.ones(n_sec, dtype=np.float64)
    scale[sec_over] = sec_cap / np.maximum(sec_w[sec_over], _EPS)
    hit = sec_over[sector_ids] & chosen_mask
    return np.where(hit, w * scale[sector_ids], w)


def _current_equity(current_w: np.ndarray) -> np.ndarray:
    """The incoming book as a non-negative equity vector summing to ``<= 1``.

    Renormalises so a float32 ``obs["portfolio"]`` that sums to 1 ± 1e-6 — or
    to 1.01, which the env really does emit (see :func:`allocate`) — does not
    leak into the returned cash weight.

    One function, called by both :func:`_finish` and :func:`band_suppression`,
    so the band diagnostic can never disagree with the book the band actually
    ran against.  Two copies of this arithmetic is the shape of the
    ``min_trade_value`` trap in CLAUDE.md.
    """
    cur_eq: np.ndarray = np.clip(current_w[1:], 0.0, None)
    cur_total = float(cur_eq.sum()) + max(float(current_w[0]), 0.0)
    if cur_total > _EPS:
        cur_eq = cur_eq / cur_total
    return cur_eq


def _clearing_prefix(dev: np.ndarray, budget: float, band: float) -> tuple[np.ndarray, float]:
    """The largest *prefix* of the biggest legs whose executed moves clear the band.

    ``dev`` is ``[M]`` positive move magnitudes competing for ``budget`` rupees
    (as a NAV fraction).  Returns ``(keep, scale)``: the legs to execute and the
    common factor they execute at, so leg ``i`` moves ``scale * dev[i]``.

    Sort descending.  For a prefix of size ``m`` with cumulative gross ``G_m``
    the whole prefix executes at ``s_m = min(1, budget / G_m)`` and its smallest
    member moves ``f(m) = s_m · dev_m``.  ``s_m`` and ``dev_m`` are both
    non-increasing in ``m``, so ``f`` is, and ``{m : f(m) > band}`` is a prefix
    — take the largest.  When the budget does not bind (``s = 1``) this
    degenerates to the plain band test.

    Prefix, not maximum cardinality — and the difference is real
    -----------------------------------------------------------
    The monotonicity argument proves this is the largest feasible **prefix of
    the descending sort**.  It does *not* prove it is the largest feasible
    **subset**, and it is not: dropping a large leg frees budget for several
    small ones.  Tie-free counterexample — ``dev = [0.0333, 0.0333, 0.0333,
    0.010, 0.010, 0.010]``, ``budget = 0.05``, ``band = 0.005``: this function
    keeps 3 legs at ``scale = 0.500501``, while ``{0.0333, 0.0333, 0.010,
    0.010, 0.010}`` is feasible at ``scale = 0.517598`` — 5 legs, every one
    moving >= 0.005176, gross exactly 0.05.  An exhaustive subset search finds a
    strictly larger feasible set in 59 of 3000 random draws (2.0%); both figures
    are regenerated by
    ``tests/unit/test_allocator.py::test_clearing_prefix_is_a_prefix_not_a_maximum_subset``.

    That is a deliberate choice, not a bug to fix later: the legs are ranked by
    deviation, so preferring the prefix spends the budget on the names furthest
    from target, and it errs toward **fewer** trades — which is what a per-scrip
    flat fee wants.  Maximising the count would be maximising the number of
    ₹15.34 debits.  The postconditions the caller relies on hold either way and
    are what the tests assert: every executed leg moves strictly more than
    ``band``, and the executed gross is ``<= budget``.
    """
    order = np.argsort(-dev, kind="stable")
    g = np.cumsum(dev[order])
    scale = np.where(g > budget, budget / np.maximum(g, _EPS), 1.0)
    m = int(np.count_nonzero(scale * dev[order] > band))
    keep = np.zeros(dev.shape[0], dtype=np.bool_)
    if m == 0:
        return keep, 0.0
    keep[order[:m]] = True
    return keep, float(scale[m - 1])


def _apply_no_trade_band(
    w_eq: np.ndarray, cur_eq: np.ndarray, params: AllocatorParams
) -> np.ndarray:
    """Suppress names inside the band; spend the turnover budget on the rest.

    Returns the equity target.  A suppressed name is assigned ``cur_eq[i]``
    **by construction** — its delta is the literal ``0.0``, never a shrunken
    delta — so ``target - current`` is exactly zero for it.

    What that does NOT say: "and therefore no order"
    ------------------------------------------------
    It said exactly that until it was measured, and the measurement refuted it.
    A zero delta **in weight space** is not a zero order, because the env holds
    shares, not weights, and two lossy conversions sit between the two:

    * ``obs["portfolio"]`` is cast to **float32** (`panel_env._build_obs`), so
      the weight this function pins a name to reproduces its position only to
      ~6e-8 relative;
    * ``panel_env._step_target`` sized the order with
      ``floor(target_value / open)``, and ``floor(400 x (1 - 6e-8))`` is 399 —
      a one-share SELL, worth the share price, clearing both the 0.5-share
      integrality guard and ``min_trade_value`` on any name above ₹500, and
      paying the flat ₹15.34 demat debit (`costs.py:55`).

    Measured in the real env (PanelTradingEnv + monthly RebalanceSchedule +
    ZerodhaEquityDeliveryCostModel, band 0.005, counts taken from the cost
    model): with ``open == prev_close`` exactly, **22 of 86 pinned names (25.6%)
    still traded, every one a sell**.  The env now snaps a request back to the
    held position when it rounds to it, which removes that class entirely
    (`tests/unit/test_weight_to_share_orders.py`,
    `scripts/probe_share_rounding.py`).

    A second class survives and is **not** fixed here.  Weights are marked at
    the previous close and filled at the open, so pinning a name across an
    overnight gap asks the env for a move of ``gap x position``.  Below half a
    share that is snapped away; above it, it is a genuine order on a name this
    function reports as suppressed.  Whether it bites is a property of the book
    geometry, not of the band: at K=30 / ₹10 lakh a position is ₹33k against a
    ₹704 median close, so a 1% gap is 0.47 of a share and is absorbed; at ₹1
    crore the same 1% is 4.7 shares and executes.  **So ``n_suppressed`` is an
    upper bound on the orders removed, and at large capital a loose one.**  The
    only real repair is to stop deciding orders from a weight vector — the band
    would have to reach the env as a set of names not to touch — and that
    changes the env's interface, so it is recorded rather than done
    (`audit/P3_P4_P5_INTEGRATION.md`).

    "Exactly" is also against ``cur_eq``, the book *after*
    :func:`_current_equity` renormalises it, which is not always the book the
    env holds: when the env's cash goes negative its ``Box(0, 1)`` clip reports
    a ``portfolio`` summing to ~1.0098 (measured, see :func:`allocate`), and no
    vector this function can return both sums to 1 and reproduces that book.
    Every name is then trimmed by the same ~1% — with the band off as much as
    on, so it is a property of the env's clipped observation, not of the band.

    Order: band first, budget on what survives — but tested against the
    move the budget will actually permit
    -----------------------------------------------------------------------
    Step 1 is the plain band on the *raw* deviation: a name is a candidate only
    when ``|target − current| > band``.  Step 2 lets the turnover budget bind on
    what that leaves, which is the whole point of doing the band first —
    suppressing names removes their delta from the gross, so the budget has a
    smaller bill and moves the survivors further, which is what makes each
    ₹15.34 worth paying.

    Step 3 is not optional, and it is where the naive orderings fail:

    * *Band, then budget, and stop.*  When the budget binds — the normal case,
      not a corner: the R5 monthly grid runs at ~0.31 gross per rebalance
      against a 0.30 budget (`audit/R4_R5_RESULTS.md`, 3.73 turnover/yr over 12
      rebalances) — every surviving delta is scaled by ``lambda = budget /
      gross``, and a name that deviated by just over the band now *moves* by far
      less than the band.  That is precisely the trade the band refused,
      reinstated by the budget one line later.
    * *Budget, then band.*  The budget freed by the suppressed names is never
      re-offered to anyone: the book under-trades and still pays a per-name fee
      for whatever sits just above the scaled band.
    * *Suppress, rescale, suppress, … to a fixed point.*  It drops names in
      batches, and on a symmetric move it drops them **all**.  Rotating 30 names
      of 0.0333 each under a 0.30 budget gives ``lambda = 0.15``, so every leg
      moves 0.005; against a 0.011 band all 60 legs fall short in one round and
      the book freezes — permanently, since the next rebalance sees the same
      target and the same book.

    So the surviving candidates are ranked and cut with :func:`_clearing_prefix`
    instead, which keeps the largest set whose executed moves all clear the
    band.  On that same rotation it trades 13 names out and 13 in at 0.0115
    each rather than freezing (``test_band_does_not_freeze_a_book_it_cannot
    _rotate_in_one_step``).

    Postcondition: ``gross <= turnover_budget``, and **every** name with a
    nonzero delta — buy or sell — moves strictly more than ``band``.

    Sells and buys are cut separately, and why that is not a detail
    --------------------------------------------------------------
    Ranking all legs together by ``|Δ|`` picks a set that can be almost all
    buys — on the rotation above, where every leg is the same size, the tie
    break hands the top 29 places to the 29 buys — and buys have to be funded.
    With no idle cash and no sells selected there is nothing to fund them with,
    the whole set collapses to zero, and the book freezes for a *second* reason.
    So each side gets its own cut of the budget, in the proportion the two sides
    already stand in (``lambda · gross_sell`` and ``lambda · gross_buy``), which
    is exactly what proportional scaling would have given them.

    The buy side is additionally capped at the cash it can actually raise: idle
    cash plus the proceeds of the sells that survived their own cut.  Without
    a band the result is a convex combination of two vectors that each sum to
    ``<= 1`` and cannot overspend; with one it can, because the band suppresses
    small *sells* as readily as small buys and a fully invested book that keeps
    its trims but takes its adds needs cash it never raised.  Sells are never
    capped by this: they are what raises the cash, and they are the leg the flat
    fee is billed on (delivery brokerage is 0 and ``dp_charge`` is sell-side,
    `costs.py:87-96`).  The ``1 - 1e-12`` haircut leaves ~₹1e-6 of a ₹10 lakh
    book unspent so the assembled vector cannot round *above* 1.0 and trip the
    renormalisation guard in :func:`_finish` — which would nudge every
    suppressed name off its current weight and hand the env an order for each.
    """
    band = params.no_trade_band
    raw = w_eq - cur_eq
    cand = np.abs(raw) > band
    if not bool(cand.any()):
        return cur_eq.copy()

    gross = float(np.abs(raw[cand]).sum())
    lam = 1.0 if gross <= params.turnover_budget else params.turnover_budget / gross

    delta = np.zeros_like(raw)

    sell = cand & (raw < 0.0)
    sell_dev = -raw[sell]
    keep_s, scale_s = _clearing_prefix(sell_dev, lam * float(sell_dev.sum()), band)
    delta[sell] = np.where(keep_s, -sell_dev * scale_s, 0.0)
    proceeds = float(sell_dev[keep_s].sum()) * scale_s

    buy = cand & (raw > 0.0)
    buy_dev = raw[buy]
    free_cash = max(1.0 - float(cur_eq.sum()), 0.0)
    funds = (free_cash + proceeds) * (1.0 - 1e-12)
    keep_b, scale_b = _clearing_prefix(
        buy_dev, min(lam * float(buy_dev.sum()), funds), band
    )
    delta[buy] = np.where(keep_b, buy_dev * scale_b, 0.0)

    out: np.ndarray = cur_eq + delta
    return out


def _finish(w_eq: np.ndarray, current_w: np.ndarray, params: AllocatorParams) -> np.ndarray:
    """Apply the no-trade band and the turnover budget, and assemble ``[cash, w]``.

    Turnover budget semantics: with ``g = Σ|w_eq − cur_eq|`` (gross, two-sided,
    equities only), if ``g > budget`` the returned equity vector is
    ``cur_eq + (budget / g) · (w_eq − cur_eq)`` — the point on the segment
    from the current book to the target at which the budget binds exactly.
    Cash is whatever is left, so the result sums to 1 by construction; it is
    non-negative because both endpoints are.  The cash floor is a property of
    the *target*: a partial move from a book below the floor lands between the
    two, and reaches the floor over successive rebalances.

    The budget binds on ``current_w`` marked at the **previous close**, which
    is what the env reports, while the env **fills at the open**.  The value
    actually traded therefore differs from the value budgeted by the overnight
    gap on the traded names, and ``info["turnover"]`` can land a few percent
    above ``turnover_budget`` (measured 0.300 → 0.305 and 0.250 → 0.260 on a
    1.5%-daily-vol synthetic panel).  It is not closable inside this signature,
    which is handed no price at which the trade will fill, and it is
    second-order against what the budget is for: stopping a signal flip from
    rotating the whole book, which is a turnover of 2.0.

    ``no_trade_band > 0`` replaces the single scaling step above with
    :func:`_apply_no_trade_band`, which decides which names to touch and how
    far to move them **together** — the band has to be tested against the move
    the budget will actually permit, not against the unscaled deviation.  With
    the band on, the budget binds as an upper bound rather than exactly:
    suppressing a name removes its delta from the gross, so the result can, and
    usually does, sit under the budget.  ``no_trade_band == 0.0`` (the default)
    takes the ``else`` branch below untouched, which is why every pre-band
    number reproduces bit for bit.
    """
    cur_eq = _current_equity(current_w)

    if params.no_trade_band > 0.0:
        w_eq = _apply_no_trade_band(w_eq, cur_eq, params)
    else:
        delta = w_eq - cur_eq
        gross = float(np.abs(delta).sum())
        if gross > params.turnover_budget + 1e-12 and gross > _EPS:
            w_eq = cur_eq + (params.turnover_budget / gross) * delta

    # Both branches keep every weight in [0, cur_eq[i] ∨ w_eq[i]] and the sum at
    # or below 1, so neither guard below fires on any input we can construct;
    # they hold the postcondition unconditionally.  Note that a `w_eq / eq_sum`
    # here would break the band's exactness (see _apply_no_trade_band).
    w_eq = np.maximum(w_eq, 0.0)
    eq_sum = float(w_eq.sum())
    if eq_sum > 1.0:
        w_eq = w_eq / eq_sum
        eq_sum = 1.0
    out = np.empty(w_eq.shape[0] + 1, dtype=np.float64)
    out[0] = 1.0 - eq_sum
    out[1:] = w_eq
    return out


# ── band diagnostics (not wired into anything) ───────────────────────────────


@dataclass(frozen=True)
class BandSuppression:
    """What ``params.no_trade_band`` removed from one rebalance's order list.

    Counts are in **weight space**: a name counts as traded when its target
    differs from its current weight by more than 1e-12 of NAV.  The env applies
    a second, independent filter downstream — ``min_trade_value`` (₹500,
    `costs.py`) and a 0.5-share integrality guard — so ``n_traded`` is an upper
    bound on the orders that actually reach the ledger, not a prediction of
    them.  The two filters are different instruments: the band is a fraction of
    NAV and scales with capital, ``min_trade_value`` is a rupee floor and does
    not.
    """

    n_traded_unbanded: int      # names that would move with the band switched off
    n_traded: int               # names that move with the band in force
    n_suppressed: int           # the difference: orders the band did not send
    n_sold_unbanded: int        # sells only — what the flat demat fee bills for
    n_sold: int
    gross_unbanded: float       # Σ|Δw|, the quantity turnover_budget constrains
    gross: float


def band_suppression(
    r_hat: np.ndarray,
    vol: np.ndarray,
    mask: np.ndarray,
    sector_ids: np.ndarray,
    current_w: np.ndarray,
    params: AllocatorParams,
) -> BandSuppression:
    """How many names ``params.no_trade_band`` kept out of the order list.

    Same arguments as :func:`allocate`, so a caller that is already allocating
    can log the counterfactual with the inputs it has in hand::

        from trader.allocator.deterministic import band_suppression
        rep = band_suppression(r_hat[sig], vol[sig], mask, sids, cur, params)
        logger.info("band suppressed %d of %d names", rep.n_suppressed,
                    rep.n_traded_unbanded)

    It runs :func:`allocate` twice — once with the band, once with it forced to
    zero — rather than re-deriving the band arithmetic, so the count is the
    real mechanism's count and cannot drift away from it.  Deliberately **not**
    wired into `scripts/run_allocator.py`; that is a one-line call at the
    rebalance branch, and this stream does not own that file.
    """
    banded = allocate(r_hat, vol, mask, sector_ids, current_w, params)
    unbanded = allocate(
        r_hat, vol, mask, sector_ids, current_w, replace(params, no_trade_band=0.0)
    )
    cur_eq = _current_equity(np.asarray(current_w, dtype=np.float64))
    d_on = banded[1:] - cur_eq
    d_off = unbanded[1:] - cur_eq
    return BandSuppression(
        n_traded_unbanded=int((np.abs(d_off) > _EPS).sum()),
        n_traded=int((np.abs(d_on) > _EPS).sum()),
        n_suppressed=int((np.abs(d_off) > _EPS).sum() - (np.abs(d_on) > _EPS).sum()),
        n_sold_unbanded=int((d_off < -_EPS).sum()),
        n_sold=int((d_on < -_EPS).sum()),
        gross_unbanded=float(np.abs(d_off).sum()),
        gross=float(np.abs(d_on).sum()),
    )
