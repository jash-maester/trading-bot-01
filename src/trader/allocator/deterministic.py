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
7. turnover budget — if the gross equity move from ``current_w`` exceeds
   ``turnover_budget``, move along the straight line towards the target so the
   budget binds exactly
8. return ``[cash, w_1..w_N]`` summing to 1, all ≥ 0

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
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_EPS = 1e-12


@dataclass(frozen=True)
class AllocatorParams:
    """Every free parameter of the allocator.  Nothing here is learned."""

    k: int = 30                      # names to hold
    max_name_weight: float = 0.10    # matches env's max_weight_per_name
    max_sector_weight: float = 0.25
    turnover_budget: float = 0.30    # max gross traded fraction of NAV per rebalance
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
    # 7/8. turnover budget, then assemble [cash, w]
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


def _finish(w_eq: np.ndarray, current_w: np.ndarray, params: AllocatorParams) -> np.ndarray:
    """Apply the turnover budget and assemble ``[cash, w_1..w_N]``.

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
    """
    cur_eq = np.clip(current_w[1:], 0.0, None)
    # Renormalise the incoming book so a float32 `obs["portfolio"]` that sums
    # to 1 ± 1e-6 does not leak into the returned cash weight.
    cur_total = float(cur_eq.sum()) + max(float(current_w[0]), 0.0)
    if cur_total > _EPS:
        cur_eq = cur_eq / cur_total

    delta = w_eq - cur_eq
    gross = float(np.abs(delta).sum())
    if gross > params.turnover_budget + 1e-12 and gross > _EPS:
        w_eq = cur_eq + (params.turnover_budget / gross) * delta

    w_eq = np.maximum(w_eq, 0.0)
    eq_sum = float(w_eq.sum())
    if eq_sum > 1.0:
        w_eq = w_eq / eq_sum
        eq_sum = 1.0
    out = np.empty(w_eq.shape[0] + 1, dtype=np.float64)
    out[0] = 1.0 - eq_sum
    out[1:] = w_eq
    return out
