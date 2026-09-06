"""Deterministic allocation layer (R5, `10_architecture_revamp.md` §5).

    from trader.allocator import AllocatorParams, RebalanceSchedule, allocate

Consumes the supervised signal's ``r_hat`` and produces target weights for
``PanelTradingEnv.step_weights``; the schedule tells the env which days it may
trade on.  No learned parameters anywhere in this package.

:class:`AllocatorParams` is the single source of truth for every allocator knob.
:mod:`trader.allocator.sizing` answers "what K can this account run?" *before* a
run starts and deliberately holds no knobs of its own — it imports its constants
from ``trader.env.costs`` and takes everything else as an argument, so there is
no second place for ``k``, ``min_trade_value`` or the demat fee to be defined
and drift.
"""
from __future__ import annotations

from trader.allocator.deterministic import (
    AllocatorParams,
    BandSuppression,
    allocate,
    band_suppression,
)
from trader.allocator.rebalance import RebalanceAnchor, RebalanceFreq, RebalanceSchedule
from trader.allocator.sizing import (
    KCapacity,
    capacity_table,
    fee_drag_estimate,
    max_supportable_k,
    supportable_k_detail,
)

__all__ = [
    "AllocatorParams",
    "BandSuppression",
    "KCapacity",
    "RebalanceAnchor",
    "RebalanceFreq",
    "RebalanceSchedule",
    "allocate",
    "band_suppression",
    "capacity_table",
    "fee_drag_estimate",
    "max_supportable_k",
    "supportable_k_detail",
]
