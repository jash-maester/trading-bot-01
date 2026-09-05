"""Deterministic allocation layer (R5, `10_architecture_revamp.md` §5).

    from trader.allocator import AllocatorParams, RebalanceSchedule, allocate

Consumes the supervised signal's ``r_hat`` and produces target weights for
``PanelTradingEnv.step_weights``; the schedule tells the env which days it may
trade on.  No learned parameters anywhere in this package.
"""
from __future__ import annotations

from trader.allocator.deterministic import AllocatorParams, allocate
from trader.allocator.rebalance import RebalanceAnchor, RebalanceFreq, RebalanceSchedule

__all__ = [
    "AllocatorParams",
    "RebalanceAnchor",
    "RebalanceFreq",
    "RebalanceSchedule",
    "allocate",
]
