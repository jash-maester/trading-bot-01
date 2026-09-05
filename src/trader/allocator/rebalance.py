"""Rebalance schedules — pure functions over trading dates.

Rebalance frequency is the single most expensive choice in the system
(`10_architecture_revamp.md` §4): delivery STT is 0.1% on both legs and STCG
is 20% under twelve months, so every needless rebalance day is paid for twice.
This module decides *which* trading days are rebalance days; `PanelTradingEnv`
enforces it by holding the book on every other day.

Nothing here knows about holidays.  A "trading day" is whatever date the
caller passes in — the panel's own date column, i.e. the observed NSE calendar
from `trader.data.calendar.build_calendar`.  The first trading day of a week or
month is therefore found by comparing *consecutive observed dates*, which is
exactly right when Monday or the 1st is a holiday: the next observed date is
the first trading day, whatever its weekday or day-of-month.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from typing import Literal

import numpy as np

RebalanceFreq = Literal["daily", "weekly", "monthly"]
RebalanceAnchor = Literal["first", "last"]

_FREQS: tuple[str, ...] = ("daily", "weekly", "monthly")
_ANCHORS: tuple[str, ...] = ("first", "last")


def _period_key(d: date, freq: str) -> tuple[int, int]:
    """Identify the (week | month) a date belongs to; daily = the date itself."""
    if freq == "monthly":
        return (d.year, d.month)
    if freq == "weekly":
        iso = d.isocalendar()
        return (iso.year, iso.week)
    return (d.toordinal(), 0)


@dataclass(frozen=True)
class RebalanceSchedule:
    """Which trading days the book is allowed to trade on.

    Parameters
    ----------
    freq:
        ``"daily"`` — every trading day (the env's historical behaviour).
        ``"weekly"`` — the first (or last) trading day of each ISO week.
        ``"monthly"`` — the first (or last) trading day of each calendar month.
    anchor:
        ``"first"`` (default) trades on the first observed trading day of the
        period, which needs only the *previous* trading date to detect.
        ``"last"`` trades on the last observed trading day, which needs the
        *next* trading date; :meth:`is_rebalance_day` therefore requires
        ``next_date`` for it.  The env precomputes a mask over its whole
        calendar via :meth:`mask`, so both anchors cost nothing per step.
    """

    freq: RebalanceFreq = "monthly"
    anchor: RebalanceAnchor = "first"

    def __post_init__(self) -> None:
        if self.freq not in _FREQS:
            raise ValueError(f"freq must be one of {_FREQS}, got {self.freq!r}")
        if self.anchor not in _ANCHORS:
            raise ValueError(f"anchor must be one of {_ANCHORS}, got {self.anchor!r}")

    def is_rebalance_day(
        self,
        d: date,
        prev_date: date | None,
        next_date: date | None = None,
    ) -> bool:
        """True if `d` is a rebalance day.

        ``prev_date`` / ``next_date`` are the adjacent *observed trading* dates
        (``None`` at the calendar's edges).  A ``"first"``-anchored schedule
        rebalances when `d` falls in a different period from ``prev_date``; a
        ``"last"``-anchored one when `d` falls in a different period from
        ``next_date``.  The calendar's edges count as period boundaries, so the
        first date of a calendar is always a rebalance day under ``"first"``
        and the last date always is under ``"last"``.
        """
        if self.freq == "daily":
            return True
        if self.anchor == "first":
            return prev_date is None or _period_key(d, self.freq) != _period_key(
                prev_date, self.freq
            )
        return next_date is None or _period_key(d, self.freq) != _period_key(
            next_date, self.freq
        )

    def mask(self, dates: Sequence[date]) -> np.ndarray:
        """Boolean ``[T]`` mask of rebalance days over an ordered trading calendar."""
        T = len(dates)
        out = np.ones(T, dtype=np.bool_)
        if self.freq == "daily" or T == 0:
            return out
        keys = np.array([_period_key(d, self.freq) for d in dates], dtype=np.int64)
        changed = np.any(keys[1:] != keys[:-1], axis=1)   # [T-1] boundary between i-1 and i
        if self.anchor == "first":
            out[1:] = changed
        else:
            out[:-1] = changed
        return out
