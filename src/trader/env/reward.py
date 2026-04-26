"""Reward functions for PanelTradingEnv."""
from __future__ import annotations

import math
from abc import ABC, abstractmethod


class RewardFn(ABC):
    @abstractmethod
    def reset(self) -> None:
        """Reset internal state at the start of an episode."""

    @abstractmethod
    def __call__(self, log_return: float) -> float:
        """Return scalar reward given the period log return."""


class DifferentialSharpe(RewardFn):
    """Moody & Saffell (2001) online differential Sharpe ratio.

    Maintains exponential moving averages of r and r^2, then computes
    the gradient of the Sharpe ratio at each step.

    Parameters
    ----------
    eta:
        EMA decay factor (≈ 1/window).
    """

    def __init__(self, eta: float = 1.0 / 60.0) -> None:
        self._eta = eta
        self._A = 0.0  # EMA of r
        self._B = 0.0  # EMA of r^2
        self._step = 0

    def reset(self) -> None:
        self._A = 0.0
        self._B = 0.0
        self._step = 0

    def __call__(self, log_return: float) -> float:
        eta = self._eta
        A_prev = self._A
        B_prev = self._B

        delta_A = log_return - A_prev
        delta_B = log_return ** 2 - B_prev

        denom = (B_prev - A_prev ** 2) ** 1.5
        if self._step < 2 or denom < 1e-12:
            dsr = 0.0
        else:
            dsr = (B_prev * delta_A - 0.5 * A_prev * delta_B) / denom

        self._A = A_prev + eta * delta_A
        self._B = B_prev + eta * delta_B
        self._step += 1

        # Guard against any numerical edge cases
        if math.isnan(dsr) or math.isinf(dsr):
            return 0.0
        return float(dsr)


class LogReturn(RewardFn):
    """Plain daily log return — ablation alternative to DSR."""

    def reset(self) -> None:
        pass

    def __call__(self, log_return: float) -> float:
        return log_return


class ExcessLogReturn(RewardFn):
    """Pass-through reward for *excess* log return.

    The env is responsible for subtracting the benchmark log return before
    calling this — see ``PanelTradingEnv`` and the ``use_excess_returns``
    flag.  This class exists so :data:`_REWARD_MAP` in ``train.py`` has a
    distinct entry that documents the choice; functionally it is identical
    to :class:`LogReturn`.

    Why excess return: with raw `LogReturn`, a long-only agent in a +16%
    market gets positive reward just by being invested.  With excess return
    (portfolio_log_return − benchmark_log_return) the agent is rewarded
    only when it *beats* the benchmark — which is the actual objective.
    """

    def reset(self) -> None:
        pass

    def __call__(self, log_return: float) -> float:
        return log_return
