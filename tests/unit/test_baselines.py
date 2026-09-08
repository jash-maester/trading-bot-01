"""Tests for the baseline agents' shared episode runner."""
from __future__ import annotations

import pytest


def test_rebalance_count_is_trading_days_not_every_step() -> None:
    """`sold/reb` must divide by rebalances, not sessions.

    Live defect until 2026-09-08: at monthly cadence over 1,919 sessions the
    baseline divided by 1,919 rather than 91, so equal-weight's scrips-sold
    per rebalance read 2.3 where the truth is 49.5 — beside an allocator column
    that always divided by real rebalances. Twenty-one times apart, under one
    header, in the table `11_cost_defect_and_fix_plan.md` reasons about.
    """
    from trader.env.baselines import run_baseline_episode

    class _Env:
        """A 10-step env that only trades on every 5th step."""

        def __init__(self) -> None:
            self.t = 0

        def is_rebalance_step(self) -> bool:
            return self.t % 5 == 0

        def reset(self, seed: int | None = None):  # noqa: ANN201
            self.t = 0
            return {"nav": 100.0}, {}

        def step(self, action):  # noqa: ANN001, ANN201
            self.t += 1
            info = {"nav": 100.0 + self.t, "turnover": 0.0,
                    "n_scrips_sold": 2, "n_legs": 1}
            return {"nav": info["nav"]}, 0.0, self.t >= 10, False, info

    class _Agent:
        def reset(self, info=None) -> None:  # noqa: ANN001
            pass

        def act(self, obs):  # noqa: ANN001, ANN201
            return None

    _, _, diag = run_baseline_episode(_Env(), _Agent(), seed=0)  # type: ignore[arg-type]
    assert diag["steps"] == 10.0
    assert diag["rebalances"] == 2.0, "steps 0 and 5 are the only rebalance steps"
    # 10 steps x 2 scrips = 20 sold, over 2 rebalances.
    assert diag["scrip_sell_days"] == 20.0
    assert diag["scrip_sell_days_per_rebalance"] == pytest.approx(10.0)
