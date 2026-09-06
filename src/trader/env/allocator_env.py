"""AllocatorEnv — one RL step is one rebalance period, one action is a few scalars.

This is the R6 environment of ``10_architecture_revamp.md`` §5.  It sits *on
top of* :class:`~trader.env.panel_env.PanelTradingEnv` and the deterministic
allocator, and it exists to make three of §1's structural causes impossible
rather than merely discouraged:

**§1.1 — the action cannot churn.**  The agent never touches a weight vector.
It emits ``3 + n_sectors`` bounded scalars — K, a turnover budget, a cash floor,
and one tilt per sector — which parameterise
:func:`trader.allocator.deterministic.allocate`.  Exploration noise moves K by a
few names, and the *sampled turnover budget is itself the bound on the trade*,
so a noisy action cannot produce a big trade by construction.  The old policy
sampled 505 logits and the sampling alone moved 28.9% of NAV per day.

The bound is not exact, and calling it "hard" was an overstatement.  The
allocator budgets ``|Δw|`` against a book marked at the previous close while
the env fills at the open, so realised turnover exceeds the sampled budget by
the overnight gap on the traded names.  Measured on a synthetic panel over 48
steps per level: **+0.0006** with no gap, **+0.0037** at 1% gap sd, **+0.0107**
at 3% (``tests/unit/test_allocator_env.py::
test_turnover_budget_slack_on_a_gapped_panel_is_measured_not_assumed``).  That
is second-order against what the budget is for — stopping a signal flip from
rotating the whole book, a turnover of 2.0 — but it is slack, not zero.

**§1.2 — the critic can see the state.**  The observation is the allocator's own
state: the parameters in force, the weight summary, per-sector exposure,
realised turnover, drawdown, regime, and time.  There is **deliberately no
per-stock tensor**.  The old critic conditioned ``V(s)`` on the mean of 504
per-stock embeddings, which is near-constant across states.

**§1.3 — the reward pays for selection, not for beta.**  The inner env must be
configured with ``use_excess_returns=True``, so each daily reward is the
portfolio log return *minus the equal-weight benchmark*.  Sitting in the market
earns nothing.

Step semantics
--------------
One :meth:`AllocatorEnv.step` advances the inner env from one rebalance day to
the next: it trades once, on the rebalance day, and then **holds** through every
day until the next one — the inner env's ``rebalance_schedule`` decides which
days those are, and on a hold day it charges no cost and reports zero turnover.
With ``freq="monthly"`` an episode of 24 steps is two years of trading.

Reward per period
-----------------
::

    r = Σ_days (excess log return)
        − turnover_penalty  · Σ_days (realised gross turnover)
        − drawdown_penalty  · max(0, drawdown_from_peak − drawdown_threshold)

The turnover term uses the env's **new** turnover semantics (gross traded value
/ pre-trade NAV, ``panel_env.py`` "Turnover"), which is proportional to the fees
the trade actually incurs.  The old metric measured NAV drift — a full rotation
reported 0.0 — and was only fixed in B4, so any ``turnover_penalty`` inherited
from a pre-B4 config means something different now and is **UNCALIBRATED**
(``10`` §8).

Alignment
---------
``r_hat`` and ``vol`` are aligned to the *inner env's* calendar and universe at
construction.  A date with no OOS prediction is a **hold**: the target is the
book already held, so a signal gap costs nothing rather than liquidating into
cash.  The panels on disk are stale relative to ``active_tickers()`` (504 vs
163), so nothing here assumes the two agree — the ticker order of the signal
must equal the ticker order of the env and that is checked, not assumed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, SupportsFloat

import numpy as np
import polars as pl
from gymnasium import Env, spaces

from trader.allocator.deterministic import AllocatorParams, allocate
from trader.data.regime_features import REGIME_DIM
from trader.data.universe import SECTOR_IDS
from trader.env.panel_env import PanelTradingEnv
from trader.env.reward import ExcessLogReturn, LogReturn

# Sector ids are 1-indexed; id 0 means "unsectored" and the allocator never
# selects it (`deterministic.py` step 1).  The tilt vector and the sector
# exposure block of the observation therefore cover ids 1..N_SECTORS.
N_SECTORS: int = len(SECTOR_IDS)


class InnerEnvMisconfigured(ValueError):
    """The wrapped :class:`PanelTradingEnv` would make the reward mean something else."""


# ── action encoding ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ActionRanges:
    """The bounded scalars the policy samples, and what each maps onto.

    The policy emits ``a ∈ [0, 1]^A`` (a Beta per dimension,
    :mod:`trader.models.allocator_policy`).  Every entry below is an affine map
    from ``[0, 1]`` onto a range that is *safe at both ends*: no value of ``a``
    can produce an allocation the book cannot fund or a trade the cost model
    would choke on.  That is the whole safety argument of this action space —
    it is a box, not a simplex.

    ==== ============================ ==================== ======================
    idx  meaning                      range                effect
    ==== ============================ ==================== ======================
    0    ``k``                        ``[10, 60]`` names   how many names to hold
    1    ``turnover_budget``          ``[0.05, 1.0]``      hard cap on gross |Δw|
    2    ``cash_floor``               ``[0.0, 0.5]``       min cash weight target
    3..  ``tilt[s]``, s = 1..S        ``[-1, +1]``         ``r_hat ×= 1 + tilt``
    ==== ============================ ==================== ======================

    Ranges, and why each end is where it is:

    ``k ∈ [10, 60]``
        Below ~10 names a 10% per-name cap cannot place the book (10 × 0.10 =
        1.0 exactly), so smaller K only adds cash; above 60 the portfolio is
        indistinguishable from equal-weight, which is the baseline.  §5's
        allocator spec says K ≈ 20–40; the range brackets it on both sides so
        the agent can be told it is wrong.
    ``turnover_budget ∈ [0.05, 1.0]``
        Gross two-sided convention (``deterministic.py`` "Units"): 1.0 is one
        full side of the book, 2.0 a complete rotation.  The upper bound is
        deliberately *below* a full rotation — a monthly full rotation at the
        verified Indian delivery round-trip is roughly 2.4%/yr of cost before
        tax.  The lower bound is not 0: a budget of exactly zero freezes the
        book and makes the sector tilts unobservable, which starves the
        gradient.
    ``cash_floor ∈ [0, 0.5]``
        Half the book is as far as a long-only equity strategy should be
        allowed to de-risk; beyond that it is a cash fund, and its excess
        return versus an equal-weight equity benchmark says nothing about
        selection.
    ``tilt[s] ∈ [-1, +1]``
        Multiplies ``r_hat`` for every name in sector ``s`` by ``1 + tilt``, so
        the factor spans ``[0, 2]``.  **Note the asymmetry** — this is a
        conviction *scale*, not an additive preference: a name with negative
        ``r_hat`` is pushed further down by a positive tilt, and a tilt of
        exactly ``-1`` zeroes a sector's signal (making its names rank at 0,
        mid-pack, rather than excluding them).  That is the pinned R6 spec
        (``10`` §5, "sector tilt"); an additive tilt would be the obvious
        alternative and is not what is specified.
    """

    k_min: int = 10
    k_max: int = 60
    turnover_min: float = 0.05
    turnover_max: float = 1.0
    cash_floor_min: float = 0.0
    cash_floor_max: float = 0.5
    tilt_min: float = -1.0
    tilt_max: float = 1.0
    n_sectors: int = N_SECTORS

    def __post_init__(self) -> None:
        if not 1 <= self.k_min <= self.k_max:
            raise ValueError(f"need 1 <= k_min <= k_max, got {self.k_min}, {self.k_max}")
        if not 0.0 <= self.turnover_min <= self.turnover_max:
            raise ValueError("need 0 <= turnover_min <= turnover_max")
        if not 0.0 <= self.cash_floor_min <= self.cash_floor_max < 1.0:
            raise ValueError("need 0 <= cash_floor_min <= cash_floor_max < 1")
        if self.tilt_min > self.tilt_max:
            raise ValueError("need tilt_min <= tilt_max")
        if self.n_sectors < 1:
            raise ValueError(f"n_sectors must be >= 1, got {self.n_sectors}")

    @property
    def action_dim(self) -> int:
        """``3 + n_sectors``  — K, turnover budget, cash floor, one tilt each."""
        return 3 + self.n_sectors

    def decode(
        self,
        action: np.ndarray,
        *,
        max_name_weight: float,
        max_sector_weight: float,
        vol_lookback: int,
        no_trade_band: float = 0.0,
    ) -> tuple[AllocatorParams, np.ndarray]:
        """``a ∈ [0, 1]^A`` → ``(AllocatorParams, tilt[n_sectors])``.

        ``action`` is clipped into ``[0, 1]`` first: the Beta policy cannot
        leave the box, but a hand-written or replayed action might, and an
        out-of-range K would raise from ``AllocatorParams.__post_init__``
        mid-rollout.

        ``no_trade_band`` is a **fixed** parameter of the env here, not a fourth
        action dimension: it arrives from ``AllocatorEnvConfig`` and is passed
        straight through.  Making it learnable would change ``action_dim`` from
        ``3 + n_sectors`` to ``4 + n_sectors`` and invalidate every checkpoint
        and the ``ranges:`` block of `configs/env/allocator.yaml`, so it is a
        decision of its own rather than a side effect of wiring the band in.
        """
        a = np.clip(np.asarray(action, dtype=np.float64).reshape(-1), 0.0, 1.0)
        if a.shape[0] != self.action_dim:
            raise ValueError(f"action must have {self.action_dim} entries, got {a.shape[0]}")
        k = int(round(self.k_min + a[0] * (self.k_max - self.k_min)))
        budget = self.turnover_min + a[1] * (self.turnover_max - self.turnover_min)
        cash_floor = self.cash_floor_min + a[2] * (self.cash_floor_max - self.cash_floor_min)
        tilt = self.tilt_min + a[3:] * (self.tilt_max - self.tilt_min)
        params = AllocatorParams(
            k=k,
            max_name_weight=max_name_weight,
            max_sector_weight=max_sector_weight,
            turnover_budget=float(budget),
            no_trade_band=no_trade_band,
            cash_floor=float(cash_floor),
            vol_lookback=vol_lookback,
        )
        return params, tilt

    def encode(self, params: AllocatorParams, tilt: np.ndarray) -> np.ndarray:
        """Inverse of :meth:`decode`, up to K's integer rounding.

        The observation carries the parameters *in force* on the unit scale the
        policy works in, so the actor never has to learn the affine maps above.
        """

        def _inv(x: float, lo: float, hi: float) -> float:
            return 0.0 if hi <= lo else float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))

        out = np.empty(self.action_dim, dtype=np.float32)
        out[0] = _inv(float(params.k), float(self.k_min), float(self.k_max))
        out[1] = _inv(params.turnover_budget, self.turnover_min, self.turnover_max)
        out[2] = _inv(params.cash_floor, self.cash_floor_min, self.cash_floor_max)
        out[3:] = [
            _inv(float(t), self.tilt_min, self.tilt_max) for t in np.asarray(tilt).reshape(-1)
        ]
        return out


# ── signal panel ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SignalPanel:
    """OOS predictions and realised vol, as dense ``[T, N]`` arrays.

    ``tickers`` must be in ``active_tickers()`` order — the pinned signal
    artefact contract — and :class:`AllocatorEnv` checks it against the inner
    env's universe rather than trusting it.
    """

    dates: list[date]
    tickers: list[str]
    r_hat: np.ndarray      # [T, N] float64, NaN = no prediction for that cell
    vol: np.ndarray        # [T, N] float64, <= 0 or non-finite = never selected

    # CONTRACT on `vol`: ``vol[t]`` is the volatility known **before date t's
    # open**, i.e. computed from data up to and including t-1.  It is NOT
    # ``realized_vol_20d`` as of date t.
    #
    # This is load-bearing, not a nicety.  `realized_vol_20d[t]` includes
    # `log_return_1d[t] = log(close_t / close_{t-1})` (features.py:241-244,
    # 261-265) — date t's *close*.  The allocator's trade is filled at date t's
    # OPEN, so sizing it with same-day vol reads a price that has not printed
    # yet.  Measured: perturbing only `realized_vol_20d` on the trade date moved
    # the executed target weights by L1 0.268551.  `PanelTradingEnv` already
    # excludes day t from its own feature window (panel_env.py:467) for exactly
    # this reason.
    #
    # `from_artifacts` performs the shift.  Anything constructing a SignalPanel
    # directly (tests, notebooks) must pass an already-lagged vol.

    def __post_init__(self) -> None:
        T, N = len(self.dates), len(self.tickers)
        for name, arr in (("r_hat", self.r_hat), ("vol", self.vol)):
            if arr.shape != (T, N):
                raise ValueError(f"{name} must be [T, N] = ({T}, {N}), got {arr.shape}")

    @classmethod
    def from_artifacts(
        cls,
        signal_dir: Path,
        panel_path: Path,
        tickers: list[str],
        *,
        horizon: str = "r_hat_20d",
        vol_lookback: int = 20   # load-bearing: names the panel column, see vol_column_for,
    ) -> SignalPanel:
        """Read ``<signal_dir>/predictions.parquet`` and a feature panel.

        The predictions file is the pinned R4 artefact: columns ``date``,
        ``ticker``, ``r_hat_5d``, ``r_hat_20d``, one row per (date, tradeable
        ticker), OOS only.  The vol column comes from the feature panel and is
        **derived from** ``vol_lookback`` via :func:`vol_column_for` rather than
        named separately — the project's panels carry ``realized_vol_20d``
        annualised, which is what the allocator's inverse-vol sizing expects.

        The vol column is **lagged one trading day** on the way in, so the
        returned ``vol[t]`` is knowable before date ``t``'s open.  See the
        CONTRACT note on :attr:`SignalPanel.vol`.
        """
        preds = pl.read_parquet(Path(signal_dir) / "predictions.parquet")
        if horizon not in preds.columns:
            raise KeyError(f"{horizon!r} not in predictions.parquet ({preds.columns})")
        vol_column = vol_column_for(vol_lookback)
        panel = pl.read_parquet(panel_path, columns=["date", "ticker", vol_column])
        r_dates, r_arr = _pivot(preds, tickers, horizon)
        v_dates, v_arr = _pivot(panel, tickers, vol_column)
        # The signal is OOS-only, so its calendar is a subset of the panel's.
        # Align vol onto the signal's dates; a missing vol row is NaN, which the
        # allocator treats as "never selected".
        v_index = {d: i for i, d in enumerate(v_dates)}
        vol = np.full_like(r_arr, np.nan)
        for i, d in enumerate(r_dates):
            j = v_index.get(d)
            # LAG BY ONE PANEL ROW.  `j - 1` walks the *panel's* calendar, not
            # the signal's: the signal is OOS-only so its dates can have gaps,
            # and the value we want is the previous trading day's, not the
            # previous prediction's.  `j == 0` has no predecessor, so that date
            # gets NaN and its names are simply not candidates that day — the
            # honest reading of "no vol history yet".  See the CONTRACT note on
            # the `vol` field for why this shift exists.
            if j is not None and j >= 1:
                vol[i] = v_arr[j - 1]
        return cls(dates=r_dates, tickers=list(tickers), r_hat=r_arr, vol=vol)


def _pivot(frame: pl.DataFrame, tickers: list[str], column: str) -> tuple[list[date], np.ndarray]:
    """Long ``(date, ticker, column)`` → ``(dates, [T, N])`` with NaN for gaps."""
    dates = sorted({_as_date(d) for d in frame["date"].to_list()})
    d_index = {d: i for i, d in enumerate(dates)}
    t_index = {t: i for i, t in enumerate(tickers)}
    out = np.full((len(dates), len(tickers)), np.nan, dtype=np.float64)
    rows = frame.select(["date", "ticker", column]).rows()
    for d, t, v in rows:
        j = t_index.get(t)
        if j is None or v is None:
            continue
        out[d_index[_as_date(d)], j] = float(v)
    return dates, out


def vol_column_for(vol_lookback: int) -> str:
    """The panel column implied by ``vol_lookback`` — the ONLY way to name it.

    ``vol_lookback`` and ``vol_column`` used to be two independent knobs
    (`configs/env/allocator.yaml`, `AllocatorParams.vol_lookback`, and this
    env's config all carried one or both), with ``vol_lookback`` never read
    anywhere.  A declared-and-never-read knob that duplicates a live one is the
    `min_trade_value: 500` trap named in CLAUDE.md: the two silently disagree
    and the dead one is the one people read.  Deriving the column from the
    lookback makes them incapable of disagreeing.
    """
    if vol_lookback < 1:
        raise ValueError(f"vol_lookback must be >= 1, got {vol_lookback}")
    return f"realized_vol_{int(vol_lookback)}d"


def _as_date(d: Any) -> date:
    return d if isinstance(d, date) else date.fromisoformat(str(d)[:10])


# ── env config ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class AllocatorEnvConfig:
    """Everything about the period-level MDP that is not the inner env.

    ``periods_per_episode`` is the episode length in *rebalance periods*, not
    days: ``10`` §5 specifies 12–36 so credit assignment is over tens of
    decisions rather than the 252 the old daily env used.

    ``turnover_penalty`` and ``drawdown_penalty`` are the two reward terms this
    env adds on top of the inner env's excess log return.  Both are
    **UNCALIBRATED**: no run has ever set them against the post-B4 turnover
    metric, and ``10`` §8 says explicitly that the right value needs a run, not
    a guess.  The defaults are placeholders chosen for scale, not for
    performance: ``turnover_penalty=0.02`` charges 2 bp of reward per 1% of
    gross turnover, which is the same order as the verified delivery round-trip
    cost the inner env already deducts from NAV, so it acts as a mild
    additional brake rather than a second full charge.
    """

    periods_per_episode: int = 24
    max_days_per_period: int = 45
    max_name_weight: float = 0.10
    max_sector_weight: float = 0.25
    # Per-name no-trade band (P3), in NAV fraction: a name is traded only when
    # |target - current| exceeds it.  0.0 switches the mechanism off and the
    # allocator takes its pre-band code path unchanged.  Read from
    # `configs/env/allocator.yaml` by `scripts/train_allocator_rl.py` and
    # threaded into `AllocatorParams` by `ActionRanges.decode` -- all three
    # sites exist, so this is not the `min_trade_value: 500` dead key.
    no_trade_band: float = 0.0
    vol_lookback: int = 20   # load-bearing: names the panel column, see vol_column_for
    turnover_penalty: float = 0.02
    drawdown_penalty: float = 1.0
    drawdown_threshold: float = 0.10
    ranges: ActionRanges = field(default_factory=ActionRanges)

    def __post_init__(self) -> None:
        if not 1 <= self.periods_per_episode:
            raise ValueError(f"periods_per_episode must be >= 1, got {self.periods_per_episode}")
        if self.max_days_per_period < 1:
            raise ValueError(f"max_days_per_period must be >= 1, got {self.max_days_per_period}")
        if self.turnover_penalty < 0.0 or self.drawdown_penalty < 0.0:
            raise ValueError("reward penalties must be >= 0")
        if not 0.0 <= self.drawdown_threshold < 1.0:
            raise ValueError(f"drawdown_threshold must be in [0, 1), got {self.drawdown_threshold}")


# ── the env ───────────────────────────────────────────────────────────────────


class AllocatorEnv(Env):  # type: ignore[type-arg]
    """Gymnasium env whose step is one rebalance period of the allocator.

    Observation — a flat ``float32`` vector, allocator state only::

        [0 : A]              action in force, on the policy's own [0, 1] scale
        [A]                  cash weight
        [A + 1]              largest single-name weight
        [A + 2 : A + 2 + S]  weight in sector 1..S
        [A + 2 + S]          realised gross turnover over the last period
        [A + 3 + S]          drawdown from the episode's peak NAV, in [0, 1]
        [A + 4 + S : ... ]   regime vector, REGIME_DIM entries
        [last]               t_frac — periods elapsed / periods_per_episode

    with ``A = ranges.action_dim`` and ``S = ranges.n_sectors``.  Every entry is
    a property of the portfolio or the market, never of an individual stock:
    the critic sees the entire state it is asked to value.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        inner_env: PanelTradingEnv,
        signal: SignalPanel,
        config: AllocatorEnvConfig | None = None,
        *,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.cfg = config or AllocatorEnvConfig()
        self.inner = inner_env
        self.ranges = self.cfg.ranges
        _require_excess_reward_config(inner_env)

        if list(signal.tickers) != list(inner_env.universe):
            raise ValueError(
                "signal ticker order must equal the env's universe order "
                f"({len(signal.tickers)} vs {len(inner_env.universe)} names; "
                "the pinned contract is active_tickers() order on both sides)"
            )
        self.signal = signal
        self._n = len(inner_env.universe)

        # Align the signal onto the env's own calendar once, up front.  Dates
        # the signal does not cover stay NaN and are treated as hold days.
        env_dates = [_as_date(d) for d in inner_env.dates]
        sig_index = {d: i for i, d in enumerate(signal.dates)}
        T = len(env_dates)
        self._r_hat = np.full((T, self._n), np.nan, dtype=np.float64)
        self._vol = np.full((T, self._n), np.nan, dtype=np.float64)
        n_covered = 0
        for i, d in enumerate(env_dates):
            j = sig_index.get(d)
            if j is not None:
                self._r_hat[i] = signal.r_hat[j]
                self._vol[i] = signal.vol[j]
                n_covered += 1
        self.n_signal_dates: int = n_covered

        A, S = self.ranges.action_dim, self.ranges.n_sectors
        self.obs_dim: int = A + 5 + S + REGIME_DIM
        self.observation_space = spaces.Box(-np.inf, np.inf, (self.obs_dim,), np.float32)
        self.action_space = spaces.Box(0.0, 1.0, (A,), np.float32)
        if seed is not None:
            self.action_space.seed(seed)

        # Episode state
        self._inner_obs: dict[str, np.ndarray] = {}
        self._period = 0
        self._peak_nav = 0.0
        self._last_turnover = 0.0
        self._last_action = np.full(A, 0.5, dtype=np.float32)
        self._last_params: AllocatorParams | None = None

    # ── Gymnasium API ────────────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        inner_obs, _ = self.inner.reset(seed=seed, options=options)
        self._inner_obs = inner_obs
        self._period = 0
        self._peak_nav = float(inner_obs["nav"])
        self._last_turnover = 0.0
        self._last_action = np.full(self.ranges.action_dim, 0.5, dtype=np.float32)
        self._last_params = None
        return self._build_obs(), {"date": self.inner.dates[self.inner.day_index]}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, SupportsFloat, bool, bool, dict[str, Any]]:
        """Trade once on the rebalance day, then hold to the next one."""
        params, tilt = self.ranges.decode(
            action,
            max_name_weight=self.cfg.max_name_weight,
            max_sector_weight=self.cfg.max_sector_weight,
            vol_lookback=self.cfg.vol_lookback,
            no_trade_band=self.cfg.no_trade_band,
        )
        day = self.inner.day_index
        current_w = np.asarray(self._inner_obs["portfolio"], dtype=np.float64)
        target_w, has_signal = self._target_weights(day, tilt, params, current_w)

        excess_log_return = 0.0
        gross_turnover = 0.0
        terminated = truncated = False
        info: dict[str, Any] = {}
        n_days = 0

        for _ in range(self.cfg.max_days_per_period):
            if n_days > 0 and self.inner.is_rebalance_step():
                break
            step_target = target_w if n_days == 0 else current_w
            inner_obs, reward, terminated, truncated, info = self.inner.step_weights(step_target)
            # `reward` is the daily EXCESS log return: the inner env is required
            # (above) to have use_excess_returns=True and turnover_penalty=0, so
            # the only per-period turnover charge is the one applied below.
            excess_log_return += float(reward)
            gross_turnover += float(info["turnover"])
            self._inner_obs = inner_obs
            n_days += 1
            if terminated or truncated:
                break

        nav = float(self._inner_obs["nav"])
        self._peak_nav = max(self._peak_nav, nav)
        drawdown = 0.0 if self._peak_nav <= 0.0 else 1.0 - nav / self._peak_nav
        dd_excess = max(0.0, drawdown - self.cfg.drawdown_threshold)

        period_reward = (
            excess_log_return
            - self.cfg.turnover_penalty * gross_turnover
            - self.cfg.drawdown_penalty * dd_excess
        )

        self._period += 1
        self._last_turnover = gross_turnover
        self._last_action = np.asarray(
            np.clip(np.asarray(action, dtype=np.float32).reshape(-1), 0.0, 1.0), dtype=np.float32
        )
        self._last_params = params

        episode_over = self._period >= self.cfg.periods_per_episode
        step_info: dict[str, Any] = {
            "k": params.k,
            "turnover_budget": params.turnover_budget,
            "cash_floor": params.cash_floor,
            "sector_tilt": tilt.astype(np.float32),
            "excess_log_return": excess_log_return,
            "turnover": gross_turnover,
            "drawdown": drawdown,
            "nav": nav,
            "n_days": n_days,
            "had_signal": has_signal,
            "date": self.inner.dates[min(self.inner.day_index, len(self.inner.dates) - 1)],
            "costs_paid": float(info.get("costs_paid", 0.0)),
        }
        return (
            self._build_obs(),
            period_reward,
            bool(terminated),
            bool(truncated or episode_over),
            step_info,
        )

    # ── internals ────────────────────────────────────────────────────────────

    def _target_weights(
        self,
        day: int,
        tilt: np.ndarray,
        params: AllocatorParams,
        current_w: np.ndarray,
    ) -> tuple[np.ndarray, bool]:
        """Target ``[N+1]`` from the signal at ``day``; hold when there is none."""
        r_hat = self._r_hat[day]
        if not np.isfinite(r_hat).any():
            # No OOS prediction for this date.  Holding is the honest default:
            # liquidating to cash would be a signal-driven trade taken on the
            # absence of a signal, and it would be charged for.
            return _renormalised(current_w), False
        sector_ids = np.asarray(self._inner_obs["sector_ids"], dtype=np.int64)
        # Standardise the cross-section BEFORE tilting.  The tilt is applied
        # multiplicatively, `r_hat * (1 + tilt)`, and the documented meaning of
        # `tilt = -1` is "this sector ranks mid-pack".  That is only true if
        # zero IS mid-pack, i.e. if r_hat is cross-sectionally centred.  R4's
        # predictions are standardised forward returns, but nothing in the
        # pinned contract requires it, and on an uncentred cross-section (say
        # every r_hat positive in a bull-market window) `tilt = -1` maps the
        # sector to 0, which is the WORST rank, silently turning "neutral" into
        # "exclude the sector entirely".  Centring here makes the action's
        # documented semantics true by construction rather than by assumption.
        # allocate() ranks by r_hat, so a shared affine rescale is otherwise a
        # no-op: this changes only how the tilts compose, which is the point.
        tilted = _standardise_cross_section(r_hat) * (1.0 + _tilt_per_name(tilt, sector_ids))
        target = allocate(
            tilted,
            self._vol[day],
            np.asarray(self._inner_obs["mask"], dtype=bool),
            sector_ids,
            _renormalised(current_w),
            params,
        )
        return _renormalised(target), True

    def _build_obs(self) -> np.ndarray:
        cfg, ranges = self.cfg, self.ranges
        A, S = ranges.action_dim, ranges.n_sectors
        portfolio = np.asarray(self._inner_obs["portfolio"], dtype=np.float64)
        eq = portfolio[1:]
        sector_ids = np.asarray(self._inner_obs["sector_ids"], dtype=np.int64)
        sec_w = np.bincount(np.clip(sector_ids, 0, S), weights=eq, minlength=S + 1)[1:]

        nav = float(self._inner_obs["nav"])
        drawdown = 0.0 if self._peak_nav <= 0.0 else 1.0 - nav / max(self._peak_nav, 1e-8)

        obs = np.zeros(self.obs_dim, dtype=np.float32)
        obs[:A] = self._last_action
        obs[A] = portfolio[0]
        obs[A + 1] = float(eq.max()) if eq.size else 0.0
        obs[A + 2 : A + 2 + S] = sec_w
        obs[A + 2 + S] = self._last_turnover
        obs[A + 3 + S] = drawdown
        obs[A + 4 + S : A + 4 + S + REGIME_DIM] = np.asarray(
            self._inner_obs["regime"], dtype=np.float32
        )
        obs[-1] = self._period / cfg.periods_per_episode
        return np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)


def _standardise_cross_section(r_hat: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-sd across the finite names of one day; NaNs preserved.

    A degenerate cross-section (all names equal, or fewer than two finite) is
    returned centred but unscaled — there is no spread to normalise, and
    dividing by ~0 would manufacture one.
    """
    out = np.asarray(r_hat, dtype=np.float64).copy()
    finite = np.isfinite(out)
    if not finite.any():
        return out
    vals = out[finite]
    mu = float(vals.mean())
    sd = float(vals.std())
    out[finite] = (vals - mu) / sd if sd > 1e-12 else vals - mu
    return out


def _tilt_per_name(tilt: np.ndarray, sector_ids: np.ndarray) -> np.ndarray:
    """Broadcast a per-sector tilt onto names; unsectored names get 0.0."""
    t = np.asarray(tilt, dtype=np.float64).reshape(-1)
    padded = np.concatenate([[0.0], t])                     # index 0 = unsectored
    idx = np.clip(sector_ids, 0, padded.shape[0] - 1)
    return np.asarray(padded[idx], dtype=np.float64)


def _renormalised(w: np.ndarray) -> np.ndarray:
    """Clip to non-negative and scale to sum exactly 1.

    ``obs["portfolio"]`` is float32 and sums to 1 ± 1e-6; ``step_weights``
    rejects anything off by more than 1e-6, so the round trip through float32
    has to be repaired before the vector goes back in.
    """
    x = np.clip(np.asarray(w, dtype=np.float64), 0.0, None)
    total = float(x.sum())
    if total <= 0.0:
        out = np.zeros_like(x)
        out[0] = 1.0
        return out
    return x / total


def _require_excess_reward_config(env: PanelTradingEnv) -> None:
    """Refuse an inner env whose per-day reward is not the excess log return.

    :meth:`AllocatorEnv.step` sums the inner env's daily rewards and calls the
    result "the period's excess log return".  That identity holds only when the
    inner env subtracts the equal-weight benchmark (``use_excess_returns=True``,
    ``10`` §1.3) and applies no turnover penalty of its own — otherwise the
    period reward silently becomes raw return, or turnover is charged twice at
    two different scales.  Both failures are invisible in a training curve,
    which is why this raises rather than warns.

    Every check below reads a **public property** of ``PanelTradingEnv``
    (`use_excess_returns`, `turnover_penalty`, `reward_fn`, `rebalance_schedule`)
    rather than a private attribute through ``getattr(..., default)``.  That
    matters: a defaulted ``getattr`` fails *open*, so renaming the attribute
    upstream would silently switch the guard off instead of breaking loudly.
    """
    if not env.use_excess_returns:
        raise InnerEnvMisconfigured(
            "AllocatorEnv requires the inner PanelTradingEnv to be built with "
            "use_excess_returns=True: the period reward is defined as excess log "
            "return vs equal-weight (10_architecture_revamp.md §1.3). With raw "
            "log return the agent is paid for being long in a 16.4%/yr market."
        )
    penalty = float(env.turnover_penalty)
    if penalty != 0.0:
        raise InnerEnvMisconfigured(
            f"inner PanelTradingEnv has turnover_penalty={penalty}; AllocatorEnv "
            "applies the turnover charge once, per period, via "
            "AllocatorEnvConfig.turnover_penalty. Set the inner env's to 0.0."
        )
    if not isinstance(env.reward_fn, (LogReturn, ExcessLogReturn)):
        # A DifferentialSharpe inner env would make the period reward a sum of
        # DSR increments, which is not an excess log return and does not
        # annualise the way AllocatorEnv's reporting assumes.
        raise InnerEnvMisconfigured(
            f"inner PanelTradingEnv uses reward_fn={type(env.reward_fn).__name__}; "
            "AllocatorEnv sums daily rewards into a period excess log return, "
            "which requires LogReturn or ExcessLogReturn."
        )
    if env.rebalance_schedule is None:
        # Without a schedule every inner day is a rebalance day, so one
        # AllocatorEnv step advances exactly ONE day rather than one rebalance
        # period.  `periods_per_episode` (read as months), the x12
        # annualisation and the whole monthly-cadence cost argument are all
        # silently wrong in that case, with no error and no log line.
        raise InnerEnvMisconfigured(
            "inner PanelTradingEnv was built with rebalance_schedule=None, so a "
            "'rebalance period' would be a single day: one AllocatorEnv step "
            "must advance exactly one rebalance period. Pass an explicit "
            "RebalanceSchedule(freq=...) to the inner env."
        )
