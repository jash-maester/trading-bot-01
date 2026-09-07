"""Portfolio risk overlays: volatility target, drawdown brake, per-name stop.

`audit/P3_P4_P5_RESULTS.md` measures the allocator at a -53.0% maximum drawdown
against equal-weight's -51.5%, and -68.4% at ₹1 lakh. Nothing in the allocator
targets volatility or limits drawdown; it was never asked to. These are the
three candidate instruments, kept deliberately separate so a run can enable one
at a time and the measurement says which did the work.

**Read the drawdown decomposition before choosing.** The allocator's drawdown
is 1.5 points worse than holding the whole market, so almost all of it is
market-wide rather than name-specific. That is an argument that the two
portfolio-level controls should dominate the per-name stop, and it is exactly
what run 1 of the risk sweep is there to test rather than assume.

All three are **overlays**: they take the allocator's target and scale or veto
parts of it. None of them changes `allocate()`, so with every control off the
result is bit-identical to the unoverlaid path.

Sign conventions, because these are easy to get backwards:

* ``exposure_scale`` is in ``[0, 1]`` and multiplies the **equity** block of a
  target weight vector; the remainder becomes cash. It never exceeds 1, so no
  control here can introduce leverage.
* a stop fires on a **loss**, so the comparison is ``return <= -stop_loss``
  with ``stop_loss`` given as a positive fraction.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_EPS = 1e-12
#: Trading days per year, for annualising a realised volatility.
_ANNUALISE = 252.0


@dataclass(frozen=True)
class RiskParams:
    """Every knob of the three overlays. ``None`` disables a control entirely."""

    #: Annualised portfolio volatility to target, e.g. 0.15. None = no vol target.
    vol_target: float | None = None
    #: Days of NAV history the realised vol is measured over.
    vol_lookback: int = 60
    #: Floor on the exposure scale a vol target may impose, so a vol spike
    #: cannot take the book to zero and strand it out of the recovery.
    vol_floor: float = 0.30
    #: Drawdown from peak beyond which exposure is reduced, e.g. 0.15. None = off.
    drawdown_threshold: float | None = None
    #: Exposure retained when the drawdown brake is fully applied.
    drawdown_floor: float = 0.30
    #: Drawdown at which the brake reaches ``drawdown_floor`` (linear between).
    drawdown_full: float = 0.35
    #: Per-name loss from entry that forces a sale, e.g. 0.15. None = off.
    stop_loss: float | None = None
    #: Steps a stopped name is barred from being re-bought.
    stop_cooldown_steps: int = 21

    def __post_init__(self) -> None:
        for name in ("vol_target", "drawdown_threshold", "stop_loss"):
            v = getattr(self, name)
            if v is not None and not 0.0 < v < 1.0 and name != "vol_target":
                raise ValueError(f"{name} must be in (0, 1) when set, got {v}")
        if self.vol_target is not None and self.vol_target <= 0.0:
            raise ValueError(f"vol_target must be > 0 when set, got {self.vol_target}")
        if self.vol_lookback < 2:
            raise ValueError(f"vol_lookback must be >= 2, got {self.vol_lookback}")
        for name in ("vol_floor", "drawdown_floor"):
            v = getattr(self, name)
            if not 0.0 <= v <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {v}")
        if self.drawdown_threshold is not None and self.drawdown_full <= self.drawdown_threshold:
            raise ValueError(
                f"drawdown_full ({self.drawdown_full}) must exceed drawdown_threshold "
                f"({self.drawdown_threshold})"
            )
        if self.stop_cooldown_steps < 0:
            raise ValueError(f"stop_cooldown_steps must be >= 0, got {self.stop_cooldown_steps}")

    @property
    def any_enabled(self) -> bool:
        return (
            self.vol_target is not None
            or self.drawdown_threshold is not None
            or self.stop_loss is not None
        )

    @property
    def needs_intraperiod_trading(self) -> bool:
        """Only the stop can demand a trade on a scheduled hold day."""
        return self.stop_loss is not None


@dataclass
class RiskState:
    """What the overlay reports about the step it just saw. Diagnostics only."""

    exposure_scale: float = 1.0
    realised_vol: float = float("nan")
    drawdown: float = 0.0
    n_stopped: int = 0
    n_quarantined: int = 0


class RiskOverlay:
    """Stateful portfolio overlay. One instance per backtest run.

    Call :meth:`update` once per env step with that step's NAV, closes and
    post-trade weights, then :meth:`apply` on a target before handing it to the
    env. :meth:`stops_to_execute` reports names that must be sold *now*, which
    on a scheduled hold day requires ``step_weights(..., force=True)``.
    """

    def __init__(self, params: RiskParams, n_names: int) -> None:
        self.p = params
        self.n = int(n_names)
        self._navs: list[float] = []
        self._peak = 0.0
        # Entry price per held name; NaN when not held.
        self._entry = np.full(self.n, np.nan, dtype=np.float64)
        self._held = np.zeros(self.n, dtype=bool)
        # Steps remaining before a stopped name may be re-bought.
        self._cooldown = np.zeros(self.n, dtype=np.int64)
        self._stopped = np.zeros(self.n, dtype=bool)
        self.state = RiskState()

    # ── state ────────────────────────────────────────────────────────────────

    def update(self, nav: float, closes: np.ndarray, weights: np.ndarray) -> None:
        """Absorb one env step. ``weights`` is the post-trade ``[N]`` equity block."""
        nav = float(nav)
        self._navs.append(nav)
        self._peak = max(self._peak, nav)

        held = np.asarray(weights, dtype=np.float64) > _EPS
        px = np.asarray(closes, dtype=np.float64)
        # A name newly held records its entry; a name released forgets it, so a
        # re-entry later is measured from the new price and not the old one.
        entered = held & ~self._held
        exited = ~held & self._held
        self._entry[entered] = px[entered]
        self._entry[exited] = np.nan
        self._held = held

        self._cooldown = np.maximum(self._cooldown - 1, 0)
        self._stopped = self._stopped_mask(px)

        self.state = RiskState(
            exposure_scale=self.exposure_scale(),
            realised_vol=self.realised_vol(),
            drawdown=self.drawdown(),
            n_stopped=int(self._stopped.sum()),
            n_quarantined=int((self._cooldown > 0).sum()),
        )

    def _stopped_mask(self, closes: np.ndarray) -> np.ndarray:
        if self.p.stop_loss is None:
            return np.zeros(self.n, dtype=bool)
        with np.errstate(invalid="ignore", divide="ignore"):
            ret = np.where(
                np.isfinite(self._entry) & (self._entry > 0.0),
                closes / np.maximum(self._entry, _EPS) - 1.0,
                0.0,
            )
        return bool_(self._held & (ret <= -float(self.p.stop_loss)))

    # ── measurements ─────────────────────────────────────────────────────────

    def realised_vol(self) -> float:
        """Annualised stdev of daily NAV log returns over ``vol_lookback``."""
        if len(self._navs) < 3:
            return float("nan")
        nav = np.asarray(self._navs[-(self.p.vol_lookback + 1) :], dtype=np.float64)
        if nav.size < 3 or np.any(nav <= 0.0):
            return float("nan")
        r = np.diff(np.log(nav))
        if r.size < 2:
            return float("nan")
        return float(np.std(r, ddof=1) * np.sqrt(_ANNUALISE))

    def drawdown(self) -> float:
        """Fractional drawdown from the running NAV peak, >= 0."""
        if not self._navs or self._peak <= 0.0:
            return 0.0
        return float(max(0.0, 1.0 - self._navs[-1] / self._peak))

    def exposure_scale(self) -> float:
        """Product of the vol-target and drawdown-brake scales, in ``[0, 1]``."""
        scale = 1.0
        if self.p.vol_target is not None:
            rv = self.realised_vol()
            if np.isfinite(rv) and rv > _EPS:
                scale *= min(1.0, max(self.p.vol_floor, self.p.vol_target / rv))
        if self.p.drawdown_threshold is not None:
            dd = self.drawdown()
            if dd > self.p.drawdown_threshold:
                span = self.p.drawdown_full - self.p.drawdown_threshold
                frac = min(1.0, (dd - self.p.drawdown_threshold) / max(span, _EPS))
                scale *= 1.0 - frac * (1.0 - self.p.drawdown_floor)
        return float(min(1.0, max(0.0, scale)))

    # ── the overlay itself ───────────────────────────────────────────────────

    def stops_to_execute(self) -> np.ndarray:
        """Held names whose loss from entry has breached the stop, ``[N]`` bool."""
        return self._stopped.copy()

    def register_stops(self) -> int:
        """Mark the current stops as acted on and start their cooldown."""
        n = int(self._stopped.sum())
        if n and self.p.stop_cooldown_steps:
            self._cooldown[self._stopped] = int(self.p.stop_cooldown_steps)
        self._stopped = np.zeros(self.n, dtype=bool)
        return n

    def apply(self, target_w: np.ndarray) -> np.ndarray:
        """Overlay a ``[N+1]`` target (cash at 0). Returns a new vector summing to 1.

        Quarantined names are zeroed, then the equity block is scaled by
        :meth:`exposure_scale`; whatever is removed becomes cash. The weights of
        the surviving names are **not** renormalised back up, because the point
        of both controls is to hold less equity, not to concentrate the same
        exposure into fewer names.
        """
        w = np.asarray(target_w, dtype=np.float64).copy()
        if w.shape != (self.n + 1,):
            raise ValueError(f"target_w must be shape [N+1]={self.n + 1}, got {w.shape}")
        if not self.p.any_enabled:
            # Bit-identical, not merely equal to within rounding. The sweep's
            # control arm runs through this path, and reassembling the vector
            # from `1 - sum(eq)` moves the cash weight by ~1e-16 — enough to
            # make a "no overlay" arm differ from no overlay at all.
            return w
        eq = w[1:]
        if self.p.stop_cooldown_steps:
            eq[self._cooldown > 0] = 0.0
        eq *= self.exposure_scale()
        out = np.empty_like(w)
        out[1:] = np.maximum(eq, 0.0)
        out[0] = max(0.0, 1.0 - float(out[1:].sum()))
        return out


def bool_(a: np.ndarray) -> np.ndarray:
    """`np.asarray(..., dtype=bool)` that keeps mypy strict happy."""
    return np.asarray(a, dtype=bool)
