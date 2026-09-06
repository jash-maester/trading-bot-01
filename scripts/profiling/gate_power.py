"""Power and false-positive rate of the R4 gate: every-window rule vs window-level t.

Regenerates the table in ``12_gate_decision.md``.  Every parameter is a
measurement from the r4_v2 run (``audit/r4_v2/gate.json``, 2026-09-06):

* 8 windows x 242 scorable OOS days each;
* daily-IC lag-1 autocorrelation 0.75 (``ic_acf_lag1`` at h=5);
* daily-IC standard deviation 0.094 (pooled mean 0.0392 / ICIR 0.418);
* the every-window rule's interval is the real moving-block bootstrap with
  ``block = h = 5``, as in ``supervised.bootstrap_mean_ci``.

Each trial draws a fresh 8-window daily-IC panel from an AR(1) with those
parameters and a chosen true mean, then asks each rule for a verdict.  The
pass rate at true IC 0 is the false-positive rate; elsewhere it is power.

    uv run python scripts/profiling/gate_power.py
"""
from __future__ import annotations

import numpy as np

from trader.training.supervised import t_critical_95

W, N, PHI, SD, H = 8, 242, 0.75, 0.094, 5
MIN_MEAN_IC = 0.02
MIN_POSITIVE_FRACTION = 0.75
TRIALS = 600

rng = np.random.default_rng(0)


def sim_daily(mu: float, trials: int) -> np.ndarray:
    """``[trials, W, N]`` daily ICs: AR(1) with stationary sd ``SD`` plus ``mu``."""
    eps_sd = SD * np.sqrt(1.0 - PHI**2)
    x = np.empty((trials, W, N))
    x[:, :, 0] = rng.normal(0.0, SD, (trials, W))
    for t in range(1, N):
        x[:, :, t] = PHI * x[:, :, t - 1] + rng.normal(0.0, eps_sd, (trials, W))
    return x + mu


def block_boot_lo(
    x: np.ndarray, block: int = H, n_boot: int = 300, alpha: float = 0.05
) -> np.ndarray:
    """Lower percentile of the mean under a wrapped moving-block bootstrap, per window."""
    trials = x.shape[0]
    nb = int(np.ceil(N / block))
    starts = rng.integers(0, N, (trials, W, n_boot, nb))
    idx = (starts[..., None] + np.arange(block)) % N
    idx = idx.reshape(trials, W, n_boot, -1)[..., :N]
    samp = np.take_along_axis(x[:, :, None, :], idx, axis=3)
    return np.quantile(samp.mean(axis=3), alpha / 2, axis=2)


def every_window_rule(x: np.ndarray) -> np.ndarray:
    wm = x.mean(axis=2)
    lo = block_boot_lo(x)
    return ((wm > MIN_MEAN_IC) & (lo > 0.0)).all(axis=1)


def window_level_rule(x: np.ndarray) -> np.ndarray:
    wm = x.mean(axis=2)
    m = wm.mean(axis=1)
    sd = wm.std(axis=1, ddof=1)
    t = m / (sd / np.sqrt(W))
    pos = (wm > 0.0).mean(axis=1)
    return (m > MIN_MEAN_IC) & (t > t_critical_95(W - 1)) & (pos >= MIN_POSITIVE_FRACTION)


def main() -> None:
    print(f"{'true IC':>8} {'every-window':>13} {'window-level t':>15}")
    print(f"({TRIALS} trials per row)")
    for mu in (0.0, 0.01, 0.02, 0.03, 0.04, 0.06):
        x = sim_daily(mu, TRIALS)
        ew = every_window_rule(x).mean()
        wl = window_level_rule(x).mean()
        print(f"{mu:>8.2f} {ew:>13.1%} {wl:>15.1%}")


if __name__ == "__main__":
    main()
