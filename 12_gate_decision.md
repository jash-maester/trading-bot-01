# 12 — The R4 gate: decision and evidence

Decided 2026-09-06. The R4 signal gate is now a **window-level test**. The
every-window rule it replaces is retained as a reported diagnostic and is no
longer decisive. This document is the record: what the question was, what was
measured, what was decided, and what the decision does not settle.

---

## 1. The question

R4 trains a cross-sectional return model and scores it out of sample on eight
walk-forward windows covering 2016-07 to 2024-06. The gate answers one
question: *is there enough evidence of predictive skill that building an
allocator and an RL layer on this signal is not building on sand?* CLAUDE.md
rule 1 forbids building on a gate that has not passed.

Two runs (`r4_v1`, `r4_v2`) both returned FAIL under the rule then in force,
each on two of eight windows, each by a hair. Meanwhile the allocator built on
the same signal beat equal-weight at every cadence with a null-signal control
correctly below it (`audit/R4_R5_RESULTS.md`). Either the signal was bad and
the allocator result was luck, or the gate was asking a question the data
could not answer at that resolution.

## 2. The rule that failed

Per horizon, **every** window had to satisfy all of: >= 30 scorable days,
mean rank IC > 0.02, and a moving-block bootstrap 95% interval on the daily IC
series excluding zero.

The r4_v2 record at the 5-day horizon:

| Window | Mean IC | 95% interval | n_days | n_eff | Verdict |
|---|---|---|---|---|---|
| W1 | +0.0499 | +0.0305, +0.0668 | 242 | 34 | pass |
| W2 | +0.0625 | +0.0413, +0.0827 | 244 | 40 | pass |
| W3 | +0.0229 | **−0.0066**, +0.0527 | 239 | 29 | fail: interval |
| W4 | +0.0295 | +0.0083, +0.0493 | 241 | 35 | pass |
| W5 | +0.0413 | +0.0200, +0.0604 | 246 | 35 | pass |
| W6 | +0.0458 | +0.0236, +0.0678 | 244 | 34 | pass |
| W7 | +0.0433 | +0.0287, +0.0571 | 242 | 41 | pass |
| W8 | **+0.0180** | +0.0023, +0.0343 | 242 | 49 | fail: mean |

Source: `audit/r4_v2/summary.json`. Eight of eight positive. The two failures
are on different conditions and W8 misses the floor by 0.002.

## 3. What was measured

### 3.1 The resolution of one window

Daily ICs are strongly autocorrelated — measured lag-1 ACF 0.75 at h=5, 0.91
at h=20 — because consecutive forward returns overlap and a 60-day-lookback
encoder scores near-identical inputs on consecutive days. At ACF 0.75 a
242-day window carries an *effective* sample of about 35 days
(`n · (1−r)/(1+r)`). The `n_eff` column above is that number per window.

With daily-IC sd 0.094 (pooled mean 0.0392 / ICIR 0.418), the standard error
of one window's mean is roughly 0.094/√35 ≈ 0.016. A true IC of 0.04 gives an
expected t of 2.5 in one window — and about a one-in-four chance that a
perfectly good window's interval brushes zero. Eight independent one-in-four
chances is not a rule a good signal can be expected to pass.

### 3.2 Power, simulated

`scripts/profiling/gate_power.py` draws 8 windows × 242 days of daily IC from
an AR(1) with the measured ACF and sd, adds a chosen true mean, and asks each
rule for a verdict. The every-window rule uses the real moving-block bootstrap
(block = h = 5). 600 trials per row.

| True IC | Every-window rule | Window-level rule |
|---|---|---|
| 0.00 | 0.0% | 0.0% |
| 0.01 | 0.0% | 2.7% |
| 0.02 | 0.0% | 47.7% |
| 0.03 | 4.8% | 97.0% |
| 0.04 | **33.5%** | 100.0% |
| 0.06 | 92.7% | 100.0% |

Both rules reject a null signal every time. The every-window rule passes a
signal with a true IC of 0.04 — the size of the one we have — one time in
three, and one of 0.03 one time in twenty. It effectively demands a true IC
near 0.06 to pass reliably, which is not the bar anyone set. The 47.7% at
exactly 0.02 is what a threshold should do at its own threshold.

### 3.3 The window-level statistic on the real runs

Treating each window's OOS mean IC as one observation:

| Run | Horizon | Windows | Mean of window ICs | sd | t (df=7) | t_crit | Positive |
|---|---|---|---|---|---|---|---|
| r4_v2 | 5d | 8 | +0.0392 | 0.0148 | **7.50** | 2.365 | 8/8 |
| r4_v2 | 20d | 8 | +0.0437 | 0.0233 | **5.31** | 2.365 | 8/8 |
| r4_v1 | 5d | 8 | +0.0414 | 0.0155 | 7.54 | 2.365 | 8/8 |
| r4_v1 | 20d | 8 | +0.0454 | 0.0295 | 4.36 | 2.365 | 8/8 |

Regenerate: `uv run python scripts/regate.py audit/r4_v2`.

## 4. The decision

Per horizon, over windows with a usable OOS record (>= 30 scorable days;
fewer fails the horizon outright, as before):

1. **at least 4 windows** — a t-test on fewer is noise with a decimal point;
2. **mean of the window ICs > 0.02** — materiality: the effect has to be
   large enough to survive Indian delivery costs at all;
3. **one-sample t on the window ICs > the two-sided 95% critical value at
   n−1 df** — agreement: the windows say the same thing beyond chance;
4. **>= 75% of windows positive** — robustness: one strong era is not
   carrying the mean.

One clean horizon is a PASS. The every-window rule is still evaluated and
written to `gate.json` as `strict_every_window` with its reasons, so nothing
it would have said is hidden. `gate.json` carries `gate_rule` naming which
rule produced the verdict; consumers (`run_allocator.py`,
`train_allocator_rl.py`) read `verdict` and are unaffected.

Implementation: `supervised.gate_verdict`, `WindowLevelStats`,
`t_critical_95` (a table; scipy is not a dependency). Tests:
`tests/unit/test_supervised.py::test_gate_*` including the r4_v2 shape as a
literal fixture.

### Why this and not the alternatives

- **Keep the rule and raise the sample.** Each window is one year of test
  data by design (so that the allocator's yearly refit is what gets tested).
  Longer windows means fewer of them, and the effective sample per window is
  set by autocorrelation, not calendar length.
- **Pooled daily bootstrap only.** The pooled interval is also miscalibrated
  (11.6% false-positive at h=5 under a persistent null, per
  `bootstrap_mean_ci`'s own docstring) and cannot see whether one era is
  carrying the rest. The window-level t sidesteps the daily autocorrelation
  entirely by using window means, and the sign check covers the era question.
- **HAC / Newey-West on the daily series.** Measured at 13.6% false-positive
  in the integration pass; same root cause — too few effective observations
  for any reweighting to reach nominal.
- **Drop significance and keep only the mean.** Loses the guard against a
  wildly dispersed set of windows whose mean happens to clear 0.02.

### Is this moving the goalposts?

The new rule passes where the old one failed, so the question has to be
asked. The answer rests on §3.2: the old rule failed two-thirds of good
signals of exactly this size. A rule that cannot pass the thing it is meant
to detect is not a high bar, it is a broken instrument. The new rule's
false-positive rate is the same as the old one's (zero at null), its power is
what a gate's power should be, and the retired rule's verdict is still
printed beside every new one.

## 5. What this does not settle

- **Survivorship.** The universe is today's 504 active names, not
  point-in-time. A window-level t of 7.5 on a survivor-biased universe is a
  precise measurement of a possibly biased quantity. This gate is a screen on
  signal *presence*; it says nothing about whether the signal survives a
  point-in-time universe. That is P4 in `11_cost_defect_and_fix_plan.md` and
  it is the next thing that can invalidate everything.
- **The 20-day horizon is thinner than its t suggests.** Its per-window n_eff
  is 8–21 days. The window-level t (5.31) is honest about that only insofar
  as the window means themselves are; treat 20d as supported but less so
  than 5d.
- **0.02 is a chosen floor, not a derived one.** The allocator turns IC 0.039
  into Sharpe 2.03 against 1.44. What IC the allocator needs to beat
  equal-weight *net of the flat demat fee at ₹1 lakh* has not been measured
  and would be the principled way to set this number.
- **The economic result remains the real validation.** CLAUDE.md rule 2:
  nothing is validated without the downstream result and a run ID. The gate
  passing means the RL stage may be *attempted*; it does not mean the signal
  is proven.
