# R5's gate, evaluated for the first time

`PROGRESS.md` carries R5's acceptance criterion verbatim:

> Beats best R2 baseline net of cost+tax, **paired bootstrap CI excluding zero**

The margin had been measured many times. **The interval never had been.** Until
now R5 was a measurement wearing a gate's name, and `CLAUDE.md` rule 1 forbids
building R6 or R7 on a gate that never opened — which is exactly the failure
that rule exists to prevent, recorded against M5.

```
uv run python scripts/run_allocator.py data=kite_v1 \
  +split=oos_r4_v2 +signal_tag=r4_v2 +require_gate_pass=true +apply_tax=true \
  '++allocator.nav_dir=audit/navs'
uv run python scripts/allocator_gate.py --nav-dir audit/navs
```

---

## Result: the statistical criterion is met, 8 arms of 8

Paired daily log-return difference against `EqualWeightRebalanced` at the same
cadence, moving-block bootstrap, block 21 (the monthly holding period), 10,000
resamples. Walk-forward OOS span `oos_r4_v2`, 1,920 sessions, after tax and
every Zerodha charge.

| arm | excess/yr | 95% CI | t | ACF₁ |
|---|---|---|---|---|
| K=20 no stop | **+0.1191** | [+0.0614, +0.1782] | 4.37 | 0.086 |
| K=20 stop 15% | +0.1035 | [+0.0322, +0.1757] | 3.17 | 0.065 |
| K=20 volstop | +0.0942 | [+0.0196, +0.1708] | 2.81 | 0.070 |
| K=20 stop 10% | +0.0824 | [+0.0049, +0.1650] | 2.41 | 0.066 |
| K=30 no stop | +0.1061 | [+0.0570, +0.1557] | 4.51 | 0.060 |
| K=30 stop 15% | +0.0881 | [+0.0235, +0.1530] | 3.00 | 0.054 |
| K=30 volstop | +0.0846 | [+0.0129, +0.1584] | 2.64 | 0.059 |
| K=30 stop 10% | +0.0777 | [+0.0033, +0.1549] | 2.38 | 0.053 |

Every interval excludes zero. The tightest margin is K=30 with a 10% stop, whose
lower bound is +0.0033 — a hair above zero, and worth remembering before anyone
quotes the stopped arms as comfortably ahead.

On the unseen 2025-26 holdout the same test gives 8 of 8 again, wider:
K=20 no stop **+0.1883/yr, CI [+0.0903, +0.2810]**. That table carries its own
caveat — `r4_v2_holdout` has no `gate.json`, so the runner stamps *"SIGNAL GATE
DID NOT PASS: these numbers are NOT evidence"* across it. The holdout is
corroboration, not the gate.

## Why paired, and why blocks

**Paired**, because both arms trade the same universe over the same days. Their
returns share every market-wide move, and testing them as two independent
samples would drown a real difference in volatility that cancels exactly.

**Blocks**, because a daily difference need not be i.i.d. — positions are held a
month, so consecutive excesses share a book. `bootstrap_mean_ci`'s own
calibration study found the i.i.d. version excluded zero in 33.7% of trials at a
nominal 5% for a persistent series, which is why the same moving-block
implementation the R4 gate uses is reused here rather than a fresh one.

The measured ACF₁ of the paired difference is **0.05–0.09**, an order of
magnitude below the 0.70–0.83 seen on raw IC series. Pairing removes the common
market factor, which is what carried most of the persistence. Block 21 is
therefore conservative for this statistic, not marginal — stated because a
reader should be able to check that the block choice was not doing the work.

## Two reasons this is NOT yet a PASS on the ledger

**1. R2 has not passed, and R5's criterion names it.** The criterion says "best
R2 baseline". R2 asks for 5 baselines × 3 frequencies × 4 benchmarks and only
`EqualWeightRebalanced` at one cadence has ever been run. All five agents exist
(`trader/env/baselines.py`: `EqualWeightRebalanced`,
`EqualWeightFrozenUniverse`, `MomentumTopK`, `SixtyFortyCash`, `RandomPolicy`) —
`run_allocator.py` simply only calls one. So the comparison above is against the
strongest baseline *measured*, not the strongest that exists, and marking R5
PASS while its predecessor is incomplete is precisely what `CLAUDE.md` rule 1
forbids. Completing R2 is cheap and is now the binding constraint.

**2. The universe is not point-in-time.** Measured 2026-09-08: of the 605 names
carrying ≥₹5 crore of median daily turnover in 2021, this universe contains
**292 — 48.3%** — and 55 of those 605 had stopped trading by 2026.

That second point does **not** invalidate the interval above, and the distinction
matters. This is a *paired* test: both arms trade the same universe on the same
days, so a survivorship-inflated universe lifts both and largely cancels in the
difference. What it does bound is the *level* — "the allocator beats equal-weight
by 12%/yr on this universe" is supported; "an investor would have earned that"
is not, until the universe is rebuilt from the full-market bhavcopy
(`scripts/fetch_bhavcopy.py`).

## Status

**R5: CONDITIONAL PASS.** The statistical criterion is met with room to spare on
both the walk-forward span and the holdout. It becomes an unconditional pass
when R2's baseline grid is run and no baseline beats `EqualWeightRebalanced`.

Artefact: `audit/r5_gate.json` (walk-forward), `audit/r5_gate_holdout.json`.
