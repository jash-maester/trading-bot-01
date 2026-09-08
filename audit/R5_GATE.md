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

## Headline: it depends entirely on which baseline, and equal-weight is the wrong one

Against `EqualWeightRebalanced`, **8 arms of 8** clear the interval. Against
`MomentumTopK` — which `scripts/run_baselines.py` measured on 2026-09-08 as the
**stronger** baseline at monthly cadence — **1 arm of 8** clears it, marginally.

R5's criterion says "best R2 baseline". That is the momentum comparison, not the
equal-weight one. **R5 does not pass as written.**

## Against equal-weight: 8 arms of 8

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

## Against the actual strongest baseline: 1 arm of 8

`MomentumTopK` (K=20, monthly) returns **CAGR 0.273** against equal-weight's
0.257, net of the same costs and tax. It is a worse portfolio on every other
axis — Sharpe 0.896 vs 1.280, drawdown −0.725 vs −0.522, turnover 21.4 vs 0.81 —
but the criterion says "beats", and beating means return.

| arm | excess/yr | 95% CI | t | verdict |
|---|---|---|---|---|
| **K=20 no stop** | **+0.1060** | **[+0.0039, +0.2089]** | **2.02** | **PASS** |
| K=20 stop 15% | +0.0903 | [−0.0267, +0.2128] | 1.58 | FAIL |
| K=20 volstop | +0.0811 | [−0.0432, +0.2098] | 1.37 | FAIL |
| K=20 stop 10% | +0.0692 | [−0.0578, +0.2028] | 1.15 | FAIL |
| K=30 no stop | +0.0929 | [−0.0125, +0.1983] | 1.72 | FAIL |
| K=30 stop 15% | +0.0749 | [−0.0427, +0.1949] | 1.30 | FAIL |
| K=30 volstop | +0.0714 | [−0.0558, +0.2007] | 1.19 | FAIL |
| K=30 stop 10% | +0.0646 | [−0.0636, +0.1951] | 1.07 | FAIL |
| *equal_weight* | *−0.0132* | *[−0.1299, +0.1028]* | *−0.22* | *FAIL* |

Every arm still beats momentum **on average** — the mean excess is positive
throughout, and the allocator's CAGR of 0.416 is far above momentum's 0.273. What
fails is significance: a concentrated 20-name momentum book is volatile, so the
paired difference is noisy and the intervals are wide. Only the unstopped K=20
arm separates from zero, and its lower bound is **+0.0039**.

The last row is worth its own sentence: **equal-weight itself loses to momentum**
(−0.0132/yr), which is exactly why testing against equal-weight flattered every
arm above.

### What this means

* **R5 does not pass its criterion.** One configuration of eight does, at
  t = 2.02, and it is the arm with no risk control — the one with the −0.589
  drawdown that the stops exist to fix.
* The **stopped arms all fail** against this bar. They buy Sharpe and drawdown
  by giving up return, and against a high-return baseline that trade stops
  clearing the interval.
* At **weekly** cadence momentum collapses (CAGR 0.174, Sharpe 0.581) and
  equal-weight is strongest again. Momentum's win is specific to monthly.

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

## The R2 grid that produced the real bar

`scripts/run_baselines.py`, `oos_r4_v2`, after tax:

| baseline | freq | Sharpe | CAGR | MDD | Turn | sold/reb | DP ₹ |
|---|---|---|---|---|---|---|---|
| equal_weight | monthly | 1.280 | 0.257 | −0.522 | 0.81 | 47.9 | 69,122 |
| equal_weight_frozen | monthly | 1.256 | 0.256 | −0.534 | 0.80 | 42.1 | 60,746 |
| **momentum_topk** | monthly | 0.896 | **0.273** | −0.725 | 21.44 | 19.7 | 28,394 |
| sixty_forty | monthly | 1.408 | 0.155 | −0.325 | 0.39 | 26.6 | 38,411 |
| random | monthly | 0.954 | 0.188 | −0.576 | 7.52 | 141.1 | 203,439 |
| equal_weight | weekly | 1.216 | 0.245 | −0.531 | 1.07 | 20.6 | 127,874 |
| momentum_topk | weekly | 0.581 | 0.174 | −0.769 | 48.29 | 14.7 | 91,058 |

`random` is in the grid as a floor, not a contender: an arm that cannot beat
uniformly random logits over the same tradeable set is measuring the cadence and
the cost model rather than any selection. It clears 0.188 CAGR here, which is
itself a reminder of how much of a long-only Indian equity backtest is simply
the market.

**A defect this grid exposed.** `scrip_sell_days_per_rebalance` divided by every
*session* for baselines and by every *rebalance* for the allocator — two numbers
under one header, twenty-one times apart. Equal-weight read **2.3** where the
truth is **47.9**. The rupee column was always absolute and correct, so cost
conclusions drawn from `dp_charges_paid` stand; the per-rebalance column in any
table printed before 2026-09-08 does not. Fixed in
`trader/env/baselines.run_baseline_episode`, with a regression test.

## The other reason this is not a PASS

**The universe is not point-in-time.** Measured 2026-09-08 by
`scripts/pit_universe_report.py`: of the names clearing ≥₹5 crore of median
daily turnover, the 645 panelled names cover only **60.0%** on average over
2011–2020, and at 2016-01-01 twenty-seven of the ninety-nine missing had stopped
trading altogether — ABIRLANUVO, ALBK, ANDHRABANK, AMTEKAUTO, CAIRN, FRL. No
list drawn today can contain those.

*(An earlier version of this document quoted 48.3%. That compared against
`active_tickers()`, roughly ten points of which is `INACTIVE_SECTORS` holding
out five sectors to cap the observation width — a compute decision, not
survivorship. 60.0% against `all_tickers()` is the honest measure.)*

That second point does **not** invalidate the interval above, and the distinction
matters. This is a *paired* test: both arms trade the same universe on the same
days, so a survivorship-inflated universe lifts both and largely cancels in the
difference. What it does bound is the *level* — "the allocator beats equal-weight
by 12%/yr on this universe" is supported; "an investor would have earned that"
is not, until the universe is rebuilt from the full-market bhavcopy
(`scripts/fetch_bhavcopy.py`).

## Status

**R5: FAIL as written.** Its criterion is "beats best R2 baseline … paired
bootstrap CI excluding zero". The best R2 baseline at monthly cadence is
`MomentumTopK`, and against it exactly one of eight arms clears the interval, at
t = 2.02 with a lower bound of +0.0039.

Recorded plainly rather than softened, per `CLAUDE.md`: *"A failed gate is a
legitimate and often good outcome. Never soften a failure, never partially pass,
never 'pass with caveats.'"* The 8-of-8 result against equal-weight is real and
worth keeping, but it is not the gate.

What would change it, honestly:

* **A stronger allocator**, not a weaker baseline. Momentum-top-20 at monthly is
  a real bar and the allocator only just clears it.
* **Point-in-time universe.** Momentum-top-K is the strategy most exposed to a
  survivorship-selected universe — it buys whatever ran hardest, and a list
  chosen in 2026 guarantees those names survived. Rebuilding the universe may
  lower momentum's bar more than it lowers the allocator's. That is a
  prediction, recorded before the rebuild, and it may be wrong.

Artefacts: `audit/r5_gate.json` (vs equal-weight),
`audit/r5_gate_vs_momentum.json` (vs the real bar),
`audit/r5_gate_holdout.json`.
