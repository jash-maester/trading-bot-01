# P3 / P4 / P5 — measured results

Runs of 2026-09-06, all on the r4_v2 signal, split `oos_r4_v2`
(2016-07-01..2024-06-28, 1980 dates), MLflow experiment `allocator`, gate
verdict PASS. Logs in `audit/r4_v2/grid_*.log`. Every arm carries a
null-signal control.

These supersede `audit/R4_R5_RESULTS.md` entirely: the environment's execution
rule changed under them (`592e175`), so no figure there is comparable.

---

## 0. The re-baseline, and why the edge got bigger

`floor(target_value / open)` no longer decides *whether* a trade happens, only
its size. That removed a class of spurious one-share sells created by float32
weights. Monthly, band 0:

| Strategy | K | Sharpe | CAGR | MaxDD | Turn | DP fees | vs EW |
|---|---|---|---|---|---|---|---|
| equal_weight | - | 1.336 | 0.270 | -0.515 | 0.81 | ₹69,843 | — |
| null_signal (control) | 30 | 1.275 | 0.258 | -0.544 | 3.54 | ₹215,435 | -0.012 |
| allocator 20d | 20 | 1.906 | 0.479 | -0.558 | 3.62 | ₹147,172 | **+0.209** |
| allocator 20d | 30 | 1.902 | 0.452 | -0.530 | 3.55 | ₹166,761 | **+0.182** |
| allocator 5d | 30 | 1.697 | 0.402 | -0.553 | 3.54 | ₹179,693 | +0.132 |

The gap over equal-weight went from +0.074 to +0.182 against the superseded
table. The env fix removed spurious fees from *both* arms, but the allocator
was paying far more of them — it sells ~116 names per rebalance against
equal-weight's ~2 — so it gained more. The control still sits just below
equal-weight, which is the property that makes the gap attributable to signal.

---

## 1. P3 — the no-trade band. Hypothesis confirmed.

Monthly, K=30, 20d, band swept. The claim under test: a band cuts the **number
of names sold** (what the flat ₹15.34 bills) much faster than it cuts rupee
turnover.

| Band | Sharpe | CAGR | MaxDD | Turn | Sold/step | DP fees | vs EW |
|---|---|---|---|---|---|---|---|
| 0.000 | 1.902 | 0.452 | -0.530 | 3.554 | 115.6 | ₹166,761 | +0.182 |
| 0.002 | 1.779 | 0.408 | -0.539 | 3.462 | 38.2 | ₹55,147 | +0.138 |
| 0.005 | 1.737 | 0.384 | -0.529 | 2.500 | 18.5 | ₹26,615 | +0.114 |
| 0.010 | 1.815 | 0.385 | -0.506 | 1.995 | 13.1 | ₹18,945 | +0.115 |
| **0.020** | **1.992** | 0.414 | **-0.454** | 2.093 | 12.7 | ₹18,270 | +0.144 |

**The mechanism works exactly as predicted.** From band 0 to 0.020, names sold
fall **9.1x** (115.6 → 12.7) while turnover falls only **1.7x** (3.554 →
2.093). Demat fees fall 9.1x, from ₹166,761 to ₹18,270.

**But it costs return, and the shape is not monotone.** CAGR falls from 0.452
to a trough of 0.384–0.385 at bands 0.005–0.010, then recovers to 0.414 at
0.020. Sharpe does the same in reverse: worst at 0.005, and at band 0.020 it is
*better than no band at all* (1.992 vs 1.902) on a materially shallower
drawdown (-0.454 vs -0.530).

Reading it honestly: a band of 0.020 buys a **9x reduction in the fee that
dominates this system, a better Sharpe and a better drawdown, for 3.8
percentage points of CAGR**. Whether that is a good trade depends on whether
you are optimising return or risk-adjusted return. It is not a free win, and
the intermediate bands are strictly worse than both ends.

The non-monotonicity was predicted before the run (fewer, larger trades) and is
now confirmed on real data rather than synthetic.

---

## 2. P5 — capital. The baseline is uninvestable at ₹1 lakh.

The same strategy at `initial_cash=100000`:

| Strategy | K | Band | Sharpe | CAGR | MaxDD | Sold/step | DP fees |
|---|---|---|---|---|---|---|---|
| **equal_weight** | - | - | **0.000** | **0.000** | **0.000** | **0.0** | **₹0** |
| null_signal | 30 | 0.000 | 1.201 | 0.321 | -0.649 | 23.1 | ₹33,364 |
| allocator | 20 | 0.000 | 1.545 | 0.458 | -0.684 | 34.9 | ₹50,300 |
| allocator | 20 | 0.005 | 1.756 | 0.417 | -0.577 | 13.4 | ₹19,328 |
| allocator | 30 | 0.000 | 1.550 | 0.436 | -0.654 | 33.1 | ₹47,799 |
| allocator | 30 | 0.005 | 1.715 | 0.387 | -0.564 | 13.3 | ₹19,236 |

> **Equal-weight over 504 names returns exactly zero at ₹1 lakh because it
> never trades at all.** Each target position is ₹198, which floors to zero
> shares for any stock priced above ₹198 and is below the ₹500 minimum trade
> value regardless. The book is never built. Turnover 0, fees ₹0, NAV
> unchanged.

This is P5's central prediction, measured rather than derived. It has two
consequences:

1. **The `vs EW` column is meaningless at ₹1 lakh** and is omitted above. An
   allocator "beating" a baseline that never traded is not a result. Any
   comparison at this capital needs a baseline that can actually be held —
   a 30-name equal-weight book, or an index proxy.
2. **The band matters more at small capital.** At ₹1 lakh a 0.005 band takes
   K=20 from Sharpe 1.545 to 1.756 and cuts fees 2.6x, a larger improvement
   than the same band gives at ₹10 lakh. Small accounts are exactly where the
   flat fee bites, so this is the expected direction.

The allocator itself is viable at ₹1 lakh at K=20–40; it is the 504-name
baseline that is not.

---

## 3. P4 — survivorship. The edge survives the restriction.

Restricted to the 360 of 645 panel tickers (277 of 504 active) with at least
252 tradeable days before 2014-12-31, so no name selected by post-2014 listing
hindsight can carry it.

The decision rule was **pre-registered before the arm ran**: PASS iff
`G_restricted >= 0.5 · G_unrestricted` and `G_restricted > 0`, where G is the
allocator-minus-equal-weight CAGR gap *within* an arm.

| Arm | equal_weight CAGR | allocator CAGR | Control vs EW | G |
|---|---|---|---|---|
| Unrestricted | 0.270 | 0.452 | -0.012 | **+0.1815** |
| Restricted (pre-2015) | 0.266 | 0.373 | -0.027 | **+0.1074** |

    threshold = 0.5 x 0.1815 = +0.0907
    G_restricted = +0.1074  >=  +0.0907   and  > 0
    VERDICT: PASS — the edge survives at 59% of its unrestricted size.

The control stays below equal-weight in both arms, so the admissibility
condition holds too.

**What this does and does not establish.** It establishes that the edge is not
an artifact of picking names that listed and did well after 2014. It does
**not** establish that the strategy is free of survivorship, because of a
finding that is worse than the CLAUDE.md entry suggested:

> All 645 tickers share a single last-tradeable date, 2026-09-04. **Zero
> delistings across 21 years.** That is the signature of a present-day
> instrument dump. Companies that died between 2016 and 2026 are absent
> entirely, and their absence cannot be bounded from this data because the data
> holds no sample of them. `market.universe_snapshots` is empty (verified,
> `select count(*)` → 0), has no read site, and the only write path inserts a
> single row dated today — so it cannot be backfilled from anything in this
> repo.

Confounds in the restricted arm, printed by the arm on every run and recorded
here rather than buried: breadth falls 362.8 → 276.7 tradeable env names/date,
so K=30 becomes 8.27% → 10.84% of the cross-section (the allocator is more
selective, not a clean one-factor change); sector retention is 49–86% and
uneven; and 227 of 504 active tickers become all-zero phantom columns, so this
split must never be handed to a graph model.

---

## 4. Where this leaves things

- **The standing configuration is monthly, K=20–30, 20-day horizon**, now at
  Sharpe 1.90–1.91 and a +0.18 to +0.21 CAGR gap over equal-weight, with the
  control correctly at the baseline.
- **The band is a real lever with a real price**, and 0.020 is the interesting
  setting: 9x lower fees, better Sharpe, better drawdown, 3.8 points less CAGR.
- **Survivorship passed its pre-registered test**, which is the strongest
  evidence yet that the edge is not an artifact — bounded by the fact that
  delisting bias remains entirely unmeasured.
- **₹1 lakh is viable for the allocator but not for the wide baseline.** A new
  small-capital baseline is needed before any ₹1 lakh comparison means
  anything.

Nothing here has been walked forward on the 2025+ holdout, which remains
untouched.

---

## 5. The holdout, and R6

Added 2026-09-07. The 2025+ holdout has now been spent, once.

### 5.1 Getting a signal onto unseen data

`train_signal.py` writes predictions only for the windows it fits, so r4_v2
stopped at 2024-06-28 and there was nothing to trade after it.
`scripts/predict_signal.py` runs the frozen encoder forward over any panel.
It is inference only: the normalisation buffers ride inside the checkpoint, so
the new panel cannot enter them, and no `gate.json` is written because a gate
belongs to the fit.

The model's last training window ended **2021-12-31**, so the holdout asks it
to extrapolate **4.68 years**. That is a real weakness of a frozen-model
holdout — a deployment would refit first — and it makes the result below a
lower bound rather than a fair estimate.

### 5.2 Deterministic allocator on 2025-06-27..2026-09-04

| Strategy | K | Band | Sharpe | CAGR | MaxDD | vs EW |
|---|---|---|---|---|---|---|
| equal_weight | - | - | 0.685 | 0.090 | -0.120 | — |
| null_signal (control) | 30 | 0.000 | 0.587 | 0.082 | -0.135 | -0.008 |
| **allocator** | **20** | 0.000 | **1.892** | **0.320** | -0.117 | **+0.230** |
| allocator | 20 | 0.005 | 1.546 | 0.267 | -0.122 | +0.177 |
| allocator | 30 | 0.000 | 1.495 | 0.231 | -0.117 | +0.141 |

**The edge survives on data no part of this pipeline has seen**, with the null
control sitting just below equal-weight exactly as it does in sample. This is
the strongest evidence the project has produced.

Caveats that bound it: the span is ~15 months and one regime; the drawdown of
-12% says nothing about behaviour in a 2018 or 2020; survivorship still applies
and is arguably worse here, since names that traded in 2025 and delisted before
the 2026 dump are absent; and the model is 4.68 years stale.

### 5.3 R6 — the RL layer does not earn its place

First full PPO run over the allocator: 104 updates, 20,000 periods, 8 envs,
obs_dim 42, action_dim 17, on the 2016-2024 span. Training excess log return
rose steadily (0.231 → 0.258 by update 60), so it did learn something.

Evaluated at the distribution mean against the identical env driven by fixed
midpoint parameters:

| Span | Arm | Sharpe | CAGR | Turnover |
|---|---|---|---|---|
| Training (2016-2024) | fixed params | 2.468 | 0.582 | 6.47 |
| Training (2016-2024) | **RL policy** | 2.553 | **0.597** | 5.26 |
| Holdout (2025-26) | fixed params | 1.256 | **0.188** | 6.41 |
| Holdout (2025-26) | RL policy (update 100) | 1.006 | 0.156 | 5.05 |
| Holdout (2025-26) | RL policy (update 50) | 1.048 | 0.166 | 5.64 |

**It beats fixed parameters in sample by +1.5 CAGR points and loses out of
sample by 2.2 to 3.2.** Both checkpoints lose, so this is not a late-training
artefact. The policy did learn to trade less (turnover 6.4 → 5.1), it just did
not learn anything that generalised.

Read plainly: **the deterministic allocator is better than the RL layer over
it**, and R6 does not currently justify its complexity. That is a legitimate
and useful outcome, and it is the answer `10_architecture_revamp.md` §5 asked
for when it said "RL, IF IT RETURNS".

What would change the verdict, in rough order of promise: a longer holdout than
15 months; a refit signal rather than a 4.7-year-stale one; a larger training
budget than 20,000 periods; and reward shaping that does not already give the
fixed parameters most of what the policy could add.

---

## 6. Against NIFTY 50, decomposed

Added 2026-09-07. Equal-weight is the *internal* control; NIFTY 50 is the
benchmark an investor actually has. Both are needed, and running them together
splits the headline alpha into two very different components.

`scripts/benchmark_vs_nifty.py`, K=20 monthly 20d, beta/alpha from an OLS of
daily log returns on the index's.

### In sample, 2016-07..2024-06 (1919 days)

| Arm | CAGR | Sharpe | Beta | Alpha/yr | Corr | IR | Up cap | Down cap |
|---|---|---|---|---|---|---|---|---|
| ^NSEI | 0.139 | 0.776 | — | — | — | — | — | — |
| equal_weight | 0.265 | 1.336 | 0.833 | 13.7% | 0.796 | 0.949 | 0.938 | 0.801 |
| **allocator** | 0.470 | 1.906 | 0.857 | **32.0%** | 0.714 | 1.774 | 1.026 | 0.721 |

### Holdout, 2025-06..2026-09 (294 days)

| Arm | CAGR | Sharpe | Beta | Alpha/yr | Corr | IR | Up cap | Down cap |
|---|---|---|---|---|---|---|---|---|
| ^NSEI | -0.058 | -0.484 | — | — | — | — | — | — |
| equal_weight | 0.088 | 0.685 | 0.817 | 14.6% | 0.821 | 1.952 | 0.837 | 0.657 |
| **allocator** | 0.314 | 1.892 | 0.872 | **39.3%** | 0.749 | 3.435 | 0.950 | — |

### The decomposition, and why it matters

The alpha over NIFTY is **two effects stacked**, and only the second is skill:

| Step | In sample | Holdout |
|---|---|---|
| NIFTY → equal-weight (**being in this universe**) | +12.6 pp | +14.6 pp |
| equal-weight → allocator (**selection within it**) | +20.5 pp | +22.6 pp |

Both components are remarkably stable across two periods that look nothing
alike — one where the index compounded at 13.9% and one where it lost money.

**The universe premium is where survivorship most likely lives.** Equal-weighting
504 names beats the large-cap index by ~14%/yr in both spans. Some of that is a
genuine size premium: this is a mid- and small-cap-skewed book and NIFTY 50 is
not. But the universe contains **zero delistings across 21 years** (§3), so a
portion of that 14 points is the arithmetic of holding only companies that
survived. That portion cannot be measured from this data, and it contaminates
the *first* row of the decomposition, not the second.

**The selection component is the defensible one.** It is measured against
equal-weight *on the same survivor-biased universe*, so the bias is present in
both arms and largely differences out — which is also why `null_signal` sitting
just below equal-weight matters. +20.5 and +22.6 percentage points is the number
to quote for what the signal adds.

Beta is 0.86 and 0.87. This is **not** a market-neutral strategy; it is a long
equity book carrying close to full market risk, and in a 2018 or a 2020 it will
behave like one. What it does add, in sample, is asymmetry: 103% of the index's
up moves against 72% of its down moves.
