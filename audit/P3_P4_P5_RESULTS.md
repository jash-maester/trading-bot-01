# P3 / P4 / P5 — measured results

Runs of 2026-09-06, all on the r4_v2 signal, split `oos_r4_v2`
(2016-07-01..2024-06-28, 1980 dates), MLflow experiment `allocator`, gate
verdict PASS. Logs in `audit/r4_v2/grid_*.log`. Every arm carries a
null-signal control.

These supersede `audit/R4_R5_RESULTS.md` entirely: the environment's execution
rule changed under them (`cec3c3e`), so no figure there is comparable.

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
