# Is a news veto on the stop worth building? Measured without buying any news

`13_fundamentals_and_news.md` §4 designs the news arm as a **veto on the
stop-loss**, not a ranking signal: a held name breaching its stop is sold if the
fall carries adverse news and held if it looks like noise. That is the IndiGo
case — a sound company falling on sentiment and recovering, against one whose
fall is structural.

§4 also says the arm "cannot be validated" because we have no news history, and
proposes validating it forward on the stop events the strategy generates. That
is true of the *signal*. It is not true of the **ceiling**, and the ceiling can
be measured today, for free, from events the strategy has already produced.

```
RUN_TAG=stopev bash scripts/stop_events_run.sh      # on the training box
uv run python scripts/stop_veto_headroom.py
```

`scripts/run_allocator.py` now records every stop it fires — entry price, exit
close, threshold, position weight — under `+allocator.stop_events_dir=`. The
events below come from the same runs that produced the tables in `audit/F3`,
not from a replay that might diverge from them.

---

## Verdict

**Do not build it now.** Even a veto with *perfect foresight* is worth about
**+1.7 to +2.3% a year** at the horizon where the money is actually idle, and
nothing measurable separates the names that recover from the names that do not:
R4's own score gives Spearman +0.014, the depth of the fall gives −0.068, and
49–52% of stopped names recover — a coin flip.

The blocker is not cost. At ~40 calls a day the plan's $19/year estimate holds.
The blocker is that the arm has a bounded and modest upside, no measurable
prior that news would capture any of it, and no way to find out before running
it live.

The same events surfaced a promising lead that needed no new data source —
60-day mean reversion, and a 21-step quarantine that might be blocking it. It
tested well in sample and **failed to replicate on the holdout**; §5 has both
tables. The default is unchanged.

---

## 1. What the stops actually do

Three arms, `r4_v2` on `oos_r4_v2` (2016-09 … 2024-06, 7.9 years), K=20 monthly,
after tax.

| arm | stops/yr | median loss at stop | median weight |
|---|---|---|---|
| `stop10` (fixed 10%) | 95.1 | −12.1% | 0.4% |
| `stop15` (fixed 15%) | 56.9 | −16.5% | 0.5% |
| `volstop` (1.0× vol) | 72.8 | −10.7% | 0.6% |

**95 stops a year, not the ~50 §4 assumed.** The call budget it sized (held ~20
+ candidates ~20, daily) is unaffected — that is a per-day figure — but the
number of decisions a veto would be making is roughly double what was planned
for.

## 2. Which horizon is the right one, and why it decides the answer

When a stop fires, the proceeds sit in cash. The cooldown bars *re-buying that
name* for 21 steps, but the cash itself is redeployed at the next monthly
rebalance. That gap is measured, not assumed, because the sign of the whole
result depends on it:

| arm | cash idle until the next rebalance |
|---|---|
| `stop10` | mean 10.5d, median 11d, p10 3, p90 18 |
| `stop15` | mean 11.0d, median 11d, p10 4, p90 19 |
| `volstop` | mean 10.3d, median 10d, p10 3, p90 18 |

**So 10 days is the window over which stopping actually costs or saves
anything.** The 21d and 60d columns overstate both the cost and the ceiling,
because by then the money is working again somewhere else.

That distinction is not a technicality. It flips the sign of the result.

| arm | horizon | sold name | market | **excess** | recovered | cost of stopping |
|---|---|---|---|---|---|---|
| `stop10` | 5d | −0.11% | −0.13% | +0.02% | 46.1% | **−2.07%** |
| `stop10` | **10d** | −0.26% | −0.06% | **−0.21%** | 49.3% | **−3.93%** |
| `stop10` | 21d | +2.03% | +0.89% | +1.15% | 53.7% | +4.20% |
| `stop10` | 60d | +8.46% | +4.10% | +4.36% | 54.4% | +33.49% |
| `stop15` | **10d** | −1.36% | −1.03% | **−0.32%** | 50.8% | **−3.28%** |
| `volstop` | **10d** | −0.04% | −0.27% | **+0.23%** | 52.4% | **−0.49%** |

A negative cost means stopping *saved* return. At 10 days, in all three arms,
**stopping is already the right call** — the sold names drift down a little
further, and the excess over the market is within a quarter of a percent of
zero.

The strong recovery shows up at 21 and 60 days, well after the cash has been
redeployed.

## 3. The ceiling on a perfect veto

An oracle that held every name which recovered and sold every one that fell —
no real veto beats this:

| arm | 10d ceiling | 21d ceiling | 60d ceiling |
|---|---|---|---|
| `stop10` | **+2.28%/yr** | +3.57%/yr | +8.25%/yr |
| `stop15` | **+1.67%/yr** | +2.69%/yr | +4.91%/yr |
| `volstop` | **+2.13%/yr** | — | — |

Against a 0.365 CAGR on the `stop10` arm, +2.28%/yr is not nothing. It is the
number a news veto is competing for, and it requires being right on **every
one** of 95 calls a year.

## 4. Nothing available separates the two groups

If something we already have predicted which stopped names recover, the answer
would be to read that, not to buy news.

| arm | horizon | R4's 20d score at the stop | depth of the fall |
|---|---|---|---|
| `stop10` | 10d | +0.014 (p 0.70) | −0.068 (p 0.062) |
| `stop15` | 10d | −0.029 (p 0.54) | +0.056 (p 0.23) |
| `volstop` | 10d | −0.024 (p 0.57) | −0.015 (p 0.72) |
| `stop10` | 60d | +0.024 (p 0.52) | **−0.148 (p 0.000)** |
| `stop15` | 60d | **+0.110 (p 0.020)** | −0.015 (p 0.75) |

At the 10-day horizon that matters, everything is noise. The two significant
numbers are both at 60 days, and they point at §5 rather than at news.

## 5. The lead this produced, and why it did not survive

At 60 trading days a stopped name beats the market by **+4.36%** (`stop10`), and
**the depth of the fall that triggered the stop predicts the size of the
recovery**: Spearman −0.148, p 0.000, n 734. Deeper falls bounce harder. That is
idiosyncratic mean reversion, it is large, and it is the strongest relationship
anywhere in this document.

Meanwhile the overlay bars re-buying a stopped name for 21 steps. If R4 would
have picked the name back up inside that window, the quarantine is refusing a
trade the data says was good. `scripts/cooldown_sweep.sh` tests exactly that —
same signal, same K, same cadence, same thresholds, only `stop_cooldown_steps`
varying, for both stop shapes.

**In sample it looked like a real finding.** On `oos_r4_v2`, K=20 monthly 20d,
after tax:

| cooldown | volstop Sharpe | volstop CAGR | fixed-10% CAGR |
|---|---|---|---|
| 0 | 1.940 | 0.396 | 0.370 |
| **5** | **1.946** | **0.397** | **0.376** |
| 21 (current) | 1.928 | 0.381 | 0.365 |
| 42 | 1.911 | 0.368 | 0.368 |

Monotone in the volatility-scaled arm, +1.6pp of CAGR from shortening 21 → 5,
Sharpe up as well, and both stop shapes agreeing. The higher turnover and the
extra ₹8,452 of demat fees are already inside those CAGRs.

**On the unseen 2025-26 holdout it reverses and collapses.**

| cooldown | volstop Sharpe | volstop CAGR | volstop MDD | fixed-10% CAGR |
|---|---|---|---|---|
| 0 | 2.365 | 0.279 | −0.070 | 0.253 |
| 5 | 2.353 | 0.275 | −0.070 | 0.254 |
| **21 (current)** | **2.391** | **0.279** | **−0.067** | 0.254 |
| 42 | 2.375 | 0.270 | −0.069 | 0.248 |

The current default is at or near the top of both shapes — best Sharpe, best
drawdown, tied best CAGR — and the whole spread is 0.9pp against 1.6pp in
sample. The in-sample ordering was almost certainly selection across the eight
arms I chose to look at.

**No change to `stop_cooldown_steps`. It stays at 21.**

Two honest caveats on the holdout, in both directions. It is 355 sessions with
~17 rebalances, so it has little power to resolve a 1pp effect — this is not
proof the effect is absent, only that it did not replicate. And
`r4_v2_holdout` carries no `gate.json`, so the runner stamped *"SIGNAL GATE DID
NOT PASS: these numbers are NOT evidence"* across the table; they are
measurements of a signal whose gate was never computed.

What this cost was about twenty minutes, and what it bought was not changing a
production default on a number that does not replicate.

## 6. Predictions, scored

Recorded in `scripts/stop_veto_headroom.py` before the numbers existed:

> stopped names average between −2% and +1% over the next 21 days … roughly
> 40–45% recovering … the oracle ceiling is worth several points of gross CAGR
> … R4's score separates the two groups hardly at all, |Spearman| < 0.05.

- 21-day average −2% to +1% — **wrong.** It is +2.03% and +1.89%, above my
  range. These names bounce harder than I expected.
- 40–45% recovering — **wrong.** 46–49% at 5 days and 54–55% at 21 days. I
  under-estimated it at every horizon.
- Ceiling worth several points — **right at 21d and 60d**, overstated for the
  10-day horizon that turns out to be the relevant one (+1.7 to +2.3%).
- R4's score separates hardly at all — **right**: +0.014 at 10d, and the one
  value above 0.05 (+0.060 at 21d) is not significant.

Two of four wrong, and being wrong about the bounce is what made the horizon
question decisive rather than incidental.

And on the cooldown sweep, recorded in `scripts/cooldown_sweep.sh` before it
ran:

> the effect is SMALL and possibly negative … I expect |delta CAGR| < 1pp
> between cooldown 0 and 21.

Wrong in sample — it was +1.5pp, and monotone. Right on the holdout, where
cooldown 0 and 21 both give 0.279. The prediction was correct about the world
and wrong about the in-sample run, which is the failure mode this whole section
exists to catch.

## 7. What would change the verdict

Not a better sentiment model. The quantity a news veto has to forecast is the
**10-day idiosyncratic return of a name that has just fallen 10–16%**, and that
quantity averages −0.32% to +0.23% across the arms with a 49–52% split. There is
no mispricing here for news to resolve at that horizon.

The arm becomes worth revisiting if either of these changes:

* **The rebalance cadence lengthens.** At monthly, cash is idle ~10 days. At
  quarterly it would be idle ~30, the relevant horizon moves toward the window
  where stopped names do recover strongly, and the ceiling roughly triples.
* **The book gets big enough that +2%/yr of ceiling justifies unvalidatable
  engineering.** At ₹1 lakh the perfect-foresight ceiling is ~₹2,300/yr against
  ~₹1,600/yr of API cost. At ₹10 lakh it is ₹23,000 against the same ₹1,600, and
  capturing even a fifth of it would pay.

Neither is true today.
