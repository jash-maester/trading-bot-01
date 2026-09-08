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

The same events surfaced a better lead that needs no new data source, in §5.

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
rebalance — on average about 10 trading days later. **So 10 days is the window
over which stopping actually costs or saves anything.** The 21d and 60d columns
overstate both the cost and the ceiling, because by then the money is working
again somewhere else.

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

## 5. The lead this actually produced

At 60 trading days a stopped name beats the market by **+4.36%** (`stop10`), and
**the depth of the fall that triggered the stop predicts the size of the
recovery**: Spearman −0.148, p 0.000, n 734. Deeper falls bounce harder. That is
idiosyncratic mean reversion, it is large, and it is the strongest relationship
anywhere in this document.

Meanwhile the overlay bars re-buying a stopped name for 21 steps. If R4 would
have picked the name back up inside that window, the quarantine is refusing a
trade the data says was good.

`scripts/cooldown_sweep.sh` tests exactly that — same signal, same K, same
cadence, same thresholds, only `stop_cooldown_steps` varying over {0, 5, 21,
42} for both stop shapes. It needs no new data source and no forward
validation.

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
