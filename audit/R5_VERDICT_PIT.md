# R5 verdict on a point-in-time universe

Four views of the same allocator, on a universe rebuilt from NSE's full-market
archive so that membership is decided from data available at the time
(`audit/S1_SURVIVORSHIP.md`). This is the assessment; the measurements live in
`audit/S2_ALLOCATOR_ON_PIT.md`.

---

## The four views

| # | span | windows | best arm | excess/yr | 95% CI | t | arms clearing |
|---|---|---|---|---|---|---|---|
| 1 | 2016–24, monthly, no band | 8 | K=20 no band | **−0.0139** | — | — | 0 of 13 |
| 2 | 2016–24, band swept | 8 | K=20 band .010 monthly | +0.0374 | [−0.0016, +0.0672] | 1.98 | 0 of 19 |
| 3 | 2016–24, + cadence | 8 | K=20 band .010 **quarterly** | +0.0421 | [+0.0011, +0.0830] | 2.11 | 1 of 17 |
| 4 | **2005–24** | **13** | K=20 band .010 quarterly | **+0.0456** | **[+0.0111, +0.0802]** | **2.83** | 3–4 of 49 |
| 5 | **2025–26 holdout, unseen** | — | K=20 band .010 **monthly** | +0.0171 | [−0.0520, +0.0842] | 0.40 | **0 of 57** |

## What is established

**The signal is real.** R4's gate passes on 13 windows spanning 2005–2024:
5d mean IC +0.0240, t 6.43, 12/13 windows positive. Extending from 8 windows to
13 *raised* t from 4.55 to 6.43 while leaving the IC unchanged — the signature
of a real effect measured over more independent periods, not a diluted one.

**The no-trade band is the single most important lever, and it replicates
everywhere.** At K=20 it takes the allocator from below equal-weight to above
it, cutting scrips sold per rebalance from 77.8 to 8.1. Its ordering holds on
the in-sample span, the extended span, and — as a point estimate — the holdout.

**Cost is not the constraint.** Isolated: the tax drag at band 0.010 is 0.6pp,
*below* equal-weight's own 0.7pp, and the edge is +0.0374 with tax and +0.0370
without. The demat bill falls to ₹1,810 on ₹1 lakh. Backtest and paper broker
agree to 0.15%.

**Concentration is not the constraint either.** Raising K is monotonically
worse — K=250 reaches −0.045 against equal-weight — so the earlier failure was
never about holding too few names.

## What is NOT established

**The quarterly refinement does not survive the holdout, and it reverses.**

| arm | 2005–24 (13 windows) | 2025–26 holdout |
|---|---|---|
| band .010 **quarterly** | **+0.0456** (t 2.83) | **+0.0015** (≈ 0) |
| band .010 **monthly** | +0.0385 (t 2.68) | **+0.0171** (t 0.40) |

In sample quarterly beats monthly; out of sample monthly beats quarterly and
quarterly is indistinguishable from zero. Neither difference is significant, so
this is not proof the effect is absent — but it is the opposite of
corroboration, and quarterly was the change that produced the first passing
gate. **It should not be treated as validated.**

**Nothing clears on the holdout — 0 of 57 against either baseline.** That is
mostly power: 295 sessions of a market where equal-weight returned 1.8% gives
intervals of ±6 to ±25pp, which would miss an effect several times this size.
The holdout can confirm a sign, not a magnitude.

**The multiple-comparison arithmetic does not support a clean pass.** The
21-year result is 3–4 passes from 49 arms; at 95% you expect ~2.5 by chance. A
Bonferroni correction at α/49 needs t ≈ 3.4 and the best arm reaches 2.83. What
argues against pure noise is the *structure* — the arms that clear are the three
highest point estimates, share one configuration family (band 0.010, no stop),
and clear against both baselines — but structure is an argument, not a test.

**Every risk-controlled arm fails.** All stopped and vol-stopped variants miss
(t 1.28–1.87), so the passing arms carry a **−0.49 maximum drawdown** over the
span. A book that halves is a real objection independent of any statistic.

## Verdict

**R5: NOT PASSED.** The criterion is a paired interval excluding zero against
the best baseline, and the only span where that holds is one on which the
configuration was selected. The genuinely unseen span does not corroborate the
specific choice that produced the pass.

What the evidence does support, stated at its actual strength:

* a **real signal** (R4, 13 windows, t 6.43) — high confidence;
* a **real and mechanically-explained cost effect** from the no-trade band —
  high confidence, replicated in three views;
* an **edge of roughly +3 to +4.5%/yr before selection adjustment** —
  moderate confidence;
* **quarterly over monthly** — low confidence, contradicted out of sample;
* **tradeable today** — not supported.

## What would change it

Not another sweep. At 49 arms the search is already the dominant source of
uncertainty, and a fiftieth arm makes it worse rather than better.

The one thing that generates independent evidence is **forward paper trading**
of a single pre-committed configuration — no re-selection, no re-tuning — with
a review point fixed in advance. Data that has not happened yet cannot be
overfitted, which is exactly the property every span above lacks.

If that is done, the configuration to commit to is **K=20, band 0.010,
monthly** rather than quarterly: it is nearly as strong in sample (+0.0385,
t 2.68), it is the better of the two out of sample, and it does not rest on the
one refinement the holdout contradicts.
