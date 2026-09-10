# The matched null control: the signal does not beat random out of sample

R5's allocator was always compared against `equal_weight`. Against a **random
signal running the identical machinery** — same universe, same K, same 1% band,
same stop, same costs, same tax — it wins every matched pair in sample and
loses almost every one out of sample.

```
uv run python scripts/run_allocator.py data=bhav_v1 +split=oos_r4_pit_long \
  +signal_tag=r4_pit_long +require_gate_pass=true +apply_tax=true \
  '++allocator.null_control=true' '++allocator.k_grid=[20,30]' \
  '++allocator.band_grid=[0.010]' '++allocator.freq_grid=[monthly,quarterly]' \
  '++allocator.horizon_grid=[20d]' '++allocator.nav_dir=audit/navs_nullctl_long'
# and the same with +split=holdout +signal_tag=r4_pit_long_holdout
```

---

## The control had to be repaired first

`run_allocator.py`'s `null_control` took `band_grid[0]` and applied **no risk
overlay**. With `band_grid=[0.0,0.005,0.010]` it therefore ran band **0.0** with
no stop, against arms carrying band 0.010 *and* a stop — pricing
"signal + band + stop" against "nothing". It also wrote no NAV file, so
`audit/navs_long` and `audit/navs_p4` contain no control at all and every risk
comparison made before 2026-09-09 was uncontrolled.

Its own docstring said the control "must run the same machinery". It did not.
Fixed to sweep the same `risk_grid × band_grid` and write NAVs.

## In sample, the signal wins every pair

Sharpe, K=30, band 0.010, 2005–2024, after tax and all charges:

| arm | null (random) | allocator (R4) | signal adds |
|---|---|---|---|
| monthly, none | 0.684 | 0.759 | +0.075 |
| monthly, stop10 | 0.841 | 0.853 | +0.012 |
| monthly, stop15 | 0.805 | 0.834 | +0.029 |
| monthly, volstop | 0.796 | 0.841 | +0.045 |
| quarterly, none | 0.772 | 0.815 | +0.043 |
| quarterly, stop10 | 0.870 | **1.024** | +0.154 |
| quarterly, stop15 | 0.775 | 0.887 | +0.112 |
| quarterly, volstop | 0.922 | 0.993 | +0.071 |

**8 of 8 to the signal** — but the increments are small next to what the
*machinery* delivers on its own. Equal-weight monthly is Sharpe 0.487; a random
top-30 with the band reaches 0.684, and with a stop 0.841. The signal then adds
0.012.

The same decomposition on return. R5's headline was **+0.0431/yr** over
equal-weight (K=30, band .01, monthly). The random control gets **+0.0261** of
that by itself. **The signal's marginal contribution is ~+0.017/yr, not
+0.043** — roughly 40% of what the equal-weight comparison attributed to it.

## Out of sample, it loses

Holdout 2025-04 → 2026-09, same matched pairs, K=30:

| arm | null (random) | allocator, 13-window | allocator, 8-window |
|---|---|---|---|
| monthly, none | 0.082 | **−0.044** | 0.224 |
| monthly, stop10 | 0.119 | **−0.338** | 0.188 |
| monthly, stop15 | 0.039 | **−0.229** | — |
| monthly, volstop | **0.307** | **−0.248** | 0.163 |
| quarterly, none | 0.340 | 0.271 | 0.270 |
| quarterly, stop10 | 0.291 | 0.016 | 0.262 |
| quarterly, stop15 | 0.407 | 0.212 | 0.275 |
| quarterly, volstop | **0.433** | −0.134 | 0.300 |

Random wins **8 of 8** against the 13-window model and **5 of 7** against the
8-window one — **13 of 15 overall**.

**Correction, 2026-09-10.** An earlier version of this line quoted a sign test
at p ≈ 0.004. That was wrong: the 15 pairs are not independent. They share the
same 15 months, the same random book (identical seed, identical null rows in
both runs), and two signal artefacts from the same model family with
overlapping training. The effective number of independent comparisons is
nearer two or three than fifteen, at which a sign test says nothing. What the
table supports is weaker and should be stated as such: **the point estimates
went the wrong way, consistently across variants, on one 15-month draw.** That
is evidence against the signal converting to portfolio value; it is not a
refutation at any conventional level, and given the holdout's ±6–25pp
intervals it could not have been.

**This is the fourth reversal.** R5's four arms reversed perfectly; quarterly
cadence vanished; low-vol lost to the model; and now the signal's advantage
over random is in-sample only.

## A dissociation worth recording

`r4_pit_long_holdout` scores **+0.0459** rank IC at 20d on the holdout against
`r4_pit_holdout`'s **+0.0338** — the higher-IC model produces the *worse*
portfolio, and by a wide margin (monthly stop10: −0.338 against +0.188).

Rank IC over the full cross-section is not what a top-30 long-only book
consumes. The same disconnect appeared in the fundamentals work, where IC rose
while top-K forward returns did not (`audit/F3_FUNDAMENTALS_VERDICT.md`).
**R4's gate metric does not measure what the allocator needs**, which is a
defect in the gate, not in this run.

## What actually survives

The portfolio-construction effects, and they survive with a random signal:

| holdout, monthly | Sharpe | CAGR | MDD |
|---|---|---|---|
| equal_weight | 0.151 | +0.018 | −0.137 |
| **null/volstop, band .010** | **0.307** | +0.021 | **−0.061** |

Concentrating from ~350 names to 30, applying a 1% no-trade band and a
volatility-scaled stop **doubles Sharpe and more than halves drawdown against
equal-weight on unseen data, with no model at all**. On return it is a wash
(+0.0024/yr).

Caveats, stated rather than buried: 15 months, one path per arm, and 26 arms in
the table — any single Sharpe here is noisy, and `null/volstop` being the best
of them is partly luck. The load-bearing statistic is the **paired** 13-of-15,
which does not depend on picking an arm.

## Conclusion

**No signal-driven edge is demonstrated, and the holdout cannot demonstrate
one either way.** R4's IC is real and passes its own gate. Whether it converts
into portfolio value once concentration, the band and the stop are controlled
for is *not established*: in sample it does, modestly (~+0.017/yr); on the
holdout the point estimates go the other way, on a sample far too short to
detect an effect of that size. Read "does not survive" throughout this document
as "is not confirmed", which is what the evidence actually supports.

What is left is a risk result, not an alpha result: a smaller, banded, stopped
book has roughly half the drawdown of equal-weight and about the same return.
That is worth something to an investor, and it needs no model — which also
means the daily job needs no Kite-dependent scoring, no artefact drift and no
retraining.

`audit/P1_PAPER_PREREGISTRATION.md` is **withdrawn** in its current form: it
froze a model-based configuration that this control refutes.
