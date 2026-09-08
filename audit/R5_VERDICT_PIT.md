# R5 verdict on a point-in-time universe

Five views of the same allocator, on a universe rebuilt from NSE's full-market
archive so membership is decided from data available at the time
(`audit/S1_SURVIVORSHIP.md`). Measurements live in
`audit/S2_ALLOCATOR_ON_PIT.md`; this is the assessment.

Every number below regenerates from:

```
data/signal/r4_pit_long/gate.json                 # signal gate, 13 windows
audit/r5_gate_long_vs_equal_weight_monthly.json   # allocator, 2005-2024
audit/r5_gate_long_vs_equal_weight_quarterly.json
audit/r5_holdout_pit_vs_equal_weight.json         # allocator, 2025-2026
audit/r5_holdout_pit_vs_equal_weight_frozen.json
```

Paired daily log-return difference, moving-block bootstrap, block 21, 10,000
resamples, after tax and every Zerodha charge.

---

## Verdict: R5 NOT PASSED

The criterion is a paired interval excluding zero against the best R2 baseline.
Three arms clear it on 2005–2024. **None clears it on the 2025–26 holdout, and
the four candidates rank in exactly the reverse order out of sample.**

---

## 1. The signal is real. This part is settled.

`data/signal/r4_pit_long/gate.json`, `window-level-t/v2`, PASS:

| span | windows | 5d mean IC | t | positive | 20d t |
|---|---|---|---|---|---|
| 2016–2024 | 8 | +0.0225 | 4.55 | 7/8 | 3.19 |
| **2005–2024** | **13** | **+0.0240** | **6.43** | **12/13** | **4.19** |

Adding five windows *raised* t from 4.55 to 6.43 while leaving the IC
essentially unchanged (+0.0225 → +0.0240). That is what a real effect measured
over more independent periods does; a spurious one dilutes. Worst window of
thirteen is −0.0028.

## 2. The no-trade band is the finding. It replicates eight times out of eight.

Excess vs equal-weight, `rnone`, by band — every column monotone in the band:

| | K20 monthly | K20 quarterly | K30 monthly | K30 quarterly |
|---|---|---|---|---|
| **2005–2024** | | | | |
| band 0.000 | +0.0036 | +0.0256 | +0.0078 | +0.0242 |
| band 0.005 | +0.0164 | +0.0260 | +0.0204 | +0.0310 |
| band 0.010 | **+0.0302** | **+0.0456** | **+0.0385** | **+0.0446** |
| **2025–26 holdout** | | | | |
| band 0.000 | −0.0029 | −0.0115 | −0.0034 | −0.0118 |
| band 0.005 | +0.0035 | −0.0087 | −0.0052 | −0.0042 |
| band 0.010 | **+0.0167** | **−0.0004** | **+0.0078** | **+0.0050** |

Band 0.010 is the best of the three in **all eight** K × cadence × span
combinations, and the only band that is positive on the holdout anywhere. The
mechanism was stated in advance and measured: scrips sold per rebalance fall
77.8 → 8.1, and the tax drag falls from 1.9pp to 0.6pp — *below* equal-weight's
own 0.7pp (`S2` §6). This is the one claim here that survives every test put to
it.

Note also that extending to 2005 moved the band-0 base case from **−0.0139**
(8 windows) to **+0.0036**: the allocator's raw loss to equal-weight was itself
a feature of the shorter span.

## 3. What does not survive: the choice of K and cadence

Three arms clear against both baselines on 2005–2024. Here they are with the
fourth candidate, on every comparison run:

| arm | 2005–24 vs EW-m | vs EW-q | holdout vs EW | vs EW-frozen |
|---|---|---|---|---|
| K20 band .01 **quarterly** | **+0.0456** t 2.83 ✓ | +0.0410 t 2.57 ✓ | **−0.0004** t −0.01 | +0.0130 t 0.24 |
| K30 band .01 **quarterly** | +0.0446 t 2.74 ✓ | +0.0400 t 2.47 ✓ | +0.0050 t 0.09 | +0.0184 t 0.35 |
| K30 band .01 **monthly** | +0.0385 t 2.68 ✓ | +0.0339 t 2.35 ✓ | +0.0078 t 0.18 | +0.0212 t 0.52 |
| K20 band .01 **monthly** | +0.0302 t 2.26 ✓ | +0.0255 t 1.91 ✗ | **+0.0167** t 0.40 | **+0.0301** t 0.75 |

**Read the two ends of that table.** The in-sample ranking is
`K20q > K30q > K30m > K20m`. The holdout ranking, against *both* baselines
independently, is exactly `K20m > K30m > K30q > K20q` — a perfect reversal. The
best arm in sample is the worst out of sample; the worst is the best.

A perfect reversal of four items happens by chance about 1 time in 24, and all
four holdout differences sit well inside their own intervals, so this is
suggestive rather than demonstrated. But it points the same way as the
arithmetic: **the in-sample ranking of these arms carries no usable
information**, and selecting the top of it is not a neutral procedure.

Quarterly specifically was the change that produced the first passing gate
(`S2` §7). It is the arm that vanishes: **−0.0004 against equal-weight on the
holdout**, indistinguishable from zero, and negative at every band. It should
not be treated as validated.

## 4. Nothing clears the holdout, and that is mostly power

**0 of 57 arms** against equal-weight, 0 of 57 against equal-weight-frozen. 295
sessions of a market where equal-weight returned 1.8% gives intervals of ±6 to
±25pp — wide enough to miss an effect several times the size of anything
claimed here. The holdout can confirm a *sign* and a *ranking*; it cannot
confirm a magnitude, and its failure to clear is not evidence against.

What it *can* do, it did: it tested the band ordering (which held, 4 of 4) and
the cadence choice (which reversed).

One more thing it says plainly: the highest point estimate on the holdout is
`momentum_topk_monthly` at **+0.0449** — a plain baseline, ahead of every
allocator arm. At t 0.31 that is noise too, but it is a reminder of which bar
R5's criterion actually names.

## 5. The multiple-comparison arithmetic

The 2005–24 pass is 4 of 49 arms against one baseline, 3 of 49 against the
other. At 95% you expect ~2.5 false passes from 49 tries. A Bonferroni
correction at α/49 needs t ≈ 3.4; the best arm reaches **2.83**.

What argues against pure noise is structure — the arms that clear are the three
highest point estimates, share one configuration family (band 0.010, no stop),
and clear against both baselines. Structure is an argument, not a test.

## 6. Two things the failure is *not*

**Not the backtest.** `scripts/broker_parity.py`: identical target weights
through the env and the paper broker agree to 0.15% in sample and 0.27% on the
holdout, against a 5% tolerance. PASS both.

**Not cost.** The edge is +0.0374 with tax and +0.0370 without (`S2` §6). There
is no tax overhang left to recover.

**But every risk-controlled arm still fails.** All stopped and vol-stopped
variants miss (t 1.28–1.87), so the three passing arms carry a **−0.49 maximum
drawdown**. A book that halves is a real objection independent of any statistic.

## 7. Confidence, claim by claim

| claim | confidence | on what |
|---|---|---|
| The signal has predictive content | **high** | 13 windows, t 6.43, 12/13 positive |
| The 0.010 band is the right band | **high** | best in 8 of 8 span × K × cadence cells |
| The edge is ~+3 to +4.5%/yr before selection adjustment | moderate | 3 arms clear, but 49 tried |
| Quarterly beats monthly | **low — contradicted** | reverses on the holdout, −0.0004 |
| Any specific K is right | **low** | ranking reverses out of sample |
| Tradeable today | **not supported** | R5's gate is unmet |

## 8. Status and what would change it

**R5: NOT PASSED.** Recorded plainly per `CLAUDE.md`: *"A failed gate is a
legitimate and often good outcome. Never soften a failure, never partially
pass, never 'pass with caveats.'"* Phase 6 stays blocked under rule 1.

Not another sweep. At 49 arms the search is already the dominant source of
uncertainty, and §3 is direct evidence that ranking arms on this span selects
against out-of-sample performance. A fiftieth arm makes it worse.

The one thing that generates independent evidence is **forward paper trading**
of a single pre-committed configuration — no re-selection, no re-tuning, a
review date fixed in advance. Data that has not happened yet cannot be
overfitted, which is the property all five views above lack.

**If that is done, commit to K=30, band 0.010, no stop, monthly.** Not because
it is the best arm — it is third of four in sample — but because it is the only
one of the four that clears both baselines on 2005–2024 *and* stays positive
against both baselines on the holdout. Given a ranking that reverses, the arm
that is never bad is worth more than the arm that was once best. Execution
through the local paper broker and the Postgres ledger; no order-placing Kite
endpoint at any point.
