# The allocator on a point-in-time universe

Phases 3–5 of the survivorship rebuild, plus the K/band sweep that followed.
`audit/S1_SURVIVORSHIP.md` covers the universe itself; this is what the strategy
does once it stands on one.

```
RUN_TAG=p3 bash scripts/pit_phase3_chain.sh     # allocator + paired gate
KSET=20,30,50   RUN_TAG=swA bash scripts/pit_sweep.sh
KSET=100,150,250 RUN_TAG=swB bash scripts/pit_sweep.sh
RUN_TAG=p4 bash scripts/pit_phase4_chain.sh     # 2025-26 holdout
RUN_TAG=p5 bash scripts/pit_phase5_chain.sh     # broker parity
```

---

## Verdict

**R5 fails its gate, and the failure is smaller than it first looked.** As
configured (K=20, no band) every arm sits *below* equal-weight. With a 1%
no-trade band the sign flips and the edge lands near **+3.2%/yr on two
independent spans** — but neither clears a 95% interval, so it is undemonstrated
rather than disproved.

## 1. As configured, the allocator loses

`oos_r4_pit`, monthly, K=20, after tax:

| | CAGR | Sharpe | vs EW |
|---|---|---|---|
| equal_weight | 0.119 | 0.617 | — |
| allocator/none | 0.105 | 0.558 | −0.0139 |
| null_signal (random, same K) | 0.078 | 0.419 | −0.0409 |

The null control decomposes it exactly: concentrating into 20 of ~350 names and
trading 3.45× turnover **costs 4.1pp**; the signal **recovers 2.7pp** of that.
Net −1.4pp. The signal is worth something — consistent with R4's gate passing —
and it does not pay for the concentration.

On the fixed 504-name universe the same allocator beat equal-weight by
**+0.159**. That entire edge was the universe.

## 2. K was the wrong lever, and the sweep says so cleanly

My hypothesis was that top-20 of ~350 is too concentrated for an IC of +0.0225.
Wrong, and monotonically so:

| K (band 0) | 20 | 50 | 100 | 150 | 250 |
|---|---|---|---|---|---|
| vs EW | −0.014 | −0.015 | −0.016 | −0.028 | **−0.045** |

Diluting toward equal-weight makes it worse, not better. K=250 — a deliberately
near-degenerate arm included as a control — comes out strongly negative, which
is the check that the sweep measures something real rather than converging on
the baseline.

## 3. The no-trade band is the lever

K=20, `oos_r4_pit`:

| band | CAGR | Sharpe | MDD | scrips sold/reb | demat ₹ | vs EW |
|---|---|---|---|---|---|---|
| 0.000 | 0.105 | 0.558 | −0.542 | 77.8 | 113,409 | −0.0139 |
| 0.005 | 0.134 | 0.709 | −0.538 | 15.2 | 22,105 | +0.0148 |
| **0.010** | **0.157** | **0.841** | −0.494 | **8.1** | **11,827** | **+0.0374** |

A 90% cut in scrips sold. The volstop variant at band 0.010 reaches **Sharpe
1.075 on a −0.285 drawdown**, against equal-weight's 0.617 and −0.589.

Worth noting what this is *not*: the demat saving is ~₹101,600 on ₹1M over
7.9 years ≈ 1.3%/yr, while CAGR improves 5.2pp. Most of the gain is therefore
**not** the flat fee — it is the 0.1% STT on both legs, the tax churn from
realising gains monthly, and simply holding winners longer. Turnover was the
problem; the demat bill was only its most visible part.

## 4. Both spans agree on the size, neither can prove it

| span | best arm | excess/yr | 95% CI | t |
|---|---|---|---|---|
| in-sample (1,982 sessions) | K=20 band 0.010 | +0.0329 | [−0.0016, +0.0672] | **1.98** |
| holdout (295 sessions) | K=20 band 0.010 | +0.0324 | [−0.0266, +0.0906] | 0.81 |

Two independent spans, the same point estimate to within 0.0005. **0 of 19 and
0 of 24 arms clear.**

The holdout's failure is power, not evidence against: 295 sessions of a market
where equal-weight returned **1.8%** gives a ±9pp interval, which would miss an
effect twice this size. The in-sample miss at t 1.98 is a genuine near-thing,
and it is the best of 36 arms — so the point estimate owes something to
selection that the holdout replication does not.

**The band ordering replicates out of sample**, which is the strongest single
result here:

| band | in-sample vs EW | holdout vs EW |
|---|---|---|
| 0.000 | −0.0139 | −0.0063 |
| 0.005 | +0.0148 | −0.0005 |
| 0.010 | +0.0374 | +0.0198 |

## 5. The backtest is not the problem

`scripts/broker_parity.py`, identical target weights through the env and the
paper broker:

| span | max \|gap\| | drift/day | verdict |
|---|---|---|---|
| in-sample | 0.15% | +0.0006% | **PASS** (tol 5%) |
| holdout | 0.27% | +0.0003% | **PASS** |

This rules out the comfortable explanation. The numbers above describe a system
that could actually be run; the strategy simply does not clear its bar yet.

## 6. What would settle it

Not another sweep of the same space — that is how a t of 1.98 becomes a t of
2.1 by accident. The honest options, in order of what they would actually
resolve:

* **Isolate the tax drag.** `apply_tax=true` is on throughout, and STCG at 20%
  on monthly turnover is a large and completely uninvestigated term. If the
  edge is 3%/yr gross of tax and tax costs 2%, that is a different problem from
  a 3%/yr net edge.
* **Longer holding.** Monthly is the shortest cadence tested here. Quarterly
  would cut both turnover and the STCG/LTCG boundary at once.
* **More span, not more arms.** The in-sample window is 7.9 years and the
  bhavcopy reaches back to 2010; extending the walk-forward earlier adds
  independent windows, which is what a window-level t-test actually needs.
