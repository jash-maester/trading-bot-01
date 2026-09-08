# The fundamental blend lifts IC by 18% and earns nothing

Steps 5–7 of the fundamentals plan, and the end of that line of work.
`audit/F2_FUNDAMENTAL_IC.md` established that a three-feature earnings-change
score is near-independent of R4 and lifts pooled 20-day rank IC from +0.0439 to
+0.0517. It closed by saying the lift "is not yet money" and that the allocator
run would settle it.

It settled it. **The lift is real, it is stable, it is not selection bias, and
none of it reaches the top 20 names a long-only book actually buys.**

Regenerate with:

```
uv run python scripts/blend_signal.py --signal-tag r4_v2 --out-tag r4_v2_fund
uv run python scripts/topk_diagnostic.py --k-grid 20,30
RUN_TAG=fund bash scripts/fundamentals_chain.sh          # on the training box
```

---

## 1. The money test

Identical allocator grid, after tax and every Zerodha charge, on
`oos_r4_v2` (2016-09 … 2024-06). The two tables differ in exactly one thing:
the signal driving them.

| arm | K | r4_v2 CAGR | r4_v2_fund CAGR | Δ | r4_v2 Sharpe | fund Sharpe |
|---|---|---|---|---|---|---|
| none | 20 | 0.416 | 0.412 | **−0.004** | 1.660 | 1.669 |
| stop10 | 20 | 0.365 | 0.362 | **−0.003** | 1.910 | 1.880 |
| stop15 | 20 | 0.394 | 0.388 | **−0.006** | 1.894 | 1.871 |
| volstop | 20 | 0.381 | 0.375 | **−0.006** | 1.928 | 1.906 |
| none | 30 | 0.397 | 0.384 | **−0.013** | 1.680 | 1.664 |
| stop10 | 30 | 0.358 | 0.331 | **−0.027** | 1.947 | 1.866 |
| stop15 | 30 | 0.372 | 0.363 | **−0.009** | 1.884 | 1.854 |
| volstop | 30 | 0.368 | 0.347 | **−0.021** | 1.968 | 1.916 |

Eight arms, eight of them worse. The unselected `r4_v2_fundall` control behaves
the same way. Drawdown improves marginally on the unstopped arms (−0.589 →
−0.570 at K=20) and the demat bill falls slightly with turnover, but nothing
here is a gain worth the coverage.

## 2. Why a bigger IC bought nothing

Rank IC scores agreement over all 504 names. A long-only allocator taking K=20
sees only the extreme top of that ranking and is completely blind to how the
other 484 are ordered. A signal can therefore become materially better at
ordering the middle, lift IC, and leave the top 20 essentially unchanged.

`scripts/topk_diagnostic.py` measures the thing the allocator actually buys —
the plain mean 20-day forward return of the top K names, no costs, no tax, no
rebalancing rules between the ranking and the outcome:

| K | r4_v2 | r4_v2_fund | Δ | t | windows up |
|---|---|---|---|---|---|
| 20 | +0.02933 | +0.02895 | −0.00039 | −0.22 | 3/8 |
| 30 | +0.02774 | +0.02669 | −0.00105 | −0.88 | 3/8 |
| 100 | +0.02174 | +0.02129 | −0.00045 | −0.91 | 2/8 |

Nothing, at any K. W1 and W2 come out at **exactly** +0.00000, which is the
check that the machinery is sound: those windows have no fundamental coverage,
so the blend must be bit-identical there, and it is.

The decile spread tells the same story from the other side. At 20d it improves
+13% (0.01631 → 0.01852) — but that is top-50 minus bottom-50, and a long-only
book never touches the bottom. Most of the measured improvement lives in the
half of the spread we cannot trade.

## 3. The filter shape fails too

Re-ranking the whole covered subset spreads the information over ~400 names,
most never bought. A *filter* concentrates it exactly where the allocator
looks: take R4's top N, keep the K with the best fundamentals.

| shape | mean Δ per 20d | t | windows up | names swapped |
|---|---|---|---|---|
| top 40 → best 20 | −0.00289 | −0.69 | 3/5 | 10.1 of 20 |
| top 60 → best 20 | +0.00153 | +0.29 | 4/6 | 13.2 of 20 |
| top 60 → best 30 | −0.00261 | −0.74 | 2/5 | 14.8 of 30 |
| top 90 → best 30 | −0.00012 | −0.03 | 4/6 | 20.1 of 30 |

Not one is significant, and the deltas straddle zero. The filter is swapping
roughly **half the book** and changing the outcome by nothing.

## 4. This is not a coverage problem

The obvious objection is that fundamentals only cover part of the panel, so the
information never reaches the names being bought. Measured, it does:

| window | panel covered | of R4's top 20 |
|---|---|---|
| W3 | 4.4% | 0.7 / 20 |
| W4 | 58.0% | 11.3 / 20 |
| W5 | 74.7% | 15.2 / 20 |
| W6 | 75.3% | 15.5 / 20 |
| W7 | 76.3% | 14.1 / 20 |
| W8 | **80.1%** | 14.6 / 20 |

**This corrects a figure in `F2`.** That document quotes "20.4% mean grid
coverage", which averages over 2016–2018 when coverage was zero and is
misleading as a statement about what the allocator sees. Within a tradeable
window from W5 on, coverage is 74–78% of the panel and three-quarters of R4's
top 20 carry a filing. The blend had every opportunity to act on the names
being bought.

## 4b. Re-measured on complete data, after both coverage bugs were fixed

Everything above was first measured on a dataset that was missing **every
bank** — 233 rows across all 33 bank tickers parsed into pure nulls — and six
symbols whose names contain an ampersand. Banks are among the heaviest index
weights, so the verdict was re-run rather than assumed to hold.

`scripts/refetch_fundamentals.sh`, 5.4 minutes (175 documents fetched, 9,338
re-parsed from cache). Field coverage went from 97.3% to **100.0%**, tickers
from 446 to 453 in-universe (500 overall), and 351 rows that carried no figures
now carry them. Eight ampersand tickers came back, not the six the missing-list
suggested: `ARE&M`, `GET&D`, `GMRP&UI`, `GVT&D`, `J&KBANK`, `L&TFH`, `M&M`,
`M&MFIN`.

The result does not move.

| measure | missing banks | complete data |
|---|---|---|
| pooled 5d IC lift | +0.0020 | +0.0020 |
| pooled 20d IC lift | +0.0078 | +0.0076 |
| top-20 forward return Δ | −0.00039 (t −0.22) | **−0.00031 (t −0.17)** |
| top-30 forward return Δ | −0.00105 (t −0.88) | **−0.00106 (t −0.88)** |
| windows up, K=20 | 3/8 | 3/8 |

So the negative result is not an artefact of the missing banks, and the
allocator tables in §1 stand as measured.

## 5. What still stands from F2

None of this retracts the IC result, which was measured correctly:

- earnings **change** carries rank IC, levels carry none;
- the change score is near-independent of R4 (cross-sectional corr +0.022) and
  survives residualising on it almost intact;
- the lift is not feature selection — the unselected all-change control keeps
  ~80% of it;
- the per-window trace respects the coverage boundary exactly.

What was wrong was the inference, in F2's own closing line, that an IC lift
would translate. **Pooled rank IC is the wrong yardstick for a long-only top-K
book, and this is the cost of having used it.** A signal change should be
judged on top-K forward return from the start; `scripts/topk_diagnostic.py`
exists so the next one is.

## 6. Predictions, scored

Recorded before running, in F2:

> "the change component lifts slightly, the level component gives none and
> mildly overfits, and the quality filter is worth more than either as a
> ranking feature."

Change lifts — right. Levels give none — right. Quality filter worth more —
wrong; it would be built from levels, which carry nothing.

Recorded before the filter test in this document:

> "the filter also fails, because the diagnostic already shows the fundamental
> score adds nothing in the top tail at any K."

Right, at all four settings.

## 7. Verdict and what happens to the work

**The fundamental blend does not ship.** `r4_v2` remains the signal.

The artefacts stay, because they are cheap to keep and the data is now correct:

- `data/ext/fundamentals.parquet` — 9,533 company-quarters from NSE's own XBRL
  across 500 tickers at 100% field coverage, no vendor, no paid API;
- `src/trader/data/fundamental_features.py` and its 17 tests;
- `scripts/blend_signal.py`, `fundamental_ic.py`, `fundamental_orthogonality.py`,
  `topk_diagnostic.py`, `fundamentals_chain.sh`.

Two coverage bugs found along the way are worth keeping regardless of this
verdict, because both were silent and both class of error will recur: an
unencoded `&` made six symbols fetch as empty bodies, and the banking taxonomy
made 233 rows across all 33 bank tickers parse into pure nulls. See the commit
`Fix two silent coverage holes in the NSE fundamentals`.

**What would change the verdict.** Not a different blend weight or a different
filter — those are the same information reshaped. It would take fundamentals
that say something about the top tail specifically, which realistically means
data this source does not carry: a balance sheet (ROE, leverage, accruals),
analyst estimates to measure a true surprise against, or the guidance and
management commentary in the filing text. The plan's news arm is the honest
next candidate, and it should be judged on top-K return, not IC.
