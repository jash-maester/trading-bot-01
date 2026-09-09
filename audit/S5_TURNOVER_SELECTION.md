# The surviving rule cannot be implemented with the obvious selection

`audit/S4_NULL_CONTROL.md` established that R4's signal does not beat a random
one out of sample, while the *machinery* — 30 names, a 1% no-trade band, a
volatility stop — improved risk against equal-weight and survived on the
holdout. That result was produced by a **random** picker, which nobody runs.

This tested the obvious replacement: **top-30 by 20-day median turnover**.
Pre-committed before measuring, monthly, K=30, band 0.010, four risk overlays,
with the matched null control. **It is much worse than random, on both spans.**

```
uv run python scripts/make_turnover_signal.py \
    --panel data/panels_bhav/oos_r4_pit_long.parquet \
    --like data/signal/r4_pit_long --out-tag turnover_long
uv run python scripts/run_allocator.py data=bhav_v1 +split=oos_r4_pit_long \
    +signal_tag=turnover_long +apply_tax=true '++allocator.null_control=true' \
    '++allocator.k_grid=[30]' '++allocator.band_grid=[0.010]' \
    '++allocator.freq_grid=[monthly]' '++allocator.horizon_grid=[20d]'
```

---

## In sample, 2005–2024

Sharpe / CAGR / max drawdown, K=30, band 0.010, monthly, after tax:

| overlay | equal_weight | null (random) | **turnover** |
|---|---|---|---|
| — | 0.487 / +0.097 / −0.595 | — | — |
| none | | 0.684 / +0.123 / −0.508 | **0.487 / +0.093 / −0.479** |
| stop10 | | 0.841 / +0.129 / −0.385 | **0.486 / +0.083 / −0.407** |
| stop15 | | 0.805 / +0.128 / −0.421 | **0.465 / +0.081 / −0.414** |
| volstop | | 0.796 / +0.122 / −0.404 | **0.431 / +0.074 / −0.430** |

## On the unseen holdout

| overlay | equal_weight | null (random) | **turnover** |
|---|---|---|---|
| — | 0.151 / +0.018 / −0.137 | — | — |
| none | | 0.082 / +0.010 / −0.128 | **−0.338 / −0.045 / −0.135** |
| stop10 | | 0.119 / +0.010 / −0.075 | **−0.759 / −0.076 / −0.110** |
| stop15 | | 0.039 / +0.004 / −0.087 | **−0.492 / −0.055 / −0.117** |
| volstop | | 0.307 / +0.021 / −0.061 | **−0.731 / −0.064 / −0.099** |

**Random beats turnover in 8 of 8 matched arms**, and on the holdout the
turnover book loses 4.5–7.6% a year in absolute terms.

## Why, and what it costs the earlier finding

Top-30 by turnover is the mega-caps: the same names every month, highly
correlated with each other, and no exposure to the mid-caps that carried the
return. It still cuts drawdown against equal-weight (−0.479 against −0.595) —
but it buys that by giving up return, which is a bad trade and not the one the
random book was making.

The random book kept the return (+0.123 against equal-weight's +0.097) *and*
cut the drawdown. So the mechanism was never "concentration" as such. It was
**breadth across the whole eligible liquidity spectrum**, with the band and the
stop trimming cost and tail losses. Concentrating into the largest names
destroys exactly the part that was working.

This was the failure mode named in advance:

> "If the turnover book's drawdown advantage over equal-weight is much smaller
> than the random book's, that would mean the effect came from holding
> diversified-but-small positions rather than from concentration per se — and
> the rule as specified wouldn't be the right one to forward-test."

That is what happened. Per the same pre-commitment, no alternative selection
was tried. Sweeping selections until one works is the procedure that produced
four reversals in this project already.

## What is actually left

Re-reading S4's result with this in hand, it is weaker than it appeared:

| holdout, vs equal_weight | ΔCAGR | MDD |
|---|---|---|
| null/none | **−0.0080** | −0.128 vs −0.137 |
| null/volstop | +0.0024 | **−0.061** vs −0.137 |

Out of sample the random book's *return* advantage is already gone — it is
negative without the stop and a rounding error with it. **The only thing that
survives on unseen data is a drawdown reduction produced by a volatility
stop**, and a stop cutting tail losses is close to mechanical: that is what a
stop does, in any strategy, with or without a signal.

So the sequence across this project is:

1. the universe was survivorship (`S1`) — ~12 of ~26 CAGR points;
2. the allocator's edge over equal-weight was mostly machinery (`S4`) —
   ~+0.026 of ~+0.043;
3. the signal does not beat random out of sample (`S4`) — 13 of 15 pairs;
4. the machinery cannot be run with a sensible selection (`S5`) — 8 of 8 to
   random;
5. what remains out of sample is a stop reducing drawdown at no return gain.

Nothing in that chain is alpha.

## Status

**No forward test is warranted on this evidence.** `P1` was already withdrawn;
nothing here revives it. The honest recommendation is option **B** — treat the
research programme as concluded — with the caveat that this is a conclusion
about *these* features, *this* horizon and *this* universe, not a claim that
Indian equities are unpredictable.
