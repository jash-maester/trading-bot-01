# The top-K gate: R4's information is in the names it would avoid

R4 is gated on full-cross-section rank IC and passes (13 windows, 20d
+0.0280, t 4.19). The allocator buys the top 30. Scored on **that** — the mean
20-day forward return of the top 30, net of a 23 bps round-trip cost proxy,
against the eligible-universe mean, per walk-forward window, versus 20 random
books on the identical support — the signal is indistinguishable from random.
Scored on the **bottom** 30, it is strongly, consistently negative. The rank
IC was real; it lives in the tail a long-only book never touches.

```
uv run python scripts/topk_gate.py --signal r4_pit_long --panel data/panels_bhav/full.parquet \
    --k 30 --horizon 20 --n-null 20 --which top      # audit/topk_gate/r4_pit_long_top30.json
uv run python scripts/topk_gate.py ... --which bottom  # audit/topk_gate/r4_pit_long_bottom30.json
```

Windows are the artefact's own (`summary.json`), so this scores exactly the
spans the IC gate scored. Hygiene, not search: nothing was trained against it.

---

## Top 30 — what a long-only book buys

Mean per 20d, net of cost, excess over the eligible-universe mean:

| | signal | random ×20 mean | diff | windows > 0 | t (crit) | rank / 21 | verdict |
|---|---|---|---|---|---|---|---|
| `r4_pit_long`, 13 win | −0.00241 | −0.00212 | −0.00029 | 4/13 | −2.01 (2.18) | **12** | FAIL |
| `r4_pit`, 8 win | −0.00160 | −0.00221 | +0.00061 | 3/8 | −0.91 (2.37) | **7** | FAIL |

Dead centre of the null distribution, both artefacts. The random books' mean is
negative by almost exactly the cost proxy (turnover ≈ 90% × 23 bps ≈ −0.0021),
which is the sanity check that random selection has zero gross excess and the
cost model is doing what it says. The signal's gross excess is also ≈ 0.

Grinold's arithmetic predicts ~+0.45%/month gross from IC 0.03 at this
breadth. Measured: nothing. The theory assumes the IC is spread evenly over the
ranking. It is not.

## Bottom 30 — what it would short

| | signal | random ×20 mean | diff | windows > 0 | t (crit) | rank / 21 |
|---|---|---|---|---|---|---|
| `r4_pit_long`, 13 win | **−0.00851** | −0.00185 | **−0.00666** | 2/13 | **−3.20** (2.18) | **21** |
| `r4_pit`, 8 win | −0.00973 | −0.00241 | −0.00732 | 1/8 | −2.04 (2.37) | **21** |

The signal's bottom 30 underperforms the universe by **≈ −10%/yr**, negative
in 11 of 13 windows, and is worse than **every one of 20 random bottom-30s**
on both artefacts. On the 13-window artefact it clears the window-level t and
the paired t against random (−2.59); on the 8-window one it has the same sign,
the same rank, and misses only on the smaller-sample critical value.

## What this reconciles

| observation | source | now explained |
|---|---|---|
| Rank-IC gate passes at t 6.43 | `r4_pit_long/gate.json` | the IC is carried by the bottom tail |
| Higher-IC model gives the worse portfolio | `S4` | more of the IC in the tail, none more at the top |
| Signal loses to random on the holdout, 13 of 15 | `S4` | top-30 was never better than random in sample either |
| Fundamentals raised IC, not top-K return | `F3` | same shape: information landed where nothing is bought |
| Warm-up rank 17 of 21 | `P2` | a top-30 book of a bottom-tail signal |

**R4 is a screen, not a picker.** Everything this project built on top of it
assumed the opposite.

## What it does not establish

That a *screened* book — equal-weight the universe minus the signal's bottom
tail — beats equal-weight or a random screen out of sample. That is a new
hypothesis, derived from in-sample data, and it has one strike against it
already: `S2` §2's K sweep at band 0 found top-250 (i.e. excluding the bottom
~100) **worse** than equal-weight by −0.045. That sweep had no band, and the
band was the lever that suppressed boundary churn everywhere else — but it
means the screen is untested, not supported.

If it is tested, it is tested **once**, pre-stated, with the band, against a
random screen of the same size on both spans, and the exclusion fraction is
fixed on stated reasoning before the run — not swept. `S5` is what sweeping
selections until one works looks like.

## Status

Step 1 of the post-R5 plan is done: R4 now carries a gate that measures what
the allocator consumes, and on it R4 fails as a long-only picker while showing
significant short-leg information. `audit/topk_gate/*.json` hold the numbers;
`scripts/topk_gate.py` regenerates them. The forward test (`P2`) continues
unchanged — its book is the top-30 that this document says carries no
information, which is a prediction the forward data will now test.
