# Do NSE quarterly fundamentals carry signal?

Steps 1–4 of the fundamentals plan (`13_fundamentals_and_news.md` §3b). Every
number below regenerates from two committed scripts:

```
uv run python scripts/fundamental_ic.py             # standalone IC, stability, availability
uv run python scripts/fundamental_orthogonality.py  # is it already inside R4?
```

Data: `data/ext/fundamentals.parquet` — 9,370 company-quarters, 491 tickers,
2017-03-31 … 2024-12-31, parsed from NSE's own XBRL filings by
`scripts/fetch_xbrl_figures.py`. No vendor, no paid API.

> The tables below were measured on that snapshot, which was missing every
> bank and every ampersand symbol. Both holes were fixed the same day; the
> dataset is now 9,533 company-quarters over 500 tickers at 100% field
> coverage, and re-running on it moves the IC lift by 0.0002 and the verdict
> not at all. See `audit/F3_FUNDAMENTALS_VERDICT.md` §4b.

---

## Verdict

**The earnings *change* features carry real, stable signal that R4 does not
already contain. The *level* features carry nothing.** A 50/50 rank blend of R4
and a three-feature fundamental score lifts 5-day IC from **+0.0245 to +0.0320**
and 20-day from **+0.0417 to +0.0545** — about +31% at both horizons — on the
same days and the same names, all five scorable windows positive.

That lift is not yet money. It is an IC measured on the 20.4% of the panel that
has a filing, and it has not been through the allocator, costs, or tax. The next
step is the only one that settles it.

---

## 1. Levels are a stock fixed effect, exactly as predicted

`13_fundamentals_and_news.md` argued that a fundamental *level* barely moves
inside a 20-day horizon, so a model can only use it to memorise which tickers
did well. The measurement agrees, and the way it agrees is worth seeing.

| feature | 5d IC | t | windows positive |
|---|---|---|---|
| `f_net_margin` | +0.0094 | 0.73 | 2/6 |
| `f_pbt_margin` | +0.0093 | 0.74 | 2/6 |
| `f_employee_cost_ratio` | −0.0010 | −0.20 | 3/6 |
| `f_finance_cost_ratio` | −0.0102 | −1.17 | 3/6 |
| `f_tax_rate` | +0.0000 | 0.01 | 3/6 |
| `f_earnings_yield` | +0.0103 | 1.18 | 3/6 |

Not one is significant, and the per-window trace shows why — margin is positive
in W3–W4 then flips and *stays* negative through W5–W8:

```
f_net_margin  20d   W3:+0.066 W4:+0.046 W5:-0.074 W6:-0.030 W7:-0.044 W8:-0.014
```

A sign that flips once and holds is a regime, not a factor. Pooled over all
windows this feature reads +0.0322 with t = 0.72, which is how a dead feature
looks when the mean is quoted without the trace.

## 2. Changes carry signal, in every window

| feature | 5d IC | t | 20d IC | t | positive |
|---|---|---|---|---|---|
| `f_eps_growth_yoy` | **+0.0282** | 3.53 | **+0.0634** | 2.11 | 6/6 |
| `f_profit_growth_yoy` | **+0.0271** | 3.38 | **+0.0623** | 2.08 | 6/6 |
| `f_net_margin_change_yoy` | **+0.0274** | 2.78 | **+0.0559** | 2.37 | 6/6 |
| `f_profit_surprise` | +0.0061 | 1.63 | +0.0150 | 2.55 | 5/6 |
| `f_revenue_growth_yoy` | +0.0033 | 0.25 | +0.0280 | 1.52 | 4/6 |
| `f_revenue_surprise` | +0.0063 | 1.14 | +0.0086 | 0.93 | 3/6 |

Two things stand out.

**The signal is in the bottom line, not the top.** Profit and EPS growth are
significant at both horizons; revenue growth is not significant at either. A
company that grew revenue is not predictive; a company that grew *earnings* is.

**The effect decays.** `f_profit_growth_yoy` at 5d runs
`W3:+0.064 → W8:+0.013`, monotonically. The mean of +0.027 is not the number to
plan with; W8 (2023-24, 365 names, the most recent and widest cross-section) at
+0.013 is closer to what a live system would see.

## 3. It is not coverage growth

Coverage ramps hard — 20 tickers filing in 2017 against 410 in 2023 — so
"having filed at all" is correlated with the calendar and could manufacture an
IC by itself. Scored as a feature, it gives 20d IC +0.0212 at t = 1.89, 4/6
positive: not significant.

It also *cannot* contaminate the features above, for a structural reason.
`daily_scores` computes Spearman over `np.isfinite(pred)` only
(`src/trader/training/supervised.py:440`), so each feature is ranked purely
among names that have a filing. Availability is constant inside every one of
those cross-sections.

## 4. The two thin windows had to go

W1 and W2 have a **median of 0** names with a filing. They still produced ICs,
from the handful of days that scraped past `min_cross_section=10`, and those ICs
were large and dominated every pooled mean:

| feature | pooled over all 8 windows | over the 6 rankable windows |
|---|---|---|
| `f_net_margin` 5d | +0.0304 | **+0.0094** |
| `f_net_margin` 20d | +0.0322 | **−0.0084** |
| `f_profit_surprise` 5d | +0.0457 | **+0.0061** |

`scripts/fundamental_ic.py` now drops any window whose median cross-section is
under `--min-names` (default 50) and prints what it dropped.

## 5. Is it already inside R4? No.

The real risk: a company whose profit jumped usually had a price jump too, and
R4 is trained on price. If the fundamental score were a noisy restatement of
momentum, adding it would cost 4× coverage for nothing.

- **Cross-sectional corr(fundamental score, R4) = +0.0224** (sd 0.080) over
  1,269 days. Near-independent.
- **Residualising the fundamental score on R4 costs almost nothing:** 5d
  +0.0204 → +0.0200; 20d +0.0344 → +0.0340. Essentially all of it is
  orthogonal.

On the same days and the same names:

| signal | 5d IC | t | 20d IC | t |
|---|---|---|---|---|
| R4 alone (restricted to names with a filing) | +0.0245 | 6.70 | +0.0417 | 3.63 |
| Fundamental score alone | +0.0204 | 6.10 | +0.0344 | 7.20 |
| Fundamental residualised on R4 | +0.0200 | 5.70 | +0.0340 | 6.71 |
| **50/50 rank blend** | **+0.0320** | **10.18** | **+0.0545** | **5.40** |

All 5/5 windows positive. The blend lift matches what independence predicts
arithmetically — two uncorrelated signals of strength 0.0245 and 0.0204 blend to
≈ (0.0245+0.0204)/2 × √2 = 0.0317, against +0.0320 observed. The result is what
the arithmetic says it should be, which is a check that it is not an artefact.

In W8, the most recent window, the fundamental score carries **more** than R4 at
20d (+0.021 vs +0.003).

---

## What I predicted, and where I was wrong

Recorded before running, so it could be judged:

> "the change component lifts slightly, the level component gives none and
> mildly overfits, and the quality filter is worth more than either as a
> ranking feature."

- Change component lifts — **right**, and by more than "slightly" (t up to 3.5).
- Levels give none and overfit — **right**, and the mechanism was the predicted
  one.
- Quality filter worth more than either — **wrong.** A quality filter would be
  built from margins and returns, i.e. from levels, and levels carry nothing
  here. That idea is dead unless a balance sheet arrives.

## Limits of this data, stated plainly

- **P&L only.** NSE quarterly filings carry no balance sheet, so ROE,
  book-to-price and debt-to-equity are not buildable. This is a property of the
  source, not a gap in the parser.
- **625 of 10,036 documents failed to parse**, almost all banking filings, which
  use a different taxonomy declaring no undimensioned reporting period. Banks
  are therefore largely absent from the fundamental score.
- **20.4% mean grid coverage**, rising to 365 of 504 names by W8. Any use of
  this score must fall back to R4 alone where a filing does not exist.
- **W3–W8 only.** Six rankable windows, 2018-07 … 2024-06.

## A bug this found

`f_profit_growth_yoy` was computed as `x / |base| − 1`. For the 8.0% of rows
with negative net profit that is wrong in a way that inverts the ranking: a
company whose loss shrank from ₹50cr to ₹10cr reads **−1.2** and ranks among the
worst names in the cross-section, and a −50 → +50 turnaround reads exactly
**0.0**, indistinguishable from no change. Corrected to `(x − base) / |base|`,
which is identical for a positive base and monotone through zero. Locked by
`tests/unit/test_fundamental_features.py::test_growth_is_monotone_through_zero`.

## Next — ANSWERED, and the answer is no

> The IC lift is measured; the money is not. Build the blended signal and run it
> through the allocator grid with tax and the standing risk arms. That is the
> test that decides whether any of this ships.

It was built and run. **See `audit/F3_FUNDAMENTALS_VERDICT.md`.** The IC lift is
real and reaches nothing: top-20 forward return moves −0.00039 (t −0.22), every
one of eight allocator arms is flat or worse after tax, and a top-N filter fails
at four settings. The lift lands in the middle of the ranking, which a long-only
top-20 book never sees.

Two figures on this page should be read with that document beside them:

* **"20.4% mean grid coverage"** averages over 2016–2018, when coverage was
  zero. Within a tradeable window from W5 on it is 75–80%, and 14–15 of R4's top
  20 carry a filing. Coverage was never the limitation.
* the closing inference that an IC lift would translate into return was wrong.
  Pooled rank IC is the wrong yardstick for a long-only top-K book;
  `scripts/topk_diagnostic.py` is the right one.
