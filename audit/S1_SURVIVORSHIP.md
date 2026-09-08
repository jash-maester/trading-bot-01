# The universe was doing most of the work

`CLAUDE.md` lists survivorship under *Known-dangerous ground* and
`audit/P4_survivorship.md` called delisting an unrecoverable limit. Both were
right that it mattered and neither said how much. This measures it.

**Headline: on a point-in-time universe, momentum-top-20 goes from CAGR +0.273
to −0.002.** It loses all of its return. Every other baseline falls too, but
none by nearly as much — and momentum was the bar R5's gate had to clear.

Regenerate:

```
uv run python scripts/fetch_bhavcopy.py --from 2010-01-01     # 35 min
uv run python scripts/fetch_corporate_actions.py
RUN_TAG=pit bash scripts/pit_rebuild_chain.sh                 # on the box
```

---

## 1. What was rebuilt

| artefact | |
|---|---|
| bhavcopy rows | 8,161,339 over 4,138 sessions, 2010-01-04 … 2026-09-04 |
| securities seen | 6,593 (4,337 distinct EQ/BE symbols) |
| corporate actions | 1,118 splits/bonuses/consolidations, 675 applied in-store |
| store after prefilter | 1,504 tickers ever clearing ₹5cr median turnover |
| panel | 4,138 days × 1,504 tickers, 1.36M tradeable rows |
| ever-eligible names | 1,105 over 201 months |

Prices are back-adjusted from NSE's corporate-actions feed. Verified on three
splits whose ratios are public record — raw, each shows a crash; adjusted, an
ordinary day:

| ticker | ex-date | raw step | adjusted step |
|---|---|---|---|
| NESTLEIND | 2024-01-05 | −90.2% | **−2.2%** |
| HDFCBANK | 2019-09-19 | −49.7% | **−1.5%** |
| IRCTC | 2021-10-28 | −77.9% | **−13.0%** |

`scripts/bhavcopy_to_store.py` refuses to write unless all three come out
under 25%, because the previous adjustment layer failed silently and produced
a store that looked fine.

## 2. How much of the investable universe the old list held

Point-in-time eligibility: ≥₹5cr median daily turnover over a trailing year,
≥100 sessions, top 504 by turnover, decided from sessions **strictly before**
each date.

| date | eligible | in 645 panelled | in 504 traded | missing |
|---|---|---|---|---|
| 2011-01-01 | 249 | 48.2% | 40.6% | 129 |
| 2014-01-01 | 171 | 63.2% | 54.4% | 63 |
| 2016-01-01 | 252 | 60.7% | 50.4% | 99 |
| 2019-01-01 | 345 | 63.2% | 50.4% | 127 |
| 2022-01-01 | 504 | 68.8% | 52.0% | 157 |
| 2024-07-01 | 504 | 77.6% | 62.7% | 113 |
| **mean** | **346** | **66.2%** | **53.7%** | **109** |

Only the "in 645" column is survivorship. The gap to "in 504" is
`INACTIVE_SECTORS` capping the observation width, which is a compute decision.
An earlier version of this work quoted the 504 figure as if it were all
survivorship; it is not.

At 2019-01-01 the 127 missing names split **67 vanished / 60 omitted** — more
than half had stopped trading by 2026. Widening a ticker list drawn today
recovers the omitted and can never recover the vanished, which is why this
needed the full-market archive rather than a bigger ticker file.

> **A limit of that split.** "Vanished" means *stopped trading under this
> symbol*, which conflates a death with a rename: ALBK and ANDHRABANK were
> merged away, but AMARAJABAT → ARE&M and CADILAHC → ZYDUSLIFE are the same
> companies still trading. The bhavcopy carries ISIN and could disambiguate
> these; that has not been done, so the vanished count is an upper bound on
> true delisting.

## 3. What it was worth — the measurement that matters

Same span, same costs, same tax, same cadence. Only the universe differs.

| baseline | 2026-chosen 504 | point-in-time | change |
|---|---|---|---|
| equal_weight | 0.257 | 0.132 | −12.5pp |
| equal_weight_frozen | 0.256 | **0.137** | −11.9pp |
| **momentum_topk** | **0.273** | **−0.002** | **−27.5pp** |
| sixty_forty | 0.155 | 0.082 | −7.3pp |
| random | 0.188 | 0.060 | −12.8pp |

Sharpe moves the same way: momentum 0.896 → −0.008, equal-weight 1.280 → 0.681.

**Momentum-top-K was almost entirely survivorship.** That is the mechanism
working exactly as predicted: it buys whatever ran hardest, and a list drawn in
2026 guarantees those names survived to be drawn. A prediction recorded in
`scripts/pit_rebuild_chain.sh` before the run said momentum would fall furthest;
it fell more than twice as far as equal-weight.

Everything else fell too, which was also predicted and is the more sobering
half: **roughly 12 points of the ~0.26 CAGR that a simple long-only Indian
equity backtest produced on this universe was the universe.**

## 4. What this does and does not settle

**Settles.** `audit/R5_GATE.md` marked R5 FAIL because only 1 of 8 arms cleared
a paired interval against momentum's 0.273. That bar was an artefact. The
comparison has to be re-run against `equal_weight_frozen` at 0.137.

**Does not settle.** The allocator's +0.416 was measured on the same
contaminated universe and will fall too — by how much is unknown until R4 is
retrained and re-run. **No claim that R5 now passes is supported by anything on
this page.** The bar moved; where the strategy lands is Phase 3.

## 5. Residual limitations, stated

* **39.9% of ever-eligible names have no NSE industry.** The classification is a
  current constituent list and cannot label 14 years of delisted names. Those
  take the explicit unknown bucket (id 99), so the allocator's sector cap treats
  them as one group — a real constraint on Phase 3.
* **ISIN is unavailable before ~2011**: the classic bhavcopy added the column
  later, so rename resolution cannot reach the earliest years.
* **Dividends are not adjusted for**, matching Kite's convention deliberately.
* **625 banking XBRL documents** remain unparsed from the separate fundamentals
  line; unrelated to prices, noted so the two are not confused.
