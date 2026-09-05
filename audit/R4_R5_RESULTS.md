# R4 signal + R5 allocator — first results on the rebuilt panel

Run 2026-09-05 on the RTX 4060 box. Logs in `audit/r4_v1/`. MLflow experiments
`signal` (8 runs) and `allocator` (21 runs) on the box, port 5555.

Panel: rebuilt at 645 tickers from `data/kite_ohlcv`, 3-month purge, all 15
features live. Model width is `active_tickers()` = 504.

## R4 — verdict FAIL

Walk-forward span bounded to 2010-01-01..2024-12-31, 8 windows, OOS
2016-07-01..2024-06-30. The 2025+ holdout was not touched.

| Horizon | Pooled IC | 95% CI | ICIR | Hit | n_days | n_eff | ACF(1) |
|---|---|---|---|---|---|---|---|
| 5d | +0.0415 | +0.0329, +0.0501 | 0.442 | 0.689 | 1468 | 207 | 0.752 |
| 20d | +0.0456 | +0.0297, +0.0626 | 0.509 | 0.718 | 1348 | 69 | 0.903 |

Per-window 5d IC: W1 +0.057, W2 +0.060, W3 +0.020, W4 +0.037, W5 +0.036,
W6 +0.057, W7 +0.042, W8 +0.023. All 8 positive; 7 of 8 clear individually.

**The gate FAILED**, and the reason is its every-window rule, not an absence of
signal. W3 (test 2018-07..2019-06) misses 5d by 0.0002 of mean IC and its CI
includes zero; W3/W5/W8 miss at 20d. `gate.json` records all seven reasons.

Two standing caveats apply to the CI column and are not fixed:
- The interval is **not a 5% test**. Measured false-positive rate under a strict
  null is ~11.6% at 5d even with the moving-block bootstrap, because at ACF 0.75
  a 250-day window carries ~207 effective observations, not 1468.
- **512 of 1980 OOS days (26%) are unscored**, matching the known warm-up-prefix
  gap: 59 days per window at lookback 60, plus forward-window truncation. This
  is lost power, not contamination.

## R5 — the allocator beats equal-weight for the first time

`+require_gate_pass=false`, so every MLflow run carries
`signal_gate_verdict=FAIL`. **These numbers do not constitute a passed gate.**
Backtest span is the R4 OOS slice (`oos.parquet`, 1921 dates, signal populates
56.8% of the grid).

Monthly cadence, the one `10_architecture_revamp.md` §4 recommends:

| Strategy | K | Horizon | Sharpe | CAGR | MaxDD | Turnover | vs EW |
|---|---|---|---|---|---|---|---|
| equal_weight | - | - | 1.369 | 0.272 | -0.532 | 1.13 | — |
| null_signal (control) | 30 | - | 1.271 | 0.152 | -0.428 | 3.90 | -0.121 |
| allocator | 30 | 20d | **2.019** | 0.338 | -0.463 | 3.78 | **+0.066** |
| allocator | 20 | 20d | 2.005 | 0.361 | -0.495 | 3.80 | +0.089 |

The **null-signal control lands below equal-weight** (-0.121 CAGR) while the real
signal lands above it. The gain therefore does not come from concentration or
from the allocator machinery; a random signal pushed through the identical path
underperforms. That control is what makes this result worth taking seriously.

Cadence matters roughly as predicted: monthly Sharpe 2.02 against weekly 1.64
and daily 1.51 at K=30/20d, with turnover 3.8 / 13.4 / 57.7 per year.

## Open, and load-bearing

1. **The daily arm is not believed.** Equal-weight at daily cadence returns
   CAGR -0.003 / Sharpe -0.019 against monthly +0.272 / 1.369, on turnover of
   only 4.74x per year. At ~0.3% round-trip that is ~1.4%/yr of cost and cannot
   explain a 27-point gap. Something in the daily path is wrong. It does not
   change the monthly ranking but it is not yet ruled out as also affecting it.
2. **Survivorship inflates every absolute number here.** The universe is today's
   504 active names, not point-in-time; `market.universe_snapshots` is still
   empty. All arms share the bias, so the *comparison* is far more trustworthy
   than any CAGR in this file.
3. R6 is unlaunched. `train_allocator_rl.py` refuses a non-PASS gate by design.
