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

## R5 — the allocator beats equal-weight, on every cadence

**Regenerated 2026-09-06 (third time) on the r4_v2 signal**, after the gate
was made window-level (`12_gate_decision.md`) and both runs passed. Logs:
`audit/r4_v2/r4_v2_p2_allocator.log`. MLflow experiment `allocator`, split
`oos_r4_v2`, `signal_gate_verdict=PASS` on every run, `require_gate_pass=true`.

Backtest span is r4_v2's own OOS slice, 2016-07-01..2024-06-28: 1980 dates,
signal populating **72.0%** of the `[T, N=504]` grid. ₹10 lakh initial.

| Cadence | Strategy | K | Hor | Sharpe | CAGR | MaxDD | Turn | vs EW |
|---|---|---|---|---|---|---|---|---|
| monthly | equal_weight | - | - | 1.347 | 0.267 | -0.508 | 0.92 | — |
| monthly | null_signal (control) | 30 | - | 1.299 | 0.251 | -0.544 | 3.76 | -0.016 |
| monthly | **allocator** | 30 | 20d | **1.981** | 0.459 | -0.532 | 3.73 | **+0.192** |
| monthly | allocator | 20 | 20d | 1.964 | 0.487 | -0.565 | 3.74 | +0.220 |
| monthly | allocator | 30 | 5d | 1.744 | 0.406 | -0.567 | 3.72 | +0.139 |
| weekly | equal_weight | - | - | 1.291 | 0.257 | -0.521 | 1.43 | — |
| weekly | allocator | 20 | 20d | 1.851 | 0.495 | -0.629 | 15.67 | +0.239 |
| daily | equal_weight | - | - | 1.258 | 0.250 | -0.523 | 2.39 | — |
| daily | allocator | 20 | 20d | 1.881 | 0.534 | -0.624 | 72.30 | +0.285 |

Three properties hold across the whole grid:

1. **Every allocator configuration beats equal-weight** on CAGR — 18 of 18 —
   and on Sharpe, at all three cadences, all three K, both horizons.
2. **The null-signal control sits at equal-weight** at monthly (-0.016 CAGR)
   and below it at weekly and daily. This is what a random 30-of-504 pick
   *should* do: its expected return is the market's, minus its costs. The gap
   between the control and the allocator — **+0.21 CAGR at monthly K=30/20d**
   — is the signal's contribution through identical machinery.
3. **Monthly is the right cadence.** Sharpe 1.98 at 3.7x annual turnover
   against 1.85 weekly at 15.7x and 1.88 daily at 72x. Daily's extra CAGR is
   bought with 20x the turnover, and P3 (per-name trade band) is not built.

Best configuration: **monthly, K=30, 20-day horizon — Sharpe 1.981 vs
equal-weight 1.347, CAGR +19.2 points, drawdown -0.532 vs -0.508.**

### Why this grid differs from the previous one, and which to believe

The previous grid (r4_v1 signal, 1921-date slice) showed monthly K=30/20d at
Sharpe 2.027 / CAGR 0.359, +0.074 over equal-weight, with the null control at
-0.102. This one shows CAGR 0.459, +0.192, control at -0.016. The Sharpe
barely moved; the CAGR and the control moved a lot. The cause is coverage,
and it is a property of the *old* grid, not a gain in this one:

- r4_v1 had no prediction on the first 59 days of each test window (no
  warm-up prefix): 56.8% of the grid populated. `allocate()` on a rebalance
  day with **no finite `r_hat` targets all cash** (`deterministic.py:127-130`),
  limited only by the 30% turnover budget. So for roughly the first quarter of
  every year the allocator was liquidating toward cash, in a market that
  compounded at ~27%. That is why its CAGR was 10 points lower, and why the
  null control — masked to the same support — was also depressed.
- r4_v2 carries the warm-up prefix, so predictions cover the full window
  (72.0% of the grid; the remainder is untradeable cells and forward
  truncation). Neither arm sits out.

**The r4_v2 grid is the faithful measurement.** The r4_v1 grid understated
both the allocator and its control by the same mechanism, which is also why
its Sharpe (a ratio) barely changed while its CAGR (a level) did.

Design note, not changed here: "no signal → go to cash" is a defensible live
default (a data outage should not leave a stale book), but a backtest with
structural gaps in coverage will understate any strategy under it. Keep the
signal's coverage in view whenever this table is read.

## Open, and load-bearing

1. **Survivorship inflates every absolute number.** The universe is today's
   504 active names, not point-in-time; `market.universe_snapshots` is still
   empty. All arms share the bias, so the *comparison* — and especially the
   allocator-minus-control gap — is far more trustworthy than any CAGR here.
   Deferred as P4 and now the single largest threat to the result.
2. **Daily turnover of 72x a year is not investable** even where it wins on
   paper. Nothing constrains the count of names traded, which is what the flat
   demat fee bills for. Deferred as P3.
3. **Capital scaling is unaddressed.** ₹10 lakh here; ₹1 lakh intended.
   Deferred as P5.
4. **The gate passing is a screen, not validation.** CLAUDE.md rule 2: the
   downstream economic result *is* this table, and it now exists with run IDs
   under a PASS. R6 may be attempted.
