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

**Regenerated 2026-09-06** after two execution defects were fixed (P1 in
`11_cost_defect_and_fix_plan.md`, plus the free-shares bug it exposed). The grid
first published here is superseded and must not be compared against this one:
they are different execution models. Logs: `audit/r4_v1/r4_v1_p2_allocator.log`.

`+require_gate_pass=false`, so every MLflow run carries
`signal_gate_verdict=FAIL`. **These numbers do not constitute a passed gate.**
Backtest span is the R4 OOS slice (1921 dates, signal populates 56.8% of grid),
₹10 lakh initial capital.

| Cadence | Strategy | K | Hor | Sharpe | CAGR | MaxDD | Turn | vs EW |
|---|---|---|---|---|---|---|---|---|
| monthly | equal_weight | - | - | 1.437 | 0.285 | -0.509 | 0.94 | — |
| monthly | null_signal | 30 | - | 1.325 | 0.184 | -0.461 | 3.73 | -0.102 |
| monthly | **allocator** | 30 | 20d | **2.027** | 0.359 | -0.464 | 3.71 | **+0.074** |
| monthly | allocator | 20 | 20d | 2.020 | 0.382 | -0.490 | 3.73 | +0.097 |
| weekly | equal_weight | - | - | 1.374 | 0.274 | -0.522 | 1.43 | — |
| weekly | allocator | 20 | 20d | 1.726 | 0.386 | -0.546 | 13.34 | +0.112 |
| daily | equal_weight | - | - | 1.337 | 0.266 | -0.524 | 2.41 | — |
| daily | allocator | 20 | 5d | 1.729 | 0.429 | -0.578 | 58.12 | +0.163 |

Three properties hold across the whole grid and each is a check that could have
failed:

1. **Every allocator configuration beats equal-weight** on CAGR, at all three
   cadences, all three K, both horizons — 18 of 18.
2. **The null-signal control lands below equal-weight** at every cadence
   (-0.102 monthly, -0.217 weekly, -0.414 daily). A random signal pushed through
   the identical allocator and the same candidate set underperforms, so the gain
   is not concentration and not the allocator machinery.
3. **Monthly dominates on risk-adjusted return** (Sharpe 2.027 against 1.726
   weekly and 1.729 daily) while trading a fifteenth as much, which is the
   cadence argument in `10_architecture_revamp.md` §4 holding up under cost.

The best configuration is monthly, K=30, 20-day horizon: Sharpe 2.027 against
equal-weight's 1.437, CAGR +7.4 points, on a shallower drawdown.

## Open, and load-bearing

1. **This rests on a FAILED signal gate.** R4 did not clear, and the interval
   behind it is not a 5% test. Nothing here changes that.
2. **Survivorship inflates every absolute number.** The universe is today's 504
   active names, not point-in-time; `market.universe_snapshots` is still empty.
   All arms share the bias, so the *comparison* is far more trustworthy than any
   CAGR in this file. Deferred as P4.
3. **Daily turnover of 58x a year is not investable** even where it wins on
   paper, and no control in the system constrains the count of names traded,
   which is what the flat demat fee bills for. Deferred as P3.
4. **Capital scaling is unaddressed.** These use ₹10 lakh; intended live capital
   is ₹1 lakh, where a 504-name book is uninvestable. Deferred as P5.
5. R6 is unlaunched. `train_allocator_rl.py` refuses a non-PASS gate by design.
