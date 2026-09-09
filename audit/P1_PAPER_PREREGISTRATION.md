# Forward paper trading: pre-registration

**Status: REGISTERED, not yet started.** Nothing below may be changed once the
first rebalance executes. If any of it changes, the clock resets to zero and
this document is superseded by a new one with a new start date.

Written 2026-09-09, before any forward data exists.

---

## Why this exists

Every span this project has measured has been searched over. R5 tried 49 arms
on 2005–2024 and 3–4 cleared; the 2025–26 holdout cleared 0 of 57 and ranked
the four candidates in **exactly reverse order** (`audit/R5_VERDICT_PIT.md`).
The same reversal then appeared one layer down in the signal work
(`audit/S3_CROSS_SECTIONAL_INPUTS.md`): a factor scoring 2× the model in sample
lost to it out of sample.

Three reversals is a property of the setup, not bad luck — roughly 14 effective
independent periods against dozens of candidates. **Any procedure that selects
on measured performance is anti-predictive here.** Forward data is the only
kind that cannot be searched, which is the entire reason for this exercise.

The pre-commitment *is* the experiment. A configuration chosen after seeing the
data it is judged on is worth nothing, and that is precisely what a document
written afterwards would be.

## The configuration, frozen

| parameter | value | where |
|---|---|---|
| signal | `r4_pit_long`, 20d horizon | `data/signal/r4_pit_long` |
| top-K | **30** | `allocator.k` |
| no-trade band | **0.010** | `allocator.no_trade_band` |
| risk overlay | **none** | `allocator.risk_grid` name `none` |
| rebalance | **monthly** | `allocator.freq` |
| max name weight | 0.10 | `configs/allocator/default.yaml` |
| max sector weight | 0.25 | ” |
| universe | point-in-time `LiquidityRule` | ≥₹5 crore median turnover, 365d lookback, ≥100 sessions, ≤504 names |
| costs | Zerodha delivery, in full | STT 0.1% both legs, DP ₹15.34/scrip/sell-day, stamp 0.015% buy |
| tax | STCG 20% + 4% cess, LTCG 12.5% | 12-month boundary |
| capital | ₹1,000,000 notional | |
| execution | local paper broker → Postgres ledger | **no order-placing Kite endpoint, ever** |

### Why K=30, band 0.010, monthly, no stop

Not because it is the best arm — it is **third of four** in sample. It is the
only one of the four candidates that clears both baselines on 2005–2024 *and*
stays positive against both baselines on the 2025–26 holdout:

| arm | 2005–24 vs EW-m | vs EW-q | holdout vs EW | vs EW-frozen |
|---|---|---|---|---|
| K20 b.01 quarterly | +0.0456 ✓ | +0.0410 ✓ | −0.0004 | +0.0130 |
| K30 b.01 quarterly | +0.0446 ✓ | +0.0400 ✓ | +0.0050 | +0.0184 |
| **K30 b.01 monthly** | **+0.0385 ✓** | **+0.0339 ✓** | **+0.0078** | **+0.0212** |
| K20 b.01 monthly | +0.0302 ✓ | +0.0255 ✗ | +0.0167 | +0.0301 |

Given a ranking that reverses out of sample, the arm that is never bad is worth
more than the arm that was once best. The band at 0.010 is the one component
with independent support: best of three bands in **all eight** span × K ×
cadence cells, with a measured mechanism (scrips sold per rebalance 77.8 → 8.1;
tax drag 1.9pp → 0.6pp, below equal-weight's own 0.7pp).

`none` rather than a stop because **no stopped arm passed anything**, in sample
or out. The cost is accepted openly: the in-sample maximum drawdown of this
family is **−0.49**. See "abandonment" below.

## Retraining: a fixed procedure, not a choice

`r4_pit_long`'s final window trained to 2022-01-02. Scoring 2026 data against
it is 4.7 years of extrapolation, and it widens every month. A frozen model
would be measuring decay, not the strategy.

So the model is retrained **annually, on 1 April**, by a procedure fixed here:

* identical config (`train=r4_pit`, `model=signal`), identical hyperparameters,
  identical seed, `xs_normalise` null;
* walk-forward windows ending at the most recent completed quarter;
* the resulting artefact is used **whatever its gate says** — the gate is
  recorded, never used to select. A refit that fails its gate is a finding to
  report, not a reason to substitute a different model.

**No hyperparameter, feature, architecture or horizon may change.** If one
does, the clock resets. This is the difference between a fixed procedure
(preserves the evidence) and a choice (destroys it).

## Baselines

Recorded every day alongside the strategy, on the same universe, same costs,
same tax:

* `equal_weight`, monthly — the primary comparison;
* `equal_weight_frozen`, monthly — the secondary;
* `momentum_topk`, monthly — because R5's criterion names the *best* baseline,
  and momentum was the bar on the fixed universe even though it collapsed to
  −0.002 on the point-in-time one.

The statistic is the one already used: paired daily log-return difference,
moving-block bootstrap, block 21, 10,000 resamples.

## What is decided in advance

**Review dates: 12, 24, 36 and 48 months after the first rebalance.** Fixed
calendar dates, not "when it looks interesting."

**No verdict before 24 months.** The holdout gave a ±0.068 half-width on 15
months; excluding zero at a +0.035 effect needs roughly 4.5 years of daily
data. A 12-month review can check sign and mechanics — turnover near the
in-sample 1.1, scrips sold per rebalance near 8, DP charges near ₹5,000 per
₹1 lakh — and nothing more. **Reporting a 12-month t-statistic as evidence is
forbidden by this document.**

**Success** = the paired bootstrap CI over the forward period, against
`equal_weight` monthly, excludes zero on the positive side. Nothing less
counts, and no other statistic may be substituted later.

**Abandonment**, any of the following at a scheduled review:

1. cumulative annualised excess below **−10%** — clearly broken, not unlucky;
2. maximum drawdown breaches **−0.55**, worse than anything in sample;
3. realised turnover exceeds **2×** the in-sample 1.129, meaning the band is
   not doing what it did in the backtest;
4. a defect is found in the strategy's own execution path — abandon, fix,
   restart the clock. Fixing and continuing is not permitted.

**Not grounds for abandonment:** a negative excess that is inside its interval.
That is the outcome this design expects to see often, and reacting to it is
exactly the selection this document exists to prevent.

## What gets recorded, daily

Ledger row per fill; per-day NAV for the strategy and all three baselines;
target vs realised weights; every charge itemised; the signal's own rank IC
against realised forward returns as it becomes available; and the model
artefact SHA in force that day.

## Honest statement of what this can and cannot deliver

It cannot deliver a tradeable verdict in 2026, and probably not in 2027. At
the measured effect size it needs about **four to five years**. It is started
now because calendar time is the only input that cannot be bought later, and
because every alternative — another sweep, another architecture, another
factor — adds to a multiplicity problem that is already the dominant source of
uncertainty in every number this project has produced.

If the honest answer in 2031 is "no edge", that is a real answer, arrived at
for the first time by a method that could have produced one.

## Signatures

Configuration frozen: 2026-09-09.
First rebalance: **not yet executed** — pending the training box, which holds
the model artefact and the point-in-time panel.
