# Forward paper trading: pre-registration

**Status: WITHDRAWN 2026-09-09, before any forward data existed.**

The configuration frozen below is model-based, and the matched null control
(`audit/S4_NULL_CONTROL.md`) refutes it: against a **random** signal running
identical machinery — same universe, same K=30, same 1% band, same stop, same
costs and tax — R4's signal wins 8 of 8 matched pairs in sample and loses
**13 of 15 out of sample** (sign test p ≈ 0.004). The signal's apparent
+0.0431/yr over equal-weight is ~+0.0261 machinery and only ~+0.017 signal, and
the +0.017 does not survive.

Withdrawing costs nothing, because no forward data has been collected. That is
the entire point of pre-registering before starting rather than after: this
document did its job by being falsifiable before the clock started, not after
four years of trading a rule that a coin flip reproduces.

**What survives is a risk result with no model in it** — a 30-name book with a
1% band and a volatility stop had Sharpe 0.307 against equal-weight's 0.151 and
a −0.061 drawdown against −0.137 on the unseen holdout, at the same return.
Whether *that* is worth forward-testing, and with what selection rule in place
of the model, is an open decision recorded at the end of this document.

Everything below is retained unaltered as the record of what was frozen and
why. It is not a live proposal.

---

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

---

# Withdrawal note and the open decision

## Why it was withdrawn

`audit/S4_NULL_CONTROL.md`. The control that should have run alongside every
allocator result was broken — it used `band_grid[0]` (0.0) and no risk overlay
against arms carrying band 0.010 and a stop, and wrote no NAV — so every risk
comparison before 2026-09-09 was uncontrolled. Repaired and re-run, it shows
the signal's advantage over random selection is in-sample only.

## What the evidence now supports

1. **No signal-driven edge is demonstrated.** R4's rank IC is real and passes
   its gate, but does not convert into portfolio value once concentration, the
   band and the stop are controlled for. A model with *higher* holdout IC
   (+0.0459 vs +0.0338) produced the *worse* portfolio, so the gate metric is
   not measuring what the allocator consumes.
2. **A portfolio-construction effect does survive**, out of sample, with a
   random signal: ~30 names instead of ~350, a 1% no-trade band and a
   volatility-scaled stop roughly halve drawdown at unchanged return.

## The open decision — for the user, not for me

"Random selection" is not something anyone would run live. Replacing it
requires a selection rule, and every candidate is now untested:

| option | what it would test | honest status |
|---|---|---|
| **A.** Forward-test the risk rule with a defensible non-model selection (e.g. top-30 by median turnover) | Does the drawdown halving hold live? | The rule is supported; *this specific selection* has never been measured |
| **B.** Forward-test nothing; treat the project as concluded | — | Defensible. Nothing has demonstrated alpha |
| **C.** Fix the gate metric first — score the signal on top-K forward return rather than full-cross-section IC — and only then reconsider | Whether a signal selected for what the allocator actually consumes does better | Real work, and it re-opens the search this project has repeatedly been burned by |

Option A must not silently become "pick the selection rule that backtests best"
— that is the search which produced four reversals. If A is chosen, the
selection rule is to be fixed on stated reasoning **before** it is measured,
and a new pre-registration written with a new start date.
