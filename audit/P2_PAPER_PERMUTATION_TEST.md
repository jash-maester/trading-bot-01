# Forward paper test: pre-registration (permutation design)

**Status: REGISTERED 2026-09-10. Clock not yet started.** The first scored
session is the first NSE session *after* 2026-09-09, recorded by the first run
of `scripts/paper_daily.sh` that includes it. Nothing in "Frozen" below may
change afterwards; if it does, the clock resets and this document is
superseded.

Supersedes `audit/P1_PAPER_PREREGISTRATION.md` (withdrawn: it froze a
model-based configuration chosen from a ranking that reversed, and compared it
to equal-weight, which a random signal beats by +0.026/yr through machinery
alone — `audit/S4_NULL_CONTROL.md`).

---

## The question, stated once

**Does R4's signal add value over random selection, in a fixed configuration,
on data that did not exist when the configuration was chosen?**

Not "does it beat equal-weight" (machinery does that), not "which arm is
best" (rankings reverse here — four times), and not "is the drawdown better"
(a stop does that with any signal). One question, one statistic, one decision
date.

## Frozen

| | |
|---|---|
| signal artefact | `data/signal/r4_pit_long` (13 windows, 2005–2024; last window trained to 2022-01-02) |
| universe | point-in-time `LiquidityRule`; ≥₹5 cr median turnover, 365-day lookback, ≥100 sessions, ≤504 names |
| book | **K=30, no-trade band 0.010, monthly, volstop** (`stop_vol_mult=1.0`, 20-day horizon, 21-day cooldown) |
| caps | max name 0.10, max sector 0.25, turnover budget 0.30 |
| costs | full Zerodha delivery: STT 0.1% both legs, DP ₹15.34/scrip/sell-day, stamp 0.015% buy; STCG 20%+cess, LTCG 12.5% |
| capital | ₹1,000,000 notional |
| null | **20 random books**, seeds 1–20, identical machinery, scores masked to the signal's support |
| data | NSE daily bhavcopy + NSE index archive (`^NSEI`). **No Kite.** |
| panel | forward panel from 2024-06-01 (365 d eligibility + 60 d features ahead of the warm-up) |
| execution | `PanelTradingEnv` (broker parity verified 0.15% / 0.27%, `audit/S2` §5) |
| runs on | the Mac mini, `launchd`, daily 14:00 UTC (19:30 IST, after NSE publishes) |

Why volstop and not none: chosen on the stated reasoning that a per-name
volatility-scaled threshold treats a utility and a small-cap differently,
where one fixed number does not. **Not** because it scored best — on the
warm-up it did not (see below). The other three overlays (none, stop10,
stop15) are run and recorded for context and may never be used for selection.

## The statistic

For every session D after the freeze date, the **recorded** daily log return
of each of the 21 volstop books. Cumulative sum over the scored period. The
signal book's **rank among 21** (1 = best).

**Recorded** means: taken from the first run that included D, from that run's
own replay (`nav_D / nav_{D-1}`), appended once to `audit/paper/record.jsonl`
and never rewritten. Each run replays the whole span from scratch — that is
how position state, the band and the stops are carried without persisting
anything — but back-adjustment is retroactive (a future split rescales past
`macd`, `atr_14`, `dollar_volume_20`), so a replay can re-decide its own
history. A real book does not. Divergence between today's replay and the
previous snapshot is measured and logged every run; it is never absorbed into
the record.

## Warm-up baseline — frozen, and the honest starting position

The replay from the forward panel's first full-lookback date to the freeze
date, all of it after the model's training end. Computed 2026-09-10, before
any scored data existed:

| 2024-08-29 → 2026-09-09, 505 sessions | cumulative log return |
|---|---|
| signal / volstop | +0.0165 |
| equal_weight, monthly | +0.0127 |
| null / volstop × 20 | mean **+0.0384** (run of record; pre-run estimate was +0.0393 — see note) |
| **signal rank among 21** | **17** |

*Note, 2026-09-10 11:05 UTC.* The pre-run estimate was computed on a store
built with 1,118 corporate actions. The first run of `paper_daily.sh`
refreshed the feed to 1,119 and rebuilt the store, which re-adjusted three
names with ex-dates inside the warm-up span — `FCL.NS` (×2.5, through
2025-10-30), `BAJFINANCE.NS` (×0.4, through 2025-06-13), `NAZARA.NS` (×2.0,
through 2025-09-25). Re-scaled history moves `dollar_volume_20`, which feeds
eligibility, which moves the support the random books draw from: their mean
shifted by −0.0009. The signal book never held those names, so its NAV was
unchanged to 3.3e-11 relative. Rank unchanged. This is the retroactive
re-adjustment mechanism the design anticipates, observed on the first run,
before any scored data existed.

**Rank 17 of 21.** On two years of unseen data the signal book sits in the
bottom quartile of its own random distribution. That is the starting position
and it is consistent with `S4`. It is not the test — the scored period begins
after it — and it is recorded here precisely so it cannot be forgotten if the
forward result looks better.

**Determinism reference:** signal/volstop NAV on 2026-09-09 =
**1,016,641.125632** from 1,000,000.00, as computed by the run of record
(`2026-09-10T11:05:43Z`, snapshot `audit/paper/snapshots/2026-09-09`). The
pre-run estimate was 1,016,641.125598; the 3.4e-5 rupee gap is 3.3e-11
relative — floating-point noise after the store rebuild described above.
Tolerance is **relative 1e-9** (`paper_record.py`). Every subsequent run's
replay is compared to the previous snapshot on every overlapping date; a
divergence above tolerance is logged with the books affected. A corporate
action re-adjusting history is the expected reason; anything else is a
defect.

## Retraining — a fixed procedure

Annually on **1 April**, first on 2027-04-01: `train=r4_pit model=signal`,
identical hyper-parameters, seed and `xs_normalise=null`, walk-forward windows
ending at the most recent completed quarter. The artefact replaces
`SIGNAL_DIR` **whatever its gate says**; the gate is recorded, never used to
select. The record notes the artefact SHA in force each day. No
hyper-parameter, feature, architecture or horizon may change.

## Reviews and the decision

| when | what happens |
|---|---|
| +12, +24, +36 months | mechanics only: turnover near the warm-up 3.79, ~11 scrips sold per rebalance, DP charges, replay divergence log, data gaps. **Rank is reported, not acted on.** |
| **+48 months** | **the verdict** |

**Success** = rank **1 of 21** at 48 months (permutation p ≈ 1/21 ≈ 0.048).
Secondary, reported alongside: moving-block bootstrap CI of daily
(signal − null mean) excludes zero. Nothing else counts, and no statistic may
be substituted later. **A 12-month rank or t-statistic reported as evidence
is forbidden by this document.**

**Abandonment**, mechanical only, at any run:

1. more than 5 consecutive sessions with unrecoverable data;
2. realised annual turnover above **2×** the warm-up (7.6);
3. absolute drawdown beyond **−0.55**;
4. a defect in the execution path — abandon, fix, restart the clock.

**A bad rank is not grounds for abandonment.** Reacting to it is the
selection this design exists to prevent.

## The economic bar, stated now

If the signal is real, the in-sample increment over machinery was ~+0.017/yr.
**Even a full success here is worth roughly +1–2%/yr net.** That is written
down today so the number cannot be re-litigated in 2030 by whoever is reading
this. It is also why the loop runs Kite-free on a machine that is on anyway:
the test is only worth running if running it costs almost nothing.

## Power, honestly

At +1.7%/yr against the observed dispersion this needs roughly four years to
reach conventional significance. That is the review horizon. Nothing sooner is
evidence.

## Not permitted

Any change to the Frozen table; adding, removing or re-choosing overlays;
changing seeds; re-picking K, band or cadence; substituting the artefact
outside the annual rule; rewriting any line of `record.jsonl`.

## Signatures

Frozen: 2026-09-10.
Warm-up baseline recorded: 2026-09-10, run `2026-09-10T11:05:43Z`
(`audit/paper/record.jsonl` line 1, date 2026-09-09).
First scored session: **pending** — the first NSE session after 2026-09-09,
filled by the run that records it.
