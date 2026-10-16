# P6 — Third live book: 12-1 momentum, 15 names, monthly

**Pre-registered 2026-10-16 (~14:15 IST), before the book's first trade.**
Requested by the owner the same day after the strategy review ("lets go with
your 10-15 stock suggestion ... keep all three running parallelly ... so that
we can compare and contrast the overall maximum profit gained by each
different strategy"). Nothing above "Operations" is edited after the first
fill. P2, P3 and both existing live books are not touched.

## Why this book

The project's own evidence says the r4 model's top 30 is no better than random
top-30 books (`audit/S6_TOPK_GATE.md:28-29`, ranks 12/21 and 7/21) and that the
portfolio machinery (stop, band, inverse-vol), not the signal, carried the
in-sample edge (`audit/S4_NULL_CONTROL.md:106-109`, `audit/S5_TURNOVER_SELECTION.md:91-93`).
This book swaps the model for the simplest published cross-sectional signal and
cuts the name count so fixed per-sale charges weigh less at ₹1L
(DP ₹15.34 per scrip sold, `src/trader/env/costs.py:55`: ≈0.46% of a ₹3,300
position at K=30, ≈0.23% of a ₹6,700 position at K=15).

**Correction recorded here, found while building this book:** the strategy
review cited R5's best holdout baseline, `momentum_topk_monthly` (+0.0449 vs
equal-weight, t 0.31, `audit/R5_VERDICT_PIT.md:110-113`), as support for 12-1
momentum. That baseline ranks on **20-day** momentum (`log_return_20d`,
`src/trader/env/baselines.py:149-160`), not 12-1. 12-1 momentum has **not**
been tested anywhere in this repo. Its support is external literature only
(Jegadeesh & Titman 1993; the IIMA Indian momentum factor). This book is its
first test here, and it is not a significance test (see Reading).

## Rules (fixed)

| | P6 momentum book | for comparison: monthly live book (P2 rules) |
|---|---|---|
| Signal | `log(adj_close[d−21] / adj_close[d−252])`, panel sessions; tradeable rows with both prices > 0 (`scripts/momentum_signal.py`) | r4_pit_long model |
| Names (K) | **15** | 30 |
| Rebalance | monthly, first session of the month | same |
| Everything else | identical: inverse-vol weights, 10% name cap, 25% sector cap, band 0.010, turnover budget 0.30 per rebalance (lifted for first funding), whole shares + top-up, ₹500 minimum trade, volstop 1.0×daily vol×√20 clipped 5–30% with 21-session cooldown, Zerodha delivery costs | |

* Executor: `scripts/live_paper.py`, unchanged except `K` read from
  `TRADER_LIVE_K` (default 30, so the other two books are unaffected).
* Capital ₹1,00,000. Folder `audit/paper/live_momentum/`.
* Schedule (`docker/paper.crontab`): signal rebuilt 21:50 and 08:20; trade
  09:18 / 09:22 / 09:32; marks every 15 min; close mark 15:37.
* **First funding: Mon 2026-10-19 at 09:18 IST** at live Kite quotes, signal
  dated 2026-10-16. First rebalance 2026-11-02.

A rehearsal on 15 Oct closing prices (`trade --prices panel --dry`, nothing
written) bought 15 names, 99.4% invested, ₹118 charges. Top scores were large
genuine rallies (e.g. STLTECH, CUPID, BLISSGVS: smooth price paths, no 1-day
move beyond ±35%, so not split or demerger artefacts).

## Known risks, stated now

* Momentum crashes: the Nifty200 Momentum 30 index fell ≈−31.8% from Sep 2024
  to Apr 2025 versus ≈−15.4% for the Nifty (external, cited in the 16 Oct
  review; UNVERIFIED here). With 15 names the swings will be larger than the
  other two books'.
* The 12-1 winners skew to small, volatile names. Inverse-vol weighting
  shrinks them; the 10% cap and whole shares bind on expensive ones.
* Universe is the panel's ~500 tradeable names, which is not point-in-time
  (`CLAUDE.md`, survivorship).

## Reading (what this book can and cannot show)

The three live books are compared **descriptively**: cumulative return net of
all costs, maximum drawdown and turnover, on the dashboard daily. **No winner
is declared from them.** One path of 15 names over one year cannot separate
skill from luck. There is no random-book null for P6 yet; until one is built
and pre-registered, any "P6 beat P2" statement is unsupported (CLAUDE.md
rule 2).

* Fixed descriptive read: **2027-10-19** (12 months), cumulative return and
  max drawdown of all three books, recorded once under Results.
* Not permitted before then: changing K, the lags, the cadence, or any shared
  rule; pausing or resetting the book after a loss; selecting between books on
  interim numbers.

## Operations

*(appended as it runs; not part of the pre-registration)*

## Results

*(appended at the read)*
