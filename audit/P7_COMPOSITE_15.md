# P7 — Book 4: three price signals, within sector, 15 names, monthly

**Pre-registered 2026-10-18 (~14:30 IST), before the signal was computed on any
period and before any backtest or live number exists.** Owner's request
(16 Oct: "aggregate all these simple strategies that are found online and rank
stocks per sector ... creating another Book, say Book 4"); mix chosen by the
owner on 18 Oct from options set out in `14_weekend_scoping_notes.md` §B.
Nothing above "Operations" is edited after the first result. P2, P3, P6 and
their books are not touched.

## Signal (fixed; `scripts/composite_signal.py`)

For ticker i on date d, in panel sessions:

| | definition | higher means |
|---|---|---|
| momentum | `log(adj_close[d−21] / adj_close[d−252])` (12-1, identical to Book 3) | stronger past-year trend |
| low volatility | `−std` of daily log returns over the 252 sessions to d | steadier |
| 52-week high | `adj_close[d] / max(adj_close[d−251..d])` | nearer its 1-year high |

* Each is z-scored across tradeable names on d **within the name's sector**
  when that sector has ≥ 5 such names, otherwise across all names.
* Score = plain mean of the three z-scores. **Equal weights, fixed now.**
* A name needs a positive price on all 253 sessions d−252..d.

## Book rules (fixed)

Identical to Book 3 (`audit/P6_MOMENTUM_15.md`) except the signal:
- K = 15, monthly (first session of the month).
- Inverse-vol weights, 10% name cap, 25% sector cap.
- Band 0.010, turnover budget 0.30 (lifted for the first funding).
- Whole shares plus top-up, ₹500 minimum trade.
- Volstop (1.0×daily vol×√20, clipped 5–30%), 21-session cooldown.
- Zerodha delivery costs.

Live: ₹1,00,000, `audit/paper/live_composite/`. Trade 09:19 / 09:23 / 09:33,
marks every 15 min, close mark 15:38, signal rebuilt 21:55 and 08:25. **First
funding Mon 2026-10-19**, signal dated 2026-10-16; first rebalance 2026-11-02.

## Historical test (fixed; run once)

`scripts/run_allocator.py` with the existing matched null control (20 stable
random signals, identical machinery), `data=bhav_v1` (point-in-time universe).
K=[15], band 0.010, monthly, horizon 20d, tax on, and the default risk arms,
whose volstop arm is the one this book trades. Two periods, as S4:
- `oos_r4_pit_long`, 2011-07-04 → 2024-07-02
- `holdout`, 2025-04-01 → 2026-09-04

**Criterion, for the volstop arm on each period:** cumulative log return
ranks **1st of 21** against its own 20 random books, **and** beats
equal-weight. A pass on `oos_r4_pit_long` is in-sample in the weak sense that
these signals are published and were known to work before 2011. The
**holdout is the read that matters** (≈17 months, so low power: a fail there
is weak evidence against, a pass is weak evidence for).

**Information only, never used to select:** the same run for momentum alone
(Book 3's signal) at K=15.

**Whatever the result, the book goes live on 19 Oct.** A failed historical
test is recorded plainly, and the live book then runs as a descriptive
comparison like Book 3. Nothing is re-tuned in response.

Known limits: delisting bias is unrecoverable (`audit/P4_survivorship.md:15`)
and flatters every signal alike. Tested on one mix, so no multiple-testing
correction is needed for the criterion. The info-only momentum run is a
second look and is not eligible for any decision.

## Reading the live book

Descriptive, alongside Books 1–3. Fixed read on **2027-10-19** (12 months):
cumulative return, max drawdown, turnover. No winner declared from one path.
Not permitted before then: changing the mix, the weights, K or any shared
rule; pausing or resetting after a loss.

## Operations

*(appended as it runs; not part of the pre-registration)*

**2026-10-18 wiring.** The live signal (`scripts/composite_signal.py`, forward
panel, through 2026-10-16) is built nightly at 21:55 and 08:25. The book is in
`docker/paper.crontab` and on the dashboard as "composite". A rehearsal on
16 Oct closing prices (`trade --prices panel --dry`, nothing written) bought
15 names, 99.5% invested, ₹118 charges, mostly large caps (GRASIM,
OBEROIRLTY, LAURUSLABS, INDHOTEL, LT, ICICIBANK …).

**Probe.** A first holdout run with a single random book (seed 42), 14 s and
1.4 GB, was a timing probe only. Its output is superseded by the 20-seed run
below and is not a result. With 20 seeds the holdout took 62 s.

**Timestamp correction (2026-10-18, appended).** The header says this was
pre-registered "~14:30 IST". The pre-registration commit `b909994` is dated
2026-10-18 14:05:02 +0530 and the holdout result commit `5f3a7aa` 14:11:49
+0530, so "~14:30" is wrong. The ordering it claims still holds: the rules
were committed before the result. The holdout NAVs start 2025-06-27 rather
than the registered 2025-04-01 because of the 60-session lookback; the window
is otherwise as registered.

## Results

**Holdout (`holdout`, NAVs 2025-06-27 → 2026-09-03, 1.19 y; 60-session
lookback).** Regenerate with `uv run python scripts/p7_score.py
audit/navs_p7/holdout_composite3`, from the 20-seed `run_allocator.py`
command in "Historical test". MLflow experiment `allocator`: volstop arm
`104ba1fc52d9459ba555c29b160bb69d`, equal-weight
`f900fa08f4ee4316ae3f0807d5673337`.

| arm | cum log | rank / 21 | random median | vs EW |
|---|---|---|---|---|
| **volstop (verdict)** | **+0.0874** | **1** | +0.0369 | **+0.0164** |
| none | +0.1560 | 1 | +0.0670 | +0.0850 |
| stop10 | +0.1032 | 1 | +0.0304 | +0.0322 |
| stop15 | +0.1151 | 1 | +0.0554 | +0.0441 |
| equal-weight | +0.0710 | | | |

**Holdout criterion: PASS** (rank 1/21 and above equal-weight). As
pre-registered, 1.19 years is low power: this is weak evidence for, not a
validation (CLAUDE.md rule 2). Observed, not acted on: on this period the
volstop arm earned the least of the four signal arms.

**`oos_r4_pit_long` (2011–2024): FAIL.** NAVs 2011-09-28 → 2024-07-01
(12.76 y). Regenerate with `uv run python scripts/p7_score.py
audit/navs_p7/oos_composite3`, from the same 20-seed command with
`+split=oos_r4_pit_long +signal_tag=composite3_oos
'++allocator.nav_dir=audit/navs_p7/oos_composite3'`. MLflow experiment
`allocator`: volstop arm `35138ab6b94542c1beabbcba7d3b9b39`, equal-weight
`a03cbce3a42447bc924bd1806fcada5e`. Log `logs/retrain/p7_oos_20261018.log`.

| arm | cum log | CAGR | MDD | rank / 21 | random median | vs EW |
|---|---|---|---|---|---|---|
| **volstop (verdict)** | **+1.6131** | +13.48% | −35.8% | **4** | +1.5702 | **−0.2243** |
| none | +1.8458 | +15.57% | −38.5% | 2 | +1.7465 | +0.0084 |
| stop10 | +1.6340 | +13.66% | −33.2% | 4 | +1.5618 | −0.2034 |
| stop15 | +1.7279 | +14.50% | −34.8% | 1 | +1.6036 | −0.1095 |
| equal-weight | +1.8375 | +15.49% | −51.2% | | | |

**Criterion: FAIL** (rank 4/21, and 0.22 log below equal-weight). Recorded
plainly. As registered, the book stays live and nothing is re-tuned. Observed,
not acted on:
- Over 13 years the signal sits only modestly above its own random books
  (+0.04 log over the random median in the verdict arm).
- Every stopped arm trails equal-weight. Equal-weight's −51.2% drawdown is
  the cost of that return; the stops cut drawdown to about −33 to −36%.
- The holdout PASS above is therefore one short period against a 13-year
  FAIL: weak evidence either way, and the longer period does not support it.

**First attempt failed to run (14:24, 15 s):** `stable_null_signal` raised
"2011-07-04 is outside the null signal's anchor span". Its anchor is
2020-01-01, so **no random-book test could run on any pre-2020 period** after
2026-09-10. The registered test was not runnable as written (CLAUDE.md
rule 4). Fixed in `src/trader/allocator/null_signal.py`: pre-anchor dates
draw from a separate `":pre"` stream. Every cell on or after the anchor was
checked bit-identical to the old code (3 seeds × 827 dates × 400 names), so
P2 and P3 reproduce unchanged. Test:
`tests/unit/test_null_signal_stability.py::test_pre_anchor_dates_are_stable_and_leave_post_anchor_cells_unchanged`.
Peak RSS of the successful run: about 30 MB sampled from the parent process,
which UNDERSTATES the true peak if work ran in child processes. Wall time
7.3 min (14:26:16 → 14:33:32).

The info-only momentum runs are not yet run.

*(live read at 2027-10-19)*
