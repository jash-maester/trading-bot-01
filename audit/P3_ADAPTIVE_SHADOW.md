# P3 — Adaptive shadow book: refit and rebalance twice a month

**Pre-registered 2026-09-30 (~23:00 IST), before any refit for this book has been
run or seen.** Requested by the owner the same evening ("set up the ... biweekly
refit shadow book ... rebalance a bit every now and then, and self adapt
depending upon the data"). Nothing above "Results" is edited after the first
refit. P2 (`audit/P2_PAPER_PERMUTATION_TEST.md`) is not touched: its book, its
model, its record and its schedule stay exactly as they are.

## Question

Does a book whose model is **refitted every half-month** on the most recent data
the model's windowing allows, and which **rebalances on the same half-month
cadence**, do better than the frozen P2 signal book run under otherwise
identical rules?

## What differs from the P2 signal book — exactly two things

| | P2 signal book (frozen) | P3 adaptive book |
|---|---|---|
| Model | `r4_pit_long`, fixed until the annual refit (1 Apr) | refitted after the close of the session before every rebalance |
| Rebalance | monthly, first session of the month | **semimonthly**: first session on/after the 1st and on/after the 16th |

Everything else is identical: K=30, no-trade band 0.010, max weight 0.10,
turnover budget **0.30 per rebalance** (so up to twice the monthly turnover),
volatility stop (1.0× daily vol × √20, clipped 5–30%, 21-session cooldown),
Zerodha delivery costs, capital-gains tax, ₹1,00,000 cash deployed in full at
the first rebalance (`initial_full_deploy`), whole-share top-up
(`topup_cash`), universe from the panel, horizon 20d.

The two changes are tested together because the owner asked for both, and
refitting twice a month while trading once a month would leave every other
refit unused. A result therefore cannot say which of the two caused it.

## Refit procedure (fixed)

* `train=r4_pit model=signal`, identical hyper-parameters, seed and
  `xs_normalise=null`. No hyper-parameter, feature, architecture or horizon may
  change.
* One window, anchored at the end of the data: train 5 years, purge 3 months,
  validation 12 months, purge 3 months, **test 1 month** ending at the latest
  session in the data (`walk.test_months=1`, `walk.n_windows=1`). The standard
  12-month test would push every training row back another 11 months; one
  month is the shortest the training code supports. So a refit's newest
  *training* row is ~19 months old and its validation (early-stopping) year
  ends ~4 months ago. This lag is inherent to the model's design and is part
  of what is being tested.
* Data: a panel built from the NSE bhavcopy store through the refit session
  (`data/panels_refit/`), same feature pipeline as every other panel.
* The artefact replaces the previous one **whatever its gate says**. The gate is
  recorded, never used to select.
* A refit that fails or is not run leaves the previous model in force, and the
  miss is logged in `audit/paper/adaptive/refits.jsonl`.

**No lookahead:** the signal dated `d` (which trades at `d+1`'s open) comes from
the latest refit whose data ends on or before `d`.

## Control

20 random books with the same semimonthly schedule and the same stable null
signals (seeds 1–20, as P2), plus the other machinery P2 runs (equal-weight and
the three other stop overlays), replayed nightly from the same data.

## Record

* Start: first rebalance **2026-10-01** at the open (refit on data through
  2026-09-30).
* End-of-day replay only, like P2. The live ₹1,00,000 book on the dashboard
  stays the frozen P2 book.
* `audit/paper/adaptive/record.jsonl`, append-only; determinism checked against
  the previous snapshot exactly as in P2.

## Criterion (fixed now)

Read **once, at 24 months** of record (first eligible 2028-10-01). Before that,
any number is plumbing, not evidence, and is shown only with that caveat.

**PASS requires both:**

1. Paired difference, adaptive minus frozen P2 signal book, in daily log
   returns aggregated to non-overlapping 20-session blocks: mean > 0 with
   t > t_crit (two-sided 95%, df = blocks − 1).
2. The adaptive book ranks **1st of 21** against its own 20 semimonthly random
   books on cumulative log return.

Anything else is FAIL. A failed result is recorded plainly.

## Not permitted

Changing the refit cadence, the window, any hyper-parameter, K, band, stop or
turnover budget; selecting or skipping a refit because of its gate or its
predictions; rewriting any line of the record.

## Compute (measured 2026-09-30, before this was written)

| | wall | peak memory | source |
|---|---|---|---|
| one-window refit, Mac MPS | 1,057 s (17.6 min) | 3.45 GB | MLflow run `372aae5fc5a9420ab52c8febfc9d74ee`, `logs/retrain/refit_probe_20260930.*` |
| 60 steps + one validation pass, Docker CPU | ~141 s (vs 36.4 s on MPS) | 7.4 GB | `/tmp/cpu_probe.log` in the paper container |

A refit therefore runs on the host's GPU (~18 min, twice a month), not in the
Docker VM (~60 min at 7.4 GB of its 9 GB). Where the host job is scheduled is
recorded under Operations.

## Operations

*(appended as the pipeline is wired; not part of the pre-registration)*

**2026-09-30 wiring.**

| piece | where | when |
|---|---|---|
| refit | `scripts/refit_adaptive.py` on the **host** (MPS) | eve of each 1st/16th rebalance (`--if-due`; also catches up a missed refit) |
| host scheduler | `ops/launchd/com.paperbook.refit-adaptive.plist` → `scripts/refit_adaptive_host.sh` | nightly 21:25; not yet loaded, see below |
| stitched signal | `scripts/adaptive_signal.py` (newest refit with data ≤ signal date; superseded refits cached) | nightly, in the container |
| replay + record | `scripts/adaptive_daily.sh` → `audit/paper/adaptive/` | 21:40 and 08:10, after P2's runs; P2's script is untouched |
| schedule | `RebalanceSchedule("semimonthly")` | first session on/after the 1st and the 16th |
| dashboard | Experiment page, "Adaptive shadow book" | live |

**Host scheduler, not yet active.** macOS blocks background jobs from this
USB volume (the reason the paper loop moved to Docker). The launchd job needs
the owner to grant Full Disk Access to `/bin/bash` (System Settings → Privacy &
Security → Full Disk Access), then:

    cp ops/launchd/com.paperbook.refit-adaptive.plist ~/Library/LaunchAgents/
    launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.paperbook.refit-adaptive.plist

**Correction to the compute table (2026-09-30, 23:20).** The CPU probe's log,
`/tmp/cpu_probe.log` in the paper container, was deleted by the agent during
clean-up after the table was written. The durable record of the probe is
MLflow run `157d7bb54b1c4be5ab76f553730f1c10` (`refit_cpu_probe_W1`, 130.6 s
start to end). Its peak memory, 7,401 MB as printed in-session, has **no
surviving artefact: UNVERIFIED.** The decision it supported (refit on the host
GPU, not the 9 GB VM) rests equally on the ~3.9× slower CPU step time.

**First refit.** `adaptive_20260930`, data through 2026-09-30, 589 s on MPS,
early stop at epoch 9 (best at 4), gate FAIL (recorded, not used). MLflow runs
`da7186b3fe7646f2b8250d5015ce055f`, `798f344a29644db68cc5ea52cdc45afd`;
encoder SHA `5ecc53e8…`. Serves the 2026-10-01 rebalance.

**Disk.** A refit writes `embeddings.npy` (238 MB) and `windows/` (228 MB),
which prediction never reads; `refit_adaptive.py` deletes both after a
successful refit, leaving ~2 MB per refit. The model file, index, summary and
gate are kept.

Until then a due refit is run by hand (`uv run python scripts/refit_adaptive.py
--if-due`); the next is due on the evening of Thu 15 Oct 2026. A refit that is
not run leaves the previous model in force, as pre-registered.

## Results

*(appended at the read)*
