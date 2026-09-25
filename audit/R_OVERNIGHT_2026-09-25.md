# Overnight research loop — 2026-09-25/26 (pre-registration + results)

The user authorised an unattended "autoresearch" loop while asleep: finish the
parity retrain, run inference, and tune judiciously. This file is written
**before any overnight result exists**. Everything below "Results" is appended
as runs finish; nothing above it is edited afterwards.

## Rules for the loop

This project's central finding is that selecting on measured performance is
anti-predictive here: four in-sample rankings reversed out of sample
(`audit/R5_VERDICT_PIT.md`, `S3`, `S4`, `S5`). So an autonomous loop that
tunes until something looks good would manufacture exactly the result the
project has learned to distrust. The loop is therefore constrained:

1. **One pre-registered variant per hypothesis.** No sweeps, no second tries
   on a failed hypothesis with different knobs.
2. **Pass criteria fixed here, now.** A result that misses them is reported as
   a failure with its numbers, not re-analysed until it passes.
3. **In-sample walk-forward first; holdout at most once**, and only for a
   variant that passed in-sample. The holdout is never used to choose.
4. **Nothing touches `audit/P2`.** The forward test's frozen artefact
   (`r4_pit_long`), its record and its container stay as they are.
5. **Every run is reported**, including crashes and nulls.

## E0 — MPS parity retrain (the requested long run)

`bash scripts/retrain_parity.sh` → tag `r4_pit_long_mps`; pruned TCN, fp32
"highest". Criteria (fixed in `scripts/compare_signal_artefacts.py` before the
run): **same gate verdict; |Δ mean 20d IC| < window SE; median per-date rank
correlation of r_hat_20d ≥ 0.8.** Then inference on the 2025-26 holdout and
the top-K gate, for the record.

## E1 — the screen (the open item from `audit/S6`)

S6: R4's information sits in its **bottom** tail (bottom-30 rank 21/21,
t −3.20), not its top. The natural long-only use is a screen: hold the
eligible universe minus the signal's worst names.

**Variant (one):** keep all tradeable predicted names except the **bottom 30**
by `r_hat_20d` (K=30 fixed from S6, not chosen here), equal-weight, re-formed
every 20 sessions, 23 bps round-trip cost proxy on the changed fraction.
Measured as excess over the equal-weight universe per walk-forward window,
against **20 random screens** of the same size (same support).

**Pass:** window-level mean excess > 0 with t > t_crit(95%) **and** paired
difference vs the random-screen mean > 0 with t > t_crit, on `r4_pit_long`'s
13 windows. **Expected size, stated now:** 30/~400 × 0.85%/20d ≈ +0.06%/20d
gross, ~0.8%/yr — small; a pass would be a real but modest result.

## E2 — a loss that puts information where the book looks

Hypothesis: the masked MSE on z-scored returns rewards getting the whole
cross-section right, and the model found its easiest wins in the bottom tail.
A **listwise** loss (ListNet: per-date softmax cross-entropy between
`softmax(target)` and `softmax(prediction)` over tradeable names) weights the
top of the target ranking most heavily, so it should move information toward
the names a long-only top-K buys.

**Variant (one):** identical to `r4_pit_long` (config, windows, seed, data,
pruned TCN, fp32 highest) except `train.loss=listnet` (temperature 1 on the
z-scored target). Tag `r4_pit_long_listnet`.

**Pass (in-sample, 13 windows), all three:**
- top-30 gate (`scripts/topk_gate.py`): paired t vs 20 random books >
  t_crit **and** rank ≤ 2 of 21;
- the rank-IC gate still PASSES (information not destroyed to get there);
- top-30 mean net excess > `r4_pit_long`'s (−0.00241 per 20d).

**Only if it passes:** one look at the 2025-26 holdout with the same top-30
gate, reported whichever way it goes. If it fails, E2 is closed — no second
loss function tonight.

## Results

*(appended below as runs finish)*
