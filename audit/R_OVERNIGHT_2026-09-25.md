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

### E1 — FAIL (2026-09-25 ~23:00 IST)

`scripts/topk_gate.py --signal r4_pit_long --which screen --k 30`
(`audit/topk_gate/r4_pit_long_screen30.json`), 13 windows:

| | per 20d, net of cost | t (crit 2.18) | |
|---|---|---|---|
| screen excess over the universe | **+0.00039** (9/13 windows > 0) | **1.04** | ✗ |
| paired vs 20 random screens | +0.00086, **rank 1 of 21** | **2.43** | ✓ |

Fails the pre-registered criterion, which required both. What it does show:
the bottom-tail information is real — excluding the names R4 flags beats
excluding random names of the same count, and ranks first of 21 — but the
absolute gain (~0.5%/yr) is not distinguishable from zero after costs. The
random screens lose −0.00047 per 20d to turnover alone, which is most of the
edge. Closed; no alternative K or cost assumption is tried.

### E0 — NOT AT PARITY (2026-09-26 00:00 IST)

MPS retrain `r4_pit_long_mps`, 74 min (pruned TCN, fp32 highest; panel
SHA256 verified identical to the CUDA run's). Report:
`logs/retrain/r4_pit_long_mps_20260925T1715Z.parity.txt`.

| criterion | CUDA `r4_pit_long` | MPS `r4_pit_long_mps` | |
|---|---|---|---|
| gate verdict | PASS | PASS | ✓ |
| 20d mean IC (window t, positive) | +0.0280 (4.19, 12/13) | +0.0252 (2.56, 10/13) | Δ −0.0028 < SE 0.0067 ✓ |
| 5d mean IC (window t) | +0.0240 (6.43) | +0.0217 (4.29) | |
| median per-date rank corr, r_hat_20d | — | **0.757** (p10 0.413, p90 0.922) | < 0.8 ✗ |
| top-30 overlap per date | — | median 57% | |

Per-window 20d IC correlates +0.895 across the 13 windows: statistically the
same signal. On any given day the two rank names differently. Verdict as
pre-registered: **NOT AT PARITY.**

### E0b — pre-registered 2026-09-26 00:05, before running

The 0.8 bar in E0 was set with no reference for how much two ordinary
retrains disagree. E0b measures that floor: identical to E0 (MPS, pruned,
fp32 highest, same panel/config/windows) except **`seed=43`** instead of 42.
Tag `r4_pit_long_mps_s43`. Queued after E2.

**Reading, fixed now:** compare `r4_pit_long_mps` vs `r4_pit_long_mps_s43`.
- If their median per-date rank corr is **≤ 0.807** (within 0.05 of E0's
  0.757), the CUDA-vs-MPS difference is the size of ordinary seed-to-seed
  variation: the platform is not the cause, and the 0.8 bar was miscalibrated.
- If it is **≥ 0.85**, retrains on one platform agree much more closely than
  across platforms, and the platform itself matters for the 2027 refit.
- In between: inconclusive, reported as such.

E0's verdict is not changed by E0b under any outcome.

### E0 follow-ups — S6's structure replicates on the independent retrain

Recorded, not used to select anything (`audit/topk_gate/r4_pit_long_mps_*30.json`):

| | CUDA `r4_pit_long` (S6) | MPS `r4_pit_long_mps` |
|---|---|---|
| top-30, rank of 21 vs random | 12 (paired t −0.23) | **9** (paired t 0.10) — no information |
| bottom-30, window t / rank | −3.20 / 21 | **−3.07 / 21** — worse than every random book |
| holdout 20d IC, monthly blocks | +0.0459 (t 1.73) | +0.0338 (t 1.69); Δ t −0.81 |

A retrain on a different platform, with a different float path, that ranks
names differently day to day (median rank corr 0.757), reproduces S6 exactly:
nothing at the top, strong negative information at the bottom. The finding
is a property of this model class and objective, not of one training run —
which is also the motivation for E2.

### E2 — FAIL, all three criteria; closed (2026-09-26 02:13 IST)

`r4_pit_long_listnet`, identical to `r4_pit_long` except `loss=listnet`
(`audit/topk_gate/r4_pit_long_listnet_top30.json`, `logs/retrain/E2.*`):

| criterion | needed | result |
|---|---|---|
| top-30 vs 20 random books | paired t > 2.18 and rank ≤ 2 | t −1.02, **rank 21 of 21** ✗ |
| rank-IC gate | PASS | **FAIL** — 20d −0.0035 (t −0.29), 5d −0.0098 ✗ |
| top-30 net excess | > −0.00241 (MSE model) | −0.00573 ✗ |

No holdout look (rule 3). No second loss function (rule 1).

**Why — diagnosed, not tuned.** Daily returns are heavy-tailed, so a softmax
over z-scored targets puts most of its mass on a few extreme winners, and the
model learns to predict extremes — i.e. volatility:

| | rank corr(prediction, realized_vol_60d) | (prediction, beta) | top-30 mean vol percentile |
|---|---|---|---|
| MSE `r4_pit_long` | −0.162 | −0.232 | 58% |
| ListNet | **+0.140** | −0.137 | **71%** |

The listwise loss turned the model into a high-volatility picker, and high
volatility underperforms on this universe (the low-vol effect in `S3`). A
top-heavy listwise loss on heavy-tailed daily returns is a volatility bet in
disguise — worth knowing before anyone reaches for a ranking loss again. Any
future attempt would have to neutralise volatility in the target or the
book, which is a new hypothesis, not a retune of this one.
