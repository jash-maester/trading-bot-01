# Cross-sectional input normalisation: a refuted hypothesis

R4's target is a per-date cross-sectional z-score. Its inputs were standardised
by one global `(mean, std)` per feature, frozen over the whole train split. The
model was asked a relative question and handed absolute numbers. Giving it the
ranking operation on its inputs made it **worse on both spans**, and the gate
fails.

```
RUN_TAG=xs bash scripts/pit_xs_chain.sh          # train, split, holdout, compare
uv run python scripts/signal_feature_diagnostic.py --tag r4_pit_long
SIGNAL=r4_pit_xs BASE=r4_pit_long uv run python scripts/xs_head_to_head.py
```

Parity: same config file, same panel, same 13 windows, same seed as
`r4_pit_long`. Exactly two overrides — the tag and `train.xs_normalise=rank`.

---

## The hypothesis, and why it was worth a run

On `r4_pit_long`'s own OOS rows at 20d, blocked by year, identical dates and
names (`scripts/signal_feature_diagnostic.py`):

| signal | IC | t | years up |
|---|---|---|---|
| trained 15-feature model | +0.0275 | 2.75 | 11/14 |
| **−`realized_vol_60d`, ranked** | **+0.0535** | 3.30 | 12/14 |
| −`realized_vol_20d`, ranked | +0.0462 | — | 12/14 |
| naive equal-weight composite of all 15 | +0.0066 | 0.51 | 3/6 |

A cross-sectional rank of one raw feature, nothing fitted, roughly doubled the
trained model. The model beat a *naive* combination (+0.0275 vs +0.0066), so it
was doing real work — it simply could not reach what one ranked feature got.
The architectural reading was that the ranking operation was missing, and
`cross_sectional_normalise` supplied it: van der Waerden scores within each
date's tradeable cross-section.

## The result: worse, on both spans

**Signal gate, 13 windows 2005–2024** (`data/signal/*/gate.json`):

| | base `r4_pit_long` | `r4_pit_xs` |
|---|---|---|
| 5d mean IC | +0.0240, t 6.43, 12/13 | +0.0169, t 3.72, 11/13 |
| 20d mean IC | +0.0280, t 4.19, 12/13 | +0.0131, t 1.52, 11/13 |
| verdict | **PASS** | **FAIL** |

**Head to head on identical rows** (`scripts/xs_head_to_head.py`):

| span | horizon | base | xs | difference | blocks favouring xs |
|---|---|---|---|---|---|
| in-sample, 14 years | 5d | +0.0247 | +0.0182 | −0.0065 (t −1.34) | 6/14 |
| in-sample, 14 years | 20d | +0.0275 | +0.0138 | −0.0138 (t −1.08) | 7/14 |
| **holdout, 15 months** | 5d | +0.0254 | +0.0039 | −0.0215 (t −1.49) | 7/15 |
| **holdout, 15 months** | 20d | **+0.0459** | **+0.0117** | **−0.0342** (t −1.19) | **3/15** |

**Stated honestly: the paired differences are not significant.** At t −1.08 to
−1.49 the two models are not statistically distinguishable, and a reader should
not be told they are. What decides this is the gate — R4's actual criterion —
which the base passes and this fails, and the consistency of the sign: on the
holdout at 20d only **3 of 15 months** favour the new model.

## Where it broke, which points at the mechanism

Per-window 20d IC:

| | W5 | W6 | W7 | W8 | W9 |
|---|---|---|---|---|---|
| base | +0.0632 | +0.0358 | +0.0608 | +0.0526 | +0.0283 |
| xs | +0.0329 | +0.0056 | +0.0029 | +0.0227 | +0.0083 |

The damage is concentrated exactly where the base model was **strongest**
(W5–W9, tests spanning 2015–2020). A change that merely added noise would not
target the best windows; one that removed information the model was relying on
would.

**The likely mechanism — a hypothesis, not a measurement.** The encoder is a
TCN reading a 60-day trajectory per stock. Ranking each day independently
across the cross-section destroys the time axis within that window:
`log_return_1d`'s cross-sectional rank is close to white noise day over day,
where its raw value carries the momentum structure a convolution exists to
read. The fix addressed the representation the *target* needed and broke the
one the *encoder* needed. Testing it properly would mean normalising only the
slow-moving features (`realized_vol_60d`, `dollar_volume_20`) and leaving the
return series raw — not attempted here.

## What is now established, and worth more than the failure

**1. The model is not short of cross-sectional context in its inputs.** That
was the hypothesis and it is refuted. Whatever limits R4, this is not it.

**2. The "one feature beats the model" gap is in-sample only, and reverses.**
Measured on the holdout with identical rows and blocking:

| holdout, 20d | IC | t | months up |
|---|---|---|---|
| `r4_pit_long` | **+0.0459** | 1.73 | 11/15 |
| −`realized_vol_60d` | +0.0312 | 0.72 | 10/15 |

In sample the feature beats the model roughly 2:1; out of sample the model
beats the feature. The motivating gap was largely an artefact of the span it
was drawn from — low-vol had exceptional years inside the training window
(2018 +0.1616, 2019 +0.1292) and does not repeat them. This is the same
in-sample/out-of-sample reversal recorded in `audit/R5_VERDICT_PIT.md` §3,
appearing again one layer down, and it was not anticipated before the run.

**3. R4's signal generalises better than any single factor tried against it.**
+0.0280 in sample and +0.0459 on unseen data at 20d. That is the most
reassuring number in this document and it belongs to the model that was
already there.

## Prediction, recorded before the run

> "I expect 20d IC to rise materially from +0.0275 — toward the 0.04s if the
> hypothesis is right. If it lands within noise of +0.0275, the hypothesis is
> wrong and I'll say so plainly. The real risk is that per-date normalisation
> also strips market-level information the model was genuinely using, in which
> case this could come out *worse*."

It came out worse: +0.0131 on the gate, +0.0117 on the holdout. The stated
risk was the outcome, so the run was correctly specified and the hypothesis was
simply wrong.

## Status

**Not adopted.** `train.xs_normalise` stays in the codebase, defaulting to
`null`, with this result recorded against it so the option is not retried
blind. `r4_pit_long` remains the signal of record. No allocator work follows
from this run — a signal that fails its own gate cannot feed R5 under
`CLAUDE.md` rule 1.

Artefacts: `data/signal/r4_pit_xs/gate.json`, `logs/xs_compare.log`.
