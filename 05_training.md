# 05 — Training, Evaluation, Walk-Forward

## Status

**PARTIAL** — last verified against `60ba1a6` (the A0/A1/A2 audit reports,
2026-09-05). PPO, walk-forward windowing, the metric set and the MLflow wiring
all exist and run. Nothing has been *validated*: M5's acceptance gate — *"matches
or beats `EqualWeightRebalanced` on the val split over 3 seeds"* — has never
passed, and M6, M7, Phase 1 and Phase 2 were built on top of it anyway.

Known broken:

- **The hyperparameter block below was stale** — `rollout_length: 512` and
  `ent_coef: 0.01` against an actual 256 and 1e-4. Corrected here.
- **`target_kl` is not configurable** (`ppo.py:68`), and it early-stops most
  epochs, so `n_epochs` is largely notional (A1 §7.12).
- **Dropout is inverted** — off during updates, on during val/test evaluation
  (`ppo.py:218`, `ppo.py:445`; A1 §7.1).
- **All five evaluation "episodes" are the same deterministic date range**, so
  `val_std_sharpe` measures action noise, not window variance (A1 §7.2).
- **The purge gap is 19–23 trading days against 60-day features** — 3 months is
  the minimum that clears (A1 §3).
- **The walk-forward "full panel" has two month-long holes** (A1 §7.3).
- **All four training entrypoints hardcode `data/panels`**, so the Kite panel is
  unreachable (A0 §3.2, A1 §7.4).
- **No `test_*` metric had ever been logged** before A1 ran one window (A1 §5.1).

Every RTX 5090 figure has been replaced with a measured RTX 4060 8 GB one.

---

## Algorithm

**PPO** with clipped surrogate, GAE, Gaussian policy over allocation
logits. Chosen because:
- Continuous action, stable, widely validated.
- Plays well with vectorized envs.
- Off-the-shelf implementations work; easy to audit.

SAC was considered and deprioritized: higher tuning burden with
continuous logits that are then softmaxed, and the replay buffer
doesn't help much when the env is cheap to re-rollout.

## Implementation

Single-file CleanRL-style PPO at `src/trader/training/ppo.py`, not a
heavy framework. Reasons:
- Easy to read, modify, and instrument for MLflow.
- No dependency on SB3's env assumptions (which occasionally bite
  custom envs).
- We still keep a SB3 sanity harness in `scripts/train_sb3.py` to
  cross-check the baseline.

Key hyperparameters, **as actually configured** in
`configs/train/ppo_baseline.yaml` (the previous version of this block claimed
`rollout_length: 512` and `ent_coef: 0.01`; both were wrong):

```yaml
# configs/train/ppo_baseline.yaml
total_steps: 8_000_000     # NOT 2M — 4x the workload every budget assumes
n_envs: 16
n_steps: 256               # rollout length; 16 x 256 = 4096 transitions/rollout
n_minibatches: 32          # -> minibatch of 128 transitions
n_epochs: 4                # -> 128 gradient steps per update
learning_rate: 3.0e-4
gamma: 0.995
gae_lambda: 0.95
clip_coef: 0.2
ent_coef: 0.0001           # NOT 0.01
vf_coef: 0.5
max_grad_norm: 0.5
anneal_lr: true            # linear decay
normalize_rewards: true    # RunningMeanStd scaling before GAE
aux_return_loss_coef: 0.0  # Phase 2 aux head; dead unless the model also opts in
```

Derived shapes at N=504 (A2 §1): 4,096 transitions per rollout, 128 gradient
steps per update, 488.28 updates for 2M env steps, 1,953 for the configured 8M.

> **DEFECT — `target_kl` is not configurable and is doing more than intended.**
> `PPOConfig.target_kl = 0.02` (`ppo.py:68`) is never populated from Hydra;
> `runner.py:277-299` omits it and no `configs/train/*.yaml` declares the key.
> In A1's walk-forward run it early-stopped the epoch loop (`ppo.py:404-406`) on
> essentially every update, so **`n_epochs: 4` is largely notional** (A1 §7.12).

Reward scaling: `normalize_rewards: true` divides by a running std estimate to
keep gradients sane. **PopArt for the value head is PLANNED, NOT IMPLEMENTED** —
no audit found it and no config key selects it. **UNVERIFIED** whether any
value-head normalisation exists.

## Walk-forward protocol

Why walk-forward: a single train/val/test split picks up luck or
regime-specific artifacts. Walk-forward rotates the windows annually,
simulating how the system would actually be retrained in production.

The windows `compute_windows()` actually produces from
`configs/walk/default.yaml` (`train_years: 5`, `val_months: 12`,
`test_months: 12`, `purge_months: 1`, `n_windows: 4`, `step_months: 12`) over
`data_start=2014-01-01, data_end=2024-12-30` — measured, A1 §3.2:

```
Window 1:  train ..2018-12-31   val 2019-02-01..2020-01-31   test 2020-03-01..
Window 2:  train ..2019-12-31   val 2020-02-01..2021-01-31   test 2021-03-01..
Window 3:  train ..2020-12-31   val 2021-02-01..2022-01-31   test 2022-03-01..
Window 4:  train ..2021-12-31   val 2022-02-01..2023-01-31   test 2023-03-01..
```

That is one year earlier than the block this document used to carry. Seeds are
`[42, 43, 44]` — **three**, not five (`configs/walk/default.yaml:11`).

> **DEFECT — the 1-month purge is a third of the feature lookback.** Measured
> across all four windows and both boundaries: **19–23 trading days** against
> 60-day `realized_vol_60d` and `beta_nifty_60d`, leaving 36–40 contaminated
> rows at **8 of 8** boundaries. **3 months (59–63 trading days) is the minimum
> that clears, and it clears W1 by exactly zero days; 4 months is the first
> value with margin.** At 6 months only three windows fit the panel. Full table
> and the EWM residual analysis in `02_data_pipeline.md` § Splits and A1 §3.
>
> Two knock-on effects specific to this document:
> - Evaluation always begins at panel index 60, and the first 60 val rows are
>   the ones that cannot be computed without pre-split history — so **36 of the
>   182 evaluated steps (19.8%) on W1/val** contain at least one contaminated
>   row (A1 §3.6).
> - `scripts/walk_forward.py:64-75` rebuilds the "full panel" from the
>   *already purged* splits, so the purge months are missing from disk: 41
>   trading days gone, two month-long holes (2022-01, 2023-01). W3/val and
>   W4/val silently end a month early; W2/test and W3/test carry a 20–21 day
>   hole mid-episode, across which the env treats rows either side as
>   consecutive sessions (A1 §7.3).
>
> `tests/unit/test_walk_forward.py:56-72` asserts a **calendar**-month gap of
> 28–32 days and never compares it to the feature window, so it passes on
> exactly the configuration that leaks.

Each window trains from scratch OR warm-starts from the previous
window's final checkpoint (two configs; report both).

> **DEFECT — walk-forward cannot see the Kite panel.**
> `scripts/walk_forward.py:59` hardcodes `orig_cwd / "data" / "panels"` and
> never reads `cfg.data.panels_root`, which `configs/data/kite_v1.yaml:19` sets
> to `data/panels_kite`. `data=kite_v1` changes nothing about which panel
> walk-forward trains on. The same hardcode is in `scripts/train.py:33` and
> `scripts/paper_run.py:266`; only `scripts/build_features.py:55` honours the
> key (A0 §3.2). **All four entrypoints must be changed together.**

> **DEFECT — `_rebuild_eval_model` is a hand-maintained copy of the model
> builder.** `walk_forward.py:351-453` duplicates `runner.py:174-223` and says
> so in its own `TODO`. `strict=True` on `load_state_dict` (`:492`) catches
> shape drift but not a silently different `num_sectors`, `dropout` or
> normaliser source (A1 §7.13).

Final reported performance is the concatenation of the test segments
across all windows.

## Evaluation metrics

Computed by `trader.training.eval_metrics`, logged per episode and
aggregated per window:

- **CAGR** (net of costs).
- **Annualized Sharpe** (√252 * mean / std of daily log returns).
- **Sortino** (downside deviation).
- **Max drawdown**, **Calmar** (CAGR / |MDD|).
- **Turnover** (annualized, sum of |Δw|). **DEFECT: this is not what the env
  reports.** `panel_env.py:286-288` computes both weight vectors from the same
  *post-trade* share vector, so `info["turnover"]` measures NAV drift, not
  trading — measured, a full book rotation reports 0.0027 while a hold on the
  next step reports 0.0045 (see `03_environment.md` § Reward). Every turnover
  figure this project has reported, and the `turnover_penalty` term in the
  reward, derive from it.
- **Hit rate** (% of months beating NIFTY 50).
- **Alpha, beta vs NIFTY 50** (OLS on daily excess returns).
- **Average sector concentration (HHI)**.
- **Cost drag** (gross CAGR − net CAGR).
- **Information ratio vs equal-weight baseline**.

Plots saved to MLflow per run:
- Equity curves (agent, 4 baselines).
- Underwater (drawdown) curves.
- Weight heatmap over time.
- Sector exposure over time.
- Rolling 252-day Sharpe.

## Baselines — must be run before any RL reporting

Baselines from `env/baselines.py` are evaluated on the identical test
window, with identical costs. The RL agent's result is only meaningful
relative to these. Expected ordering in a healthy run:

```
RandomPolicy  <  60/40  <  EqualWeight  <  NIFTY50  ≲  Momentum  ≲  RLAgent
```

**Two of the six terms in that ordering are not what they are labelled**, so the
prediction is untestable as written:

- `EqualWeight` rebalances **every step** (`baselines.py:57-62`), not monthly,
  so it pays the full turnover drag rather than a fraction of it.
- `BuyAndHoldIndex` holds **equal weight across the whole universe from day 1**
  (`baselines.py:65-81`) — it tracks no index, so it is not a proxy for
  `NIFTY50` in the ordering above.
- `Momentum` is `MomentumTopK` at **K=5** ranking on **`log_return_1d`**
  (`baselines.py:91,110`) — a half-cash portfolio driven by a one-day reversal
  signal, not 20-day momentum. See `03_environment.md` § Baselines for both
  defects and the measured cash fractions.

Also note the ordering's premise. On a 504-name midcap-tilted NSE universe,
equal-weight beating the Nifty 50 is the likelier outcome, and `ARCHITECTURE.md`
reports equal-weight at 16.4% CAGR. `EqualWeight < NIFTY50` should not be
assumed.

**The benchmark set the project should report against** (`09_revamp_and_audit.md`
§4.3): Nifty 50 TRI, Nifty 500 TRI, equal-weight universe, and momentum top-20
monthly. The momentum baseline is the one that matters — if it wins, the RL is
contributing nothing.

If the RL agent underperforms equal-weight after costs on the test
windows across multiple seeds, it is not working — debug before
changing architecture. **It has underperformed, and the debugging did not
happen:** M5's gate never passed, and M6/M7/Phase 1/Phase 2 were built anyway.
The most complete stored run (`c39b8f5`, 4,096 steps) reports mean Sharpe
**−1.259**, mean CAGR **−16.1%**, mean max drawdown **−26.9%**; the longest
(500,000 steps) reports mean Sharpe **−0.963**, mean CAGR **−10.2%** (A0 §4.2).

## Seeding and statistical reporting

- Each config runs with 5 seeds. Report **mean ± std** for every
  metric. **As configured it is 3** (`configs/walk/default.yaml:11`,
  `seeds: [42, 43, 44]`), and every stored run is seed 42.
- Walk-forward: 4 windows × 5 seeds = 20 test segments. **As configured it is
  4 × 3 = 12**, and zero have been run to completion. Use a paired bootstrap CI
  vs equal-weight to report significance — with a minimum-n guard, which
  `walk_forward.py:684-717` does not have.
- No cherry-picking. MLflow runs are the ledger; any figure in a
  writeup must link to a run.

> **DEFECT — the five evaluation "episodes" are one window sampled five times.**
> `panel_env.py:75-88` clamps `episode_length` to `len(dates) − lookback − 1`
> on a short panel, and `:207-211` then draws the start index from
> `[lookback, max(lookback, len(dates) − episode_length − 1)]` — after the clamp
> both bounds are equal, so the draw is **deterministic**. Measured (A1 §7.2):
> W1/val and W1/test both give five reset start indices of `[60,60,60,60,60]`.
> `n_episodes: int = 5` (`runner.py:405`) therefore buys five draws of the
> policy's *action noise* on one identical date range. `val_std_sharpe` and
> `test_std_sharpe` measure sampling noise, and `paired_bootstrap_ci`
> (`walk_forward.py:684-717`) is pairing single-window point estimates with no
> minimum-n guard — at n=1 it returned a zero-width "95% CI" reported as a
> significance result. Training episodes are unaffected (five distinct starts).

> **DEFECT — evaluation and training use different envs.**
> `runner.py:412-420` builds the eval env with only `lookback`,
> `episode_length` and `initial_cash`, dropping `reward_fn`, `turnover_penalty`
> and `use_excess_returns` that `runner.py:150-160` did set.
> `walk_forward.py:129-141` mirrors this deliberately so the baseline arm stays
> comparable. NAV-derived metrics are unaffected, but any experiment varying
> `env.reward` or `env.turnover_penalty` **is evaluated under the defaults**
> (A1 §7.10).

> **The decision metric has no artefact.** `corr(val_sharpe, test_sharpe) =
> −0.86` — the number `HANDOFF.md` §2, `09_revamp_and_audit.md` §3 and
> `scripts/walk_forward.py:21` all treat as decisive — is **unreproducible in
> this repository**. Before A1's run, no `test_*` metric had ever been logged,
> `data/walks/` was empty, and no `window` or `seed` tag existed (A1 §5.1).
> Separately, on n=4 windows the Fisher-transform 95% CI for r=−0.86 is
> (−0.997, +0.583) — it **includes zero** (`09_revamp_and_audit.md` §3).

## Overfitting guardrails

1. **Early stopping on val Sharpe** (patience = 10 evaluation rounds).
2. **Regularization**:
   - Entropy bonus (`ent_coef: 0.0001`, lowered from 0.001 because entropy crept
     the policy σ from 0.37 to 0.68).
   - Turnover penalty in the reward — **but see the turnover defect above; the
     term penalises NAV volatility, not trading.**
   - DropEdge on the GNN — **dead, see below.**
   - Dropout in heads — **dead, see below.**

   > **DEFECT — every declared regulariser is off during training and on during
   > evaluation.** `ppo.py:218` calls `self.model.eval()` at the top of every
   > update iteration, before the rollout, to make `log_prob_old` deterministic.
   > `ppo.py:445` calls `self.model.train()` only **after** the `for update`
   > loop. Nothing switches back in between, so the gradient updates at
   > `ppo.py:342-398` run in eval mode: `tcn.dropout: 0.1`, `graph.dropout: 0.1`
   > and `graph.drop_edge_prob: 0.1` never take effect.
   >
   > The mirror image is worse. `_evaluate_split` (`runner.py:397-440`) is called
   > at `runner.py:359` and `:371`, after `trainer.train()` returned and left the
   > model in **train** mode, and never calls `model.eval()` — so **every
   > reported val and test metric is measured with dropout active**. Measured
   > spread over 8 identical forwards: 1.6e-02 with dropout on, exactly 0.0 with
   > it off (A1 §7.1). `paper_run.py:209` and `walk_forward.py:492` (the
   > shuffled arm) *do* call `eval()`, so the shuffle check and the real arm are
   > compared under different stochasticity.
3. **Sanity tests** that must pass during training:
   - Train-vs-test Sharpe gap < 1.5× (otherwise: flag as overfitting).
   - Turnover must not collapse to zero (cash-hoarding failure mode)
     nor explode past 20× annual turnover.
4. **Shuffled-label check** (optional but gold-standard): shuffle
   ticker identities in a held-out run; the agent should *not* be able
   to exceed chance. If it can, there's a data leak. Wired at
   `configs/walk/default.yaml:20` (`shuffle_check: false`).

   **It had never been run until A1 ran it** (A1 §5). One window, one seed,
   10,240 env steps:

   | arm | test `mean_sharpe` |
   |---|---:|
   | agent, real ticker labels | 3.1447 |
   | agent, **shuffled** ticker labels | 3.1597 |
   | `EqualWeightRebalanced`, same window | 3.9577 |

   Red flag (b) — shuffled ≫ equal-weight — does **not** fire: no leak is
   reaching the policy through a channel that survives relabelling, which
   matches the call-graph result that `next_day_returns` has exactly one
   consumer (`ppo.py:387-388`, A1 §1). Red flag (a) — shuffled ≈ real — *does*
   fire, but at 10,240 steps the policy is near-initial and allocates near
   uniformly, so that is a property of the budget, not a finding. The two arms
   are also not evaluated under the same conditions (the dropout defect above).
   **The check now runs end-to-end and cannot produce meaningful evidence either
   way until the dropout and single-window defects are fixed and it is run on a
   trained policy over ≥3 window×seed pairs.**

   The command documented at `scripts/walk_forward.py:15-17` (`+walk.shuffle_check=true`)
   **fails** — the key has existed since `configs/walk/default.yaml:20`, so
   Hydra rejects the `+` prefix. Use `walk.shuffle_check=true` (A1 §7.5).

## Curriculum (optional)

- Start episodes in low-volatility periods, then expand. Only use if
  base training is unstable.
- More useful in practice: warm-start the policy head from **behavior
  cloning on the momentum baseline**, then fine-tune with PPO. Cuts
  early-training variance significantly.

## MLflow

MLflow runs on **port 5555**, not 5000 — AirPlay owns 5000 on macOS. Backing
store `mlruns/mlflow.db`.

- Params: full Hydra config dumped as artifact.
- Metrics: all eval metrics per epoch and per window.
- Artifacts: model checkpoints every N epochs, equity curves, attention
  weight snapshots from the GNN (for interpretability).
- Tags: `git_sha`, `data_version`, `universe_version`, `cost_model`,
  `seed`.

> **DEFECT — run provenance is not recoverable from MLflow.** All five stored
> runs log exactly **35 params** — `model.*`, `train.*`, `seed`, `device`,
> `n_params`, `n_tickers`. There is **no `env.*` param, no `data.*` param, and
> no panel path or panel SHA256** on any run, and the only tags present are
> `mlflow.runName` / `mlflow.source.*` / `mlflow.user` — **none of the five tags
> listed above exists** (A0 §4.2, A1 §5.1). Which panel a run saw can only be
> inferred from `n_tickers = 163` and the timestamp.
>
> `runner.py:327-336` does now log `env.reward_fn_resolved` and
> `env.use_excess_returns_resolved`, but every stored run predates that code.
>
> All five runs also **predate both correctness fixes**: `9019892` (costs, tax)
> and `a5d8b79` (beta) were committed at 16:50 and 16:51 on 2026-09-04; the
> latest run started at 15:28. Every stored run therefore carries the
> 22%-understated cost model, zero capital-gains tax, and `beta_nifty_60d`
> constant 1.0. **Treat all five as void.**
>
> All eleven Hydra run logs under `outputs/2026-09-04/` are **0 bytes** —
> Loguru is not attached to Hydra's file sink, so no run has produced a durable
> text log (A0 §4.3).

## Compute budget — RTX 4060 8 GB, measured

The RTX 5090 this section used to budget against does not exist. Development is
on an M4 Mac mini (24 GB unified, MPS); the target is an **RTX 4060 Laptop,
8,188 MiB, 24 SM**. Numbers from A2.

**Where the time goes** (instrumented, N=504, minibatch 128; proportions only —
see A2 §0.1):

| Phase | % of wall |
|---|---:|
| update forward | 49.2 |
| backward | 41.3 |
| rollout forward | 8.7 |
| optimiser (`Adam.step` + `clip_grad_norm_`) | 0.013 |
| rollout-buffer flatten (`_batch_obs`) | 0.007 |
| **env stepping** | **0.003** |
| host↔device transfers | 0.0001 |

Forward+backward is **99.2%**. The three things this document used to assume
were expensive — env stepping, the optimiser, data movement — are together
0.02%. *"Rollout is CPU-bound (env stepping)"* and *"GPU update is tiny"* were
both wrong, and in the same direction: **the observation, not the parameter
count, sets the price.**

**The arithmetic** (A2 §6.1): 7.83 MFLOP per stock-sequence × 64,512 sequences
per gradient step → 1,549 GFLOP fwd+bwd per gradient step → **96.8 PFLOP for 2M
env steps**, of which the TCN encoder is **97.9%**. The configured
`total_steps: 8_000_000` is 387 PFLOP.

**The hardware** (A2 §3, measured on the card): FP32 matmul peak **9.05
TFLOPS**. The ~15 TFLOPS figure used in earlier planning is the **TF32** number
(15.34); BF16 peaks at 30.96. At the shapes the TCN actually runs, the card
sustains 1.30 TFLOPS on the 15-channel input conv and 2.93 TFLOPS on the 64→64
inner convs, which are 94.1% of the MACs.

| Reference rate | 2M steps takes |
|---|---|
| `Conv1d(64→64,k=3)` at production shape, FP32 — a generous ceiling for this graph | **9.2 h** |
| FP32 matmul peak, 100% utilisation | **3.0 h** |
| TF32 matmul peak, 100% utilisation | 1.75 h |

So **"2M steps in under 2 hours" is below its own theoretical floor** on the
current graph, and no amount of engineering on data movement changes that —
the whole PCIe bill is 41.7 GiB and ~5 s per update, 40 minutes per 2M-step
run (A2 §4.2). The gate is reachable only by *removing* work: caching the TCN
encoder cuts 96.8 PFLOP to 2.06 (A2 §10).

**Memory is the binding constraint, not VRAM capacity per se.**

| | measured |
|---|---|
| VRAM at N=504 / L=60 / mb=128 | **7,762–7,766 MiB of 8,188** — ~95%, at 34 W with memory-controller utilisation 1–4% |
| host rollout buffer at N=504 | 6.95 GiB |
| host RSS during update | **15.83 GiB**; peak during the buffer flatten ≈20.8 GiB |
| system RAM required | **≥24 GiB** at 504 tickers — not the ≥16 GiB previously stated. 645 tickers does not fit in the box's 29 GiB WSL allowance |

Minibatch 256 at 504 tickers cannot fit, and neither can 645 tickers at
minibatch 128.

**NOT MEASURED on the 4060** (A2 §0.2 — the box refused SSH mid-audit):
uninstrumented wall clock per gradient step, achieved FLOPS, the kernel table,
the scaling curves and the OOM ceiling grid. `scripts/profiling/a2_all.sh` runs
all of them unattended. **Until those exist, no wall-clock claim about the 4060
is admissible in this document.**

**Also not on the critical path any more:** `runner.py:244-265` rebinds
`model.forward` to its compiled version on the grounds that
`get_action_and_value` calls `self.forward`. Phase 1 introduced
`_forward_shared`, which `get_action_and_value` now calls directly, so for
`ActorCritic` `torch.compile` reaches only the once-per-rollout bootstrap value
call (A1 §7.9). `configs/model/mlp_regime.yaml:38` still sets `compile: true`.

## Acceptance criteria for Phase 4

| Criterion | State |
|---|---|
| `make train CONFIG=train/ppo_baseline` trains a non-GNN baseline to near equal-weight performance on the validation window | **FAILS.** This is M5's gate and it has never passed. Best stored result: mean Sharpe −1.259 / CAGR −16.1% at 4,096 steps; −0.963 / −10.2% at 500,000 steps (A0 §4.2). Both against an `EqualWeightRebalanced` that is itself mis-specified |
| `make train CONFIG=train/ppo_gnn` runs end-to-end over one walk-forward window in under 8 hours | **NOT MEASURED on the 4060** (A2 §0.2). Restate it against 96.8 PFLOP / 9.05 TFLOPS: 2M steps has a 3.0 h floor at 100% FP32 utilisation and a 9.2 h realistic figure at the 2.93 TFLOPS this graph sustains. A one-window budget must be derived from those, not asserted |
| 5 seeds per config, all logged to MLflow with tags | **FAILS** — 3 seeds configured, 1 seed used, and none of the five specified tags is logged (A0 §4.2) |
| Evaluation script produces a one-page PDF report comparing RL agent vs baselines on the test windows | **UNVERIFIED** — no audit exercised it |
| All overfitting guardrail tests pass (or their alerts are visible in MLflow) | **FAILS** — the regularisers those guardrails assume are inactive during training (dropout defect above), and the val/test gap they measure is computed with dropout on |

Criteria this phase should have had:

- **The purge between any two segments is ≥ the longest feature window in
  trading days.**
- **Val and test evaluation runs in `eval()` mode.**
- **The five evaluation episodes have ≥2 distinct start indices.**
- **A `test_*` metric exists in MLflow before any val→test correlation is
  quoted.**
