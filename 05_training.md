# 05 — Training, Evaluation, Walk-Forward

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

Key hyperparameters (starting point; hydra-configurable):

```yaml
# configs/train/ppo_gnn.yaml
total_steps: 2_000_000
num_envs: 16
rollout_length: 512
num_epochs: 4
minibatch_size: 4096
gamma: 0.995
gae_lambda: 0.95
clip_coef: 0.2
ent_coef: 0.01
vf_coef: 0.5
max_grad_norm: 0.5
learning_rate: 3.0e-4
lr_schedule: linear_decay
target_kl: 0.02
normalize_advantage: true
```

Reward scaling: divide by a running std estimate to keep gradients sane;
PopArt-style for the value head.

## Walk-forward protocol

Why walk-forward: a single train/val/test split picks up luck or
regime-specific artifacts. Walk-forward rotates the windows annually,
simulating how the system would actually be retrained in production.

```
Window 1:  train 2014..2019  val 2020  test 2021
Window 2:  train 2015..2020  val 2021  test 2022
Window 3:  train 2016..2021  val 2022  test 2023
Window 4:  train 2017..2022  val 2023  test 2024
(etc.)
```

Purge gap: 1 month between train/val and val/test to avoid overlap in
rolling-window features.

Each window trains from scratch OR warm-starts from the previous
window's final checkpoint (two configs; report both).

Final reported performance is the concatenation of the test segments
across all windows.

## Evaluation metrics

Computed by `trader.training.eval_metrics`, logged per episode and
aggregated per window:

- **CAGR** (net of costs).
- **Annualized Sharpe** (√252 * mean / std of daily log returns).
- **Sortino** (downside deviation).
- **Max drawdown**, **Calmar** (CAGR / |MDD|).
- **Turnover** (annualized, sum of |Δw|).
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

If the RL agent underperforms equal-weight after costs on the test
windows across multiple seeds, it is not working — debug before
changing architecture.

## Seeding and statistical reporting

- Each config runs with 5 seeds. Report **mean ± std** for every
  metric.
- Walk-forward: 4 windows × 5 seeds = 20 test segments. Use a paired
  bootstrap CI vs equal-weight to report significance.
- No cherry-picking. MLflow runs are the ledger; any figure in a
  writeup must link to a run.

## Overfitting guardrails

1. **Early stopping on val Sharpe** (patience = 10 evaluation rounds).
2. **Regularization**:
   - Entropy bonus (ent_coef).
   - Turnover penalty in the reward.
   - DropEdge on the GNN.
   - Dropout in heads.
3. **Sanity tests** that must pass during training:
   - Train-vs-test Sharpe gap < 1.5× (otherwise: flag as overfitting).
   - Turnover must not collapse to zero (cash-hoarding failure mode)
     nor explode past 20× annual turnover.
4. **Shuffled-label check** (optional but gold-standard): shuffle
   ticker identities in a held-out run; the agent should *not* be able
   to exceed chance. If it can, there's a data leak.

## Curriculum (optional)

- Start episodes in low-volatility periods, then expand. Only use if
  base training is unstable.
- More useful in practice: warm-start the policy head from **behavior
  cloning on the momentum baseline**, then fine-tune with PPO. Cuts
  early-training variance significantly.

## MLflow

- Params: full Hydra config dumped as artifact.
- Metrics: all eval metrics per epoch and per window.
- Artifacts: model checkpoints every N epochs, equity curves, attention
  weight snapshots from the GNN (for interpretability).
- Tags: `git_sha`, `data_version`, `universe_version`, `cost_model`,
  `seed`.

## Compute budget (ballpark on the 5090)

- 2M steps × 150 tickers × 60 lookback × 16 envs.
- Rollout is CPU-bound (env stepping) + GPU forward.
- GPU update is tiny (~150k params, minibatch 4096) — expect 1–2 ms.
- End-to-end: 3–6 hours per seed per walk-forward window. With 5 seeds
  × 4 windows, plan overnight runs per config.

## Acceptance criteria for Phase 4

- `make train CONFIG=train/ppo_baseline` trains a non-GNN baseline to
  near equal-weight performance on the validation window.
- `make train CONFIG=train/ppo_gnn` runs end-to-end over one walk-
  forward window in under 8 hours.
- 5 seeds per config, all logged to MLflow with tags.
- Evaluation script produces a one-page PDF report comparing RL agent
  vs baselines on the test windows.
- All overfitting guardrail tests pass (or their alerts are visible in
  MLflow).
