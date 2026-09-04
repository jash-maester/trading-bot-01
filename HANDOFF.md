# HANDOFF — Machine transfer & current development state

**Written:** 2026-09-04
**From:** MacBook (`/Users/jash/Work/Trading_Bot`, Apple Silicon / MPS)
**To:** Mac mini (fresh clone)
**Repo:** `git@github.com:jash-maester/trading-bot-01.git`, branch `main`

This document exists because all Phase 1 / Phase 2 work described below was
sitting **uncommitted** on the MacBook. It is now committed and pushed, so a
fresh clone gets everything. Read this before writing any new code.

---

## 1. TL;DR — where the project actually is

| Item | State |
|---|---|
| Last milestone commit | `b7d5ac8` — M7 Integration (walk-forward training) |
| Work added on top | **Phase 1** (regime conditioning) + **Phase 2** (auxiliary return head) |
| Code complete | Yes — both phases fully wired end-to-end |
| Unit tests | **140/140 pass** (44 new: 10 regime-features, 16 FiLM, 18 aux-head) |
| `ruff check .` | Clean |
| `mypy src` | Clean (37 files) |
| **Trained / validated** | **NO.** Zero training runs exist for either phase. |
| Next action | Run the Phase 1 A/B walk-forward — see §6 |

> ⚠️ **The most important line in this document:** the architecture is built
> and unit-tested, but **no walk-forward experiment has been run on it yet**.
> `mlruns/` is empty. Any comment in a config calling something a "winner" is
> aspirational — it describes intent, not a measured result. Do not report
> Phase 1 as validated until §6 has actually been run.

---

## 2. The problem this work is solving

M7 walk-forward on the MLP baseline surfaced a pathology:

> **`corr(val_sharpe, test_sharpe) = −0.86`**

Validation performance was *anti*-predictive of test performance. The
diagnosis: the policy specialises to whichever market regime dominates the
train+val window, and the following year (test) tends to be in the opposite
regime — so the specialised policy inverts. Model selection on val was
actively picking the worst test performers.

The fix is not more regularisation. It is to give the model an **exogenous
regime signal** it can condition on, so that "what to do" becomes a function
of "what regime are we in" rather than being baked into the weights.

Two phases were designed against this:

- **Phase 1 — regime conditioning (FiLM).** Feed a market-regime vector into
  the network so behaviour is explicitly regime-dependent.
- **Phase 2 — auxiliary return prediction.** Add a dense supervised gradient
  so the encoder learns return-predictive representations instead of relying
  on PPO's sparse, noisy signal.

---

## 3. What was built

### 3.1 Phase 1 — market-regime conditioning

**New module: `src/trader/data/regime_features.py`** (272 lines)

Computes a 6-dimensional daily regime vector from the cross-section of stock
returns plus the tradeability mask:

| Feature | Meaning |
|---|---|
| `mkt_vol_20d` | Annualised std of equal-weight benchmark log-return, 20d |
| `mkt_breadth_20d` | Fraction of stocks with positive cumulative 20d return |
| `mkt_dispersion_20d` | Cross-sectional std of daily returns (do stocks disagree?) |
| `mkt_trend_20d` | Cumulative equal-weight log return, 20d |
| `mkt_acceleration` | `trend_20d − trend_60d` — regime-change indicator |
| `mkt_vol_of_vol_60d` | 60d rolling std of `mkt_vol_20d` — is the vol regime itself stable? |

Public API: `compute_regime_features`, `compute_regime_stats`,
`regime_stats_to_tensors`, `save_regime_stats`, `load_regime_stats`,
plus `REGIME_COLS` / `REGIME_DIM`.

**Leakage discipline:** row `t` uses returns through `t−1` only — information
known at the *start* of trading day `t`, matching the env's `features`
lookback convention. Warm-up rows are zero-filled. Normalisation stats are
computed **from the train panel only**, per walk-forward window, so no
val/test statistics leak backwards. This is the single most important
invariant in the new code — there are dedicated tests for it in
`tests/unit/test_regime_features.py`.

**New model components (`src/trader/models/encoders.py`):**

- `RegimeNormalizer` — frozen z-score for the `[B, R]` regime tensor. Mirrors
  `FeatureNormalizer`. Identity when stats are absent (tests / ablations).
- `FiLM` — Feature-wise Linear Modulation (Perez et al. 2018). Produces
  `(γ, β)` from the regime vector via a small MLP and applies
  `out = γ·x + β` to per-stock embeddings. The same `(γ, β)` applies to every
  stock, which is correct: regime is a market-wide property.

  Two design points worth preserving:
  - *Why FiLM, not concatenation?* Concat grows the input dim and forces every
    layer to relearn regime-dependence. FiLM is **multiplicative** — it gates
    *which features matter in which regime*, which is exactly the inductive
    bias the `−0.86` problem calls for.
  - *Identity init.* The final linear is zero-initialised, so `Δγ = Δβ = 0`
    and `γ = 1` at step 0. **At initialisation the model is bit-identical to
    the non-FiLM baseline**, and only gradually learns modulation. This makes
    the A/B in §6 a clean test of the conditioning, not of a different init.

**Wiring (`src/trader/models/actor_critic.py`):**

```
features → TCN encoder → [FiLM?] → CrossStockAttention → [FiLM?] → actor
                                                                 → critic (+regime concat?)
                                                                 → [aux head?]
```

Three independently toggleable insertion points, so each can be ablated:

| Flag | Effect |
|---|---|
| `model.regime_film_encoder` | FiLM after the TCN encoder |
| `model.regime_film_attn` | FiLM after CrossStockAttention |
| `model.regime_in_critic` | Raw regime concatenated into the critic input |

**With all three false, the model is bit-exact identical to pre-Phase-1.**
`regime_dim` is auto-detected from the obs space in `ActorCritic.from_obs_space`.

**Env (`src/trader/env/panel_env.py`):** the observation dict gained a
`regime` key (shape `[R]`). It is *always* emitted — models that don't use it
just ignore the key.

### 3.2 Phase 2 — auxiliary next-day return prediction

**`ReturnPredictionHead` (`src/trader/models/heads.py`)** — a small per-stock
MLP over the same embeddings `z` used by actor/critic, predicting next-day
per-stock log returns. Trained jointly with PPO via masked MSE
(`aux_return_loss`, tradeable stocks only).

Rationale, since it drove the design:
- PPO's **value loss** gives one scalar per batch element, so gradient
  pressure on *per-stock* representations is weak (mediated by `mean(z, dim=1)`).
- PPO's **policy loss** is sparse and noisy — advantage reflects the combined
  effect of all decisions, not per-stock truth.
- **This head** supervises every per-stock embedding directly against a target
  known one step ahead. (Cf. UNREAL, Jaderberg et al. 2017.)

Output layer is small-initialised (gain 0.01) so initial predictions are
O(1e-3) — the same order as real daily log returns. Without this the head's
initial MSE swamps the PPO losses for the first few hundred updates.

**Env:** obs gained `next_day_returns` (shape `[N]`). **This is a target, not
an input.** The policy/value forward path never reads it — doing so would be
a look-ahead leak. Only `PPOTrainer` consumes it, in the update step.

**PPO (`src/trader/training/ppo.py`):** new `aux_return_loss_coef` (default
`0.0`). The aux path activates only when *all three* hold: coefficient > 0,
the model has the head, and the obs buffer carries the target. Rollout uses
the cheap `get_action_and_value`; only the update step calls
`get_action_value_and_aux`, so there is no rollout cost. Logged to MLflow as
`losses/aux_return_mse`.

### 3.3 Also included

- **`torch.compile` support** — `model.compile: true`. Note the subtlety
  documented in `runner.py`: we rebind `model.forward` to its compiled version
  rather than replacing the module, because PPO's hot path calls
  `self.forward(obs)` internally, which would otherwise bypass the compiled
  graph. Wrapped in try/except → falls back to eager. Expect ~1.3–2× on CUDA,
  ~1.0–1.2× on MPS.
- **`Makefile`** — `sync` rsync source fixed from `./*` to `./` (the glob
  skipped dotfiles).
- **`pyproject.toml`** — trimmed a stale comment block; `torch-backend = "auto"`
  is unchanged and still correct for both MPS and CUDA.
- **mypy cleanup** — 7 pre-existing type errors in the new code fixed
  (dict-splat inference in `FiLM` construction, two stale `type: ignore`s,
  one over-broad ignore code).

---

## 4. File map

**New files**

| File | Lines | Purpose |
|---|---:|---|
| `src/trader/data/regime_features.py` | 272 | Regime feature computation + stats |
| `configs/model/mlp_regime.yaml` | 38 | Phase 1 model config |
| `configs/model/mlp_regime_aux.yaml` | 36 | Phase 2 (Phase 1 + aux head) |
| `configs/train/ppo_aux.yaml` | 26 | PPO with `aux_return_loss_coef: 0.1` |
| `tests/unit/test_regime_features.py` | 286 | Regime features, incl. leakage tests |
| `tests/unit/test_regime_film.py` | 342 | FiLM correctness + identity-init |
| `tests/unit/test_aux_return_head.py` | 436 | Aux head + masked loss |

**Modified files**

| File | Change |
|---|---|
| `src/trader/models/actor_critic.py` | FiLM wiring, `_forward_shared`, `get_action_value_and_aux`, regime auto-detect |
| `src/trader/models/encoders.py` | `RegimeNormalizer`, `FiLM` |
| `src/trader/models/heads.py` | `ReturnPredictionHead`, `aux_return_loss`, critic regime concat |
| `src/trader/training/runner.py` | Regime stats, model-cfg plumbing, `torch.compile` |
| `src/trader/training/ppo.py` | `aux_return_loss_coef`, aux loss in update, MLflow logging |
| `src/trader/env/panel_env.py` | `regime` + `next_day_returns` obs keys |
| `configs/config.yaml` | Default model → `mlp_regime` |
| `configs/model/mlp_baseline.yaml` | Explicit `false` flags (self-documenting) |
| `configs/model/gnn_v1.yaml`, `gnn_intra_only.yaml` | No-op flags for Hydra override compat |
| `configs/train/ppo_baseline.yaml`, `ppo_gnn.yaml` | `aux_return_loss_coef: 0.0` |
| `Makefile`, `pyproject.toml` | See §3.3 |

---

## 5. Mac mini setup — from zero to training

### 5.1 Clone

```bash
git clone git@github.com:jash-maester/trading-bot-01.git ~/Work/Trading_Bot && cd ~/Work/Trading_Bot
```

(HTTPS instead, if SSH keys aren't set up on the Mac mini:
`git clone https://github.com/jash-maester/trading-bot-01.git ~/Work/Trading_Bot`)

### 5.2 Install

```bash
uv sync
```

`torch-backend = "auto"` picks MPS on Apple Silicon. No `uv.toml` needed
unless you later move to the CUDA box.

### 5.3 Verify the transfer landed intact

```bash
uv run ruff check . && uv run mypy src && uv run pytest tests/unit/ -q
```

Expected: `All checks passed!` / `Success: no issues found in 37 source files` /
**`140 passed`**. If you see 96, the Phase 1/2 files did not come across.

### 5.4 ⚠️ Rebuild the data — it is NOT in git

`data/` is ~95 MB and gitignored (`*.parquet`, `data/raw/`, `data/ohlcv/`).
**A fresh clone has no data.** Either rebuild it (~15–25 min, mostly the
Yahoo Finance download) or rsync `data/` over from the MacBook.

```bash
make db-up          # postgres:16 on 5432, MLflow on 5555
make db-migrate     # once

uv run python scripts/build_universe.py data=universe_v1    # ~5s
uv run python scripts/fetch_data.py     data=universe_v1    # ~10-20 min
uv run python scripts/build_features.py data=universe_v1    # ~2-3 min
```

Produces `data/panels/{train,val,test}.parquet` with purge gaps
(train 2014→2021, val 2022-02→2022-12, test 2023-02→present).

> **macOS note:** MLflow uses port **5555**, not 5000 — port 5000 is taken by
> AirPlay Receiver. Override with `MLFLOW_PORT=<port> make db-up`.

Faster alternative — copy the data straight across from the MacBook:

```bash
rsync -avz --progress <macbook-user>@<macbook-host>.local:~/Work/Trading_Bot/data/ ~/Work/Trading_Bot/data/
```

---

## 6. Next action — the experiment that has not been run

Phase 1 needs a head-to-head walk-forward against the baseline. `mlp_regime`
shares every hyperparameter with `mlp_baseline`, so the comparison isolates
the conditioning effect rather than a parameter-budget difference.

```bash
# Control
uv run python scripts/walk_forward.py model=mlp_baseline seed=42

# Phase 1
uv run python scripts/walk_forward.py model=mlp_regime   seed=42
```

**The metric that matters is not Sharpe.** It is `corr(val, test)` across
walk-forward windows. Baseline is `−0.86`. Phase 1 succeeds if that moves
meaningfully toward zero or positive — i.e. validation becomes usable for
model selection again. A Phase 1 run with a better mean Sharpe but a still
strongly negative correlation has **not** solved the problem.

Run multiple seeds before concluding anything; single-seed walk-forward
differences on this setup are well within noise.

Then the ablation, to find which insertion point is doing the work:

```bash
uv run python scripts/walk_forward.py model=mlp_regime model.regime_film_attn=false
uv run python scripts/walk_forward.py model=mlp_regime model.regime_in_critic=false
uv run python scripts/walk_forward.py model=mlp_regime model.regime_film_encoder=false
```

Only if Phase 1 shows signal, move to Phase 2 (both flags are required —
the head is built by the model config, the loss weight by the train config):

```bash
uv run python scripts/walk_forward.py model=mlp_regime_aux train=ppo_aux
# sweep: train.aux_return_loss_coef={0.05,0.1,0.2}
```

---

## 7. Gotchas for whoever picks this up

1. **Two flags are needed for Phase 2.** `model.use_aux_return_head=true` builds
   the head; `train.aux_return_loss_coef>0` trains it. Setting only the first
   gives you dead parameters and no error message. `configs/train/ppo_aux.yaml`
   pairs them correctly.
2. **`next_day_returns` is a label, never an input.** It lives in the obs dict
   purely as transport to the trainer. If you ever find yourself reading it in
   a forward path, you have introduced look-ahead bias.
3. **Regime stats are per walk-forward window, train-split only.** Reusing one
   global stat set across windows would leak.
4. **The GNN configs carry no-op regime flags.** Regime conditioning is *not*
   implemented for `GNNActorCritic` — the flags exist only so generic Hydra
   overrides don't fail. Wiring FiLM into the GNN path is open work.
5. **`compile: true` is on in both new model configs.** If you hit odd
   recompilation stalls or want clean stack traces while debugging, set
   `model.compile=false` first.
6. **Default model changed** in `configs/config.yaml`: `mlp_baseline` →
   `mlp_regime`. Bare `scripts/train.py` no longer runs the baseline. Pass
   `model=mlp_baseline` explicitly for the control.
---

## 8. Environment notes

- **Development machine:** macOS / Apple Silicon, MPS backend.
- **Intended training machine:** RTX 5090 (Blackwell GB202), CUDA 13.0,
  driver 595 — see `make sync` / `make sync-data` for the rsync deploy path.
- **Python 3.12**, `uv` for dependency management.
- MLflow on **5555** (AirPlay owns 5000 on macOS).
