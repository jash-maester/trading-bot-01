# HANDOFF — Machine transfer & current development state

**Written:** 2026-09-04
**Updated:** 2026-09-04 — transfer completed and verified on the Mac mini.
**From:** MacBook (`/Users/jash/Work/Trading_Bot`, Apple Silicon / MPS)
**To:** Mac mini — **`/Users/jash/storage/trading-bot-01`** (M4, 24 GB, macOS 26.6.2)
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

> **Status: DONE.** This section is a record of the completed transfer, not a
> to-do list. Everything below was executed on the Mac mini on 2026-09-04 and
> verified. Re-read it only if you are setting up a *third* machine.

### 5.0 Where the code actually lives

| | Path |
|---|---|
| MacBook (source) | `/Users/jash/Work/Trading_Bot` |
| **Mac mini (current)** | **`/Users/jash/storage/trading-bot-01`** |

The transfer was a **direct copy of the working tree**, not a fresh `git clone`
— so `data/` (96 MB, gitignored) came across with it and did **not** need
rebuilding. Every path in the configs is repo-relative (`data/ohlcv`,
`data/panels`, …), so nothing needed editing for the new location.

Mac mini hardware: **Apple M4, 24 GB unified memory, macOS 26.6.2, arm64.**

### 5.1 Install

```bash
cd /Users/jash/storage/trading-bot-01
uv sync
```

Verified on the mini: `uv` 0.9.16, Python 3.12.12, **torch 2.11.0 with
`torch.backends.mps.is_available() == True`**, 353 packages.

> ⚠️ **`torch-backend = "auto"` in `pyproject.toml` is a dead key.** uv 0.9.16
> rejects it under `[tool.uv]` and prints a `TOML parse error … unknown field`
> warning on *every* `uv` invocation, then ignores it. This is harmless on
> macOS (there is only one arm64 torch wheel and it has MPS built in), but it
> means the setting **is not doing anything on the CUDA box either**. When you
> move to the 5090, select the backend explicitly:
>
> ```bash
> UV_TORCH_BACKEND=auto uv sync     # or: uv sync --torch-backend=cu130
> ```
>
> `uv sync` has no `--torch-backend` flag in 0.9.16 — it is `uv pip`-only plus
> the `UV_TORCH_BACKEND` env var. Moving the key to `[tool.uv.pip]` silences
> the warning but still would not affect `uv sync`.

### 5.2 Verify the transfer landed intact

```bash
uv run ruff check . && uv run mypy src && uv run pytest tests/unit/ -q
```

Actual result on the mini: `All checks passed!` /
`Success: no issues found in 37 source files` / **`140 passed`**.
(If you ever see 96, the Phase 1/2 files did not come across.)

With the services of §5.3 up, the *full* suite including DB integration tests
passes: `uv run pytest -q` → **`159 passed`**.

### 5.3 Services — native, no Docker

**Docker is not installed on the Mac mini yet** (no Docker Desktop, no
colima, no podman), so `make db-up` / `make db-down` / the whole
`docker/docker-compose.yml` path does not work here *for now*. Both services
run natively instead.

The compose path is deliberately left intact and unchanged — Docker is
planned for this machine, and it is still the path on the CUDA box. Once
Docker is installed here, `make db-up` works again and you can pick either;
just don't run both at once, since they both bind 5432 and 5555. The native
targets below are additive, not a replacement.

**Postgres 16 — Homebrew service (installed & running):**

```bash
brew install postgresql@16
brew services start postgresql@16          # auto-restarts at login
export PATH="/opt/homebrew/opt/postgresql@16/bin:$PATH"   # keg-only

# one-time role + db + schemas (already done)
psql -h localhost -d postgres -c \
  "CREATE ROLE trader LOGIN PASSWORD 'trader' SUPERUSER;"
createdb -h localhost -O trader trader
psql -h localhost -U trader -d trader -f docker/init-scripts/001_schema.sql
```

**MLflow — from the project venv (no container):**

```bash
cd /Users/jash/storage/trading-bot-01
mkdir -p mlruns logs
nohup .venv/bin/mlflow server \
  --host 127.0.0.1 --port 5555 \
  --backend-store-uri "sqlite:///$(pwd)/mlruns/mlflow.db" \
  --artifacts-destination "$(pwd)/mlruns/artifacts" \
  --serve-artifacts \
  > logs/mlflow.log 2>&1 &
```

The compose file's MLflow used a Docker volume; this uses a local SQLite
backend store plus a local artifact root under `mlruns/`. Same API, same port,
and `runner.py`'s hardcoded `http://localhost:5555` needs no change.

Health checks:

```bash
pg_isready -h localhost -p 5432          # => accepting connections
curl -s http://127.0.0.1:5555/health     # => OK
```

> **macOS note:** MLflow uses port **5555**, not 5000 — 5000 is taken by
> AirPlay Receiver (`ControlCenter`). Confirmed still true on this mini.

**Migrations (already applied — `alembic_version` is at `0001`):**

```bash
set -a && . ./.env && set +a      # nothing in the codebase auto-loads .env
uv run alembic upgrade head
```

This created 5 `market.*` and 6 `ledger.*` tables.

> **Gotcha:** no module calls `load_dotenv()`. `docker compose` used to read
> `.env` for you; running natively, **you must `set -a && . ./.env && set +a`
> yourself** before anything that touches Postgres. It happens to work without
> that today only because every default in `engine.py` / `alembic/env.py`
> matches the `.env` values (`trader` / `trader` / `localhost` / `5432`).

### 5.4 Data — already present, no rebuild needed

`data/` is 96 MB / 1,835 files and came over with the copy: `data/panels/{train,val,test}.parquet`
(+ `.sha256` sidecars), `data/ohlcv/year=*/ticker=*.parquet`, `data/raw/universe_v1.parquet`.
Panels cover train 2014→2021, val 2022-02→2022-12, test 2023-02→present, with
purge gaps.

Only if you ever need to rebuild from scratch (~15–25 min, mostly the Yahoo
download):

```bash
uv run python scripts/build_universe.py data=universe_v1    # ~5s
uv run python scripts/fetch_data.py     data=universe_v1    # ~10-20 min
uv run python scripts/build_features.py data=universe_v1    # ~2-3 min
```

### 5.5 What did *not* come across

`mlruns/` was **not** transferred — it is a fresh, empty tracking store. The
`corr(val_sharpe, test_sharpe) = −0.86` figure in §2 and every number in
`Report/main.tex` live only in the MacBook's MLflow. **If you want the
baseline number to compare Phase 1 against, you must either rsync the
MacBook's `mlruns/` over or re-run the `mlp_baseline` control from §6.**

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

- **Current machine:** Mac mini, Apple **M4 / 24 GB unified memory**,
  macOS 26.6.2 arm64, MPS backend. Repo at
  `/Users/jash/storage/trading-bot-01`.
- **Docker not installed here yet** (planned). Until it is, Postgres 16 runs
  as a Homebrew service and MLflow from the project venv — see §5.3, or
  `make services-up`. The `db-up` / `db-down` compose targets are untouched
  and will work on the mini as soon as Docker is installed; don't run the
  native and compose stacks simultaneously (both bind 5432 / 5555).
- **Intended training machine:** RTX 5090 (Blackwell GB202), CUDA 13.0,
  driver 595 — see `make sync` / `make sync-data` for the rsync deploy path.
  Note `make sync`'s `REMOTE_DIR` default is still
  `/home/jash/trading-agent/Trading_Bot`; `SERVER`/`REMOTE_DIR` are
  command-line overridable.
- **Python 3.12**, `uv` 0.9.16 for dependency management. torch 2.11.0.
- MLflow on **5555** (AirPlay owns 5000 on macOS).
- Nothing auto-loads `.env` — `set -a && . ./.env && set +a` before any
  Postgres-touching command.
