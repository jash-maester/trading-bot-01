# Local RL Trading Agent — Plan

A structured, milestone-driven plan you can hand to Claude Code to
build a local, paper-trading-capable RL trading system for Indian
equities. One end-to-end path first, then specialize.

## How to use this

1. Read `00_overview.md` first. Especially the **Honest caveats**
   section — the plan pushes back on a few parts of the original spec
   and explains why.
2. Review `08_open_decisions.md` and override any defaults you
   disagree with before starting. (Shorting? Data frequency? Universe
   size? All listed there.)
3. Open Claude Code in an empty repo, and paste the **Prompt seed**
   from `07_roadmap.md` Milestone 0 to start.
4. Proceed milestone by milestone. Each milestone has explicit
   acceptance criteria that must pass before the next one starts.

## Files in this plan

| File | Purpose |
| --- | --- |
| `00_overview.md` | Goal, architecture, design principles, caveats, assumptions |
| `01_setup_and_infrastructure.md` | Repo layout, deps, Docker, Postgres, MLflow, tooling |
| `02_data_pipeline.md` | Sources, universe, alignment with masks, features |
| `03_environment.md` | `PanelTradingEnv` spec: obs, action, reward, costs |
| `04_models.md` | TCN encoder + Hetero GAT + actor-critic heads |
| `05_training.md` | PPO, walk-forward eval, metrics, overfitting guards |
| `06_paper_broker.md` | DB schema, `Broker` ABC, paper simulator, live stub |
| `07_roadmap.md` | 11 milestones (M0–M10) with acceptance criteria and prompt seeds |
| `08_open_decisions.md` | Defaults chosen; override before starting |

## Key design departures from the original brief

Summarized so you know what to argue with:

1. **Non-existent stocks are masked, not zero-filled.** Zero-fill would
   leak a fake regime-shift signal. The mask is part of the observation
   and the action space.
2. **Graph edges are learned by backprop through a GAT, not set by an
   RL action.** Same interpretability outcome ("learned sector
   relations"), vastly better credit assignment.
3. **One hierarchical actor-critic in v1, not two agents.** A second
   agent is an open door in v2, not a starting point.
4. **LLM/sentiment is offline-only.** Static weekly embeddings if at
   all. No per-step LLM calls.
5. **Baselines are first-class citizens.** The RL agent has to beat
   equal-weight and momentum after real Indian transaction costs
   before any result is reported.
6. **"Real-time adaptation" is walk-forward retraining on a schedule,
   not online gradient updates during live trading.**

## Hardware assumptions

- NVIDIA RTX 5090 (Blackwell, GB202), driver 595, CUDA 13.0.
- Intel Core Ultra 9 285K (Arrow Lake).
- Plenty of RAM and fast local SSD.

The plan is not compute-bound; model sizes are modest on purpose so
the first result ships quickly.

## If something fails an acceptance criterion

- Do not skip ahead.
- Roll back the commit and diagnose.
- The invariants (especially in M3 alignment and M4 env) catch silent
  correctness bugs that you would otherwise notice only in returns —
  at which point the cost to debug is 10× higher.

---

## Operations Guide

Everything you need to go from a fresh clone to a running training job,
step by step. Run these in order the first time; individual sections
can be re-run independently after that.

---

### 0. Prerequisites

| Tool | Minimum version | Install |
|---|---|---|
| Python | 3.12 | `pyenv install 3.12` or system package |
| uv | 0.4+ | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| Docker + Compose | v2 | https://docs.docker.com/get-docker/ |
| Git | any | system package |
| **Linux only** CUDA toolkit | 13.0 (driver ≥ 570) | https://developer.nvidia.com/cuda-downloads |

Verify GPU (Linux/CUDA server):
```bash
nvidia-smi          # should show driver 595, CUDA 13.0
nvcc --version      # confirm toolkit version
```

---

### 1. Clone and install dependencies

```bash
git clone <repo-url> Trading_Bot
cd Trading_Bot
```

**macOS (Apple Silicon — MPS backend):**
```bash
uv sync
```

**Linux + CUDA 13.0 (RTX 5090 / Blackwell, driver 595):**
```bash
uv sync
```
`pyproject.toml` sets `torch-backend = "auto"` (uv 0.11+), which
automatically selects the highest available CUDA wheel for your driver.
Driver 595 / CUDA 13.0 will pick up a `cu130` build if published, or
fall back to `cu128` (fully compatible with driver 595).

To pin a specific CUDA version, create a `uv.toml` in the project root:
```toml
[tool.uv]
torch-backend = "cu130"   # or "cu128" as fallback
```

Verify PyTorch sees the GPU after install:
```bash
uv run python -c "import torch; print(torch.cuda.get_device_name(0))"
# Expected: NVIDIA GeForce RTX 5090
```

Activate the virtual environment (optional — all `uv run` commands work without it):
```bash
source .venv/bin/activate
```

---

### 2. Verify the install (linting, type-checking, unit tests)

Run these after every `uv sync` and after any code change before training.

```bash
# Linting — must report "All checks passed!"
uv run ruff check .

# Auto-fix safe lint issues (import order, deprecated syntax, etc.)
uv run ruff check --fix .

# Type checking — must report "Success: no issues found"
uv run mypy src

# Unit tests — must be 82/82 passed (no DB or data required)
uv run pytest tests/unit/ -v

# Full test suite (includes integration tests — requires DB to be up)
uv run pytest -v
```

Or use the Makefile shortcuts:
```bash
make lint      # ruff check
make format    # ruff format (auto-fixes style)
make test      # pytest tests/unit/
```

---

### 3. Start infrastructure (PostgreSQL + MLflow)

```bash
make db-up
```

This starts two Docker containers:
- **postgres:16** on port `5432`
- **MLflow** on port `5555` (UI at http://localhost:5555)

Run the database migration (only needed once, or after a schema change):
```bash
make db-migrate
```

Stop everything:
```bash
make db-down
```

> **MLflow port conflict on macOS:** Port 5000 is used by AirPlay.
> MLflow is configured to use port 5555 instead. Override with
> `MLFLOW_PORT=<port> make db-up` if needed.

---

### 4. Data pipeline

Run steps 1 → 3 in order. Each step is idempotent — safe to re-run.

**Step 1 — Build universe snapshot** (~5 seconds):
```bash
uv run python scripts/build_universe.py data=universe_v1
```
Writes the ticker list and sector map; optionally inserts a DB row.

**Step 2 — Fetch OHLCV data from Yahoo Finance** (~10–20 minutes):
```bash
uv run python scripts/fetch_data.py data=universe_v1
```
Downloads daily OHLCV for ~163 NSE tickers (2014–present).
Files are cached at `data/raw/yfinance/<ticker>/`. Re-running skips
already-cached tickers instantly.

To fetch a custom date range:
```bash
uv run python scripts/fetch_data.py data=universe_v1 data.start_date=2018-01-01 data.end_date=2024-12-31
```

**Step 3 — Build aligned feature panel** (~2–3 minutes):
```bash
uv run python scripts/build_features.py data=universe_v1
```
Produces three split files with purge gaps between them:

| File | Date range | Purpose |
|---|---|---|
| `data/panels/train.parquet` | 2014-01-01 → 2021-12-31 | Training |
| `data/panels/val.parquet` | 2022-02-01 → 2022-12-31 | Validation |
| `data/panels/test.parquet` | 2023-02-01 → present | Hold-out test |

Each file has a `.sha256` sidecar for integrity verification.

---

### 5. Training

Make sure `data/panels/train.parquet` exists before running.
MLflow must be up (`make db-up`) to record metrics.

**Ablation 1 — MLP baseline (no graph):**
```bash
uv run python scripts/train.py model=mlp_baseline seed=42
```

**Ablation 2 — Full hetero-GNN (all 4 edge types):**
```bash
uv run python scripts/train.py model=gnn_v1 seed=42
```

**Ablation 3 — Intra-sector-only GNN:**
```bash
uv run python scripts/train.py model=gnn_intra_only seed=42
```

Override any config value on the command line (Hydra syntax):
```bash
# Change seed, total steps, or learning rate without editing YAML
uv run python scripts/train.py model=gnn_v1 seed=123 train.total_steps=2000000 train.learning_rate=3e-4
```

**Force CPU** (useful for debugging shape errors):
```bash
CUDA_VISIBLE_DEVICES="" uv run python scripts/train.py model=mlp_baseline seed=42
```

Checkpoints are saved to `checkpoints/` every N steps (configured in
`configs/train/defaults.yaml`). MLflow run URLs are printed to stdout.

---

### 6. View results in MLflow

Open http://localhost:5555 in a browser.

Runs are grouped under the `trading_bot` experiment. Each run logs:
- Hyperparameters (`seed`, `n_params`, all `train.*` and `model.*` values)
- Training metrics per update step (loss, entropy, value loss, etc.)
- Episode metrics (Sharpe, CAGR, max drawdown, turnover)
- Validation Sharpe at the end of training
- `config.yaml` artifact (full Hydra config snapshot)
- Model checkpoints as artifacts

---

### 7. Integration tests (requires DB)

```bash
make db-up
make db-migrate
uv run pytest tests/integration/ -v
make db-down
```

---

### Quick-reference cheat sheet

```bash
# --- Setup ---
uv sync                                                    # install / update deps
uv run ruff check --fix . && uv run mypy src               # lint + types
uv run pytest tests/unit/ -v                               # unit tests

# --- Services ---
make db-up                                                 # start postgres + mlflow
make db-migrate                                            # apply schema migrations
make db-down                                               # stop services

# --- Data ---
uv run python scripts/build_universe.py data=universe_v1
uv run python scripts/fetch_data.py data=universe_v1
uv run python scripts/build_features.py data=universe_v1

# --- Train ---
uv run python scripts/train.py model=mlp_baseline seed=42
uv run python scripts/train.py model=gnn_v1 seed=42
uv run python scripts/train.py model=gnn_intra_only seed=42

# --- MLflow UI ---
open http://localhost:5555
```
