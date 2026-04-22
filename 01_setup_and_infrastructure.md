# 01 — Setup and Infrastructure

## Tooling choices

- **Python** 3.12
- **Package manager** `uv` (fast, deterministic, replaces pip/poetry)
- **Lint / format** `ruff` (format + lint)
- **Types** `mypy` in strict mode on `src/`, permissive on `scripts/`
- **Tests** `pytest`, `pytest-xdist`, `hypothesis` for property tests
- **Config** `hydra-core` with OmegaConf
- **Logging** `loguru`
- **Experiment tracking** `mlflow` (local server, file + sqlite backend)
- **DB** PostgreSQL 16 in Docker, `sqlalchemy` + `psycopg[binary]`,
  migrations via `alembic`
- **DL** PyTorch (CUDA 12.8 build for RTX 5090 Blackwell), `torch-geometric`
- **RL** baseline implementation from CleanRL style (single-file PPO),
  with `stable-baselines3` as a fallback sanity check
- **RL env** `gymnasium` (NOT legacy `gym`)
- **Data** `polars` for heavy transforms, `pandas` at the env boundary,
  `pyarrow` for Parquet, `yfinance`, `kaggle`
- **CLI** `typer`

## Repo layout

```
trading-agent/
├── pyproject.toml
├── uv.lock
├── Makefile
├── README.md
├── .env.example
├── .gitignore
├── docker/
│   ├── docker-compose.yml
│   ├── postgres.Dockerfile
│   └── init-scripts/
│       └── 001_schema.sql            # bootstrap; real schema via alembic
├── configs/                           # Hydra configs
│   ├── config.yaml
│   ├── data/
│   │   ├── default.yaml
│   │   └── universe_v1.yaml
│   ├── env/
│   │   └── panel_daily.yaml
│   ├── model/
│   │   ├── mlp_baseline.yaml
│   │   └── gnn_v1.yaml
│   ├── train/
│   │   ├── ppo_baseline.yaml
│   │   └── ppo_gnn.yaml
│   └── broker/
│       ├── paper.yaml
│       └── zerodha.yaml              # stub
├── src/
│   └── trader/
│       ├── __init__.py
│       ├── cli.py                    # typer entry point
│       ├── config.py                 # pydantic models for configs
│       ├── data/
│       │   ├── __init__.py
│       │   ├── universe.py
│       │   ├── sources/
│       │   │   ├── base.py
│       │   │   ├── yfinance_source.py
│       │   │   ├── kaggle_source.py
│       │   │   └── zerodha_source.py # stub
│       │   ├── alignment.py          # panel + is_tradeable mask
│       │   ├── features.py
│       │   ├── storage.py            # parquet + PG metadata
│       │   └── sentiment.py          # optional, stub in v1
│       ├── env/
│       │   ├── __init__.py
│       │   ├── panel_env.py          # PanelTradingEnv(gym.Env)
│       │   ├── costs.py              # STT, GST, brokerage, slippage
│       │   ├── reward.py
│       │   └── baselines.py          # buy-and-hold, momentum, 60/40
│       ├── models/
│       │   ├── __init__.py
│       │   ├── encoders.py           # TCN, optional transformer
│       │   ├── graph.py              # hetero GAT
│       │   ├── heads.py              # actor/critic heads with masked softmax
│       │   └── actor_critic.py
│       ├── training/
│       │   ├── __init__.py
│       │   ├── ppo.py                # single-file PPO, CleanRL-style
│       │   ├── walk_forward.py
│       │   ├── eval_metrics.py
│       │   └── callbacks.py
│       ├── broker/
│       │   ├── __init__.py
│       │   ├── base.py               # Broker ABC
│       │   ├── paper_broker.py
│       │   ├── zerodha_broker.py     # stub
│       │   └── schema.py             # sqlalchemy models
│       ├── db/
│       │   ├── __init__.py
│       │   ├── engine.py
│       │   └── migrations/           # alembic
│       └── utils/
│           ├── seeding.py
│           ├── time_utils.py         # NSE calendar
│           └── logging.py
├── scripts/
│   ├── bootstrap_db.py
│   ├── fetch_data.py
│   ├── build_universe.py
│   ├── build_features.py
│   ├── train.py
│   ├── evaluate.py
│   └── paper_run.py
├── tests/
│   ├── unit/
│   ├── property/
│   └── integration/
└── notebooks/                         # exploratory, not CI-gated
```

## pyproject.toml (essentials)

```toml
[project]
name = "trader"
version = "0.1.0"
requires-python = ">=3.12,<3.13"
dependencies = [
  "gymnasium>=0.29",
  "numpy>=1.26",
  "polars>=1.0",
  "pandas>=2.2",
  "pyarrow>=16",
  "yfinance>=0.2.40",
  "kaggle>=1.6",
  "sqlalchemy>=2.0",
  "psycopg[binary]>=3.2",
  "alembic>=1.13",
  "hydra-core>=1.3",
  "omegaconf>=2.3",
  "pydantic>=2.7",
  "typer>=0.12",
  "loguru>=0.7",
  "mlflow>=2.14",
  "tqdm>=4.66",
  # torch, torch-geometric installed separately due to CUDA index
]

[project.optional-dependencies]
dev = [
  "pytest>=8",
  "pytest-xdist>=3",
  "hypothesis>=6",
  "ruff>=0.5",
  "mypy>=1.10",
  "ipykernel>=6",
]

[tool.ruff]
line-length = 100
target-version = "py312"

[tool.mypy]
python_version = "3.12"
strict = true
files = ["src"]
```

Install torch + torch-geometric against the CUDA 12.8 index separately
(pin the wheels that support Blackwell at the time you build this).

## Makefile targets

```
make setup           # uv sync, pre-commit install
make db-up           # docker compose up -d postgres mlflow
make db-down
make db-migrate      # alembic upgrade head
make data-fetch      # scripts/fetch_data.py
make features        # scripts/build_features.py
make train CONFIG=train/ppo_baseline
make eval  RUN=<mlflow-run-id>
make paper CONFIG=broker/paper
make test
make lint
make format
```

## docker-compose.yml

Services:
- `postgres` (16-alpine) — exposes 5432, volume `pgdata`, healthcheck.
- `pgadmin` (dpage/pgadmin4) — optional, port 5050.
- `mlflow` — custom image or `ghcr.io/mlflow/mlflow`, file-backed tracking
  and sqlite metadata, volume `mlruns`.

Env vars read from `.env` (see `.env.example`). Do not commit `.env`.

## Database

Two logical schemas:

1. `market` — reference data:
   - `instruments`, `sectors`, `universe_snapshots`,
     `corporate_actions`, `trading_calendar`
2. `ledger` — paper-trading state:
   - `orders`, `fills`, `positions`, `portfolio_snapshots`, `pnl_daily`,
     `strategy_runs`

Bulk historical OHLCV is NOT stored in Postgres — it lives in partitioned
Parquet (`data/ohlcv/year=YYYY/ticker=XYZ.parquet`). Postgres holds
metadata, universe, and ledger state.

## Configuration pattern

Hydra composed config, e.g.:

```bash
python scripts/train.py \
  data=universe_v1 \
  env=panel_daily \
  model=gnn_v1 \
  train=ppo_gnn \
  train.total_steps=2_000_000 \
  env.initial_cash=1_000_000
```

All Hydra configs are validated into Pydantic models in
`src/trader/config.py` at entry so downstream code is typed.

## Determinism

`src/trader/utils/seeding.py::seed_everything(seed)` seeds
`random`, `numpy`, `torch`, `torch.cuda`, and sets
`torch.backends.cudnn.deterministic = True`. Env uses
`np.random.Generator` seeded from the same seed.

## Acceptance criteria for Phase 0

- `make setup && make db-up && make test` exits 0 on a clean clone.
- `alembic upgrade head` creates all tables.
- MLflow UI reachable at `http://localhost:5000`.
- `python -c "import torch; print(torch.cuda.is_available())"` prints
  `True` and detects the RTX 5090.
- `ruff check .` and `mypy src` exit clean.
