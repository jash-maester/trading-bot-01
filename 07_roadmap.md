# 07 — Roadmap: Milestones for Claude Code

Each milestone is scoped to land in a single Claude Code session.
Acceptance criteria are tests or commands that must pass before moving
on. The **prompt seed** at the bottom of each milestone is what to
paste into Claude Code to start that milestone.

Sequence is deliberate. Do not jump ahead; later milestones depend on
invariants established earlier.

---

## M0 — Bootstrap

**Scope**
- Repo skeleton from `01_setup_and_infrastructure.md`.
- `pyproject.toml` with uv, ruff, mypy, pytest.
- Hydra config skeleton (empty but valid).
- `docker-compose.yml` with `postgres`, `mlflow`, optional `pgadmin`.
- `.env.example`, `.gitignore`, `README.md` stub, `Makefile`.

**Acceptance**
- `uv sync` succeeds.
- `docker compose up -d` brings up `postgres` and `mlflow`; healthchecks
  pass.
- `ruff check .` and `mypy src` pass on an empty `src/trader` package.
- `pytest` runs (0 tests, 0 failures).

**Prompt seed**
> Implement Milestone 0 from `07_roadmap.md` exactly as specified in
> `01_setup_and_infrastructure.md`. Create the directory layout, the
> `pyproject.toml`, the `Makefile`, the `docker-compose.yml`, and an
> empty but valid Hydra config tree. Do not implement any application
> code yet. End by running `make setup && make db-up && make test` and
> report the output.

---

## M1 — Database schema

**Scope**
- Alembic initial migration creating schemas `market` and `ledger` and
  all tables from `02_data_pipeline.md` and `06_paper_broker.md`.
- `src/trader/db/engine.py` exposing a configured SQLAlchemy engine.
- `src/trader/broker/schema.py` SQLAlchemy models matching the DDL.
- `scripts/bootstrap_db.py` runs `alembic upgrade head`.

**Acceptance**
- `make db-migrate` on a fresh DB creates all tables.
- Integration test: round-trip insert/select through SQLAlchemy models
  for each table.
- `mypy src/trader/db src/trader/broker/schema.py` clean.

**Prompt seed**
> Implement Milestone 1 from `07_roadmap.md`. The schema definitions
> are in `02_data_pipeline.md` (market tables) and `06_paper_broker.md`
> (ledger tables). Use SQLAlchemy 2.0 declarative models and Alembic.
> Add integration tests that actually hit the dockerized Postgres.

---

## M2 — Data ingestion

**Scope**
- `src/trader/data/sources/{base,yfinance_source,kaggle_source,zerodha_source}.py`
- `src/trader/data/universe.py` with NIFTY 50 + sectoral lists.
- `src/trader/data/storage.py` for Parquet partitioning.
- `scripts/build_universe.py` and `scripts/fetch_data.py`.
- Raw data cache layout under `data/raw/yfinance/...`.

**Acceptance**
- `scripts/build_universe.py` creates a `universe_snapshots` row and
  `configs/data/universe_v1.yaml` is loadable by Hydra.
- `scripts/fetch_data.py data.start=2014-01-01 data.end=2024-12-31`
  fetches the universe (idempotent on rerun).
- Unit test: known split adjustment (pick a real Indian stock split,
  assert continuity on adjusted close).
- `zerodha_source.py` exists with the interface but raises
  `NotImplementedError`.

**Prompt seed**
> Implement Milestone 2 from `07_roadmap.md`. Follow `02_data_pipeline.md`
> for universe construction rules, source interface, and caching. The
> Zerodha source is interface-only. Add unit tests including a
> known-split adjustment test.

---

## M3 — Alignment and features

**Scope**
- `src/trader/data/alignment.py` producing a `[T, N]` panel with
  `is_tradeable` mask using NSE trading calendar.
- `src/trader/data/features.py` implementing the feature list from
  `02_data_pipeline.md`.
- `scripts/build_features.py` producing `train.parquet`,
  `val.parquet`, `test.parquet` with SHA256 sidecars and a row in
  `market.dataset_versions`.

**Acceptance**
- Mask correctness test: a ticker listed in 2017 has `is_tradeable=False`
  across 2014–2016.
- Feature no-lookahead test: static scan rejects `.shift(-k)` or
  `rolling(...).apply(lambda x: x[-0:])` patterns.
- No NaN in numeric columns on `is_tradeable=True` rows.
- Determinism: rerun produces identical SHA256.

**Prompt seed**
> Implement Milestone 3 from `07_roadmap.md`. Follow the alignment
> rules in `02_data_pipeline.md` — especially: no zero-filling of
> non-existent stocks, no interpolation across multi-day gaps. Add all
> listed acceptance tests.

---

## M4 — Environment + baselines

**Scope**
- `src/trader/env/panel_env.py` — `PanelTradingEnv(gym.Env)` per
  `03_environment.md`.
- `src/trader/env/costs.py` — `ZerodhaEquityDeliveryCostModel`.
- `src/trader/env/reward.py` — differential Sharpe + turnover penalty.
- `src/trader/env/baselines.py` — BuyAndHold, EqualWeight, MomentumTopK,
  SixtyForty, Random.

**Acceptance**
- All env invariants in `03_environment.md` pass as tests.
- Baselines run on train split, log metrics to MLflow.
- Full episode rollout (252 days, 150 tickers) in under 200 ms on CPU.
- Vectorized env (`AsyncVectorEnv`, 16 envs) works without
  pickling errors.

**Prompt seed**
> Implement Milestone 4 from `07_roadmap.md`. The environment spec is
> `03_environment.md`. Use Gymnasium, not legacy gym. Implement the cost
> model exactly as specified (Zerodha equity delivery, Indian fees).
> Include all 5 baselines as agents that emit the same logit vector as
> the RL policy.

---

## M5 — Baseline MLP policy + PPO (no graph yet)

**Scope**
- `src/trader/models/encoders.py` — TCN.
- `src/trader/models/heads.py` — actor/critic heads with masked softmax.
- `src/trader/models/actor_critic.py` — simple encoder → mean-pool →
  heads (no GNN).
- `src/trader/training/ppo.py` — single-file CleanRL-style PPO.
- `src/trader/training/eval_metrics.py`.
- `scripts/train.py` hydra entry point.

**Acceptance**
- Runs `total_steps=200_000` end-to-end in under 30 min on the 5090.
- Beats `RandomPolicy` and matches or beats `EqualWeightRebalanced` on
  the val split (net of costs) over 3 seeds.
- MLflow run has params, metrics, equity curve artifact.
- Overfitting guardrail tests wired in.

**Prompt seed**
> Implement Milestone 5 from `07_roadmap.md`. Use `04_models.md` for
> the encoder and heads (skip the GNN — next milestone). Use
> `05_training.md` for PPO and evaluation. Keep the PPO implementation
> in a single file for readability. Run 3 seeds on the `ppo_baseline`
> config and report MLflow URLs.

---

## M6 — Hetero GNN model

**Scope**
- `src/trader/models/graph.py` — Hetero GAT with sector hierarchy per
  `04_models.md`.
- `configs/model/gnn_v1.yaml`.
- Sector graph construction utilities (intra-sector, inter-sector,
  membership edges).
- DropEdge regularization.

**Acceptance**
- Forward pass under 20 ms on batch `[8, 150, 60, 15]` on the 5090.
- Attention weights on untradeable neighbors are ≈ 0 (unit test).
- Ablation runs (no-graph vs graph-v1 vs only-intra-sector) all
  trainable and logged to MLflow.

**Prompt seed**
> Implement Milestone 6 from `07_roadmap.md`. Follow `04_models.md`
> section "Hetero graph" and "Hetero GAT". Use `torch_geometric`'s
> `HeteroConv` with `GATv2Conv`. Keep the existing PPO training
> harness; only the model changes. Add the three ablation configs.

---

## M7 — Walk-forward training

**Scope**
- `src/trader/training/walk_forward.py`.
- Config-driven windowing from `05_training.md`.
- Aggregated per-window reporting.

**Acceptance**
- Running one full walk-forward (4 windows × 5 seeds) for the GNN
  config completes in the expected budget and produces per-window and
  aggregated metrics.
- Paired bootstrap CI of RL agent vs equal-weight is reported.
- Shuffled-ticker-label sanity check is wired (optional flag).

**Prompt seed**
> Implement Milestone 7 from `07_roadmap.md`. Walk-forward protocol is
> in `05_training.md`. Add per-window and aggregated metric logging to
> MLflow, and the shuffled-label sanity check behind a flag.

---

## M8 — Paper broker

**Scope**
- `src/trader/broker/base.py` (ABC).
- `src/trader/broker/paper_broker.py`.
- `scripts/paper_run.py`.
- `scripts/ledger.py` for `show`, `report`, `compare`.

**Acceptance**
- Golden-file integration test: fixed seed, fixed dates → fixed fills,
  fixed NAV, fixed snapshots.
- `scripts/paper_run.py` completes a 1-year session in under 30 min.
- `scripts/ledger.py report --run <id> --format pdf` generates a
  stable report.

**Prompt seed**
> Implement Milestone 8 from `07_roadmap.md`. The broker spec is
> `06_paper_broker.md`. Use the SQLAlchemy models from M1. Implement
> T+1 settlement for cash. Add the golden-file integration test.

---

## M9 — Zerodha adapter stub

**Scope**
- `src/trader/broker/zerodha_broker.py` implementing `Broker` ABC with
  `NotImplementedError` bodies and clear TODO comments.
- `configs/broker/zerodha.yaml` with `enabled: false`.
- `docs/going_live.md` runbook (empty structure is fine for v1).

**Acceptance**
- Importing the module does no network calls.
- Paper-run config continues to work unchanged.
- A test confirms that calling any live method raises `NotImplementedError`.

**Prompt seed**
> Implement Milestone 9 from `07_roadmap.md`. Stub only. Do not add
> kite-connect as a required dependency; make it optional. Reference
> the Kite Connect Python SDK in TODO comments but do not call it.

---

## M10 — End-to-end report and iteration loop

**Scope**
- `scripts/full_report.py` produces a single PDF comparing the RL
  agent, baselines, and per-window breakdowns.
- A `CHANGELOG.md` entry for the first real benchmark.
- `docs/interpretation.md` explaining attention-weight exports from the
  GNN.

**Acceptance**
- Running `make full-report` after a completed walk-forward produces
  the PDF and an MLflow artifact link.
- Report includes at least: equity curves, drawdown, per-sector
  allocation over time, GNN attention heatmap, statistical significance
  table vs equal-weight.

**Prompt seed**
> Implement Milestone 10 from `07_roadmap.md`. Produce a single
> end-to-end reporting script and attach an interpretation doc.

---

## After M10

Only then consider:
- Sentiment embeddings (`02_data_pipeline.md` optional section).
- Hierarchical two-level policy (see `04_models.md` "second agent").
- Intraday data and a new env variant.
- Live adapter (separate project stream with its own risk review).

Do not let any of these slip ahead of M10. The baseline has to exist
and beat naive benchmarks before adding complexity.
