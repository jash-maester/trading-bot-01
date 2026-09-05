# 07 — Roadmap: Milestones for Claude Code

## Status

**PARTIAL — and the sequence was broken.** Last verified against `60ba1a6`
(the A0/A1/A2 audit reports, 2026-09-05).

M0–M4 and M6–M8 have code that runs. **M5's acceptance gate never passed** —
*"matches or beats `EqualWeightRebalanced` on the val split (net of costs) over
3 seeds"* — and M6 (hetero GNN), M7 (walk-forward), Phase 1 (regime FiLM) and
Phase 2 (auxiliary return head) were built on top of it regardless: roughly
2,500 lines of model code and 44 tests standing on a gate that never opened.
That is precisely what the paragraph below forbids.

This is not a code-quality problem — `ruff` is clean, `mypy` is clean, 368 tests
pass. It is a sequencing problem, and it compounds.

**This roadmap is superseded from M5 onward** by the R0–R7 sequence in
`09_revamp_and_audit.md` §5. M0–M4 stand as history. Every milestone below has
had its acceptance criteria annotated with measured state; the RTX 5090
criteria in M5 and M6 are void and have been replaced with RTX 4060 8 GB
equivalents.

---

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

**Acceptance — annotated 2026-09-05**
- `scripts/build_universe.py` creates a `universe_snapshots` row and
  `configs/data/universe_v1.yaml` is loadable by Hydra.
  **FAILS: the table has 0 rows** and the row it would write is today's
  `all_tickers()` back-stamped onto `cfg.data.start_date` — not point-in-time
  (A0 §2.4). Both of `universe_v1.yaml`'s own keys are dead (A0 §3.1).
- `scripts/fetch_data.py data.start=2014-01-01 data.end=2024-12-31`
  fetches the universe (idempotent on rerun). **UNVERIFIED** — idempotence
  never tested by an audit.
- Unit test: known split adjustment (pick a real Indian stock split,
  assert continuity on adjusted close). **UNVERIFIED.**
- `zerodha_source.py` exists with the interface but raises
  `NotImplementedError`. **SUPERSEDED:** `ZerodhaSource.fetch_ohlcv`
  (`sources/zerodha_source.py:274`) is live and read-only; only
  `fetch_corporate_actions` still raises (`:327`).
- **Missing criterion this milestone needed:** the universe returned by
  `all_tickers()` matches the ticker set of the panels built from it. It does
  not — 163 vs 645 (A0 §2.6).

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

**Acceptance — annotated 2026-09-05**
- Mask correctness test: a ticker listed in 2017 has `is_tradeable=False`
  across 2014–2016. **UNVERIFIED** as a specific test; the mask itself does
  hold — no zero-close row is ever marked tradeable (A0 §2.2).
- Feature no-lookahead test: static scan rejects `.shift(-k)` or
  `rolling(...).apply(lambda x: x[-0:])` patterns. **UNVERIFIED.** Note that a
  static scan cannot catch the leak that actually exists here, which is
  cross-boundary rolling windows, not forward shifts (A1 §3).
- No NaN in numeric columns on `is_tradeable=True` rows. **HOLDS**, but it is
  the wrong invariant: `beta_nifty_60d` was constant 1.0 on all 276,005
  tradeable rows and has no NaN (A0 §2.8). **The missing criterion is nonzero
  variance per feature.**
- Determinism: rerun produces identical SHA256. **PASSES** — all six sidecars
  across both panel roots verify (A0 §2.6).

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
  **As built:** `LogReturn` is what runs; `DifferentialSharpe` is written,
  unit-tested and never selected (A0 §2.5).
- `src/trader/env/baselines.py` — BuyAndHold, EqualWeight, MomentumTopK,
  SixtyForty, Random.

**Acceptance — annotated 2026-09-05**
- All env invariants in `03_environment.md` pass as tests. **PASSES as a
  suite**, but no test covers the reported `turnover`, which measures NAV drift
  rather than trading (`panel_env.py:286-288`; see `03_environment.md`).
- Baselines run on train split, log metrics to MLflow. **FAILS in substance:**
  `EqualWeightRebalanced` rebalances every step, not monthly
  (`baselines.py:57-62`); `MomentumTopK` runs at K=5 (~47% cash under the
  10% cap) ranking on `log_return_1d` rather than `log_return_20d`
  (`baselines.py:91,110`); and `BuyAndHoldIndex` holds equal weight across the
  whole universe from day 1 rather than tracking NIFTY 50 monthly
  (`baselines.py:65-81`). **Three of five baselines are not the thing they are
  named.** See `03_environment.md` § Baselines.
- Full episode rollout (252 days, 150 tickers) in under 200 ms on CPU.
  **UNVERIFIED** — never measured as stated. A2 §4.1 puts env stepping at
  0.003% of a PPO update at 504 tickers, so the criterion's purpose (making
  vectorisation worthwhile) is moot either way.
- Vectorized env (**`SyncVectorEnv`**, 16 envs) works without pickling errors.
  **PASSES.** The spec said `AsyncVectorEnv`; the code uses `SyncVectorEnv`
  (`runner.py:18,171`) and should keep doing so — env stepping is 0.2% of wall
  clock, so async buys nothing (A2 §4.1).

**Prompt seed**
> Implement Milestone 4 from `07_roadmap.md`. The environment spec is
> `03_environment.md`. Use Gymnasium, not legacy gym. Implement the cost
> model exactly as specified (Zerodha equity delivery, Indian fees).
> Include all 5 baselines as agents that emit the same logit vector as
> the RL policy.

**Do not re-run this prompt seed.** "Implement the cost model exactly as
specified" is how `src/trader/env/costs.py` acquired a 22% understatement from a
stale document. `CLAUDE.md` rule 5: the constants live in the module and are
edited in place against `zerodha.com/charges`; `03_environment.md` describes
them, it does not specify them.

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

**Acceptance — annotated 2026-09-05. THIS GATE NEVER PASSED.**
- ~~Runs `total_steps=200_000` end-to-end in under 30 min on the 5090.~~
  **VOID — that machine does not exist.** RTX 4060 8 GB equivalent, from A2's
  measured figures: 200,000 env steps is 48.83 updates × 198.3 TFLOP =
  **9.68 PFLOP**. At the 2.93 TFLOPS this graph sustains on the 4060 that is
  **~55 minutes**; at the card's 9.05 TFLOPS FP32 peak, 18 minutes. **A 30-minute
  criterion is therefore not met by the current graph** and can only be met by
  encoder caching. Wall clock on the 4060 is **NOT MEASURED** (A2 §0.2) — run
  `scripts/profiling/a2_all.sh` before asserting any timing.
- Beats `RandomPolicy` and matches or beats `EqualWeightRebalanced` on
  the val split (net of costs) over 3 seeds. **FAILS — and this is the finding
  that matters most in the project.** Best stored result is mean Sharpe −1.259 /
  CAGR −16.1% at 4,096 steps and −0.963 / −10.2% at 500,000 steps, all seed 42,
  all on the dead-beta panel with the understated cost model (A0 §4.2). Three
  seeds have never been run. The comparison baseline is itself mis-specified
  (M4 above).
- MLflow run has params, metrics, equity curve artifact. **PARTIAL** — 5 runs
  exist with 35 params each, no `env.*`/`data.*` params, none of the five
  specified tags, and no `test_*` metric was logged until A1 ran one window
  (A0 §4.2, A1 §5.1).
- Overfitting guardrail tests wired in. **PARTIAL** — wired, but the
  regularisers they assume are inactive during training and active during
  evaluation (`ppo.py:218`, `ppo.py:445`; A1 §7.1).

> **`CLAUDE.md` rule 1: never implement a milestone whose predecessor's
> acceptance criteria have not demonstrably passed.** M6, M7, Phase 1 and
> Phase 2 all violated it against this gate. Do not build further on M5 —
> `09_revamp_and_audit.md` §5 replaces the sequence from here.

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

**Acceptance — annotated 2026-09-05. Built before M5's gate opened.**
- ~~Forward pass under 20 ms on batch `[8, 150, 60, 15]` on the 5090.~~
  **VOID — deleted, not translated.** The machine does not exist and the shape
  is not the one that runs (production is `[128, 504, 60, 15]`). Replacement, at
  the production shape and measured: one gradient step is **1,549 GFLOP**, of
  which the TCN is 97.9%; peak VRAM at that shape is **7,762–7,766 MiB of
  8,188** on the 4060 (A2 §4.4, §6.1). Wall clock **NOT MEASURED** (A2 §0.2).
- Attention weights on untradeable neighbors are ≈ 0 (unit test). **The test
  passes and does not test the failing case.** Every mask in
  `tests/unit/test_graph.py` is constant along the batch dimension, which is
  exactly the case `graph.py:334` handles correctly. With a batch-varying mask,
  **87.6% of minibatch elements get the wrong adjacency** on `data/panels` and
  94.3% on `data/panels_kite` (A1 §6).
- Ablation runs (no-graph vs graph-v1 vs only-intra-sector) all
  trainable and logged to MLflow. **FAILS — no GNN run exists in MLflow**, and
  every `gnn_v1` / `gnn_intra_only` result would be uninterpretable while
  `graph.py:334` stands (A1 §6.4). `model.graph.num_sectors` is also hardcoded
  to 8 against a 14-sector universe (A1 §7.6).

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

**Acceptance — annotated 2026-09-05. Built before M5's gate opened.**
- Running one full walk-forward (4 windows × 5 seeds) for the GNN
  config completes in the expected budget and produces per-window and
  aggregated metrics. **FAILS** — `data/walks/` was empty until A1's run, and
  the configured seed list is 3, not 5 (`configs/walk/default.yaml:11`). The
  "expected budget" is undefined; see `05_training.md` § Compute budget for the
  measured 4060 arithmetic.
- Paired bootstrap CI of RL agent vs equal-weight is reported. **PARTIAL and
  misleading** — `walk_forward.py:684-717` has no minimum-n guard and the five
  evaluation episodes are one deterministic date range, so at n=1 it returned a
  zero-width "95% CI" as a significance result (A1 §5.4, §7.2).
- Shuffled-ticker-label sanity check is wired (optional flag). **Wired but
  never run** until A1 ran it; it produced no evidence of a leak and cannot
  produce meaningful evidence either way until the dropout and single-window
  defects are fixed (A1 §5). The command documented at
  `scripts/walk_forward.py:15-17` fails as written — drop the `+` prefix.
- **Also:** `scripts/walk_forward.py:59` hardcodes `data/panels`, so
  `data=kite_v1` changes nothing (A1 §7.4), and the "full panel" it rebuilds
  from the purged splits has two month-long holes (A1 §7.3).

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

**Acceptance — annotated 2026-09-05**
- Golden-file integration test: fixed seed, fixed dates → fixed fills,
  fixed NAV, fixed snapshots. **UNVERIFIED** — no audit exercised it.
- `scripts/paper_run.py` completes a 1-year session in under 30 min.
  **UNVERIFIED.** The broker has run: `ledger` holds 3 `strategy_runs`, 1,900
  orders, 1,900 fills, 501 `portfolio_snapshots` (A0 §4.3) — priced off the
  stale, dividend-adjusted, dead-beta `data/panels`, because
  `scripts/paper_run.py:266` hardcodes it.
- `scripts/ledger.py report --run <id> --format pdf` generates a
  stable report. **UNVERIFIED.**
- **Known divergence:** the paper broker applies `min_trade_value: 500`
  (`paper_broker.py:1038`) and the backtest env does not implement a minimum
  trade size at all. Both floor to whole shares, so rounding is *not* the
  difference (A0 §1 row 12). The 11% NAV divergence quoted in
  `ARCHITECTURE.md:147` has no regenerating artefact — **UNVERIFIED**.

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

---

## What actually happens next

Not M8. The sequence from here is **R0–R7 in `09_revamp_and_audit.md` §5**, with
A0–A2 complete and this document's reconciliation (A3) being the first unit
permitted to write. In outline:

- **R1 — one panel, one truth.** Point-in-time universe, one reachable panel
  root, purge ≥ feature window, every feature with nonzero variance.
- **R2 — honest baselines.** Fix `MomentumTopK` (K and ranking feature) and the
  turnover metric first; then the benchmark table, net of cost and tax. This
  ends the ambiguity about whether anything beats equal-weight.
- **R3 — the compute bug.** Encoder caching is not one of three candidate fixes,
  it is the whole fix: the TCN is 97.9% of all arithmetic (A2 §10).
- **R4 — supervised cross-sectional model.** Rank IC with a confidence interval.
  Go/no-go on whether signal exists at all.
- **R5–R7** — deterministic allocator, then RL narrowly scoped, then regime
  conditioning only if still warranted.

Phase 1 (regime FiLM) should be **shelved rather than run** until the pathology
it targets is re-measured on a fixed panel with more windows — on n=4 the
`r = −0.86` that motivated it has a 95% CI of (−0.997, +0.583), which includes
zero (`09_revamp_and_audit.md` §3).
