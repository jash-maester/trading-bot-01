# 08 — Open Decisions

## Status

**PARTIAL** — last verified against `60ba1a6` (the A0/A1/A2 audit reports,
2026-09-05). Most defaults below are still the live defaults. Five have drifted
from the code and are corrected in place; one explicit non-decision is
**violated**.

Known broken:

- **"Survivorship-bias-aware universe" is listed as a non-decision and is not
  implemented** — `market.universe_snapshots` has 0 rows and no read site
  (A0 §2.4).
- **The reward default (Differential Sharpe) is not what runs** — `LogReturn`
  is (A0 §2.5).
- **Universe size is 645 fetched / 504 active**, not 130–160 (A0 §2.6).
- **Eight declared config keys do nothing**, and the 10% per-name cap every
  document quotes is not a config key at all. Table at the end of this
  document.
- **Rebalance frequency has never been a decision** and is the most expensive
  unexamined choice in the system.

## Market scope

- **Default:** Indian equities only (NSE).

## Data frequency

- **Default:** Daily bars.
- **Alternatives:** 15-min or 1-min intraday.
- **Why the default:** Daily is enough to prove the architecture with
  10+ years of history. Intraday adds an order of magnitude to storage,
  features, and compute, and changes the slippage model materially.

## Universe size

- **Default as of `a5d8b79`:** **645 tickers fetched across 14 sectors;
  504 traded.** `INACTIVE_SECTORS` holds 5 sectors (141 names) out of training
  purely to keep observation width — and therefore wall-clock cost per run — at
  504 rather than 645 (`universe.py:273-282`). Sector assignment from NSE
  Indices' "Industry" column (`ARCHITECTURE.md:37`; **UNVERIFIED** against a
  downloaded NSE file).
- **Superseded default:** NIFTY 50 + 20 per sector × ~6 sectors ≈ 130–160.
- **Alternatives:** top 500 by ADV; NIFTY 500 full.
- **Why it changed:** the 8-sector taxonomy excluded eight NIFTY 50 names.
- **The cost of the change, which was not priced:** the panels on disk still
  hold 163 tickers, of which 143 are in `active_tickers()`, so **361 of 504
  observation columns are permanently zero — 71.6% of the compute budget spent
  on padding** (A0 §2.6, A2 §1.1). Widening the universe before rebuilding the
  panel bought nothing and cost 3.5×. See `02_data_pipeline.md`.
- "Small enough that a Hetero GAT fits comfortably" is no longer true: at
  N=504 / L=60 / minibatch 128 the 4060 sits at ~95% of VRAM, and 645 tickers
  does not fit at all (A2 §8.1).

## Action space

- **Default:** Continuous target-allocation logits; env applies masked
  softmax. Long-only, cash allowed, per-name cap 10%. **The 10% cap is not a
  config key** — it is a Python default (`panel_env.py:49`) plus a hardcoded
  fallback at `scripts/paper_run.py:274` reading an `env.max_weight_per_name`
  that `configs/env/panel_daily.yaml` never declares (A0 §3.3). Changing it
  requires editing two source files. The cap also interacts with `MomentumTopK`:
  at K=5 it forces ~47% cash (`03_environment.md` § Baselines).
- **Alternatives:** Per-ticker {-1, 0, +1} discrete; Dirichlet; long/short.
- **Why the default:** Easiest to train, matches how real portfolios
  are rebalanced.

## Shorting and leverage

- **Default:** No shorts, no leverage. Gross = net = 100% max.
- **Alternatives:** Add shorts with stock-borrow fees, or cap leverage
  at 1.5×.
- **Why the default:** Shorting Indian cash equities has securities-
  lending constraints; leverage adds margin model complexity. Defer.

## Reward

- **Default in the code:** **plain log return minus turnover penalty** —
  `configs/env/panel_daily.yaml:5` `reward: log_return`,
  `turnover_penalty: 0.001` (A0 §2.5).
- **Intended default, never selected:** Differential Sharpe (Moody–Saffell)
  minus turnover penalty. `DifferentialSharpe` is implemented and unit-tested
  at `env/reward.py:18` and no config chooses it. **PLANNED, NOT
  IMPLEMENTED.**
- **Alternatives:** Sortino, CVaR, excess-over-equal-weight
  (`ExcessLogReturn` / `use_excess_returns`, also never switched on).
- **Why the intended default:** DSR gives a dense, risk-adjusted reward per
  step. CVaR is more principled but harder to train stably.
- **Caveat on the turnover term:** it does not measure turnover.
  `panel_env.py:286-288` computes both weight vectors from the same post-trade
  share vector, so the penalty tracks NAV drift, not trading. Measured in
  `03_environment.md` § Reward. **Choosing between reward functions is
  premature while the penalty term is mis-computed.**

## Rebalance frequency

Not previously a decision here, and it should have been.

- **Default in the code:** **daily**, never justified. The env steps one
  trading day and the agent re-allocates every step.
- **Why it matters more in India than anywhere else:** STT is 0.1% on **both**
  legs of delivery, and STCG is 20% + 4% cess under 12 months. Weekly cuts
  turnover 4–5×; monthly cuts it further and starts moving part of the book past
  the 12-month LTCG boundary at 12.5%.
- **Proposed in `09_revamp_and_audit.md` §4.1:** monthly default, with weekly
  and daily as ablations. **Not adopted here** — A3 reconciles, it does not
  decide.
- **The argument that was originally made for it is false.** §4.1 first claimed
  the agent was unfairly compared against monthly-rebalancing baselines paying
  less tax. `EqualWeightRebalanced` rebalances **every step**
  (`baselines.py:57-62`), so the baselines churn just as hard and pay the same
  drag (A0 §5, finding 16). The case for monthly stands on cost, not fairness.
- **UNVERIFIED:** the "~1.7× turnover ≈ 4 pp/yr of drag" figure. It has no
  regenerating artefact, and the turnover metric it derives from is the broken
  one above.

## Model architecture

- **Default in the code:** **`mlp_regime`** — TCN encoder (3 blocks, d=64) +
  cross-stock attention + FiLM regime conditioning + masked-softmax
  actor-critic. `use_graph: false` (`configs/model/mlp_regime.yaml:23`).
  **The GNN is not the default and never has been in a stored run.**
- **Intended default:** TCN encoder + Hetero GATv2 + masked-softmax
  actor-critic.
- **Alternatives:** Small Transformer encoder (**not reachable** —
  `model.encoder` is a dead key); GraphSAGE; no-graph MLP.
- **Why the intended default:** TCN is fast and sample-efficient for short
  lookbacks; GATv2 gives you learned sector relations after training.
- **Before choosing the GNN:** `graph.py:334` tiles `mask[0]` across the batch,
  so 87.6% of minibatch elements get the wrong adjacency (A1 §6), and
  `num_sectors` is hardcoded to 8 against 14 sectors (A1 §7.6). Both are
  described in `04_models.md`. A GNN-vs-MLP comparison cannot mean anything
  until they are fixed.
- `09_revamp_and_audit.md` §4.2 argues the whole RL path — PPO, actor-critic,
  FiLM and the GNN, ~2,500 lines — should become non-load-bearing until R6, in
  favour of supervised cross-sectional prediction. **Not adopted here**; it is a
  decision that belongs to R4, and it is recorded so it is a decision rather
  than a side effect.

## Graph structure

- **Default:** Three relations (intra-sector stock–stock, stock–sector
  membership, sector–sector). Learnable attention; optional correlation
  prior on edge features.
- **Alternatives:** Dynamic graph where edges are re-sampled per day
  from rolling correlations.
- **Why the default:** Static topology + learnable attention is stable
  and interpretable. Dynamic topology is research-scale work.

## Broker for live

- **Default:** Zerodha Kite Connect. **Market data is live and read-only**
  (`instruments`, `historical_data`); **execution remains paper-only** and
  `ZerodhaBroker` does not exist. Never call an order-placing Kite endpoint.
- **Alternatives:** Upstox, Fyers, Dhan.
- **Why the default:** Kite is the most widely used programmatic
  broker in India with a mature Python SDK.
- **Operational:** access tokens expire around 6 AM daily and need a browser
  login. Market data needs the paid Connect subscription — the free Personal
  tier returns `PermissionException`.
- `broker.type` and `broker.enabled` are both **dead keys**:
  `scripts/paper_run.py:298` constructs `PaperBroker` unconditionally, so
  `broker=zerodha` would still run the paper broker (A0 §3.1).

## Training schedule

- **Default:** Walk-forward, 4 windows, **3 seeds** per window
  (`configs/walk/default.yaml:11`, `seeds: [42, 43, 44]` — the "5 seeds" this
  section used to claim is not configured anywhere, and every stored run is
  seed 42). Warm-start across windows is a configurable toggle, off by default.
- **The purge between windows is 1 month, which is 19–23 trading days against
  60-day features.** 3 months is the minimum that clears; 4 gives margin
  (A1 §3.4). See `05_training.md`.
- `n_episodes: 5` inside evaluation is **not** five seeds and not five windows —
  it is five draws of action noise on one deterministic date range (A1 §7.2).
- **Alternatives:** Single train/val/test split (cheaper, less robust).
- **Why the default:** Walk-forward is the honest way to report
  out-of-sample performance.

## LLM / sentiment

- **Default:** Not in v1. Interfaces scaffolded, offline-only.
- **Alternatives:** Weekly FinBERT embeddings joined to features.
- **Why the default:** Sentiment is a known weak signal that's easy to
  overfit to. Defer until the price-only baseline is strong.

## "Real-time adaptation"

Your original brief asked for the agent to "learn and adapt in real
time." I'm splitting this into two things:

- **Default (what v1 does):** Scheduled walk-forward retraining — the
  agent is re-trained on a rolling window every quarter (or year). At
  inference time it does **not** update weights.
- **Alternative (v2+):** Online learning with a slow stream of gradient
  updates during paper trading.
- **Why the default:** Online updates on live PnL signal are a great
  way to destroy a model. Walk-forward retraining captures most of the
  benefit without the risk.

## Data start date

- **Default:** 2014-01-01 through latest available. 10 years of daily. This is
  what `data/panels` holds and what every run has used.
- **Already built, unreachable:** `data/panels_kite` starts **2005-01-03**
  (4,711 train dates). Kite depth was measured on 2026-09-04: 2005-01-03 for
  established names, listing date for newer ones (COALINDIA 2010-11, HAL
  2018-03, NYKAA 2021-11).
- **Two cautions found on review, before choosing a longer start:** 11.3% of
  rows in the 2005–2013 extension are under ₹20 against a ₹0.05 tick
  (BAJFINANCE oscillates ₹0.50↔₹1.50 through 2008–09 — ±50% steps of pure
  rounding noise), and pre-2010 microstructure had far wider spreads than the
  cost model assumes. **Print the first non-null date per ticker and the
  low-price row fraction before choosing.**
- **Why the current default:** 10 years covers multiple regimes (2015 selloff,
  2018 small-cap crash, 2020 COVID, 2022 rate cycle). Older data is
  noisier and less relevant to current microstructure.
- **For real depth**, NSE bhavcopy archives go back further and give
  point-in-time series codes (EQ / BE / T2T), which Kite cannot provide at all
  — and which the survivorship defect means this project needs.

## Explicit non-decisions

The following are not parameters — they are fixed by the plan and
should not be changed without a strong reason. **Four of the five are currently
not satisfied.** Status verified against `60ba1a6`:

| Non-decision | State |
|---|---|
| **No look-ahead. Ever.** | **VIOLATED, structurally.** The purge is 19–23 trading days against 60-day rolling features at 8 of 8 walk-forward boundaries and both fixed-split boundaries (A1 §3). Separately, `next_day_returns` is a genuine one-step-ahead label emitted in every observation (`panel_env.py:372,386`); it is *contained* — the only consumer is `ppo.py:387-388` for the auxiliary head, confirmed by call graph and by the shuffled-ticker arm — but the obs dict is passed wholesale into every forward method, so nothing but convention keeps it out (A1 §1). |
| **Masks, not zero-fill.** | **PARTIAL — it is mask *and* zero-fill.** Non-tradeable rows are sentinel-zeroed, not null (`alignment.py:20-25,77-79`): 11.1% of `data/panels` train rows carry `close == 0.0`. No zero-close row is ever tradeable, so the mask holds — but the invariant as written is not what the data does (A0 §2.2). |
| **Real transaction costs, always.** | **PARTIAL.** The cost model is now correct (`9019892`, ₹237.82 per ₹1L delivery round trip) and `03_environment.md` has been corrected to match. But `env/tax.py` exists and **is imported by nothing** — no backtest pays capital-gains tax — and all five stored MLflow runs predate the cost fix (A0 §4.2). |
| **Survivorship-bias-aware universe.** | **VIOLATED.** `market.universe_snapshots` holds 0 rows and has no read site; the one write site would back-stamp today's membership onto the start of history (A0 §2.4). `00_overview.md` calls this "the silent killer". It invalidates every backtest the project has produced. |
| **Baselines first; RL agent must beat them after costs.** | **VIOLATED both ways.** M5's gate never passed, and M6/M7/Phase 1/Phase 2 were built anyway. And the baselines themselves are wrong: `EqualWeightRebalanced` rebalances every step rather than monthly, and `MomentumTopK` runs at K=5 ranking on `log_return_1d` (A0 §5). |

If anything on this list starts to feel like a constraint worth
relaxing, that is usually a sign the project has drifted. **It has drifted.**

---

## Dead and mis-wired configuration

Every key in `configs/**/*.yaml` was traced to a read site in `src/` and
`scripts/` (A0 §3). These are the ones that do not do what they appear to.

**Declared and never read** — changing them silently does nothing:

| Key | File | Value | What happens instead |
|---|---|---|---|
| `env.transaction_cost_model` | `configs/env/panel_daily.yaml:4` | `zerodha_equity_delivery` | `runner.py:150-160` never passes `cost_model`; `panel_env.py:57` falls back to the same class, so the value happens to match |
| `broker.type` | `broker/paper.yaml:2`, `broker/zerodha.yaml:2` | `paper` / `zerodha` | `paper_run.py:298` constructs `PaperBroker` unconditionally |
| `broker.enabled` | `broker/zerodha.yaml:4` | `false` | nothing reads it; `ZerodhaBroker` does not exist |
| `model.encoder` | all five `configs/model/*.yaml` | `tcn` | `runner.py:177-197` hardcodes the TCN; `encoder: gru` is ignored |
| `model.graph.hidden_dim` | `gnn_v1.yaml:14`, `gnn_intra_only.yaml:14` | `64` | `runner.py:203-211` never reads it; the GNN uses `encoder.out_dim`. The config's own comment already says so |
| `data.universe_version` | `data/universe_v1.yaml:8` | `1` | nothing reads it |
| `data.universe_snapshot_path` | `data/universe_v1.yaml:9` | `data/raw/universe_v1.parquet` | nothing reads it; `build_universe.py:32` hardcodes the same path when writing. This is the survivorship dead end |

**Read, but not where the documents imply:**

| Key | Read at | Not read at | Consequence |
|---|---|---|---|
| `env.min_trade_value: 500` | `paper_run.py:279` → `paper_broker.py:1038` | `panel_env.py` — the string does not appear in the file | The backtest env has **no minimum trade size**; the broker drops every delta under ₹500. This, not share rounding, is the structural difference between the two execution paths |
| `data.panels_root` | `build_features.py:55` | `train.py:33`, `walk_forward.py:59`, `paper_run.py:266` — all hardcode `data/panels` | `data=kite_v1` builds into `data/panels_kite` and **no consumer can read it** |

**Read with no declared key:**

| Read site | Key | Default used |
|---|---|---|
| `paper_run.py:274` | `env.max_weight_per_name` | `0.10` — **not declared in `configs/env/panel_daily.yaml`.** The 10% per-name cap every document quotes exists only as a Python default (`panel_env.py:49`) |

**Declared but stale against the code:**

| Key | File | Value | Code truth |
|---|---|---|---|
| `model.graph.num_sectors` | `gnn_v1.yaml:13`, `gnn_intra_only.yaml:13` | `8` ("matches SECTOR_IDS in universe.py") | `SECTOR_IDS` runs 1..14. On a panel rebuilt from the current universe this raises `index 13 is out of bounds` in `CriticHead` and silently mis-routes edges in `HeteroGNN` (A1 §7.6) |

**Assume other keys are dead until traced.** The list above is A0's complete
sweep as of `60ba1a6`; it will drift again.

---

**When you've reviewed the above:** do **not** start at M0 in `07_roadmap.md`.
That sequence is superseded from M5 onward. Start at **R1** in
`09_revamp_and_audit.md` §5 — A0, A1 and A2 are complete and A3 (this
reconciliation) is done.
