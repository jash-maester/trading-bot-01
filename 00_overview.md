# 00 — Overview

## Goal

A local, paper-trading-capable reinforcement learning system that learns a
daily portfolio-allocation policy over an Indian equity universe
(NIFTY 50 + sector-top stocks), using a graph-aware neural network over
intra-sector and inter-sector relationships, trained on 10+ years of
historical data, with a Postgres-backed simulated broker and stubbed hooks
for live Zerodha Kite trading.

## Non-goals (v1)

- Real-money live trading. (Interfaces only; disabled by default.)
- Sub-minute / tick-level data. (Daily bars, with room to extend later.)
- LLM-in-the-loop at inference time. (Sentiment, if used, is pre-computed
  offline into static per-ticker embeddings.)
- Multi-user, multi-account, hardened production deployment.

## System diagram

```
+------------------------+
|    Data pipeline       |   yfinance  ->  primary
|  (Parquet + Postgres)  |   Kaggle/HF ->  backfill / delisted
|                        |   Zerodha   ->  live adapter stub
+-----------+------------+
            |
            v                     historical panel +
+------------------------+        feature tensors +
|   PanelTradingEnv      |        is_tradeable mask
|   (Gymnasium.Env)      |  <-- observation, reward, done
|                        |  --> action = target allocation vector
+-----------+------------+
            |
            v
+------------------------+
|   Policy / Value net   |   per-ticker TS encoder (TCN)
|   ActorCritic          |   -> Hetero Graph Attention Network
|                        |   -> masked softmax allocation head
+-----------+------------+
            |
            v
+------------------------+
|   RL trainer (PPO)     |   walk-forward windows
|   MLflow tracking      |   baselines to beat
+-----------+------------+
            |
            v
+------------------------+
|   Broker interface     |
|   PaperBroker  (PG)    |   fills, slippage, fees (STT/GST/...)
|   ZerodhaBroker stub   |   interface only, no creds required
+------------------------+
```

## Design principles

1. **Strict temporal separation.** No look-ahead. Features at day `t` use
   only data with timestamp `<= t - 1` for actions executed on day `t`.
2. **Universe as of date.** Historical universe snapshots prevent
   survivorship bias; see `02_data_pipeline.md`.
3. **Mask, don't impute.** Stocks not yet listed (or delisted) are masked
   out of the observation and action space for the relevant dates. No
   zero-filling.
4. **Broker-agnostic.** `Broker` ABC with `PaperBroker` and
   `ZerodhaBroker` (stub). Live adapter is behind a flag that is off by
   default.
5. **Reproducibility.** Seeds pinned, data snapshotted with a hash,
   `uv lock` for deps, MLflow for runs, dockerized Postgres.
6. **Baselines first.** Buy-and-hold NIFTY 50, equal-weight monthly
   rebalance, momentum top-5, and 60/40 are implemented and measured
   before the RL agent. The RL agent has to beat them net of costs.
7. **One end-to-end path before specialization.** A minimal flat-feature
   PPO run must work on the full pipeline before GNN / sentiment /
   hierarchical heads are added.

## Honest caveats — read before building

- **Zero-filling non-existent stocks poisons the agent.** Use the mask.
- **Graph edges-as-actions** would make credit assignment a nightmare; a
  Graph Attention Network with learnable attention trained by backprop
  gives you the same "learned sector relations" interpretation cleanly.
- **Two separate RL agents** doubles debugging pain. Start hierarchical
  within one actor-critic; split only if v1 is bottlenecked and you can
  articulate why.
- **Finance LLM per step** is latency-prohibitive and adds non-stationary
  noise. Use it offline, weekly, to produce static embeddings per
  (ticker, week) if at all.
- **Survivorship bias** is the silent killer. Selecting "today's top 20
  per sector" and backtesting on 10 years is selecting winners after the
  fact.
- **Transaction costs, STT, GST, stamp duty, SEBI charges, slippage,
  impact** must be modeled, or the agent will learn to churn.
- **Indian equities settle T+1** (cash segment). The paper broker models
  this; allocations become positions at next-day open with realistic
  costs.
- **A well-built RL trader frequently loses to a cheap momentum or 60/40
  baseline.** That is the yardstick, not "does it make money."

## Assumptions (flag these in `08_open_decisions.md`)

1. Indian equities (NSE) primary; yfinance with `.NS` suffix.
   US / global tickers are an optional extension.
2. Daily bars for training.
3. Long-only, no leverage, no shorting in v1.
4. Broker for live = Zerodha Kite Connect.
5. PyTorch with CUDA 12.8+ for RTX 5090 (Blackwell) — nightly or 2.6+
   release as of early 2026.
6. Cash allocation is a valid "position" — the agent can go to cash.

## Phases

- **P0** Setup + infrastructure ................. `01_setup_and_infrastructure.md`
- **P1** Data pipeline + universe ................ `02_data_pipeline.md`
- **P2** Gym environment + baselines ............. `03_environment.md`
- **P3** Model: encoder + GNN + actor-critic ..... `04_models.md`
- **P4** RL training + walk-forward evaluation ... `05_training.md`
- **P5** Paper broker + live stub ................ `06_paper_broker.md`
- **P6** End-to-end runs + reports

Concrete milestones with acceptance criteria are in `07_roadmap.md`.
Decisions you should lock before starting are in `08_open_decisions.md`.
