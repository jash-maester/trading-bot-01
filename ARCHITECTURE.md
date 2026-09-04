# ARCHITECTURE

**Written:** 2026-09-04
**Repo:** `trading-bot-01` — daily-bar, long-only portfolio allocation over NSE
equities, trained with PPO.
**Purpose of this file:** a single accurate picture of what the system *is*, what
it *isn't*, and where it is weak — written to be handed to a reviewer.

Read `HANDOFF.md` for machine/environment state and `07_roadmap.md` for the
milestone plan. This file is the architecture only.

> **The one line that matters most:** the agent has never beaten a naive
> equal-weight rebalance. See [Benchmark status](#benchmark-status) — the
> headline "16%" figure people quote is the *baseline's* return, not the
> model's.

---

## 1. System diagram

```
================================================================================
 TRADING-BOT-01  ARCHITECTURE                       as of 2026-09-04
 Daily-bar long-only portfolio allocation over NSE equities, trained with PPO
================================================================================

 ┌─ DATA ──────────────────────────────────────────────────────────────────────┐
 │                                                                             │
 │  Zerodha Kite Connect (historical daily bars, 2005-01-03 -> today)          │
 │    - split/bonus adjusted, NOT dividend adjusted  -> PRICE-return series    │
 │    - 3 req/s, 2000 CALENDAR-day cap per request                             │
 │         │                                                                   │
 │         v                                                                   │
 │  kite_symbols.py   rename / series-suffix / delisting resolution            │
 │         │          (LTIM->LTM, STLTECH->STLTECH-BE, MCDHOLDING=dead)        │
 │         v                                                                   │
 │  universe.py    645 tickers / 14 sectors (NSE Indices "Industry" column)    │
 │                 504 ACTIVE for training (5 sectors held out)                │
 │         │                                                                   │
 │         v                                                                   │
 │  alignment.py + NSE calendar  -> [T, N] panel + is_tradeable mask           │
 │         │                        (mask, never zero-fill)                    │
 │         v                                                                   │
 │  corporate_actions.py   demerger detection + masking (event day + 60d)      │
 │         │                                                                   │
 │         v                                                                   │
 │  features.py   15 columns, ALL derived from adj_close:                      │
 │     log_return_{1,5,20}d  realized_vol_{20,60}d  rsi_14                     │
 │     macd macd_signal macd_hist  bbw_20  z_close_20                          │
 │     volume_z_20  dollar_volume_20  atr_14  beta_nifty_60d                   │
 │         │                                                                   │
 │         +--> regime_features.py  -> 6-dim MARKET-WIDE regime vector         │
 │              mkt_vol_20d, breadth_20d, dispersion_20d,                      │
 │              trend_20d, acceleration, vol_of_vol_60d                        │
 │         v                                                                   │
 │  panels: train 2005->2023 | val 2024 | test 2025-01 -> 2026-09              │
 │          1-month purge gaps; norm stats from TRAIN ONLY                     │
 │                                                                             │
 │  [X] NO news. NO sentiment. NO fundamentals. NO order book. Price+volume    │
 │      only. data/sentiment.py was specified in M0 and never written.         │
 └─────────────────────────────────────────────────────────────────────────────┘
                                    │
 ┌─ ENVIRONMENT: PanelTradingEnv (Gymnasium) ──────────────────────────────────┐
 │                                                                             │
 │  OBS   features        [60, 504, 15]   <- 60-day lookback per stock         │
 │        mask            [504]           tradeable                            │
 │        sector_ids      [504]                                                │
 │        portfolio       [505]           current weights (cash at index 0)    │
 │        cash, nav, t_frac, recent_return_1d, recent_vol_20d                  │
 │        regime          [6]                                                  │
 │        next_day_returns[504]  <- LABEL ONLY, never read in forward pass     │
 │                                                                             │
 │  ACT   505 logits -> masked softmax -> target weights                       │
 │        long-only, cash allowed, 10% per-name cap                            │
 │                                                                             │
 │  FILL  next-day OPEN, integer shares (floor), slippage                      │
 │  COST  brokerage 0 | STT 0.1% BOTH legs | stamp 0.015% buy                  │
 │        exch 0.00307% | SEBI 1e-6 | GST 18% | DP 15.34/scrip/sell-day        │
 │  TAX   STCG 20% + 4% cess (<=12mo) | LTCG 12.5% above 1.25L/FY  [NOT WIRED] │
 │  REWARD  log_return - turnover_penalty                                      │
 │          (excess_log_return exists, OFF and never run)                      │
 └─────────────────────────────────────────────────────────────────────────────┘
                                    │
 ┌─ MODEL: ActorCritic  (~107k params) ────────────────────────────────────────┐
 │                                                                             │
 │   features [B, 60, 504, 15]                                                 │
 │        │                                                                    │
 │        v   reshape -> [B*504, 15, 60]                                       │
 │   TCN encoder  3 x Conv1d(64), k=3, dilated, dropout 0.1                    │
 │        │       -> per-stock embeddings [B, 504, 128]                        │
 │        │                                                                    │
 │        ├── FiLM(regime)  gamma,beta  identity-init      [Phase 1]           │
 │        v                                                                    │
 │   CrossStockAttention  MHA, 4 heads, over the 504 stock axis                │
 │        │                                                                    │
 │        ├── FiLM(regime)                                 [Phase 1]           │
 │        v                                                                    │
 │        ├──> actor  -> 505 logits -> masked softmax                          │
 │        ├──> critic (mean-pool + regime concat) -> V(s)  [Phase 1]           │
 │        └──> ReturnPredictionHead -> next-day returns    [Phase 2, OFF]      │
 │                                                                             │
 │   ALT PATH (unused in Phase 1/2): graph.py HeteroGAT                        │
 │        GATv2Conv over (stock-stock intra-sector),                           │
 │        (sector-contains-stock), (sector-sector) + DropEdge                  │
 │        [X] regime FiLM is NOT wired into this path                          │
 └─────────────────────────────────────────────────────────────────────────────┘
                                    │
 ┌─ TRAINING ──────────────────────────────────────────────────────────────────┐
 │  SyncVectorEnv(16)  ->  PPO (CleanRL-style, single file)                    │
 │     n_steps 256, batch 4096, 4 epochs x 32 minibatches = 128 grad steps     │
 │     gamma .995  lambda .95  clip .2  ent 1e-4  target-KL early stop         │
 │        │                                                                    │
 │        v                                                                    │
 │  walk_forward.py   4 windows x 3 seeds, 5y train / 12m val / 12m test       │
 │        │           + equal-weight arm + paired bootstrap CI                 │
 │        │           + corr(val,test)  <- THE decision metric                 │
 │        │           + shuffled-ticker leak check (flag, off)                 │
 │        v                                                                    │
 │  MLflow :5555   |  eval_metrics (8) + QuantStats (46 + tearsheet)           │
 └─────────────────────────────────────────────────────────────────────────────┘
                                    │
 ┌─ EXECUTION (built, never run against live data) ────────────────────────────┐
 │  PaperBroker -> Postgres ledger.*                                           │
 │    weights -> integer shares (floor, 500 min, 10% cap)                      │
 │    T+1 cash settlement | FIFO lots | costs + tax accrual                    │
 │    strategy_runs -> orders -> fills -> positions -> snapshots -> pnl_daily  │
 │  ZerodhaBroker (live orders)  [NOT BUILT — deliberately]                    │
 └─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Known flaws

Ordered by how much I'd want a reviewer to attack them.

| # | Flaw | Detail |
|---|---|---|
| 1 | **Action space** | 505-wide masked softmax, one flat policy. Credit assignment across 504 names from a single scalar reward. No hierarchy, no sector budgeting, no top-K concentration. |
| 2 | **Reward function** | `log_return - turnover_penalty`. No risk term beyond turnover, no drawdown penalty, no tax, no benchmark-relative term. A long-only agent in a +16%/yr market is rewarded for simply being invested. |
| 3 | **Horizon** | Single-step daily, `gamma=0.995` over a 252-day episode. Nothing encodes a holding-period objective, and ~1.7× turnover means essentially all gains are short-term, taxed at 20%. |
| 4 | **No exogenous information** | Price and volume only. No news, earnings, sentiment, macro, flows, or fundamentals. |
| 5 | **Model tiny relative to input** | 107k params against a `[B,60,504,15]` observation. Measured: **96% of wall clock is the backward pass**; memory-bandwidth bound, not capacity bound. |
| 6 | **`corr(val,test) = −0.86` unsolved** | Validation is *anti*-predictive; selecting on val picks the worst test performers. Phase 1/2 target this and have never been trained. |
| 7 | **No risk management** | No stop-loss, no vol targeting, no position sizing beyond softmax weights, no exposure limits beyond the 10% per-name cap. |
| 8 | **GNN path orphaned** | Regime conditioning not wired in. Per-batch tradeable mask still uses `mask[0]` tiled across the batch (`graph.py:334`). |
| 9 | **`min_trade_value: 500` is dead config** | Declared in `configs/env/panel_daily.yaml`, never read by the env. Measured **11% NAV divergence** between backtest and paper broker because of it. |
| 10 | **Long-only, no leverage, no shorts** | Structurally cannot profit in a down market beyond going to cash. |

---

## 3. Benchmark status

**Margin over benchmark: negative.** There is no version of this system that has
beaten a naive equal-weight rebalance.

| Agent | Split | Sharpe | CAGR | vs baseline |
|---|---|---|---|---|
| `equal_weight_rebalanced` | train | **1.04** | **+16.4%** | — *(this is the bar)* |
| r6 `gnn_intra_only` — best ever recorded | train | +0.031 | +7.3% | **−9.1 pp** |
| r6 `mlp_baseline` (completed) | train | −1.12 | −13.8% | **−30.2 pp** |
| r6 `mlp_baseline` | val | −1.66 | — | worse out of sample |

The best result in the project's history was an **incomplete, in-sample** run
reaching less than half the return of a trivial equal-weight rebalance. The only
completed run lost money. **No test-split evaluation has ever finished.**

### Those numbers are void

They were produced under conditions since found to be broken:

- **`beta_nifty_60d` was constant 1.0** across all 276,005 tradeable training
  rows — 1 of 15 features dead, and because the stored stats recorded mean 0.0
  for it, that channel acted as a fixed non-zero bias into every convolution.
- **Transaction costs understated by 22%** — STT was charged sell-side only
  when delivery incurs 0.1% on *both* legs.
- **Zero capital-gains tax** against ~1.7× annual turnover — worth ~3.4 pp/yr.
- Old 163-ticker universe on a dividend-adjusted (total-return) panel.

`data/walks/` is empty; no walk-forward summary exists. **The honest current
state is that there is no measured benchmark margin, in either direction.**

### The actual first milestone

Not "beat 16%". The system has never reached 16%. The first milestone is
**beating a naive equal-weight rebalance at all, on a test split, with correct
costs and taxes.**

---

## 4. What exists vs. what doesn't

**Built and tested:** data pipeline, universe/symbol resolution, corporate-action
detection, `PanelTradingEnv` + 5 baselines, cost model, tax model, TCN +
cross-attention actor-critic, regime FiLM (Phase 1), auxiliary return head
(Phase 2), hetero-GAT, PPO, walk-forward with bootstrap CI + `corr(val,test)` +
shuffle check, QuantStats reporting, paper broker + ledger, Kite read-only feed.

**Not built:** news/sentiment ingestion, hierarchical or multi-agent policy,
live order placement, `scripts/full_report.py` (M10), `docs/interpretation.md`.

**Built but never trained:** Phase 1 regime conditioning, Phase 2 auxiliary head.
Zero runs exist for either.

---

## 5. Compute reality

Measured on an Apple M4 Mac mini, 24 GB unified, MPS. There is no CUDA machine.

| Universe | est. steps/sec | 2M steps, one run |
|---|---:|---:|
| 163 (old) | ~14.8 | ~1.6 days |
| **504 (active)** | **~4.8** | **~4.8 days** |
| 645 (fetched) | ~3.7 | ~6.3 days |

Profiled split of one PPO update: **env stepping 0.2%, rollout forward 3.7%,
update forward+backward 96.2%**. Larger minibatches are *slower* per sample and
OOM at 512 — the observation, not the parameter count, sets the cost.

A single Phase 1 A/B (1 window × 3 seeds × 2 arms) at 504 names is roughly
**29 days** at 2M steps. This is the binding constraint on the whole programme.
