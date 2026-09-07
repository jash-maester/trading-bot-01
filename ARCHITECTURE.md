# ARCHITECTURE

**Written:** 2026-09-04 · **Rewritten:** 2026-09-07
**Repo:** `trading-bot-01` — daily-bar, long-only portfolio allocation over NSE
equities.
**Purpose of this file:** a single accurate picture of what the system *is*,
what it *isn't*, and where it is weak — written to be handed to a reviewer.

Read `HANDOFF.md` for machine/environment state, `10_architecture_revamp.md`
for why the design changed, `12_gate_decision.md` for how the signal is judged,
and `audit/P3_P4_P5_RESULTS.md` for the current measured numbers.

> **The one line that matters most has changed.** The previous version of this
> file opened: *"the agent has never beaten a naive equal-weight rebalance."*
> That is no longer true. A **supervised signal driving a deterministic
> allocator** beats equal-weight in sample, on a survivorship-restricted arm,
> and on a genuinely unseen 2025-26 holdout. The **PPO layer over that
> allocator does not**, and was measured and set aside.

---

## 0. What changed, and why the old file was wrong

Everything below the data layer was replaced between 2026-09-05 and 09-07. The
old architecture — one PPO policy emitting 505 softmax logits — was retired
after `10_architecture_revamp.md` diagnosed three structural causes for it
never beating the baseline:

1. **The policy class manufactured turnover.** Sampling a 505-dim Gaussian and
   softmaxing it meant exploration noise alone churned 29% of NAV per day,
   worth 3–9 pp/yr.
2. **The critic could not see the state.** `V(s)` conditioned on a mean-pool
   over 504 stock embeddings.
3. **The reward paid for beta.** Raw log return in a bull market rewards being
   long, not selecting.

The replacement separates *prediction* from *allocation*, so each can be
measured on its own terms. That separation is the single most important
architectural fact about the system now.

---

## 1. System diagram

```
================================================================================
 TRADING-BOT-01  ARCHITECTURE                        as of 2026-09-07
 Daily-bar long-only portfolio allocation over NSE equities
 SUPERVISED SIGNAL -> DETERMINISTIC ALLOCATOR -> ENV   (PPO layer: rejected)
================================================================================

 ┌─ DATA ──────────────────────────────────────────────────────────────────────┐
 │  Zerodha Kite Connect (daily bars, 2005-01-03 -> 2026-09-04)                │
 │    split/bonus adjusted, NOT dividend adjusted -> PRICE-return series       │
 │    3 req/s, 2000 CALENDAR-day cap per request                               │
 │         v                                                                   │
 │  kite_symbols.py   rename / series-suffix / delisting resolution            │
 │         v                                                                   │
 │  universe.py    645 fetched / 14 sectors; 504 ACTIVE for training           │
 │         v                                                                   │
 │  alignment.py + NSE calendar -> [T, N] panel + is_tradeable mask            │
 │  corporate_actions.py  demerger detection + masking (event day + 60d)       │
 │         v                                                                   │
 │  features.py   15 columns, all from adj_close/volume:                       │
 │     log_return_{1,5,20}d realized_vol_{20,60}d rsi_14 macd macd_signal      │
 │     macd_hist bbw_20 z_close_20 volume_z_20 dollar_volume_20 atr_14         │
 │     beta_nifty_60d          <- ALIVE since the 2026-09-05 rebuild           │
 │     + regime_features.py -> 6-dim market-wide regime vector                 │
 │         v                                                                   │
 │  panels_kite/  train 2005->2023-12 | val 2024-04->12 | test 2025-04->2026-09│
 │                3-MONTH purge gaps; norm stats from TRAIN ONLY               │
 │                full.parquet (no purge holes) drives the walk-forward        │
 │                                                                             │
 │  features_ext.py  delivery %, FII/DII flows, bulk deals   [BUILT, NEVER RUN]│
 │  [X] NO news, sentiment, fundamentals or order book.                        │
 └─────────────────────────────────────────────────────────────────────────────┘
                                    │
 ┌─ R4: SUPERVISED SIGNAL  (this is where the alpha is) ───────────────────────┐
 │  SignalModel = shared TCNEncoder + one ReturnPredictionHead per horizon     │
 │    in: [B, L=60, N=504, F=15]   out: r_hat_5d, r_hat_20d  per (date, name)  │
 │    target: forward log return over t+1..t+h, z-scored ACROSS the            │
 │            cross-section; unlabelled rows are NEVER zero-filled             │
 │    363,394 params, embed_dim 128, TCN 4x128, k=3, dropout 0.1               │
 │                                                                             │
 │  Trained by WALK-FORWARD, 8 windows, 5y train / 12m val / 12m test,         │
 │  3-month purge. Test segments carry a `lookback-1` WARM-UP PREFIX from      │
 │  inside the purge gap (unlabelled, unscored) so the first prediction lands  │
 │  on the first real OOS date instead of 59 days into it.                     │
 │                                                                             │
 │  GATE (12_gate_decision.md) — window-level, replaced 2026-09-06:            │
 │    mean of per-window OOS rank ICs > 0.02                                   │
 │    one-sample t on the window ICs > 95% critical value at n-1 df            │
 │    >= 75% of windows positive, >= 4 windows                                 │
 │  Artefacts: predictions.parquet | embeddings.npy | index.json | gate.json   │
 └─────────────────────────────────────────────────────────────────────────────┘
                                    │
 ┌─ R5: DETERMINISTIC ALLOCATOR  (no learned parameters at all) ───────────────┐
 │  allocate(r_hat, vol, mask, sector_ids, current_w, params) -> w[N+1]        │
 │    1. candidates = tradeable & finite r_hat & vol>0 & sector_id>0           │
 │    2. rank by r_hat, take top K                                             │
 │    3. weight proportional to 1/vol                                          │
 │    4. cap per name (10%), redistributing to uncapped names                  │
 │    5. cap per sector (25%), same                                            │
 │    6. cash floor                                                            │
 │    7. NO-TRADE BAND  (P3) — pin a name whose weight moved less than the     │
 │       band, solved JOINTLY with the budget via a sorted clearing prefix     │
 │    8. turnover budget — move partially along the segment to target          │
 │                                                                             │
 │  RISK OVERLAY (risk.py) — applied AFTER allocate, one arm at a time:        │
 │    vol target | drawdown brake | per-name stop + quarantine cooldown        │
 │    A stop may FORCE a trade on a scheduled hold day (step_weights(force=))  │
 └─────────────────────────────────────────────────────────────────────────────┘
                                    │
 ┌─ ENVIRONMENT: PanelTradingEnv (Gymnasium) ──────────────────────────────────┐
 │  OBS   features [60,504,15] | mask | sector_ids | portfolio [505]           │
 │        cash, nav, t_frac, recent_return_1d, recent_vol_20d | regime [6]     │
 │  ACT   step_weights(target_w) — explicit weights, NOT logits                │
 │        (the 505-logit softmax path survives only for the retired baselines) │
 │  CADENCE  rebalance_schedule daily | weekly | monthly; hold days carry the  │
 │        book, mark to market, charge nothing                                 │
 │  FILL  next-day OPEN, integer shares (floor), ATR/ADV slippage              │
 │        EXECUTION GATE — a trade happens only if BOTH:                       │
 │          |delta| >= 0.5 shares         (integrality)                        │
 │          |delta| * open >= min_trade_value = Rs 500   (economic)            │
 │        and a request that ROUNDS to the position already held is not an     │
 │        order. Both rules mirrored exactly in PaperBroker.                   │
 │  COST  brokerage 0 | STT 0.1% BOTH legs | stamp 0.015% buy | exch 0.00307%  │
 │        SEBI 1e-6 | GST 18% | DP Rs 15.34 per scrip per SELL DAY  <- FLAT,   │
 │        and 83-98% of all cost measured. Turnover is the WRONG instrument    │
 │        for it; `n_scrips_sold` is the right one.                            │
 │  TAX   STCG 20% + 4% cess | LTCG 12.5% above 1.25L/FY   [MODELLED, NOT      │
 │        WIRED INTO THE BACKTEST LOOP]                                        │
 │  REWARD  log_return | excess_log_return - turnover_penalty                  │
 └─────────────────────────────────────────────────────────────────────────────┘
                                    │
 ┌─ R6: PPO OVER THE ALLOCATOR   [BUILT, MEASURED, REJECTED] ──────────────────┐
 │  AllocatorEnv: one step = one REBALANCE PERIOD (monthly)                    │
 │    obs [42]: allocator state only — params, weight summary, sector          │
 │      exposure, realised turnover, drawdown, regime, time. NO per-stock      │
 │      tensor, so the critic can see the whole state.                         │
 │    act [17] = 3 + 14 sectors: K, turnover budget, cash floor, sector tilts  │
 │      Beta distribution per dim -> bounded by construction, analytic entropy │
 │      in ACTION space                                                        │
 │  Verdict: beats fixed parameters in sample by +1.5 CAGR pts, LOSES out of   │
 │  sample by 2.2-3.2 on both checkpoints. See section 3.3.                    │
 └─────────────────────────────────────────────────────────────────────────────┘
                                    │
 ┌─ EXECUTION (built, never run against live data) ────────────────────────────┐
 │  PaperBroker -> Postgres ledger.*                                           │
 │    weights -> integer shares (floor, Rs 500 min, 10% cap)                   │
 │    T+1 cash settlement | FIFO lots | costs + tax accrual                    │
 │  ZerodhaBroker (live orders)  [NOT BUILT — deliberately]                    │
 │  Kite is READ-ONLY: instruments, historical_data. Never an order endpoint.  │
 └─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Known flaws

Ordered by how much I'd want a reviewer to attack them. Items 1–5 in the
previous version of this file are gone; what replaced them is smaller and
better understood, which is the point of the rewrite.

| # | Flaw | Detail |
|---|---|---|
| 1 | **Survivorship is unbounded** | All 645 tickers share one last-tradeable date, 2026-09-04. **Zero delistings in 21 years.** That is a present-day instrument dump; companies that died are absent and their absence *cannot be measured from data containing no sample of them*. `market.universe_snapshots` is empty, has no read site, and its only write path inserts one row dated today. This is the single largest threat to every number in this repo. |
| 2 | **The holdout is spent, and it was 15 months** | 2025-06 to 2026-09, one regime, max drawdown 13%. Nothing in it resembles 2018 or 2020. Re-tuning against it now would convert a real test into a fitted one. |
| 3 | **Tax is modelled but not wired** | `tax.py` has FIFO lots, §111A/§112A and cess, and the backtest loop does not call it. At ~3.7x annual turnover essentially all gains are short-term at 20%, so every CAGR here is pre-tax and overstated by roughly a fifth. |
| 4 | **No exogenous information** | Price and volume only. `features_ext.py` (delivery %, FII/DII flows, bulk deals) is built and tested but its data was never fetched and it has never been run. |
| 5 | **Drawdowns are large and only partly controlled** | -53% unmitigated over 2016-24, -68% at Rs 1 lakh. The per-name stop cuts it to -38% (section 3.4), which is an improvement, not a solution. |
| 6 | **The gate's interval is not a 5% test** | Documented, measured (11.6% false-positive at h=5 under a persistent null) and mitigated by moving to a window-level t-test, but the underlying effective sample is ~35 days per window and no reweighting fixes that. |
| 7 | **`corr(val,test) = −0.86` never revisited** | Measured under the retired architecture. It has not been re-measured under the supervised signal, so it is neither refuted nor confirmed. |
| 8 | **Long-only, no leverage, no shorts** | Structurally cannot profit in a down market beyond going to cash. |
| 9 | **GNN path orphaned** | `graph.py` HeteroGAT is built, unwired, and unused by the current design. |
| 10 | **The 4.7-year-stale holdout model** | The holdout was traded with an encoder whose last training window ended 2021-12. A deployment would refit first, so the holdout number is a lower bound, not a fair estimate. |

**Fixed since the last version, and worth recording because each was expensive:**
`beta_nifty_60d` dead (rebuilt panel, all 15 features live); `min_trade_value`
unread by the env (wired, with an 11% NAV divergence behind it); shares moving
to target without the cash leg when a trade was suppressed; `floor()` deciding
*whether* a trade happens rather than its size, creating one-share phantom
sells; the every-window gate that could not pass a signal as good as the one it
guarded; turnover measuring NAV drift rather than traded value.

---

## 3. Benchmark status

**Margin over benchmark: positive, and measured out of sample.** This section
is a complete replacement — every figure in the previous version is void.

### 3.1 In sample, 2016-07 to 2024-06 (`oos_r4_v2`, monthly, 20d horizon)

| Strategy | K | Sharpe | CAGR | MaxDD | vs EW |
|---|---|---|---|---|---|
| `equal_weight` | — | 1.336 | 0.270 | -0.515 | — |
| `null_signal` (control) | 30 | 1.275 | 0.258 | -0.544 | **-0.012** |
| **allocator** | 20 | 1.906 | 0.479 | -0.558 | **+0.209** |
| **allocator** | 30 | 1.902 | 0.452 | -0.530 | **+0.182** |

The null-signal control — a random signal through the identical allocator,
masked to the same candidate set — lands *below* equal-weight. That is what
makes the gap attributable to the signal rather than to concentration or to the
allocator machinery.

### 3.2 Out of sample, 2025-06-27 to 2026-09-04 (the holdout, spent once)

| Strategy | K | Sharpe | CAGR | MaxDD | vs EW |
|---|---|---|---|---|---|
| `equal_weight` | — | 0.685 | 0.090 | -0.120 | — |
| `null_signal` (control) | 30 | 0.587 | 0.082 | -0.135 | -0.008 |
| **allocator** | 20 | **1.892** | **0.320** | -0.117 | **+0.230** |

At Rs 1 lakh, K=20: mean month **+2.32%**, median +2.35%, best +19.3%, worst
-7.0%, 11 of 15 months positive, monthly sd 6.02%. Before tax.

### 3.3 The RL layer, and why it was set aside

| Span | Arm | Sharpe | CAGR | Turnover |
|---|---|---|---|---|
| Train 2016-24 | fixed params | 2.468 | 0.582 | 6.47 |
| Train 2016-24 | RL policy | 2.553 | **0.597** | 5.26 |
| Holdout 2025-26 | fixed params | 1.256 | **0.188** | 6.41 |
| Holdout 2025-26 | RL (update 100) | 1.006 | 0.156 | 5.05 |
| Holdout 2025-26 | RL (update 50) | 1.048 | 0.166 | 5.64 |

It learned to trade ~20% less and nothing that generalised. The structural
reason is sample size, not tuning: the action space is 17 dims of which **14 are
per-sector tilts**, and the training span holds roughly **four non-overlapping
two-year windows**. 800 episodes are those four periods resampled ~200 times.
More steps would make this worse. The cheap next experiment is to delete the
sector tilts and retrain on three dimensions.

### 3.4 Risk overlays, 2016-24, monthly K=30

| Arm | Sharpe | CAGR | MaxDD | vs EW |
|---|---|---|---|---|
| none | 1.902 | 0.452 | -0.530 | +0.182 |
| vol target (0.15) | 1.771 | 0.318 | -0.492 | +0.048 |
| drawdown brake | 2.082 | 0.382 | **-0.353** | +0.112 |
| **per-name stop (15%)** | **2.167** | 0.426 | -0.379 | **+0.156** |

**The per-name stop is the best of the three**, and this refuted a prediction
made before the run. The argument against it was that the drawdown is only 1.5
points worse than the market's, so it is market-wide and a per-name instrument
should not help. That reasoning was wrong. Volatility targeting is the one that
does not work: it is worse than no control at all on Sharpe and costs 13 points
of return.

### 3.5 Survivorship arm (pre-registered decision rule)

| Arm | EW CAGR | Allocator CAGR | Control vs EW | Gap G |
|---|---|---|---|---|
| Unrestricted | 0.270 | 0.452 | -0.012 | +0.1815 |
| Restricted to pre-2015 listings | 0.266 | 0.373 | -0.027 | +0.1074 |

Rule declared before the run: PASS iff `G_restricted >= 0.5·G_unrestricted` and
`> 0`. Threshold +0.0907, measured +0.1074. **PASS** — the edge survives at 59%
of its unrestricted size, which rules out post-2014 listing selection as its
source. It does not address delisting bias, which remains unbounded.

---

## 4. What exists vs. what doesn't

**Built, tested and measured:** data pipeline; universe/symbol resolution;
corporate-action detection; `PanelTradingEnv` + 5 baselines; cost model
(delivery *and* intraday); tax model; supervised signal model + walk-forward +
window-level gate; deterministic allocator with no-trade band; risk overlays;
capital-capacity derivation (`sizing.py`); survivorship arm; frozen-model
inference onto unseen panels; PPO over the allocator; paper broker + Postgres
ledger; Kite read-only feed; MLflow + QuantStats.

**Built but never run:** `features_ext.py` and its three NSE sources (data never
fetched); embedding cache; regime FiLM; hetero-GAT.

**Not built:** news/sentiment ingestion; live order placement; point-in-time
universe; hierarchical or multi-agent policy; `scripts/full_report.py`.

**Deliberately not built:** any Kite order-placing path.

---

## 5. Compute reality

Training runs on an **RTX 4060 Laptop, 8 GiB, under WSL2**
(`jashm@192.168.1.7`, `D:\trading-bot-01`). Development is on an M4 Mac mini.

The binding constraint is **VRAM, not FLOPs**, and WSL2 oversubscribes silently
into host RAM rather than failing — so exceeding it does not OOM, it pages over
PCIe at 20–30x cost. Measured for the R4 signal model, N=504, L=60:

| batch_days | peak VRAM | s/step | verdict |
|---:|---:|---:|---|
| 4 | 1.81 GiB | 0.131 | fits |
| 8 | 3.57 GiB | 0.231 | fits |
| **12** | **5.36 GiB** | **0.345** | **chosen** |
| 16 | 7.12 GiB | 2.903 | spills |
| 32 | 14.20 GiB | 21.130 | spills |

The shipped default of 32 was a 60x penalty. Batch 16 spills despite reading
under 8 GiB, because the display holds ~0.5 GiB.

Reference wall clocks: full R4 walk-forward (8 windows) 2h47m; R5 allocator grid
~4 min; R6 PPO (20,000 periods, 104 updates) ~3 min; holdout inference ~7 s.

An unexplained 2.9x slowdown was observed between two identical R4 runs
(41s → 125s per epoch on bit-identical work, same losses, same epoch counts, no
throttling, host idle). Cause not identified; nvidia-smi inside WSL cannot see
Windows-side GPU consumers.
