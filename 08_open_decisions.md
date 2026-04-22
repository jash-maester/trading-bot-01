# 08 — Open Decisions

Defaults are set so Claude Code can start M0 today. If you disagree
with any default, override it before kicking off.

## Market scope

- **Default:** Indian equities only (NSE).

## Data frequency

- **Default:** Daily bars.
- **Alternatives:** 15-min or 1-min intraday.
- **Why the default:** Daily is enough to prove the architecture with
  10+ years of history. Intraday adds an order of magnitude to storage,
  features, and compute, and changes the slippage model materially.

## Universe size

- **Default:** NIFTY 50 + 20 per sector × ~6 sectors ≈ 130–160 unique
  tickers.
- **Alternatives:** top 500 by ADV; NIFTY 500 full.
- **Why the default:** Large enough to have sector structure; small
  enough that a Hetero GAT fits comfortably.

## Action space

- **Default:** Continuous target-allocation logits; env applies masked
  softmax. Long-only, cash allowed, per-name cap 10%.
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

- **Default:** Differential Sharpe (Moody–Saffell) minus turnover
  penalty.
- **Alternatives:** Log return, Sortino, CVaR.
- **Why the default:** DSR gives a dense, risk-adjusted reward per
  step. CVaR is more principled but harder to train stably.

## Model architecture

- **Default:** TCN encoder + Hetero GATv2 + masked-softmax actor-critic.
- **Alternatives:** Small Transformer encoder; GraphSAGE; no-graph
  MLP (kept as ablation, not main).
- **Why the default:** TCN is fast and sample-efficient for short
  lookbacks; GATv2 gives you learned sector relations after training.

## Graph structure

- **Default:** Three relations (intra-sector stock–stock, stock–sector
  membership, sector–sector). Learnable attention; optional correlation
  prior on edge features.
- **Alternatives:** Dynamic graph where edges are re-sampled per day
  from rolling correlations.
- **Why the default:** Static topology + learnable attention is stable
  and interpretable. Dynamic topology is research-scale work.

## Broker for live

- **Default:** Zerodha Kite Connect, stub only in v1.
- **Alternatives:** Upstox, Fyers, Dhan.
- **Why the default:** Kite is the most widely used programmatic
  broker in India with a mature Python SDK.

## Training schedule

- **Default:** Walk-forward, 4 windows, 5 seeds per window. Warm-start
  across windows is a configurable toggle, off by default.
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

- **Default:** 2014-01-01 through latest available. 10 years of daily.
- **Alternatives:** Go back to 2004 with Kaggle/NSE archive data,
  accepting lower data quality pre-2010.
- **Why the default:** 10 years covers multiple regimes (2015 selloff,
  2018 small-cap crash, 2020 COVID, 2022 rate cycle). Older data is
  noisier and less relevant to current microstructure.

## Explicit non-decisions

The following are not parameters — they are fixed by the plan and
should not be changed without a strong reason:

- No look-ahead. Ever.
- Masks, not zero-fill.
- Real transaction costs, always.
- Survivorship-bias-aware universe.
- Baselines first; RL agent must beat them after costs.

If anything on this list starts to feel like a constraint worth
relaxing, that is usually a sign the project has drifted.

---

**When you've reviewed the above, tell Claude Code to start at M0 in
`07_roadmap.md` and proceed milestone by milestone.**
