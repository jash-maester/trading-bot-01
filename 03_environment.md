# 03 — The Environment (`PanelTradingEnv`)

## Summary

A `gymnasium.Env` that steps one trading day at a time over a universe
of N tickers. The agent's action at day `t` is a **target portfolio
allocation** vector. The environment enforces the tradeable mask,
applies transaction costs and slippage, advances the panel by one day,
marks the portfolio at the new closing prices, and returns the reward.

The env is intentionally minimal — no feature engineering inside it.
Features are precomputed (see `02_data_pipeline.md`) and loaded by the
env at construction time.

## Constructor

```python
class PanelTradingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        panel_path: Path,                  # train/val/test.parquet
        universe: list[str],
        feature_columns: list[str],
        lookback: int = 60,                # trading days of history in obs
        episode_length: int = 252,         # 1 trading year per episode
        initial_cash: float = 1_000_000.0, # INR
        cost_model: CostModel = ...,
        reward_fn: RewardFn = DifferentialSharpe(window=60),
        allow_short: bool = False,
        max_weight_per_name: float = 0.10,
        turnover_penalty: float = 0.0,     # added to reward fn
        seed: int | None = None,
    ): ...
```

## Observation space

A `Dict` so architectures can pick what they need:

```python
spaces.Dict({
    "features": spaces.Box(
        low=-np.inf, high=np.inf,
        shape=(lookback, N, F), dtype=np.float32,
    ),                                # panel window up to day t-1
    "mask": spaces.MultiBinary(N),    # is_tradeable on day t
    "sector_ids": spaces.Box(
        low=0, high=S, shape=(N,), dtype=np.int32,
    ),
    "portfolio": spaces.Box(
        low=0.0, high=1.0,
        shape=(N + 1,), dtype=np.float32,
    ),                                # current weights, index 0 = cash
    "cash": spaces.Box(0, np.inf, shape=(), dtype=np.float32),
    "nav": spaces.Box(0, np.inf, shape=(), dtype=np.float32),
    "t_frac": spaces.Box(0, 1, shape=(), dtype=np.float32),  # progress
})
```

Feature windows end at day `t-1`; the action executed on day `t` is
filled at day `t` close (or day `t+1` open — see Fill model).

## Action space

**Continuous target-allocation logits**, shape `(N + 1,)`, index 0 is
cash. The env applies a **masked softmax** inside `step`:

```python
logits = clip(action, -10, 10)
logits = where(mask, logits, -inf)    # cash always allowed
target_w = softmax(logits)            # sums to 1
target_w = cap_and_renormalize(target_w, max_weight_per_name)
```

Why logits, not weights:
- Lets PPO use a Gaussian policy over an unconstrained space (easy,
  stable), while the env enforces the simplex constraint.
- Masking is clean (set logit to `-inf` for untradeable names).
- Avoids Dirichlet reparameterization pain.

Alternative considered and rejected for v1:
- Per-ticker discrete {sell, hold, buy} — too coarse, doesn't express
  allocation magnitude.
- Dirichlet action — fiddly to train and no meaningful gain.

v2 can add a **long/short** flavor by moving to real-valued target
weights in `[-max_w, +max_w]` with an L1 budget.

## Fill model (daily)

For a daily env, executing at "now" is an unrealistic optimism. Model:

1. Action arrives at end-of-day `t-1`.
2. Orders are submitted "at the open" of day `t`.
3. Fill price = `open_t * (1 + slippage_t)` where
   `slippage_t = k * (atr_14 / close) * sign(delta_qty) *
                 sqrt(|delta_qty| / adv_20)`.
4. Fees (see Cost model) are charged per leg.
5. Mark-to-close at `close_t` for reward computation.

This is realistic enough for Indian equities cash segment and matches
how the paper broker will execute during paper trading.

## Cost model

Per trade value `V`:

- **Brokerage**: min(20, 0.0003 * V) per order (Zerodha-like flat).
- **STT**: 0.001 * V on sell side (delivery equity).
- **Exchange txn charge**: 0.0000322 * V (NSE).
- **GST**: 0.18 * (brokerage + exchange charge).
- **SEBI**: 0.000001 * V.
- **Stamp duty** (buy only): 0.00015 * V.
- **DP charges** (on sell, delivery): flat ~15.93 INR per scrip per day.

Encapsulated in `trader.env.costs.ZerodhaEquityDeliveryCostModel`.
Swappable via config; `ZeroCostModel` exists for sanity-only runs and
is flagged as such in metrics.

## Reward

Default: **differential Sharpe ratio** (Moody & Saffell 2001) on daily
log portfolio returns, minus turnover penalty.

```
r_t  = log(NAV_t / NAV_{t-1})              # net of costs
DSR  = (B_{t-1} * delta_A - 0.5 * A_{t-1} * delta_B) / denom
where A, B are EMA of r and r^2, delta_* their updates.
```

Reward returned per step:
```
reward = DSR_t - lambda_turnover * turnover_t
```

Alternatives (ablate, don't default):
- Plain daily log return.
- Sortino-style downside-only penalty.
- CVaR-based (harder to train stably).

## `step` and `reset`

```python
def reset(self, seed=None, options=None):
    # Sample a start date from the valid window for this split,
    # reset portfolio to {cash: initial_cash, positions: 0},
    # return (obs, info).

def step(self, action: np.ndarray):
    # 1. Compute target weights from logits + mask.
    # 2. Compute target shares (integer-lot for realism).
    # 3. Execute fills at next-day open with slippage and costs.
    # 4. Mark-to-close, compute daily log return, update DSR state.
    # 5. Advance pointer by 1 trading day.
    # 6. Build next obs. Set terminated/truncated.
    # 7. Populate info:
    #    { "nav": ..., "turnover": ..., "gross_exposure": ...,
    #      "sector_exposure": {...}, "fills": [...],
    #      "weights": np.ndarray, "date": datetime,
    #      "costs_paid": float }
    return obs, reward, terminated, truncated, info
```

`terminated=True` only on failure conditions (NAV hits a hard floor
at 10% of initial cash — signals a broken run).
`truncated=True` at end of `episode_length` window.

## Vectorization

`gymnasium.vector.AsyncVectorEnv` across independent starting dates
within the training window. This gives batch diversity and is critical
for PPO stability.

## Baselines (implemented in `env/baselines.py`)

All baselines are agents that emit the same action-logit vector, so
they use the same env and the same metrics pipeline:

1. **BuyAndHoldIndex** — weights follow NIFTY 50 composition monthly.
2. **EqualWeightRebalanced** — `1/N_tradeable` monthly.
3. **MomentumTopK** — top K=5 by trailing 12-1 month return, equal
   weighted, rebalanced monthly.
4. **SixtyFortyCash** — 60% equal-weight equity, 40% cash.
5. **RandomPolicy** — sanity floor.

The RL agent has to beat these after costs on the test split to be
considered a real result.

## Invariants / tests

- Sum of post-softmax weights is 1.0 to within 1e-6 after masking.
- Masked names have zero weight in every step.
- If the action is the current allocation, turnover ≈ 0 and cost ≈ 0.
- If cash weight is 1.0 on every step, reward equals 0 (no price risk,
  but DSR is defined to be ≈0) and NAV is flat minus zero fees.
- No NaN in obs or reward across a full training window.
- Seeded `reset()` calls produce identical trajectories.

## Acceptance criteria for Phase 2

- `pytest tests/integration/test_panel_env.py` passes all invariants.
- A random policy and the 4 baselines produce sensible metrics on the
  train split, logged to MLflow.
- Walltime for a full 252-day episode rollout is under 200 ms on CPU
  for a 150-ticker universe (so vectorization actually helps the GPU).
