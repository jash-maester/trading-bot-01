# 03 — The Environment (`PanelTradingEnv`)

## Status

**PARTIAL** — last verified against `60ba1a6` (the A0/A1/A2 audit reports,
2026-09-05). The env, the masked-softmax action path, the fill model and the
cost model are implemented and correct. The cost model was corrected in
`9019892` and this document was corrected to match in A3.

Known broken:

- **The reward is not Differential Sharpe.** What runs is
  `log_return − 0.001 × turnover`. DSR is written, unit-tested and never
  selected (A0 §2.5).
- **`turnover` is not turnover.** `panel_env.py:286-288` computes both
  weight vectors from the *same post-trade* share vector, so the metric
  measures NAV drift, not trading. Measured below.
- **Three of the five baselines are not what this document called them.**
  `MomentumTopK` is broken twice — `K=5` leaves ~47% cash under the 10% cap,
  and it ranks on `log_return_1d`, not `log_return_20d`
  (`baselines.py:91,110`). `EqualWeightRebalanced` rebalances every step, not
  monthly (`baselines.py:57-62`). `BuyAndHoldIndex` holds equal weight across
  the whole universe from day 1 and tracks no index at all
  (`baselines.py:65-81`) — **found in A3, not covered by A0/A1/A2**.
- **The env has no minimum trade size** while the paper broker does, which is
  the structural divergence between the two execution paths (A0 §3.2).
- **`env.max_weight_per_name` is not a declared config key** — the 10% cap
  every document quotes exists only as a Python default (A0 §3.3).
- Vectorisation is `SyncVectorEnv`, not `AsyncVectorEnv`, and that does not
  matter — env stepping is 0.2% of wall clock (A2 §4.1).

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
        cost_model: CostModel | None = None,   # -> ZerodhaEquityDeliveryCostModel()
        reward_fn: RewardFn | None = None,     # -> LogReturn()   NOT DifferentialSharpe
        allow_short: bool = False,
        max_weight_per_name: float = 0.10,
        turnover_penalty: float = 0.0,     # added to reward fn
        use_excess_returns: bool = False,
        seed: int | None = None,
    ): ...
```

Signature as of `panel_env.py:39-58`. Both `cost_model` and `reward_fn` default
to `None` and are resolved inside `__init__` (`panel_env.py:57-58`), so an env
constructed without them silently gets the Zerodha delivery cost model and
`LogReturn` — which is what every training run has used.

> **DEFECT — `max_weight_per_name` cannot be set from the env config.**
> `configs/env/panel_daily.yaml` declares no such key. The 10% per-name cap
> exists only as this Python default and as a literal at
> `scripts/paper_run.py:274`, which reads `env.max_weight_per_name` with a
> hardcoded `0.10` fallback for a key nothing declares (A0 §3.3). Changing the
> cap requires a code edit in two places.

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

**Implemented as specified** — `panel_env.py:245-262`: target shares are
`floor(target_equity_value / open)`, the slippage formula matches, fill price is
`open * (1 + slip)`, and NAV is marked at `close_t` (`:281`).

> **DEFECT — the env has no minimum trade size; the paper broker does.**
> `env.min_trade_value: 500` (`configs/env/panel_daily.yaml:12`) is read only by
> `scripts/paper_run.py:279` → `PaperBrokerConfig` → `paper_broker.py:1038`. The
> string does not appear in `panel_env.py` at all. **This, not share rounding,
> is the structural difference between backtest and paper execution** — both
> floor to whole shares (`panel_env.py:247`, `paper_broker.py:1034`), refuting
> the long-standing claim in `06_paper_broker.md:158-159` that the broker rounds
> to nearest (A0 §1 row 12). The 11% NAV divergence between the two paths quoted
> in `ARCHITECTURE.md:147` has **no regenerating artefact in the repo —
> UNVERIFIED** (A0 §4.3).

## Cost model

Indian equity **delivery** (CNC), per the published schedule at
<https://zerodha.com/charges>. These constants are load-bearing and
India-specific.

| Component | Delivery rate | Legs |
|---|---|---|
| Brokerage | **0** — delivery is free at Zerodha | — |
| STT | **0.1%** of turnover | **both** buy and sell |
| Exchange transaction charge | **0.00307% NSE**, 0.00375% BSE | both |
| SEBI turnover fee | **₹10 per crore** (0.0001%) | both |
| GST | **18%** on (brokerage + SEBI + exchange) | both |
| Stamp duty | **0.015%** | buy only |
| DP charge | **₹15.34** per scrip per sell day | sell only |

The DP charge is ₹3.50 CDSL + ₹9.50 broker + ₹2.34 GST, levied once per distinct
scrip on any day that scrip is sold — not per order and not per share.

Implemented by `trader.env.costs.ZerodhaEquityDeliveryCostModel`
(`src/trader/env/costs.py`), which also carries the intraday (MIS) schedule and
the BSDA demat AMC slabs. `ZeroCostModel` exists for sanity-only runs and is
flagged as such in metrics.

> **Do not regenerate `src/trader/env/costs.py` from this table, or from any
> document.** `CLAUDE.md` rule 5. The constants live in the module; edit them
> there, in place, citing `zerodha.com/charges`. This section describes that
> module — it is not a specification the module should be rebuilt from. The
> previous version of this section described STT as sell-side-only, brokerage as
> `min(20, 0.0003·V)`, GST on (brokerage + exchange) only, and DP as ₹15.93.
> Regenerating the module from it reintroduces a **22.0% understatement of every
> delivery round trip**.

Reference values, reproduced by A0 against the live module (`uv run`,
`ZerodhaEquityDeliveryCostModel`):

```
DELIVERY  buy ₹1L = 118.74   sell ₹1L = 119.08   round trip = 237.82
INTRADAY  buy ₹1L =  30.34   sell ₹1L =  52.34   round trip =  82.68
the stale spec model above    round trip = 185.58   -> 78.0% of the truth
```

Capital-gains tax is modelled separately in `src/trader/env/tax.py` (added in
`9019892`): STCG 20% + 4% cess under 12 months, LTCG 12.5% beyond.

> **DEFECT — the tax model is not wired in.** `env/tax.py` is imported by
> nothing in `env/` or `training/` (A0, "three findings"). No backtest currently
> pays capital-gains tax. At the observed ~1.7× turnover the omission is worth
> roughly 4 pp/yr (`09_revamp_and_audit.md` §4.1 — **UNVERIFIED**, that figure
> has no regenerating artefact in the repo).

> **DEFECT — `env.transaction_cost_model` is a dead config key.**
> `configs/env/panel_daily.yaml:4` declares `zerodha_equity_delivery`, but
> `training/runner.py:150-160` never passes `cost_model` and `panel_env.py:57`
> falls back to `ZerodhaEquityDeliveryCostModel()`. The value happens to match,
> so changing it silently does nothing (A0 §3.1).

## Reward

### What is implemented

**Plain daily log return minus a turnover penalty.** `panel_env.py:294-297`:

```
reward_t = log(NAV_t / NAV_{t-1}) - lambda_turnover * turnover_t
```

Wiring: `configs/env/panel_daily.yaml:5` sets `reward: log_return`;
`training/runner.py:141-142` maps that through `_REWARD_MAP` (`runner.py:52-56`)
to `LogReturn` (`env/reward.py:65`), which is a pass-through.
`lambda_turnover = 0.001` (`configs/env/panel_daily.yaml:11`).
`use_excess_returns` is false (`:10`), so the reward input is the raw log
return, not an excess return over the equal-weight benchmark.

`ExcessLogReturn` (`env/reward.py:75`) exists and is functionally identical to
`LogReturn` — the env does the subtraction itself (`panel_env.py:292-293`) — and
has never been switched on. No stored run carries an `env.*` parameter at all
(A0 §4.2).

> **DEFECT — the `turnover` term does not measure turnover.**
> `panel_env.py:279` assigns `self._shares = target_shares` (post-trade) and
> `:286-288` then computes **both** weight vectors from that same post-trade
> share vector:
>
> ```
> prev_eq_w = (self._shares * closes) / self._prev_nav
> new_eq_w  = (self._shares * closes) / new_nav
> turnover  = sum(|new_eq_w - prev_eq_w|)
> ```
>
> The two differ only in the NAV denominator, so the quantity is
> `equity_value × |1/NAV_prev − 1/NAV_new|` — daily NAV drift scaled by equity
> exposure. It is uncorrelated with how much was traded.
>
> Measured in A3 (`PanelTradingEnv`, `data/panels/train.parquet`, 20 tickers,
> lookback 60, seed 0; step 1 rotates the entire book from 10 names to 19
> different names, step 2 repeats the same action so nothing rebalances):
>
> ```
> step 1  full rotation   costs = ₹2,116.55   turnover = 0.002727
> step 2  hold            costs =    ₹93.08   turnover = 0.004547
> ```
>
> The hold reports **1.7× the turnover of the full rotation** while paying 23×
> less in fees. Consequences: the `turnover_penalty: 0.001` term penalises
> volatility rather than trading, and the "Turnover (annualized, sum of |Δw|)"
> metric in `05_training.md` and the ~1.7× turnover figure quoted in
> `09_revamp_and_audit.md` §4.1 both derive from this quantity. **Not covered by
> A0/A1/A2 — found in A3.** Fixing it means capturing the pre-trade weights
> before `:279`.

### Differential Sharpe — PLANNED, NOT IMPLEMENTED

The intended default was the **differential Sharpe ratio** (Moody & Saffell
2001) on daily log portfolio returns, minus turnover penalty:

```
r_t  = log(NAV_t / NAV_{t-1})              # net of costs
DSR  = (B_{t-1} * delta_A - 0.5 * A_{t-1} * delta_B) / denom
where A, B are EMA of r and r^2, delta_* their updates.

reward = DSR_t - lambda_turnover * turnover_t
```

`DifferentialSharpe` is implemented and unit-tested at `env/reward.py:18` and is
**never selected**: no config sets `reward: differential_sharpe`, and no run has
ever used it (A0 §2.5). Every result the project has produced was trained on
plain log return. Treat any claim that this system optimises a risk-adjusted
objective as false until a config selects it and a run logs it.

Alternatives (ablate, don't default):
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

`gymnasium.vector.SyncVectorEnv` across **16** independent starting dates within
the training window (`training/runner.py:18,171`). This gives batch diversity
for PPO.

This document previously specified `AsyncVectorEnv`. **Do not pursue it.**
Environment stepping is **0.2%** of wall clock on the M4 (`CLAUDE.md`) and
**0.003%** of a PPO update at 504 tickers, against 99.2% for forward+backward
(A2 §4.1). Parallelising the env cannot buy more than a rounding error, and the
`AsyncVectorEnv` pickling constraints are a real cost for no measurable gain.

## Baselines (implemented in `env/baselines.py`)

All baselines are agents that emit the same action-logit vector, so
they use the same env and the same metrics pipeline:

1. **BuyAndHoldIndex** — **equal weight across the whole tradeable universe on
   the first day of the episode, never rebalanced** (`baselines.py:65-81`,
   docstring "Buy equal weight on first tradeable day, never rebalance"). It
   does **not** follow NIFTY 50 composition and it is not monthly — it holds no
   index at all. As a benchmark it is "buy-and-hold equal weight", which is a
   reasonable thing to have but is not the thing this line used to name.
   *(Not covered by A0/A1/A2 — found in A3.)*
2. **EqualWeightRebalanced** — `1/N_tradeable`, **rebalanced every step**.
3. **MomentumTopK** — **K=20** by trailing 20-day log return, equal weighted,
   rebalanced every 21 steps (`baselines.py:91,103`).
4. **SixtyFortyCash** — 60% equal-weight equity, 40% cash, rebalanced each step
   (`baselines.py:116-137`). Matches its description.
5. **RandomPolicy** — sanity floor; uniform random logits over tradeable names
   (`baselines.py:140-156`). Matches its description.

**Three of the five baselines are not what this document called them.** Only
`SixtyFortyCash` and `RandomPolicy` survive unchanged. Since the whole point of
the baseline set is to be the bar the agent must clear, none of the historical
"the agent failed to beat X" statements means what it appears to.

The RL agent has to beat these after costs on the test split to be
considered a real result. It has never done so — M5's gate ("matches or beats
`EqualWeightRebalanced` on the val split over 3 seeds") has never passed.

> **DEFECT — `EqualWeightRebalanced` rebalances every step, not monthly.**
> `baselines.py:57-62` returns fresh equal-weight logits on **every** `act()`
> call with no rebalance counter; its own docstring says "rebalanced each step".
> The baselines therefore churn as hard as the agent and pay the same turnover
> drag. Any argument that the agent is unfairly compared against
> lower-turnover, lower-tax baselines is **false** — `09_revamp_and_audit.md`
> §4.1 originally rested on it and has been corrected (A0 §5, finding 16).

> **DEFECT 1 of 2 in `MomentumTopK` — K=5 leaves the baseline ~47% in cash.**
> `baselines.py:91` defaults `k=5`. Five names at equal weight is 20% each,
> which the 10% per-name cap forbids, so the cap-and-renormalise step dumps the
> residual into cash. Measured against the real `masked_softmax` path:
>
> ```
> k=5 : cash= 47.1%  equity= 52.9%  max=10.0%
> k=10: cash=  3.5%  equity= 96.5%  max= 9.6%
> k=20: cash=  1.8%  equity= 98.2%  max= 4.9%
> ```
>
> **The specified value is K=20.** At K=5 the "momentum" baseline is roughly a
> half-cash portfolio, which is not the bar anyone intended to clear.

> **DEFECT 2 of 2 — it does not rank on momentum.** `baselines.py:110` scores
> names with `features[-1, :, 0]`, and `FEATURE_COLS[0]` is `log_return_1d`
> (`data/features.py:16-17`), not `log_return_20d` (`FEATURE_COLS[2]`). The
> comment at `baselines.py:105-109` admits the shortcut: *"We need
> log_return_20d, but we don't know its feature index here. Fall back to the
> last row of features col 0 as a proxy score."* So "momentum top-K" has been a
> **one-day** cross-sectional reversal signal for its entire life. Any statement
> that the agent failed to beat momentum is a statement about the wrong baseline
> (A0 §5 finding 15, A1 §7.11). The class docstring at `baselines.py:85-88`
> still claims the 20-day feature; the docstring is the intent, the code is not.
>
> The two defects are independent: fixing K without fixing the ranking gives a
> 20-name one-day-reversal basket.

## Invariants / tests

- Sum of post-softmax weights is 1.0 to within 1e-6 after masking.
- Masked names have zero weight in every step.
- If the action is the current allocation, turnover ≈ 0 and cost ≈ 0.
- If cash weight is 1.0 on every step, reward equals 0 and NAV is flat minus
  zero fees. (With `LogReturn` this is exact; the parenthetical about DSR being
  "defined to be ≈0" applied to a reward function that is not in use.)
- No NaN in obs or reward across a full training window.
- Seeded `reset()` calls produce identical trajectories.

## Acceptance criteria for Phase 2

| Criterion | State |
|---|---|
| `pytest tests/integration/test_panel_env.py` passes all invariants | **PASSES** as a suite (368 tests green), but no test exercises the `turnover` invariant, and the defect above shows why that matters |
| A random policy and the 4 baselines produce sensible metrics on the train split, logged to MLflow | **FAILS as stated** — two of the five baselines are wrong (see defects above), so "sensible" cannot be assessed. MLflow holds 5 runs, none of them baseline-only (A0 §4.2) |
| Walltime for a full 252-day episode rollout under 200 ms on CPU for 150 tickers | **UNVERIFIED** — never measured as stated. A2 §4.1 measures env stepping at 504 tickers as 0.027 s across 512 synthetic steps, a lower bound, and 0.003% of a PPO update. The criterion's premise ("so vectorization actually helps the GPU") is void either way: env stepping is not on the critical path |

A criterion this phase should have had: **the reported `turnover` responds to
trading**. A single test that rotates the book and asserts turnover ≈ 2.0 would
have caught the defect above.
