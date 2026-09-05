# 10 — Architecture Revamp

**Written:** 2026-09-05, after A0–A3 and the B1–B7 fixes (commit `46bf578`)
**Supersedes:** `04_models.md` §"Actor and Critic Heads", `05_training.md` §"Algorithm",
and the architectural half of `09_revamp_and_audit.md` §4.2
**Question answered:** the agent has never beaten `EqualWeightRebalanced`. Is
that the approach, the implementation, or the apparatus — and what should be
built instead?

Every number carries `file:line`, a pasted command, or **UNVERIFIED**. The
simulations in §1 are reproducible from the snippets shown.

---

## 0. Verdict

**The agent never beat the baseline because its policy class manufactures
turnover by construction, its critic cannot see the state, and its reward pays
for market beta rather than selection.** Those three are design choices, not
bugs, and they sit underneath the twelve defects the audits found. Fixing the
defects (done) makes the *experiment* honest; it does not make the *architecture*
capable of winning.

**The idea was not wrong from the start. This instance of it was wrong in almost
every specific**, and it was pointed at the hardest possible version of the
problem: daily rebalancing of 500 Indian equities, net of 0.2% round-trip STT and
20% STCG, from 15 price-derived features. No policy architecture closes that gap.
The rebalance frequency and the feature set close it or nothing does.

**The per-sector hierarchical idea does not address any of the three causes.**
Fourteen Gaussian-over-logits sub-policies produce fourteen copies of the same
noise; the governor solves a 14-dimensional problem a heuristic already solves.
It is a plausible *later* refinement, not a fix.

**What to build:** the supervised cross-sectional model the project already
half-wrote (`ReturnPredictionHead`, whose own docstring makes this argument), a
deterministic allocator over it, monthly rebalancing, and richer features. RL
returns — if it returns — over a **small** action space with a policy class that
cannot churn.

---

## 1. Why it never beat equal-weight — three structural causes

### 1.1 The policy class manufactures turnover

The policy is a **505-dimensional independent Gaussian over logits**
(`actor_critic.py:242-247`: `dist = Normal(mean, std)`, `action = dist.sample()`),
and the env then applies `masked_softmax` to the sampled logits
(`panel_env.py:482-510`). The rollout executes the *sampled* action, not the mean
(`ppo.py:252`). Initial σ is `exp(-1.0) = 0.368` (`heads.py:114`).

Simulated with the real `masked_softmax`, 504 tickers, 10% cap, a "confident"
policy mean (top-20 at logit +4, rest 0):

```
policy MEAN portfolio: top-20 hold 69.3%   cash 0.02%   rest 30.7%
sigma=0.368: sampled vs mean  L1 dist mean=0.289  max=0.438
sigma=0.135: sampled vs mean  L1 dist mean=0.107  max=0.160
```

Two things in that output.

**Even a confident policy cannot concentrate.** Logit +4 on twenty names — a
strong preference — still leaves **30.7% smeared across the other 484**, because
softmax over 505 entries with a 10% cap has nowhere else to put mass. The policy
class structurally cannot express "hold these twenty and nothing else".

**Exploration alone turns over 29% of NAV every day.** The L1 distance between
the sampled portfolio and the policy mean *is* the turnover the sampling injects,
before the policy intends any trade. Priced at the verified delivery round-trip
cost (`₹237.82 per ₹2L traded`, `costs.py`, commit `9019892`):

| σ | daily turnover from noise | annual cost, delivery, before tax |
|---|---|---|
| 0.368 (init) | 0.289 | **8.7% / yr** |
| 0.135 (collapsed) | 0.107 | **3.2% / yr** |

The equal-weight baseline's entire CAGR on the old panel was 16.4%. **The policy
spends 3–9 points of that on its own sampling noise**, every year, in every
configuration. This is not a hyperparameter. It is what a Gaussian-over-logits
policy through a softmax *is*.

Three secondary consequences of the same design:

- **The importance ratio is a product of 505 terms.** `log_prob` is
  `dist.log_prob(action).sum(-1)` (`actor_critic.py:249`). Small per-dimension
  drift compounds multiplicatively, so `approx_kl` explodes on tiny weight
  changes. The smoke run hit `KL 0.386` and `0.734` against `target_kl=0.02` on
  its *first* epochs — PPO was doing one epoch and stopping, essentially not
  learning.
- **Softmax is shift-invariant, so the policy wastes a dimension** and the
  likelihood of "the same portfolio" depends on which of infinitely many logit
  vectors was sampled. The ratio is measuring something that is not portfolio
  change.
- **The entropy bonus is meaningless.** `ent_coef × entropy` sums Gaussian
  entropies in *logit* space (`actor_critic.py:250`). The logged `ent=-95.28` at
  164 dims and `-293` at 506 dims are exactly `(N+1) × H(σ)` — a function of the
  action count and σ, saying nothing about portfolio diversity. A policy
  concentrated in one stock and one spread across 500 have the same entropy at
  the same σ.

### 1.2 The critic cannot see the state

`CriticHead.forward` conditions `V(s)` on `z.mean(dim=1)` — the **mean over 504
per-stock embeddings** (`heads.py`, `z_mean = z.mean(dim=1)`) — plus seven
portfolio scalars and a 14-dim sector-exposure vector. The mean of 504
embeddings is nearly constant across states; the per-stock information the actor
uses is averaged away before the critic sees it.

The project's own `ReturnPredictionHead` docstring says this outright
(`heads.py:145-147`): *"PPO value loss sees only one scalar per batch element, so
the critic gradient pressure on per-stock representations is weak (mediated by
`mean(z, dim=1)`)."* A critic that cannot distinguish states produces advantages
that are noise, and PPO with noisy advantages learns the noise.

### 1.3 The reward pays for beta, not alpha

`reward = log_return − turnover_penalty × turnover` (`panel_env.py:317`,
`reward.py:65-72`). With `use_excess_returns=False` — the default, and the only
setting ever run — that is the raw daily log return of the portfolio.

The old panel spans 2014–2024, a market in which equal-weight compounded at
16.4%/yr. **An agent that does nothing but stay invested collects positive reward
on average every day.** The signal that distinguishes selection from exposure —
excess return over the universe — was implemented (`ExcessLogReturn`,
`reward.py:75`) and never switched on. The agent was never asked to beat the
market; it was asked to be in it, and the turnover penalty that should have
stopped it churning was measuring NAV drift (B4) until yesterday.

### 1.4 How the three compound

A policy that must churn, judged by a critic that cannot see, on a reward that
rewards being long. Under those conditions the *best available* behaviour is a
noisy equal-weight portfolio paying 3–9%/yr for the noise — which is exactly what
r6's `−1.12` train Sharpe against equal-weight's `1.04` looks like. The GNN,
FiLM, and auxiliary head were built on top of this. They could not have helped.

---

## 2. Was it wrong from the start?

Separate the idea from the instance.

**The idea** — learn a daily allocation over a cross-section from price history,
with sector structure — is defensible and has a literature. Not wrong.

**The instance** made six choices that were each individually enough to lose:

| # | Choice | Verdict |
|---|---|---|
| 1 | Gaussian-over-logits → softmax policy | **Wrong.** Known-bad for portfolios; Dirichlet or direct-weight policies are standard, and even those churn. §1.1 |
| 2 | Mean-pooled critic | **Wrong.** §1.2 |
| 3 | Raw log-return reward in a bull market | **Wrong.** §1.3 |
| 4 | **Daily** rebalancing in a 0.2%-round-trip, 20%-STCG market | **Wrong for any approach.** `09` §4.1 |
| 5 | GNN, FiLM, aux head built before the MLP baseline cleared its gate | Sequencing, not architecture — `09` §1 |
| 6 | Data layer: non-point-in-time universe, 22-day purge, dead beta, wrong turnover, 22% cost understatement | **Would have sunk any model.** A0–A3 |

Choices 4–6 are not RL problems; they would defeat a supervised model, a
momentum rule, or a human. Choices 1–3 are RL-specific and fixable. **So: RL was
not the wrong tool, but this RL was wrong in almost every specific, and it was
never once run on honest data.** The claim "the graph + RL approach is not
leading to anything meaningful" is true as an observation and unsupported as a
conclusion — no clean experiment has been run on which to conclude anything.

One thing *was* bad from the start, and it is the one that matters most for
profit: **choice 4.** Every architectural decision downstream inherits a problem
where the required edge exceeds what daily technical features can plausibly
supply.

---

## 3. The per-sector hierarchical idea, evaluated

The proposal: train one agent per sector over its ~40–120 names, plus a
meta-governor allocating across sectors; or ensemble them.

**What it would fix.** Action-space width per agent: 505 → 40–120. Sector
membership is genuine structure. Each sub-agent could be validated independently.

**What it would not fix — all three §1 causes:**

- **Each sub-agent is still a Gaussian-over-logits policy.** The noise turnover
  in §1.1 is per-logit; it does not shrink because there are fewer logits per
  head. Fourteen heads produce fourteen copies of the same churn.
- **Each sub-agent still gets one scalar reward per day**, now with *worse*
  attribution: its reward depends on a sector budget it did not choose, and the
  governor's reward depends on picks it did not make. Two levels of credit
  assignment, not one.
- **The governor's problem does not need learning.** Allocating across 14 sectors
  by inverse-vol, momentum, or equal-weight is a one-line heuristic that is hard
  to beat and impossible to overfit.

**And it is not a compute win.** Measured in A2 (`audit/A2_compute.md`): cost
per stock is flat in N (12,516 → 13,659 µs/stock from N=25 to N=300); splitting
300 names into 6 sector models gains **1.09×**. Run count multiplies instead —
14 sub-agents × windows × seeds + governor — and every level's environment
shifts as the other learns, so you also pay for alternating freeze-and-train.

**Where it does belong:** as a *later* refinement of the allocator in §5, once a
supervised signal exists — "which sector should the budget favour this month" is
a reasonable small RL problem (14 actions, monthly) *on top of* a working
per-stock ranking. That is R6's slot, not R1's.

---

## 4. What actually maximises profit — the levers, sized

Ordered by expected magnitude. Only the first two are measured; the rest are
literature-informed and marked as such.

| Lever | Expected | Basis |
|---|---|---|
| **Stop paying for noise** (policy class, §1.1) | **+3–9 pp/yr** | measured, this document |
| **Rebalance monthly, not daily** | **+2–4 pp/yr** | STT 0.1% both legs + STCG 20%; magnitude was previously derived from the broken turnover metric and needs re-measuring after B4 — **UNVERIFIED until Q3** |
| **Point-in-time universe** | not a return lever — a validity one | `09` §2.1; every backtest so far is invalid without it |
| **Richer features** | UNVERIFIED; literature suggests rank IC 0.02→0.05+ | NSE delivery %, FII/DII flows, bulk/block deals, corporate announcements — all free, structured, timestamped |
| **Allocator discipline** — inverse-vol sizing, sector caps, turnover budget | +1–3 pp/yr, UNVERIFIED | standard in practice; this is what R5 builds |
| **Model capacity / architecture** | ~0 until the above are done | the encoder is 107k params and bandwidth-bound; it is not the constraint |

The honest reading: **the top three levers are not model changes.** Two are
cost-side, one is data validity. The project has spent its effort on the bottom
row.

---

## 5. The revamped architecture

Same data layer (post-R1), same TCN encoder. What changes is everything above
the encoder.

```
 ┌─ SIGNAL (supervised, R4) ──────────────────────────────────────────────────┐
 │  TCN encoder  ->  per-stock embedding z_i  [N, 128]                        │
 │        │                                                                   │
 │        v                                                                   │
 │  ReturnPredictionHead (EXISTS: heads.py:139)                               │
 │     -> predicted cross-sectionally-standardised forward return r̂_i       │
 │        horizons: 5d and 20d                                                │
 │  loss: masked MSE on real forward returns — 504 labels/day, not 1        │
 │  eval: rank IC, ICIR per period, bootstrap CI       <- THE go/no-go gate  │
 └────────────────────────────────────────────────────────────────────────────┘
                                    │  r̂  [N]
 ┌─ ALLOCATOR (deterministic, R5) ────────────────────────────────────────────┐
 │  rank r̂  ->  top-K (K ≈ 20–40)                                            │
 │  size by inverse realised vol                                              │
 │  cap: 10% per name, 25% per sector                                         │
 │  turnover budget: max X% of NAV per rebalance                              │
 │  rebalance: MONTHLY (weekly / daily as ablations)                          │
 │  -> target weights w  [N+1]                                                │
 │  no learned parameters; every knob is a number you can read               │
 └────────────────────────────────────────────────────────────────────────────┘
                                    │  w
 ┌─ EXECUTION (unchanged) ────────────────────────────────────────────────────┐
 │  PanelTradingEnv  (corrected costs, corrected turnover, tax accrual)       │
 │  PaperBroker -> ledger.*                                                   │
 └────────────────────────────────────────────────────────────────────────────┘

 ┌─ RL, IF IT RETURNS (R6) ───────────────────────────────────────────────────┐
 │  Frozen encoder + frozen signal head (cached: [T, N, 128], ~670 MiB)      │
 │  Policy over a SMALL action space — the allocator's free parameters:      │
 │     K, turnover budget, cash fraction, sector tilt      (≈ 4–20 dims)     │
 │  Policy class: Beta / squashed-Gaussian over bounded scalars              │
 │     -> cannot churn; a sampled action is a slightly different K,          │
 │        not a slightly different 505-vector                                 │
 │  Reward: EXCESS log return vs equal-weight (exists: reward.py:75)          │
 │          − turnover_penalty (now real, B4)  − drawdown term                │
 │  Critic: sees the allocator state directly — no mean-pool over N          │
 │  Horizon: monthly decisions; 12–36 steps per episode, not 252             │
 └────────────────────────────────────────────────────────────────────────────┘
```

Why each block:

**Signal first, supervised.** Cross-sectional regression gives ~504 labelled
examples per day where PPO gives one scalar. The project's own aux-head docstring
(`heads.py:139-160`) makes exactly this argument and cites the literature
(UNREAL; Théate & Ernst 2021). It stopped at "auxiliary". Make it primary. The
evaluation — rank IC with a CI — is cheap, runs in minutes, and is the only
experiment that can return a clean *no*.

**Allocator without parameters.** Every knob is inspectable and every failure
mode is diagnosable. If this does not beat equal-weight net of cost and tax, no
learned allocator over the same signal will, and you have found out in hours.

**RL over the allocator, not over the stocks.** The action space shrinks from
505 to a handful of bounded scalars. A Beta or squashed-Gaussian over "K in
[10, 60]" can be sampled all day without generating turnover — the churn was
never a property of RL, it was a property of *what was being sampled*. The
critic sees the allocator's actual state. The reward is excess return, so
sitting long earns nothing. Decisions are monthly, so an episode is 12–36 steps
and credit assignment is tractable.

**Encoder caching becomes free.** Once the encoder is frozen for R4's signal, the
47× from A2 (`audit/A2_compute.md`) applies, and `504 × L=60` drops from a
spilling 17.9 GiB to comfortably under 8. The compute problem and the modelling
problem have the same fix.

---

## 6. Revised roadmap — delta to `09` §5

The gate structure holds. Three units change content.

**R1 — add** point-in-time universe handling (already top of the list), and
**switch the default reward to `use_excess_returns=True`** so no future run can
collect beta as reward by accident.

**R4 — promote, and reuse.** The supervised model is `TCNEncoder +
ReturnPredictionHead` trained standalone on 5d/20d cross-sectionally
standardised forward returns. Do not write a new model. Gate unchanged: OOS rank
IC > 0.02 across windows, bootstrap CI excluding zero. Add a **feature-liveness
check** to the gate — every input column nonzero variance — because B7 showed a
dead feature hides for months.

**R5 — specify the allocator** as in §5: top-K, inverse-vol, 10%/25% caps,
turnover budget, monthly. Ablate K ∈ {20, 30, 40} and frequency ∈ {daily,
weekly, monthly}. Gate unchanged.

**R6 — rewrite.** Replace "PPO learns the allocator's free parameters" with the
concrete spec in §5: frozen encoder and signal head, Beta/squashed-Gaussian
policy over 4–20 bounded scalars, excess-return reward, monthly steps, critic on
allocator state. **Retire the 505-dim Gaussian policy.** Keep `ActorHead` in
`experimental/` (A4) with a note pointing at §1.1.

**R7 — unchanged.** Regime conditioning re-enters only if the R7 correlation CI
excludes zero, and then as a FiLM on the *allocator* policy, not the encoder.

**New: R8 — feature expansion.** After R5 clears or fails cleanly: NSE delivery
percentage (MTO file), FII/DII daily flows, bulk/block deals, corporate
announcement calendar. All free, structured, timestamped. Re-run R4's IC gate
per feature group. This is the lever most likely to move the signal, and it is
gated behind proving the pipeline can measure a signal at all.

**Sector hierarchy → R6 refinement.** If R6's small-action RL clears, "sector
tilt" is one of its bounded scalars. If a per-sector *signal* model beats the
pooled one at R4's IC gate, use it. Neither needs a governor agent.

---

## 7. What to stop doing

- **Stop sampling 505 logits.** Every run that does so pays 3–9%/yr for nothing.
- **Stop rewarding raw log return** in a bull-market panel. Flip
  `use_excess_returns` on, permanently.
- **Stop rebalancing daily** as the default. It is the single most expensive
  choice in the system and it was never justified.
- **Stop adding encoder-side capacity** — GNN, FiLM, wider TCN — until R4 shows
  the encoder's *current* output carries signal. A2 measured the encoder at
  97.9% of compute; making it bigger makes the thing that is not the bottleneck
  slower.
- **Stop treating "not beating the baseline" as a result.** No experiment to
  date was run on honest data with a working reward. The first one that is, is
  Q3.

---

## 8. What this does not settle

- Whether 15 technical features carry enough cross-sectional signal to clear
  Indian transaction costs at monthly frequency. **R4 answers this; nothing
  else does.**
- The correct `turnover_penalty` now that turnover is real. It needs a run, not
  a guess (`configs/env/panel_daily.yaml:11`).
- Whether the TCN is the right encoder at all. It is adequate to *test for
  signal*; if R4 passes, that is when to ask.
- Total-return vs price-return. Dividends are out by decision; the measured
  sector-correlated gap (COALINDIA 8.06%/yr vs BAJFINANCE 0.34%) will bias
  sector ranking and should be revisited before any live P&L is reported.
