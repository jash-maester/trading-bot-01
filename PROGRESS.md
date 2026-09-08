# PROGRESS

**Last updated:** 2026-09-08 by session 3 (survivorship rebuild)
**Plan:** `09_revamp_and_audit.md` (process) · `10_architecture_revamp.md` (architecture)
**Rules:** `CLAUDE.md`

State file for the gated audit-and-rebuild programme. Written to be resumable
cold: a session that has never seen this repository should be able to read this
file, `09_revamp_and_audit.md` and `CLAUDE.md`, and know exactly what to do next.

---

## Unit ledger

Dependency order:

```
  A0 ──> A1 ──> A2 ──> A3 ──> A4* ──> A5 ──> R1 ──> R2 ──┬─> R4 ──> R5 ──> R6 ──> R7
                                                          └─> R3 (parallel with R2)
  * A4 is optional — see 09 §6
```

| Unit | Status | Gate | Evidence | Session |
|---|---|---|---|---|
| A0 — read-only forensics | **PASS** | Written answers to all 14 contradictions + 7 R0 questions | `audit/A0_findings.md` | 2 |
| A1 — leakage & correctness | **PASS** | Six checks answered with call graphs / measurements | `audit/A1_leakage.md` | 2 |
| A2 — compute forensics | **PASS (partial)** | Profile + measured H2D bytes + encoder-invocation ratio | `audit/A2_compute.md` | 2 |
| A3 — reconcile specs | **PASS** | Every spec carries a `## Status` block + verifying commit | `02/03/05/08_*.md` | 2 |
| A4 — quarantine (optional) | DEFERRED to before R6 | Untrained code moved to `experimental/`, tests still green | `experimental/*/README.md` | — |
| A5 — standing rules | **DONE** | `CLAUDE.md` exists with the five rules | `CLAUDE.md` | 1 |
| R1 — one panel, one truth | **PARTIAL → point-in-time panel BUILT** | Deterministic SHA256; every feature nonzero variance; purge ≥ lookback; point-in-time universe | `data/panels_bhav`: 4,138 days × 1,504 tickers from the full-market bhavcopy, back-adjusted from NSE's corporate-actions feed, eligibility decided strictly before each date. `audit/S1_SURVIVORSHIP.md`. The Kite panel (`data/panels_kite`, 645 fixed names) remains what every pre-2026-09-08 number was measured on. | 3 |
| R2 — honest baselines | **PARTIAL, and re-measured** | 5 baselines × 3 frequencies × 4 benchmarks, net of cost **and tax** | All 5 run at monthly + weekly, after tax, MLflow `baselines`. On the FIXED universe `MomentumTopK` wins at 0.273 vs equal-weight's 0.257. On the POINT-IN-TIME universe it collapses to **−0.002** and `equal_weight_frozen` (0.137) is strongest — `audit/S1_SURVIVORSHIP.md`. Daily cadence and the 4-benchmark leg still outstanding. | 3 |
| R3 — kill the compute bug | NOT_STARTED | 2M steps < 2h on the 4060, **conditional on encoder caching** | timed run + run ID | — |
| R4 — supervised cross-sectional | **PASS** (r4_v2) | Window-level: mean of per-window OOS rank IC > 0.02, window t > t_crit(95%), >= 75% windows positive (`12_gate_decision.md`) | `data/signal/r4_v2/gate.json`: 5d +0.0392 t 7.50, 20d +0.0437 t 5.31, 8/8 windows positive | 2026-09-06 |
| R5 — deterministic allocator | **RUN — FAIL on the point-in-time universe** | Beats best R2 baseline net of cost+tax, **paired bootstrap CI excluding zero** | PIT, 2005–2024, 13 windows: 4 of 49 arms clear vs `equal_weight_monthly`, 3 of 49 vs `equal_weight_quarterly`; best is K=20 band 0.010 quarterly, +0.0456/yr, CI [+0.0111, +0.0802], t 2.83. **The 2025–26 holdout clears 0 of 57 and ranks the four candidates in exactly reverse order**; quarterly lands at −0.0004. The 0.010 no-trade band replicates 8 of 8 cells and is the one robust result. Prior fixed-universe numbers were survivorship. `audit/R5_VERDICT_PIT.md`, `audit/S2_ALLOCATOR_ON_PIT.md`. | 3 |
| R6 — reinstate RL | **RUN — FAIL** | Beats R5's allocator | Loses out of sample. Diagnosed as 17 policy parameters against ~4 independent 2-year windows. | 3 |
| R7 — regime conditioning | NOT_STARTED | `corr(val,test)` CI over ≥8 windows excludes zero, then Phase 1 A/B | walk-forward summary | — |

## The survivorship rebuild — phase ledger

Started 2026-09-08 after measuring that the 504-name universe came from a 2026
instrument dump. `audit/S1_SURVIVORSHIP.md` has the numbers.

| # | Phase | Status | What it settles |
|---|---|---|---|
| — | Full-market bhavcopy 2010–2026 | **DONE** | 8.16M rows, 4,337 EQ/BE symbols, 0 failures |
| — | Corporate actions | **DONE** | 1,118 actions; 3 known splits verified end to end |
| 1 | Baselines on the PIT universe | **DONE** | Momentum-top-K: CAGR +0.273 → **−0.002** |
| 2 | Retrain R4 on the PIT universe | **RUNNING** | Whether any signal survives a real universe |
| 3 | R5 allocator + gate on PIT | armed, auto-fires after 2 | Does the edge survive? |
| 4 | Holdout 2025–26 on PIT | not started | Unseen-data confirmation |
| 5 | Paper-trading parity | not started | Backtest matches the broker tick-for-tick |
| 6 | Live readiness | not started | Auth, daily job, monitoring, kill-switch |

**NOTHING IS TRADEABLE YET.** Every number in the R4/R5 rows above was measured
on the fixed 504-name universe, and Phase 1 showed roughly 12 points of a ~0.26
CAGR long-only backtest on that universe was the universe itself. A model
trained on the point-in-time panel is a *different model* and does not exist
until Phase 2 finishes.

**Where R5 stands after 2026-09-08:**

1. **The paired bootstrap CI now exists, and R5 fails it.** Against
   `EqualWeightRebalanced` all 8 arms clear. Against `MomentumTopK` — which R2
   measured as the *stronger* baseline at monthly cadence, and which
   equal-weight itself loses to — only the unstopped K=20 arm clears, at
   t = 2.02 with a lower bound of +0.0039. R5's criterion says "best R2
   baseline", so the momentum comparison is the gate and the answer is FAIL.
   Recorded plainly per `CLAUDE.md`; the 8-of-8 equal-weight result is real but
   is not the gate.
2. **Every arm still beats momentum on average** (mean excess positive
   throughout, allocator CAGR 0.416 vs momentum 0.273). What fails is
   significance: a concentrated 20-name momentum book is volatile and the
   paired difference is wide. This is a power problem as much as an edge
   problem.
3. **The universe is still not point-in-time.** Measured on 2010–2020
   bhavcopy by `scripts/pit_universe_report.py`: of the names clearing ≥₹5cr
   median daily turnover, the **645 panelled names cover 60.0%** on average,
   and at 2016-01-01 twenty-seven of the ninety-nine missing had stopped
   trading altogether (ABIRLANUVO, ALBK, ANDHRABANK, AMTEKAUTO, CAIRN, FRL) —
   names no list drawn today can contain. Coverage against the 504 *traded*
   names reads 49.5%, but ~10pp of that is `INACTIVE_SECTORS` capping the
   observation width, which is a compute decision rather than survivorship; an
   earlier entry here quoted the conflated figure. This does NOT invalidate the
   R5 interval, which is paired and so largely cancels a shared universe bias;
   it bounds the LEVEL of every absolute number. The full-market bhavcopy
   backfill (`scripts/fetch_bhavcopy.py`, classic archive with ISIN back to
   2010) is the fix and is in progress.

<!-- superseded: -->
**Previously: A3 — reconcile the specs.** A0, A1 and A2 all pass; their
headline claims were independently re-verified rather than accepted on report.
A2 passes *partially*: four 4060 measurements are marked NOT MEASURED because the
box began refusing SSH mid-run and has not recovered. None of A2's conclusions
depend on them — the R3 verdict rests on the corrected FLOP count and the
measured FP32 peak, both in hand.

**To finish A2 later** (needs `Restart-Service sshd` or a reboot on the Windows
box first, which cannot be done from the Mac):
`make win-push`, then
`ssh jashm@192.168.1.7 'wsl.exe -e bash /mnt/d/trading-bot-01/scripts/profiling/a2_all.sh'`.

---

## Open questions for the human

1. **Total-return or price-return?** (`09` §2 row 3, §4.1.) Dividends are
   currently out of scope by decision, making the panel price-return. The
   measured yfinance-vs-Kite gap is strongly sector-correlated — COALINDIA
   8.06%/yr against BAJFINANCE 0.34%/yr, oil/gas/power ~4.0% against banking
   ~0.6% — so a price-return panel systematically under-ranks high-yield sectors
   in a model whose graph is built from sector membership. **Needs an explicit
   decision before R1 rebuilds the panel.**

2. **Data start date.** 2005 is available and measured, but 11.3% of rows in the
   2005–2013 extension are under ₹20 against a ₹0.05 tick (BAJFINANCE oscillates
   ₹0.50↔₹1.50 through 2008–09 — pure rounding noise), and pre-2010
   microstructure had far wider spreads than the cost model assumes. Start at
   2005, 2010, or 2014?

3. **Share rounding.** The env floors; the paper broker rounds to nearest
   (`09` §2 row 12). Suspected cause of most of the measured 11% NAV divergence.
   Which is correct?

4. **Sector holdout.** 141 names across 5 sectors are fetched and panelled but
   excluded from training via `INACTIVE_SECTORS`. Keep at 504, or widen to 645?

5. **A4 quarantine** — worth the refactor churn, or skip?

---

## Findings that change the plan

### Architectural (2026-09-05) — `10_architecture_revamp.md`

- **The policy class manufactures turnover.** A 505-dim Gaussian over logits
  (`actor_critic.py:242-247`) pushed through `masked_softmax`: exploration noise
  alone turns over **29% of NAV per day** at initial σ, costing **3–9 pp/yr** in
  verified delivery charges before tax. Even a confident policy cannot hold
  fewer than ~30% in names it does not want. This is a design property, not a
  hyperparameter, and it sits under every result ever recorded.
- **The critic cannot see the state** — `V(s)` conditions on `z.mean(dim=1)`
  over 504 embeddings. The project's own `ReturnPredictionHead` docstring
  (`heads.py:145-147`) says so.
- **The reward pays for beta** — raw log return in a 16.4%/yr bull-market panel;
  `ExcessLogReturn` exists (`reward.py:75`) and was never switched on.
- **The per-sector hierarchical idea addresses none of these** and A2 measured
  it as a 1.09× compute non-win. It belongs as a bounded "sector tilt" scalar
  inside a small-action RL layer in R6, not as a governor agent.
- **Consequence for the roadmap:** R4 becomes primary (supervised signal via the
  existing head), R5 a parameter-free allocator at monthly frequency, R6 RL over
  ~4–20 bounded allocator scalars with a Beta/squashed-Gaussian policy. The
  505-dim Gaussian policy is retired. New R8: feature expansion (NSE delivery %,
  FII/DII flows, bulk deals) gated behind R4's IC gate.

### From A0/A1/A2 (2026-09-05) — all verified independently

- **The good panel is unreachable.** `data.panels_root` is read only by
  `build_features.py:55`. `train.py:33`, `walk_forward.py:59`, `paper_run.py:266`
  and `evaluate.py:43` all hardcode `data/panels`, so `data/panels_kite/` is
  written and then orphaned. No config override can select it.
- **`num_sectors` will crash on rebuild — introduced by this session.** The
  universe now has 14 sectors; `num_sectors` defaults to 8 in `heads.py:181`,
  `actor_critic.py:30` and `graph.py:82`. Verified: `sector_id=14` →
  `RuntimeError: index 13 is out of bounds for dimension 1 with size 8`.
- **Dropout is inverted.** `ppo.py:218` calls `.eval()` before the rollout and
  `.train()` is only restored at `ppo.py:445`, *after* the whole loop. So every
  gradient update runs in eval mode (no dropout at all during training) and
  val/test evaluation runs in train mode (dropout on). Exactly backwards.
- **The purge is short on all 8 boundaries** (19–23 trading days vs 60-day
  features), and the new `kite_v1` split has it too despite a comment claiming
  otherwise. Minimum clean purge is 3 months; 4 gives margin.
- **`graph.py:334` is a correctness bug**, not a simplification: 87.6% (yfinance)
  / 94.3% (Kite) of minibatch elements get the wrong adjacency, swinging a fixed
  transition's `log_prob` by 0.68 nats against `clip_coef=0.2`. Every GNN result
  is uninterpretable, and this is a live alternative explanation for the
  instability the report attributes to dropout.
- **Both momentum baselines are wrong**: `MomentumTopK` ranks on `log_return_1d`
  (`baselines.py:110`, with a comment admitting it) and holds 47% cash at K=5.
- **`beta_nifty_60d` was never fixed in code.** `features.py:293` still does
  `pl.lit(1.0)` when the index is absent, and that file is untouched since
  2026-04-22. Only the Kite *config* fetches `^NSEI`. Any rebuild with a config
  lacking the index silently kills the feature again, with no error.
- **My compute arithmetic was wrong twice** — see the R3 gate in `09` §5.
  7.83 MFLOP/sequence not 3.3 (the TCN has 7 conv layers, not 3); 9.05 TFLOPS
  FP32 not ~15 (15.34 is TF32). The gate was below its own theoretical floor.
- **Host-to-device traffic is confirmed but irrelevant**: 27.79 GiB/update
  verified to 0.02%, but ~5.0 s/update — 0.0001% of wall clock. The real
  signature is a VRAM cliff; at 504 tickers **L=60 costs 24× L=30**. Encoder
  caching (TCN = 97.9% of arithmetic, 47× reduction) is the only fix that matters.

### From the pre-commit review of the plan (2026-09-04)

Recorded during the pre-commit review of `09_revamp_and_audit.md`. Full
verification log at `09` §9.

- **The universe is not point-in-time.** `market.universe_snapshots` has 0 rows
  and no read site anywhere in `src/`. Survivorship bias is live and always has
  been, despite `00_overview.md` listing it as a non-negotiable. **This
  invalidates every backtest the project has produced** and is now the
  highest-priority item in R1.

- **The purge gap is insufficient.** 22 trading days against 60-day rolling
  features leaves 38 trading days of overlap between train and val feature
  windows. Verified against `compute_windows()` on real dates.

- **The rollout-buffer diagnosis in the original draft was wrong.** The buffer is
  CPU-resident (`ppo.py:98,228,334`), not a VRAM overflow. The real cost is
  ~27.7 GB of host→device traffic per update. R3's prescribed fix is unchanged
  and still correct, but the reasoning had to be restated or A2 would find
  nothing wrong.

- **`MomentumTopK` K=5 holds 47.1% cash** under the 10% per-name cap. The
  momentum benchmark has been understated, and the agent still failed to clear
  it. K=20 gives 1.8% cash.

- **`corr(val,test) = −0.86` has a 95% CI of (−0.997, +0.583)** at n=4 windows —
  it includes zero. Phase 1 may be targeting a pathology that has not been shown
  to exist. R7 re-measures over ≥8 windows before anything is trained.

- **The plan was stale on 2026-09-04's work.** The cost model is fixed and
  verified to the paisa, the tax model exists (though unwired), `beta_nifty_60d`
  was found dead and fixed, corporate-action masking exists, the universe is
  645/504 from NSE's own classification, QuantStats is wired, and the paper
  broker is built. A0 should confirm these rather than rediscover them.

- **Target hardware changed twice.** The RTX 5090 is gone; an RTX 4060 8 GB is
  incoming. VRAM is not binding (~3 GB of activations at 504 names); **system RAM
  is** — the host-side rollout buffer is 6.9 GB, so ≥16 GB is required.

---

## Run queue

Full detail in `09_revamp_and_audit.md` §10. Nothing launched.

| # | Run | Est. | Gates on |
|---|---|---|---|
| Q1 | Config confirmation, 504/L=30/mb=64 | ~20 min | B1, B2 |
| Q2 | Panel rebuild (R1) | ~20–40 min CPU | B1, B2, B3, B7 |
| Q3 | Baseline table (R2) — **the bar** | ~2–4 h | Q2, B4, B5 |
| Q4 | Supervised rank IC (R4) — **go/no-go on signal** | ~1–2 h | Q2 |
| Q5 | Encoder-caching validation (R3) | ~1 h | Q4 |
| Q6 | Deterministic allocator (R5) | ~2–4 h | Q3, Q4 |
| Q7 | PPO 2M steps, 1 window 1 seed | ~7–8 h | Q1–Q6 |
| Q8 | Phase 1 A/B, 3 seeds × 2 arms | ~45–48 h | Q7 |
| Q9 | Full walk-forward | ~8–10 days | Q8 |

Q1 first: every estimate below it is extrapolated from a synthetic measurement on
a 72%-padded workload, and 20 minutes replaces all of them with a real number.

---

## Session log

### 2026-09-05 — session 2 (cont.) — B1–B7 fixed, architecture reviewed

B1–B7 landed (commit `46bf578`, 405 tests). Then a design-level review of the
core — not the bug list — found the three structural causes above. The finding
that reframes everything: the agent's *sampling noise* costs more per year than
the baseline's entire edge over cash. Written up as `10_architecture_revamp.md`.

**Next:** Q1 (config confirmation) is unblocked. R1 gains `use_excess_returns`
default-on. R4 is reframed as the primary model; see `10` §5–6.

### 2026-09-04 — session 2 — training box + A0/A1/A2 launched

Stood up the RTX 4060 box (`jashm@192.168.1.7`, `D:\\trading-bot-01`): key auth,
CUDA torch 2.11.0+cu130, 346 unit tests passing, Postgres + MLflow healthy under
a compose project now explicitly named `trading-bot` (it was defaulting to
`docker`, from the directory name). Automation is a set of `win-*` Makefile
targets plus `scripts/win_bootstrap.sh`.

Four bugs surfaced by running on a second machine rather than trusting the first:

- `--exclude='data/'` is unanchored, so rsync matched it at every level and
  silently omitted `src/trader/data/`. The same footgun was already present in
  the pre-existing `sync`/`sync-data` targets. Both anchored to `/data/`.
- `ruff check .` passed on macOS and reported 14 I001 errors on WSL from
  byte-identical files with the same ruff 0.15.11 — `src`-layout inference
  differs on a `/mnt/d` DrvFs mount. `known-first-party` pinned.
- Pinning `mlflow==3.11.1` was not enough: pip drifted starlette/anyio to a
  combination whose WSGI middleware raises `module 'anyio' has no attribute
  'from_thread'` on every request. Both pinned.
- Docker's `credsStore: desktop.exe` needs an interactive Windows logon and
  fails over SSH even for public images. Worked around with a project-local
  `DOCKER_CONFIG`.

**Blocker found for any measurement:** `active_tickers()` returns 504 but both
panels still contain 163 tickers. A smoke run on the box built a 505-wide action
space over mostly-empty columns — visible as `ent=-293.37` (log 505) against the
Mac's `-95.27` (log 164). Any A/B run in that state produces a number that has
to be thrown away. The 656-ticker raw store is complete and covers all 645
universe names, so the rebuild is unblocked but has not been run.

Also observed and passed to A0: `Feature stats: mean range [-0.05525, 1.288e+09]`
— a feature carrying values near 1.3 billion, almost certainly un-normalised
`dollar_volume_20`.

**Next:** A0/A1/A2 reports, then their gates.

### 2026-09-04 — session 1 — bootstrap

Created `09_revamp_and_audit.md`, `CLAUDE.md` and this file. Reviewed the
incoming plan against the live repository before committing to it rather than
adopting it as written.

**Verified and held:** insufficient purge gap; broken `MomentumTopK` K=5;
Fisher CI on `corr=−0.86` including zero; the FLOP arithmetic; dead
`min_trade_value`.

**Verified and refuted:** the rollout buffer exceeding 8 GB VRAM (it is
CPU-resident); the existence of a `docs/` directory (all specs are at root, and
paths were corrected throughout); VRAM as the 4060's binding constraint.

**Found beyond the plan:** the universe is not point-in-time — empty
`universe_snapshots`, no read site. Promoted to the top of R1.

**Corrections folded in:** R3's gate made conditional on encoder caching (the
arithmetic puts a 100%-utilisation floor at ~44 min, so 2 h is ~37% — reachable
with the cache, not without); A4 marked optional; §4.2's consequence stated
plainly (~2,500 lines go dormant until R6); a verification log added as §9.

**Next:** A0 — read-only forensics. No file may be modified except
`audit/A0_findings.md`. Do not fix anything found; recording it is the
deliverable.

**Not done, deliberately:** A0 was not started in this session, per the
bootstrap instruction to create the state files and stop.
