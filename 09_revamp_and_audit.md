# 09 — Revamp Plan and Audit Protocol

**Written:** 2026-09-04
**Supersedes:** parts of `07_roadmap.md` (M5 onward) and `08_open_decisions.md`
**Status:** plan reviewed and corrected against the live repository on
2026-09-04. Every factual claim below carries either a `file:line`, a pasted
command output, or an explicit `UNVERIFIED` marker. See §9 for the verification
log — what was checked, what held, and what did not.

**Project state when written:** M5's acceptance criterion — *"matches or beats
`EqualWeightRebalanced` on the val split over 3 seeds"* — has never been met, and
M6, M7, Phase 1 and Phase 2 were built on top of it anyway.

> **Note on file locations.** The original draft of this plan referenced
> `docs/09_*.md`, `docs/PROGRESS.md` and `docs/audit/`. **There is no `docs/`
> directory in this repository** — all nine specs (`00_overview.md` …
> `08_open_decisions.md`), plus `HANDOFF.md` and `ARCHITECTURE.md`, live at the
> repo root. Paths below have been corrected to match. Audit reports go to
> `audit/` at root.

---

## 1. The finding that matters most

`07_roadmap.md` opens with: *"Sequence is deliberate. Do not jump ahead; later
milestones depend on invariants established earlier."*

M5's gate never opened. M6 (hetero GNN), M7 (walk-forward), Phase 1 (regime
FiLM, ~600 lines + 26 tests) and Phase 2 (aux head, ~200 lines + 18 tests) were
built regardless — roughly 2,500 lines of model code and 44 tests standing on a
gate that never passed.

This is not a code-quality problem. `ruff` is clean, `mypy` is clean, 368 tests
pass. It is a **sequencing** problem, and it compounds: every layer added on top
of an unvalidated base makes it harder to find out whether the base works.

The revamp below is mostly about unwinding that, not about new architecture.

---

## 2. Document contradictions

Cases where two of the project's own documents assert incompatible facts.
Resolve each before writing code — regenerating a module from a stale spec
re-injects bugs already fixed.

**Status column added on review.** Several rows were fixed in code on 2026-09-04
but remain wrong in the specs; those are marked `CODE FIXED / SPEC STALE` and
are the highest-risk kind, because the spec is now actively dangerous.

| # | Subject | `00`–`08` say | `ARCHITECTURE.md` says | Status |
|---|---|---|---|---|
| 1 | Data source | yfinance primary, Kaggle fallback, Zerodha `NotImplementedError` | Zerodha Kite Connect | **CODE CHANGED** — Kite is live and read-only; yfinance path still present |
| 2 | Date range | train 2014–2021, val 2022, test 2023–2024 | train 2005–2023, val 2024, test 2025-01–2026-09 | **CODE CHANGED** — both panels exist side by side |
| 3 | Price series | `auto_adjust=True` → total return | split/bonus only → price return | **UNRESOLVED — decide explicitly** |
| 4 | Universe size | 130–160 tickers | 645 fetched / 504 active | **CODE CHANGED** |
| 5 | Survivorship | `universe_snapshots`, "universe as of date", a stated non-negotiable | current NSE index membership | **CONFIRMED BROKEN — see §2.1** |
| 6 | Reward | Differential Sharpe − turnover penalty | `log_return − turnover_penalty` | **SPEC STALE** |
| 7 | STT | 0.001 sell side only | 0.1% both legs | **CODE FIXED / SPEC STALE** |
| 8 | Brokerage | `min(20, 0.0003·V)` | 0 (delivery is free) | **CODE FIXED / SPEC STALE** |
| 9 | DP charge | ₹15.93 | ₹15.34 | **CODE FIXED / SPEC STALE** |
| 10 | Vectorisation | `AsyncVectorEnv` | `SyncVectorEnv(16)` | **SPEC STALE** — and irrelevant; env stepping is 0.2% of wall clock |
| 11 | Rollout / entropy | `rollout_length: 512`, `ent_coef: 0.01` | `n_steps 256`, `ent 1e-4` | **SPEC STALE** |
| 12 | Share rounding | paper broker "nearest, not truncated" | env: floor | **UNRESOLVED — likely most of the 11% NAV divergence** |
| 13 | Target hardware | RTX 5090, CUDA 12.8 | M4 Mac mini, MPS | **BOTH STALE** — target is now RTX 4060 8 GB |
| 14 | Sector holdout | not mentioned | 5 of 14 sectors held out | **CODE CHANGED** — `INACTIVE_SECTORS` |

Rows 1–3 together mean **nobody currently knows what data the trained models
saw**: different source, different span, and a switch from total-return to
price-return targets partway through. Results from before and after are not
comparable.

Rows 7–9 are fixed in code (`src/trader/env/costs.py`, commit `9019892`,
verified exact: ₹1L buy + ₹1L sell = ₹237.82 delivery, ₹82.68 intraday) but
still wrong in `03_environment.md`. **Until that spec is corrected, any
regeneration of `costs.py` reintroduces a 22% cost understatement.**

### 2.1 Survivorship bias is live — confirmed

```
$ psql -c "select count(*) from market.universe_snapshots"
0
$ grep -rn "universe_snapshot" --include='*.py' src/ scripts/
src/trader/db/market_models.py:47:    __tablename__ = "universe_snapshots"
```

The table is **empty** and the only reference in the codebase is its own
definition — there is **no read site** in the environment or the panel builder.
The universe is not point-in-time. `00_overview.md` lists survivorship bias
among the "explicit non-decisions" and calls it *"the silent killer."*

This is arguably a larger finding than anything else in this document, and it
invalidates every backtest the project has produced.

---

## 3. Phase 1 may be solving a pathology that does not exist

`HANDOFF.md` §2 states the problem as `corr(val_sharpe, test_sharpe) = −0.86`,
diagnoses regime specialisation, and builds FiLM conditioning against it. The
engineering is careful — identity init so the A/B is clean, train-only regime
stats, dedicated leakage tests. It is good work.

Two things undercut it.

**The correlation was measured on runs `ARCHITECTURE.md` declares void** —
`beta_nifty_60d` constant at 1.0 acting as a fixed bias into every convolution,
costs understated 22%, zero capital-gains tax, old 163-ticker total-return panel.
A pathology measured under those conditions is not yet a pathology.

**With four windows it is indistinguishable from noise.** Fisher transform on
`r = −0.86`, `n = 4` (verified on review):

```
z    = atanh(-0.86)              = -1.293
SE   = 1/sqrt(n-3) = 1/sqrt(1)   =  1.000
95%  = -1.293 ± 1.96             = (-3.25, +0.67)
tanh                             = (-0.997, +0.583)
```

**The CI spans −1 to +0.58 and comfortably includes zero.** Seeds do not rescue
this: three seeds inside one window are not three independent observations of the
val→test relationship. The effective sample size is the number of *windows* —
four.

Before any Phase 1 A/B: re-measure on a fixed panel, with more windows, and
report the CI beside the point estimate. If the CI still includes zero, there is
nothing for FiLM to fix and Phase 1 should be **shelved rather than run**.

Keep the code. It is well built and cheap to revisit. Do not spend a month of
compute on it before confirming the target is real.

---

## 4. Decisions to amend in `08_open_decisions.md`

### 4.1 Rebalance frequency — promote to first-order

Currently daily, never justified. In Indian equities this is the single most
expensive choice in the system:

- STT is 0.1% on **both** legs of delivery.
- STCG is 20% + 4% cess under 12 months.
- At the observed ~1.7× turnover that is roughly **4 pp/yr of drag**.

Weekly cuts turnover 4–5×. Monthly cuts it further and starts moving part of the
book past the 12-month LTCG boundary at 12.5%. **That is worth 2–3 pp/yr — more
than any plausible model improvement.**

The baselines (`EqualWeightRebalanced`, `MomentumTopK`) already rebalance
monthly, so the agent is compared against baselines paying a fraction of its tax
bill.

**New default: monthly, with weekly and daily as ablations.**

### 4.2 Supervised-first

`04_models.md` rejects edge-weights-as-actions on credit-assignment grounds. The
same argument applies one level up and was not carried through: PPO gives one
scalar reward per day to attribute across 505 actions.

Cross-sectional return prediction on the same panel gives ~504 × T labelled
examples instead of T scalar rewards — three orders of magnitude more supervisory
signal from identical data. `ReturnPredictionHead` already exists as Phase 2.
**Invert it: prediction becomes the main path, allocation becomes a deterministic
function of predictions.** RL returns later as a layer on top of a signal that
already works, learning the turnover/holding-period trade-off rather than trying
to discover alpha.

> **Consequence, stated plainly.** This makes PPO, the actor-critic, FiLM and the
> GNN — roughly 2,500 lines — non-load-bearing until R6. That is the correct
> call, but it should be a decision rather than a side effect.

### 4.3 Benchmark set

`05_training.md` predicts `Random < 60/40 < EqualWeight < NIFTY50 ≲ Momentum
≲ RLAgent`. On a 504-name midcap-tilted NSE universe, equal-weight beating Nifty
50 is likelier, and `ARCHITECTURE.md` bears that out at 16.4% CAGR.

**`MomentumTopK` with `K=5` is broken — verified empirically:**

```
k=5 : cash= 47.1%  equity= 52.9%  max=10.0%
k=10: cash=  3.5%  equity= 96.5%  max= 9.6%
k=20: cash=  1.8%  equity= 98.2%  max= 4.9%
```

Five names at equal weight is 20% each, which the 10% per-name cap forbids, so
the baseline sits ~47% in cash. The momentum bar has been **understated**, and
the agent still failed to clear it. **Set K=20** (`src/trader/env/baselines.py:91`).

**Report against all of:** Nifty 50 TRI, Nifty 500 TRI, equal-weight universe,
momentum top-20 monthly. The momentum baseline is the one that matters — if it
wins, the RL is contributing nothing.

### 4.4 Data start date

`08` chose 2014 on the grounds that older data is noisier. Kite depth was
**measured** on 2026-09-04: 2005-01-03 for established names, listing date for
newer ones (COALINDIA 2010-11, HAL 2018-03, NYKAA 2021-11).

Two cautions found on review: **11.3% of rows in the 2005–2013 extension are
under ₹20 against a ₹0.05 tick** (BAJFINANCE oscillates ₹0.50↔₹1.50 through
2008–09 — ±50% steps of pure rounding noise), and pre-2010 microstructure had far
wider spreads than the cost model assumes. **Print the first non-null date per
ticker and the low-price row fraction before choosing a start.**

For real depth, NSE bhavcopy archives go back further and give point-in-time
series codes (EQ / BE / T2T), which Kite cannot provide at all — and which §2.1
shows this project needs.

### 4.5 Target hardware

Every acceptance criterion referencing the 5090 is unverifiable — that machine is
gone. Rewrite against the **RTX 4060 8 GB**.

Measured/derived budget for planning:

| Machine | FP32 | 2M steps @ 504 names |
|---|---|---|
| M4 Mac mini (MPS) | ~4 TFLOPS | ~4.8 days |
| **RTX 4060 8 GB** | **~15 TFLOPS** | **~1.3 days** (before R3 fixes) |

VRAM is **not** binding — see §5 R3. System RAM is: the host-side rollout buffer
is 6.9 GB at 504 names, so **≥16 GB system RAM is required**.

---

## 5. Revised roadmap

Milestones renumbered R0–R7 to avoid collision with M0–M10. Each has a hard gate.
**A failed gate stops the sequence — the rule the original roadmap had and the
project did not follow.**

### R0 — Fact-finding (read-only)

Answered inside A0. Seven questions:

1. Which source populated `data/panels/*.parquet` — yfinance or Kite?
2. First and last non-null date per ticker; median history length.
3. Is the panel price-return or total-return?
4. Is the universe point-in-time? *(Pre-answered: no — §2.1. Verify the read
   path anyway.)*
5. Which reward function does `reward.py` actually implement?
6. Actual ticker count and provenance of the 5-sector holdout.
7. `git log` for `costs.py` — when were STT/DP corrected, and which stored runs
   predate the fix?

**Gate:** a written answer to all seven. No code until it exists.

### R1 — One panel, one truth

- Pick a source, delete the other paths. Recommendation: **NSE bhavcopy for
  history and point-in-time series codes, Kite for recent and live.**
- **Rebuild the universe point-in-time** from
  `niftyindices.com/Monthly_Report/IndexInclExcl.xls`. This is the §2.1 fix and
  the highest-priority item in R1.
- **Widen the purge gap to exceed the longest feature lookback.** Currently 22
  trading days against 60-day features — 38 days of overlap (verified). A
  3-month purge is the minimum consistent with the feature set.
- Decide total-return vs price-return. If TR, build the dividend series from NSE
  corporate actions and benchmark against Nifty TRI.
- Key the panel on Kite `instrument_token`, not `tradingsymbol`, so renames
  (LTIM→LTM, TATAMOTORS→TMPV) cannot orphan history.
- Add daily cross-sectional rank/z normalisation of every feature.
- Add delivery percentage (NSE MTO file) and NSE published impact cost.
- Decide the low-price floor for the 2005–2013 extension (§4.4).

`beta_nifty_60d` is **already fixed** (commit `a5d8b79`); verify rather than redo.

**Gate:** panel SHA256 deterministic; every feature has nonzero variance across
tradeable rows; `is_tradeable` reconciles against bhavcopy series codes; purge
gap ≥ longest lookback.

### R2 — Honest baselines

Re-run all five baselines on the rebuilt panel at **daily / weekly / monthly**,
with corrected costs and **STCG/LTCG wired into the accounting** (the tax model
exists at `src/trader/env/tax.py` but is not connected to the env or reward).
Add the circuit-limit fill rule — an order does not fill if `open == upper_band`
(buy) or `lower_band` (sell). Cap fills at 5% of ADV. Fix `MomentumTopK` K→20.

**Gate:** a table of 5 baselines × 3 frequencies × 4 benchmarks, net of costs and
tax. **This table is the bar. Nothing else in the project means anything without
it.**

*Expect a deflating and useful result: monthly equal-weight or monthly
momentum-20 may already clear Nifty 50 TRI net of everything.*

### R3 — Kill the compute bug (may run parallel with R2)

> **Corrected diagnosis.** The original draft stated that a 4096-transition
> rollout buffer at `[60,504,15]` fp32 is 7.4 GB and "cannot be resident" on an
> 8 GB card. **It is already not on the GPU.** `ppo.py:98` documents the buffer
> as CPU-resident, `:228` does `.cpu().numpy()`, and `:334` moves each minibatch
> with `.to(self.device)` inside the inner loop. There is no VRAM overflow.
>
> The real cost is **data movement**: 4 epochs × 32 minibatches = 128 transfers
> of 217 MB each = **~27.7 GB host→device per update**. On MPS unified memory
> that is cheap; over PCIe on the 4060 it will not be.
>
> The prescribed fix is unchanged and still correct — but for this reason, not
> the stated one. Anyone chasing a VRAM problem will find nothing wrong.

Diagnosis order — twenty minutes:

1. `nvidia-smi dmon` during a run. Utilisation under 40% ⇒ data movement.
2. `torch.profiler` with `record_shapes=True` over three updates. Thousands of
   sub-millisecond `aten::conv1d` calls ⇒ a Python loop over the stock axis.
3. `torch.cuda.max_memory_allocated()` ⇒ confirm what is actually resident.

Then fix, in order:

- **Panel resident on GPU; observations become date indices.** The panel is
  ~150 MB fp32 / ~75 MB fp16 at 504 names (measured). Load once, slice the 60-day
  window on-device inside the forward pass. The rollout buffer becomes 4096
  integers, and the 27.7 GB/update of transfers disappears.
- **`torch.autocast` + pad input channels 15 → 16.** A 15-channel conv cannot use
  tensor cores. Typically 2–3× for ten minutes of work.
- **Cache encoder outputs.** The TCN embedding for `(date, stock)` is a pure
  function of the panel — independent of portfolio, cash and action. Pretrain the
  encoder supervised, freeze, cache to `[T, N, 128]` (~670 MB fp16 at 504 names),
  and let PPO train only attention and heads.

**Gate (corrected):** 2M steps in under 2 hours on the 4060, **conditional on
encoder caching landing**. The arithmetic — ~40 PFLOP total against ~15 TFLOPS —
puts a 100%-utilisation floor at ~44 minutes, so 2 hours is ~37% utilisation.
That is reachable *with* the cache and not without it. If the cache is rejected,
restate the gate rather than failing the sequence against an unreachable target.

### R4 — Supervised cross-sectional model

Same TCN encoder, same panel. Predict forward 5-day and 20-day
cross-sectionally-standardised returns. Evaluate by **rank IC and ICIR per
period**, not portfolio return. Runs in minutes, not days.

**Gate:** out-of-sample rank IC > 0.02 sustained across walk-forward windows,
with a bootstrap CI excluding zero. If this fails, there is no signal in these 15
features and no policy architecture will find one. **That is a real and valuable
answer — stop and change features, not models.**

### R5 — Deterministic allocator

Predictions → weights, no learning: top-K by predicted return, inverse-vol
scaled, sector-capped, turnover-limited, 10% per-name cap. Runs through the
existing cost/tax model and paper broker.

**Gate:** beats the best R2 baseline, net of costs and tax, on test splits, with
a paired bootstrap CI excluding zero.

### R6 — Reinstate RL, narrowly scoped

Only after R5 clears. PPO learns the allocator's free parameters — when to trade,
how much turnover to accept, holding-period trade-offs — on top of a frozen,
working signal. Reward becomes **excess log return vs Nifty 50** (the
`use_excess_returns` flag exists and has never been switched on) plus a drawdown
penalty, or restore the Differential Sharpe the spec originally called for.

**Gate:** beats R5's deterministic allocator. If it doesn't, ship R5 — a working
supervised strategy beats a novel one that loses money.

### R7 — Regime conditioning, if still warranted

Re-measure `corr(val, test)` with CI on the rebuilt panel across **≥8 windows**.
If the CI still excludes zero, run the Phase 1 A/B and the three ablations
exactly as `HANDOFF.md` §6 specifies — that protocol is good. If it includes
zero, close Phase 1 as *"problem not confirmed."*

---

## 6. Claude Code audit protocol

Run A0 through A2 before any code is written. The failure mode to guard against
is generating more code on a foundation nobody has verified.

**Reports go to `audit/` at the repo root**, not `docs/audit/`.

### A0 — Read-only forensics → `audit/A0_findings.md`

> Audit only. **Do not modify any file** except the report.
>
> For each of the 14 contradictions in §2, determine what the code actually
> does, citing `file:line`. Three verdicts: matches the spec docs, matches
> `ARCHITECTURE.md`, or **matches neither** — flag those loudly.
>
> Rows 7–9 are already fixed in code as of commit `9019892`; confirm rather than
> rediscover, and record which specs remain stale.
>
> Answer the seven R0 questions with evidence. Load `data/panels/train.parquet`
> (and val/test) and report real numbers: shape, date range, per-ticker
> first/last non-null date, median history length, tickers with <2 years,
> per-feature mean/std/min/max/distinct-count over `is_tradeable==True` rows
> only, and the fraction of cells where `is_tradeable` is True.
>
> Determine whether prices are dividend-adjusted: pick three known high-dividend
> names (COALINDIA, ONGC, ITC), find ex-dividend dates from any in-repo source,
> and check for the drop.
>
> Confirm or refute that `beta_nifty_60d` was constant 1.0. *(Pre-verified: it
> was, across all 276,005 tradeable training rows, and the stored stats recorded
> mean 0.0 for it. Confirm independently.)*
>
> Trace every key in `configs/**/*.yaml` to a read site in `src/`. List every key
> declared but never read. `min_trade_value` in `configs/env/panel_daily.yaml` is
> one known case; find the rest.
>
> `git log` for `costs.py` and `features.py`. Give the commit and date where
> STT/DP were corrected and where beta was fixed. List which stored MLflow runs
> predate those commits. *(Pre-verified: `trading_bot` holds 1 run and
> `data/walks/` is empty — confirm.)*

### A1 — Leakage and correctness → `audit/A1_leakage.md`

> Read-only.
>
> 1. Trace every path that can read `next_day_returns`. `HANDOFF.md` §7 asserts
>    only `PPOTrainer` consumes it. Verify by call graph, and confirm the obs
>    dict is not passed wholesale into any forward method.
> 2. Verify normalisation statistics are train-split-only, per walk-forward
>    window, for **both** `FeatureNormalizer` and `RegimeNormalizer`.
> 3. **Purge gap.** *(Pre-verified as insufficient: `train_end 2009-12-31 →
>    val_start 2010-02-01` = 32 calendar / 22 trading days, against 60-day
>    `realized_vol_60d` and `beta_nifty_60d` — 38 trading days of overlap.)*
>    Confirm, quantify across all windows, and state the minimum purge that
>    eliminates it.
> 4. Confirm `regime_features.py` row `t` uses returns through `t−1` only.
> 5. Check whether the shuffled-ticker leak check has ever been run. If not, run
>    it and report.
> 6. `graph.py:334` — the per-batch mask uses `mask[0]` tiled across the batch.
>    Determine whether this is a correctness bug or a deliberate simplification,
>    and what it does when a batch spans dates with different tradeable sets.

### A2 — Compute forensics → `audit/A2_compute.md`

> May add scripts under `scripts/profiling/`; change no production code.
>
> Profile one PPO update on the 4060 with `torch.profiler`
> (`record_shapes=True`, `profile_memory=True`). Report: wall-clock split across
> rollout / forward / backward / optimiser; the ten most expensive kernels; total
> H2D and D2H bytes per update; peak allocated memory; and whether any op runs
> more than 100 times per update with a leading dim of 1 (the signature of a
> per-stock Python loop).
>
> **Measure H2D explicitly.** The expected figure is ~27.7 GB per update (128
> minibatch transfers × 217 MB at 504 names). Confirm or refute.
>
> Count how many times the TCN encoder forward is invoked per gradient step, and
> how many *distinct* `(date, stock)` pairs those invocations cover. Report the
> ratio — it is the size of the prize for encoder caching.
>
> Compare measured against the analytic estimate: ~3.3 MFLOP per stock-sequence,
> 64,512 sequences per gradient step, ~640 GFLOP fwd+bwd per step, ~40 PFLOP for
> 2M steps. State the achieved fraction of the 4060's ~15 TFLOPS FP32.

### A3 — Reconcile the specs

> Now you may write. Update `02_data_pipeline.md`, `03_environment.md`,
> `05_training.md` and `08_open_decisions.md` to describe the system as it is.
>
> - `03_environment.md` cost model → STT 0.1% both legs, brokerage 0, exchange
>   0.00307% NSE / 0.00375% BSE, DP ₹15.34, GST on (brokerage + SEBI + exchange).
>   **Cite `zerodha.com/charges`; do not regenerate `costs.py` from the doc.**
> - Reward section → document what `reward.py` implements; mark Differential
>   Sharpe **planned but not implemented** if that is what A0 found.
> - Replace every RTX 5090 criterion with an RTX 4060 8 GB equivalent.
> - `MomentumTopK` K=5 → K=20, noting the 10% cap interaction and the measured
>   47.1% cash figure.
> - Document that the universe is not point-in-time (§2.1) until R1 fixes it.
>
> Add a `## Status` block at the top of each spec: `IMPLEMENTED` / `PARTIAL` /
> `PLANNED`, with the commit that last verified it.

### A4 — Quarantine (optional)

> Move to `experimental/` anything built past the M5 gate that has never been
> trained: the GNN path, Phase 1 FiLM, Phase 2 aux head. Keep tests running
> against them. Add a one-line README in each explaining what gate must pass
> before it returns to `src/`.
>
> **Marked optional on review.** This is a large refactor touching imports and 44
> tests and yields no new information. Do it if the trunk's honesty matters more
> than the churn; skip it without blocking the sequence.

### A5 — Standing rules

Written to `CLAUDE.md` at the repo root. See that file.

---

## 7. What good looks like

| Week | Deliverable |
|---|---|
| 1 | A0–A2 reports. R0 gate answered. You know what data you have. |
| 2 | R1: one rebuilt **point-in-time** panel, deterministic hash, features with variance, purge gap ≥ lookback. |
| 3 | R2: the baseline table — 5 × 3 × 4, net of cost and tax. **This ends the ambiguity.** |
| 3 | R3 in parallel: 2M steps under 2 hours, or a written explanation of why not. |
| 4–5 | R4: rank IC with CI. Go/no-go on whether signal exists at all. |
| 6 | R5 if R4 cleared: deterministic allocator vs the R2 table. |

Two outcomes are both fine. Either R4 shows signal and there is a real strategy by
week 6, or it doesn't and six months of training policies over uninformative
features has been avoided. **The current setup cannot distinguish those two
worlds. That is the actual problem.**

---

## 8. Kronos and news — where they fit

Neither belongs before R4.

**Kronos** is one cheap experiment alongside R4, not a workstream. Load
`Kronos-small` frozen, one forward pass per `(date, stock)` over a 150-name
subset, cache the hidden state, and measure whether a linear probe on those
embeddings beats a linear probe on the 15 features by rank IC. Two days. If it
loses there it will not win inside a policy. It is univariate-per-series with no
cross-sectional view, so at best it improves per-stock representation — it cannot
solve selection among 504 names, and being public it cannot contain information
the OHLCV lacks. *(UNVERIFIED: this model has not been evaluated in-repo.)*

**News** is the least of the problems and the most expensive to do correctly.
Point-in-time Indian news back to 2014 with trustworthy timestamps is a vendor
contract or a multi-month build with a high chance of undetectable lookahead. The
cheap correct version is **NSE corporate announcements and earnings calendars** —
free, structured, timestamped, one to two weeks — giving event-window masking and
better corporate-action handling as a side effect. `08` concluded "defer until the
price-only baseline is strong." That was right and still is.

---

## 9. Verification log

What was checked against the live repository on 2026-09-04, before this plan was
committed.

| Claim | Method | Result |
|---|---|---|
| Purge gap < feature lookback | `compute_windows()` on real dates | **CONFIRMED** — 22 trading days vs 60-day lookback, 38 days overlap |
| `MomentumTopK` K=5 broken | `masked_softmax` with the 10% cap | **CONFIRMED** — 47.1% cash at K=5; 1.8% at K=20 |
| `corr=−0.86` CI includes zero | Fisher transform recomputed | **CONFIRMED** — (−0.997, +0.583) |
| FLOP arithmetic | recomputed from TCN shapes | **CONFIRMED** — 3.3 MFLOP/seq, 640 GFLOP/step, 40 PFLOP/2M |
| `min_trade_value` dead | grep for read sites | **CONFIRMED** — declared, never read |
| Universe not point-in-time | SQL + grep | **CONFIRMED** — 0 rows, no read site |
| Rollout buffer exceeds 8 GB VRAM | read `ppo.py:98,228,334` | **REFUTED** — buffer is CPU-resident; real cost is ~27.7 GB H2D per update |
| `docs/` directory exists | `ls` | **REFUTED** — no `docs/`; all specs at root |
| Panel size on GPU | computed from shape | **CONFIRMED** — ~150 MB fp32 / ~75 MB fp16 at 504 names |
| 4060 VRAM binding | activations at mb=128 | **REFUTED** — ~3 GB at 504 names; system RAM (6.9 GB buffer) is the real constraint |

**Not verified, carried forward as stated:** the Kronos claims (§8), NSE bhavcopy
depth and series-code availability (§4.4), and the assertion that share-rounding
mismatch causes most of the 11% NAV divergence (§2, row 12) — that one is
plausible and untested.
