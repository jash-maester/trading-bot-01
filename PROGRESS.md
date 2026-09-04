# PROGRESS

**Last updated:** 2026-09-04 by session 1 (bootstrap)
**Plan:** `09_revamp_and_audit.md`
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
| A0 — read-only forensics | NOT_STARTED | Written answers to all 14 contradictions + 7 R0 questions | `audit/A0_findings.md` | — |
| A1 — leakage & correctness | NOT_STARTED | Six checks answered with call graphs / measurements | `audit/A1_leakage.md` | — |
| A2 — compute forensics | NOT_STARTED | Profile + measured H2D bytes + encoder-invocation ratio | `audit/A2_compute.md` | — |
| A3 — reconcile specs | NOT_STARTED | Every spec carries a `## Status` block + verifying commit | `02/03/05/08_*.md` | — |
| A4 — quarantine (optional) | NOT_STARTED | Untrained code moved to `experimental/`, tests still green | `experimental/*/README.md` | — |
| A5 — standing rules | **DONE** | `CLAUDE.md` exists with the five rules | `CLAUDE.md` | 1 |
| R1 — one panel, one truth | NOT_STARTED | Deterministic SHA256; every feature nonzero variance; purge ≥ lookback; point-in-time universe | panel hashes | — |
| R2 — honest baselines | NOT_STARTED | 5 baselines × 3 frequencies × 4 benchmarks, net of cost **and tax** | metrics table + run IDs | — |
| R3 — kill the compute bug | NOT_STARTED | 2M steps < 2h on the 4060, **conditional on encoder caching** | timed run + run ID | — |
| R4 — supervised cross-sectional | NOT_STARTED | OOS rank IC > 0.02 across windows, bootstrap CI excluding zero | metrics table + run ID | — |
| R5 — deterministic allocator | NOT_STARTED | Beats best R2 baseline net of cost+tax, paired bootstrap CI excluding zero | metrics table + run ID | — |
| R6 — reinstate RL | NOT_STARTED | Beats R5's allocator | run ID | — |
| R7 — regime conditioning | NOT_STARTED | `corr(val,test)` CI over ≥8 windows excludes zero, then Phase 1 A/B | walk-forward summary | — |

**Current unit: A0.** Nothing before it is outstanding — A5 was completed during
bootstrap because the rules govern every later unit.

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

## Session log

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
