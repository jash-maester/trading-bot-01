# CLAUDE.md

Standing rules for this repository. These exist because each one has already
been violated at least once, at cost. See `09_revamp_and_audit.md` for the
incidents behind them.

---

## The five rules

**1. Never implement a milestone whose predecessor's acceptance criteria have not
demonstrably passed.** If asked to, say so and stop.

> M5's gate — *"matches or beats `EqualWeightRebalanced` on the val split over 3
> seeds"* — never passed. M6, M7, Phase 1 and Phase 2 were built on top of it
> anyway: ~2,500 lines of model code and 44 tests standing on a gate that never
> opened. `07_roadmap.md` opens by forbidding exactly this.

**2. Never call a config, model, or result a "winner", "baseline-beating", or
"validated" without an MLflow run ID.** Aspirational comments in configs are
forbidden.

> `configs/config.yaml` described `mlp_regime` as the "Phase 1 winner" when no
> walk-forward had ever been run on it. `HANDOFF.md` §1 had to carry a warning
> that *"any comment in a config calling something a 'winner' is aspirational."*
> That warning should have been a rule, not a footnote.

**3. Any number in a document must link to a run or a script that regenerates
it.** "UNVERIFIED" is an acceptable answer; a confident guess is not.

**4. When a spec and the code disagree, report it — do not silently follow
either.** There are 14 known contradictions catalogued in
`09_revamp_and_audit.md` §2. Assume more exist.

**5. Cost-model constants are load-bearing and India-specific. Never regenerate
`src/trader/env/costs.py` from a document.** Edit in place, citing
`zerodha.com/charges`.

> The specs still describe STT as sell-side-only, brokerage as `min(20,
> 0.0003·V)`, and DP as ₹15.93. All three are wrong. Regenerating `costs.py`
> from `03_environment.md` reintroduces a 22% understatement of every delivery
> round trip.

---

## Verification standard

Every factual claim cites `file:line`, pasted command output, or a run ID.

Numbers in reports must be reproducible: paste the command, or commit the script
that generates them.

A gate passes only with a durable artefact — a committed report, a deterministic
hash, a metrics table, or an MLflow run ID. An assessment that something "looks
right" is not a gate.

**A failed gate is a legitimate and often good outcome.** Record it plainly.
Never soften a failure, never partially pass, never "pass with caveats."

---

## Known-dangerous ground

Things that have already caused silent, expensive errors here.

| Area | The trap |
|---|---|
| **Feature liveness** | `beta_nifty_60d` was constant 1.0 across all 276,005 tradeable training rows because `^NSEI` was never fetched — and the stored stats recorded mean 0.0 for it, so a dead channel acted as a fixed bias into every convolution. **Check every feature has nonzero variance before trusting a run.** |
| **Corporate actions** | `auto_adjust=True` handles splits and dividends but **not demergers** — value moves to a separate listed entity, so the parent shows a fake catastrophic loss (NIITLTD −76.13%, MASTEK −66.00%). |
| **Symbol renames** | A current instrument dump has no memory of renames. `LTIM→LTM`, `TATAMOTORS→TMPV`, `STLTECH→STLTECH-BE`, `MCDHOLDING` delisted. A symbol-keyed join silently drops history. |
| **Survivorship** | `market.universe_snapshots` is empty and has no read site. The universe is **not** point-in-time despite `00_overview.md` calling this a non-negotiable. |
| **Purge gaps** | The walk-forward purge is 22 trading days against 60-day rolling features — 38 days of overlap between train and val feature windows. |
| **Dead config** | `min_trade_value: 500` is declared in `configs/env/panel_daily.yaml` and never read. Measured 11% NAV divergence between backtest and paper broker. Assume other keys are dead until traced. |
| **Phantom sectors** | `all_tickers()` once unioned `NIFTY_50` with `SECTOR_MAP`, giving unsectored names `sector_id == 0` — an id absent from `SECTOR_IDS`, forming a phantom sector node in the graph model. Fixed; regression test exists. |

---

## Environment

- **Docs live at the repo root**, not in `docs/`. Numbered specs `00_*.md` …
  `09_*.md`, plus `HANDOFF.md`, `ARCHITECTURE.md`, `PROGRESS.md`. Audit reports
  go to `audit/`.
- **Nothing auto-loads `.env`.** Run `set -a && . ./.env && set +a` before
  anything touching Postgres or Kite.
- **Services run natively on the Mac mini**, not in Docker: Postgres 16 via
  Homebrew, MLflow from the project venv. `make services-up` / `services-status`.
  The `docker compose` path is intact for machines that have Docker.
- **MLflow is on port 5555**, not 5000 — AirPlay owns 5000 on macOS.
- **Kite access tokens expire around 6 AM daily** and require a browser login to
  refresh. Market data (historical, quote, ohlc, ltp) needs the paid Connect
  subscription; the free Personal tier returns `PermissionException`.
- **Never call an order-placing Kite endpoint.** Read-only: `instruments`,
  `historical_data`. Execution goes through the local paper broker and the
  Postgres ledger.

---

## Compute

Target hardware is an **RTX 4060 8 GB**; development happens on an **M4 Mac mini
(24 GB unified, MPS)**. The RTX 5090 referenced throughout `00`–`08` no longer
exists — treat every acceptance criterion mentioning it as void.

Measured on the M4: env stepping is **0.2%** of wall clock, rollout forward
**3.7%**, update forward+backward **96.2%**. Cost scales linearly in ticker
count, and larger minibatches are *slower* per sample and OOM at 512 — the
observation, not the parameter count, sets the price. Optimisation advice that
assumes a compute-bound model is wrong here.
