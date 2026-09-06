# The daily arm: diagnosis and fix plan

Investigated 2026-09-06 after `audit/R4_R5_RESULTS.md` recorded equal-weight at
daily cadence returning -0.37%/yr against +27.2%/yr monthly, on turnover that
could not explain the gap.

## It is not the data

The panel is fine. Feature liveness passes on all 15 columns, the purge gaps are
correct, and the same panel produces a sane monthly result. The cause is a
**flat fee meeting trades too small to carry it**, and it is a defect this
repository has already documented and then walked past.

## Measured cause

Zerodha's demat debit charge is **₹15.34 per distinct scrip per selling day**,
flat — it does not scale with trade value. `costs.py` models it correctly
(`costs.py:55`, applied per-scrip via a 0/1 indicator at `panel_env.py:394`).

The environment decides whether a trade happens on a **share-count** threshold:

    panel_env.py:392   traded = np.abs(delta_shares) >= 0.5

The paper broker decides on a **rupee-value** threshold:

    paper_broker.py:1038   if abs(delta) * open_px < self._cfg.min_trade_value:  # drop

`min_trade_value: 500` is declared in `configs/env/panel_daily.yaml:25` and is
read by the broker and **never by the environment**. This is precisely the dead
key CLAUDE.md lists under Known-dangerous ground, with the note "Measured 11% NAV
divergence between backtest and paper broker." That divergence is this.

Equal-weight over the R4 OOS slice, 1860 steps, ₹10 lakh initial:

| Cadence | Sell trades | Median sell | Below ₹500 | Flat fee total | Fee share of all cost |
|---|---|---|---|---|---|
| daily | 83,933 | ₹102 | 94.5% | ₹1,287,532 | 97.7% |
| monthly | 10,250 | ₹499 | 50.1% | ₹157,235 | 89.8% |

At daily cadence the median sell trade is ₹102 and pays ₹15.34 to settle — a
**15% charge on the median trade**, and on 99.4% of trades the fee exceeds 1% of
the trade value. Total flat fees come to 129% of initial capital over 7.4 years,
about 17%/yr, which is what erases the daily arm.

Turnover was a misleading diagnostic precisely because this cost is **not
proportional to traded value**. 4.74x annual turnover at ~0.3% suggests 1.4%/yr;
the real bill was 17%/yr, because what drives it is the *number of distinct
names sold per day*, which no control in the system constrains.

## What the fix reveals underneath

> **CORRECTED 2026-09-06.** The exploratory measurement first recorded here —
> daily equal-weight jumping to +62.7%/yr once the ₹500 rule was applied — was
> an artefact of a SECOND bug that the rule activated, not a real result, and
> the survivorship reading built on it was wrong. `panel_env.py` assigned
> `self._shares = target_shares` unconditionally while the cash leg was masked
> by `traded`, so every suppressed order moved the position **for free**. It was
> dormant while the only guard was `>= 0.5` shares. With both fixed, daily
> equal-weight is 1.337 Sharpe / +26.6% CAGR — unremarkable, and in line with
> monthly. Survivorship (P4) remains a real concern; the +62.7% was simply never
> evidence for it. Final numbers: `audit/R4_R5_RESULTS.md`.

Applying the broker's ₹500 rule to the environment showed the monthly arm
improving and the daily arm changing sign, and **every number in the first
`audit/R4_R5_RESULTS.md` is void**, monthly included: half its sell trades were
sub-₹500 too.

## Plan

**P1 — Wire `min_trade_value` into the environment.** Add it as a constructor
parameter read from config, and gate execution on `abs(delta) * fill_price >=
min_trade_value` so the environment and broker agree by construction. The
existing share-count guard stays as a separate integrality check. Acceptance: a
parity test driving the same target weights through both paths and asserting the
same set of executed trades, which is the test whose absence let an 11%
divergence persist. Note dropping a trade leaves the position where it was, so
weights drift from target — that is correct and must be asserted, not corrected.

**P2 — Regenerate every R5 number.** Re-run the allocator grid and rewrite
`audit/R4_R5_RESULTS.md`, marking the current table superseded. Do not compare
any new figure against the old one; they are different cost models.

**P3 — Give the allocator a per-name no-trade band.** `turnover_budget`
constrains rupees and is the wrong instrument for a per-name fee. Add a
deviation band so a name is only traded when its weight has drifted beyond it,
which directly targets the count that sets the bill. This subsumes P1 for the
allocator path but does not replace it: the environment must still be correct for
the baselines.

**P4 — Treat survivorship as the live confound.** It now dominates the daily
arm's absolute numbers. `market.universe_snapshots` is still empty with no read
site. Until it is backfilled, no absolute CAGR here should be quoted, only
comparisons between arms sharing the bias, and daily-cadence results should not
be quoted at all.

**P5 — Scale the name count to capital.** The intended live capital is ₹1 lakh,
while these runs use ₹10 lakh. At ₹1 lakh an equal-weight 504-name book holds
₹198 per position, so a ₹15.34 debit fee is 7.7% per sale: uninvestable. At
K=30 it is ₹3,333 a position and 0.46% per sale, which is workable. The maximum
supportable name count is a function of capital and should be derived and
asserted, not chosen by hand.

## Sequencing

P1 then P2 are the unblock and must land together, since P2 is the evidence P1
worked. P3 is the profit lever. P4 and P5 are correctness bounds on what any of
these results are allowed to claim, and P4 in particular gates whether the daily
arm is ever reportable.
