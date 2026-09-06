# P3 / P4 / P5 — integration pass

Written 2026-09-06. Three streams built on disjoint files, three verifiers
returned **FAIL** on all three, and this pass applies every must-fix and every
handed-back patch, wires the seams none of them was allowed to touch, and
records what is still open.

No backtest was run, no grid, no training, no GPU. Every number below comes
from a committed script or a committed test, named in place.

**One line:** all three streams' mechanisms are now reachable from the config
and the CLI and are measured on the quantity that costs money; one of the three
central claims was false and its cause was a defect in the environment, not in
the allocator; and **the standing R5 table is no longer reproducible from this
tree** — the env's execution rule changed and the grid has to be re-run before
anything in `audit/R4_R5_RESULTS.md` is quoted again.

---

## 1. The one that mattered: the env invented orders

P3's central claim was that a name inside `no_trade_band` "produces no order and
no ₹15.34 demat debit". The verifier refuted it and the refutation reproduces.

`PanelTradingEnv` holds **shares** and is handed **weights**, and two lossy
conversions sat between them:

* `obs["portfolio"]` is cast to **float32** (`panel_env._build_obs`), so a
  weight that pins a name to its current position reproduces it only to ~6e-8
  relative;
* `_step_target` computed `target_shares = floor(target_value / open)`, and
  `floor(400 × (1 − 6e-8))` is **399** — a one-share SELL, worth the share
  price, clearing the 0.5-share integrality guard (it is a whole share) and
  clearing `min_trade_value` on any name above ₹500 (61.1% of the panel by last
  close), and paying the flat ₹15.34.

A second, independent class: weights are marked at the previous close and filled
at the open, so pinning a name across an overnight gap asks the env to move
`gap × position`.

**The fix.** `floor` still sets the *size* of a trade that happens — that is
what stops a buy overspending its target value — but it no longer decides
*whether* one happens:

```python
target_shares = np.where(
    np.abs(requested - self._shares) < 0.5, self._shares, target_shares
)
```

A request that **rounds to the position already held is not an order.** It only
ever removes orders, never adds or enlarges one, and it cannot block a
liquidation (a full exit requests 0.0 against a position of at least one share).
The identical rule is now in `PaperBroker._materialise_targets`, because the two
paths disagreeing about `min_trade_value` is what produced the 11% NAV
divergence CLAUDE.md lists under "Dead config".

Regression: `tests/unit/test_weight_to_share_orders.py`, 5 tests. Verified
failing with the mechanism reverted — 2 of the 5 fail, the three controls stay
green:

```
FAILED test_a_pinned_book_emits_no_order_when_nothing_moved
FAILED test_a_pinned_book_emits_no_order_across_a_sub_half_share_gap
2 failed, 3 passed
```

### What it costs, measured

`scripts/probe_share_rounding.py`, one cell (monthly, K=30, 20d), 500 steps of
`data/panels_kite/oos.parquet` + `data/signal/r4_v1`, ₹10 lakh, band 0.0. Run on
each side of the change:

| metric | floor decides | snap decides | delta |
|---|---|---|---|
| final NAV | 983,453.50 | 987,037.46 | **+3,583.96 (+0.36%)** |
| executed legs | 2,160 | 2,144 | −16 |
| scrip-sell-days | 1,546 | 1,531 | −15 |
| DP fees ₹ | 23,715.64 | 23,485.54 | −230.10 |
| Σ daily turnover | 7.4158 | 7.2366 | −0.1791 |

**So `audit/R4_R5_RESULTS.md` is superseded.** Its numbers were measured under
the old rule. They are not badly wrong — this is a 0.36% NAV effect over ~2
years on one cell — but they are not reproducible from this tree, and P4's
pre-registered threshold (§3 of `audit/P4_survivorship.md`) is a *ratio* against
`G_unrestricted`, so both sides of it must come from the same code.

One integration test's recorded bounds moved with it, documented in place:
`test_monthly_cadence_cuts_a_persistent_baselines_turnover_far_less_than_21x`,
where daily equal-weight turnover falls 3.276 → 2.930 while monthly barely moves
1.608 → 1.600. Same mechanism as the `min_trade_value` fix before it: a trade
too small to carry a flat fee is overwhelmingly a daily-cadence phenomenon,
because one day of drift is ≈ √21 times smaller than twenty-one days of it.

### What the fix does NOT do

A gap worth **more than half a share** is a genuine order on a name the band
reports as suppressed, and no change inside `panel_env.py` can prevent it: the
env is never told which names the band pinned. Whether it bites is a property of
the book geometry, not of the band —

| capital, K | position | 1% gap | median share | verdict |
|---|---|---|---|---|
| ₹10 lakh, K=30 | ₹33k | ₹330 | ₹704 | 0.47 share — absorbed |
| ₹1 crore, K=30 | ₹3.3 lakh | ₹3,300 | ₹704 | 4.7 shares — executes |

so `BandSuppression.n_suppressed` is an **upper bound on orders removed, and a
loose one at large capital**. Pinned as a characterisation test
(`test_the_band_does_not_survive_a_gap_worth_more_than_half_a_share`) rather
than left as prose. The real repair is an interface change — the band would have
to reach the env as a set of names not to touch — and that is listed as open
below.

---

## 2. P3 — the band

**Docstrings corrected** (`src/trader/allocator/deterministic.py`):

* the module docstring's step 7 and `_apply_no_trade_band` no longer claim a
  pinned name produces no order. They state what is true — the delta is exactly
  zero *in weight space* — and carry the measurement above;
* `_clearing_prefix` claimed to keep "the *most* legs the budget can move" and
  to "never over-suppress". Both are false: the monotonicity argument proves it
  finds the largest feasible **prefix of the descending sort**, not the largest
  feasible **subset**. Counterexample, tie-free, regenerated by
  `test_clearing_prefix_is_a_prefix_not_a_maximum_subset`:

  ```
  dev = [0.0333, 0.0333, 0.0333, 0.010, 0.010, 0.010], budget 0.05, band 0.005
  shipped rule keeps 3 legs at scale 0.500501
  largest feasible subset  5 legs at scale 0.517598, min move 0.005176
  exhaustive search finds a larger feasible set in 59 of 3000 draws (2.0%)
  ```

  The behaviour is kept and the claim weakened: preferring the prefix spends the
  budget on the names furthest from target and errs toward **fewer** trades,
  which is what a per-scrip flat fee wants. The postconditions the caller relies
  on — every executed leg clears the band, gross ≤ budget — are unchanged and
  are what the tests assert. (CLAUDE.md rule 4: this was a spec and its code
  disagreeing inside one docstring.)

**The headline was quoted against the wrong baseline.** `n_sold` counts
weight-space sells, and roughly two thirds of the band-0 sell tail is sub-₹500
dust that `min_trade_value` was never going to execute, so the fee is never
billed on it. `_band_sweep` now carries a `billed` column at two capitals,
filtering on `|Δw|·NAV ≥ DEFAULT_MIN_TRADE_VALUE` imported from `costs.py`:

```
 AR(1) rho=0.9, N=504, K=30, budget=0.30, 3 seeds x 60 periods
   band   traded    sold   gross  bill10L  bill1Cr   sold%  gross%  bill10L%  bill1Cr%  fee %NAV/yr
  0.000   183.46  155.46  0.3000    50.78    83.29   100.0   100.0     100.0     100.0         0.93
  0.002    49.14   24.66  0.2772    24.66    24.66    15.9    92.4      48.6      29.6         0.45
  0.005    23.42   10.16  0.2087    10.16    10.16     6.5    69.6      20.0      12.2         0.19
  0.010    11.22    4.83  0.1700     4.83     4.83     3.1    56.7       9.5       5.8         0.09
```

At ₹10 lakh a 0.005 band leaves **20.0%** of the billed sells, not 6.5%, and the
band-0 flat-fee drag is **0.93% of NAV/yr**, not 2.86% — the ratios above 0.002
are unchanged, only the baseline was wrong, and the baseline is what every ratio
divides by. Corrected in all three places it appeared: the sweep test, the
comment block in `configs/env/allocator.yaml`, and here.

**Seams wired.** The band was unreachable outside Python:

* `scripts/run_allocator.py` takes `+allocator.no_trade_band=` and
  `+allocator.band_grid=[...]` (the grid defaults to the single value, so the
  two knobs cannot disagree about what ran), logs `no_trade_band` as an MLflow
  param, and logs the band's suppression counterfactual per rebalance when the
  band is on;
* `AllocatorEnvConfig.no_trade_band` exists, `ActionRanges.decode` threads it
  into `AllocatorParams`, and `scripts/train_allocator_rl.py` reads the config
  key — so `no_trade_band:` in `configs/env/allocator.yaml` is now **live** and
  is uncommented. Making it a fourth *action* dimension was deliberately not
  done: it would change `action_dim` from `3 + n_sectors` to `4 + n_sectors` and
  invalidate every checkpoint and the `ranges:` block. That is a decision of its
  own.

---

## 3. P4 — survivorship

Every must-fix applied in `audit/P4_survivorship.md`; the substantive ones:

* **A fabricated command output was in the durable artefact.** A `select
  count(*)` was shown returning a three-column `rows | min | max` header. The
  fact was true, the transcript was invented. Replaced with the real output,
  with the correction recorded in place rather than silently swapped.
* **The section 1 claim was over-broad and false.** `allocate()` does *not*
  return 0.0 for any masked-out name: the mask gates candidacy, not holdings, so
  a held name that goes untradeable drains over several rebalances (0.14 of NAV
  each after one, from five names at 0.2) and with a band above its weight is
  pinned indefinitely. Narrowed to what is true and what P4 needs — *a name that
  has never been tradeable can never acquire weight* — with the counterexample
  now a test.
* **Breadth was quoted on a universe the allocator never sees.** 469.5 → 359.7
  is panel-wide over 645 tickers; the env trades `active_tickers()` = 504, where
  it is **364.9 → 276.7** and `k=30` goes **8.22% → 10.84%**, not 6.4% → 8.3%.
  Equal-weight holds ~365 names, not ~470. Both rows now print from
  `survivorship_arm.py --dry-run`, labelled.
* **Two omitted confounds added**: sector retention is 49–86% per sector, not
  uniform, so `max_sector_weight` binds on a different mix; and 227 of 504
  active tickers become all-zero columns with `sector_id == 0` in the restricted
  split — CLAUDE.md's phantom-sector condition, harmless for the allocator and
  equal-weight, fatal for a graph model. Both printed on every run.
* **The residual's sign rested on a wrong argument.** "Equal-weight holds 470
  names and the allocator holds 30" is a statement about variance, and variance
  produces no bias in a difference of means. What sets the sign is the rate at
  which the truncated names would have been *picked*. Measured
  (`--tail-exposure`): the top-30 by `r_hat_20d` land in the bottom decile of
  realised forward-20d return **11.16%** of the time against 10%, and the bottom
  5% **6.26%** against 5% — 1.1–1.3x, not 12x. Sign supported, **magnitude still
  UNVERIFIED** and unrecoverable from a panel with zero deaths.
* **A decision rule is now pre-registered** before the arm runs: PASS iff
  `G_restricted ≥ 0.5 · G_unrestricted` and `G_restricted > 0`, with `G` the
  allocator-minus-equal-weight CAGR gap *within* an arm. Without a number, §3's
  limit 3 (a lower allocator CAGR is "expected rather than informative") left no
  result that could fail.
* `run_allocator.py` now logs `universe_effective` alongside `universe_size`, so
  a restricted run reads `504 / 277` and is distinguishable from `504 / 504`.

---

## 4. P5 — capital and K

* **`supportable_k_detail(...).binding` was capital-dependent**, contradicting
  the module's own headline. It compared the *floored integers* `k_fee` vs
  `k_mtv`; near the crossover those tie at one capital and separate at another.
  At f = 0.00625697: ₹1 lakh gives 40 = 40 ("both"), ₹10 lakh gives 407 vs 400
  ("min_trade_value"). Now derived from the real-valued bounds, so the invariant
  holds by construction. `k_max` was never affected — it is `min` of the two
  floors either way — so **no capacity number moved**.
* **Its test was a seed lottery.** The committed body hard-coded
  `random.Random(7)`; run over seeds 0–399 that exact body fails on **139**.
  Replaced with a 200-seed sweep that also asserts the label equals what `f` vs
  `f*` predicts, plus the counterexample budget pinned explicitly. Both verified
  failing against the old labelling before the fix was accepted.
* **"The bill is set by K and cadence" is wrong for the module's own formula.**
  Outside the saturation cap `R` cancels: `drag = D·k·turnover/(2·a·C)`, and
  `fee_drag_estimate(1e5, 30, R, 3.73)` returns 0.858273% for R = 12, 52 and 252
  alike. The daily arm's 19.4x is entirely its turnover (72.30 vs 3.73), not its
  cadence. Corrected in the docstring and in the test's; a new test pins R's one
  real degree of freedom (where the cap bites) so the parameter is not held in
  place only by a mutation that happens to cross it.
* **Blocked item 3 was withdrawn**: it asked for `initial_cash` to be pinned in
  the env config because it "disagrees with the broker", and all three configs
  already pin `1_000_000` (`configs/env/panel_daily.yaml:3`,
  `configs/env/allocator.yaml:17`, `configs/broker/paper.yaml:3`), and
  `audit/R4_R5_RESULTS.md:42` already states "₹10 lakh initial". The *substance*
  survives and is real: every R5 number was measured at 10x the intended live
  account, which multiplies flat-fee drag by exactly 10 (0.0858% → 0.8583%/yr at
  K=30 monthly). Re-measuring means running the grid.
* **Seams wired.** `trader.allocator` exports the sizing API.
  `run_allocator.py` **warns** — never refuses — when a configured K exceeds
  `max_supportable_k` at the configured capital, and stamps `max_supportable_k`
  and `k_is_supported` into every MLflow run. A refusal would make it impossible
  to measure what the constraint costs, which is a legitimate thing to want.
  P5's blocked item proposed `_fail`; the WARN is the deliberate deviation.

`AllocatorParams` stays the single source of truth for allocator knobs.
`sizing.py` holds none: it imports `_DP_CHARGE` and `DEFAULT_MIN_TRADE_VALUE`
from `trader.env.costs` and takes everything else as an argument.

---

## 5. Turnover is no longer the only thing the grid reports

The dominant cost of this system is flat and per-scrip, so the grid was
reporting the wrong column. `PanelTradingEnv` now exposes
`info["n_scrips_sold"]` and `info["n_legs"]` — the count the ₹15.34 is actually
billed on, from the executed order list rather than from a weight-space estimate
— and every arm of `run_allocator.py` logs, per run:

`scrip_sell_days`, `scrip_sell_days_per_rebalance`, `dp_charges_paid`,
`legs`, `fee_drag_estimate` (P5's closed form against that run's own measured
turnover, a floor because it models sells as full exits),
`dp_charges_minus_equal_weight`, and — when the band is on —
`names_suppressed_per_rebalance`. The printed table gains `band`, `sold/reb` and
`DP ₹` columns.

---

## 5b. The wiring, exercised end to end — NOT a result

`run_allocator.py`'s rewritten `main()` cannot be unit-tested (it is a Hydra
entry point that talks to MLflow), so it was run once against a **stubbed**
MLflow module on `PYTHONPATH` — no server, no run written, nothing logged
anywhere. `k=600` was included deliberately to fire the P5 warning.

```
uv run python scripts/run_allocator.py data=kite_v1 +split=oos \
  +signal_tag=r4_v1 +require_gate_pass=false \
  '+allocator.k_grid=[30,600]' '+allocator.freq_grid=[monthly]' \
  '+allocator.horizon_grid=[20d]' '+allocator.band_grid=[0.0,0.005]' \
  +allocator.null_control=true
```

```
universe 504 from active_tickers(); 504 of them have rows in .../oos.parquet
WARNING  k=600 exceeds max_supportable_k=400 at initial_cash=1,000,000 ...
         Measuring it anyway — see P5, src/trader/allocator/sizing.py.

R4 signal gate [r4_v1]: FAIL — ...
*** SIGNAL GATE DID NOT PASS: these numbers are NOT evidence ***
strategy                  K   band     freq  hor  Sharpe    CAGR     MDD    Turn sold/reb      DP ₹  vs EW Δ CAGR
equal_weight              -      -  monthly    -   1.427   0.289  -0.515   0.841      2.5    72,297            —
null_signal (control)   600  0.000  monthly    -   1.394   0.224  -0.476   3.094    118.3   165,074      -0.0647
allocator                30  0.000  monthly  20d   2.014   0.368  -0.470   3.608     97.3   135,836      +0.0786
allocator                30  0.005  monthly  20d   1.893   0.399  -0.512   3.299     23.2    32,444      +0.1103
allocator               600  0.000  monthly  20d   1.394   0.224  -0.476   3.094    118.3   165,074      -0.0647
allocator               600  0.005  monthly  20d   1.698   0.126  -0.236   0.891      5.2     7,194      -0.1635
```

**Read this as a wiring check and nothing else.** The r4_v1 gate is FAIL, the
split is `oos` and not the standing `oos_r4_v2`, and there is no MLflow run ID,
so under CLAUDE.md rules 1 and 2 not one number in it is quotable. What it does
establish: the band grid sweeps, the P5 warning fires and stamps
`k_is_supported=False`, `universe_effective` is logged, and every arm reports
the scrip-sell-day count and the rupee DP bill. The k=30 rows are also a
sanity check on P3's synthetic sweep — 97.3 → 23.2 billed sells per rebalance at
a 0.005 band is 23.9%, against 20.0% predicted at ₹10 lakh
(`test_band_cuts_name_count_faster_than_turnover`). The k=600 row shows what an
unsupportable K does: `turnover` collapses to 0.891 because the targets are
unreachable, which is precisely what the warning is warning about.

---

## 6. Still open — each needs a decision, none is a caveat on the above

1. **The standing R5 table must be regenerated.** Not a defect: a consequence of
   §1. Until it is, no figure in `audit/R4_R5_RESULTS.md` is quotable, and P4's
   threshold has no baseline.
2. **The band cannot suppress an order it cannot see.** Fixing the residual in
   §1 means changing what the allocator hands the env — a set of names not to
   touch, or an order list instead of a weight vector. That changes
   `step_weights`' contract, `PaperBroker`'s target interface and
   `AllocatorEnv`, so it is a design decision, not a patch. Until it is taken,
   a band setting must be justified on `scrip_sell_days` from a real run, never
   on `n_suppressed`.
3. **Should the band be an action dimension?** `AllocatorEnvConfig` holds it as
   a fixed parameter. Making it learnable is `action_dim = 4 + n_sectors` and
   invalidates every checkpoint and the `ranges:` block.
4. **₹10 lakh or ₹1 lakh?** The env and broker configs agree on ₹10 lakh; the
   intended live account is ₹1 lakh, and P5 shows that is a 10x multiplier on
   the only cost that matters. Changing the default silently voids the grid;
   keeping it means every result is about an account nobody has. Needs a call,
   then a re-measurement.
5. **`data/panels_kite/oos_r4_v2.parquet` and `data/signal/r4_v2/` are not on
   this machine.** `oos_pre2015.parquet` restricts the **r4_v1** slice. Do not
   compare it against the r4_v2 standing table — that confounds the restriction
   with the warm-up-prefix coverage change worth ~10 CAGR points. Rebuild the
   arm wherever the r4_v2 artefacts live.
6. **Point-in-time universe.** Still the only real repair for delisting bias,
   still needs an external data source, still out of scope.
