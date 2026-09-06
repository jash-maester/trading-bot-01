# Future extension — intraday trading

Written 2026-09-06 at the user's request, as a plan only. Nothing here is
implemented and nothing here is scheduled. The question asked was: the original
scope was a "safer" low-risk daily strategy — what if we shifted to intraday,
cash in and cash out the same day, and took profit that way? How would the
architecture change?

This document answers that with the numbers this repo already has, says what
would survive and what would have to be rebuilt, and gives a staged path with
explicit kill criteria. It ends with a recommendation that is not the obvious
one.

---

## 0. Verdict up front

**Intraday is cheaper per trade and far more expensive per year.** It is not
safer, for reasons that have little to do with overnight gaps. And the single
thing that makes it a different project is not the model — it is that we have
no intraday data, and the tax treatment changes from capital gains to
speculative business income.

The honest summary is: *worth a bounded feasibility spike, not a pivot.* And
there is a middle path (§7) that captures a real part of the benefit for a
fraction of the work and none of the risk.

---

## 1. The cost inversion, which is genuinely counterintuitive

Per round trip, intraday is **much cheaper** than delivery. `costs.py` already
models both products correctly. Intraday (MIS) pays brokerage of 0.03% capped
at ₹20 per order, STT of 0.025% on the **sell leg only**, cheaper stamp duty,
and — decisively — **no DP fee at all**, because nothing ever leaves the demat
account.

Round-trip cost of one position, buy and sell, by trade size:

| Trade value | Delivery | Intraday | Delivery / Intraday |
|---|---|---|---|
| ₹2,000 | 0.989% | 0.106% | 9.3x |
| ₹5,000 | 0.529% | 0.106% | 5.0x |
| ₹20,000 | 0.299% | 0.106% | 2.8x |
| ₹50,000 | 0.253% | 0.106% | 2.4x |
| ₹200,000 | 0.230% | 0.059% | 3.9x |

Regenerate from `trader.env.costs`. The delivery column falls with size because
the flat ₹15.34 DP fee amortises; the intraday column is nearly flat until the
₹20 brokerage cap binds.

**This is the strongest argument for intraday and it is real.** The flat demat
fee — the thing that dominates 83–98% of costs in every backtest in this repo,
that broke the daily arm, and that P3 and P5 exist to work around — simply does
not exist intraday. An entire class of problem disappears.

### And then frequency undoes it

At ₹10 lakh and K=30, a ₹33,333 position:

| Strategy | Round trips/yr | Annual cost | % of NAV |
|---|---|---|---|
| Monthly delivery, full rotation | 12 | ₹32,220 | 3.2% |
| Weekly delivery, full rotation | 52 | ₹139,621 | 14.0% |
| **Intraday, flat every night** | **250** | **₹265,703** | **26.6%** |

Being flat every night means a full round trip every single session. Cheaper per
trip, twenty times as many trips.

> **The break-even: an intraday strategy must gross roughly 23 percentage
> points more per year than the monthly delivery strategy before it earns a
> single rupee more.** The current allocator grosses about 46% at monthly.
> Intraday must gross about 69% to match it.

That is the number that governs everything below, and it should be the first
thing any intraday proposal is measured against.

---

## 2. The tax change, which is a genuine landmine

This is the part most easily missed, and it cuts against intraday.

Indian tax treats a same-day buy-and-sell as **speculative business income**,
not capital gains — regardless of how the order was tagged. `costs.py:19-24`
already notes this distinction. Consequences:

| | Current (delivery) | Intraday |
|---|---|---|
| Head of income | Capital gains, §111A | Speculative business income |
| Rate | 20% flat (STCG) | **Slab rate**, up to 30% + 4% cess |
| Loss set-off | Against capital gains | Only against *speculative* income |
| Loss carry-forward | 8 years | **4 years** |
| Bookkeeping | Trade log | Business books; audit likely at volume |

So the tax rate is **worse**, the loss relief is **narrower**, and the
compliance burden is **higher**. `src/trader/env/tax.py` models §111A/§112A
with FIFO lots and would not merely need new rates — the whole capital-gains
framing, the twelve-month holding line, and the ₹1.25L LTCG exemption become
irrelevant. It would need a parallel speculative-income module.

The one benefit: business income permits deducting expenses (data feeds,
hardware, connectivity), which capital gains does not.

---

## 3. "Safer" deserves scrutiny

The intuition is that being flat overnight removes risk. Being flat overnight
removes exactly one risk — the overnight gap — and introduces several:

- **Forced square-off.** Positions must be closed by ~3:20pm or the broker
  closes them at whatever price is available. You become a *known* forced
  seller at a *known* time, every day. That is a structurally bad position.
- **Leverage.** MIS offers roughly 5x. It will be available and it will be
  tempting, and it multiplies both directions. The current strategy is
  unlevered by construction.
- **Cost drag as a certainty.** The 26.6%/yr above is not a risk, it is a
  guaranteed loss that the strategy must out-earn before anything else.
- **Slippage becomes the dominant unknown.** The current model,
  `atr_frac · √(Δshares/ADV)` in `panel_env.py`, is calibrated for daily bars.
  Intraday, the bid-ask spread crossed on every trip dominates, and **we have
  no spread data at all**.

Against that, the daily strategy's overnight gap risk is real but is already
diversified across 30 names and is the compensated risk you are being paid for.

**Net: intraday is a higher-variance, higher-cost, more operationally fragile
strategy.** It is not the safe version of what we have.

---

## 4. The real blocker: we have no intraday data

Everything in this repo is daily bars. `data/kite_ohlcv/` holds 656 tickers of
daily OHLCV from 2005. An intraday strategy needs minute bars at minimum.

Rough scale: 504 tickers × ~375 minutes/session × 250 sessions/year ≈ **47
million bars per year**, against roughly 126,000 daily bars per year. Two years
of minute data is ~800x the current panel.

Constraints already documented in `CLAUDE.md`:

- Kite historical data is capped at **2000 calendar days per request** and
  rate-limited to **3 requests/second**; minute data has a much shorter
  per-request window than daily, so the fetch is thousands of requests.
- Kite's minute history typically reaches back only ~60 days for most
  instruments. **Multi-year intraday backtesting is likely impossible from this
  data source**, which alone may end the discussion.
- Tick data and order-book depth are not available on this subscription at all.

Without a spread or depth model, an intraday backtest is not merely imprecise
— it is measuring a different game than the one it would trade.

---

## 5. What survives, and what would be rebuilt

Considerable infrastructure transfers. The alpha does not.

| Component | Fate | Why |
|---|---|---|
| `costs.py` | **Survives intact** | Already models MIS correctly; pass `intraday=True` |
| `walk_forward.py`, purge logic, the gate | **Survives** | Methodology is cadence-agnostic (`12_gate_decision.md`) |
| `TCNEncoder`, `SignalModel` | **Survives structurally** | Retrain on minute bars; architecture is fine |
| `allocator/deterministic.py` | **Mostly survives** | Top-K, caps, turnover budget all still apply |
| Feature pipeline | **Rebuilt** | 15 daily features → intraday microstructure features |
| `panel_env.py` | **Rebuilt** | T+1 settlement, daily steps, overnight marks all wrong |
| `tax.py` | **Replaced** | Capital gains → speculative business income (§2) |
| The R4 signal itself | **Discarded** | 5/20-day horizon says nothing about minutes |
| P3 no-trade band | **Obsolete** | Exists for the DP fee, which intraday does not pay |
| P5 capital sizing | **Replaced** | Binding constraint becomes margin, not the flat fee |

The last three rows matter: **the work currently in flight is largely specific
to the delivery cost structure.** That is an argument for finishing the current
line rather than abandoning it mid-way.

The signal point deserves emphasis. Our measured edge is a rank IC of 0.039 at
a 5-day horizon — a slow cross-sectional effect. Intraday alpha is a different
phenomenon (order flow, microstructure, opening auction dynamics, intraday
reversal), competed for by co-located firms with infrastructure we do not have
and cannot get. Nothing about our 0.039 transfers.

---

## 6. If we did it anyway: staged path with kill criteria

Each stage kills the project cheaply if it fails. Do not proceed past a failed
stage.

| Stage | Question | Effort | Kill criterion |
|---|---|---|---|
| **H0** | How much minute history can Kite actually give us? | 1 day | < 2 years for the universe → **stop** |
| **H1** | Is there *any* measurable intraday cross-sectional signal? | 3 days | Rank IC not clearing the window-level gate → **stop** |
| **H2** | Does it survive a spread model? | 2 days | Edge < 2x modelled spread cost → **stop** |
| **H3** | Does it clear the 23-point break-even net of cost? | 3 days | Gross uplift < 23 pp vs monthly → **stop** |
| **H4** | Does it survive slab-rate tax and forced square-off? | 2 days | Net after-tax below current strategy → **stop** |
| **H5** | Paper-trade live for one quarter | 3 months | Any divergence from backtest > 10% → **stop** |

H0 is a one-day check that very likely ends it. **Do H0 before anything else,
and before any modelling thought at all.** It costs a day and it is the
question everything else depends on.

---

## 7. The recommendation, which is a middle path

Two things are true at once: the intraday *cost structure* is genuinely
attractive, and the intraday *game* is one we are poorly positioned to play.
There is a way to take the first without the second.

**Use intraday data for execution, not for alpha.**

Keep the daily cross-sectional signal, the monthly cadence, the allocator, and
the delivery product exactly as they are. Then improve *how* the monthly
rebalance is executed:

- Instead of filling at the next open (what `panel_env.py` models today), work
  the order across the session — VWAP or TWAP over a window.
- This attacks slippage, which is currently modelled as
  `atr_frac · √(Δshares/ADV)` and never measured against reality.
- It needs only ~60 days of minute data for the names being traded, which is
  comfortably inside what Kite provides.
- It changes no tax treatment, adds no leverage, keeps the DP fee structure we
  have already engineered around, and cannot lose more than the slippage it is
  trying to save.

Expected value is modest — perhaps 10 to 30 basis points per rebalance, so
maybe 1 to 4 percentage points a year at monthly cadence — but the downside is
bounded near zero and the work is a week rather than a quarter.

A second, smaller idea worth noting: **the intraday cost schedule makes small
accounts viable in a way delivery does not.** At ₹1 lakh, the flat DP fee makes
a 30-name delivery book expensive (P5 exists for this). Intraday pays no DP fee
at all, so the capital constraint that P5 is currently deriving would largely
vanish. If the ₹1 lakh account turns out to be badly constrained by P5's
findings, that is the one result that would make revisiting this document
worthwhile.

---

## 8. What this document does not settle

- **No intraday signal has been tested.** Everything above is about cost,
  tax, data availability and risk structure. Whether an exploitable intraday
  cross-sectional effect exists in NSE equities at our latency is an open
  empirical question, and H1 is where it would be answered.
- **The 23-point break-even assumes full daily rotation.** A strategy holding
  through several sessions but flat overnight only some days would sit between
  the delivery and intraday rows. That intermediate is not costed here.
- **Margin and leverage are not modelled anywhere in this repo.** The env is
  unlevered by construction. Any intraday work needs a margin model before it
  can claim a return number.
- **No number in §1 or the break-even has an MLflow run behind it.** They are
  closed-form evaluations of the committed cost model, not backtests
  (CLAUDE.md rule 3). They are reproducible from `trader.env.costs`, and that
  is the only claim made for them.
