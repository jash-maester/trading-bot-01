# 13 — Kronos, fundamentals and news: what to integrate and in what order

Written 2026-09-07 at the user's request. Plan only; nothing here is
implemented. Two of the premises behind the request turn out to be wrong, and
saying so is most of the value of this document.

---

> **STATUS 2026-09-08. Both data arms are closed, both negative.**
>
> **Fundamentals (§3, §3b)** — real, stable, independent rank IC that reaches
> nothing a long-only top-20 book buys. `audit/F2_FUNDAMENTAL_IC.md` for what
> the features carry, `audit/F3_FUNDAMENTALS_VERDICT.md` for why it does not
> translate. §3b.6 records the evaluation rule that came out of it.
>
> **News (§4)** — not built, and the reason is stronger than "cannot be
> validated". The ceiling was measurable without buying any news:
> `audit/N1_STOP_VETO_HEADROOM.md`. Cash from a stop is idle ~10 days, not 21
> (measured), and at 10 days stopping is ALREADY the right call in all three
> arms. A veto with perfect foresight is worth +1.7 to +2.3%/yr, nothing
> available separates the recoverers from the fallers (R4's own score: Spearman
> +0.014), and 49–52% recover — a coin flip. §4's design and cost estimate were
> sound; the upside is bounded and modest. §7 of N1 says what would reopen it.
>
> A follow-up lead from the same events — 60-day mean reversion against a
> 21-step quarantine — tested well in sample (+1.6pp CAGR) and **failed to
> replicate on the holdout**. The default is unchanged.
>
> **Kronos (§2)** was measured separately: beats the TCN by +0.0107, not
> significant, but far more stable.

## 0. Verdict

| Proposal | Reality | Verdict |
|---|---|---|
| Wire **Kronos** for fundamentals | Kronos is OHLCV-only. It has **no fundamentals and no news**. | Premise wrong — but Kronos is worth doing for a *different* reason (§2) |
| Use **Monid.ai** equities API | Its equities catalogue says *"No endpoints here yet."* Monid is an API **router**, not a data source. | Not usable today (§5) |
| Add **news**, twice daily | Feasible forward, but **no history means no backtest and no gate** | Do it, but as a *veto*, not a ranking signal (§4) |
| Add **fundamentals** | Genuinely available for NSE and genuinely backtestable | The best of the three, with one severe trap (§3) |
| Concatenate a per-stock **score** into the model | Right in outline; wrong in three details | Signal model yes, RL agent no; split level from change (§3b) |

**Recommended order: Kronos encoder → fundamentals → news veto.** That is
deliberately the reverse of the order the request implies, and §1 is why.

---

## 1. The binding constraint is sample size, not features

Before adding any data source, the thing that killed the last attempt has to be
stated, because it applies to everything below.

R6 failed out of sample for a measured, structural reason
(`ARCHITECTURE.md` §3.3): its action space was 17 dimensions and the training
span holds roughly **four non-overlapping two-year windows**. It fitted
parameters against a sample that could not support them.

The supervised signal has the same exposure. It trains on 8 windows across
2016–2024, and its whole edge is a rank IC of 0.039. **Every feature group added
is more parameters against the same fixed, small sample.** A new data source is
not free even when the data itself is free.

That gives a test any proposal here has to pass:

> Does this add **information** the price series does not already carry, or does
> it add **parameters**? Sources that mostly restate what price already knows
> make the model worse, not better.

Fundamentals mostly pass that test — a balance sheet is not in the price series.
Kronos passes it in an unusual way: it adds no features at all, it adds
*pretraining*, which moves in the opposite direction and makes the sample
problem smaller. News passes it too, but cannot be validated (§4).

---

## 2. Kronos — not fundamentals, but the most promising item here

**What it actually is.** The first open-source foundation model for financial
candlesticks, decoder-only transformer, trained on data from 45+ global
exchanges. It takes OHLC (volume and amount optional) and forecasts future
OHLCV. A tokenizer quantises continuous K-line data into hierarchical discrete
tokens, then an autoregressive transformer models the sequence.

| Model | Params | Context | Licence |
|---|---|---|---|
| Kronos-mini | 4.1M | 2048 | MIT |
| Kronos-small | 24.7M | 512 | MIT |
| Kronos-base | 102.3M | 512 | MIT |
| Kronos-large | 499.2M | 512 | closed |

**It contains no fundamentals and no news.** It is exactly the same input class
we already use. So it cannot answer the question that prompted the request.

**Why it is still the first thing I would do.** Our `TCNEncoder` is 363k
parameters trained *from scratch* on 8 years of one market. Kronos-base is 102M
parameters pretrained on 45 exchanges. Swapping a from-scratch encoder for a
pretrained one is the standard remedy for exactly the small-sample problem in
§1, and it is the only proposal here that reduces our parameter burden instead
of increasing it.

**How it would fit.** Not as a forecaster. We would use its **hidden states as
per-stock embeddings**, replacing `TCNEncoder` inside `SignalModel` and keeping
everything above it — the cross-sectional layer, the two return heads, the
walk-forward, the gate. Kronos is per-series and has no cross-sectional view, so
our cross-stock attention stays exactly where it is.

Three arms worth measuring, cheapest first:

1. **Frozen Kronos embeddings + our heads.** No fine-tuning. This is the
   cleanest test of whether its pretraining transfers to NSE at all, and it is
   compatible with the embedding cache already built (`embedding_cache.py`),
   which is worth ~47x on a frozen encoder.
2. **Fine-tuned Kronos-small.** More capacity to overfit; run only if (1) shows
   signal.
3. **Kronos forecast as a feature.** Feed its predicted forward return as one
   extra column alongside the 15. Cheapest of all and tests the model's own
   output rather than its representation.

**Gate:** the existing R4 window-level gate, unchanged, plus a paired comparison
against the current TCN on the same windows and seeds. If frozen Kronos
embeddings do not beat a from-scratch TCN on rank IC, stop — the transfer did
not happen and nothing downstream will rescue it.

**Cost:** one 102M-parameter forward pass per (stock, date) to build the cache.
On the 4060 that is a few hours once, then free. It is a bigger model than
anything we run today and the VRAM cliff in `ARCHITECTURE.md` §5 applies, so
profile the batch size before committing to a long run.

**Risk to name up front:** foundation models for financial time series are a new
and contested claim. "Trained on 45 exchanges" does not mean it learned anything
transferable to NSE mid-caps at a 5–20 day horizon. Arm (1) is designed to find
that out in one run rather than after a month of integration.

---

## 3. Fundamentals — backtestable, and one trap that would invalidate everything

This is the item that actually answers the original question, and the one where
the data genuinely exists.

**Availability.** Several providers cover NSE/BSE fundamentals: `indianapi.in`,
FinEdge, Twelve Data, Finnhub, and a RapidAPI Indian Stock Exchange endpoint.
Coverage claims run to 5,200+ NSE and BSE listed companies with P&L, balance
sheet, cash flow and shareholding.

**Volume is tractable.** 504 companies × ~32 quarters over 8 years ≈ **16,000
records**. That is a one-time fetch, not a streaming cost, and it is small
enough to store in the existing Postgres schema.

**Candidate features**, all cross-sectional and all slow-moving: earnings yield,
book-to-price, return on equity, debt-to-equity, sales growth, accruals,
promoter-holding change, earnings surprise versus the prior quarter.

> ### The trap: point-in-time, and it is the same shape as our survivorship bug
>
> Fundamentals are **announced with a lag** and **restated afterwards**. A Q3
> figure is not knowable on the last day of Q3; it becomes knowable on its
> announcement date, typically 4–8 weeks later. Nearly every cheap API returns
> *current, restated* values with no announcement date attached.
>
> Joining those to a panel by quarter-end gives the model a number no investor
> could have had — and, worse, a number that was later *corrected to be right*.
> That is not a small lookahead. It is the single most reliable way to
> manufacture a spectacular backtest, and this repo has already been burned by
> its cousin: a universe of 645 names with zero delistings.
>
> **Hard requirement: no fundamental datum enters the panel without an
> announcement date, and it becomes visible only on the day after.** If a
> provider cannot supply announcement dates, use a conservative fixed lag (45
> days after quarter end) and say so in the artefact. A provider that offers
> neither is not usable at any price.

### F0 result, run 2026-09-07: dates exist, but not where the money is

Audited before fetching anything, because the announcement date decides whether
this branch is usable at all.

* **Commercial fundamentals APIs do not document announcement dates.**
  `indianapi.in`'s page describes income statements, balance sheets and cash
  flows, and says nothing about a result-declaration date anywhere in its
  endpoint documentation. The same is true of the other candidates' public
  material. Absence of documentation is not proof of absence, but a field this
  load-bearing being undocumented is itself a bad sign.
* **NSE publishes the dates itself.** Its corporate-filings section carries
  financial results and board-meeting announcements — SEBI's Listing
  Obligations regulations *require* a listed company to tell the exchange when
  its board will consider results — with the broadcast date attached. That is
  the authoritative record of when a number became public, and it is exactly
  the field the commercial APIs omit.
* **We already have the machinery to read it.** R8 built
  `src/trader/data/sources/nse_flows.py` with the browser-like session warm-up
  NSE requires. Confirmed the hard way during this audit: a plain unauthenticated
  fetch of the NSE financial-results page **times out**, which is the same
  behaviour that warm-up exists to defeat.

**Verdict: F0 PASSES, with a split design.** Take the *figures* from a
commercial API (cheap, structured, historical) and the *dates* from NSE (free,
authoritative, already scrapable), then join on (symbol, period) and expose each
figure only from the day after its broadcast date. If the join fails for a
company-quarter, that row is dropped rather than lagged by a guess — a fixed
45-day fallback is the fallback for a *provider* with no dates at all, not for a
row we simply failed to match.

That split is more work than one API call and it is the difference between a
usable feature and a lookahead generator.

**Gate:** run R4's existing IC gate on the extended feature group, exactly as
`features_ext.py` was built to do. The comparison is against the same
walk-forward without the group. Fundamentals are a months-to-years effect and
our horizon is 5–20 days, so a small or zero IC lift is the *expected* outcome
and would not be a failure of the plumbing.

**Cost:** one-time historical fetch, then quarterly refresh. Likely free tier or
low tens of dollars.

---

## 3b. How a fundamental score should enter the model

The obvious design — compute one score per stock and concatenate it to the
model's input — is right in outline and wrong in three details, each of which
decides whether it helps or quietly hurts.

### 3b.1 Where it attaches: the signal model, never the RL agent

Into `SignalModel`: **yes**. That is the natural home. The score is
cross-sectional, per stock and per date, it slots in beside the existing 15
feature columns, and R4's rank-IC gate measures it directly with no new
machinery.

Into `AllocatorEnv`'s observation: **no, and this one is structural.** That
observation is deliberately 42 dimensions of *portfolio state* with no per-stock
tensor (`ARCHITECTURE.md` §1). That was the fix for the retired critic, which
mean-pooled 504 stock embeddings and therefore could not see the state it was
valuing (`10_architecture_revamp.md` §1.2). Concatenating a per-stock score
there means adding 504 dimensions and reintroducing precisely the defect that
was removed. Fundamentals belong in the thing that **ranks stocks**, not in the
thing that **sizes the book**.

### 3b.2 One composite score is probably the wrong shape

Collapsing earnings yield, ROE, debt-to-equity and growth into a single number
makes a modelling decision by hand before the model sees anything: it fixes the
weights, discards the components, and makes it impossible to know afterwards
which part did the work. The model is capable of learning that combination.

The counter-argument is real and it is §1: six raw features is six more things
to fit against roughly four independent periods. So measure **both**:

* the raw components as separate columns; and
* one **unfitted** composite — an equal-weight average of cross-sectional
  z-scores. Unfitted is load-bearing. A composite whose weights were optimised
  on the same data is an overfitted model with fewer visible parameters, which
  is worse than the honest version because the overfitting is hidden.

### 3b.3 The timescale mismatch, and the memorisation risk it creates

We predict 5–20 day returns. Fundamentals change quarterly and their documented
predictive power is over one to five years. Within any 20-day window a
fundamental *level* is essentially constant.

A near-constant per-stock feature behaves like a **stock fixed effect**. That is
the specific danger: instead of learning "profitable companies outperform", the
model can learn "these particular 40 tickers did well between 2016 and 2024".
That memorises names rather than a relationship, it will not transfer, and — the
reason to take it seriously — it would look excellent in sample.

**Split level from change.** The *level* is the slow factor carrying the
mismatch problem. The *change* — earnings surprise, margin inflection, estimate
revision — is an event, is higher frequency, and is far more plausible at a
20-day horizon. Feed both; expect the change to carry whatever signal exists.

### 3b.4 Cheaper uses that add no parameters at all

Given §1, two uses are worth testing **before** the ranking feature, because
neither fits a single new parameter:

* **a quality floor on the candidate set** — weak balance sheets never enter the
  top K. This is closer to how fundamentals are actually used, and it is
  immune to the fixed-effect problem because it never enters the ranking;
* **the stop-loss veto** (§4) — a strong company falling is noise, a weak one
  falling may not be. This is the IndiGo case stated precisely.

### 3b.5 Two checks before believing any of it

1. **Standalone IC first.** Measure the score's own rank IC before adding it to
   anything, so its contribution is known independently rather than inferred
   from a lift.
2. **Stability across windows.** A genuine factor shows up broadly across the 8
   walk-forward windows. A memorised set of names shows up in the windows
   containing those names and nowhere else. Concentration is the tell, and
   `summary.json` already carries per-window ICs to check it against.

**Prediction, recorded in advance so it can be wrong:** the change component
gives a small lift, the level component gives none and mildly overfits, and the
quality filter is worth more than either as a ranking feature.

### 3b.6 A THIRD check, which these two were not enough without

Run 2026-09-08. Both checks above passed convincingly and the conclusion they
supported was still wrong.

The change score cleared them: 6/6 windows positive at both horizons, t up to
3.5, near-independent of R4 (cross-sectional corr +0.022), pooled 20d IC up
from +0.0439 to +0.0517, and an unselected control keeping 80% of the lift. It
then earned **nothing** — every one of eight allocator arms flat or worse after
tax, and top-20 forward return moving −0.00039 (t −0.22).

The reason is structural, not a fluke of this dataset. **Rank IC scores
agreement over all 504 names; a long-only book of K=20 sees only the extreme
top of the ranking and is blind to how the other 484 are ordered.** A feature
can be genuinely informative about the middle of the cross-section, lift IC,
and never touch what gets bought. The decile spread does not rescue this
either: it improved +13%, but it is top-50 minus bottom-50 and half of that
improvement lives in a tail a long-only book cannot trade.

So, as a standing rule for every candidate on this page including the news arm:

3. **Top-K forward return, before any allocator run.** The plain mean forward
   return of the K names the signal selects, no costs and no rebalancing rules
   in between. `scripts/topk_diagnostic.py`. If that does not move, the
   allocator will not either, and the IC number is not evidence about this
   strategy.

Full write-up: `audit/F3_FUNDAMENTALS_VERDICT.md`. Prediction scoring:
change lifts — right; levels give none — right; quality filter worth more —
wrong, it would be built from the levels, which carry nothing.

## 4. News — real information, but it cannot be validated

News is the highest-frequency of the three and the most likely to matter at our
horizon. It also has a problem that no amount of money quite solves.

**The blocker: we have no history.** Everything in this repo is validated by
walk-forward over 2016–2024. News gathered from today forward cannot be
backtested on that span, so it cannot pass the R4 gate, and per `CLAUDE.md` rule
1 nothing may be built on a gate that never opened. Buying history is possible
in principle but the archives that carry Indian small-cap headlines with
timestamps are neither cheap nor complete.

Two honest options: **start collecting now and have a usable panel in 12–24
months**, or **use news in a role that does not require a backtest**. The second
is the one worth doing.

### The design: a veto on held names, not a ranking signal over 504

This is where the cost question and the IndiGo question meet.

Fetching news for all 504 names twice daily is 1,008 calls a day. Fetching it
only for the ~20 names held plus the ~20 candidates for the next rebalance is
40 calls a day:

| Strategy | Calls/day | $/day | $/yr |
|---|---|---|---|
| All 504, twice daily | 1,008 | 1.31 | 478 |
| All 504, once daily | 504 | 0.66 | 239 |
| **Held + next candidates (~40), daily** | **40** | **0.05** | **19** |
| Market-level only, twice daily | 2 | 0.00 | 1 |

At Monid's quoted $0.0013 per call. Backfilling 8 years for all 504 names would
be ~1,016,000 calls ≈ $1,321 — cheap in absolute terms, but the archive
completeness, not the price, is what makes it unattractive.

**A veto is not a ranking signal and does not need a rank IC.** It answers one
question about the ~40 names we care about: *is this fall a story or is it
noise?* That is precisely the IndiGo case — a strong company falling on
sentiment, which recovers, versus a company whose fall reflects something
structural, which does not.

Concretely, it would gate the **stop-loss**, not the allocator:

- name breaches its volatility-scaled stop **and** carries materially negative
  news → sell, and extend the cooldown;
- name breaches its stop with **no** adverse news → treat as noise, hold or
  reduce rather than exit.

That is a small, testable, cheap intervention on top of a mechanism that already
works, and it does not require news history to justify — it can be validated
forward, on the ~50 stop events a year the strategy actually generates.

**What must not be done:** using a large language model to produce a per-stock
sentiment score, feeding it in as a 16th feature, and quoting the IC. That is
unbacktestable, unstable across model versions, and would fail the §1 test.

---

## 5. Monid.ai — a router, not a data source

**What it actually is:** a registry that lets an agent discover and call across
1,700+ tools from 55+ providers, billed per call from one balance (their example
rate: $0.0013), with no subscription. It routes to Exa and Octen for search,
Apify and Browserbase for scraping, and similar.

**Its equities catalogue is empty:** *"No endpoints here yet — the catalog grows
weekly."*

So Monid supplies **no equity data today**. Its value here is narrow and
conditional: it is a reasonable way to reach a *news search* endpoint without
signing a separate contract, at the 40-calls-a-day budget above. It is not a
route to fundamentals, and it should not be a dependency — anything built on it
must work against a direct provider too.

**Recommendation:** do not integrate Monid now. Re-check when the equities
catalogue is non-empty. If a news veto goes ahead, evaluate Monid against a
direct search API on price *and* on whether headlines carry usable timestamps.

---

## 6. Staged plan, with kill criteria

Each stage ends the line cheaply if it fails.

| # | Stage | Question | Effort | Kill criterion |
|---|---|---|---|---|
| K0 | Kronos smoke | Does it load and embed our panel at all? | 1 day | Cannot run at our shape on 8 GiB → stop |
| K1 | Frozen Kronos embeddings vs TCN | Does pretraining transfer to NSE? | 2 days | Rank IC not above the from-scratch TCN → **stop the whole Kronos line** |
| K2 | Kronos forecast as a 16th feature | Cheaper alternative to K1 | 1 day | No IC lift → drop |
| F0 | Fundamentals provider audit | Does anyone supply **announcement dates**? | 1 day | Nobody does → use a 45-day fixed lag or **stop** |
| F1 | Historical fetch, 504 × 32 quarters | — | 2 days | Coverage below ~80% of the universe → stop |
| F2 | Standalone IC of the score | What does it carry on its own? | 0.5 day | IC indistinguishable from zero → stop before integrating |
| F3 | Level vs change, as separate arms | Which component carries it? | 1 day | Neither lifts → keep pipeline, drop features |
| F4 | Unfitted composite vs raw components | Is the sample big enough for components? | 0.5 day | — (chooses the shape, does not kill) |
| F5 | Per-window stability of the lift | Factor or memorised names? | 0.5 day | Lift concentrated in 1–2 windows → **discard, it is names not a factor** |
| F6 | Quality floor on the candidate set | Zero-parameter alternative | 1 day | No improvement → drop the filter, keep the features |
| N0 | Stop-event census | How many stop events a year is a veto worth? | 0.5 day | Under ~20/yr → not worth any integration |
| N1 | News veto, forward paper only | Does it separate story from noise? | 2 days + 6 months | No separation after 50 events → drop |

**K1 and F0 are the two that decide everything.** Both are one to two days and
both can kill their entire branch.

---

## 7. What I would do first, and what I would not

**First: K1.** It is the only item that attacks the constraint in §1 rather than
adding to it, it needs no new provider, no contract and no money, and it is two
days to a yes or no.

**Second: F0.** One day, and it determines whether fundamentals are usable at
all. If no provider carries announcement dates, that is worth knowing before any
fetching.

**Third, and only after F0 clears: F2 before F3.** Measure what the score
carries on its own before measuring what it adds. A lift is easy to misread; a
standalone IC is not. And run F5 — per-window stability — on any lift that does
appear, because §3b.3's failure mode produces a strong average and a
concentrated distribution, which is indistinguishable from a real factor if you
only look at the mean.

**Not now: Monid.** Its equities catalogue is empty; there is nothing to
integrate.

**Not at all: an LLM sentiment score as a model feature.** Unbacktestable,
version-unstable, and it fails the test in §1.

**A note on the framing that started this.** The instinct — that a strong
company falling on sentiment should be held, not sold — is right, and the
volatility-scaled stop already implements a crude version of it by giving a
name room proportional to its own normal movement. News would sharpen that from
*"how much does this normally move?"* to *"is there a reason this moved?"* That
is a genuine improvement and it is the right long-term use of news here. It is
also the one use that does not require history we do not have.
