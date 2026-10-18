# Weekend scoping notes: intraday and Book 4 (gathered 2026-10-16)

Raw findings kept for the weekend planning session. **Nothing here is decided,
built or scheduled.** These are inputs to a plan, not the plan. External
numbers come from the research passes of 16 Oct (an Opus pass on practitioner
reality and a Fable pass on modelling literature) and carry their URLs. Repo
numbers cite files. Anything not checked is marked UNVERIFIED.

---

## A. Intraday

### A1. Verdict both research passes reached independently

No RL, graph, news or LLM-agent approach has independent, live, cost-inclusive
evidence of picking intraday winners at retail minute latency. The credible
evidence covers **execution** (RL/LOB, mostly simulator-based) and **slow
post-news drift** (daily horizon, US, concentrated in shorts and small caps).
Starting Monday means **collecting data, not trading a strategy**. A paper
book on an untested rule would skip gates H1–H4 (`future_HF_extension.md:191-198`).

### A2. Costs, checked against zerodha.com/charges on 16 Oct; `costs.py` matches

| Position | MIS round trip | CNC round trip |
|---|---|---|
| ₹10k | ₹10.63 (0.106%) | ₹37.59 (0.376%) |
| ₹25k | ₹26.57 (0.106%) | ₹70.96 (0.284%) |
| ₹1L | ₹82.68 (0.083%) | ₹237.82 (0.238%) |

- **Fees per year on a ₹1L account:** one full round trip a day costs **20.7%
  of NAV**. Three a day cost 62%, and ten a day 207%.
- **Spread on top:** about 0.04–0.08% per round trip for large caps, from
  NSE's impact-cost figures dated March 2019 (stale). That is roughly 10–20%
  of NAV a year at one round trip a day. Mid and small caps: UNVERIFIED.
- **Comparison:** a 15-name delivery book rotated in full every month costs
  at most about 5.4% a year.
- **Tax:** speculative business income, taxed at slab rates up to 30% plus
  cess. Losses set off only against speculative income, carried forward 4
  years.
- **Budget 2026:** left intraday STT unchanged at 0.025% on the sell side.
  F&O STT went up from 1 Apr 2026.

### A3. Realistic outcomes

SEBI's FY23 study of intraday cash traders:
- 71% of individual intraday traders lost money; 80% of those making more
  than 500 trades a year did.
- Of traders with ₹1 cr+ turnover a year, 76% lost. Three ₹1L round trips a
  day comes to about ₹1.5 cr a year.
- Loss-makers' trading costs added another 57% on top of their losses.

SEBI's F&O studies: 93% lost over FY22–24, and 91% in FY25.

Forum evidence:
- Reddit was blocked to the fetcher, so there is **no Reddit evidence**.
- The TradingQnA posts found are self-reported and carry no verified P&L.
  One example: 7–9% a month in a backtest became about 0.88% a month after
  costs and size.
- Zerodha itself notes that the P&Ls shared publicly are mostly the
  profitable ones.

### A4. Kite Connect: what paper-only intraday can use

- ₹500/month per app, including WebSocket and historical candles.
- WebSocket: up to 3,000 instruments per connection, 3 connections per key.
  `full` mode carries 5-level depth.
  - Ticks are about one-per-second snapshots, not every trade.
  - **No historical depth.**
- SEBI's retail-algo rules (static IP, algo ID, the 10 orders/second
  threshold, market protection) apply to **order endpoints only**. Data use
  is unaffected. We never place orders.
- MIS square-off is now **15:25**, or **15:12** for CAS stocks.
- Minute-history depth: Kite staff say back to about 2015 for listed names.
  Delisted tokens are retired, so that history is survivorship-biased.
  **To be measured: this is H0.**

### A5. Contradictions in `future_HF_extension.md` to correct (CLAUDE.md rule 4)

1. `:147-149` says minute history is about 60 days. Kite staff say about
   2015; 60 days is the per-request window. H0 settles it.
2. `:142` says these constraints are documented in CLAUDE.md. They are not.
3. `:150` says no depth on this subscription. Live 5-level depth is
   available; only historical depth is missing.
4. `:113` says square-off is about 3:20. It is 15:25 now, or 15:12 for CAS.
5. `:71-73` says "the current allocator grosses about 46%". That predates
   the survivorship findings (`audit/S5_TURNOVER_SELECTION.md:91`, about 12
   of about 26 CAGR points). Flag it, and don't reuse it as a benchmark.

### A6. Modelling literature (Fable pass)

**Kronos** (arXiv 2508.02739):
- NSE is inside its pretraining data.
- **We already tested it here.** It added a non-significant +0.0107 daily IC
  over the TCN (t 2.15, n=8; commits `7f9192e`, `506ef91`, `8cf71f7`). No
  top-K test was ever run on it.
- The only independent test (GitHub issue #375, US stocks): zero-shot IC
  +0.022, beaten by 5-day reversal at +0.032 (t 1.97). 91% of *Kronos's*
  gross P&L came from 5 of 85 dates.

**RL:** works for execution and market making on LOB data, mostly in
simulators. Not for directional picking (review arXiv 2106.00123; FinRL-Meta
authors concede overfitting).

**Graph models:** on Qlib's CSI300 benchmark, GATs (RankIC 0.0462) do not beat
Linear (0.0472) or XGBoost (0.0505). MASTER, HATS and RSR are author-reported
with no independent replication.

**News:**
- Equities absorb earnings news in about 10–15 minutes.
- Lopez-Lira & Tang's LLM-sentiment drift is daily-horizon. It is
  unprofitable at a 20 bp round trip and concentrated in the short leg.
- No minute-level NSE announcement study was found (UNVERIFIED).
- What is tradeable is post-news drift at daily horizon, which belongs to the
  delivery books.

**LLM trading agents:** Sharpe falls 51–62% once moved to windows after the
model's training cutoff ("Profit Mirage", arXiv 2510.07920). The leakage
sits in the weights themselves.

**JEPA** (read as the owner's "JEV"): embeddings track volatility, not
direction. A regime feature at most.

### A7. Candidate Phase-0/1 work (proposed 16 Oct, not approved)

- **Phase 0, no trading:**
  - A tick and depth recorder for about 200 liquid names, 09:00–15:35, about
    0.8 GB/day raw. The storage volume has 406 GiB free (`df`, 16 Oct).
  - The H0 minute-history probe, then a backfill on go-ahead: about 14,200
    requests, about 80 min at 3 requests/second.
  - An NSE corporate-announcement logger running from Monday. Forward-only,
    so it is leak-free.
  - The A5 corrections.
- **Phase 1, pre-registered tests, each with a kill criterion:**
  1. VWAP/TWAP execution of the delivery rebalances. Kill if it saves less
     than 10 bp per rebalance.
  2. End-of-day reversal (last-half-hour reversal, Baltussen–Da–Soebhag).
     Kill on the window-level gate, or if long top-K is less than 2× a
     10.6 bp round trip.
  3. Kronos-small zero-shot on minute bars. Kill if it does not beat a
     one-feature reversal baseline.
  4. Post-announcement drift, daily horizon, for the delivery books. Kill
     after 50 events with no separation.
- **Phase 2, only if something passes H1–H4:** an unlevered ₹1L intraday
  paper book with spread-crossing fills and a speculative-income tax module.
- **Out of scope:** RL or LLM directional agents, intraday GNNs,
  news-reaction trading, leverage, F&O.

---

## B. Book 4: a composite of simple signals (owner's idea, 16 Oct)

### B1. Correction to carry forward

The one-line rule that beat Kronos in issue #375 is **5-day** reversal on US
stocks: long-short, weekly, t 1.97. It is not usable as a weekly delivery
strategy here. Weekly rotation costs about 14% of NAV a year at ₹10L, K=30
(`future_HF_extension.md:64`), and more at ₹1L. Reversal can only be a small
component of a monthly book.

Also: R5's `momentum_topk` baseline is **20-day** momentum
(`src/trader/env/baselines.py:149-160`), not 12-1. 12-1 is untested in this
repo apart from the live P6 book (`audit/P6_MOMENTUM_15.md`).

### B2. Candidate signals

| Signal | Rank on | Evidence | Data |
|---|---|---|---|
| 12-1 momentum | past-year return, skip a month | strongest; IIMA Indian factor | prices (Book 3) |
| Low volatility | 252-day realised vol, ascending | long global record, mixed in India; NSE index funds exist | prices |
| 52-week-high proximity | close / 252-day high | George & Hwang 2004 | prices |
| 1-month reversal | last month's return, ascending | classic, decays, costly | prices; small weight only |
| Value (earnings yield) | EPS / price | long record; drought 2010–2020 | `data/ext/fundamentals.parquet` (9,533 company-quarters, `audit/F3_FUNDAMENTALS_VERDICT.md:174`); usability for E/P UNVERIFIED |
| Quality / earnings change | improving earnings | raised IC but not top-K here (`audit/F3_FUNDAMENTALS_VERDICT.md`) | fundamentals; weak |
| Trend overlay | Nifty vs 200-DMA, go to cash | cuts drawdown, usually costs return | needs index data, never fetched (CLAUDE.md, feature liveness) |
| Delivery % (India-specific) | share of volume delivered | weak or UNVERIFIED | `data/ext/delivery.parquet` |

### B3. Shape proposed

- Z-score each chosen signal **within sector** and average them.
- Top 15, monthly.
- Same allocator, stop, caps, band, budget and costs as Books 1–3.
- ₹1L live paper book, `audit/paper/live_composite/` (name TBD).

### B4. Discipline (the trap is the factor zoo)

- **Fix the mix before seeing results.** For example, equal weights of
  momentum, low-vol and 52-week-high. Published anomalies lose about half
  their returns after publication (McLean–Pontiff), and choosing the best of
  N tested signals mostly chooses luck.
- **One historical run** through the existing 20-random-book null control,
  PIT universe 2005–2026 (`audit/S4_NULL_CONTROL.md`, `audit/S6_TOPK_GATE.md`
  machinery). Delisting bias is unrecoverable (`audit/P4_survivorship.md:15`)
  and flatters all signals alike.
- Per-signal results are reported as information, never used to select.
- Go live with the pre-registered mix whatever the result.
- **Compute:** price-only signals, so the guess is tens of minutes on CPU
  (UNVERIFIED). Run a probe and get the owner's go-ahead first.

---

## C. Open questions for the weekend

1. Phase 0 recorder: run in the paper container or as a separate service?
   Depth sampling rate versus storage?
2. Book 4's exact mix and K; whether value is usable from the fundamentals
   data.
3. Whether to add the Nifty index reference line (₹1L in an index ETF) as the
   honest benchmark for all books.
4. Advisor survey (Fable pass, 16 Oct). See D below.

---

## D. What credible advisors say (survey, 16 Oct; SEBI-registered, academic or long-record voices only, no finfluencers)

- **Active picking mostly loses to the index after costs.** SPIVA India
  YE2024: 60% of large-cap active funds trailed the benchmark over 1y, 75%
  over 3y, **93% over 5y** and 74% over 10y
  (https://www.spglobal.com/spdji/en/spiva/article/spiva-india-year-end-2024/).
  Caveat: the benchmark changed in 2024, so the editions may not be
  comparable. Buffett's bet: an S&P index fund returned 7.1%/yr against 2.2%
  for a hedge-fund basket, 2008–16.
- **Short-horizon retail trading loses on aggregate.**
  - SEBI: 71% of intraday traders lost, and 93% of F&O traders.
  - Kamath (Zerodha): fewer than 1% of active traders beat an FD over 3 years.
  - Barber–Odean: the most active fifth of 66,465 US households earned
    11.4%/yr against the market's 17.9%.
  - Fewer than 1% of Taiwanese day traders show reliable net alpha.
  - Every voice surveyed is negative on intraday.
- **Factor investing:**
  - Asness and Swedroe back momentum and value, but warn of multi-year
    "winters". AQR's AUM fell from $226bn to $110bn in one.
  - Shenoy (Capitalmind, which sells momentum products) cut the 2020
    drawdown by moving to cash, and had a bad 2022.
  - Primary reference on momentum crashes: Daniel–Moskowitz, NBER w20439.
- **Test discipline the sources name:**
  - The Harvey–Liu–Zhu t > 3 hurdle.
  - Probability of Backtest Overfitting and the Deflated Sharpe ratio
    (Bailey and López de Prado), which discount for the number of configs
    tried. **Worth adding to the Book 4 test.**
- **Framing:** an index-fund core plus a capped satellite for picking and
  trading. Quoted caps: Swedroe about 5%, Bernstein 5–10% (UNVERIFIED), and
  Radhika Gupta's 80/20 split, so at most 20% in narrow bets. Real money
  only after basic financial security (Halan, Pattabiraman).
- **Red flags that apply here:** overtrading, overconfidence, overfitting
  from many trials, leverage and F&O, and reading a short live record
  (Gupta: "the most dangerous number is the 1-year return").
