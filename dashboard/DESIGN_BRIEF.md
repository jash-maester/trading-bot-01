# Paper-trading monitor — design brief

**For:** Claude Design, working in parallel with engineering on this repo's dashboard.
**Status:** brief, 2026-09-30. The current UI works and shows correct data; the owner
dislikes its design. This document is everything a designer needs: who uses it, what
every screen must show, the exact data behind each element, the states each screen
can be in, and the rules the content must not break.

> Example values in this brief are **illustrative** (taken from a rehearsal on closing
> prices, not a live result). Never ship them as real numbers.

---

## 1. What this product is

A private, single-user web dashboard for an **algorithmic stock-investing experiment
on India's NSE, run entirely on paper**. No real money moves and no orders are ever
sent to a broker.

Two things run side by side, and the dashboard monitors both:

| | **Live book** (the user's "portfolio") | **Forward paper test** (the experiment) |
|---|---|---|
| What | ₹1,00,000 of simulated cash, invested by the algorithm at 09:16 IST on 2026-09-30, then managed by its rules | The same algorithm replayed end-of-day against 20 books that pick stocks at random with identical rules, plus an equal-weight book |
| Prices | Live Kite quotes, marked every 15 min in market hours | NSE's official daily prices, recorded after each session closes |
| Question it answers | "How is my portfolio doing right now?" | "Is the algorithm actually better than random stock-picking?" |
| Update cadence | Intraday (09:16 trade; marks 09:15–15:45; close mark 15:35) | Once a day (~21:00 IST), plus a 07:30 catch-up |
| Emotional register | Portfolio app: holdings, today's move, total P&L | Lab notebook: honest, sceptical, "don't over-read early results" |

**The single most important design tension:** the live book invites the user to feel
good or bad about daily P&L; the experiment exists to say that early P&L is noise and
that the algorithm has **not** yet shown it beats random selection. The design must
give the portfolio view the polish of a good brokerage app **without** making short-run
P&L look like evidence of skill.

## 2. The user

One person: the owner of the project. Technically fluent (built the system with an AI
engineer), retail investor in India, reads tables well ("I understand better from
tabular representation"). Uses it:

- **Every trading morning before 09:16 IST:** log in to Kite (the broker) so the system
  can read prices. This is a hard dependency — forget it and nothing trades.
- **Mid-morning (~10:30) and during the day:** glance at the portfolio. What did it buy,
  how much, how is each stock and sector doing today and overall.
- **Evenings / weekly:** check the experiment's standing and that the data pipeline is
  healthy.
- On a desktop browser mostly; **phone use is likely** for the quick checks.

Primary jobs, in order:

1. See at a glance: portfolio value, today's ₹ and % move, total ₹ and % P&L.
2. See what it holds: stock, sector, quantity, buy price, current price, 1-day and
   total P&L per stock, weight.
3. See the sector picture: where the money is, which sectors are up or down
   (the owner explicitly asked for a **scrollable, sector-wise graph view**).
4. Do the daily Kite login in one click, and see instantly whether it's still valid.
5. Know that everything is working: data fresh, schedules fired, no errors.
6. Follow the experiment honestly: algorithm vs random vs equal-weight, with the
   right caveats.
7. Read the research trail: what was tried, what failed, why.

## 3. Non-negotiable content rules

These come from the project's standing rules (`CLAUDE.md`) and apply to every screen.

1. **PAPER ONLY must always be visible.** Every screen carries a clear, calm (not
   alarming) marker that this is simulated money and no orders are placed.
2. **No "winner" language.** Never label the algorithm, a config or a result as
   "winning", "beating the market", "validated", "outperforming" unless the
   pre-registered test has passed. Neutral words: "ahead of / behind", "rank",
   "difference".
3. **The rank is not evidence yet.** The experiment's rank (algorithm vs 20 random
   books) must be shown with a visible caveat until 48 months of record exist. The
   success criterion is exactly: *rank 1 of 21 at 48 months; nothing else counts.*
   Don't make the rank the hero number.
4. **Every number traces to a source.** Tooltips / footnotes name the file the number
   came from (e.g. `audit/paper/live/holdings_latest.json`). Keep this affordance —
   subtle, but present.
5. **Failures are shown plainly.** A failed check is red and says what failed. No
   softening, no "passed with warnings".
6. **No trading controls.** No buy/sell buttons, no order forms, no "execute".
   The only action that talks to the broker is the daily login.

## 4. Information architecture

Current navigation (a flat radio list in a sidebar) with proposed priority:

| # | Screen | Primary question | Priority |
|---|---|---|---|
| 1 | **Portfolio** (today: "Live book") | How is my ₹1L doing? | Home screen |
| 2 | **Kite login** | Is the broker connection alive? Log in. | Must be reachable in one tap from anywhere; surfaced as a global status |
| 3 | **Experiment** (today: "Overview" + "P&L") | Is the algorithm better than random? | Secondary |
| 4 | **System health** (today: "Health & ops") | Is everything running? | Secondary; global status chip on every screen |
| 5 | **Research log** (today: "Research & reasoning") | What was tried and what failed? | Tertiary |
| 6 | **Warm-up history** (today: "Warm-up context") | What did the pre-test replay look like? | Tertiary; context only, could fold into Experiment |

Design is free to restructure (tabs, top nav, merged screens), as long as every
element in §5 remains reachable.

### Global elements (every screen)

- **Paper-only marker** (rule 1).
- **Kite connection chip:** `Connected · valid until 06:00` / `Login needed` / `Could
  not verify`. Tapping goes to Kite login. Currently a sidebar badge.
- **Pipeline health chip:** OK / WARN / STALE / FAIL, tapping goes to System health.
- **Market status:** Pre-open / Open / Closed / Holiday / Weekend (IST). *New — not
  built yet.* Derivable from time + whether today's quotes exist.
- **Last updated** time (IST) for the data on screen.
- **Reload** control (data is cached ≤ 60 s).

## 5. Screens in detail

### 5.1 Portfolio (home)

**Header metrics** (from `holdings_latest.json`):

| Metric | Field | Format | Notes |
|---|---|---|---|
| Starting capital | `capital` | ₹1,00,000 | constant |
| Current value | `nav` | ₹ | cash + holdings at LTP |
| Today's P&L | `day_chg_rs`, `day_chg_pct` | ±₹, ±% | vs yesterday's close; vs fill price for shares bought today |
| Total P&L | `pnl_rs`, `pnl_pct` | ±₹, ±% | vs ₹1,00,000, net of all charges |
| Invested (cost) | Σ holdings `cost` | ₹ + "N stocks" | includes buy charges |
| Cash | `cash` | ₹ | usually small (~0.5%) |
| Realised P&L | `realised_rs` | ±₹ | from sales (rebalances, stops) |
| Charges paid | `charges_rs` | ₹ | STT, stamp, exchange, GST, DP |
| Last marked | `ts` | HH:MM IST | staleness cue if > 20 min in market hours |

**Trading & risk strip** (from `state.json`):

- Last rebalance date (`last_rebalance`); **next rebalance** = first trading session
  of next month (*new*, derive it).
- Stops pending for next open (`pending_stops`: list of tickers) — these will be
  **sold at the next 09:16**. Needs prominence when non-empty.
- Re-entry bans (`cooldown`: ticker → sessions left).
- One-line rule explainers: *monthly rebalance on the first session of the month;
  volatility stop sells a stock that closes below its stop level, at the next open,
  and bars it for 21 sessions.*

**Holdings table** (one row per stock, `holdings[]` in the snapshot):

| Column | Field | Notes |
|---|---|---|
| Stock | `symbol` | e.g. `HDFCBANK` |
| Sector | `sector` | NSE industry name; "Unclassified (ETF / other)" exists (e.g. NIFTYBEES, an index ETF) |
| Qty | `qty` | whole shares |
| Avg buy | `avg_price` | ₹ |
| Invested | `cost` | ₹ incl. charges |
| LTP | `ltp` | ₹ |
| 1D % / 1D ₹ | `day_chg_pct`, `day_chg_rs` | |
| Value | `value` | ₹ |
| Total P&L ₹ / % | `pnl_rs`, `pnl_pct` | |
| Weight % | `value / nav` | cap is 10% per stock |
| Stop level | `stop_level` | ₹; null until the first close mark. Show distance from LTP. |
| Opened | `opened` | date first bought |

Needs: sort by any column, sticky header, mobile layout (cards or a condensed table),
red/green that is also distinguishable without colour (±sign, arrows).
Typical size: **25–35 rows** (rehearsal: 27 stocks).

**Sector views** (the owner's explicit ask: "scrollable graph view sector wise"):

- Sector summary table: sector, # stocks, weight %, value, P&L ₹, 1D ₹
  (`D.sector_summary`).
- Treemap today: sector → stock, box size = value, colour = total P&L %.
- A **scrollable horizontal bar list** grouped by sector: one bar per stock, length =
  value, colour = profit/loss, hover shows P&L and 1D %.
- Design freedom here — the goal is "which sectors did it put my money in, and how is
  each doing", readable in 5 seconds.

**Intraday chart:** portfolio value through the day (`marks.csv`: `ts, nav, pnl_rs,
day_chg_rs`, every 15 min). Multi-day once history accumulates — needs a range
selector (1D / 1W / 1M / All).

**Per-stock history:** multi-select stocks, plot total P&L over time
(`marks_holdings.jsonl`, one snapshot per mark).

**Trade log / fills** (`ledger.jsonl`, one line per simulated fill):
`ts, trade_date, kind (rebalance / volstop; absent on the first purchase), side (BUY/SELL), ticker, sector,
qty, price, value, charges, realised_rs (sells), target_weight, quote_ts`.
Group by trade day; show a per-day summary (`trades.jsonl`: buys, sells, ₹ bought/sold,
charges, stops executed).

**Honesty footer:** a short standing note that early P&L is noise and the algorithm has
not shown it beats random picks (link to Experiment).

**States:**

| State | When | Show |
|---|---|---|
| Not deployed | before the first 09:16 purchase | countdown to 09:16, Kite status, "₹1,00,000 in cash" |
| Pre-open | 00:00–09:16 on a trading day | yesterday's close values, today's actions expected (rebalance? stops?) |
| Live | 09:16–15:30 | live marks |
| After close | 15:35 onwards | final marks for the day, stops flagged for tomorrow |
| Holiday / weekend | no NSE trades today | last close; "Market closed" |
| Rehearsal data | `prices != "live"` | loud banner: not live quotes |
| Unpriced stock | `unpriced[]` non-empty | row warning; valued at buy price |
| Stale marks | last mark > 20 min old in market hours | warning with the time |
| Token missing/expired | Kite chip not green | banner linking to Kite login |

### 5.2 Kite login

Purpose: the daily broker login, in one click, before 09:16.

- Status card: **Valid / Expired / Not set / Could not verify**, minted at, expires at
  (~06:00 IST next morning).
- Primary button: **Log in to Kite** (opens `kite.zerodha.com/connect/login?...`).
  Kite redirects back to `http://127.0.0.1:8501/?...&request_token=...`; the app
  captures it automatically and shows success/failure.
- Fallback: paste the redirect URL or token (masked input).
- Explainer: why it's needed, that only login/profile endpoints are used, where the
  token is stored (owner-only file, never displayed).
- Error copy: token refused (single-use, expired within minutes → log in again);
  credentials missing.
- *Nice-to-have:* a morning reminder surface (e.g. the chip turns amber after 06:00
  until login; a countdown to 09:16).

### 5.3 Experiment (the forward paper test)

Source: `audit/paper/record.jsonl` — one line per recorded session:
`{run_ts, date, books: {<book_name>: {nav, ret}}}` where `ret` is the daily log return.
Books: the **signal book** (the algorithm, with the volatility stop), **equal-weight**,
**20 random books** (seeds), and 3 secondary overlays of the algorithm.

- Headline: cumulative P&L (₹ on the ₹1,00,000 notional, and %) for signal,
  equal-weight, random median, random 10th–90th percentile band.
- **Rank** of the signal among 21 books — shown with the "not evidence until 48 months"
  caveat (rule 3). Progress indicator toward 48 months (sessions, calendar months).
- Chart: cumulative P&L over sessions — signal line, equal-weight line, random books as
  a **fan/band** (p10–p90, median), optional thin lines for each random book, optional
  overlays toggle.
- Per-session table.
- Integrity notes that must remain visible when present: freeze-date mismatch, missing
  (session, book) pairs, unparseable record lines.
- Restart history: the record has been restarted 3 times (documented reasons). Show
  "since <date>" and a link to the restart notes.

Tone: scientific instrument. It should look *less* exciting than the portfolio.

### 5.4 System health

Source: `audit/paper/health.json` (`checked_at, overall, checks[{check, status, detail,
…}]`), `logs/paper/<date>.status` (run step lines), memory CSVs, determinism info.

- Overall pipeline status + each check as a row: `nse_access, data_fresh,
  feeds_aligned, record_current, last_run, scheduler_fired, determinism, scheduler` —
  status (OK/WARN/FAIL), detail text, checked-at.
- "Re-run health check" button (rate-limited; already exists).
- Latest nightly run: step timeline (store → panel → predict → allocator → record) with
  timestamps and OK/FAIL per step (from the status file lines like
  `2026-09-29T18:04:20Z allocator OK (85 books)`).
- Schedule: what runs when (IST): 07:30 catch-up, 07:50 health, 09:16/09:20/09:30
  trade, every 15 min marks, 15:35 close mark, 21:00 nightly, 21:20 health.
  *New:* show next scheduled run and last fire of each.
- Memory during the last run (line chart).
- Determinism: compared against, overlap days, max abs rel diff, books diverged.

### 5.5 Research log

- Experiments table: id, criterion, result (PASS/FAIL), file link — parsed from
  `audit/*.md`. **Failures are first-class** (most results here are FAILs, and that's
  a legitimate outcome).
- Top-K gate results table (`audit/topk_gate/*.json`).
- Document reader: render any audit markdown file.

### 5.6 Warm-up history

Replay of 2024-08 → 2026-09 that precedes the test. **Context, not the test** — must
say so prominently. Chart of signal vs random fan vs equal-weight on ₹10,00,000
notional; two summary metrics.

## 6. Formatting conventions

- **Currency:** Indian grouping — ₹1,00,000, ₹12,34,567. `₹` symbol (the current UI
  writes "Rs"; switch to ₹). Signed values show `+`/`−`. Whole rupees in summaries,
  paise in prices (₹1,202.60).
- **Percent:** 2 decimals for returns (+1.23%), 1 decimal for weights (4.8%).
- **Time:** always IST, 24-hour, `30 Sep 09:16`. The data stores ISO strings with
  `+05:30`; some health/record timestamps are UTC (`Z`) and must be converted.
- **Tickers:** display without the `.NS` suffix.
- **Colour:** profit/loss green/red must pass contrast and not be the only cue.
  Sector colours need ~20 distinguishable categories (NSE industries) or a
  top-N + "other" scheme.
- **Light and dark mode** (the owner works at night).

## 7. Current pain points (engineering's view)

- Default Streamlit look; a long single scroll per page; metrics in uniform grey boxes
  with no hierarchy.
- The portfolio's headline (value, today, total) doesn't dominate the page.
- Walls of yellow/red caveat text; caveats are necessary but should be designed
  (inline, collapsible, iconographic), not pasted.
- Jargon leaks: "log returns", "notional", "volstop", "rvolstop_s12_monthly_20d".
  Human labels needed, with the technical name in a tooltip.
- No market-hours awareness; no "what happens next" (next trade, next rebalance,
  stops scheduled).
- Not designed for phone width.
- Sidebar radio navigation.

## 8. Technical context and constraints

> **Decided 2026-09-30 (owner):** no Streamlit. The dashboard is now plain HTML,
> CSS and JavaScript (`dashboard/web/`) served by a standard-library Python
> server (`dashboard/server.py`) with a read-only JSON API (`dashboard/api.py`).
> The Portfolio design from Claude Design is in `dashboard/design/`; the other
> screens reuse its tokens and components. The notes below describe the stack
> as it was when this brief was written.

- **Current stack:** Streamlit 1.64 + Plotly 7, Python, served from a Docker container
  at `http://127.0.0.1:8501` (localhost only, no auth, single user). Memory limit 1 GB.
- **Data is files, not an API.** The app reads JSON/JSONL/CSV/Parquet written by
  scheduled jobs (§9). The repo is mounted **read-only**; the only writes are the
  health-check output and the Kite token file.
- **Refresh:** data cached ≤ 60 s; marks arrive every 15 min. Push/websocket updates are
  not required; auto-refresh every 1–5 min during market hours is enough.
- **Open decision for design + engineering:** stay on Streamlit (custom theme, CSS,
  custom components; limited control over layout and mobile) **or** move to a small
  local web app (e.g. a static React/HTML front end reading a thin read-only JSON
  endpoint). Design should not be limited by Streamlit if the better experience needs
  more; say which parts need it. Either way: no external services, no analytics, no
  CDNs that phone home with portfolio data.
- **Deliverables wanted from design:** screen designs for §5.1 (desktop + phone), §5.2,
  and the global elements; a component/visual system (type, colour tokens incl. P&L and
  status colours, number formatting, table and chart styles); treatment for the states
  tables above; lighter-touch layouts for §5.3–5.6.

## 9. Data contracts (exact shapes)

All paths relative to the repo root. `…/live/` = `audit/paper/live/`.

**`…/live/holdings_latest.json`** — latest mark (overwritten each mark):
```json
{"ts": "2026-09-30T10:30:00+05:30", "prices": "live", "cash": 416.56,
 "holdings_value": 99465.33, "nav": 99881.89, "capital": 100000.0,
 "pnl_rs": -118.11, "pnl_pct": -0.00118, "day_chg_rs": 0.0, "day_chg_pct": 0.0,
 "unpriced": [], "realised_rs": 0.0, "charges_rs": 118.12,
 "close_mark": false, "stops_pending": [], "cooldown": {},
 "holdings": [{"ticker": "HDFCBANK.NS", "symbol": "HDFCBANK", "sector": "Banks",
   "qty": 4, "avg_price": 964.7, "cost": 3860.2, "ltp": 970.1, "prev_close": 964.7,
   "value": 3880.4, "day_chg_rs": 21.6, "day_chg_pct": 0.0056, "pnl_rs": 20.2,
   "pnl_pct": 0.0052, "entry_price": 964.7, "stop_level": 901.2,
   "opened": "2026-09-30", "priced": true}]}
```
(illustrative values; fractions, not percents: `0.0056` = 0.56%)

**`…/live/marks.csv`** — one row per mark: `ts,cash,holdings_value,nav,pnl_rs,pnl_pct,day_chg_rs`

**`…/live/marks_holdings.jsonl`** — every snapshot above, one per line (history).

**`…/live/state.json`** — the book's state: `cash, holdings{ticker: {qty, avg_price,
entry_price, opened, cost, sector, stop_threshold?, stop_level?, buy_today?}},
deployed{trade_date, names, invested, charges, cash, …}, last_rebalance,
last_trade_day, pending_stops[], cooldown{ticker: sessions}, realised_rs, charges_rs`.

**`…/live/ledger.jsonl`** — one simulated fill per line (fields in §5.1).

**`…/live/trades.jsonl`** — one summary per trading action: `trade_date, kind,
buys, sells, bought_rs, sold_rs, charges, cash, names_after, stops_executed,
stops_registered`.

**`audit/paper/record.jsonl`** — experiment record (§5.3).

**`audit/paper/health.json`** — health (§5.4).

**Kite token status** — from `dashboard/kite_auth.check()`:
`{state: valid|expired|missing|error, minted_at, expires_at, user_type, broker}`.
The token itself is never available to the UI.

## 10. Daily rhythm (IST, weekdays)

| Time | Event | What the UI should reflect |
|---|---|---|
| ~06:00 | Kite token expires | chip → "Login needed" |
| 07:30 | nightly catch-up run | health updates |
| before 09:16 | **user logs in** | chip → "Connected" |
| 09:15 | market opens | status → Open |
| 09:16 (retries 09:20, 09:30) | trade: first funding / monthly rebalance / stop sales / hold | fills appear; header updates |
| every 15 min | mark | values, charts update |
| 15:30 | market closes | status → Closed |
| 15:35 | close mark: stop check | pending stops appear (sold next morning) |
| ~21:00 | experiment run records the previous session | Experiment updates |

## 11. Scope

**Must have (v1):** global elements (§4); Portfolio with header, holdings table, sector
views, intraday chart, trade log, states; Kite login; Experiment headline + chart with
caveats; System health checks + run timeline; ₹ formatting; light/dark; phone layout
for Portfolio and Kite login.

**Should have:** market status + next-event strip ("Next: rebalance on 1 Oct 09:16");
multi-day range selector; per-stock drill-down (click a row → that stock's history,
fills, stop level); comparison of the live book to the end-of-day replay (execution gap
between the 09:16 quote and NSE's official open).

**Could have:** morning login reminder; export holdings/fills to CSV; benchmark line
(NIFTY 50) on the portfolio chart (data exists in the pipeline, not yet exposed);
sector exposure over time.

**Out of scope:** any order placement or trading controls; editing the record; multi-user
accounts; notifications through external services.

## 12. Glossary

| Term | Plain meaning |
|---|---|
| Live book | The ₹1,00,000 paper portfolio traded at live prices |
| Signal / the algorithm | A machine-learning model that ranks ~500 NSE stocks by expected 20-day return; the book holds the top 30 (cap 10% each) |
| Rebalance | Monthly re-selection, first session of the month; limited to 30% turnover |
| Volatility stop (volstop) | Sells a stock that falls below its buy price by more than its usual 20-day move (5%–30%); it can't be re-bought for 21 sessions |
| Random books / null | 20 portfolios run with the same rules but random stock rankings; the yardstick |
| Equal-weight | Every eligible stock in equal amounts |
| Rank | Where the algorithm's cumulative return places among the 21 books (1 = best) |
| Mark | A valuation of the portfolio at a moment's prices |
| LTP | Last traded price |
| Charges | Zerodha delivery costs: STT, stamp duty, exchange, SEBI, GST, DP charge on sells |
| Pre-registration | The test's rules were fixed in writing before it started (`audit/P2_PAPER_PERMUTATION_TEST.md`) |
