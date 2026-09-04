# Kite vs yfinance: dataset comparison

**Generated** 2026-09-04 · Kite store `data/kite_ohlcv/` (749,938 bars, 164 instruments,
2005-01-03 → 2026-09-04) vs the existing yfinance store `data/ohlcv/`
(2014-01-01 → 2024-12-30). Overlap analysed: **2014-01-01 → 2024-12-31**.

Neither `data/ohlcv/` nor `data/panels/` was modified. The new artefacts are
`data/kite_ohlcv/` and `data/panels_kite/`.

---

## 1. Headline

| | yfinance | Kite |
|---|---|---|
| Rows in overlap | 405,134 | 405,635 |
| Tickers in overlap | 163 | 162 |
| Trading days in overlap | 2,710 | 2,721 |
| Full history start | 2014-01-01 | **2005-01-03** |
| Latest bar | 2024-12-30 | **2026-09-04** |
| Split/bonus adjusted | yes | yes |
| Dividend adjusted | yes (`auto_adjust=True`) | **no** (accepted) |
| Demerger adjusted | no | partially, inconsistently (§4) |

Kite adds **~9 years of prior history and ~1.7 years of recent history**. Over the
overlap the two feeds agree on structure but not on level, for reasons that are
almost entirely explainable.

## 2. Coverage

**Tickers.** The single ticker Kite cannot serve is `MCDHOLDING.NS` — McDowell
Holdings, suspended from NSE and BSE with no successor listing. It is recorded in
`DELISTED_SYMBOLS` so it fails loudly rather than silently shrinking the universe.
Three others resolve only because the alias map exists: `LTIM.NS → LTM`,
`STLTECH.NS → STLTECH-BE`, `TATAMOTORS.NS → TMPV`.

**Kite recovers history yfinance lost.**

| Ticker | yfinance bars | Kite bars | Note |
|---|---|---|---|
| `ARE&M.NS` | **290** | 2,716 | yfinance only has data from the Oct-2023 `AMARAJABAT → ARE&M` rename. Kite has the full series. **+2,426 bars.** |
| `ETERNAL.NS` | 828 | 854 | yfinance is missing ~21 scattered bars. Each 2-day hole cascades: `align_panel` marks the run non-tradeable, so the yfinance panel loses ETERNAL from 2023-07 to 2023-11 entirely. |
| `GRSE.NS`, `MAZDOCK.NS` | −16, −13 | | scattered yfinance holes |

**Where Kite has less.** `OLECTRA.NS` — Kite's history begins 2015-01-01 rather
than 2014-01-01 (245 fewer bars). It is a truncated start, not a hole; alignment
handles it as a later listing date.

**Calendar.** Kite has **13 trading days yfinance lacks**: 2014-03-22, 2014-10-23,
2015-02-28, 2015-11-11, 2016-10-30, 2019-02-13, 2019-03-29, 2020-02-01, 2023-11-12,
2024-01-20, 2024-03-02, 2024-05-18, 2024-12-31. These are NSE Saturday special
sessions (Budget day, Muhurat trading, disaster-recovery live sessions) plus
2024-12-31. yfinance simply does not carry them; the broker's own feed does.
yfinance has 2 days Kite lacks (2014-04-24, 2014-10-15), both likely phantom.

## 3. Close-price ratio (Kite ÷ yfinance)

401,632 joined `(ticker, date)` rows.

| stat | value |
|---|---|
| median | 1.0490 |
| mean | 1.1036 |
| p01 / p05 | 0.516 / 0.995 |
| p25 / p75 | 1.016 / 1.124 |
| p95 / p99 | 1.459 / 1.936 |
| min / max | 0.251 / 2.943 |
| within ±1% | 13.4% |
| within ±5% | 48.0% |
| within ±25% | 85.4% |

Those numbers look alarming until you split them by year:

| year | median ratio | p90 |
|---|---|---|
| 2014 | 1.125 | 1.535 |
| 2016 | 1.104 | 1.504 |
| 2018 | 1.076 | 1.359 |
| 2020 | 1.046 | 1.243 |
| 2022 | 1.027 | 1.099 |
| 2024 | **1.011** | 1.040 |

The ratio decays monotonically toward 1.0 as it approaches the present. **This is
the dividend signature**, and it is the expected consequence of a decision already
taken: yfinance back-adjusts for dividends, Kite does not, so yfinance's historical
prices are progressively *lower* the further back you go. The gap is largest for
high-yield PSU and commodity names — `HINDZINC` (median 1.81), `COALINDIA` (1.69),
`RECLTD` (1.68), `PFC` (1.58), `OIL` (1.44), `NATIONALUM` (1.40), `POWERGRID` (1.40) —
and near 1.0 for low-yield names.

**This affects levels, not returns.** A constant multiplicative factor cancels in
`log(P_t / P_{t-1})`; a dividend adjustment is a smooth drift, so it perturbs daily
returns by roughly the daily dividend accrual (basis points). Every feature in
`FEATURE_COLS` is either a return, a ratio, or a z-score, so the panels are far more
comparable than the raw ratio spread suggests. The one place the level matters is
`dollar_volume_20`, which is ~5-10% higher in the Kite panel for the same day.

## 4. Where they genuinely disagree: adjustment discontinuities

The ratio should be *smooth* within a ticker. A one-day jump in the ratio means one
feed applied a corporate-action adjustment the other did not. **37 of 162 tickers
have at least one ratio jump greater than 5%.** The largest:

| Ticker | Date | Ratio jump | Diagnosis |
|---|---|---|---|
| `CGPOWER.NS` | 2015-01-01 | ×2.93 | **Kite archive seam** — masked |
| `MASTEK.NS` | 2015-01-01 | ×2.92 | **Kite archive seam** — masked |
| `NIITLTD.NS` | 2023-06-08 | ×2.33 | demerger; yfinance shows −76.13%, Kite −44.38% (Kite partially adjusted) — masked |
| `CGPOWER.NS` | 2016-03-15 | ×0.35 | consumer-products demerger; **yfinance adjusted, Kite did not** — **open** |
| `STLTECH.NS` | 2016-06-15 | ×1.84 | Sterlite Power demerger — **open** |
| `NMDC.NS` | 2022-10-27 | ×1.79 | NMDC Steel demerger — **open** |
| `HINDZINC.NS` | 2016-04-06 | ×0.87 | special dividend, yfinance-only adjustment |
| `COALINDIA.NS` | 2014-01-17 | ×0.90 | buyback / special dividend |

**The 2015-01-01 seam is the most important finding.** For `MASTEK` and `CGPOWER`,
Kite's pre-2015-01-01 archive is already adjusted for a demerger that had not yet
happened, while its 2015+ bars are not. The two halves are on different scales and
join with a fictitious **+195% / +191% single-day return**. yfinance is continuous
across that date, which is how the artefact was identified. Both are now in
`KNOWN_CORPORATE_EVENTS` as `kind="feed_seam"` and masked. Only these two tickers
are affected — this is *not* a global Kite boundary.

## 5. Low-price quantisation in the pre-2014 extension

NSE's tick size is ₹0.05. Kite's back-adjusted prices for stocks with large
cumulative split/bonus factors fall to a few rupees in the 2005-2013 window, and the
tick then becomes a large fraction of the price:

| price band | rows | share of 2005-2013 |
|---|---|---|
| < ₹5 | 4,394 | 1.6% |
| < ₹10 | 13,567 | 5.0% |
| < ₹20 | 30,599 | 11.3% |
| < ₹50 | 78,981 | 29.2% |

Worst offenders (median 2005-2013 adjusted price): `BAJFINANCE` ₹4.00,
`MOTHERSON` ₹5.10, `WELSPUNLIV` ₹6.21, `ASHOKLEY` ₹7.90, `SONATSOFTW` ₹11.00,
`BEL` ₹13.05. `BAJFINANCE` oscillates between ₹0.50 and ₹1.50 through 2008-09,
producing a return series of ±50% and ±100% steps that is **pure rounding noise** —
five of those days appear on the triage list purely because of it.

This is the main caveat on the 9-year history extension and needs a decision (§7).

## 6. New panel: `data/panels_kite/`

| split | rows | dates | days | tickers | tradeable |
|---|---|---|---|---|---|
| train | 767,893 | 2005-01-03 → 2023-12-29 | 4,711 | 163 | 623,555 |
| val | 37,001 | 2024-02-01 → 2024-12-31 | 227 | 163 | 36,774 |
| test | 64,222 | **2025-02-01 → 2026-09-04** | 394 | 163 | 63,828 |

Purge gaps of 34 and 32 calendar days, same convention as the panel it replaces
(the month is cut from the start of the later segment). SHA256 sidecars written and
verified. The 2025-26 test window post-dates every architecture and hyperparameter
decision in the repo; normalisation statistics are computed from `train` only, so
nothing from 2025+ can reach a training-time statistic.

**`beta_nifty_60d` is a real feature again.** In the existing `data/panels`, `^NSEI`
was never fetched, so `_load_index_rets` returned nothing and beta was the constant
**1.0 on every one of 276,005 tradeable training rows** — a silently dead feature.
The Kite fetch pulls NIFTY 50 (NSE INDICES segment, instrument token 256265, daily
bars back to 2005-01-03) and stores it as `^NSEI`, so beta now has σ ≈ 0.52. This is
a real feature-distribution change between the two panels, not just a data-source
swap.

## 7. Open items

1. **Low-price quantisation (§5).** Options: exclude pre-2010 from training; add a
   minimum-adjusted-price liquidity filter; or accept the noise. Recommend a filter —
   ±50% steps on 5% of the extension is a lot of fictional volatility to hand a policy.
2. **Three unresolved demerger discontinuities** — `CGPOWER` 2016-03-15 (−71.7%),
   `NMDC` 2022-10-27 (+85.7%), `STLTECH` 2016-06-15 (+55.0%) and 2025-04-24 (+77.1%).
   Each is a real corporate action Kite did not adjust for, but the exact mechanism
   was not confirmed, so per this module's policy they are reported and **not**
   masked. Adding them is a one-line edit each in `KNOWN_CORPORATE_EVENTS`.
3. **`ADANIENT.NS` 2015-06-03 (−41.9% Kite / −38.8% yfinance).** The June-2015 Adani
   Enterprises restructuring (Adani Ports / Power / Transmission spun out). The price
   never retraces, which is the demerger signature rather than the crash signature.
   Strongest single candidate for promotion to the known-event list.
4. **`beta_nifty_60d` (§6)** — confirm the new panel is meant to carry a live beta.
   If a strict like-for-like A/B against the old panel is wanted first, delete
   `data/kite_ohlcv/year=*/ticker=^NSEI.parquet` and rebuild.
5. **`OLECTRA.NS`** starts a year late in Kite. Harmless, noted for completeness.
