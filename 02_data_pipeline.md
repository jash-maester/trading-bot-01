# 02 — Data Pipeline

## Status

**PARTIAL** — last verified against `60ba1a6` (the A0/A1/A2 audit reports,
2026-09-05). Sources, storage layout, alignment, the 15-feature set, SHA256
sidecars and `market.dataset_versions` are implemented and work.

Known broken, in order of consequence:

- **The universe is not point-in-time.** `market.universe_snapshots` holds 0
  rows and has no read site anywhere in `src/` or `scripts/` (A0 §2.4).
  Survivorship bias is live — see below.
- **The panels on disk do not match the universe.** Both hold **163 tickers**
  on the pre-`a5d8b79` 8-sector taxonomy, against a universe that now defines
  645 fetched / 504 active across 14 sectors. Only 143 overlap, so 361 of 504
  observation columns are permanently zero (A0 §2.6, A2 §1.1).
- **`beta_nifty_60d` is constant 1.0** across all 276,005 tradeable rows of
  `data/panels/train.parquet` (`data/features.py:293`, A0 §2.8).
- **The purge gap is 19–23 trading days against 60-day features** (A1 §3.2).
- **`data.panels_root` is honoured by one script only**
  (`scripts/build_features.py:55`), so `data/panels_kite` is write-only
  (A0 §3.2).

---

## Universe construction

### Composition — as built

`src/trader/data/universe.py` defines **645 tickers across 14 sectors**
(`SECTOR_IDS`, `universe.py:236-252`). `all_tickers()` returns all 645 and is
what the data pipeline fetches and panels; `active_tickers()` returns the
**504** the model actually trades and is what builds every observation space
(`training/runner.py:107`, `scripts/paper_run.py:272`).

Sector assignment comes from NSE Indices' "Industry" column
(`ARCHITECTURE.md:37`). That provenance is documented there and is not
reproducible from anything in `src/` — the sector lists are hand-maintained
literals in `universe.py`. **UNVERIFIED** against a downloaded NSE file.

Sector sizes, measured (A0 §2.6):

```
banking 121   capital_goods 99   pharma 70   auto 48   fmcg 45   chemicals 45
oil_gas_power 42   consumer_durables 41   it 36   services 26   metals 24
defence 19   construction_materials 16   telecom 13
```

`INACTIVE_SECTORS` (`universe.py:273-282`) holds **5 of those 14** out of
training — chemicals, construction_materials, consumer_durables, services,
telecom — 141 of the 645 names. Their data *is* fetched and panelled; they are
held out only to keep the observation width, and therefore the wall-clock cost
per run, at 504 rather than 645. Emptying the set trades the full 645. The five
were introduced in `a5d8b79` (2026-09-04) as a side effect of recovering eight
NIFTY 50 names (ASIANPAINT, TITAN, BHARTIARTL, ULTRACEMCO, SHREECEM, GRASIM,
ADANIPORTS, UPL) that the original 8-sector taxonomy excluded.

> **DEFECT — the panels predate the universe.** Both `data/panels` and
> `data/panels_kite` contain 163 tickers, of which only **143** are in
> `active_tickers()` and only 153 are in `all_tickers()` at all. Because
> `training/runner.py:107` builds the env on `active_tickers()` = 504,
> **361 of 504 observation columns are all-zero and permanently
> `is_tradeable=False`** (`env/panel_env.py:437-440`) while costing full
> forward and backward compute — 71.6% of the compute budget spent on zeros
> (A2 §1.1). Both panels also still carry the old 8-sector taxonomy, including
> eight names at the phantom `sector_id == 0` (the eight NIFTY names above),
> 5.54% of tradeable train rows (A1 §7.8). The `sector_id == 0` regression fix
> landed in `universe.py:295-320`; the data on disk predates it. Rebuilding one
> panel is R1 in `09_revamp_and_audit.md` §5.

**The original target, for the record:** NIFTY 50 plus top-20 per sector across
~6 sectors, 130–160 unique tickers after dedup. That is what the code did until
`a5d8b79`, and it is what the 163-ticker panels on disk still reflect.

### Survivorship bias mitigation

> **DEFECT — NOT IMPLEMENTED. The universe is not point-in-time, and this
> section describes an intention, not the system.**
>
> `market.universe_snapshots` holds **0 rows**, and the only references to it
> anywhere are its own definition (`src/trader/db/market_models.py:46-47`), the
> single write site (`scripts/build_universe.py:48,60`), a round-trip test, and
> the Alembic DDL. **There is no read site in `src/` or `scripts/`** (A0 §2.4).
> The panel builder resolves the universe by calling `all_tickers()` directly
> (`scripts/build_features.py:59`); the env builder calls `active_tickers()`
> (`training/runner.py:107`). Both return today's hand-maintained list.
>
> Worse than unwired: the one write site would not produce a point-in-time table
> even if it were run. `scripts/build_universe.py:26-27,60-64` writes **one**
> row — `symbols = all_tickers()` (today's list) stamped with
> `effective_date = cfg.data.start_date` — current membership back-stamped onto
> the start of history. The on-disk fallback `data/raw/universe_v1.parquet`
> holds 167 rows dated Apr 24, 8 of them in an `'unknown'` sector, and is read
> by nothing.
>
> `00_overview.md` lists a survivorship-bias-aware universe among the explicit
> non-decisions and calls it "the silent killer". **It is not satisfied.** This
> affects every backtest the project has produced. Fixing it is R1
> (`09_revamp_and_audit.md` §5); NSE bhavcopy archives, which carry
> point-in-time series codes (EQ / BE / T2T) that Kite cannot provide, are the
> stated route.

Naively taking "today's top 20 per sector" selects winners with
hindsight. The plan — **intended behaviour; none of the four steps below is
wired up**:

1. Pull NIFTY 50 historical constituent lists (available on NSE archive)
   and snapshot for each calendar year.
2. For each sectoral list, record the effective date of the selection.
   When backtesting over 10 years, the training data **uses the universe
   as it existed on each date** via the `universe_snapshots` table.
3. Include **delisted or merged** tickers. yfinance does not always
   serve these cleanly; Kaggle datasets (`NSE Stock Data`, `Indian Stock
   Market`) and the NSE bhavcopy archive are fallbacks.
4. Flag any ticker where more than 10% of expected trading days are
   missing, and treat those gaps honestly (not by interpolation — see
   Alignment below).

The Hydra config `configs/data/universe_v1.yaml` was meant to encode both the
current universe and the path to historical snapshot files. Both of its keys are
**dead**: nothing reads `data.universe_version` (`universe_v1.yaml:8`) or
`data.universe_snapshot_path` (`:9`), and `scripts/build_universe.py:32`
hardcodes the same path when writing, so the key could never redirect it
(A0 §3.1).

## Sources

All sources implement `trader.data.sources.base.MarketDataSource`:

```python
class MarketDataSource(Protocol):
    def fetch_ohlcv(
        self,
        tickers: list[str],
        start: datetime,
        end: datetime,
        interval: Literal["1d", "1h", "15m"] = "1d",
    ) -> pl.DataFrame: ...

    def fetch_corporate_actions(
        self, tickers: list[str]
    ) -> pl.DataFrame: ...
```

Both OHLCV stores are provenance-tagged: `data/ohlcv` rows carry
`source='yfinance'`, `data/kite_ohlcv` rows carry `source='zerodha'` (A0 §2.1).

### Primary: yfinance — populated `data/panels`

- Ticker format: `RELIANCE.NS`, `TCS.NS`, etc.
- Uses `auto_adjust=True` (`sources/yfinance_source.py:101`) so splits **and
  dividends** are baked in. The resulting panel is therefore **total-return**.
- Rate-limit aware; retry with exponential backoff.
- Cache raw responses to `data/raw/yfinance/<ticker>/<date-range>.parquet`
  to avoid re-hitting the API.
- `data/ohlcv` holds 405,134 rows / 163 tickers / 2014-01-01 → 2024-12-30, all
  `source='yfinance'`, and is what `data/panels` was built from (A0 §2.1).

### Fallback: Kaggle / HuggingFace

- For delisted names and pre-2015 history where yfinance is spotty.
- Loaded into a staging table and reconciled against yfinance where they
  overlap (log mismatches > 0.5% on adjusted close).
- `KaggleSource` exists (`sources/kaggle_source.py:11`) and no entrypoint calls
  it. Whether it has ever populated a store is **UNVERIFIED** — no audit
  covered it.

### Live: Zerodha Kite — no longer a stub

- `ZerodhaSource.fetch_ohlcv` (`sources/zerodha_source.py:274`) fetches real
  read-only historical bars; it is live, not `NotImplementedError`. Only
  `fetch_corporate_actions` still raises (`:327`), so
  `market.corporate_actions` has no producer.
- `data/kite_ohlcv` holds 2,217,214 rows / 656 tickers (incl. `^NSEI`) /
  2005-01-03 → 2026-09-04. The panels built from it, `data/panels_kite`, are
  **price-return**, not total-return.
- **Never call an order-placing Kite endpoint.** Read-only: `instruments`,
  `historical_data`.

> **DEFECT — the Kite panel is unreachable from training.** `scripts/train.py:33`,
> `scripts/walk_forward.py:59` and `scripts/paper_run.py:266` all hardcode
> `orig_cwd / "data" / "panels"`. `data.panels_root` is read only by
> `scripts/build_features.py:55`, so `data=kite_v1` builds into
> `data/panels_kite` and **no consumer can read it** (A0 §3.2, A1 §7.4).
> Everything that can currently be run is wired to the April yfinance panel.

### Price series: total return vs price return

`data/panels` is **total-return** (dividends back-adjusted); `data/panels_kite`
is **price-return**. Proven by joining the two on `(date, ticker)` over their
2014–2021 overlap and finding one-day down-steps in the ratio on ex-dividend
dates: COALINDIA 12 such days totalling 57.65% (7.43%/yr), ONGC 19 days
(4.26%/yr), ITC 11 days (3.12%/yr); median implied yield across 154 tickers
1.04%/yr (A0 §2.3). No in-repo corporate-actions dataset exists to cross-check
the ex-dates — `market.corporate_actions` has 0 rows — so the cross-feed test is
the only evidence; **UNVERIFIED** against an external ex-date list.

`adj_close` carries no information in either panel: `df.adj_close.equals(df.close)`
is `True` in both, because yfinance aliases it
(`sources/yfinance_source.py:143-149`) and Kite mirrors it
(`sources/zerodha_source.py:608`). Do not use it as a total-return marker.

**Which convention is correct has never been decided** — `09_revamp_and_audit.md`
§2 row 3 marks it UNRESOLVED. Results from before and after the switch are not
comparable.

## Storage

- **Raw OHLCV**: partitioned Parquet, `data/ohlcv/year=YYYY/ticker=XYZ.parquet`.
  Columns: `date, open, high, low, close, volume, adj_close, source`.
- **Aligned panel**: single Parquet per split,
  `data/panels/train.parquet`, `val.parquet`, `test.parquet`.
  Schema: `(date, ticker, feature_1, ..., feature_F, is_tradeable)`.
- **Metadata in Postgres** (`market.instruments`,
  `market.universe_snapshots`, `market.corporate_actions`,
  `market.trading_calendar`). **DEFECT: all four are empty.** Measured
  2026-09-04: `instruments` 0, `universe_snapshots` 0, `corporate_actions` 0,
  `trading_calendar` 0, `dataset_versions` 6 (A0 §2.4). Only
  `dataset_versions` has ever been written.
- **Data hash**: each panel file has a sidecar `.sha256` and a row in
  `market.dataset_versions` so runs are pinned to data. This part works — all
  six `.sha256` sidecars verify against `shasum -a 256` (A0 §2.6), so neither
  panel has been tampered with since it was written. They are simply old.

There is a second panel root, `data/panels_kite`, written from the Kite store.
Its layout is identical. See the defect note under *Live: Zerodha Kite* — no
training entrypoint can read it.

## Alignment — the important part

Goal: a dense `[T, N, F]` tensor with a `[T, N]` `is_tradeable` mask,
without lying about stocks that did not exist.

### Rules

1. Build the trading calendar from the NSE (holidays excluded), not from
   pandas business days (which miss Indian holidays).
2. For each `(date, ticker)`:
   - If the date is before the ticker's first trade, set
     `is_tradeable=False`, features = sentinel zeros. **The mask is the
     source of truth; the agent must never see the zeros as signal.**
   - If the ticker was delisted / suspended on that date,
     `is_tradeable=False`.
   - If there is a corporate action that invalidates the day (rare),
     mark untradeable.
   - Missing bars inside the tradeable span: forward-fill OHLC with
     a **max 1-day gap**, then mark untradeable beyond that. Volume gaps
     are filled with 0, which is honest.
3. **Do not interpolate close prices across long gaps.** Interpolation
   is a look-ahead in disguise.
4. Corporate actions: yfinance `auto_adjust=True` handles splits and dividends.
   It does **not** handle demergers — value moves to a separately listed entity,
   so the parent shows a fake catastrophic loss (NIITLTD −76.13%, MASTEK
   −66.00%; `CLAUDE.md` "Known-dangerous ground"). Verify by sampling known
   splits (Reliance 1:1 bonus, etc.) and writing a unit test against the
   expected adjusted close.
5. Symbol renames are not handled. A current instrument dump has no memory of
   them (`LTIM→LTM`, `TATAMOTORS→TMPV`, `STLTECH→STLTECH-BE`, `MCDHOLDING`
   delisted), so a symbol-keyed join silently drops history.

**What the mask actually looks like on disk.** The panels are dense: every
ticker has a row on every date and `close`/`adj_close` are never NULL —
non-tradeable rows are **sentinel-zeroed, not null**
(`data/alignment.py:20-25, 77-79`). 35,794 rows (11.1%) of `data/panels` train
and 134,306 (17.5%) of `panels_kite` train carry `close == 0.0`, and no
zero-close row is ever marked tradeable, so the mask does hold (A0 §2.2). It is
mask *and* zero-fill, not "mask, never zero-fill" — `ARCHITECTURE.md` is
inaccurate on this point, `02` is not.

Measured tradeable fractions (A0 §2.2): `data/panels` train 0.8587
(276,005 of 321,436), val 0.9876, test 0.9942.

### Feature set (initial)

Per ticker, per day, computed from the adjusted OHLCV panel:

- `log_return_1d, log_return_5d, log_return_20d`
- `realized_vol_20d, realized_vol_60d`
- `rsi_14, macd, macd_signal, macd_hist`
- `bbw_20` (Bollinger bandwidth), `z_close_20`
- `volume_z_20, dollar_volume_20`
- `atr_14` (used by the broker for slippage)
- `beta_nifty_60d`
- `sector_id` (categorical, one-hot or learned embedding)
- `is_tradeable` (mask, stored with features)

All features are computed from information strictly before the action
time. `src/trader/data/features.py` has a single function
`compute_features(df: pl.DataFrame) -> pl.DataFrame` that asserts
monotonic non-decreasing dates and no forward references.

Feature order matters and is fixed by `FEATURE_COLS` (`data/features.py:16-17`):
index 0 is `log_return_1d`, index 2 is `log_return_20d`. Two consumers index
this list positionally — see the `MomentumTopK` defect in `03_environment.md`.

> **DEFECT — `beta_nifty_60d` is silently dead whenever the index is absent.**
> `data/features.py:291-293`:
>
> ```python
> def _add_beta(df, index_rets):
>     if index_rets is None:
>         return df.with_columns(pl.lit(1.0, ...).alias("beta_nifty_60d"))
> ```
>
> `data/ohlcv` contains no `^NSEI` file, so `_load_index_rets` returned `None`
> and every row got the literal 1.0. Measured over all 276,005 tradeable rows of
> `data/panels/train.parquet`: `mean=1.0 std=0.0 min=1.0 max=1.0 nunique=1`
> (A0 §2.8). The fallback does not warn. **The fix that landed is a data fix,
> not a code fix** — `configs/data/kite_v1.yaml:32-33` sets `fetch_index: true` /
> `index_ticker: "^NSEI"` and the Kite panel has a live beta (mean 0.868, std
> 0.519). `features.py:293` is unchanged since `08d11fe` (2026-04-22) and will
> still silently emit 1.0 for any future panel built without an index series.

> **DEFECT — a zero-variance feature becomes a fixed +1 bias, not a zeroed
> channel.** `data/feature_stats.py:43-45` substitutes `(0.0, 1.0)` for mean/std
> whenever `std < 1e-8`. A constant-1.0 feature therefore normalises to
> `(1.0 − 0.0)/1.0 = 1.0` and feeds a **constant +1** into every convolution.
> `data/panels/train.feature_stats.json` records exactly that for
> `beta_nifty_60d` (A0 §2.8). The substitution is a reasonable guard against a
> divide-by-zero and a bad guard against a dead channel; it is what turned one
> pipeline bug into a silent model input. **Check every feature has nonzero
> variance before trusting a run.**

> **Scale note — `dollar_volume_20` is nine orders of magnitude above every
> return feature.** Measured over tradeable train rows: mean 1.288e+09, std
> 2.442e+09, range [33,230, 6.286e+10] (A0 §2.9). It *is* standardised at the
> model boundary (`training/runner.py:114,219-220`), so the network input is not
> raw — but it is standardised against a strongly right-skewed distribution and
> there is no log transform anywhere in `features.py`. On the Kite panel it also
> takes the value 0.0 on 1,183 tradeable rows.

### Optional: sentiment embeddings (v2)

- Weekly: scrape news headlines per ticker, run a local finance-tuned
  embedding model (e.g. `FinBERT`), store as
  `data/sentiment/weekly/ticker=XYZ.parquet` with columns
  `(week_start, emb_0..emb_d)`.
- Joined at feature-build time as additional feature columns.
- Entirely offline — the RL loop never calls an LLM.

## Splits

**As built** (A0 §2.2, measured on the files):

| Panel | train | val | test |
|---|---|---|---|
| `data/panels` (yfinance, the one that runs) | 2014-01-01 → 2021-12-31, 1,972 dates | 2022-02-01 → 2022-12-30, 228 | 2023-02-01 → 2024-12-30, 469 |
| `data/panels_kite` (unreachable) | 2005-01-03 → 2023-12-29, 4,711 | 2024-02-01 → 2024-12-31, 227 | 2025-02-01 → 2026-09-04, 394 |

The purge month is cut from the *start* of each later segment, which is why val
begins in February, not January.

> **DEFECT — the 1-month purge is far shorter than the feature lookback.**
> Features are computed on the **full continuous panel** and only then split
> (`scripts/build_features.py:90` precedes `:134-136`), so every rolling window
> — `realized_vol_60d` (`features.py:147-152`), `beta_nifty_60d`
> (`features.py:317-327`) — spans the boundary freely. `compute_windows`
> measures the gap in **calendar months** and never against the feature-window
> length.
>
> Measured (A1 §3.2-3.4): the 1-month purge is **19–23 trading days** against a
> **60-trading-day** window, leaving 36–40 contaminated rows at **8 of 8**
> walk-forward boundaries and at both fixed-split boundaries. The Kite split has
> the same defect, and `configs/data/kite_v1.yaml:42-44` asserts the opposite in
> a comment.
>
> | `purge_months` | trading-day gap (min–max) | every boundary ≥ 59? |
> |---:|---:|:---:|
> | 1 | 19 – 23 | no |
> | 2 | 38 – 42 | no |
> | **3** | **59 – 63** | **yes** — clears by zero days on W1 |
> | 4 | 79 – 86 | yes — first value with margin |
> | 6 | 120 – 129 | yes, but only 3 windows fit 2014–2024 |
>
> **The specified purge is 3 months minimum; use 4 for margin.** No finite purge
> closes the EWM features (`rsi_14`, `atr_14`, `macd*`), whose support is
> unbounded; measured, a 3-month purge leaves ≈1% of a feature σ and a 6-month
> purge ≈0.01% (A1 §3.5). That residual is second-order; the 60-day hard windows
> are not.
>
> A second consequence: `tests/unit/test_walk_forward.py:56-72` asserts
> `28 <= gap_train_val <= 32` — a **calendar**-month gap that is never compared
> to the feature window — so the test passes on exactly the configuration that
> leaks.
>
> A third: `scripts/walk_forward.py:64-75` rebuilds its "full panel" by
> concatenating the *already purged* splits, so the 41 purge-month sessions are
> simply absent — two month-long holes (2022-01, 2023-01) that silently truncate
> two walk-forward val segments by a month and sit mid-episode in two test
> segments (A1 §7.3).

Walk-forward retraining shifts these windows; see `05_training.md`.

## CLI

```bash
# 1. Build universe from yaml (hits NSE + yfinance metadata)
#    NOTE: writes ONE non-point-in-time row; see the survivorship defect above.
python scripts/build_universe.py data=universe_v1

# 2. Fetch OHLCV for the universe into raw/
python scripts/fetch_data.py data=universe_v1 \
    data.start=2014-01-01 data.end=2024-12-31

# 3. Build aligned panels with masks and features
python scripts/build_features.py data=universe_v1

# 3b. The Kite path. Builds into data/panels_kite, which nothing downstream
#     can read (see the Kite defect above).
python scripts/fetch_kite_data.py data=kite_v1
python scripts/build_features.py data=kite_v1
```

Each command writes its hash and inputs to `market.dataset_versions` so
downstream runs link deterministically to the data. That link is not carried
into MLflow: no stored run logs a panel path or panel SHA256, and none logs any
`data.*` parameter at all (A0 §4.2), so which panel a run saw can only be
inferred from `n_tickers` and the run timestamp.

## Acceptance criteria for Phase 1

| Criterion | State |
|---|---|
| `scripts/build_universe.py` produces a Postgres row in `market.universe_snapshots` and a yaml snapshot on disk | **FAILS** — the table has 0 rows, and the write it would perform is not point-in-time (A0 §2.4) |
| `scripts/fetch_data.py` caches raw data and is idempotent on re-run | **UNVERIFIED** — no audit tested idempotence |
| `scripts/build_features.py` produces `train/val/test.parquet` with a deterministic SHA256 given pinned inputs | **PASSES** — all six sidecars across both panel roots verify (A0 §2.6) |
| Unit tests: known split; `is_tradeable` False for a 2017 listing across 2014–2016; no NaN on tradeable rows; static no-lookahead scan | **UNVERIFIED as a set** — the suite is green (368 tests) but no audit checked that these four specific tests exist and assert what they claim |
| Panel shape ≈ `[~2700 trading days, ~150 tickers, ~15 features]` | **Actual:** `data/panels` train is `[1,972 dates, 163 tickers, 15 features]`; `data/panels_kite` train is `[4,711, 163, 15]`. The env is built at N=504, so the realised observation is `[60, 504, 15]` with 361 zero columns. |

Three criteria this phase should have had and did not:

- **Every feature has nonzero variance on tradeable rows.** `beta_nifty_60d`
  failed this for five months undetected.
- **The purge between any two segments is ≥ the longest feature window, counted
  in trading days.**
- **The panel's ticker set equals `active_tickers()`.** The 143/504 mismatch is
  invisible to every existing test.
