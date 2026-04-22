# 02 — Data Pipeline

## Universe construction

### Target composition (approx. 150 tickers)

- **NIFTY 50** constituents (as of chosen reference date).
- **Top 20 per sector** (configurable list):
  - Pharma / Healthcare
  - Information Technology
  - Auto + New-Age EV
  - Oil, Gas, Power, Fuel
  - Defence + PSU Capital Goods
  - Banking / Financial Services
  - FMCG (optional)
  - Metals (optional)

After dedup, expect 130–160 unique tickers.

### Survivorship bias mitigation

Naively taking "today's top 20 per sector" selects winners with
hindsight. The plan:

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

The Hydra config `configs/data/universe_v1.yaml` encodes both the
current universe and the path to historical snapshot files.

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

### Primary: yfinance

- Ticker format: `RELIANCE.NS`, `TCS.NS`, etc.
- Use `auto_adjust=True` so splits/dividends are baked in.
- Rate-limit aware; retry with exponential backoff.
- Cache raw responses to `data/raw/yfinance/<ticker>/<date-range>.parquet`
  to avoid re-hitting the API.

### Fallback: Kaggle / HuggingFace

- For delisted names and pre-2015 history where yfinance is spotty.
- Loaded into a staging table and reconciled against yfinance where they
  overlap (log mismatches > 0.5% on adjusted close).

### Live: Zerodha Kite (stub)

- `zerodha_source.py` implements the `MarketDataSource` interface but
  raises `NotImplementedError` in v1 with a TODO comment referencing the
  Kite Connect historical data API. The interface is there so training
  code is agnostic.

## Storage

- **Raw OHLCV**: partitioned Parquet, `data/ohlcv/year=YYYY/ticker=XYZ.parquet`.
  Columns: `date, open, high, low, close, volume, adj_close, source`.
- **Aligned panel**: single Parquet per split,
  `data/panels/train.parquet`, `val.parquet`, `test.parquet`.
  Schema: `(date, ticker, feature_1, ..., feature_F, is_tradeable)`.
- **Metadata in Postgres** (`market.instruments`,
  `market.universe_snapshots`, `market.corporate_actions`,
  `market.trading_calendar`).
- **Data hash**: each panel file has a sidecar `.sha256` and a row in
  `market.dataset_versions` so runs are pinned to data.

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
4. Corporate actions: yfinance `auto_adjust=True` handles most. Verify
   by sampling known splits (Reliance 1:1 bonus, etc.) and writing a
   unit test against the expected adjusted close.

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

### Optional: sentiment embeddings (v2)

- Weekly: scrape news headlines per ticker, run a local finance-tuned
  embedding model (e.g. `FinBERT`), store as
  `data/sentiment/weekly/ticker=XYZ.parquet` with columns
  `(week_start, emb_0..emb_d)`.
- Joined at feature-build time as additional feature columns.
- Entirely offline — the RL loop never calls an LLM.

## Splits

- Train: 2014-01-01 .. 2021-12-31
- Val:   2022-01-01 .. 2022-12-31
- Test:  2023-01-01 .. 2024-12-31
- Purge: 1 month gap between splits to avoid overlap in rolling features.

Walk-forward retraining shifts these windows; see `05_training.md`.

## CLI

```bash
# 1. Build universe from yaml (hits NSE + yfinance metadata)
python scripts/build_universe.py data=universe_v1

# 2. Fetch OHLCV for the universe into raw/
python scripts/fetch_data.py data=universe_v1 \
    data.start=2014-01-01 data.end=2024-12-31

# 3. Build aligned panels with masks and features
python scripts/build_features.py data=universe_v1
```

Each command writes its hash and inputs to `market.dataset_versions` so
downstream runs link deterministically to the data.

## Acceptance criteria for Phase 1

- `scripts/build_universe.py` produces a Postgres row in
  `market.universe_snapshots` and a yaml snapshot on disk.
- `scripts/fetch_data.py` caches raw data and is idempotent on re-run
  (no duplicate API calls when cache is warm).
- `scripts/build_features.py` produces `train/val/test.parquet` with a
  deterministic SHA256 given pinned inputs.
- Unit tests:
  - Known split (pick a real one, assert adjusted close continuity).
  - `is_tradeable` mask is `False` for a ticker listed in 2017 for all
    dates in 2014–2016.
  - No NaN in numeric columns on tradeable rows.
  - Feature computation contains no `.shift(-k)` or equivalent
    look-ahead (static scan in a test).
- Panel shape: roughly `[~2700 trading days, ~150 tickers, ~15 features]`.
