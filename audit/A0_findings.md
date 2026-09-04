# A0 — Read-only forensics

**Run:** 2026-09-04 · read-only · no file modified except this one.
**Repo state:** `main` @ `1c20df6` (advanced from `429470f` mid-audit; the
`docker/docker-compose.yml` and `scripts/win_bootstrap.sh` edits that were dirty
at the start were committed there, not by me). Only `audit/` was created by A0;
`scripts/profiling/` is A2's.
**Method:** every claim below carries a `file:line` or a pasted command output.
Throwaway analysis scripts were written under the session scratchpad, never into
the repo.

---

## 1. The 14 contradictions

Verdict key: **SPEC** = code matches `00`–`08`; **ARCH** = code matches
`ARCHITECTURE.md`; **NEITHER** = code matches no document.

| # | Subject | What the code actually does | `file:line` | Verdict |
|---|---|---|---|---|
| 1 | Data source | Both source classes are live. `YFinanceSource` fetches with `auto_adjust=True`; `ZerodhaSource` fetches read-only historical bars. The **stores are provenance-tagged**: `data/ohlcv` rows carry `source='yfinance'`, `data/kite_ohlcv` rows carry `source='zerodha'`. But every training entrypoint hardcodes the yfinance-derived panel. | `sources/yfinance_source.py:101`; `sources/zerodha_source.py:552`; `scripts/train.py:33`; `scripts/walk_forward.py:59`; `scripts/paper_run.py:266` | **NEITHER** — Kite exists but is unreachable from training |
| 2 | Date range | Two panels exist. `data/panels`: train 2014-01-01→2021-12-31, val 2022-02-01→2022-12-30, test 2023-02-01→2024-12-30. `data/panels_kite`: train 2005-01-03→2023-12-29, val 2024-02-01→2024-12-31, test 2025-02-01→2026-09-04. **Only the first is reachable from `train.py`/`walk_forward.py`/`paper_run.py`.** | measured (§2.2); `scripts/train.py:33`, `scripts/walk_forward.py:59`, `scripts/paper_run.py:266` | **SPEC** in practice (2014/2022/2023 splits are what runs) |
| 3 | Price series | `data/panels` is **total-return** (dividend back-adjusted). `data/panels_kite` is **price-return**. Proven by ex-dividend drops present in one and absent in the other (§2.3). `adj_close` is a byte-identical copy of `close` in both, so the column carries no information. | `sources/yfinance_source.py:143-149`; `sources/zerodha_source.py:608`; §2.3 | **NEITHER** — spec says TR, ARCH says PR, and both files exist; the one that runs is TR |
| 4 | Universe size | `all_tickers()` = **645**, `active_tickers()` = **504**, 14 sectors. But **both panels contain 163 tickers**, of which only **143 are in `active_tickers()`**. `runner.py` builds the env on `active_tickers()`, so **361 of 504 observation columns are all-zero and permanently `is_tradeable=False`.** | `data/universe.py:284-292`; `training/runner.py:107`; `env/panel_env.py:439-440`; §2.6 | **NEITHER** — 143 effective names, matching no document |
| 5 | Survivorship | Confirmed broken, independently (§2.4). `market.universe_snapshots` = 0 rows. Only write site is `scripts/build_universe.py:60`, and it writes *today's* `all_tickers()` stamped with `cfg.data.start_date` — not point-in-time even if run. **No read site anywhere in `src/` or `scripts/`.** | `db/market_models.py:46-47`; `scripts/build_universe.py:60`; psql output §2.4 | **NEITHER** — the spec's non-negotiable is unimplemented |
| 6 | Reward | `LogReturn` (pass-through) minus `turnover_penalty × turnover`. `DifferentialSharpe` is implemented but never selected; `use_excess_returns` defaults false and no run has set it. | `configs/env/panel_daily.yaml:5`; `training/runner.py:141-142`; `env/panel_env.py:294-297`; `env/reward.py:18,65` | **ARCH** |
| 7 | STT | 0.1% on **both** legs of delivery. Confirmed, not rediscovered (commit `9019892`). | `env/costs.py:81-82` | **ARCH** — `03_environment.md:111` STALE |
| 8 | Brokerage | `brokerage_rate=0.0`, `cap=inf` for delivery. Confirmed. | `env/costs.py:79-80` | **ARCH** — `03_environment.md:110` STALE |
| 9 | DP charge | `_DP_CHARGE = 15.34`. Confirmed. | `env/costs.py:55,84` | **ARCH** — `03_environment.md:116` STALE |
| 10 | Vectorisation | `SyncVectorEnv`, 16 envs. | `training/runner.py:18,171` | **ARCH** — `03_environment.md:172` STALE |
| 11 | Rollout / entropy | `n_steps: 256`, `ent_coef: 0.0001`. | `configs/train/ppo_baseline.yaml:5,11` | **ARCH** — `05_training.md:31,37` STALE |
| 12 | Share rounding | **Both floor.** Env: `np.floor(...)` at `env/panel_env.py:247`. Paper broker: `int(floor(nav * weight / open_px))` at `broker/paper_broker.py:1034`, and its own docstring says "floored to whole shares" (`:13-15`). The divergence is *not* rounding — it is `min_trade_value`, which the broker applies (`:1038`) and the env does not implement at all. | as cited | **ARCH** — `06_paper_broker.md:158-159` ("nearest, not truncated") is STALE; row 12's premise is refuted |
| 13 | Target hardware | `get_device()` resolves cuda→mps→cpu. Every stored run logged `device = mps`. No CUDA device has ever been used. | `utils/seeding.py:10-25`; MLflow param `device=mps` on all 5 runs | **NEITHER** — 5090 gone, 4060 unused, only MPS has ever run |
| 14 | Sector holdout | `INACTIVE_SECTORS = {chemicals, consumer_durables, construction_materials, services, telecom}` — 5 of 14 sectors, 141 of 645 names. Provenance: introduced in commit `a5d8b79` (2026-09-04 16:51) with the stated reason of holding observation width at 504 for wall-clock cost. **Moot in practice**: the panels contain 163 names from the *old* 8-sector taxonomy, and 8 of them carry `sector_id == 0`. | `data/universe.py:273-282`; §2.6 | **ARCH** as written, **NEITHER** as executed |

### 1.1 Specs that remain stale after `9019892`

| Spec | Line | Stale claim | Code truth |
|---|---|---|---|
| `03_environment.md` | 110 | brokerage `min(20, 0.0003·V)` | `0.0` (`costs.py:79-80`) |
| `03_environment.md` | 111 | STT `0.001·V` **sell side only** | 0.1% **both** legs (`costs.py:81-82`) |
| `03_environment.md` | 113 | GST on (brokerage + exchange) | GST on (brokerage + SEBI + exchange) (`costs.py:199`) |
| `03_environment.md` | 116 | DP ₹15.93 | ₹15.34 (`costs.py:55`) |
| `03_environment.md` | 30, 129-135 | `reward_fn = DifferentialSharpe(window=60)` | `LogReturn` (`configs/env/panel_daily.yaml:5`) |
| `03_environment.md` | 172 | `AsyncVectorEnv` | `SyncVectorEnv` (`runner.py:171`) |
| `05_training.md` | 31, 37 | `rollout_length: 512`, `ent_coef: 0.01` | 256, 1e-4 |
| `06_paper_broker.md` | 158-159 | integer lots "nearest, not truncated" | `floor` (`paper_broker.py:1034`) |
| `02_data_pipeline.md` | 18, 149-151 | 130–160 tickers, 2014/2022/2023 splits | 163 in panel; 645/504 in `universe.py` |
| `02_data_pipeline.md` | 29 | universe "as it existed on each date" via `universe_snapshots` | table empty, no read site |
| `00_overview.md`, `01_setup*.md`, `04_models.md`, `05_training.md`, `07_roadmap.md` | 108 / 15,166,241 / 11,134,162 / 151 / 154,180 | RTX 5090, CUDA 12.8 | no CUDA device exists |

Cost-model check, reproduced (`uv run`, `ZerodhaEquityDeliveryCostModel`):

```
DELIVERY  buy 1L = 118.74  sell 1L = 119.08  round trip = 237.82
INTRADAY  buy 1L =  30.34  sell 1L =  52.34  round trip =  82.68
SPEC-MODEL (03_environment.md) round trip = 185.58  -> code is 28.2% higher
```

₹237.82 / ₹82.68 match `09_revamp_and_audit.md` exactly. The spec model charges
₹185.58 against the correct ₹237.82 — a **22.0% understatement**
(185.58 / 237.82 = 0.780). Rows 7–9 confirmed, not rediscovered.

---

## 2. The seven R0 questions

### 2.1 Q1 — Which source populated `data/panels/*.parquet`?

**yfinance.** Direct, not inferred: both OHLCV stores carry a `source` column,
and the panel prices match the yfinance store byte-for-byte.

```
data/ohlcv/year=2014/ticker=COALINDIA.NS.parquet   close 2014-01-01 = 106.041351  source = yfinance
data/kite_ohlcv/year=2014/ticker=COALINDIA.NS.parquet close 2014-01-01 = 272.05     source = zerodha

data/panels/train.parquet       COALINDIA 2014-01-01 close = 106.04135131835938
data/panels_kite/train.parquet  COALINDIA 2014-01-01 close = 272.05
```

`data/ohlcv` holds 405,134 rows / 163 tickers / 2014-01-01→2024-12-30, all
`source='yfinance'`. `data/panels` is built from it (`configs/data/default.yaml`
and `universe_v1.yaml` both set `parquet_root: data/ohlcv`).

### 2.2 Q2 — Shape, date range, per-ticker history

**Panel shapes and coverage (measured):**

| Panel | split | rows | dates | tickers | `is_tradeable` True | True fraction |
|---|---|---:|---:|---:|---:|---:|
| `data/panels` (yfinance) | train | 321,436 | 2014-01-01 → 2021-12-31 (1,972) | 163 | 276,005 | **0.8587** |
| | val | 37,164 | 2022-02-01 → 2022-12-30 (228) | 163 | 36,705 | 0.9876 |
| | test | 76,447 | 2023-02-01 → 2024-12-30 (469) | 163 | 76,007 | 0.9942 |
| `data/panels_kite` (Kite) | train | 767,893 | 2005-01-03 → 2023-12-29 (4,711) | 163 | 623,555 | **0.8120** |
| | val | 37,001 | 2024-02-01 → 2024-12-31 (227) | 163 | 36,774 | 0.9939 |
| | test | 64,222 | 2025-02-01 → 2026-09-04 (394) | 163 | 63,828 | 0.9939 |

**Per-ticker first/last non-null.** The panels are dense: every ticker has a row
on every date and `close`/`adj_close` are never NULL — non-tradeable rows are
**sentinel-zeroed**, not null (`data/alignment.py:20-25, 77-79`). 35,794 rows
(11.1%) of `data/panels` train and 134,306 (17.5%) of `panels_kite` train carry
`close == 0.0`. No zero-close row is ever marked tradeable, so the mask does
hold — but `ARCHITECTURE.md`'s "(mask, never zero-fill)" is inaccurate: it is
mask *and* zero-fill.

The meaningful "first date" is therefore the first `is_tradeable` date:

| Panel | median tradeable rows/ticker | median span | min span | tickers < 2y span | tickers < 504 tradeable rows | earliest first-tradeable |
|---|---:|---:|---:|---:|---:|---|
| `data/panels` train | 1,912 | 7.76 y | 0.13 y | **5** | **5** | 2014-03-28 (132 of 163 tickers) |
| `panels_kite` train | 4,651 | 18.75 y | 0.41 y | **2** | **2** | 2005-03-31 (99 of 163 tickers) |

Latest listings in `data/panels` train: DEVYANI 2021-11-12, ETERNAL 2021-10-20,
SHYAMMETL 2021-09-21, MAZDOCK 2021-01-06, MAXHEALTH 2020-11-14.

**Raw Kite store `data/kite_ohlcv` (9,438 files, 2,217,214 rows, 656 tickers
incl. `^NSEI`), 2005-01-03 → 2026-09-04:** median 4,076 bars/ticker, median span
16.48 y; **63 tickers with < 2 years of span, 64 with < 504 bars**; 234 tickers
start at 2005-01-03; latest first bars are 2025-11/12 IPOs (ICICIAMC 2025-12-19,
WAKEFIT 2025-12-15, AEQUS 2025-12-10 …).

**Per-feature statistics over `is_tradeable == True` rows only** — the numbers
the model's normaliser sees.

`data/panels/train.parquet` (n = 276,005):

| feature | mean | std | min | max | distinct |
|---|---:|---:|---:|---:|---:|
| log_return_1d | 0.000721 | 0.02348 | −1.0787 | 0.4580 | 273,420 |
| log_return_5d | 0.003600 | 0.05365 | −1.1406 | 1.0152 | 274,956 |
| log_return_20d | 0.014611 | 0.10964 | −1.3428 | 0.9680 | 275,475 |
| realized_vol_20d | 0.33292 | 0.16585 | 0.01168 | 4.3115 | 275,948 |
| realized_vol_60d | 0.34397 | 0.14231 | 0.06511 | 2.5187 | 275,953 |
| rsi_14 | 52.3125 | 12.3534 | 3.0431 | 98.7309 | 276,005 |
| macd | 5.8957 | 106.155 | −3,554.19 | 4,194.95 | 276,005 |
| macd_signal | 5.9510 | 100.220 | −3,083.80 | 3,645.75 | 276,005 |
| macd_hist | **−0.055250** | 30.980 | −1,220.07 | 1,046.89 | 276,005 |
| bbw_20 | 0.14786 | 0.10540 | 0.004056 | 2.0724 | 275,724 |
| z_close_20 | 0.15525 | 1.30167 | −4.1449 | 4.1742 | 275,997 |
| dollar_volume_20 | **1.2881e+09** | 2.4415e+09 | 33,230 | 6.2861e+10 | 276,004 |
| volume_z_20 | 0.000174 | 1.03205 | −3.3176 | 4.2481 | 276,005 |
| atr_14 | 36.261 | 140.113 | 0.13602 | 4,019.44 | 276,005 |
| **beta_nifty_60d** | **1.0** | **0.0** | **1.0** | **1.0** | **1** |

`data/panels_kite/train.parquet` (n = 623,555), differences worth naming:

| feature | mean | std | min | max | distinct |
|---|---:|---:|---:|---:|---:|
| beta_nifty_60d | 0.86781 | 0.51940 | −2.1422 | 5.6221 | 622,373 |
| dollar_volume_20 | 1.0564e+09 | 2.1729e+09 | **0.0** | 6.3981e+10 | 622,370 |
| realized_vol_20d | 0.35018 | 0.19170 | **0.0** | 7.5520 | 619,644 |
| realized_vol_60d | 0.36248 | 0.16918 | **0.0** | 4.2936 | 619,875 |
| rsi_14 | 51.972 | 12.658 | **0.0** | **100.0** | 622,373 |
| macd_hist | 0.052129 | 25.403 | −1,229.90 | 1,322.98 | 622,373 |

The Kite panel has degenerate minima on tradeable rows that the yfinance panel
does not: 1,183 tradeable rows with `dollar_volume_20 == 0`, and `rsi_14` pinned
at exactly 0 and 100. Also 3,491 tradeable rows (0.560%) carry `volume == 0` —
these are `alignment.py:22`'s 1-day forward-filled gaps, which stay
`is_tradeable=True` with a synthetic bar. `data/panels` has 298 such rows
(0.108%).

Low-price rows on tradeable data: `data/panels` 3.42% under ₹20;
`panels_kite` 5.96%.

### 2.3 Q3 — Price-return or total-return?

**`data/panels` is total-return. `data/panels_kite` is price-return.** Both are
in the repo; the one that training reaches is the total-return one.

Direct test, joining the two panels on `(date, ticker)` over their 2014–2021
overlap and looking for one-day down-steps in `close_kite / close_yfinance` —
the signature of a dividend that one feed adjusts away and the other does not:

`COALINDIA.NS`, 12 such days over 7.76 y, total 57.65% (**7.43%/yr**):

| date | close Kite | close yfinance | ratio | ratio step | Kite return | yfinance return |
|---|---:|---:|---:|---:|---:|---:|
| 2015-03-03 | 353.40 | 160.81 | 2.1976 | −5.26% | **−3.84%** | +1.50% |
| 2016-03-14 | 276.20 | 137.49 | 2.0089 | −8.59% | **−6.96%** | +1.78% |
| 2017-03-14 | 274.55 | 145.29 | 1.8897 | −5.94% | **−6.76%** | −0.87% |
| 2018-03-16 | 259.00 | 145.14 | 1.7844 | −5.58% | **−5.85%** | −0.29% |
| 2020-11-19 | 113.60 | 77.85 | 1.4592 | −5.87% | **−4.86%** | +1.08% |
| 2021-12-06 | 138.30 | 106.45 | 1.2992 | −5.62% | **−6.99%** | −1.46% |

`ONGC.NS`: 19 such days, 33.05% total (**4.26%/yr**) — 2014-12-16, 2015-03-24,
2015-11-10, 2016-08-31, 2016-11-03, 2017-02-07, 2017-11-03, 2018-03-13,
2019-02-28, 2021-11-22 …
`ITC.NS`: 11 such days, 24.19% total (**3.12%/yr**) — 2014-06-03, 2015-06-03,
2016-05-30, 2017-06-05, 2018-05-25, 2019-05-22, 2021-06-10 …

The Kite series drops on each date and the yfinance series does not. Those are
plausible dividend yields for those three names. Across all 154 tickers with
≥500 joined days, median implied yield 1.04%/yr, max 9.08%/yr (ADANIENT).

No in-repo corporate-actions dataset exists to cross-check the ex-dates
independently: `market.corporate_actions` has **0 rows**,
`ZerodhaSource.fetch_corporate_actions` raises `NotImplementedError`
(`sources/zerodha_source.py:327`), and the cached yfinance parquets under
`data/raw/yfinance/` carry only OHLCV columns. **UNVERIFIED** against an
external ex-date list; the cross-feed test above is the only in-repo evidence,
and it is unambiguous.

`adj_close` is not usable as a marker: `df.adj_close.equals(df.close)` is `True`
in both panels — yfinance aliases it (`sources/yfinance_source.py:143-149`) and
Kite mirrors it (`sources/zerodha_source.py:608`).

### 2.4 Q4 — Is the universe point-in-time? (verify the read path)

**No.** Confirmed independently of §2.1 of the plan.

```
$ psql -h localhost -U trader -d trader -tAc "select count(*) from market.universe_snapshots"
0
$ psql ... market.corporate_actions   -> 0
$ psql ... market.instruments         -> 0
$ psql ... market.trading_calendar    -> 0
$ psql ... market.dataset_versions    -> 6
```

Reference sites for `universe_snapshots` / `UniverseSnapshot` across
`src/ scripts/ tests/ alembic/`:

```
src/trader/db/market_models.py:46,47   class + __tablename__ (definition)
scripts/build_universe.py:48,60        the only WRITE site
tests/integration/test_db_roundtrip.py:19,65,66,76   round-trip test only
alembic/versions/0001_initial.py:48,226              DDL
```

**There is no read site in `src/` or `scripts/`.** The panel builder resolves the
universe by calling `all_tickers()` directly (`scripts/build_features.py:59`),
and the env builder calls `active_tickers()` (`training/runner.py:107`,
`scripts/paper_run.py:272`) — both return today's hand-maintained list.

Worse than "unwired": the single write site would not produce a point-in-time
table if it were run. `scripts/build_universe.py:26-27,60-64` writes **one** row —
`symbols = all_tickers()` (today's list) with
`effective_date = cfg.data.start_date` — i.e. the current membership
back-stamped onto the start of history.

The on-disk fallback `data/raw/universe_v1.parquet` (config key
`data.universe_snapshot_path`) holds 167 rows dated **Apr 24**, with 8 tickers
in a `'unknown'` sector, and is read by nothing.

### 2.5 Q5 — Which reward function does `reward.py` implement?

Three classes exist: `DifferentialSharpe` (`env/reward.py:18`), `LogReturn`
(`:65`), `ExcessLogReturn` (`:75`, functionally identical to `LogReturn` — the
env does the subtraction).

**What actually runs:** `configs/env/panel_daily.yaml:5` sets
`reward: log_return`; `training/runner.py:141-142` maps that through
`_REWARD_MAP` (`:52-56`) to `LogReturn()`; `env/panel_env.py:297` computes

```
reward = float(self._reward_fn(reward_input)) - self._turnover_penalty * turnover
```

with `reward_input = log_return` because `use_excess_returns` is false
(`configs/env/panel_daily.yaml:10`, `runner.py:143`). So the implemented reward is
**`log_return − 0.001 × turnover`**. `DifferentialSharpe` is written, unit-tested
and never selected. `ExcessLogReturn` has never been switched on — no stored run
carries an `env.*` parameter at all (§4).

### 2.6 Q6 — Actual ticker count and provenance of the 5-sector holdout

```
all_tickers()    = 645     (14 sectors, SECTOR_IDS 1..14)
active_tickers() = 504
INACTIVE_SECTORS = ['chemicals','construction_materials','consumer_durables','services','telecom']
sector sizes: banking 121, it 36, pharma 70, auto 48, oil_gas_power 42, defence 19,
              fmcg 45, metals 24, capital_goods 99, chemicals 45, consumer_durables 41,
              construction_materials 16, services 26, telecom 13
```

Provenance: `data/universe.py:262-282` — added in commit `a5d8b79`
(2026-09-04 16:51) to recover eight NIFTY-50 names (ASIANPAINT, TITAN,
BHARTIARTL, ULTRACEMCO, SHREECEM, GRASIM, ADANIPORTS, UPL) that the old
8-sector taxonomy excluded, then held out purely to keep observation width at
504 rather than 645.

**The known mismatch, quantified.** Both panels contain 163 tickers, of which:

```
in active_tickers():   143     ->  361 of the 504 obs columns are dead
in all_tickers():      153     ->  10 panel tickers are in no sector at all
                                   (DEVYANI, ETERNAL, HGINFRA, LTIM, MCDHOLDING,
                                    NIITLTD, NYKAA, PGHH, WELSPUNLIV, WESTLIFE)
```

`env/panel_env.py:437-440` skips any universe ticker absent from the panel
(`if sub.is_empty(): continue`), leaving `_stacked_features[:, ni, :]` at zero
and `_stk_mask[:, ni]` at False for all 361. They cost full forward/backward
compute and contribute nothing.

Both panels also still carry the **old 8-sector taxonomy plus `sector_id == 0`**:
`{0: 8, 1: 20, 2: 19, 3: 20, 4: 19, 5: 20, 6: 19, 7: 20, 8: 18}` tickers per
sector — the eight `sector_id == 0` names are exactly ADANIPORTS, ASIANPAINT,
BHARTIARTL, GRASIM, SHREECEM, TITAN, ULTRACEMCO, UPL. The phantom-sector fix
(`data/universe.py:295-320`) landed in the code; **the panels predate it and
still contain the phantom sector.**

**Why the panels are stale — timeline from file mtimes and commit timestamps:**

```
Apr 24 14:06   data/panels/{train,val,test}.parquet written   (yfinance, 163 tickers)
Sep  4 16:34   data/panels_kite/*.parquet written             (Kite, 163 tickers, old taxonomy)
Sep  4 16:44   data/kite_ohlcv rewritten with 656 tickers     (fetch_kite_data.py)
Sep  4 16:51   commit a5d8b79 expands universe.py to 645/504
```

`reports/kite_vs_yfinance.md:3` corroborates: it describes the Kite store as
"749,938 bars, **164 instruments**". The store now holds 2,217,214 bars across
656 tickers. **`data/panels_kite` was built from the 163-ticker store and was
never rebuilt after the universe expansion.**

**Panel SHA256 sidecars all verify** (`data/panels/*.sha256` and
`data/panels_kite/*.sha256` match `shasum -a 256` on all six files), so neither
panel has been tampered with since it was written — they are simply old.

### 2.7 Q7 — `git log` for `costs.py` and `features.py`

See §4 for the full run-provenance answer.

---

### 2.8 `beta_nifty_60d` — confirmed dead in `data/panels`

**CONFIRMED, independently.** Over all 276,005 tradeable rows of
`data/panels/train.parquet`:

```
beta_nifty_60d   mean=1.0  std=0.0  min=1.0  max=1.0  nunique=1  n_nan=0
```

Root cause, still present in code: `data/features.py:291-293`

```python
def _add_beta(df: pl.DataFrame, index_rets: pl.DataFrame | None) -> pl.DataFrame:
    if index_rets is None:
        return df.with_columns(pl.lit(1.0, dtype=pl.Float64).alias("beta_nifty_60d"))
```

`data/ohlcv` contains no `^NSEI` file (`find data/ohlcv -name '*NSEI*'` → empty),
so `_load_index_rets` returned `None` and every row got the literal 1.0. The
guard is a **silent** fallback — it does not warn, and `features.py` has not been
touched since `08d11fe` (2026-04-22).

The second half of the bug is confirmed too. `data/panels/train.feature_stats.json`
records `"beta_nifty_60d": [0.0, 1.0]`, because `data/feature_stats.py:43-45`
substitutes `(0.0, 1.0)` whenever `std < 1e-8`. A constant-1.0 feature therefore
normalises to `(1.0 − 0.0)/1.0 = 1.0` — a **fixed +1 bias** into every
convolution, not a zeroed channel.

In `data/panels_kite` the feature is alive: mean 0.8678, std 0.5194, range
[−2.1422, 5.6221], 622,373 distinct values. **The fix lives in the data
(`configs/data/kite_v1.yaml:32-33` `fetch_index: true` / `index_ticker: "^NSEI"`), not in the code** —
`features.py:293` will still silently emit 1.0 for any future panel built
without an index series.

### 2.9 The un-normalised feature in the smoke run

`Feature stats: mean range [-0.05525, 1.288e+09]` is the log line at
`training/runner.py:115-119`. It reports the range of the **per-feature means**,
so the two endpoints name two different features. Reproduced exactly:

```
$ compute_feature_stats(data/panels/train.parquet, FEATURE_COLS)
Feature stats: mean range [-0.05525, 1.288e+09]  std range [0.02348, 2.442e+09]
```

- Lower endpoint **−0.05525** = `macd_hist`, mean −0.0552497.
- Upper endpoint **1.288e+09** = **`dollar_volume_20`**, mean 1,288,102,831,
  std 2,441,530,517, range [33,230, 6.2861e+10], 276,004 distinct values —
  raw INR turnover, nine orders of magnitude above every return feature.

Distribution of `dollar_volume_20` over tradeable train rows: it is the only
feature whose std exceeds its own mean by a factor >1.8 and spans five orders of
magnitude within the tradeable set. In `panels_kite` it additionally takes the
value **0.0 on 1,183 tradeable rows**.

The feature *is* standardised at the model boundary — `runner.py:114,219-220`
passes `feat_mean`/`feat_std` into the encoder — so the panel value is
un-normalised but the network input is not. Two features remain problematic
after standardisation, though: `dollar_volume_20` is standardised against a
right-skewed distribution (no log transform anywhere in `features.py`), and
`beta_nifty_60d` in `data/panels` is standardised by `(0.0, 1.0)` as above.

Running this line reproduces the smoke run's exact string only for
`data/panels`. For `data/panels_kite` the same line reads
`mean range [0.0006636, 1.056e+09]`. **The smoke run used `data/panels`.**

---

## 3. Dead config

Every key in `configs/**/*.yaml` traced to a config-access site
(`cfg.<group>.<key>`, `.get("<key>")`, `cfg["<key>"]`) in `src/` and `scripts/`,
excluding tests.

### 3.1 Declared and never read

| Key | File | Value | What the code does instead |
|---|---|---|---|
| `env.transaction_cost_model` | `configs/env/panel_daily.yaml:4` | `zerodha_equity_delivery` | `runner.py:150-160` never passes `cost_model`; `panel_env.py:57` falls back to `ZerodhaEquityDeliveryCostModel()`. The value happens to match, so changing it silently does nothing. |
| `broker.type` | `broker/paper.yaml:2`, `broker/zerodha.yaml:2` | `paper` / `zerodha` | `scripts/paper_run.py:298` constructs `PaperBroker` unconditionally. `broker=zerodha` would still run the paper broker. |
| `broker.enabled` | `broker/zerodha.yaml:4` | `false` | Nothing reads it; `ZerodhaBroker` does not exist. |
| `model.encoder` | all 5 `configs/model/*.yaml` | `tcn` | `runner.py:177-197` hardcodes the TCN; `encoder: gru` would be silently ignored. |
| `model.graph.hidden_dim` | `gnn_v1.yaml:14`, `gnn_intra_only.yaml:14` | `64` | `runner.py:203-211` never reads it; the GNN uses `encoder.out_dim`. The config's own comment already says so. |
| `data.universe_version` | `data/universe_v1.yaml:8` | `1` | Nothing reads it. |
| `data.universe_snapshot_path` | `data/universe_v1.yaml:9` | `data/raw/universe_v1.parquet` | Nothing reads it. `scripts/build_universe.py:32` **hardcodes** the same path when writing, so the key can never redirect it. This is the survivorship dead end (§2.4). |

### 3.2 Read, but not where the doc implies — the `min_trade_value` class

| Key | Read at | Not read at | Consequence |
|---|---|---|---|
| `env.min_trade_value: 500` (`configs/env/panel_daily.yaml:12`) | `scripts/paper_run.py:279` → `PaperBrokerConfig` → `paper_broker.py:1038` | `env/panel_env.py` — the string does not appear in the file | Confirmed as the known case. The **backtest env has no minimum trade size**; the paper broker drops every delta under ₹500. This, not share rounding (§1 row 12 — both floor), is the structural difference between the two execution paths. |
| `data.panels_root` | `scripts/build_features.py:55` | `scripts/train.py:33`, `scripts/walk_forward.py:59`, `scripts/paper_run.py:266` — all hardcode `orig_cwd / "data" / "panels"` | `data=kite_v1` builds panels into `data/panels_kite` and **no consumer can read them**. `data/panels_kite` is write-only from the pipeline's point of view. |
| `data.train_end`, `val_start`, `val_end`, `test_start` | `scripts/build_features.py:125-128` via `_split_date(cfg, key, fallback)` (indirect string key) | anywhere else | Live, but only at panel-build time; the fallbacks at `build_features.py:29-32` are the 2021/2022/2023 boundaries. |
| `broker.initial_cash` | `scripts/paper_run.py:276` | training path uses `env.initial_cash` (`runner.py:156`) | Two independent capital settings; `paper.yaml` says 1,000,000 and `panel_daily.yaml` says 1,000,000, so they coincide by luck. |

### 3.3 The inverse — read with no declared key

| Read site | Key | Default used |
|---|---|---|
| `scripts/paper_run.py:274` | `env.max_weight_per_name` | `0.10` — **not declared in `configs/env/panel_daily.yaml`**. The 10% per-name cap that every document quotes exists only as a Python default (`panel_env.py:49`) and is not settable from the env config. |

### 3.4 Config stale against code

| Key | File | Value | Code truth |
|---|---|---|---|
| `model.graph.num_sectors` | `gnn_v1.yaml:13`, `gnn_intra_only.yaml:13` | `8` ("matches SECTOR_IDS in universe.py") | `SECTOR_IDS` now runs 1..14. `build_sector_edges` maps `sector_id` → node index `sid − 1`; on a panel built from the current 645/504 universe it emits sector-node indices up to 13 against `S = 8` sector nodes (`models/graph.py:217,323`). Verified directly: feeding sector ids `[0..8, 14]` produces edge indices with `max idx 13`. Latent only because the GNN path is not the default and the live panels top out at `sector_id == 8`. |
| `configs/config.yaml:5-12` | — | comment block correctly flags `mlp_regime` as "the *intended* architecture, NOT a measured winner" | Consistent with CLAUDE.md rule 2 — this one is fine, recorded for completeness. |

Everything else in `configs/**` traces to a live read site: all of
`walk.*` (`scripts/walk_forward.py:83-89`, `training/walk_forward.py:576`), all of
`train.*` (`runner.py:277-292`), the remaining `model.*` (`runner.py:177-197`,
`203-211`, `256`, `273`), `env.{lookback_days,initial_cash,episode_length,
turnover_penalty,reward,use_excess_returns}` (`runner.py:141-159`),
`broker.{initial_cash,settlement_days,slippage_model,slippage_pct}`
(`paper_run.py:276-281`), `data.{start_date,end_date,parquet_root,raw_cache,
universe,lookback_days,extreme_move_threshold,fetch_index,force_refetch,
index_ticker,mask_corporate_actions,max_days_per_request}`
(`scripts/fetch_data.py`, `scripts/fetch_kite_data.py:58-66`,
`scripts/build_features.py:54-124`), and `seed` (`runner.py:99`).

---

## 4. Run provenance

### 4.1 `git log`

```
$ git log --format='%h %ad %s' --date=short -- src/trader/env/costs.py
9019892 2026-09-04 Correct the Zerodha cost model, add capital-gains tax, add the paper broker
0cc881d 2026-04-26 Opus4.7 fixes
f98342b 2026-04-24 fixing a mistake of naming the directory to be env

$ git log --format='%h %ad %s' --date=short -- src/trader/data/features.py
08d11fe 2026-04-22 M3 Phase complete; Integration tests are performed.
```

**STT and DP were corrected in `9019892`, 2026-09-04 16:50:37 +0530**
(`costs.py` +206/−42 lines; `costs.py:81-82` STT both legs, `:55` DP 15.34,
`:79-80` brokerage 0). `tax.py` was added in the same commit.

**Beta was *not* fixed in `features.py`.** That file's last change is
`08d11fe`, 2026-04-22 — five months before the beta bug was found, and
`features.py:293` still contains the silent `pl.lit(1.0)` fallback. The fix
landed in **`a5d8b79`, 2026-09-04 16:51:16 +0530**, and it is a *data* fix:
`configs/data/kite_v1.yaml:32-33` adds `fetch_index: true` / `index_ticker: "^NSEI"`,
`scripts/fetch_kite_data.py` pulls NIFTY 50 (instrument token 256265), and
`scripts/build_features.py` passes the configured start into `_load_index_rets`.
`a5d8b79` touched 21 files (+3,686/−199) and did not touch `features.py`.

Both fixes are ancestors of `HEAD`; `git merge-base --is-ancestor 7c22fce 9019892`
→ YES, `git merge-base --is-ancestor b89f156 a5d8b79` → YES.

### 4.2 MLflow

**MLflow is up at `http://localhost:5555`** (backing store `mlruns/mlflow.db`,
836 KB, artifacts under `mlruns/artifacts/1/`). It is **not** empty:
2 experiments (`Default` id 0, empty; `trading_bot` id 1) and **5 runs**, all
named `mlp_regime_seed42`, all on 2026-09-04.

| run_id | status | start | `total_steps` | metrics | `mlflow.source.git.commit` | predates `9019892`? | predates `a5d8b79`? |
|---|---|---|---:|---:|---|---|---|
| `20c8f4157b3f4df1865d83f958b98081` | FINISHED | 12:15:20 | 8,192 | 37 | `b89f156` | **yes** | **yes** |
| `bf3dd4770fc84ec8b94186ad76ba20b8` | KILLED | 12:22:51 | 61,440 | 0 | `b89f156` | **yes** | **yes** |
| `f571addaa1a744b4967ff12732434081` | KILLED | 12:25:32 | 20,480 | 9 | `b89f156` | **yes** | **yes** |
| `c39b8f5085344d1b896f09b468ad7af7` | FINISHED | 13:35:16 | 4,096 | 212 | `b89f156` | **yes** | **yes** |
| `46ae9da6e61f4aea90af71fa023f1edd` | **RUNNING** | 15:28:01 | 500,000 | 9 | `7c22fce` | **yes** | **yes** |

**All five stored runs predate both fix commits.** `9019892` and `a5d8b79` were
committed at 16:50:37 and 16:51:16; the latest run started at 15:28:01. Every
run therefore carries the 22%-understated cost model, zero capital-gains tax,
and `beta_nifty_60d` constant 1.0.

Every run also logs `n_tickers = 163` and `device = mps`, and the longest is
500,000 steps — a quarter of the 2M-step budget, and it is still marked RUNNING
with a mean Sharpe of **−0.963** and mean CAGR **−10.2%**. The most-complete run
(`c39b8f5`, 212 metrics) finished 4,096 steps in 3.5 minutes with mean Sharpe
**−1.259**, mean CAGR **−16.1%**, mean max drawdown **−26.9%**.

Corroborating the plan's pre-verification: `data/walks/` **does not exist**
(`ls: "data/walks": No such file or directory`) — no walk-forward summary has
ever been written. `checkpoints/mlp_regime_seed42/` exists and is **empty**.

**Run provenance is not recoverable from MLflow.** All 5 runs log exactly 35
params — `model.*`, `train.*`, `seed`, `device`, `n_params`, `n_tickers`. There is
**no `env.*` parameter, no `data.*` parameter, and no panel path or panel
SHA256** on any run. Which panel a run saw can only be inferred from
`n_tickers = 163` and the run timestamp. `a5d8b79`'s commit message claims
"env.* params are now logged to MLflow", and `runner.py:327-336` does log
`env.reward_fn_resolved` and `env.use_excess_returns_resolved` — but every stored
run predates that code.

### 4.3 Other run artefacts

`ledger.*` in Postgres is **not** empty — the paper broker has been run against
a panel: `strategy_runs` 3, `orders` 1,900, `fills` 1,900, `positions` 187,
`portfolio_snapshots` 501, `pnl_daily` 501, `lots` 1,020. Since
`scripts/paper_run.py:266` hardcodes `data/panels`, those fills were priced off
the stale, dividend-adjusted, dead-beta panel.

All eleven Hydra run logs under `outputs/2026-09-04/` are **0 bytes** —
`train.log` ×5, `paper_run.log` ×4, `fetch_kite_data.log` ×2. Loguru is not
attached to Hydra's file sink, so no run has produced a durable text log. The
11% NAV-divergence figure quoted in `ARCHITECTURE.md:147` and `CLAUDE.md:73`
has **no regenerating artefact in the repo — UNVERIFIED.** To verify it I would
need the paired backtest NAV series that produced it; only the paper-broker side
survives, in `ledger.portfolio_snapshots`.

---

## 5. Two further contradictions not in the §2 list of 14

Recorded because CLAUDE.md rule 4 says to assume more exist. Not fixed.

**15. `MomentumTopK` does not rank on momentum.** The docstring at
`env/baselines.py:85-88` says "Top K tickers by trailing 20-day log return …
Uses the precomputed `log_return_20d` feature". The code at
`env/baselines.py:110` reads `features[-1, :, 0]`, and `FEATURE_COLS[0]` is
`log_return_1d` (`data/features.py:16-17`), not `log_return_20d`
(`FEATURE_COLS[2]`). The comment at `:105-109` admits the fallback. So the
momentum baseline ranks on **yesterday's one-day return** — a one-day
cross-sectional momentum/reversal signal, not 20-day momentum. Any statement
that "the agent failed to beat momentum top-K" is a statement about the wrong
baseline. This is orthogonal to the K=5 vs K=20 issue in `09_revamp_and_audit.md`
§4.3.

**16. `EqualWeightRebalanced` rebalances daily, not monthly.**
`09_revamp_and_audit.md:158-160` states "The baselines (`EqualWeightRebalanced`,
`MomentumTopK`) already rebalance monthly, so the agent is compared against
baselines paying a fraction of its tax bill." `env/baselines.py:57-62` returns
fresh equal-weight logits on **every** `act()` call with no rebalance counter —
it is a daily rebalance and pays the same turnover cost as the agent.
`MomentumTopK` does rebalance every 21 steps (`env/baselines.py:91,103`), so
half the claim holds and half does not. The stated argument for §4.1's
monthly-rebalance decision rests on the half that does not.

---

## Three findings that most change what should be done next

1. **Everything that can currently be run is wired to the stale panel, and the
   good panel is unreachable.** `scripts/train.py:33`, `scripts/walk_forward.py:59`
   and `scripts/paper_run.py:266` all hardcode `data/panels` — the April yfinance
   panel with `beta_nifty_60d` constant 1.0 across all 276,005 tradeable rows and
   a `(0.0, 1.0)` normalisation that turns that dead channel into a fixed +1 bias.
   `data.panels_root` is read only by `build_features.py:55`, so `data/panels_kite`
   is write-only. The smoke run at 15:28 today reproduced
   `mean range [-0.05525, 1.288e+09]` exactly, which is the fingerprint of
   `data/panels`. R1 is not "pick a source" — it is "connect the source you
   already picked", and it is a three-line change away from being testable.

2. **The universe mismatch is worse than 504-vs-163: it is 143.** `runner.py:107`
   builds the observation on `active_tickers()` = 504, but only **143** of those
   names exist in either panel, so **361 of 504 columns are permanently zero and
   non-tradeable** (`env/panel_env.py:439-440`) while costing full forward and
   backward compute. Both panels still carry the pre-`a5d8b79` 8-sector taxonomy
   including 8 tickers at the phantom `sector_id == 0`. `data/kite_ohlcv` has held
   656 tickers since 16:44 today; `data/panels_kite` was written at 16:34 from the
   163-ticker store. Every compute estimate in `ARCHITECTURE.md` §5 and every
   FLOP figure in `09_revamp_and_audit.md` §5 R3 is priced at 504 names against a
   model that is really trading 143 — A2's profiling numbers will be
   uninterpretable until the panel is rebuilt.

3. **The baseline that is supposed to be the bar is measuring the wrong thing,
   and the bar itself is undefined.** `MomentumTopK` ranks on `log_return_1d`,
   not `log_return_20d` (`env/baselines.py:110` vs `data/features.py:16-17`), so
   the "momentum baseline" the agent failed to clear is a one-day signal.
   `EqualWeightRebalanced` rebalances daily, not monthly (`env/baselines.py:57-62`),
   which removes the turnover-asymmetry argument underpinning the §4.1
   monthly-rebalance decision. Meanwhile the tax model exists
   (`env/tax.py`, added `9019892`) and is imported by nothing in `env/` or
   `training/`. R2's "5 baselines × 3 frequencies × 4 benchmarks" table cannot be
   trusted until the baselines are fixed first — and fixing them is cheaper than
   any of the compute work in R3.
