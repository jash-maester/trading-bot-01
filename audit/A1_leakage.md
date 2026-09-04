# A1 — Leakage and correctness audit

**Unit:** A1 of the audit protocol in `09_revamp_and_audit.md` §6.
**Run:** 2026-09-04, on `/Users/jash/storage/trading-bot-01`, HEAD `429470f`.
**Scope:** read-only. Nothing was fixed. One short walk-forward run was executed
for §5 (writes to `data/walks/`, `outputs/`, MLflow only).

Every claim below carries a `file:line`, a traced call graph, or pasted output.
The throwaway measurement scripts named in each section live under
`/private/tmp/claude-501/-Users-jash-storage-trading-bot-01/bf63e648-70c7-4efd-9a78-fbe8fff88317/scratchpad/`
and are **session-scoped — they will not survive**; the A1 brief forbade
creating any file but this report, so each script's full output is pasted here
rather than referenced. Anything marked **UNVERIFIED** was not measured.

---

## 1. `next_day_returns` containment

**Verdict: `HANDOFF.md` §7's claim holds. The containment mechanism is weaker
than the document implies — the obs dict *is* passed wholesale into every
forward method; nothing but convention stops one from indexing the key.**

### 1.1 Producers

| Site | What |
|---|---|
| `src/trader/env/panel_env.py:175` | declared in `observation_space` as `Box(-inf, inf, (N,), float32)` |
| `src/trader/env/panel_env.py:372` | `next_day_returns = self._stk_log_ret[day_idx].astype(np.float32)` |
| `src/trader/env/panel_env.py:386` | emitted in the obs dict, unconditionally |

`_stk_log_ret` is `log_return_1d` stacked to `[T, N]` (`panel_env.py:134`).
The obs feature window is `self._stacked_features[day_idx - lookback : day_idx]`
(`panel_env.py:343`), i.e. rows through `day_idx - 1`. So `_stk_log_ret[day_idx]`
is strictly one step ahead of everything else in the observation. It is a
genuine label.

### 1.2 Complete static enumeration of obs-key reads in `src/` and `scripts/`

`grep -rn "obs\[\|obs\.get\|obs\.items()\|obs\.keys()\|\*\*obs" src scripts`
returns exactly these consumer sites:

| File:line | Keys read |
|---|---|
| `models/actor_critic.py:157,159,160,161` | `features`, `portfolio`, `t_frac`, `sector_ids` |
| `models/actor_critic.py:170,171,172,184,194` | `recent_return_1d`, `recent_vol_20d`, `nav_log_progress`, `regime`, `mask` |
| `models/graph.py:440,441,442,443,445,455,456,457` | the same eight keys; no `regime` |
| `training/ppo.py:387,388` | `next_day_returns`, `mask` — **the only `next_day_returns` read** |
| `training/ppo.py:92,228` / `_batch_obs:106-113` | whole-dict copy/stack, no per-key logic |
| `training/runner.py:424,431,475` | `nav`; whole-dict tensorisation |
| `training/walk_forward.py:259,345,347,512,514` | `nav`, `features`, `sector_ids`, `mask` |
| `env/baselines.py:61,78,102,104,120,150` | `mask`, `features` |
| `scripts/paper_run.py:218,317` | whole-dict tensorisation; `mask` |

No forward path names `next_day_returns`. `models/heads.py` and
`models/encoders.py` take tensors only — neither module accepts a dict, so
neither can index the key by construction.

### 1.3 Call graph to the one consumer

```
PanelTradingEnv._build_obs  (panel_env.py:338)
  └─ obs["next_day_returns"]                                      panel_env.py:386
       └─ PPOTrainer.train rollout  ppo.py:228   obs_buf.append({k: v.cpu().numpy() ...})
            └─ _batch_obs           ppo.py:292   flattens every key to [T*E, ...] on CPU
                 └─ mb_obs          ppo.py:334   {k: v[mb_inds].to(device) for k, v in obs_batch.items()}
                      ├─ model.get_action_value_and_aux(mb_obs, mb_act)   ppo.py:346  ← whole dict
                      ├─ model.get_action_and_value(mb_obs, mb_act)       ppo.py:350  ← whole dict
                      └─ aux_target = mb_obs["next_day_returns"].float()  ppo.py:387  ← the label read
                           └─ aux_return_loss(pred, target, mask)         heads.py:136
```

`aux_active` (`ppo.py:321-325`) additionally requires `aux_return_loss_coef > 0`
**and** `model.aux_return_head is not None`. With the shipped defaults
(`configs/train/ppo_baseline.yaml` `aux_return_loss_coef: 0.0`) the branch is
dead and `next_day_returns` is carried through the whole rollout buffer purely
as ballast.

### 1.4 The obs dict *is* passed wholesale

Contrary to the phrasing of the audit brief, the full dict — `next_day_returns`
included — is handed to forward methods at four sites:

- `ppo.py:346` and `ppo.py:350` (`mb_obs`, contains every key)
- `runner.py:429-433` (`obs_t`, built from `obs.items()`)
- `walk_forward.py:517-521` (`obs_t`, built from `shuffled.items()`)
- `paper_run.py:218` (`batch`, built from `obs.items()`)

Containment therefore rests entirely on `ActorCritic._forward_shared`
(`actor_critic.py:145-210`) and `GNNActorCritic.forward` (`graph.py:435-474`)
naming their inputs explicitly. There is no structural barrier — no filtered
view, no assertion, no test that the key is absent from the model's input.

### 1.5 Empirical confirmation

`scratchpad/keytrace.py` wraps the obs in a `dict` subclass that records every
`__getitem__`/`get`, and runs every public forward entry point:

```
ActorCritic(all flags on)    forward                    read=['features','mask','nav_log_progress','portfolio','recent_return_1d','recent_vol_20d','regime','sector_ids','t_frac']
                                                        NOT READ=['cash','nav','next_day_returns']
ActorCritic(all flags on)    get_action_and_value       NOT READ=['cash','nav','next_day_returns']
ActorCritic(all flags on)    get_action_value_and_aux   NOT READ=['cash','nav','next_day_returns']
ActorCritic(all flags on)    get_value                  NOT READ=['cash','nav','next_day_returns']
ActorCritic(baseline)        forward/get_action_and_value/get_value   NOT READ=['cash','nav','next_day_returns','regime']
GNNActorCritic               forward/get_action_and_value/get_value   NOT READ=['cash','nav','next_day_returns','regime']

functional: ActorCritic(all)   max|Δ output| when next_day_returns is scaled 1e6 and shifted: 0.000e+00
functional: GNN                max|Δ output| when next_day_returns is scaled 1e6 and shifted: 0.000e+00
```

Both models are bit-identical under a `×1e6 + 42` perturbation of the label.

### 1.6 Related observation

`cash` and `nav` are also emitted (`panel_env.py:384-385`) and never read by any
model. `nav` is read only for metrics (`runner.py:424`). Three of twelve obs keys
are transport, not input.

---

## 2. Normalisation statistics

**Verdict: train-split-only and per-window, as claimed, for both normalisers.
Two real but immaterial calibration defects, and one silent mis-normalisation
path in `paper_run.py`.**

### 2.1 The path is correct

| Step | Site |
|---|---|
| Window panels materialised from the full panel | `walk_forward.py:584` → `materialise_window` (`:140`) |
| `train_one_run` receives the three window paths | `walk_forward.py:605-609` (`train_panel=win_paths["train"]`, `val_panel=…["val"]`, `test_panel=…["test"]`) |
| Feature stats | `runner.py:111` `compute_feature_stats(train_panel, FEATURE_COLS)` |
| Regime stats | `runner.py:125` `compute_regime_stats(train_panel)` |
| Saved per window × seed | `walk_forward.py:604` `stats_path = win_dir / f"feature_stats_seed{seed}.json"`, passed at `:623` |
| Frozen into the model as buffers | `encoders.py:187-214` (`FeatureNormalizer`), `encoders.py:71-105` (`RegimeNormalizer`) — no running update anywhere |

`compute_feature_stats` (`feature_stats.py:36`) and `compute_regime_stats`
(`regime_features.py:197`) each open exactly one parquet: the path they are
given. `val_panel` / `test_panel` are used only by `_evaluate_split`
(`runner.py:359, 371`) and never reach either statistic. `scripts/train.py`
passes no `test_panel` at all (`scripts/train.py:34-52`), so the single-window
path cannot touch test either.

The regime-stats file path is derived by the same string rule in the writer
(`runner.py:127-130`) and the reader (`walk_forward.py:363-366`):
`feature_stats_seed42.json` → `regime_stats_seed42.json`. Consistent.

### 2.2 Defect A — stats are computed over a different cross-section than the env sees

`compute_regime_stats` builds its `[T, N]` arrays from
`tickers = df["ticker"].unique()` (`regime_features.py:204`) — **163 panel
tickers**. `PanelTradingEnv` builds its arrays over `active_tickers()`
(`runner.py:107`, `panel_env.py:67`) — **504 slots, of which 143 overlap the
panel**. The equal-weight benchmark and breadth denominators therefore differ
between the statistic and the runtime feature.

Measured (`scratchpad/regime_probe.py`, on `data/panels/train.parquet` against
the stored `data/panels/train.regime_stats.json`):

| feature | stats mean | stats std | env mean | env std | z-mean | z-std |
|---|---:|---:|---:|---:|---:|---:|
| `mkt_vol_20d` | 0.150061 | 0.075668 | 0.150671 | 0.075367 | 0.008 | 0.996 |
| `mkt_breadth_20d` | 0.562405 | 0.210083 | 0.560154 | 0.209939 | −0.011 | 0.999 |
| `mkt_dispersion_20d` | 0.019699 | 0.003995 | 0.019447 | 0.004114 | −0.063 | 1.030 |
| `mkt_trend_20d` | 0.014356 | 0.059427 | 0.013759 | 0.059214 | −0.010 | 0.996 |
| `mkt_acceleration` | −0.029145 | 0.085182 | −0.028037 | 0.084913 | 0.013 | 0.997 |
| `mkt_vol_of_vol_60d` | 0.041440 | 0.035760 | 0.040847 | 0.035618 | −0.017 | 0.996 |

`z-mean`/`z-std` are what `RegimeNormalizer` actually outputs at runtime;
perfect calibration is 0.000 / 1.000. Worst case 0.063 σ off-centre and 3 %
off-scale. Real, immaterial. `compute_feature_stats` has the same structural
mismatch (163 vs 143 tickers) and was not separately quantified — **UNVERIFIED**.

### 2.3 Defect B — `paper_run.py` ignores per-window stats

`scripts/paper_run.py:149-158` always loads `panels_root/train.feature_stats.json`
and `panels_root/train.regime_stats.json` — the *global* split's stats — no matter
which checkpoint is passed. A walk-forward checkpoint's stats live at
`data/walks/W<n>/feature_stats_seed<s>.json`. Loading such a checkpoint into
`paper_run` silently applies the wrong z-score to every feature. Not a
train/val leak; a silent correctness hole on the live path.

---

## 3. Purge gap — confirmed insufficient, quantified

**Verdict: confirmed independently and larger than the brief states. Every
boundary in every window is short. Minimum purge that closes the hard-window
overlap is 3 months (59–63 trading days); the EWM features are never fully
closed by any finite purge, but their residual is second-order.**

### 3.1 Why the overlap exists at all

Features are computed on the **full continuous panel** and only then split:
`scripts/build_features.py:90` `panel = compute_features(panel, …)` precedes
`:134-136` `_write_split(panel, …, train/val/test)`. `walk_forward.py` then
re-concatenates the three splits and re-slices by date
(`scripts/walk_forward.py:73-75`, `walk_forward.py:140`). Nothing recomputes
a feature per segment. So every rolling window — `realized_vol_60d`
(`features.py:147-152`, `rolling_std(window_size=60)`), `beta_nifty_60d`
(`features.py:317-327`, `rolling_mean(window_size=60)`) — spans the boundary
freely.

`compute_windows` (`walk_forward.py:72-121`) computes the gap in **calendar
months** (`_add_months`, `:59-70`), never in trading days, and never against the
feature-window length.

### 3.2 Confirmation and quantification — walk-forward windows

Trading-day counts taken from the true NSE calendar rebuilt from
`data/ohlcv/year=*/ticker=*.parquet` (2,710 sessions, 2014-01-01 → 2024-12-30),
not from the split panels (which have holes — see §7.3).
Script: `scratchpad/purge2.py`.

`compute_windows(data_start=2014-01-01, data_end=2024-12-30, train_years=5, val_months=12, test_months=12, purge_months=1, n_windows=4, step_months=12)`

| win | train_end | val_start | cal days | trading days | contaminated val rows | val_end | test_start | cal days | trading days | contaminated test rows |
|---|---|---|---:|---:|---:|---|---|---:|---:|---:|
| W1 | 2018-12-31 | 2019-02-01 | 32 | 23 | 36 | 2020-01-31 | 2020-03-01 | 30 | 19 | 40 |
| W2 | 2019-12-31 | 2020-02-01 | 32 | 23 | 36 | 2021-01-31 | 2021-03-01 | 29 | 20 | 39 |
| W3 | 2020-12-31 | 2021-02-01 | 32 | 20 | 39 | 2022-01-31 | 2022-03-01 | 29 | 20 | 39 |
| W4 | 2021-12-31 | 2022-02-01 | 32 | 20 | 39 | 2023-01-31 | 2023-03-01 | 29 | 20 | 39 |

"contaminated rows" = `max(0, 59 − gap)` = the number of leading rows in the
later segment whose 60-day rolling window still reaches into the earlier
segment. **8 of 8 boundaries are short.** The brief's 22-trading-day figure is
in range; the measured values are 19–23, and the val→test boundary is
consistently *worse* than train→val (19–20 days) because February is a shorter
month than January.

### 3.3 Fixed splits in `configs/data/*.yaml`

| dataset | boundary | dates | cal days | trading days | contaminated rows |
|---|---|---|---:|---:|---:|
| `data/panels` (build_features defaults, `build_features.py:29-32`) | train→val | 2021-12-31 → 2022-02-01 | 32 | 20 | 39 |
| `data/panels` | val→test | 2022-12-31 → 2023-02-01 | 32 | 21 | 38 |
| `configs/data/kite_v1.yaml:45-48` | train→val | 2023-12-31 → 2024-02-01 | 32 | **22** | **37** |
| `configs/data/kite_v1.yaml:45-48` | val→test | 2024-12-31 → 2025-02-01 | 32 | **23** | **36** |

**Yes — the new Kite split has the same defect.** `kite_v1.yaml:42-44` asserts
*"A 1-month purge is cut from the START of each later segment … so no feature
window straddles a boundary."* That statement is false: a 1-month purge is
20–23 trading days against a 60-trading-day window.

### 3.4 Minimum purge that eliminates the hard-window overlap

`scratchpad/purge2.py`, all four windows, both boundaries:

| `purge_months` | windows produced | trading-day gap min | max | every boundary ≥ 59? |
|---:|---:|---:|---:|:---:|
| 1 | 4 | 19 | 23 | no |
| 2 | 4 | 38 | 42 | no |
| **3** | **4** | **59** | **63** | **yes** |
| 4 | 4 | 79 | 86 | yes |
| 5 | 4 | 100 | 109 | yes |
| 6 | 3 | 120 | 129 | yes |

**3 months is the minimum, and it clears by exactly zero days on W1.** 4 months
is the first value with margin. At 6 months the fourth window no longer fits the
2014–2024 panel.

### 3.5 The residual that no purge removes

`rsi_14`, `atr_14`, `macd`, `macd_signal`, `macd_hist` are exponential
(`features.py:170-177, 193-210, 282-287`), so their support is unbounded.
Analytic residual weight of pre-purge data, `w(G) = (1−α)^G`
(`scratchpad/ewm.py`):

| feature | G=22 | G=40 | G=59 | G=80 | G=120 | half-life |
|---|---:|---:|---:|---:|---:|---:|
| `rsi_14`, `atr_14` (α=1/14) | 1.96e-01 | 5.16e-02 | 1.26e-02 | 2.66e-03 | 1.37e-04 | 9.4 d |
| `macd` ema26 (α=2/27) | 1.84e-01 | 4.60e-02 | 1.07e-02 | 2.12e-03 | 9.75e-05 | 9.0 d |
| `macd` ema12 (α=2/13) | 2.53e-02 | 1.25e-03 | 5.24e-05 | 1.57e-06 | 1.97e-09 | 4.1 d |
| `macd_signal` ema9 of `macd` | 7.38e-03 | 1.33e-04 | 1.92e-06 | 1.77e-08 | 2.35e-12 | 3.1 d |

Measured directly (`scratchpad/leak_measure.py`, `scratchpad/leak2.py`): the
whole feature set was recomputed twice on 10 large-cap tickers — once on the
full 2014→ history, once on a history that **begins at `val_start` 2022-02-01**
— and compared row by row, in units of the feature's own val-period σ, over
rows tradeable in both:

| feature | k=60 | k=70 | k=80 | k=100 | k=120 | k=150 | k=200 | first k with rel. diff < 1e-6 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `rsi_14` | 1.60e-02 | 6.19e-03 | 1.25e-03 | 4.45e-04 | 1.50e-04 | 8.71e-06 | 4.49e-07 | 200 |
| `macd` | 1.91e-02 | 6.29e-03 | 3.06e-03 | 6.30e-04 | 1.18e-04 | 1.34e-05 | 4.75e-07 | 216 |
| `macd_signal` | 1.93e-02 | 1.14e-02 | 4.85e-03 | 1.05e-03 | 1.82e-04 | 2.10e-05 | 7.10e-07 | 221 |
| `macd_hist` | 1.36e-02 | 9.93e-03 | 4.81e-03 | 1.20e-03 | 1.77e-04 | 2.07e-05 | 6.17e-07 | 224 |
| `atr_14` | 5.72e-04 | 2.36e-04 | 6.89e-05 | 3.42e-05 | 1.04e-05 | 1.13e-06 | 3.32e-08 | 186 |
| `realized_vol_60d` | 6.3e-15 | — | — | — | — | — | — | 0 (hard window) |
| `beta_nifty_60d` | 0 | — | — | — | — | — | — | 0 (dead, constant 1.0) |

`k` is the trading-day index inside the val panel.

The same experiment gives the hard-window number directly: **600 of 7,180
tradeable val rows (8.4 %, = the first 60 trading days × 10 tickers) are not
computable at all without pre-val history** — `is_tradeable` would be False for
them, because `realized_vol_60d` and the 20-day windows are null. Last such
date 2022-04-29.

**Summary of §3.** After a 3-month purge the hard 60-day windows are clean and
the EWM residual is ≈1 % of a feature σ, falling below 0.1 % by ~110 trading
days (≈6 months). "The minimum purge that eliminates the overlap" is 3 months
for the 60-day windows; there is no finite purge for the EWMs, and the honest
statement is that a 3-month purge reduces EWM contamination to ~1 % of a σ and
a 6-month purge to ~0.01 %.

### 3.6 Where the contamination actually lands in evaluation

Because evaluation always begins at panel index 60 (see §7.2), the *first*
evaluated val observation's feature window is rows 0–59 — precisely the rows
that cannot be computed without train data. Observations at `day_idx` 60…95
each contain at least one contaminated row: **36 of the 182 evaluated steps
(19.8 %) on W1/val**.

### 3.7 The test locks the defect in

`tests/unit/test_walk_forward.py:56-72` (`test_compute_windows_purge_gaps`)
asserts `28 <= gap_train_val <= 32` — a *calendar*-month gap. It never compares
the gap to the longest feature window, so it passes on exactly the configuration
that leaks.

---

## 4. Regime-feature causality

**Verdict: the return path is strictly causal and the test that claims so does
test it. But the *mask* path is not covered, and `mkt_breadth_20d[t]` is a
function of `tradeable_mask[t]` — day-`t` information. The module docstring's
blanket claim is inaccurate.**

### 4.1 Formula audit

All three rolling helpers use a left-shifted window
(`regime_features.py:96-130`): `out[w:] = f(x[t-w : t])`, i.e. indices `t−w …
t−1`, day `t` excluded.

| output | line | window | verdict |
|---|---|---|---|
| `mkt_vol_20d` | :133 | `std(bench[t−20:t])·√252` | causal |
| `mkt_breadth_20d` | :140-153 | `cumret_20[t] = Σ rets[t−20:t]` **but** `& tradeable_mask[t]` and `/ tradeable_mask[t].sum()` | **returns causal, mask is day-`t`** |
| `mkt_dispersion_20d` | :159 | `mean(csd_std_today[t−20:t])` | causal (`csd_std_today[t]` uses day `t`, but is shifted out) |
| `mkt_trend_20d` | :162 | `Σ bench[t−20:t]` | causal |
| `mkt_acceleration` | :165-166 | `trend20 − trend60`, both shifted | causal |
| `mkt_vol_of_vol_60d` | :172 | `std(mkt_vol[t−60:t])`, and `mkt_vol[s]` ends at `s−1` | causal |

### 4.2 Do the tests test what they claim?

`tests/unit/test_regime_features.py:49-70`, `test_regime_strictly_backward_looking`:
perturbs `log_ret[80] += 5.0`, asserts `base[:81] == pert[:81]` and
`base[81:] != pert[81:]`. Re-run independently (`scratchpad/regime_probe.py`):

```
perturbing log_return[80] only -> first changed row: 81   (test asserts >80) OK
```

The test is honest about the returns argument.

It never perturbs `tradeable_mask`, and `compute_regime_features` takes the mask
as a second, equally load-bearing argument. Perturbing only that:

```
perturbing tradeable_mask[80] only -> regime rows changed: [80, 81, ..., 119] (n=40)
row 80 changed? True   max |delta| at row 80 = 0.125000
   mkt_breadth_20d: 0.375000 -> 0.500000  (delta 0.125000)
```

So `regime[80]` **does** depend on day-80 information, through
`regime_features.py:151` (`pos = (cumret_20 > 0.0) & tradeable_mask`) and `:153`
(`breadth = n_pos / tradeable_mask.sum(axis=1)`).

**Severity: low, but the documentation is wrong.** `is_tradeable[t]` is already
in the observation (`panel_env.py:376`) and is what gates the action, so the
policy has it anyway; this adds no new future information. But
`regime_features.py:24-26` and `HANDOFF.md` §3.1 both state the invariant as
absolute, and no test covers the second argument. The line comment at
`regime_features.py:155` (`# breadth above is "as of t" using rets up to t-1 ✓`)
is a self-assessment, not a test.

### 4.3 Second-order: warm-up rows are live during evaluation

The env computes the regime tensor from the **sliced** panel
(`panel_env.py:145-147`), so a val/test panel restarts the warm-up. Rows `t<20`
have `mkt_vol_20d = 0`; rows `t<60` have `mkt_trend_60 = 0`, so
`mkt_acceleration = mkt_trend_20`; and `mkt_vol_of_vol_60d[t] = std(mkt_vol[t−60:t])`
is only free of zero-padded inputs from `t ≥ 80`. Evaluation starts at
`day_idx = 60` (§7.2), so the first ~20 evaluated steps of every val/test
episode see a regime vector with a structurally degenerate
`mkt_vol_of_vol_60d`, z-scored against train-panel statistics that had no such
degeneracy. Not leakage; a distribution shift the FiLM net was never trained on.

---

## 5. Shuffled-ticker leak check

### 5.1 Has it ever been run? — No.

State of `mlruns/mlflow.db` **before** §5.2's run (queried directly with
`sqlite3` over the `experiments`, `runs`, `latest_metrics` and `tags` tables):

| Evidence | Result |
|---|---|
| `mlruns/mlflow.db` experiments | only `Default` (0) and `trading_bot` (1) — **no `walk_forward_*` experiment existed** |
| runs in `trading_bot` | 5, all `mlp_regime_seed42`, 2026-09-04 06:45 → 09:58 (2 FINISHED, 2 KILLED, 1 RUNNING) |
| distinct MLflow metric keys | `val_*` and `qs/val*` only — **no `test_*` metric has ever been logged** |
| distinct MLflow tags | `mlflow.runName`, `mlflow.source.*`, `mlflow.user` — no `window`, no `seed` |
| `data/walks/` | empty before this audit |
| `configs/walk/default.yaml:20` | `shuffle_check: false` |

`run_walk_forward` had never executed. Since the shuffled arm is produced only
inside it (`walk_forward.py:627-637`), the check had never run.

### 5.2 It was run for this audit

Command (writes only to `data/walks/`, `outputs/`, MLflow):

```
uv run python scripts/walk_forward.py \
  model=mlp_regime model.compile=false \
  walk.shuffle_check=true walk.n_windows=1 'walk.seeds=[42]' \
  train.total_steps=10240 train.n_envs=8 train.n_steps=32 \
  train.n_minibatches=16 train.n_epochs=2 \
  train.checkpoint_interval=10 train.log_interval=5
```

MLflow run `790f9673e816489f8e414b3e59966fa5`, summary run
`065b5281f2dc421f8e8520bc88c77cab`, experiment `walk_forward_mlp_regime`
(id 2, created by this run). On-disk artefact:
`data/walks/summary_mlp_regime.json`.

| arm | test `mean_sharpe` |
|---|---:|
| agent, real ticker labels | **3.1447** |
| agent, **shuffled ticker labels** | **3.1597** |
| `EqualWeightRebalanced`, same episode window | **3.9577** |
| agent, val split | −1.3431 |
| agent, train (aggregated over episodes) | −0.0213 |

`W1/test` = 2020-03-02 … 2021-02-26, one window, one seed.

### 5.3 How to read it

The check has two red flags, defined at `walk_forward.py:305-311`.

- **(a) shuffled ≈ real → FIRES.** 3.1597 vs 3.1447, a gap of 0.015 Sharpe,
  0.5 %. Destroying the correspondence between a stock's features and that
  stock's realised return cost the policy nothing. Its test Sharpe is entirely
  market beta plus the tradeability mask; there is no cross-sectional edge to
  destroy.
- **(b) shuffled ≫ equal-weight → does NOT fire.** 3.1597 < 3.9577. Nothing is
  reaching the agent through a channel that survives relabelling. On this
  evidence there is **no observational leak** into the policy — which is the
  same conclusion §1 reaches by call graph.

### 5.4 What this run does *not* establish

- **The policy is barely trained.** 10,240 environment steps, 40 PPO updates,
  and `target_kl` (§7.12) cut most updates to a single epoch. A near-initial
  policy allocates near-uniformly, so "shuffling changed nothing" is the
  expected outcome and carries no information about leakage. Flag (a) firing
  here is a property of the budget, not a finding.
- **The two arms are not evaluated under the same conditions.** The real arm
  goes through `_evaluate_split`, which leaves the model in **train** mode with
  dropout active; the shuffled arm calls `model.eval()` at
  `walk_forward.py:492`. See §7.1. The 0.015 gap is within that confound.
- **It is one window and one seed, and the window is one deterministic date
  range** (§7.2), so `std = 0.0` on every arm is an artefact, not a measurement.
- `paired_bootstrap_ci` with `n = 1` returned
  `{"mean_diff": -0.8130, "ci_lo": -0.8130, "ci_hi": -0.8130, "p_above_zero": 0.0}`
  — a zero-width "95 % CI" reported as a significance result
  (`walk_forward.py:684-717` has no minimum-n guard).
- `val_test_corr` was empty: `val_test_correlation` needs ≥3 runs.

The correct verdict is: **the shuffled-ticker arm now runs end-to-end and
reports; it has produced no evidence of a leak; and it cannot produce
meaningful evidence either way until §7.1 and §7.2 are resolved and it is run
on a trained policy over ≥3 window × seed pairs.**

---

## 6. `graph.py:334` — the per-batch mask

**Verdict: a correctness bug, not a simplification. The justification in the
comment is true of `sector_ids` and false of `mask`, and the two are handled by
the same line.**

### 6.1 What the code does

```
graph.py:326-327   if tradeable_mask is not None:
                       stock_x = stock_x * tradeable_mask.unsqueeze(-1).float()   ← per-element, CORRECT
graph.py:329-331   # The universe is fixed, so sector assignments are the same for every
                   # batch element; we use index [0] and tile the result.
graph.py:332       sec_ids_0 = sector_ids[0]
graph.py:333-335   mask_0 = tradeable_mask[0] if tradeable_mask is not None else None
graph.py:336       base_edges = build_sector_edges(sec_ids_0, mask_0, self._relations)
graph.py:338       batched_edges = self._tile_edges(base_edges, B, N, S)
```

Node *features* are masked per batch element. Graph *topology* is built from
element 0 and tiled. The stated premise ("sector assignments are the same for
every batch element") is correct — measured, `sector_ids` differ on **0** of
1,971 consecutive-day pairs in `data/panels/train.parquet` (max Hamming
distance 0). The premise does not extend to `tradeable_mask`, which is a
function of the date.

Two failure modes follow, both visible in `build_sector_edges`
(`graph.py:135-179`):

- a name tradeable in element *b* but not in element 0 gets **no edges at all**
  (only GATv2's `add_self_loops` on the intra-sector relation), so it is
  isolated from the graph;
- a name untradeable in element *b* but tradeable in element 0 **keeps its
  edges** while its feature vector has already been zeroed at `:327`, so a zero
  vector is attended over and dilutes every neighbour in its sector.

### 6.2 Direct demonstration

`scratchpad/gnnmask.py` — `HeteroGNN(N=12, 4 sectors, 2 layers, dropout=0,
drop_edge=0)`, eval mode. Two batch elements with identical features and
different masks, compared against the same elements run alone at `B=1`:

```
element 0 (== mask[0]):  max|batched - solo| = 4.768e-07
element 1 (!= mask[0]):  max|batched - solo| = 2.701e+00

  per-node error, element 1 (untradeable slots 0,3,6):
    node  0 sector 1  err=2.141e+00 UNTRADEABLE
    node  1 sector 1  err=2.965e-01
    node  2 sector 1  err=2.688e-01
    node  3 sector 2  err=2.432e+00 UNTRADEABLE
    node  4 sector 2  err=2.133e-01
    node  5 sector 2  err=1.540e-01
    node  6 sector 3  err=2.701e+00 UNTRADEABLE
    node  7 sector 3  err=7.298e-01
    node  8 sector 3  err=6.000e-01
    node  9 sector 4  err=2.384e-07     ← sector with no masked names: unaffected
    node 10 sector 4  err=1.788e-07
    node 11 sector 4  err=2.384e-07

same two elements, order swapped:
  batched2[0] vs solo1: 3.576e-07   (mask[0]=m1, so correct)
  batched2[1] vs solo0: 1.403e+00   (mask[0]=m1 applied to the m0 element)

edges[('stock','same_sector','stock')]: from mask[0]=24  correct for element 1=12
edges[('stock','in','sector')]:         from mask[0]=12  correct for element 1=9
edges[('sector','contains','stock')]:   from mask[0]=12  correct for element 1=9
```

The error is O(1) on the embedding scale, it propagates to *tradeable*
neighbours (nodes 1, 2, 4, 5, 7, 8), and the same `(features, mask)` pair
produces different embeddings depending on which sample lands at index 0 of the
minibatch.

### 6.3 How often does it fire on real data?

`scratchpad/maskq.py`. PPO minibatch size is
`n_steps · n_envs / n_minibatches = 256·16/32 = 128`
(`configs/train/ppo_baseline.yaml`). Minibatch indices are reshuffled every
epoch (`ppo.py:329`), so a minibatch is ~128 dates drawn at random from the
rollout. Simulated with 2,000 random draws of 128 dates from each real training
panel, universe = `active_tickers()` (504 slots):

| | `data/panels/train.parquet` | `data/panels_kite/train.parquet` |
|---|---:|---:|
| dates in panel | 1,972 | 4,711 |
| slots never tradeable | 363 / 504 | 361 / 504 |
| tradeable per day (min / median / max) | 0 / 129 / 141 | 0 / 118 / 143 |
| consecutive-day pairs with an identical mask | 98.78 % | 98.70 % |
| **batch elements whose tradeable set differs from `mask[0]`** | **87.60 %** | **94.27 %** |
| mean Hamming distance to `mask[0]` | 9.74 names | 17.54 names |
| — of which tradeable-but-edgeless | 4.65 | 8.48 |
| — of which untradeable-but-still-edged | 5.09 | 9.07 |

So on the yfinance panel roughly **7 of every 8 samples in a GNN minibatch are
given the wrong adjacency**, wrong by ~9.7 of ~130 tradeable names (7.5 % of the
live cross-section); on the Kite panel, ~15 of every 16, wrong by ~17.5 names
(14.9 %). The consecutive-day figure (98.8 % identical) is what makes this easy
to miss: masks change rarely *day to day*, but a minibatch spans years.

### 6.4 Consequence inside PPO

`scratchpad/gnnreal.py` — `GNNActorCritic` at the real `N = 504`, real masks
from `data/panels/train.parquet`, one fixed transition (fixed features, fixed
action, fixed own mask), evaluated 12 times at different positions in an 8-sample
minibatch with different neighbours:

```
log_prob of the SAME transition: min=-12944.5957 max=-12943.9141 range=0.6816
implied importance ratio exp(range) = 1.977   (PPO clip_coef=0.2 clips at |ratio-1|>0.2)
```

`ppo.py:357-365` computes `ratio = exp(new_lp − old_lp)`. `old_lp` was measured
during rollout at `B = n_envs = 16` with one mask composition; `new_lp` is
measured in the update at `B = 128` with a reshuffled composition, four times
per update. A factor-of-2 swing in the ratio from minibatch composition alone
is ~10× the clip threshold. This is the same failure signature that
`ppo.py:209-217` documents ("GNN `clip_frac` collapses to 1.000 from the very
first update") and attributes entirely to dropout; forcing `eval()` removed the
dropout term but leaves this one.

### 6.5 Why no test catches it

Every mask in `tests/unit/test_graph.py` is constant across the batch:
`:39` `torch.ones(B, N)`, `:210-211` `mask[:, 0] = False` (all elements
identical), `:236-237` `B = 1`. There is no test with a mask that varies along
the batch dimension.

---

## 7. Other correctness concerns found while tracing

Recorded, not fixed. Ordered by consequence.

### 7.1 Dropout is inverted: off during training, on during evaluation

- `ppo.py:218` `self.model.eval()` runs at the top of every update iteration,
  before the rollout, with the stated aim of making `log_prob_old` deterministic.
- `ppo.py:445` `self.model.train()` runs **after** the `for update` loop.
- Nothing switches back in between, so the **gradient updates at `ppo.py:342-398`
  also run in eval mode**. `configs/model/*.yaml` `tcn.dropout: 0.1`,
  `graph.dropout: 0.1` and `graph.drop_edge_prob: 0.1` never take effect during
  training. Every declared regulariser is dead.
- `_evaluate_split` (`runner.py:397-440`) is called at `runner.py:359` and `:371`,
  i.e. after `trainer.train()` returned and left the model in **train** mode, and
  never calls `model.eval()`. **Val and test evaluation runs with dropout
  active.** `scratchpad/evalmode.py`:
  ```
  after model.train()  [dropout ON]   model.training=True   max spread over 8 identical forwards = 1.614e-02
  after model.eval()   [dropout OFF]  model.training=False  max spread over 8 identical forwards = 0.000e+00
  ```
- `walk_forward.py:492` (`shuffled_ticker_test_metrics`) *does* call
  `model.eval()`. So **the shuffled arm and the real test arm are evaluated
  under different stochasticity** — the §5 comparison is confounded by
  construction, independently of anything else.
- `scripts/paper_run.py:209` also calls `.eval()`. So live trading and the
  shuffle check are deterministic; reported val/test metrics are not.

### 7.2 All five evaluation "episodes" are the same window

`panel_env.py:75-88` clamps `episode_length` to `len(dates) − lookback − 1` when
the panel is short. `panel_env.py:207-211` then draws the start index from
`[lookback, max(lookback, len(dates) − episode_length − 1)]`. After the clamp
those two bounds are equal, so the draw is deterministic. `scratchpad/epwin.py`:

```
W1/val     n_dates= 243  episode_length(clamped)=182  5 reset start indices = [60,60,60,60,60]  distinct=1
W1/test    n_dates= 249  episode_length(clamped)=188  5 reset start indices = [60,60,60,60,60]  distinct=1
W1/train   n_dates=1230  episode_length(clamped)=252  5 reset start indices = [751,678,305,712,78]  distinct=5
```

`runner.py:405` `n_episodes: int = 5` therefore buys five draws of the *policy's
action noise* on one identical date range, not five windows. `val_std_sharpe` /
`test_std_sharpe` measure sampling noise, and `paired_bootstrap_ci`
(`walk_forward.py:684-717`) is pairing single-window point estimates. Training
episodes are unaffected.

### 7.3 The walk-forward "full panel" has two month-long holes

`scripts/walk_forward.py:64-75` rebuilds the full panel by concatenating
`data/panels/{train,val,test}.parquet` — the *already purged* splits. The purge
months are gone from disk, so they are gone from the walk-forward panel.
`scratchpad/holes.py`:

```
reconstructed walk-forward panel: 2014-01-01 2024-12-30  2669 days
true NSE calendar over same span: 2710
missing trading days: 41
   hole: 2022-01-03 .. 2022-01-31  (20 trading days)
   hole: 2023-01-02 .. 2023-01-31  (21 trading days)
```

Effect on the four default windows — `materialise_window` (`walk_forward.py:140`)
raises only on an *empty* segment, so short segments pass silently:

| segment | requested | rows present | true calendar | |
|---|---|---:|---:|---|
| W2/test | 2021-03-01 … 2022-02-28 | 228 | 248 | 20-day hole mid-episode |
| W3/val | 2021-02-01 … 2022-01-31 | 228 | 248 | **ends 2021-12-31, a month early** |
| W3/test | 2022-03-01 … 2023-02-28 | 228 | 249 | 21-day hole mid-episode |
| W4/val | 2022-02-01 … 2023-01-31 | 228 | 249 | **ends 2022-12-30, a month early** |

The other 8 segments are intact. Where the hole falls mid-segment the env treats
the rows either side as consecutive trading days (`panel_env.py:343`,
`self._dates` is just the sorted unique list), so a 60-row observation window can
span 81 real sessions.

### 7.4 `scripts/walk_forward.py` cannot see the Kite panel

`scripts/walk_forward.py:59` hardcodes `panels_root = orig_cwd / "data" / "panels"`
and never reads `cfg.data.panels_root`, which `configs/data/kite_v1.yaml:19` sets
to `data/panels_kite`. `data=kite_v1` changes nothing about which panel
walk-forward trains on.

### 7.5 The documented shuffle-check command fails

`scripts/walk_forward.py:15-17` instructs
`+walk.shuffle_check=true`, on the stated grounds that "`shuffle_check` is not
yet a key in `configs/walk/default.yaml`". It has been a key since
`configs/walk/default.yaml:20`. Run as documented:

```
Could not append to config. An item is already at 'walk.shuffle_check'.
Either remove + prefix: 'walk.shuffle_check=true'
Or add a second + to add or override 'walk.shuffle_check': '++walk.shuffle_check=true'
```

### 7.6 `sector_id == 9` will crash the critic the moment the panel is rebuilt

`ModelConfig.num_sectors` defaults to 8 (`actor_critic.py:30`) and
**`runner.py:178-196` never sets it**, so `CriticHead` is always built with
`num_sectors=8` for the non-graph models. `universe.py:237-252` defines 14
sectors; `active_tickers()` (`universe.py:284`) currently returns 504 names
spanning sector ids 1–9, with 99 in `capital_goods` (id 9). `CriticHead.forward`
(`heads.py:218-224`) does `scatter_add_` at index `sector_id − 1`:

```
sector_ids=[1, 2, 8] -> ok, value=-0.3361
sector_ids=[1, 2, 9] -> RuntimeError: index 8 is out of bounds for dimension 1 with size 8
```

This has not fired only because the panels on disk are stale (§7.7) and contain
no sector-9 name. `configs/model/gnn_v1.yaml:15` likewise hardcodes
`graph.num_sectors: 8`; in `HeteroGNN` a sector-9 membership edge would not
raise but would point at the *next batch element's* sector-0 node
(`graph.py:293-302` offsets by `b*S`).

### 7.7 Both panels are stale relative to `universe.py`

| | `data/panels` | `data/panels_kite` |
|---|---:|---:|
| tickers in the panel | 163 | 163 |
| tickers in `data/*_ohlcv/year=2022/` | 161 | **545** |
| panel tickers still in `all_tickers()` | 153 | 153 |
| panel tickers in `active_tickers()` | 143 | 143 |
| panel tickers no longer in `all_tickers()` | 10 (`LTIM.NS`, `MCDHOLDING.NS`, `NIITLTD.NS`, `NYKAA.NS`, …) | same 10 |
| panel `sector_id` disagreeing with today's `sector_id_of()` | 17 | 17 |
| distinct panel `sector_id` max | 8 | 8 |

The Kite fetch expanded the store to 545 tickers (commit `a5d8b79`, "expand the
universe to 645/504") but `data/panels_kite` was written **before** that and
holds the same 163-name universe as the April yfinance panel. `active_tickers()`
returns 504, so the env is built with **504 observation slots of which 361
(72 %) are permanently untradeable padding** (§6.3), and `n_tickers = 504` sizes
the actor's 505-dim action head and `log_std` accordingly.

### 7.8 Sector 0 is still live in the data

`CLAUDE.md` lists phantom sectors as "Fixed; regression test exists". The fix is
in `universe.py:295-320`; the panels on disk predate it:

| panel | rows with `sector_id == 0` | tradeable among them | share of all tradeable rows | tickers |
|---|---:|---:|---:|---|
| `data/panels/train.parquet` | 15,776 | 15,296 | 5.54 % | ADANIPORTS, ASIANPAINT, BHARTIARTL, GRASIM, SHREECEM, TITAN, ULTRACEMCO, UPL |
| `data/panels_kite/train.parquet` | 37,688 | 36,481 | 5.85 % | same 8 |

`build_sector_edges` (`graph.py:134, 155`) excludes `sector_id == 0` from every
edge, and `CriticHead` (`heads.py:219`) zeroes their sector exposure. So eight
large-cap NIFTY names — 5.5 % of all tradeable rows in the training panel — are
invisible to the graph and to the critic's sector vector. The regression test
passes; the data is still wrong. (`data/panels/train.parquet` has 276,005
tradeable rows, matching the figure `CLAUDE.md` cites for the `beta_nifty_60d`
incident — this is the same panel.)

### 7.9 `torch.compile` no longer reaches the hot path

`runner.py:244-265` rebinds `model.forward` to its compiled version, justified by
"PPO's hot path is `model.get_action_and_value(obs)`, which internally calls
`self.forward(obs)`". Phase 1 introduced `_forward_shared`, and
`ActorCritic.get_action_and_value` (`actor_critic.py:233`) and
`get_action_value_and_aux` (`:259`) now call `_forward_shared` directly.
`scratchpad/compilecheck.py`:

```
ActorCritic      rebound forward called by get_action_and_value: 0x   by get_value: 1x
GNNActorCritic   rebound forward called by get_action_and_value: 1x   by get_value: 1x
```

`configs/model/mlp_regime.yaml:38` and `mlp_regime_aux.yaml` set `compile: true`;
for `ActorCritic` it now compiles only the once-per-rollout bootstrap value call.

### 7.10 `_evaluate_split` builds a different env from the one trained on

`runner.py:412-420` constructs the eval env with only `lookback`,
`episode_length` and `initial_cash` from `env_kwargs`, dropping `reward_fn`,
`turnover_penalty` and `use_excess_returns` (which `runner.py:150-160` did set).
`walk_forward.py:129-141` deliberately mirrors this so the baseline arm stays
comparable. Metrics are NAV-derived so the numbers are unaffected, but any
experiment varying `env.reward` or `env.turnover_penalty` is evaluated under the
defaults, and `configs/env/panel_daily.yaml:11` `turnover_penalty: 0.001` is
silently absent from every reported val/test figure.

### 7.11 `MomentumTopK` is not a momentum baseline

`env/baselines.py:104-109` scores names with `features[-1, :, 0]` and admits in
the comment that it "picks whatever feature is at index 0".
`FEATURE_COLS[0]` is `log_return_1d` (`features.py:17`), not `log_return_20d` as
the class docstring (`baselines.py:87`) states. The baseline registered as
`momentum_top5` in `scripts/evaluate.py:60` ranks on **yesterday's one-day
return** — a short-horizon reversal signal, not 20-day momentum.

### 7.12 `target_kl` is not configurable

`PPOConfig.target_kl = 0.02` (`ppo.py:68`) is never populated from Hydra —
`runner.py:277-299` omits it, and no `configs/train/*.yaml` declares the key. In
the §5 run it early-stopped the epoch loop (`ppo.py:404-406`) on essentially
every update, so `train.n_epochs` is largely notional.

### 7.13 `_rebuild_eval_model` is a hand-maintained copy

`walk_forward.py:351-453` duplicates `runner.py:174-223` and says so
(`:359-366`, with a `TODO`). `strict=True` on `load_state_dict` (`:492`) catches
shape drift, but not a silently different `num_sectors`, `dropout`, or
normaliser source — a checkpoint that loads is not a checkpoint that behaves the
same.

---

## The three findings that most change what should be done next

**1. The `−0.86` that justifies Phase 1 and Phase 2 has no artefact in this
repository, and the code that would produce it cannot currently produce a
trustworthy one.** Before this audit's run, `mlruns/mlflow.db` held two
experiments (`Default`, `trading_bot`), five runs, and **not one `test_*`
metric had ever been logged**; `data/walks/` was empty; no `window` or `seed`
tag exists. So `corr(val_sharpe, test_sharpe) = −0.86` — the number
`HANDOFF.md` §2, `09_revamp_and_audit.md` §3 and `scripts/walk_forward.py:21`
all treat as the decision metric — is unreproducible here. Worse, the code path
that computes it draws five "episodes" that are the *same deterministic date
range* (§7.2) and evaluates them with dropout **on** (§7.1), so what it measures
is the policy's action noise on one window. ~2,500 lines of Phase 1 + Phase 2
architecture were built to move a statistic that this repository cannot
currently compute.

**2. `graph.py:334` mis-wires 87.6 % of every GNN minibatch on the yfinance
panel and 94.3 % on the Kite panel, and is a live candidate cause of the GNN
instability the code blames on dropout.** Mean 9.7 of ~130 tradeable names get
the wrong adjacency per sample (17.5 on Kite); the embedding error is O(1) and
propagates to correctly-masked neighbours; and the same transition's `log_prob`
swings by 0.68 nats — an importance ratio of 1.98 against `clip_coef = 0.2` —
purely from which sample lands at minibatch index 0. `ppo.py:209-217` documents
"GNN `clip_frac` collapses to 1.000 from the very first update" and attributes
it entirely to dropout; forcing `eval()` removed that term and left this one.
Every `gnn_v1` / `gnn_intra_only` result is uninterpretable, and no test in
`tests/unit/test_graph.py` uses a mask that varies along the batch dimension.

**3. The purge is short at all eight walk-forward boundaries *and* in the new
`kite_v1` split, the minimum that closes the 60-day windows is 3 months, and
fixing the split alone will not fix it.** Measured gaps are 19–23 trading days
against a 60-trading-day window, leaving 36–40 contaminated rows at every
boundary; `configs/data/kite_v1.yaml:42-44` claims the opposite in a comment.
The first 60 val rows are not computable at all without pre-split history
(8.4 % of tradeable val rows), and evaluation starts at exactly panel index 60,
so the single evaluated window is built from them. Separately,
`scripts/walk_forward.py:64-75` rebuilds its "full panel" from the *already
purged* splits, so the 41 purge-month sessions are simply absent — silently
truncating W3/val and W4/val by a month and putting a 20-day hole inside
W2/test and W3/test.
