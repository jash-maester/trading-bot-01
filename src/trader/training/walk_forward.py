"""Walk-forward training protocol — M7.

Slides a (train, val, test) window forward across time, training a fresh
PPO policy on each window's train segment and evaluating on val/test.
Aggregated metrics across windows × seeds give us proper statistical
power (single-seed std_sharpe ≈ 1.5 was making r6/r7 results unreadable).

Window definition follows the spec in `05_training.md`:

```
Window 1: train 2014..2018  val 2019  test 2020
Window 2: train 2015..2019  val 2020  test 2021
Window 3: train 2016..2020  val 2021  test 2022
Window 4: train 2017..2021  val 2022  test 2023
```

SPEC DIVERGENCE (unresolved, reported not silently followed): that table assumes
a purge small enough to leave val and test on calendar-year boundaries.  With the
3-month purge the code now requires, W1 is train 2014-01-01..2018-12-31,
val 2019-04-01..2020-03-31, test 2020-07-01..2021-06-30 — the segments are still
12 months each but no longer aligned to calendar years.  The spec has not been
updated; the leakage fix takes precedence over the cosmetic alignment.

A purge gap is enforced between consecutive segments to prevent rolling-window
features from leaking across the boundary.  It was 1 month, which is 17–23 NSE
trading days against 60-day features (`realized_vol_60d`, `beta_nifty_60d`) —
roughly 40 contaminated rows on every one of the 8 boundaries.  It is now 3
months, and `compute_windows` refuses anything shorter than the longest feature
lookback rather than letting the regression happen again quietly.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl
from loguru import logger
from omegaconf import DictConfig

from trader.data.features_ext import combined_max_lookback_days

if TYPE_CHECKING:  # torch is imported lazily — keeps `import walk_forward` cheap
    import torch
    from torch import nn

# NSE trades ~252 days a year, so a calendar month is ~21 trading days.  Only
# used to turn `purge_months` into a comparable number of trading days for the
# leakage guard; the real per-month count ranges 17–23 (measured over
# data/kite_ohlcv, 2014–2026), so a purge that only just clears the guard is
# worth widening by one more month.
_TRADING_DAYS_PER_MONTH = 21

# Default purge, in months.  Mirrors `configs/walk/default.yaml: purge_months`;
# the two must agree, since the config is what a real run uses and this is what
# a bare `compute_windows(...)` call gets.
DEFAULT_PURGE_MONTHS = 3

# ── Window definition ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class WindowConfig:
    """Date boundaries for one walk-forward window."""

    name: str               # e.g., "W1"
    train_start: date
    train_end: date
    val_start: date
    val_end: date
    test_start: date
    test_end: date

    def asdict_iso(self) -> dict[str, str]:
        out = {}
        for k, v in asdict(self).items():
            out[k] = v.isoformat() if isinstance(v, date) else str(v)
        return out


def _add_months(d: date, months: int) -> date:
    """Return a date `months` after `d`, clamping to month-end as needed."""
    y = d.year + (d.month - 1 + months) // 12
    m = (d.month - 1 + months) % 12 + 1
    # Clamp day to last valid day of month (handles 31 → 28/29/30)
    for trial_day in (d.day, 31, 30, 29, 28):
        try:
            return date(y, m, trial_day)
        except ValueError:
            continue
    raise ValueError(f"Could not add {months} months to {d}")


def assert_purge_clears_feature_lookback(
    purge_months: int, *, include_ext: bool = False
) -> None:
    """Raise unless `purge_months` covers the longest feature lookback.

    The purge gap exists to stop a rolling feature window from spanning a
    train/val or val/test boundary.  If the gap is shorter than the longest
    window any feature uses, it does not do that: rows on both sides are
    computed from overlapping history, and the val/test segments are
    contaminated by train data.  `purge_months: 1` gave 17–23 trading days
    against 60-day features and roughly 40 contaminated rows per boundary.

    The bound is *derived* — `MAX_FEATURE_LOOKBACK_DAYS` comes from
    `trader.data.features.FEATURE_LOOKBACK_DAYS`, which is checked against the
    window literals in that module by its own tests.  Adding a 120-day feature
    therefore tightens this guard automatically instead of quietly invalidating
    every walk-forward run.

    ``include_ext`` widens the bound to cover the R8 ext feature group, whose
    longest window (``delivery_pct_z_60d``, 61 days) lives in
    ``trader.data.features_ext`` and is invisible to
    ``MAX_FEATURE_LOOKBACK_DAYS``.  It defaults to False so no existing run
    changes behaviour, and at today's ``purge_months: 3`` (~63 trading days)
    the 61-day bound already clears — this moves no window today.  It is the
    guard against the *next* longer feature, which is when a silently
    unwidened purge would begin contaminating every boundary.
    """
    bound = combined_max_lookback_days(include_ext=include_ext)
    purge_days = purge_months * _TRADING_DAYS_PER_MONTH
    if purge_days < bound:
        needed = -(-bound // _TRADING_DAYS_PER_MONTH)
        raise ValueError(
            f"purge_months={purge_months} is ~{purge_days} trading days, shorter "
            f"than the longest feature lookback ({bound} days, from "
            f"trader.data.features.FEATURE_LOOKBACK_DAYS"
            f"{' + features_ext.EXT_FEATURE_LOOKBACK_DAYS' if include_ext else ''}"
            f"). Train and val "
            f"feature windows would physically overlap across every boundary. "
            f"Use walk.purge_months >= {needed} (configs/walk/default.yaml)."
        )


def compute_windows(
    data_start: date,
    data_end: date,
    train_years: int = 5,
    val_months: int = 12,
    test_months: int = 12,
    purge_months: int = DEFAULT_PURGE_MONTHS,
    n_windows: int = 4,
    step_months: int = 12,
    include_ext_features: bool = False,
) -> list[WindowConfig]:
    """Generate sliding walk-forward windows.

    Each window has the structure:
        ┌──────── train_years ────────┐  purge  ┌── val_months ──┐  purge  ┌── test_months ──┐

    Windows are advanced by `step_months` (default 12 — one window per
    calendar year of test data).  Stops when either `n_windows` is
    reached or a window would extend past `data_end`.

    Raises
    ------
    ValueError
        If `purge_months` is too short to clear the longest feature lookback —
        see :func:`assert_purge_clears_feature_lookback`.
    """
    assert_purge_clears_feature_lookback(purge_months, include_ext=include_ext_features)

    windows: list[WindowConfig] = []
    cursor_train_start = data_start
    for i in range(1, n_windows + 1):
        train_start = cursor_train_start
        train_end = _add_months(train_start, 12 * train_years) - timedelta(days=1)
        val_start = _add_months(train_end + timedelta(days=1), purge_months)
        val_end = _add_months(val_start, val_months) - timedelta(days=1)
        test_start = _add_months(val_end + timedelta(days=1), purge_months)
        test_end = _add_months(test_start, test_months) - timedelta(days=1)

        if test_end > data_end:
            logger.warning(
                f"Window W{i} truncated/skipped: test_end {test_end} > "
                f"data_end {data_end}"
            )
            break

        windows.append(
            WindowConfig(
                name=f"W{i}",
                train_start=train_start,
                train_end=train_end,
                val_start=val_start,
                val_end=val_end,
                test_start=test_start,
                test_end=test_end,
            )
        )
        cursor_train_start = _add_months(cursor_train_start, step_months)

    return windows


# ── Panel slicing ──────────────────────────────────────────────────────────────


def slice_panel(
    full_panel: pl.DataFrame,
    start: date,
    end: date,
) -> pl.DataFrame:
    """Return rows of `full_panel` with `start <= date <= end`."""
    if "date" not in full_panel.columns:
        raise ValueError("panel must have a 'date' column")
    return full_panel.filter(
        (pl.col("date") >= start) & (pl.col("date") <= end)
    )


def find_calendar_gaps(
    panel: pl.DataFrame,
    max_gap_days: int = 10,
) -> list[tuple[date, date, int]]:
    """Consecutive dates in `panel` more than `max_gap_days` calendar days apart.

    Exists because the "full historical panel" the driver assembles is
    ``concat(train.parquet, val.parquet, test.parquet)`` — and those three do not
    tile the history.  `build_features.py` cuts the purge months out of the
    *start of each later split* and never writes them anywhere, so the
    concatenation has a month-shaped hole at each build-time boundary.  Walk
    forward windows carry their own purge structure and land nowhere near those
    boundaries, so segments were being sliced against a calendar with whole
    months missing, with nothing in the output to say so.

    `max_gap_days=10` clears NSE's longest real closure (Diwali/holiday clusters
    plus a weekend never reaches 10 calendar days) while catching a purged month.

    Returns ``[(gap_start, gap_end, n_calendar_days), ...]``, empty when clean.
    """
    if "date" not in panel.columns:
        raise ValueError("panel must have a 'date' column")
    dates = panel["date"].unique().sort().to_list()
    gaps: list[tuple[date, date, int]] = []
    for prev, nxt in zip(dates, dates[1:], strict=False):
        span = (nxt - prev).days
        if span > max_gap_days:
            gaps.append((prev, nxt, span))
    return gaps


def materialise_window(
    full_panel: pl.DataFrame,
    window: WindowConfig,
    out_dir: Path,
    warmup_days: int = 0,
) -> dict[str, Path]:
    """Slice `full_panel` for the given window and write three parquet files.

    Returns a dict ``{'train': path, 'val': path, 'test': path}``.

    ``warmup_days`` prepends that many trading days of **feature context** to the
    test segment, taken from the purge gap that precedes it. Without it the first
    ``lookback - 1`` days of every OOS segment have no full input window and are
    never predicted — 59 days per window at lookback 60, which was 472 of 1980
    OOS days (24%) on the shipped 8-window configuration. That is lost power, not
    contamination.

    The rows are marked ``is_warmup`` and carry no labels and no score; they exist
    only so the encoder has history at the first real test date. This is what a
    live system has on that morning.

    The 3-month purge is sized for exactly this — 62 trading days against a
    59-day need — and the warm-up is asserted to stay inside it, so no test input
    window ever reaches back into validation.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    warmup_start: date | None = None
    if warmup_days > 0:
        prior = sorted(
            full_panel.filter(pl.col("date") < window.test_start)["date"].unique().to_list()
        )
        if len(prior) < warmup_days:
            logger.warning(
                f"  {window.name}/test: only {len(prior)} trading day(s) precede "
                f"{window.test_start} in the panel; wanted {warmup_days} of warm-up. "
                "The first OOS days will still be unpredicted."
            )
            warmup_start = prior[0] if prior else None
        else:
            warmup_start = prior[-warmup_days]
        # Never let the context cross into validation. Clamp rather than raise:
        # a short purge costs predictable days, which is the very thing the
        # warm-up exists to recover, and losing some of them is not a reason to
        # fail the run. What must never happen is reaching back past val_end, so
        # that is enforced here and asserted in tests.
        if warmup_start is not None and warmup_start <= window.val_end:
            usable = [d for d in prior if d > window.val_end]
            logger.warning(
                f"  {window.name}/test: a {warmup_days}-day warm-up would start "
                f"{warmup_start}, at or before val_end {window.val_end}. Clamping "
                f"to the {len(usable)} day(s) inside the purge gap; the first "
                f"{warmup_days - len(usable)} OOS day(s) stay unpredicted. Widen "
                "walk.purge_months to recover them."
            )
            warmup_start = usable[0] if usable else None

    segments = (
        ("train", window.train_start, window.train_end),
        ("val", window.val_start, window.val_end),
        ("test", warmup_start or window.test_start, window.test_end),
    )
    for name, start, end in segments:
        sub = slice_panel(full_panel, start, end)
        if name == "test" and warmup_start is not None:
            sub = sub.with_columns(
                (pl.col("date") < window.test_start).alias("is_warmup")
            )
            n_warm = sub.filter(pl.col("is_warmup"))["date"].n_unique()
            logger.info(
                f"  {window.name}/test: {n_warm} warm-up day(s) "
                f"{warmup_start}..{window.test_start} prepended as feature context "
                "(unlabelled, unscored)"
            )
        else:
            sub = sub.with_columns(pl.lit(False).alias("is_warmup"))
        if sub.is_empty():
            raise ValueError(
                f"Window {window.name} {name} segment is empty for "
                f"{start}..{end}"
            )
        path = out_dir / f"{name}.parquet"
        sub.write_parquet(path)
        paths[name] = path
        d_min = sub["date"].min()
        d_max = sub["date"].max()
        logger.info(
            f"  {window.name}/{name}: {sub.shape[0]} rows, {d_min!s}..{d_max!s}"
        )

        # A segment can be silently short in two ways: the source panel stops
        # before the segment does (truncation at the edges), or it has a hole in
        # the middle (a build-time purge month the driver concatenated over).
        # Both used to pass without a word.  Say it plainly, per segment.
        interior_gaps = find_calendar_gaps(sub)
        if interior_gaps:
            logger.error(
                f"  {window.name}/{name}: {len(interior_gaps)} calendar gap(s) "
                f"inside the segment: "
                + ", ".join(f"{a}→{b} ({n}d)" for a, b, n in interior_gaps)
                + " — this segment is missing data it was defined to include."
            )
        if isinstance(d_min, date) and isinstance(d_max, date):
            lead = (d_min - start).days
            trail = (end - d_max).days
            # ~1 week absorbs a segment boundary landing on a weekend/holiday.
            if lead > 7 or trail > 7:
                logger.warning(
                    f"  {window.name}/{name}: requested {start}..{end} but the "
                    f"panel only covers {d_min}..{d_max} "
                    f"({lead}d missing at the start, {trail}d at the end)."
                )
    return paths


# ── Comparison arms: equal-weight baseline + shuffled-ticker leak check ────────
#
# These three constants mirror `trader.training.runner._evaluate_split`, which
# is what produces the agent's own test metrics.  The episode window an env
# hands out is a pure function of the reset seed, so the baseline arm is a
# genuinely *paired* comparison only while these stay in sync with runner.py.
# (runner.py is not ours to edit; if its offsets change, these must follow.)
_EVAL_N_EPISODES = 5      # runner._evaluate_split's `n_episodes` default
_TEST_SEED_OFFSET = 1     # runner evaluates the test split with `seed + 1`
_ENV_SEED_OFFSET = 999    # runner resets with `seed + 999 + episode_index`


def _eval_env_kwargs(cfg: DictConfig) -> dict[str, Any]:
    """Env kwargs for an *evaluation* env, matching runner._evaluate_split.

    Deliberately omits `reward_fn`, `turnover_penalty` and `use_excess_returns`:
    runner builds its eval envs with the defaults too, and reward shaping does
    not affect realised NAV, only the training signal.  Matching it exactly is
    what keeps the arms comparable.
    """
    return {
        "lookback": int(cfg.env.lookback_days),
        "episode_length": int(cfg.env.episode_length),
        "initial_cash": float(cfg.env.initial_cash),
    }


def equal_weight_test_metrics(
    cfg: DictConfig,
    *,
    test_panel: Path,
    seeds: list[int],
    n_episodes: int = _EVAL_N_EPISODES,
) -> dict[int, dict[str, float]]:
    """Evaluate `EqualWeightRebalanced` on one window's test panel, per seed.

    M7's acceptance criterion is a *paired* bootstrap CI of the RL agent
    against equal-weight.  `paired_bootstrap_ci` has existed (and been
    unit-tested) since M7 but nothing ever called it, because there was no
    baseline arm to pair against — this is that arm.

    Pairing is what makes the CI tight enough to be informative: agent and
    baseline are evaluated on *identical* episode windows (same reset seeds),
    so the window-to-window variation that dominates walk-forward Sharpe
    cancels in the difference.

    Returns ``{run_seed: aggregated_metrics}``.  A seed is absent from the
    result only if evaluation failed, which is logged and never raised — a
    missing benchmark must not kill a multi-day training run.
    """
    from trader.data.features import FEATURE_COLS
    from trader.data.universe import active_tickers
    from trader.env.baselines import EqualWeightRebalanced
    from trader.env.panel_env import PanelTradingEnv
    from trader.training.eval_metrics import (
        EpisodeMetrics,
        aggregate_metrics,
        compute_episode_metrics,
    )

    out: dict[int, dict[str, float]] = {}
    kwargs = _eval_env_kwargs(cfg)
    try:
        # One env per panel rather than per seed: `reset(seed=...)` replaces the
        # env's RNG outright, so the constructor seed never influences which
        # episode windows are drawn.  Saves re-parsing the parquet per seed.
        env = PanelTradingEnv(
            panel_path=test_panel,
            universe=active_tickers(),
            feature_columns=FEATURE_COLS,
            seed=seeds[0] + _TEST_SEED_OFFSET + _ENV_SEED_OFFSET if seeds else 0,
            **kwargs,
        )
    except Exception as e:  # noqa: BLE001 — benchmark must never fail a run
        logger.warning(f"Equal-weight baseline env failed for {test_panel}: {e}")
        return out

    agent = EqualWeightRebalanced()
    for seed in seeds:
        eval_seed = seed + _TEST_SEED_OFFSET
        episodes: list[EpisodeMetrics] = []
        try:
            for ep in range(n_episodes):
                obs, _ = env.reset(seed=eval_seed + _ENV_SEED_OFFSET + ep)
                agent.reset()
                nav_series = [float(obs["nav"])]
                turnovers: list[float] = []
                done = False
                while not done:
                    obs, _, terminated, truncated, info = env.step(agent.act(obs))
                    nav_series.append(float(info["nav"]))
                    turnovers.append(float(info["turnover"]))
                    done = bool(terminated or truncated)
                episodes.append(compute_episode_metrics(nav_series, turnovers))
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Equal-weight baseline failed (seed={seed}): {e}")
            continue
        if episodes:
            out[seed] = aggregate_metrics(episodes)
    return out


# ── Shuffled-ticker-label sanity check ────────────────────────────────────────
#
# `05_training.md` §Overfitting guardrails calls this the gold-standard leak
# detector.  The idea: destroy the correspondence between a stock's features
# and that stock's realised returns, and re-run the *already trained* agent.
# If it still performs as well, the performance was never coming from
# per-stock signal, and something else in the observation is carrying the
# information — possibly the future.
#
# WHAT IS PERMUTED, per evaluation episode, with one permutation held fixed
# for every step of that episode:
#   * `obs["features"]`   along the ticker axis (axis 1 of (lookback, N, F))
#   * `obs["sector_ids"]` along its only axis, with the SAME permutation, so
#     the sector a name claims to belong to travels with its features (a GNN
#     builds its adjacency from this key; leaving it aligned would hand the
#     model a true graph over falsified nodes).
#
# WHAT IS NOT PERMUTED — and this is the point of the test:
#   * `mask`, `portfolio`, `cash`, `nav`, `t_frac`, `recent_return_1d`,
#     `recent_vol_20d`, `nav_log_progress`, `regime`, `next_day_returns`.
#   * Nothing at all inside the env: prices, costs, tradeability and the
#     action→ticker mapping are untouched.  Action index j+1 still buys the
#     real ticker j.
#
# So after permutation, slot j shows the price/volume history of ticker
# perm[j] while the P&L booked at slot j is ticker j's.  Any genuine
# cross-sectional edge ("this stock's own history predicts this stock's
# return") is destroyed; anything that does not depend on that correspondence
# survives — market-level timing from `regime`/`t_frac`, the tradeability
# mask, and plain long-the-universe beta.
#
# HOW TO READ THE RESULT.  The shuffled arm is NOT expected to score zero: a
# long-only agent keeps market beta through the shuffle, so it should land
# near the equal-weight baseline arm.  The two red flags are
#   (a) shuffled ≈ real  → the agent's edge was never cross-sectional; and
#   (b) shuffled materially ABOVE equal-weight → information is reaching the
#       agent through a channel that survives ticker relabelling, i.e. a leak.
# Off by default (`walk.shuffle_check: false`); it costs one extra evaluation
# pass per run.

def make_ticker_permutation(
    mask: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Random permutation of ticker slots, restricted to tradeable names.

    Non-tradeable slots are left in place on purpose.  A padded/untradeable
    name has near-zero features; letting one land in a tradeable slot would
    give the agent a trivially detectable "this slot is fake" tell, and the
    check would then measure the agent's ability to spot the shuffle rather
    than its dependence on real cross-sectional signal.
    """
    perm = np.arange(len(mask), dtype=np.int64)
    tradeable = np.flatnonzero(np.asarray(mask).astype(bool))
    if len(tradeable) > 1:
        perm[tradeable] = tradeable[rng.permutation(len(tradeable))]
    return perm


def apply_ticker_permutation(
    obs: dict[str, np.ndarray],
    perm: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return `obs` with only the per-stock *feature* keys permuted.

    See the module comment above for the full list of what is and is not
    permuted, and why.
    """
    out = dict(obs)
    out["features"] = np.ascontiguousarray(obs["features"][:, perm, :])
    if "sector_ids" in obs:
        out["sector_ids"] = np.ascontiguousarray(obs["sector_ids"][perm])
    return out


def _rebuild_eval_model(
    cfg: DictConfig,
    *,
    feature_stats_path: Path,
    n_tickers: int,
    device: torch.device,
) -> nn.Module:
    """Rebuild the trained architecture so a checkpoint can be loaded into it.

    MIRRORS the model construction inside `trader.training.runner.train_one_run`.
    That builder is not exposed as a function and runner.py is outside this
    module's ownership, so it is duplicated here.  The duplication is made safe
    by loading the checkpoint with ``strict=True``: an architecture that has
    drifted apart fails loudly instead of silently evaluating a different model.

    TODO(runner.py): extract a shared `build_model(cfg, ...)` and delete this.
    """
    import torch

    from trader.data.feature_stats import load_stats, stats_to_tensors
    from trader.data.features import FEATURE_COLS
    from trader.data.regime_features import (
        REGIME_DIM,
        load_regime_stats,
        regime_stats_to_tensors,
    )
    from trader.models.actor_critic import ActorCritic, ModelConfig

    feat_mean, feat_std = stats_to_tensors(load_stats(feature_stats_path), FEATURE_COLS)
    regime_path = feature_stats_path.with_name(
        feature_stats_path.stem.replace("feature_stats", "regime_stats")
        + feature_stats_path.suffix
    )
    if regime_path.exists():
        regime_mean, regime_std = regime_stats_to_tensors(load_regime_stats(regime_path))
    else:
        regime_mean = torch.zeros(REGIME_DIM)
        regime_std = torch.ones(REGIME_DIM)

    num_channels_raw = cfg.model.tcn.get("num_channels")
    num_channels = (
        [int(c) for c in num_channels_raw] if num_channels_raw is not None else None
    )
    model_cfg = ModelConfig(
        in_features=len(FEATURE_COLS),
        n_tickers=n_tickers,
        embed_dim=int(cfg.model.embed_dim),
        num_channels=num_channels,
        kernel_size=int(cfg.model.tcn.kernel_size),
        dropout=float(cfg.model.tcn.dropout),
        use_cross_attn=bool(cfg.model.get("use_cross_attn", True)),
        cross_attn_heads=int(cfg.model.get("cross_attn_heads", 4)),
        regime_dim=REGIME_DIM,
        regime_film_encoder=bool(cfg.model.get("regime_film_encoder", False)),
        regime_film_attn=bool(cfg.model.get("regime_film_attn", False)),
        regime_in_critic=bool(cfg.model.get("regime_in_critic", False)),
        regime_film_hidden=int(cfg.model.get("regime_film_hidden", 32)),
        use_aux_return_head=bool(cfg.model.get("use_aux_return_head", False)),
        aux_return_hidden=int(cfg.model.get("aux_return_hidden", 32)),
    )

    model: nn.Module
    if bool(cfg.model.get("use_graph", False)):
        from trader.models.graph import GNNActorCritic, GNNConfig

        gc = cfg.model.graph
        # `num_sectors` is deliberately NOT defaulted here.  GNNConfig derives it
        # from trader.data.universe.SECTOR_IDS via `default_num_sectors()`, so an
        # explicit fallback would override the derivation with a stale literal —
        # which is what an `8` here did once SECTOR_IDS grew.  Only pass it when a
        # config actually asks for a specific width.
        gnn_kwargs: dict[str, Any] = {}
        if gc.get("num_sectors") is not None:
            gnn_kwargs["num_sectors"] = int(gc["num_sectors"])
        gnn_cfg = GNNConfig(
            **gnn_kwargs,
            num_layers=int(gc.get("layers", 2)),
            num_heads=int(gc.get("num_heads", 2)),
            dropout=float(gc.get("dropout", 0.1)),
            drop_edge_prob=float(gc.get("drop_edge_prob", 0.1)),
            relations=str(gc.get("relations", "all")),
        )
        model = GNNActorCritic(model_cfg, gnn_cfg, feat_mean=feat_mean, feat_std=feat_std)
    else:
        model = ActorCritic(
            model_cfg,
            feat_mean=feat_mean,
            feat_std=feat_std,
            regime_mean=regime_mean,
            regime_std=regime_std,
        )
    return model.to(device)


def shuffled_ticker_test_metrics(
    cfg: DictConfig,
    *,
    test_panel: Path,
    checkpoint_dir: Path,
    feature_stats_path: Path,
    seed: int,
    n_episodes: int = _EVAL_N_EPISODES,
) -> dict[str, float] | None:
    """Re-evaluate the trained agent on the test panel with ticker labels shuffled.

    Reloads the newest checkpoint under `checkpoint_dir` (PPOTrainer writes
    `model_<update>.pt` every `train.checkpoint_interval` updates, so this is
    the last *checkpointed* policy, which may trail the final weights by up to
    one interval — a caveat worth remembering when the numbers are close).

    Uses the same episode windows as the agent's real test evaluation, so the
    two Sharpes are directly comparable.  Returns ``None`` — never raises — if
    anything is missing; this is a diagnostic, not a gate.
    """
    import torch

    from trader.data.features import FEATURE_COLS
    from trader.data.universe import active_tickers
    from trader.env.panel_env import PanelTradingEnv
    from trader.training.eval_metrics import (
        EpisodeMetrics,
        aggregate_metrics,
        compute_episode_metrics,
    )
    from trader.utils.seeding import get_device

    checkpoints = sorted(checkpoint_dir.glob("model_*.pt")) if checkpoint_dir.is_dir() else []
    if not checkpoints:
        logger.warning(
            f"Shuffle check skipped (seed={seed}): no checkpoint under {checkpoint_dir}"
        )
        return None
    if not feature_stats_path.exists():
        logger.warning(
            f"Shuffle check skipped (seed={seed}): missing {feature_stats_path}"
        )
        return None

    try:
        device = get_device()
        universe = active_tickers()
        model = _rebuild_eval_model(
            cfg,
            feature_stats_path=feature_stats_path,
            n_tickers=len(universe),
            device=device,
        )
        state = torch.load(checkpoints[-1], map_location=device, weights_only=True)
        model.load_state_dict(state["model_state"], strict=True)
        model.eval()

        env = PanelTradingEnv(
            panel_path=test_panel,
            universe=universe,
            feature_columns=FEATURE_COLS,
            seed=seed + _TEST_SEED_OFFSET + _ENV_SEED_OFFSET,
            **_eval_env_kwargs(cfg),
        )
        eval_seed = seed + _TEST_SEED_OFFSET
        episodes: list[EpisodeMetrics] = []
        with torch.no_grad():
            for ep in range(n_episodes):
                episode_seed = eval_seed + _ENV_SEED_OFFSET + ep
                obs, _ = env.reset(seed=episode_seed)
                # One permutation per episode, fixed for its whole length: a
                # per-step reshuffle would also destroy the *temporal*
                # consistency of a slot, which is a different (and much
                # weaker) test than breaking the feature↔return identity.
                perm = make_ticker_permutation(
                    obs["mask"], np.random.default_rng(episode_seed)
                )
                nav_series = [float(obs["nav"])]
                turnovers: list[float] = []
                done = False
                while not done:
                    shuffled = apply_ticker_permutation(obs, perm)
                    obs_t = {
                        k: torch.tensor(v, device=device).unsqueeze(0)
                        for k, v in shuffled.items()
                    }
                    action, _, _, _ = model.get_action_and_value(obs_t)  # type: ignore[operator]
                    a_np = action.squeeze(0).cpu().numpy()
                    obs, _, terminated, truncated, info = env.step(a_np)
                    nav_series.append(float(info["nav"]))
                    turnovers.append(float(info["turnover"]))
                    done = bool(terminated or truncated)
                episodes.append(compute_episode_metrics(nav_series, turnovers))
    except Exception as e:  # noqa: BLE001 — diagnostic must never fail a run
        logger.warning(f"Shuffled-ticker check failed (seed={seed}): {e}")
        return None

    return aggregate_metrics(episodes) if episodes else None


# ── Walk-forward driver ────────────────────────────────────────────────────────


def run_walk_forward(
    cfg: DictConfig,
    *,
    full_panel: pl.DataFrame,
    windows: list[WindowConfig],
    seeds: list[int],
    walks_root: Path,
    mlflow_experiment: str = "walk_forward",
) -> list[dict[str, Any]]:
    """Run PPO for every (window × seed) combination and return per-run metrics.

    Each run is logged to MLflow under `mlflow_experiment` with tags
    `window=<W1..>` and `seed=<n>` so it's easy to filter later.

    Returns a list of dicts, one per run, with structure::

        {
          "window":        "W1",
          "seed":          42,
          "train":         {...aggregated train metrics...},
          "val":           {...aggregated val metrics...},
          "test":          {...aggregated test metrics...},
          "baseline_test": {...equal-weight metrics on the same episodes...},
          "shuffled_test": {...shuffled-ticker metrics...} or None,
          "run_id":        "<mlflow id>",
        }

    ``baseline_test`` is the equal-weight arm required by M7's "paired
    bootstrap CI vs equal-weight" criterion — see
    :func:`equal_weight_test_metrics`.  ``shuffled_test`` is populated only
    when ``walk.shuffle_check`` is true; see
    :func:`shuffled_ticker_test_metrics` for what the shuffle does and does
    not touch.
    """
    from trader.training.runner import train_one_run

    shuffle_check = bool(cfg.get("walk", {}).get("shuffle_check", False))
    if shuffle_check:
        logger.info("Shuffled-ticker sanity check: ENABLED (walk.shuffle_check)")

    results: list[dict[str, Any]] = []
    for window in windows:
        win_dir = walks_root / window.name
        logger.info(f"\n=== {window.name} ===  {window.asdict_iso()}")
        win_paths = materialise_window(full_panel, window, win_dir)

        # Baseline arm for this window, computed once for every seed up front:
        # it depends only on the panel and the reset seeds, not on the policy,
        # and one env construction per window beats one per run.
        baseline_by_seed = equal_weight_test_metrics(
            cfg, test_panel=win_paths["test"], seeds=seeds
        )
        for seed, bl in sorted(baseline_by_seed.items()):
            logger.info(
                f"  equal-weight test Sharpe (seed={seed}): "
                f"{bl.get('mean_sharpe', float('nan')):.3f}"
            )

        for seed in seeds:
            logger.info(f"--- {window.name} seed={seed} ---")
            run_cfg = cfg.copy()
            run_cfg["seed"] = seed
            run_tag = f"{cfg.model.name}_{window.name}_seed{seed}"
            ckpt_dir = win_dir / f"checkpoints_seed{seed}"
            stats_path = win_dir / f"feature_stats_seed{seed}.json"
            res = train_one_run(
                run_cfg,
                train_panel=win_paths["train"],
                val_panel=win_paths["val"],
                test_panel=win_paths["test"],
                checkpoint_dir=ckpt_dir,
                mlflow_run_name=run_tag,
                mlflow_experiment=mlflow_experiment,
                mlflow_tags={
                    "window": window.name,
                    "seed": str(seed),
                    "train_start": window.train_start.isoformat(),
                    "train_end": window.train_end.isoformat(),
                    "val_start": window.val_start.isoformat(),
                    "val_end": window.val_end.isoformat(),
                    "test_start": window.test_start.isoformat(),
                    "test_end": window.test_end.isoformat(),
                },
                feature_stats_save_path=stats_path,
            )
            shuffled: dict[str, float] | None = None
            if shuffle_check:
                shuffled = shuffled_ticker_test_metrics(
                    run_cfg,
                    test_panel=win_paths["test"],
                    checkpoint_dir=ckpt_dir,
                    feature_stats_path=stats_path,
                    seed=seed,
                )
                if shuffled is not None:
                    logger.info(
                        f"  shuffled-ticker test Sharpe: "
                        f"{shuffled.get('mean_sharpe', float('nan')):.3f}  "
                        f"(real: {(res.test_metrics or {}).get('mean_sharpe', float('nan')):.3f})"
                    )

            results.append(
                {
                    "window": window.name,
                    "seed": seed,
                    "train": res.train_metrics,
                    "val": res.val_metrics,
                    "test": res.test_metrics,
                    "baseline_test": baseline_by_seed.get(seed),
                    "shuffled_test": shuffled,
                    "run_id": res.mlflow_run_id,
                }
            )

    return results


# ── Aggregation + bootstrap CI ─────────────────────────────────────────────────


def aggregate_walk_forward(
    results: list[dict[str, Any]],
    metric: str = "mean_sharpe",
    split: str = "test",
) -> dict[str, float]:
    """Mean / std / median of one metric across all (window × seed) runs."""
    values = [
        r[split][metric]
        for r in results
        if r.get(split) is not None and metric in r[split]
    ]
    if not values:
        return {}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "n": float(len(arr)),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def paired_bootstrap_ci(
    rl_values: list[float],
    baseline_values: list[float],
    n_boot: int = 10_000,
    alpha: float = 0.05,
    rng_seed: int = 0,
) -> dict[str, float]:
    """Bootstrap a (1-α) CI for `mean(RL) − mean(baseline)` (paired by index).

    Returns mean diff and the [lo, hi] CI.  `paired` means we resample
    indices, so seed/window pairings are preserved — the standard
    walk-forward significance test against a fixed benchmark.
    """
    if len(rl_values) != len(baseline_values):
        raise ValueError("rl/baseline lengths must match for paired bootstrap")
    if not rl_values:
        return {"mean_diff": 0.0, "ci_lo": 0.0, "ci_hi": 0.0}

    rng = np.random.default_rng(rng_seed)
    rl = np.asarray(rl_values, dtype=np.float64)
    bl = np.asarray(baseline_values, dtype=np.float64)
    n = len(rl)
    diffs = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[i] = (rl[idx] - bl[idx]).mean()

    lo, hi = np.quantile(diffs, [alpha / 2.0, 1.0 - alpha / 2.0])
    return {
        "mean_diff": float((rl - bl).mean()),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "p_above_zero": float((diffs > 0).mean()),
    }


# ── Val/test correlation — the headline walk-forward diagnostic ────────────────
#
# Why this lives here and not in a notebook: the central empirical finding of
# this project is that `corr(val_sharpe, test_sharpe)` was *negative* (−0.86 on
# the `mlp_baseline` M7 run), which means selecting a model on validation
# Sharpe actively picked the worst test performers.  Every architecture change
# since (regime FiLM conditioning, the auxiliary return head) is aimed at
# moving that number toward zero.  A figure that decides experiments has to be
# computed by the pipeline that runs them, reproducibly, not by hand.


def _average_ranks(x: np.ndarray) -> np.ndarray:
    """0-based ascending ranks of `x`, with ties collapsed to their mean rank.

    Uses the double-``argsort`` identity (``argsort(argsort(x))`` is the
    ordinal rank vector) followed by a tie-averaging pass.  scipy is
    deliberately not a dependency of this project, so ``rankdata`` is not
    available; and the tie pass genuinely matters here — two runs with
    identical Sharpe would otherwise get an arbitrary relative order that
    leaks the input order into rho.
    """
    n = len(x)
    order = np.argsort(x, kind="stable")
    ranks = np.argsort(order, kind="stable").astype(np.float64)
    sorted_x = x[order]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = (i + j) / 2.0
        i = j + 1
    return ranks


def _pearson_r(x: np.ndarray, y: np.ndarray) -> float | None:
    """Pearson correlation, or ``None`` when either series has zero variance."""
    xd = x - x.mean()
    yd = y - y.mean()
    denom = math.sqrt(float((xd * xd).sum()) * float((yd * yd).sum()))
    if not math.isfinite(denom) or denom <= 0.0:
        return None
    r = float((xd * yd).sum() / denom)
    # Guard against |r| drifting a hair past 1.0 through floating-point error.
    return max(-1.0, min(1.0, r))


def _log_beta(a: float, b: float) -> float:
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function (Lentz's method).

    Straight transcription of Numerical Recipes §6.4.  Needed only because we
    compute p-values without scipy.
    """
    tiny = 1e-30
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, 301):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 3e-16:
            break
    return h


def _incomplete_beta(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta ``I_x(a, b)``."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(a * math.log(x) + b * math.log1p(-x) - _log_beta(a, b))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def _corr_p_value(r: float, n: int) -> float:
    """Two-sided p-value for correlation `r` over `n` observations.

    Uses the usual t-approximation ``t = r * sqrt((n - 2) / (1 - r²))`` on
    ``n - 2`` degrees of freedom.  NOTE: this is *approximate* for the sample
    sizes we actually have.  It is exact for Pearson only under bivariate
    normality, and for Spearman it is an approximation that is known to be
    optimistic in small samples.  A walk-forward gives ~12 (window × seed)
    points, so read these p-values as "is this worth taking seriously" rather
    than as a formal test.
    """
    df = n - 2
    if df <= 0:
        return 1.0
    r2 = min(r * r, 1.0)
    if 1.0 - r2 <= 1e-15:
        return 0.0
    t = r * math.sqrt(df / (1.0 - r2))
    return float(_incomplete_beta(df / 2.0, 0.5, df / (df + t * t)))


def val_test_correlation(
    results: list[dict[str, Any]],
    metric: str = "mean_sharpe",
    val_split: str = "val",
    test_split: str = "test",
) -> dict[str, float]:
    """Correlation between validation and test performance across all runs.

    One (val, test) pair per (window × seed) run.  Returns *both* Pearson and
    Spearman, deliberately:

    * The historical −0.86 baseline is a **Pearson** value, so Pearson is the
      only number directly comparable to it.
    * With ~12 points a single outlier window dominates Pearson entirely.
      **Spearman** answers the question that actually matters for model
      selection — "does ranking by val Sharpe rank correctly on test?" — and is
      robust to that one bad window.

    Reporting only one of them would make it trivially easy to declare victory
    on noise, so both are always returned.

    Returns
    -------
    dict
        ``{n, pearson_r, pearson_p, spearman_rho, spearman_p, degenerate}``,
        or an **empty dict** when fewer than 3 usable pairs exist (a
        correlation over 2 points is either ±1 or undefined — meaningless).
        Runs whose val or test metrics are missing (``None``, absent key, or
        non-finite) are skipped rather than raising.  ``degenerate`` is 1.0
        when a series had zero variance, in which case correlation is
        mathematically undefined and is reported as 0.0 with p = 1.0 rather
        than as NaN — NaN propagates silently through MLflow and JSON.
    """
    xs: list[float] = []
    ys: list[float] = []
    for run in results:
        val = run.get(val_split)
        test = run.get(test_split)
        if not isinstance(val, dict) or not isinstance(test, dict):
            continue
        if metric not in val or metric not in test:
            continue
        v = float(val[metric])
        t = float(test[metric])
        if not (math.isfinite(v) and math.isfinite(t)):
            continue
        xs.append(v)
        ys.append(t)

    n = len(xs)
    if n < 3:
        return {}

    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)

    pearson = _pearson_r(x, y)
    spearman = _pearson_r(_average_ranks(x), _average_ranks(y))
    degenerate = pearson is None or spearman is None

    return {
        "n": float(n),
        "pearson_r": 0.0 if pearson is None else pearson,
        "pearson_p": 1.0 if pearson is None else _corr_p_value(pearson, n),
        "spearman_rho": 0.0 if spearman is None else spearman,
        "spearman_p": 1.0 if spearman is None else _corr_p_value(spearman, n),
        "degenerate": 1.0 if degenerate else 0.0,
    }


def paired_test_values(
    results: list[dict[str, Any]],
    metric: str = "mean_sharpe",
    rl_split: str = "test",
    baseline_split: str = "baseline_test",
) -> tuple[list[float], list[float]]:
    """Index-aligned (RL, baseline) metric values, ready for the paired bootstrap.

    Only runs that have *both* arms contribute, so the two returned lists are
    always the same length and element *i* of each comes from the same
    (window, seed) run — which is exactly what makes
    :func:`paired_bootstrap_ci` a paired test rather than a two-sample one.
    """
    rl: list[float] = []
    bl: list[float] = []
    for run in results:
        a = run.get(rl_split)
        b = run.get(baseline_split)
        if not isinstance(a, dict) or not isinstance(b, dict):
            continue
        if metric not in a or metric not in b:
            continue
        av, bv = float(a[metric]), float(b[metric])
        if not (math.isfinite(av) and math.isfinite(bv)):
            continue
        rl.append(av)
        bl.append(bv)
    return rl, bl
