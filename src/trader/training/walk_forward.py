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

A 1-month purge gap is enforced between consecutive segments to prevent
rolling-window features from leaking across the boundary.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from loguru import logger
from omegaconf import DictConfig

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


def compute_windows(
    data_start: date,
    data_end: date,
    train_years: int = 5,
    val_months: int = 12,
    test_months: int = 12,
    purge_months: int = 1,
    n_windows: int = 4,
    step_months: int = 12,
) -> list[WindowConfig]:
    """Generate sliding walk-forward windows.

    Each window has the structure:
        ┌──────── train_years ────────┐  purge  ┌── val_months ──┐  purge  ┌── test_months ──┐

    Windows are advanced by `step_months` (default 12 — one window per
    calendar year of test data).  Stops when either `n_windows` is
    reached or a window would extend past `data_end`.
    """
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


def materialise_window(
    full_panel: pl.DataFrame,
    window: WindowConfig,
    out_dir: Path,
) -> dict[str, Path]:
    """Slice `full_panel` for the given window and write three parquet files.

    Returns a dict ``{'train': path, 'val': path, 'test': path}``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    segments = (
        ("train", window.train_start, window.train_end),
        ("val", window.val_start, window.val_end),
        ("test", window.test_start, window.test_end),
    )
    for name, start, end in segments:
        sub = slice_panel(full_panel, start, end)
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
    return paths


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
          "window":  "W1",
          "seed":    42,
          "train":   {...aggregated train metrics...},
          "val":     {...aggregated val metrics...},
          "test":    {...aggregated test metrics...},
          "run_id":  "<mlflow id>",
        }
    """
    from trader.training.runner import train_one_run

    results: list[dict[str, Any]] = []
    for window in windows:
        win_dir = walks_root / window.name
        logger.info(f"\n=== {window.name} ===  {window.asdict_iso()}")
        win_paths = materialise_window(full_panel, window, win_dir)

        for seed in seeds:
            logger.info(f"--- {window.name} seed={seed} ---")
            run_cfg = cfg.copy()
            run_cfg["seed"] = seed
            run_tag = f"{cfg.model.name}_{window.name}_seed{seed}"
            res = train_one_run(
                run_cfg,
                train_panel=win_paths["train"],
                val_panel=win_paths["val"],
                test_panel=win_paths["test"],
                checkpoint_dir=walks_root / window.name / f"checkpoints_seed{seed}",
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
                feature_stats_save_path=win_dir / f"feature_stats_seed{seed}.json",
            )
            results.append(
                {
                    "window": window.name,
                    "seed": seed,
                    "train": res.train_metrics,
                    "val": res.val_metrics,
                    "test": res.test_metrics,
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
