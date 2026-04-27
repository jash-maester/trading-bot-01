#!/usr/bin/env python
"""Walk-forward training driver — M7.

Runs the configured model across multiple sliding windows × seeds and
reports per-window metrics + aggregated bootstrap CI vs equal-weight.

Usage:
    # Default 4 windows × 3 seeds for one architecture:
    uv run python scripts/walk_forward.py model=gnn_intra_only

    # Custom seeds / windows:
    uv run python scripts/walk_forward.py model=gnn_v1 \
        walk.seeds=[42,43,44,45,46] walk.n_windows=4

The full historical panel is read from the existing train/val/test
parquets and re-sliced into walk-forward segments.  Per-window panels
are written under ``data/walks/<W>/{train,val,test}.parquet``.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import hydra
from loguru import logger
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    import hydra.utils
    import polars as pl

    from trader.training.walk_forward import (
        aggregate_walk_forward,
        compute_windows,
        run_walk_forward,
    )

    orig_cwd = Path(hydra.utils.get_original_cwd())
    panels_root = orig_cwd / "data" / "panels"
    walks_root = orig_cwd / "data" / "walks"
    walks_root.mkdir(parents=True, exist_ok=True)

    # ── Load and concatenate the full historical panel ────────────────────────
    panel_paths = [
        panels_root / "train.parquet",
        panels_root / "val.parquet",
        panels_root / "test.parquet",
    ]
    available = [p for p in panel_paths if p.exists()]
    if not available:
        logger.error("No panels found under data/panels/. Run build_features.py first.")
        return
    full_panel = pl.concat([pl.read_parquet(p) for p in available]).unique(
        subset=["date", "ticker"], keep="first"
    ).sort(["date", "ticker"])
    logger.info(
        f"Full panel: {full_panel.shape[0]:,} rows, "
        f"{full_panel['date'].min()}..{full_panel['date'].max()}"
    )

    # ── Walk-forward config (with sensible defaults if not in YAML) ───────────
    walk_cfg = cfg.get("walk", {})
    n_windows = int(walk_cfg.get("n_windows", 4))
    train_years = int(walk_cfg.get("train_years", 5))
    val_months = int(walk_cfg.get("val_months", 12))
    test_months = int(walk_cfg.get("test_months", 12))
    purge_months = int(walk_cfg.get("purge_months", 1))
    step_months = int(walk_cfg.get("step_months", 12))
    seeds_list = list(walk_cfg.get("seeds", [42, 43, 44]))

    data_start = full_panel["date"].min()
    data_end = full_panel["date"].max()
    if not isinstance(data_start, date):
        data_start = date.fromisoformat(str(data_start))
    if not isinstance(data_end, date):
        data_end = date.fromisoformat(str(data_end))

    windows = compute_windows(
        data_start=data_start,
        data_end=data_end,
        train_years=train_years,
        val_months=val_months,
        test_months=test_months,
        purge_months=purge_months,
        n_windows=n_windows,
        step_months=step_months,
    )
    logger.info(f"Computed {len(windows)} windows:")
    for w in windows:
        logger.info(f"  {w.name}: {w.asdict_iso()}")

    if not windows:
        logger.error("No windows fit in the available data — check date ranges.")
        return

    # ── Drive ─────────────────────────────────────────────────────────────────
    model_name = str(cfg.model.get("name", "ppo"))
    results = run_walk_forward(
        cfg,
        full_panel=full_panel,
        windows=windows,
        seeds=seeds_list,
        walks_root=walks_root,
        mlflow_experiment=f"walk_forward_{model_name}",
    )

    # ── Aggregate + dump JSON for downstream analysis ─────────────────────────
    summary = {
        "model": model_name,
        "n_windows": len(windows),
        "n_seeds": len(seeds_list),
        "windows": [w.asdict_iso() for w in windows],
        "test_mean_sharpe": aggregate_walk_forward(results, "mean_sharpe", "test"),
        "test_mean_cagr": aggregate_walk_forward(results, "mean_cagr", "test"),
        "test_mean_max_drawdown": aggregate_walk_forward(
            results, "mean_max_drawdown", "test"
        ),
        "val_mean_sharpe": aggregate_walk_forward(results, "mean_sharpe", "val"),
        "per_run": [
            {
                "window": r["window"],
                "seed": r["seed"],
                "train_sharpe": (r["train"] or {}).get("mean_sharpe"),
                "val_sharpe": (r["val"] or {}).get("mean_sharpe"),
                "test_sharpe": (r["test"] or {}).get("mean_sharpe"),
            }
            for r in results
        ],
    }

    out_path = walks_root / f"summary_{model_name}.json"
    with out_path.open("w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info("\n=== Walk-forward summary ===")
    logger.info(f"Model: {model_name}")
    logger.info(f"Test mean_sharpe (across {summary['test_mean_sharpe'].get('n', 0):.0f} runs):")
    for k, v in summary["test_mean_sharpe"].items():
        logger.info(f"  {k}: {v:.4f}")
    logger.info(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
