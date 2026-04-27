#!/usr/bin/env python
"""Train the RL policy with PPO (single window).

Thin Hydra wrapper around :func:`trader.training.runner.train_one_run`.
For multi-window walk-forward training see ``scripts/walk_forward.py``.

Usage:
    # No-graph MLP baseline (M5):
    python scripts/train.py model=mlp_baseline seed=42

    # Full hetero-GNN (M6):
    python scripts/train.py model=gnn_v1 seed=42

    # Intra-sector-only GNN ablation (M6):
    python scripts/train.py model=gnn_intra_only seed=42
"""
from __future__ import annotations

from pathlib import Path

import hydra
from loguru import logger
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    import hydra.utils

    from trader.training.runner import train_one_run

    orig_cwd = Path(hydra.utils.get_original_cwd())
    panels_root = orig_cwd / "data" / "panels"
    train_panel = panels_root / "train.parquet"
    val_panel = panels_root / "val.parquet"

    if not train_panel.exists():
        logger.error("train.parquet not found. Run scripts/build_features.py first.")
        return

    seed = int(cfg.seed)
    model_name = str(cfg.model.get("name", "ppo"))
    run_tag = f"{model_name}_seed{seed}"

    result = train_one_run(
        cfg,
        train_panel=train_panel,
        val_panel=val_panel if val_panel.exists() else None,
        checkpoint_dir=orig_cwd / "checkpoints" / run_tag,
        mlflow_run_name=run_tag,
        mlflow_experiment="trading_bot",
        feature_stats_save_path=panels_root / "train.feature_stats.json",
    )
    logger.info(f"MLflow run: {result.mlflow_run_id}")


if __name__ == "__main__":
    main()
