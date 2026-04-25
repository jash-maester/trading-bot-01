#!/usr/bin/env python
"""Evaluate all baseline agents on train / val / test panels.

Establishes the reference bar that RL runs must beat.
Logs results to MLflow under the experiment "baselines".

Usage:
    uv run python scripts/evaluate.py                   # all splits
    uv run python scripts/evaluate.py split=val         # val only
    uv run python scripts/evaluate.py n_episodes=20     # more episodes
"""
from __future__ import annotations

import hydra
from loguru import logger
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    from pathlib import Path

    import hydra.utils
    import mlflow

    from trader.data.features import FEATURE_COLS
    from trader.data.universe import all_tickers
    from trader.env.baselines import (
        BuyAndHoldIndex,
        EqualWeightRebalanced,
        MomentumTopK,
        RandomPolicy,
        SixtyFortyCash,
    )
    from trader.env.panel_env import PanelTradingEnv
    from trader.training.eval_metrics import (
        EpisodeMetrics,
        aggregate_metrics,
        compute_episode_metrics,
    )

    orig_cwd = Path(hydra.utils.get_original_cwd())
    panels_root = orig_cwd / "data" / "panels"
    universe = all_tickers()

    n_episodes: int = int(cfg.get("n_episodes", 10))
    target_split: str = str(cfg.get("split", "all"))

    splits = {
        "train": panels_root / "train.parquet",
        "val":   panels_root / "val.parquet",
        "test":  panels_root / "test.parquet",
    }
    if target_split != "all":
        splits = {target_split: splits[target_split]}

    baselines = {
        "equal_weight":   EqualWeightRebalanced(),
        "buy_and_hold":   BuyAndHoldIndex(),
        "momentum_top5":  MomentumTopK(k=5),
        "sixty_forty":    SixtyFortyCash(),
        "random":         RandomPolicy(seed=0),
    }

    env_base = dict(
        universe=universe,
        feature_columns=FEATURE_COLS,
        lookback=int(cfg.env.lookback_days),
        episode_length=int(cfg.env.episode_length),
        initial_cash=float(cfg.env.initial_cash),
    )

    mlflow.set_tracking_uri(f"http://localhost:{cfg.get('mlflow_port', 5555)}")
    mlflow.set_experiment("baselines")

    # ── header ────────────────────────────────────────────────────────────────
    header = f"{'Agent':<20} {'Split':<6} {'Sharpe':>8} {'CAGR':>8} {'MDD':>8} {'Turn':>8}"
    separator = "-" * len(header)
    logger.info("\n" + separator)
    logger.info(header)
    logger.info(separator)

    for split_name, panel_path in splits.items():
        if not panel_path.exists():
            logger.warning(f"{panel_path} not found, skipping.")
            continue

        for agent_name, agent in baselines.items():
            env = PanelTradingEnv(panel_path=panel_path, **env_base, seed=42)  # type: ignore[arg-type]
            episode_metrics: list[EpisodeMetrics] = []

            for ep in range(n_episodes):
                obs, _ = env.reset(seed=42 + ep)
                agent.reset()
                nav_series = [float(obs["nav"])]
                turnovers: list[float] = []
                done = False

                while not done:
                    action = agent.act(obs)
                    obs, _, terminated, truncated, info = env.step(action)
                    nav_series.append(float(info["nav"]))
                    turnovers.append(float(info["turnover"]))
                    done = terminated or truncated

                episode_metrics.append(
                    compute_episode_metrics(nav_series, turnovers)
                )

            agg = aggregate_metrics(episode_metrics)
            sharpe = agg.get("mean_sharpe", 0.0)
            cagr   = agg.get("mean_cagr",   0.0)
            mdd    = agg.get("mean_max_drawdown", 0.0)
            turn   = agg.get("mean_turnover_ann", 0.0)

            logger.info(
                f"{agent_name:<20} {split_name:<6}"
                f" {sharpe:>8.3f} {cagr:>8.3f} {mdd:>8.3f} {turn:>8.3f}"
            )

            # Log to MLflow
            with mlflow.start_run(run_name=f"{agent_name}_{split_name}"):
                mlflow.log_params({
                    "agent": agent_name,
                    "split": split_name,
                    "n_episodes": n_episodes,
                    "universe_size": len(universe),
                })
                mlflow.log_metrics({
                    f"{split_name}/sharpe":       sharpe,
                    f"{split_name}/cagr":         cagr,
                    f"{split_name}/max_drawdown": mdd,
                    f"{split_name}/turnover_ann": turn,
                    f"{split_name}/sortino":      agg.get("mean_sortino", 0.0),
                    f"{split_name}/calmar":       agg.get("mean_calmar", 0.0),
                })

    logger.info(separator)
    logger.info(
        "Baselines logged to MLflow. "
        "RL agent must beat 'equal_weight' on the same split to pass M5."
    )


if __name__ == "__main__":
    main()
