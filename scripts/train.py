#!/usr/bin/env python
"""Train the RL policy with PPO.

Usage:
    python scripts/train.py train=ppo_baseline seed=42
    python scripts/train.py train=ppo_baseline seed=42 train.total_steps=200000
"""
from __future__ import annotations

from pathlib import Path

import hydra
from loguru import logger
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    import hydra.utils
    import mlflow
    from gymnasium.vector import SyncVectorEnv

    from trader.data.features import FEATURE_COLS
    from trader.data.universe import all_tickers
    from trader.models.actor_critic import ActorCritic, ModelConfig
    from trader.training.eval_metrics import aggregate_metrics
    from trader.training.ppo import PPOConfig, PPOTrainer
    from trader.utils.seeding import get_device, seed_everything

    orig_cwd = Path(hydra.utils.get_original_cwd())
    seed = int(cfg.seed)
    seed_everything(seed)
    device = get_device()
    logger.info(f"Device: {device}  seed: {seed}")

    # ── Panel paths ───────────────────────────────────────────────────────────
    panels_root = orig_cwd / "data" / "panels"
    train_panel = panels_root / "train.parquet"
    val_panel = panels_root / "val.parquet"

    if not train_panel.exists():
        logger.error("train.parquet not found. Run scripts/build_features.py first.")
        return

    universe = all_tickers()
    N = len(universe)

    env_kwargs = dict(
        panel_path=train_panel,
        universe=universe,
        feature_columns=FEATURE_COLS,
        lookback=int(cfg.env.lookback_days),
        episode_length=int(cfg.env.episode_length),
        initial_cash=float(cfg.env.initial_cash),
        turnover_penalty=float(cfg.env.turnover_penalty),
    )

    def make_env(env_seed: int) -> object:
        def _init() -> object:
            from trader.env.panel_env import PanelTradingEnv

            return PanelTradingEnv(**env_kwargs, seed=env_seed)  # type: ignore[arg-type]

        return _init

    n_envs = int(cfg.train.n_envs)
    envs = SyncVectorEnv([make_env(seed + i) for i in range(n_envs)])

    # ── Model ─────────────────────────────────────────────────────────────────
    model_cfg = ModelConfig(
        in_features=len(FEATURE_COLS),
        n_tickers=N,
        embed_dim=int(cfg.model.embed_dim),
        kernel_size=int(cfg.model.tcn.kernel_size),
        dropout=float(cfg.model.tcn.dropout),
    )
    model = ActorCritic(model_cfg)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: {n_params:,} parameters")

    # ── PPO config ────────────────────────────────────────────────────────────
    ppo_cfg = PPOConfig(
        total_steps=int(cfg.train.total_steps),
        n_envs=n_envs,
        n_steps=int(cfg.train.n_steps),
        n_epochs=int(cfg.train.n_epochs),
        n_minibatches=int(cfg.train.n_minibatches),
        gamma=float(cfg.train.gamma),
        gae_lambda=float(cfg.train.gae_lambda),
        clip_coef=float(cfg.train.clip_coef),
        ent_coef=float(cfg.train.ent_coef),
        vf_coef=float(cfg.train.vf_coef),
        max_grad_norm=float(cfg.train.max_grad_norm),
        learning_rate=float(cfg.train.learning_rate),
        anneal_lr=bool(cfg.train.anneal_lr),
        log_interval=int(cfg.train.log_interval),
        eval_interval=int(cfg.train.eval_interval),
        checkpoint_interval=int(cfg.train.checkpoint_interval),
        checkpoint_dir=Path(hydra.utils.get_original_cwd()) / "checkpoints",
    )

    # ── MLflow run ────────────────────────────────────────────────────────────
    mlflow.set_tracking_uri(
        f"http://localhost:{cfg.get('mlflow_port', 5555)}"
    )
    mlflow.set_experiment("trading_bot")

    with mlflow.start_run(run_name=f"ppo_baseline_seed{seed}") as run:
        from omegaconf import OmegaConf

        mlflow.log_params(
            {
                "seed": seed,
                "n_tickers": N,
                "n_params": n_params,
                "device": str(device),
                **{f"train.{k}": v for k, v in dict(cfg.train).items()},
                **{
                    f"model.{k}": v
                    for k, v in dict(cfg.model).items()
                    if not isinstance(v, DictConfig)
                },
            }
        )
        mlflow.log_text(OmegaConf.to_yaml(cfg), "config.yaml")

        trainer = PPOTrainer(envs, model, ppo_cfg, device, run.info.run_id)
        episode_metrics = trainer.train()

        # ── Final aggregated metrics ──────────────────────────────────────────
        if episode_metrics:
            agg = aggregate_metrics(episode_metrics)
            mlflow.log_metrics({k: v for k, v in agg.items()})
            logger.info("Training complete. Aggregated metrics:")
            for k, v in sorted(agg.items()):
                logger.info(f"  {k}: {v:.4f}")

        # ── Overfitting guardrails ────────────────────────────────────────────
        _check_guardrails(episode_metrics)

        # ── Val evaluation ────────────────────────────────────────────────────
        if val_panel.exists():
            val_metrics = _evaluate_split(
                val_panel, universe, FEATURE_COLS, model, device, env_kwargs, seed
            )
            if val_metrics:
                agg_val = aggregate_metrics(val_metrics)
                mlflow.log_metrics({f"val_{k}": v for k, v in agg_val.items()})
                logger.info(f"Val Sharpe: {agg_val.get('mean_sharpe', 0):.3f}")

    envs.close()
    logger.info(f"MLflow run: {run.info.run_id}")


# ── helpers ────────────────────────────────────────────────────────────────────


def _evaluate_split(
    panel_path: Path,
    universe: list[str],
    feature_cols: list[str],
    model: object,
    device: object,
    base_kwargs: dict[str, object],
    seed: int,
    n_episodes: int = 5,
) -> list[object]:
    import torch

    from trader.env.panel_env import PanelTradingEnv
    from trader.training.eval_metrics import compute_episode_metrics

    env = PanelTradingEnv(
        panel_path=panel_path,
        universe=universe,  # type: ignore[arg-type]
        feature_columns=feature_cols,
        lookback=int(base_kwargs.get("lookback", 60)),
        episode_length=int(base_kwargs.get("episode_length", 252)),
        initial_cash=float(base_kwargs.get("initial_cash", 1_000_000.0)),
        seed=seed + 999,
    )
    metrics = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + 999 + ep)
        nav_series = [float(obs["nav"])]
        turnovers = []
        done = False
        with torch.no_grad():
            while not done:
                obs_t = {k: torch.tensor(v, device=device).unsqueeze(0) for k, v in obs.items()}  # type: ignore[arg-type]
                action, _, _, _ = model.get_action_and_value(obs_t)  # type: ignore[union-attr]
                a_np = action.squeeze(0).cpu().numpy()
                obs, _, terminated, truncated, info = env.step(a_np)
                nav_series.append(float(info["nav"]))
                turnovers.append(float(info["turnover"]))
                done = terminated or truncated
        metrics.append(compute_episode_metrics(nav_series, turnovers))
    return metrics


def _check_guardrails(episode_metrics: list[object]) -> None:
    from trader.training.eval_metrics import EpisodeMetrics

    episodes = [m for m in episode_metrics if isinstance(m, EpisodeMetrics)]
    if not episodes:
        return

    import numpy as np

    turnovers = [e.turnover_ann for e in episodes]

    mean_turnover = float(np.mean(turnovers))
    if mean_turnover < 0.1:
        logger.warning(
            f"GUARDRAIL: turnover collapsed to {mean_turnover:.2f}/yr — cash-hoarding?"
        )
    if mean_turnover > 20.0:
        logger.warning(
            f"GUARDRAIL: turnover exploded to {mean_turnover:.2f}/yr — churning?"
        )


if __name__ == "__main__":
    main()
