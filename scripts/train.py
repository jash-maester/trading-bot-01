#!/usr/bin/env python
"""Train the RL policy with PPO.

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
    import mlflow
    import torch.nn as nn
    from gymnasium.vector import SyncVectorEnv

    from trader.data.feature_stats import (
        compute_feature_stats,
        save_stats,
        stats_to_tensors,
    )
    from trader.data.features import FEATURE_COLS
    from trader.data.universe import all_tickers
    from trader.env.reward import DifferentialSharpe, ExcessLogReturn, LogReturn
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

    # ── Feature normalisation stats ───────────────────────────────────────────
    # Computed once from the TRAIN panel so the model sees zero-mean unit-var
    # inputs.  Without this, conv weights are dominated by the largest-scale
    # features (dollar_volume_20 ≈ 1e9, atr_14 ≈ 100) and the model is blind
    # to the small-scale signals (log returns, RSI, vol).  Stats are saved
    # alongside the panel for reproducibility.
    feat_stats = compute_feature_stats(train_panel, FEATURE_COLS)
    save_stats(feat_stats, panels_root / "train.feature_stats.json")
    feat_mean, feat_std = stats_to_tensors(feat_stats, FEATURE_COLS)
    logger.info(
        f"Feature stats: mean range [{feat_mean.min().item():.4g}, "
        f"{feat_mean.max().item():.4g}]  std range "
        f"[{feat_std.min().item():.4g}, {feat_std.max().item():.4g}]"
    )

    # ── Reward function (read from cfg.env.reward; default: log_return) ────────
    _REWARD_MAP = {
        "differential_sharpe": DifferentialSharpe,
        "log_return": LogReturn,
        "excess_log_return": ExcessLogReturn,
    }
    reward_name = str(cfg.env.get("reward", "log_return"))
    reward_fn = _REWARD_MAP.get(reward_name, LogReturn)()
    use_excess = bool(cfg.env.get("use_excess_returns", False))
    # `excess_log_return` reward implies `use_excess_returns=True` even if
    # the env flag is left at its default — keep the two consistent so a
    # config typo can't silently produce the wrong reward.
    if reward_name == "excess_log_return":
        use_excess = True
    logger.info(
        f"Reward function: {reward_name} ({type(reward_fn).__name__})"
        f"  excess={use_excess}"
    )

    env_kwargs = dict(
        panel_path=train_panel,
        universe=universe,
        feature_columns=FEATURE_COLS,
        lookback=int(cfg.env.lookback_days),
        episode_length=int(cfg.env.episode_length),
        initial_cash=float(cfg.env.initial_cash),
        turnover_penalty=float(cfg.env.turnover_penalty),
        reward_fn=reward_fn,
        use_excess_returns=use_excess,
    )

    def make_env(env_seed: int) -> object:
        def _init() -> object:
            from trader.env.panel_env import PanelTradingEnv

            return PanelTradingEnv(**env_kwargs, seed=env_seed)  # type: ignore[arg-type]

        return _init

    n_envs = int(cfg.train.n_envs)
    envs = SyncVectorEnv([make_env(seed + i) for i in range(n_envs)])

    # ── Model ─────────────────────────────────────────────────────────────────
    num_channels_raw = cfg.model.tcn.get("num_channels")
    num_channels = (
        [int(c) for c in num_channels_raw] if num_channels_raw is not None else None
    )
    model_cfg = ModelConfig(
        in_features=len(FEATURE_COLS),
        n_tickers=N,
        embed_dim=int(cfg.model.embed_dim),
        num_channels=num_channels,
        kernel_size=int(cfg.model.tcn.kernel_size),
        dropout=float(cfg.model.tcn.dropout),
    )

    model: nn.Module
    use_graph = bool(cfg.model.get("use_graph", False))
    if use_graph:
        from trader.models.graph import GNNActorCritic, GNNConfig

        gc = cfg.model.graph
        gnn_cfg = GNNConfig(
            num_sectors=int(gc.get("num_sectors", 8)),
            num_layers=int(gc.get("layers", 2)),
            num_heads=int(gc.get("num_heads", 2)),
            dropout=float(gc.get("dropout", 0.1)),
            drop_edge_prob=float(gc.get("drop_edge_prob", 0.1)),
            relations=str(gc.get("relations", "all")),
        )
        model = GNNActorCritic(model_cfg, gnn_cfg, feat_mean=feat_mean, feat_std=feat_std)
        logger.info(f"Model: GNNActorCritic (relations={gnn_cfg.relations})")
    else:
        model = ActorCritic(model_cfg, feat_mean=feat_mean, feat_std=feat_std)
        logger.info("Model: ActorCritic (no graph)")

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Parameters: {n_params:,}")

    # ── PPO config ────────────────────────────────────────────────────────────
    # model_name is resolved here (before PPOConfig) so the checkpoint dir is
    # namespaced per run — parallel runs must never share a checkpoint dir.
    model_name = str(cfg.model.get("name", "ppo"))
    run_tag = f"{model_name}_seed{seed}"
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
        normalize_rewards=bool(cfg.train.get("normalize_rewards", True)),
        log_interval=int(cfg.train.log_interval),
        eval_interval=int(cfg.train.eval_interval),
        checkpoint_interval=int(cfg.train.checkpoint_interval),
        checkpoint_dir=Path(hydra.utils.get_original_cwd()) / "checkpoints" / run_tag,
    )

    # ── MLflow run ────────────────────────────────────────────────────────────
    mlflow.set_tracking_uri(
        f"http://localhost:{cfg.get('mlflow_port', 5555)}"
    )
    mlflow.set_experiment("trading_bot")
    with mlflow.start_run(run_name=f"{model_name}_seed{seed}") as run:
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
