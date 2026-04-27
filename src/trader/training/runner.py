"""Reusable PPO training runner.

Extracted from `scripts/train.py` so that both single-window training
(`scripts/train.py`) and walk-forward training (`scripts/walk_forward.py`)
can share a single tested code path.

The function takes explicit panel paths and a Hydra config, returns
aggregated train/val/test metrics, and logs everything to MLflow.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch.nn as nn
from gymnasium.vector import SyncVectorEnv
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from trader.data.feature_stats import (
    compute_feature_stats,
    save_stats,
    stats_to_tensors,
)
from trader.data.features import FEATURE_COLS
from trader.data.universe import all_tickers
from trader.env.reward import DifferentialSharpe, ExcessLogReturn, LogReturn
from trader.models.actor_critic import ActorCritic, ModelConfig
from trader.training.eval_metrics import EpisodeMetrics, aggregate_metrics
from trader.training.ppo import PPOConfig, PPOTrainer
from trader.utils.seeding import get_device, seed_everything


@dataclass
class RunResult:
    """Aggregated outputs of a single training run."""

    mlflow_run_id: str
    train_metrics: dict[str, float]
    val_metrics: dict[str, float] | None
    test_metrics: dict[str, float] | None


_REWARD_MAP: dict[str, type] = {
    "differential_sharpe": DifferentialSharpe,
    "log_return": LogReturn,
    "excess_log_return": ExcessLogReturn,
}


def train_one_run(
    cfg: DictConfig,
    *,
    train_panel: Path,
    val_panel: Path | None = None,
    test_panel: Path | None = None,
    checkpoint_dir: Path | None = None,
    mlflow_run_name: str | None = None,
    mlflow_experiment: str = "trading_bot",
    mlflow_tags: dict[str, str] | None = None,
    feature_stats_save_path: Path | None = None,
) -> RunResult:
    """Execute one full PPO training run and evaluate on val/test panels.

    Parameters
    ----------
    cfg:
        Hydra config (must have `seed`, `train.*`, `model.*`, `env.*` groups).
    train_panel:
        Path to the training parquet panel.  Feature normalisation stats
        are computed from this panel.
    val_panel, test_panel:
        Optional evaluation panels.  Each gets its own evaluation split
        and metrics are logged to MLflow under `val_*` / `test_*` prefixes.
    checkpoint_dir:
        Override for checkpoint output dir.  Default
        `<orig_cwd>/checkpoints/<model_name>_seed<seed>`.
    mlflow_run_name, mlflow_experiment, mlflow_tags:
        MLflow controls.  `mlflow_tags` is useful for walk-forward to
        tag each window run.
    feature_stats_save_path:
        If provided, save the computed (mean, std) JSON to this path.

    Returns
    -------
    RunResult
        Aggregated metrics for train / val / test plus the MLflow run id.
    """
    import mlflow

    seed = int(cfg.seed)
    seed_everything(seed)
    device = get_device()
    logger.info(f"Device: {device}  seed: {seed}  train_panel: {train_panel}")

    if not train_panel.exists():
        raise FileNotFoundError(f"train panel not found: {train_panel}")

    universe = all_tickers()
    n_tickers = len(universe)

    # ── Feature normalisation stats (from train panel only) ───────────────────
    feat_stats = compute_feature_stats(train_panel, FEATURE_COLS)
    if feature_stats_save_path is not None:
        save_stats(feat_stats, feature_stats_save_path)
    feat_mean, feat_std = stats_to_tensors(feat_stats, FEATURE_COLS)
    logger.info(
        f"Feature stats: mean range [{feat_mean.min().item():.4g}, "
        f"{feat_mean.max().item():.4g}]  std range "
        f"[{feat_std.min().item():.4g}, {feat_std.max().item():.4g}]"
    )

    # ── Reward function ───────────────────────────────────────────────────────
    reward_name = str(cfg.env.get("reward", "log_return"))
    reward_fn = _REWARD_MAP.get(reward_name, LogReturn)()
    use_excess = bool(cfg.env.get("use_excess_returns", False))
    if reward_name == "excess_log_return":
        use_excess = True
    logger.info(
        f"Reward: {reward_name} ({type(reward_fn).__name__})  excess={use_excess}"
    )

    env_kwargs: dict[str, Any] = dict(
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

    def _make_env(env_seed: int) -> Any:
        def _init() -> Any:
            from trader.env.panel_env import PanelTradingEnv

            return PanelTradingEnv(**env_kwargs, seed=env_seed)

        return _init

    n_envs = int(cfg.train.n_envs)
    envs = SyncVectorEnv([_make_env(seed + i) for i in range(n_envs)])

    # ── Model ─────────────────────────────────────────────────────────────────
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
        model = GNNActorCritic(
            model_cfg, gnn_cfg, feat_mean=feat_mean, feat_std=feat_std
        )
        logger.info(f"Model: GNNActorCritic (relations={gnn_cfg.relations})")
    else:
        model = ActorCritic(model_cfg, feat_mean=feat_mean, feat_std=feat_std)
        logger.info("Model: ActorCritic (no graph)")

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Parameters: {n_params:,}")

    # ── PPO config ────────────────────────────────────────────────────────────
    model_name = str(cfg.model.get("name", "ppo"))
    run_tag = mlflow_run_name or f"{model_name}_seed{seed}"
    if checkpoint_dir is None:
        checkpoint_dir = Path("checkpoints") / run_tag
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
        checkpoint_dir=checkpoint_dir,
    )

    # ── MLflow ────────────────────────────────────────────────────────────────
    mlflow.set_tracking_uri(f"http://localhost:{cfg.get('mlflow_port', 5555)}")
    mlflow.set_experiment(mlflow_experiment)

    with mlflow.start_run(run_name=run_tag) as run:
        if mlflow_tags:
            mlflow.set_tags(mlflow_tags)
        mlflow.log_params(
            {
                "seed": seed,
                "n_tickers": n_tickers,
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

        train_agg: dict[str, float] = {}
        if episode_metrics:
            train_agg = aggregate_metrics(episode_metrics)
            mlflow.log_metrics({k: v for k, v in train_agg.items()})
            logger.info("Training complete. Aggregated train metrics:")
            for k, v in sorted(train_agg.items()):
                logger.info(f"  {k}: {v:.4f}")

        _check_guardrails(episode_metrics)

        # ── Val / Test eval ───────────────────────────────────────────────────
        val_agg: dict[str, float] | None = None
        test_agg: dict[str, float] | None = None

        if val_panel is not None and val_panel.exists():
            val_metrics = _evaluate_split(
                val_panel, universe, FEATURE_COLS, model, device, env_kwargs, seed
            )
            if val_metrics:
                val_agg = aggregate_metrics(val_metrics)
                mlflow.log_metrics({f"val_{k}": v for k, v in val_agg.items()})
                logger.info(f"Val Sharpe: {val_agg.get('mean_sharpe', 0):.3f}")

        if test_panel is not None and test_panel.exists():
            test_metrics = _evaluate_split(
                test_panel, universe, FEATURE_COLS, model, device, env_kwargs,
                seed + 1,  # different seed for test to avoid val/test leakage
            )
            if test_metrics:
                test_agg = aggregate_metrics(test_metrics)
                mlflow.log_metrics({f"test_{k}": v for k, v in test_agg.items()})
                logger.info(f"Test Sharpe: {test_agg.get('mean_sharpe', 0):.3f}")

        envs.close()
        return RunResult(
            mlflow_run_id=run.info.run_id,
            train_metrics=train_agg,
            val_metrics=val_agg,
            test_metrics=test_agg,
        )


# ── helpers ────────────────────────────────────────────────────────────────────


def _evaluate_split(
    panel_path: Path,
    universe: list[str],
    feature_cols: list[str],
    model: nn.Module,
    device: torch.device,  # type: ignore[name-defined]  # noqa: F821
    base_kwargs: dict[str, Any],
    seed: int,
    n_episodes: int = 5,
) -> list[EpisodeMetrics]:
    import torch

    from trader.env.panel_env import PanelTradingEnv
    from trader.training.eval_metrics import compute_episode_metrics

    env = PanelTradingEnv(
        panel_path=panel_path,
        universe=universe,
        feature_columns=feature_cols,
        lookback=int(base_kwargs.get("lookback", 60)),
        episode_length=int(base_kwargs.get("episode_length", 252)),
        initial_cash=float(base_kwargs.get("initial_cash", 1_000_000.0)),
        seed=seed + 999,
    )
    metrics: list[EpisodeMetrics] = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + 999 + ep)
        nav_series = [float(obs["nav"])]
        turnovers: list[float] = []
        done = False
        with torch.no_grad():
            while not done:
                obs_t = {
                    k: torch.tensor(v, device=device).unsqueeze(0)
                    for k, v in obs.items()
                }
                action, _, _, _ = model.get_action_and_value(obs_t)  # type: ignore[operator]
                a_np = action.squeeze(0).cpu().numpy()
                obs, _, terminated, truncated, info = env.step(a_np)
                nav_series.append(float(info["nav"]))
                turnovers.append(float(info["turnover"]))
                done = terminated or truncated
        metrics.append(compute_episode_metrics(nav_series, turnovers))
    return metrics


def _check_guardrails(episode_metrics: list[Any]) -> None:
    import numpy as np

    episodes = [m for m in episode_metrics if isinstance(m, EpisodeMetrics)]
    if not episodes:
        return

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
