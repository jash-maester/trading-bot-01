"""CleanRL-style single-file PPO for PanelTradingEnv."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from loguru import logger

from trader.training.eval_metrics import EpisodeMetrics, compute_episode_metrics


@dataclass
class PPOConfig:
    total_steps: int = 200_000
    n_envs: int = 8
    n_steps: int = 256          # rollout length per env
    n_epochs: int = 4
    n_minibatches: int = 8
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    learning_rate: float = 3e-4
    anneal_lr: bool = True
    target_kl: float | None = 0.02
    normalize_advantage: bool = True
    checkpoint_dir: Path = field(default_factory=lambda: Path("checkpoints"))
    eval_interval: int = 50       # update iterations between evals
    checkpoint_interval: int = 100
    log_interval: int = 10


def _obs_to_tensor(
    obs: dict[str, np.ndarray] | list[dict[str, np.ndarray]],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Stack vectorized obs list and move to device."""
    if isinstance(obs, list):
        result: dict[str, torch.Tensor] = {}
        for key in obs[0]:
            arrays = [o[key] for o in obs]
            result[key] = torch.tensor(np.stack(arrays), device=device)
        return result
    return {k: torch.tensor(v, device=device) for k, v in obs.items()}


def _batch_obs(
    obs_buf: list[dict[str, np.ndarray]],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Flatten (n_steps, n_envs) obs buffer into (n_steps*n_envs,) tensors."""
    if not obs_buf:
        return {}
    keys = obs_buf[0].keys()
    result: dict[str, torch.Tensor] = {}
    for k in keys:
        arrays = [step[k] for step in obs_buf]
        # arrays: list of (n_envs, ...) → stack → (n_steps, n_envs, ...)
        stacked = np.stack(arrays, axis=0)
        # flatten first two dims
        shape = stacked.shape
        flat = stacked.reshape(shape[0] * shape[1], *shape[2:])
        result[k] = torch.tensor(flat, device=device)
    return result


def compute_gae(
    rewards: np.ndarray,      # [T, E]
    values: np.ndarray,       # [T+1, E]  (last = bootstrap)
    dones: np.ndarray,        # [T, E]
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Generalized advantage estimation. Returns (advantages, returns), both [T, E]."""
    T, E = rewards.shape
    advantages = np.zeros_like(rewards)
    last_gae = np.zeros(E, dtype=np.float32)

    for t in reversed(range(T)):
        next_non_term = 1.0 - dones[t]
        delta = rewards[t] + gamma * values[t + 1] * next_non_term - values[t]
        last_gae = delta + gamma * gae_lambda * next_non_term * last_gae
        advantages[t] = last_gae

    returns = advantages + values[:T]
    return advantages, returns


class PPOTrainer:
    """PPO training loop for PanelTradingEnv."""

    def __init__(
        self,
        envs: Any,                           # vectorized gymnasium env
        model: nn.Module,
        cfg: PPOConfig,
        device: torch.device,
        mlflow_run_id: str | None = None,
    ) -> None:
        self.envs = envs
        self.model = model.to(device)
        self.cfg = cfg
        self.device = device
        self._mlflow_run_id = mlflow_run_id

        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=cfg.learning_rate, eps=1e-5
        )
        cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self._n_envs = cfg.n_envs
        self._batch_size = cfg.n_steps * cfg.n_envs
        self._minibatch_size = self._batch_size // cfg.n_minibatches

    def train(self) -> list[EpisodeMetrics]:
        cfg = self.cfg
        n_updates = cfg.total_steps // self._batch_size
        global_step = 0
        episode_metrics: list[EpisodeMetrics] = []
        start_time = time.time()

        obs_np, _ = self.envs.reset()
        obs = _obs_to_tensor(obs_np, self.device)
        done = np.zeros(self._n_envs, dtype=np.float32)

        # Episode tracking buffers
        ep_nav: list[list[float]] = [[] for _ in range(self._n_envs)]
        ep_turnover: list[list[float]] = [[] for _ in range(self._n_envs)]
        ep_init_nav: list[float] = [1_000_000.0] * self._n_envs

        for update in range(1, n_updates + 1):
            # ── LR annealing ─────────────────────────────────────────────────
            if cfg.anneal_lr:
                frac = 1.0 - (update - 1) / n_updates
                for pg in self.optimizer.param_groups:
                    pg["lr"] = cfg.learning_rate * frac

            # ── Rollout collection ────────────────────────────────────────────
            obs_buf: list[dict[str, np.ndarray]] = []
            actions_buf: list[np.ndarray] = []
            log_probs_buf: list[np.ndarray] = []
            values_buf: list[np.ndarray] = []
            rewards_buf: list[np.ndarray] = []
            dones_buf: list[np.ndarray] = []

            with torch.no_grad():
                for _ in range(cfg.n_steps):
                    obs_buf.append({k: v.cpu().numpy() for k, v in obs.items()})
                    dones_buf.append(done.copy())

                    action, log_prob, _, value = self.model.get_action_and_value(obs)  # type: ignore[operator]
                    actions_buf.append(action.cpu().numpy())
                    log_probs_buf.append(log_prob.cpu().numpy())
                    values_buf.append(value.cpu().numpy())

                    obs_np_step, reward, terminated, truncated, info = self.envs.step(
                        action.cpu().numpy()
                    )
                    done = np.logical_or(terminated, truncated).astype(np.float32)
                    rewards_buf.append(np.array(reward, dtype=np.float32))
                    global_step += self._n_envs

                    # Track episode metrics
                    for ei in range(self._n_envs):
                        if isinstance(info, dict):
                            nav_i = float(info.get("nav", [ep_init_nav[ei]] * self._n_envs)[ei])
                            turn_i = float(info.get("turnover", [0.0] * self._n_envs)[ei])
                        else:
                            nav_i = ep_init_nav[ei]
                            turn_i = 0.0
                        ep_nav[ei].append(nav_i)
                        ep_turnover[ei].append(turn_i)

                        if done[ei]:
                            navs = [ep_init_nav[ei]] + ep_nav[ei]
                            m = compute_episode_metrics(navs, ep_turnover[ei])
                            episode_metrics.append(m)
                            ep_nav[ei] = []
                            ep_turnover[ei] = []

                    obs = _obs_to_tensor(obs_np_step, self.device)

                # Bootstrap value for last obs
                next_value = self.model.get_value(obs).cpu().numpy()  # type: ignore[operator]

            # ── GAE ───────────────────────────────────────────────────────────
            rewards = np.stack(rewards_buf)       # [T, E]
            values_t = np.stack(values_buf)       # [T, E]
            values_full = np.concatenate(
                [values_t, next_value[np.newaxis]], axis=0
            )                                     # [T+1, E]
            dones = np.stack(dones_buf)           # [T, E]

            advantages, returns = compute_gae(
                rewards, values_full, dones, cfg.gamma, cfg.gae_lambda
            )

            # ── Flatten to batch ──────────────────────────────────────────────
            obs_batch = _batch_obs(obs_buf, self.device)
            b_actions = torch.tensor(
                np.stack(actions_buf).reshape(self._batch_size, -1), device=self.device
            )
            b_log_probs = torch.tensor(
                np.stack(log_probs_buf).reshape(self._batch_size), device=self.device
            )
            b_advantages = torch.tensor(
                advantages.reshape(self._batch_size), device=self.device, dtype=torch.float32
            )
            b_returns = torch.tensor(
                returns.reshape(self._batch_size), device=self.device, dtype=torch.float32
            )
            b_values = torch.tensor(
                values_t.reshape(self._batch_size), device=self.device, dtype=torch.float32
            )

            # ── PPO update ────────────────────────────────────────────────────
            clip_fracs: list[float] = []
            approx_kls: list[float] = []
            pg_losses: list[float] = []
            v_losses: list[float] = []
            entropy_losses: list[float] = []

            b_inds = np.arange(self._batch_size)
            for _ in range(cfg.n_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, self._batch_size, self._minibatch_size):
                    mb_inds = b_inds[start : start + self._minibatch_size]

                    mb_obs = {k: v[mb_inds] for k, v in obs_batch.items()}
                    mb_act = b_actions[mb_inds]
                    mb_old_lp = b_log_probs[mb_inds]
                    mb_adv = b_advantages[mb_inds]
                    mb_ret = b_returns[mb_inds]
                    mb_val = b_values[mb_inds]

                    _, new_lp, entropy, new_val = self.model.get_action_and_value(  # type: ignore[operator]
                        mb_obs, mb_act
                    )

                    if cfg.normalize_advantage:
                        mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                    log_ratio = new_lp - mb_old_lp
                    ratio = log_ratio.exp()

                    with torch.no_grad():
                        approx_kl = ((ratio - 1) - log_ratio).mean().item()
                        approx_kls.append(approx_kl)
                        clip_fracs.append(
                            ((ratio - 1.0).abs() > cfg.clip_coef).float().mean().item()
                        )

                    pg_loss1 = -mb_adv * ratio
                    pg_loss2 = -mb_adv * ratio.clamp(1 - cfg.clip_coef, 1 + cfg.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss with clipping
                    v_clipped = mb_val + (new_val - mb_val).clamp(-cfg.clip_coef, cfg.clip_coef)
                    vf_loss = torch.max(
                        (new_val - mb_ret).pow(2),
                        (v_clipped - mb_ret).pow(2),
                    ).mean() * 0.5

                    entropy_loss = entropy.mean()
                    loss = pg_loss + cfg.vf_coef * vf_loss - cfg.ent_coef * entropy_loss

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
                    self.optimizer.step()

                    pg_losses.append(pg_loss.item())
                    v_losses.append(vf_loss.item())
                    entropy_losses.append(entropy_loss.item())

                if cfg.target_kl is not None and np.mean(approx_kls) > cfg.target_kl:
                    logger.debug(f"Early stop at epoch — KL {np.mean(approx_kls):.4f}")
                    break

            # ── Logging ───────────────────────────────────────────────────────
            if update % cfg.log_interval == 0:
                sps = int(global_step / (time.time() - start_time))
                recent = episode_metrics[-10:] if episode_metrics else []
                mean_sharpe = float(np.mean([m.sharpe for m in recent])) if recent else 0.0
                mean_cagr = float(np.mean([m.cagr for m in recent])) if recent else 0.0
                logger.info(
                    f"update={update}/{n_updates}  step={global_step:,}"
                    f"  sps={sps}  sharpe={mean_sharpe:.3f}  cagr={mean_cagr:.3f}"
                    f"  pg={np.mean(pg_losses):.4f}  vf={np.mean(v_losses):.4f}"
                    f"  ent={np.mean(entropy_losses):.4f}"
                    f"  clip_frac={np.mean(clip_fracs):.3f}"
                )
                self._log_mlflow(
                    global_step,
                    {
                        "charts/mean_sharpe": mean_sharpe,
                        "charts/mean_cagr": mean_cagr,
                        "losses/policy_loss": float(np.mean(pg_losses)),
                        "losses/value_loss": float(np.mean(v_losses)),
                        "losses/entropy": float(np.mean(entropy_losses)),
                        "charts/clip_fraction": float(np.mean(clip_fracs)),
                        "charts/approx_kl": float(np.mean(approx_kls)),
                        "charts/sps": sps,
                    },
                )

            if update % cfg.checkpoint_interval == 0:
                self._save_checkpoint(update)

        return episode_metrics

    def _log_mlflow(self, step: int, metrics: dict[str, float]) -> None:
        try:
            import mlflow

            mlflow.log_metrics(metrics, step=step)
        except Exception:
            pass

    def _save_checkpoint(self, update: int) -> None:
        path = self.cfg.checkpoint_dir / f"model_{update:06d}.pt"
        torch.save(
            {
                "update": update,
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
            },
            path,
        )
        logger.info(f"Checkpoint saved: {path}")
