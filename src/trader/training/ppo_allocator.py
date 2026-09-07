"""PPO over :class:`~trader.env.allocator_env.AllocatorEnv` (R6).

A deliberately small trainer for a deliberately small problem.  One environment
step is one rebalance period, an episode is 12–36 of them, and the action is
``3 + n_sectors`` bounded scalars — so a whole rollout is a few hundred
transitions of ~30 floats each and the batch fits in L2, never mind VRAM.  The
expensive object in this project is the TCN encoder (97.9% of the arithmetic,
``audit/A2_compute.md``); it does not appear here at all, because R6 runs
against a **frozen** encoder whose embeddings and predictions were computed once
(:mod:`trader.training.embedding_cache`).

Relationship to ``trader.training.ppo``
---------------------------------------
:func:`trader.training.ppo.compute_gae` and
:class:`trader.training.ppo.RunningMeanStd` are **imported**, not copied: the
advantage estimator and the reward scaler must be identical across the two
trainers or a comparison between them measures the estimator.  Everything else
is separate because the two loops disagree on the things that matter — a flat
Box observation instead of a Dict of ``[L, N, F]`` tensors, no BF16 autocast (a
3-layer MLP on ~30 inputs is latency-bound, and A2 measured autocast at 4.6×
*slower* on narrow conv1d), and no per-stock anything.

Mode discipline
---------------
Rollout under ``eval()``, the gradient update under ``train()``, and the model
left in ``eval()`` on return.  ``ppo.py`` had this exactly inverted until it was
fixed: dropout never fired in any backward pass, so TCN/attention dropout was
dead for the entire history of the project, and validation then ran *with*
dropout on.  The rule is not cosmetic — under a stochastic rollout,
``log_prob_old`` is one random draw and ``log_prob_new`` another, so the
importance ratio measures the difference between two dropout masks rather than
the weight update.
"""
from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from loguru import logger

from trader.env.allocator_env import AllocatorEnv
from trader.models.allocator_policy import AllocatorPolicy
from trader.training.ppo import RunningMeanStd, compute_gae


@dataclass
class PPOAllocatorConfig:
    """Hyper-parameters.  ``total_steps`` counts **periods**, not days.

    With ``freq="monthly"`` one step is ~21 trading days, so 20,000 steps is
    ~1,700 simulated years spread over ``n_envs`` — cheap, because no encoder
    runs.  ``n_steps`` defaults to one episode's worth so every rollout ends on
    an episode boundary and GAE never bootstraps across a reset it cannot see.

    ``ent_coef`` multiplies the **Beta's analytic entropy in action space**
    (:mod:`trader.models.allocator_policy`), which is a real statement about how
    undecided the policy is over K, the turnover budget and the tilts.  The old
    ``ent_coef`` multiplied ``(N+1)·H(σ)`` in logit space and meant nothing, so
    its historical value carries no information about a good value here: this
    default is a starting point, **UNCALIBRATED**, and needs a sweep.
    """

    total_steps: int = 20_000
    n_envs: int = 8
    n_steps: int = 24
    n_epochs: int = 10
    n_minibatches: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.003
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    learning_rate: float = 3e-4
    anneal_lr: bool = True
    target_kl: float | None = 0.02
    normalize_advantage: bool = True
    normalize_rewards: bool = True
    checkpoint_dir: Path = field(default_factory=lambda: Path("checkpoints/allocator_rl"))
    log_interval: int = 10
    checkpoint_interval: int = 50
    #: How many of the most recent checkpoints to keep on disk. A long run at
    #: interval 50 writes hundreds of files, each carrying a full policy; only
    #: the newest few are ever loaded. 0 keeps every one.
    keep_last_checkpoints: int = 3

    def __post_init__(self) -> None:
        if self.n_envs < 1 or self.n_steps < 1:
            raise ValueError("n_envs and n_steps must be >= 1")
        batch = self.n_envs * self.n_steps
        if batch % self.n_minibatches != 0:
            raise ValueError(
                f"n_envs*n_steps={batch} must be divisible by "
                f"n_minibatches={self.n_minibatches}"
            )


@dataclass
class AllocatorEpisodeSummary:
    """What one finished episode did, in period units.

    ``excess_log_return`` is the sum of the inner env's daily excess log returns
    over the whole episode — i.e. log outperformance of equal-weight, *before*
    the turnover and drawdown terms the reward adds.  It is the number to look
    at when asking "did it beat the baseline"; ``total_reward`` is what PPO
    optimised, which is not the same question.
    """

    n_periods: int = 0
    total_reward: float = 0.0
    excess_log_return: float = 0.0
    mean_turnover: float = 0.0
    max_drawdown: float = 0.0
    mean_k: float = 0.0
    final_nav: float = 0.0


class _EpisodeTracker:
    """Accumulates per-period info for one env until it reports done."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.rewards: list[float] = []
        self.excess: list[float] = []
        self.turnover: list[float] = []
        self.drawdown: list[float] = []
        self.k: list[float] = []
        self.nav: float = 0.0

    def add(self, reward: float, info: dict[str, Any]) -> None:
        self.rewards.append(reward)
        self.excess.append(float(info.get("excess_log_return", 0.0)))
        self.turnover.append(float(info.get("turnover", 0.0)))
        self.drawdown.append(float(info.get("drawdown", 0.0)))
        self.k.append(float(info.get("k", 0.0)))
        self.nav = float(info.get("nav", self.nav))

    def summary(self) -> AllocatorEpisodeSummary:
        return AllocatorEpisodeSummary(
            n_periods=len(self.rewards),
            total_reward=float(np.sum(self.rewards)),
            excess_log_return=float(np.sum(self.excess)),
            mean_turnover=float(np.mean(self.turnover)) if self.turnover else 0.0,
            max_drawdown=float(np.max(self.drawdown)) if self.drawdown else 0.0,
            mean_k=float(np.mean(self.k)) if self.k else 0.0,
            final_nav=self.nav,
        )


class PPOAllocatorTrainer:
    """PPO for a Beta policy over the allocator's parameters.

    ``envs`` is a plain sequence of :class:`AllocatorEnv` stepped in a Python
    loop.  There is no vectorised backend on purpose: an allocator step drives
    ~21 inner env days of NumPy, so the loop overhead is invisible next to it,
    and a subprocess vec-env would only add pickling and a second source of
    seed state.
    """

    def __init__(
        self,
        envs: Sequence[AllocatorEnv],
        model: AllocatorPolicy,
        config: PPOAllocatorConfig,
        device: torch.device | None = None,
        *,
        seed: int = 0,
    ) -> None:
        if not envs:
            raise ValueError("need at least one env")
        if len(envs) != config.n_envs:
            raise ValueError(f"got {len(envs)} envs but n_envs={config.n_envs}")
        obs_dim = envs[0].obs_dim
        if model.obs_dim != obs_dim or model.action_dim != envs[0].ranges.action_dim:
            raise ValueError(
                f"model expects obs_dim={model.obs_dim}, action_dim={model.action_dim}; "
                f"env provides obs_dim={obs_dim}, action_dim={envs[0].ranges.action_dim}"
            )
        self.envs = list(envs)
        self.cfg = config
        self.device = device or torch.device("cpu")
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=config.learning_rate, eps=1e-5
        )
        self._reward_rms = RunningMeanStd()
        self._batch_size = config.n_envs * config.n_steps
        self._minibatch_size = self._batch_size // config.n_minibatches
        self._seed = seed
        self.last_metrics: dict[str, float] = {}

    # ── main loop ────────────────────────────────────────────────────────────

    def train(self) -> list[AllocatorEpisodeSummary]:
        cfg = self.cfg
        n_updates = cfg.total_steps // self._batch_size
        episodes: list[AllocatorEpisodeSummary] = []
        trackers = [_EpisodeTracker() for _ in self.envs]
        start = time.time()

        obs = np.stack(
            [env.reset(seed=self._seed + i)[0] for i, env in enumerate(self.envs)]
        ).astype(np.float32)
        done = np.zeros(cfg.n_envs, dtype=np.float32)

        try:
            for update in range(1, n_updates + 1):
                if cfg.anneal_lr:
                    frac = 1.0 - (update - 1) / max(n_updates, 1)
                    for pg in self.optimizer.param_groups:
                        pg["lr"] = cfg.learning_rate * frac

                obs, done, rollout = self._collect_rollout(obs, done, trackers, episodes)
                metrics = self._update(rollout)
                metrics["episodes"] = float(len(episodes))
                if episodes:
                    recent = episodes[-cfg.n_envs :]
                    metrics["ep_excess_log_return"] = float(
                        np.mean([e.excess_log_return for e in recent])
                    )
                    metrics["ep_mean_turnover"] = float(np.mean([e.mean_turnover for e in recent]))
                    metrics["ep_mean_k"] = float(np.mean([e.mean_k for e in recent]))
                self.last_metrics = metrics

                if cfg.log_interval and update % cfg.log_interval == 0:
                    sps = update * self._batch_size / max(time.time() - start, 1e-9)
                    logger.info(
                        f"update {update}/{n_updates} "
                        + " ".join(f"{k}={v:.4f}" for k, v in metrics.items())
                        + f" periods/s={sps:.1f}"
                    )
                if cfg.checkpoint_interval and update % cfg.checkpoint_interval == 0:
                    self.save_checkpoint(update)
        finally:
            # Leave the model in eval(): whatever runs next — an evaluation
            # split, a checkpoint dump, a notebook — must not silently get
            # dropout.  `finally` so this holds on the exception path too.
            self.model.eval()
        return episodes

    # ── rollout ──────────────────────────────────────────────────────────────

    def _collect_rollout(
        self,
        obs: np.ndarray,
        done: np.ndarray,
        trackers: list[_EpisodeTracker],
        episodes: list[AllocatorEpisodeSummary],
    ) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        cfg = self.cfg
        # MODE DISCIPLINE (1/2): rollout in eval().  log_prob_old must be a
        # deterministic function of the weights, or the PPO ratio measures
        # dropout masks.  See the module docstring.
        self.model.eval()

        obs_buf = np.zeros((cfg.n_steps, cfg.n_envs, self.model.obs_dim), dtype=np.float32)
        act_buf = np.zeros((cfg.n_steps, cfg.n_envs, self.model.action_dim), dtype=np.float32)
        logp_buf = np.zeros((cfg.n_steps, cfg.n_envs), dtype=np.float32)
        val_buf = np.zeros((cfg.n_steps, cfg.n_envs), dtype=np.float32)
        rew_buf = np.zeros((cfg.n_steps, cfg.n_envs), dtype=np.float32)
        done_buf = np.zeros((cfg.n_steps, cfg.n_envs), dtype=np.float32)

        with torch.no_grad():
            for t in range(cfg.n_steps):
                obs_buf[t] = obs
                done_buf[t] = done
                obs_t = torch.as_tensor(obs, device=self.device)
                action, log_prob, _, value = self.model.get_action_and_value(obs_t)
                act_np = action.cpu().numpy().astype(np.float32)
                act_buf[t] = act_np
                logp_buf[t] = log_prob.cpu().numpy()
                val_buf[t] = value.cpu().numpy()

                next_obs = np.empty_like(obs)
                for i, env in enumerate(self.envs):
                    o, r, term, trunc, info = env.step(act_np[i])
                    rew_buf[t, i] = float(r)
                    trackers[i].add(float(r), info)
                    finished = bool(term or trunc)
                    done[i] = 1.0 if finished else 0.0
                    if finished:
                        episodes.append(trackers[i].summary())
                        trackers[i].reset()
                        o, _ = env.reset(seed=self._seed + 1000 * len(episodes) + i)
                    next_obs[i] = o
                obs = next_obs

            bootstrap = self.model.get_value(
                torch.as_tensor(obs, device=self.device)
            ).cpu().numpy()

        rewards = rew_buf
        if cfg.normalize_rewards:
            self._reward_rms.update(rewards.flatten())
            rewards = self._reward_rms.normalize(rewards).astype(np.float32)

        values_full = np.concatenate([val_buf, bootstrap[np.newaxis]], axis=0)
        # `done_buf[t]` is the flag as of *entering* step t, which is what
        # `compute_gae` consumes — identical convention to `ppo.py`.
        advantages, returns = compute_gae(
            rewards, values_full, done_buf, cfg.gamma, cfg.gae_lambda
        )
        rollout = {
            "obs": obs_buf.reshape(self._batch_size, -1),
            "actions": act_buf.reshape(self._batch_size, -1),
            "log_probs": logp_buf.reshape(self._batch_size),
            "values": val_buf.reshape(self._batch_size),
            "advantages": advantages.reshape(self._batch_size),
            "returns": returns.reshape(self._batch_size),
        }
        return obs, done, rollout

    # ── update ───────────────────────────────────────────────────────────────

    def _update(self, rollout: dict[str, np.ndarray]) -> dict[str, float]:
        cfg = self.cfg
        dev = self.device
        b_obs = torch.as_tensor(rollout["obs"], device=dev)
        b_act = torch.as_tensor(rollout["actions"], device=dev)
        b_logp = torch.as_tensor(rollout["log_probs"], device=dev)
        b_adv = torch.as_tensor(rollout["advantages"], device=dev, dtype=torch.float32)
        b_ret = torch.as_tensor(rollout["returns"], device=dev, dtype=torch.float32)

        # MODE DISCIPLINE (2/2): the gradient update runs in train(), so any
        # dropout the config asks for actually regularises something.
        self.model.train()
        idx = np.arange(self._batch_size)
        rng = np.random.default_rng(self._seed)
        approx_kl = 0.0
        clip_frac = 0.0
        pg_loss_v = v_loss_v = ent_v = 0.0
        stopped_early = False

        for _ in range(cfg.n_epochs):
            rng.shuffle(idx)
            for start in range(0, self._batch_size, self._minibatch_size):
                mb = idx[start : start + self._minibatch_size]
                mb_t = torch.as_tensor(mb, device=dev)
                _, new_logp, entropy, new_value = self.model.get_action_and_value(
                    b_obs[mb_t], b_act[mb_t]
                )
                log_ratio = new_logp - b_logp[mb_t]
                ratio = log_ratio.exp()

                with torch.no_grad():
                    # Schulman's k3 estimator — unbiased and non-negative.
                    approx_kl = float(((ratio - 1.0) - log_ratio).mean())
                    clip_frac = float(((ratio - 1.0).abs() > cfg.clip_coef).float().mean())

                mb_adv = b_adv[mb_t]
                if cfg.normalize_advantage and mb_adv.numel() > 1:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                pg_loss = torch.max(
                    -mb_adv * ratio,
                    -mb_adv * ratio.clamp(1.0 - cfg.clip_coef, 1.0 + cfg.clip_coef),
                ).mean()
                v_loss = 0.5 * ((new_value - b_ret[mb_t]) ** 2).mean()
                ent_loss = entropy.mean()
                loss = pg_loss + cfg.vf_coef * v_loss - cfg.ent_coef * ent_loss

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()   # type: ignore[no-untyped-call]
                nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    pg_loss_v = float(pg_loss.detach())
                    v_loss_v = float(v_loss.detach())
                    ent_v = float(ent_loss.detach())

            if cfg.target_kl is not None and approx_kl > cfg.target_kl:
                stopped_early = True
                break

        # Back to eval() the moment the update is done — the next rollout needs
        # it and so does anything that inspects the model between updates.
        self.model.eval()
        return {
            "pg_loss": pg_loss_v,
            "v_loss": v_loss_v,
            "entropy": ent_v,
            "approx_kl": approx_kl,
            "clip_frac": clip_frac,
            "kl_early_stop": float(stopped_early),
        }

    # ── io ───────────────────────────────────────────────────────────────────

    def save_checkpoint(self, update: int) -> Path:
        out = Path(self.cfg.checkpoint_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"allocator_policy_{update:06d}.pt"
        torch.save(
            {
                "update": update,
                "model_state": {k: v.cpu() for k, v in self.model.state_dict().items()},
                "policy_config": self.model.cfg,
                "metrics": self.last_metrics,
            },
            path,
        )
        self._prune_checkpoints()
        return path

    def _prune_checkpoints(self) -> None:
        """Keep only the ``keep_last_checkpoints`` most recent checkpoint files.

        Sorted by the zero-padded update number in the filename, not by mtime:
        a resumed run can rewrite an older update after a newer one, and mtime
        would then delete the wrong file.
        """
        keep = int(self.cfg.keep_last_checkpoints)
        if keep <= 0:
            return
        out = Path(self.cfg.checkpoint_dir)
        existing = sorted(out.glob("allocator_policy_*.pt"))
        for stale in existing[:-keep]:
            try:
                stale.unlink()
            except OSError as exc:                       # pragma: no cover
                logger.warning(f"could not remove old checkpoint {stale}: {exc}")
