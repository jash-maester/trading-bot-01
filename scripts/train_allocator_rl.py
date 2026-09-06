#!/usr/bin/env python
"""Train the small-action allocator policy with PPO (R6).

    uv run python scripts/train_allocator_rl.py \
        --signal-dir data/signal/<tag> \
        --panel data/panels/<split>/train.parquet

THIS SCRIPT REFUSES TO RUN WITHOUT A PASSING R4 GATE.
--------------------------------------------------------------------------
``<signal-dir>/gate.json`` must exist and read ``{"verdict": "PASS", ...}``.
That is CLAUDE.md rule 1 — *never implement a milestone whose predecessor's
acceptance criteria have not demonstrably passed* — written as code rather than
as a note someone can skip.  The rule exists because M5's gate never opened and
M6, M7, Phase 1 and Phase 2 were built on top of it anyway: ~2,500 lines and 44
tests standing on a gate that never passed.  R6 is the same shape of risk —
an RL layer over a supervised signal — so the check is mechanical.

There is deliberately **no ``--force``**.  If the signal has no measured rank
IC, an allocator policy trained on it is optimising noise, and a flag that lets
you find that out slowly is worse than an error that tells you now.

The script does not touch the GPU by default (``--device cpu``): the encoder is
frozen and its output is read from R4's ``predictions.parquet``, so there is no
large tensor in this loop at all.
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any

import torch
from loguru import logger

GATE_FILE = "gate.json"
PREDICTIONS_FILE = "predictions.parquet"


class SignalGateNotPassed(RuntimeError):
    """The R4 signal gate is missing, malformed, or did not pass."""


def read_signal_gate(signal_dir: Path) -> dict[str, Any]:
    """Load ``<signal_dir>/gate.json``, raising if it is absent or unreadable."""
    path = Path(signal_dir) / GATE_FILE
    if not path.exists():
        raise SignalGateNotPassed(
            f"{path} does not exist. R6 trains a policy over R4's signal; without "
            "a gate file there is no evidence the signal carries any rank IC at "
            "all, and CLAUDE.md rule 1 forbids building on an unopened gate. Run "
            "the R4 signal job first."
        )
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise SignalGateNotPassed(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise SignalGateNotPassed(f"{path} must contain a JSON object, got {type(payload)}")
    return payload


def require_passing_gate(signal_dir: Path) -> dict[str, Any]:
    """Return the gate payload, or raise :class:`SignalGateNotPassed`.

    A missing ``verdict``, a verdict other than the exact string ``"PASS"``, or
    a missing ``predictions.parquet`` are all refusals.  "FAIL" is a legitimate
    and often good outcome (CLAUDE.md, *Verification standard*) — it is simply
    not a licence to train on top of it.
    """
    payload = read_signal_gate(signal_dir)
    verdict = payload.get("verdict")
    if verdict != "PASS":
        raise SignalGateNotPassed(
            f"{Path(signal_dir) / GATE_FILE} reports verdict={verdict!r}, not 'PASS'. "
            f"mean_ic_5d={payload.get('mean_ic_5d')} "
            f"mean_ic_20d={payload.get('mean_ic_20d')} "
            f"ci=[{payload.get('ic_ci_low')}, {payload.get('ic_ci_high')}] "
            f"icir={payload.get('icir')} n_windows={payload.get('n_windows')}. "
            "A failed gate is a result, not an obstacle: record it and stop. "
            "There is no override flag."
        )
    preds = Path(signal_dir) / PREDICTIONS_FILE
    if not preds.exists():
        raise SignalGateNotPassed(
            f"gate.json says PASS but {preds} is missing; the artefact directory "
            "is incomplete and the policy would have nothing to allocate over."
        )
    logger.info(
        f"R4 gate PASS — mean_ic_5d={payload.get('mean_ic_5d')} "
        f"mean_ic_20d={payload.get('mean_ic_20d')} "
        f"icir={payload.get('icir')} n_windows={payload.get('n_windows')}"
    )
    return payload


# ── env construction ──────────────────────────────────────────────────────────


def build_envs(
    *,
    signal_dir: Path,
    panel_path: Path,
    n_envs: int,
    env_cfg: dict[str, Any],
    seed: int,
) -> list[Any]:
    """One :class:`AllocatorEnv` per parallel worker, all over the same panel."""
    from trader.allocator.rebalance import RebalanceSchedule
    from trader.data.features import FEATURE_COLS
    from trader.data.universe import active_tickers
    from trader.env.allocator_env import (
        ActionRanges,
        AllocatorEnv,
        AllocatorEnvConfig,
        SignalPanel,
    )
    from trader.env.panel_env import PanelTradingEnv
    from trader.env.reward import LogReturn

    inner_cfg = dict(env_cfg.get("inner", {}))
    # The signal artefact pins the ticker axis; the env's universe must equal it
    # or the allocator would index the wrong stock.  The panels on disk are
    # stale relative to active_tickers() (504 vs 163), so intersect explicitly
    # rather than assuming they agree.
    index = json.loads((Path(signal_dir) / "index.json").read_text())
    tickers: list[str] = list(index["tickers"])
    universe_order = active_tickers()
    # The invariant that matters: the signal tickers KNOWN to the universe must
    # appear in the same relative order as in active_tickers().  Stated that way
    # it is satisfiable by an off-universe synthetic panel (vacuously) and still
    # catches the failure that matters — a shuffled axis, which would index the
    # wrong stock through embeddings.npy.
    known = set(universe_order)
    in_universe = [t for t in tickers if t in known]
    expected_order = [t for t in universe_order if t in set(in_universe)]
    if not in_universe:
        logger.warning(
            f"none of the {len(tickers)} signal tickers are in active_tickers(); "
            "the universe-order check is vacuous. This is expected for a "
            "synthetic panel and wrong for a real one."
        )
    elif len(in_universe) < len(tickers):
        logger.warning(
            f"{len(tickers) - len(in_universe)} of {len(tickers)} signal tickers "
            "are absent from active_tickers() (stale panel / renames, see "
            "CLAUDE.md); their rows cannot be joined to the universe."
        )
    if in_universe != expected_order:
        # The pinned contract says index.json["tickers"] MUST be
        # active_tickers() order, and axis 1 of embeddings.npy MUST match it.
        # This used to be a logger.warning that then proceeded on the signal's
        # own order: on a stale panel that warning fires on every run and gets
        # ignored, and a mis-ordered axis silently indexes the wrong stock.
        # A pinned MUST is enforced, not logged.
        first = next(
            (i for i, (a, b) in enumerate(zip(in_universe, expected_order)) if a != b),
            min(len(in_universe), len(expected_order)),
        )
        raise ValueError(
            f"signal index.json tickers are not in active_tickers() order "
            f"(pinned contract). {len(in_universe)} of {len(tickers)} signal "
            f"tickers are in the universe; first order divergence at index "
            f"{first}: signal={in_universe[first:first + 3]} "
            f"expected={expected_order[first:first + 3]}. embeddings.npy axis 1 "
            f"follows index.json, so proceeding would index the wrong stock. "
            f"Rebuild the signal artefacts against the current universe."
        )

    schedule = RebalanceSchedule(
        freq=inner_cfg.get("rebalance_freq", "monthly"),
        anchor=inner_cfg.get("rebalance_anchor", "first"),
    )
    ranges = ActionRanges(**dict(env_cfg.get("ranges", {})))
    cfg = AllocatorEnvConfig(
        periods_per_episode=int(env_cfg.get("periods_per_episode", 24)),
        max_days_per_period=int(env_cfg.get("max_days_per_period", 45)),
        max_name_weight=float(env_cfg.get("max_name_weight", 0.10)),
        max_sector_weight=float(env_cfg.get("max_sector_weight", 0.25)),
        # P3's per-name no-trade band. Read here, held on AllocatorEnvConfig and
        # passed into AllocatorParams by ActionRanges.decode -- the three sites
        # that make `no_trade_band:` in configs/env/allocator.yaml a live key
        # rather than the `min_trade_value: 500` trap.
        no_trade_band=float(env_cfg.get("no_trade_band", 0.0)),
        vol_lookback=int(env_cfg.get("vol_lookback", 20)),
        turnover_penalty=float(env_cfg.get("turnover_penalty", 0.02)),
        drawdown_penalty=float(env_cfg.get("drawdown_penalty", 1.0)),
        drawdown_threshold=float(env_cfg.get("drawdown_threshold", 0.10)),
        ranges=ranges,
    )
    signal = SignalPanel.from_artifacts(
        Path(signal_dir),
        Path(panel_path),
        tickers,
        horizon=str(env_cfg.get("signal_horizon", "r_hat_20d")),
        # Derived from vol_lookback, never named independently: the two used to
        # be separate keys that could silently disagree (CLAUDE.md's
        # `min_trade_value: 500` trap).
        vol_lookback=int(env_cfg.get("vol_lookback", 20)),
    )

    envs: list[Any] = []
    for i in range(n_envs):
        inner = PanelTradingEnv(
            panel_path=Path(panel_path),
            universe=tickers,
            feature_columns=list(FEATURE_COLS),
            lookback=int(inner_cfg.get("lookback_days", 60)),
            episode_length=int(inner_cfg.get("episode_length", 756)),
            initial_cash=float(inner_cfg.get("initial_cash", 1_000_000.0)),
            reward_fn=LogReturn(),
            max_weight_per_name=cfg.max_name_weight,
            # Both are load-bearing: AllocatorEnv refuses any other combination.
            turnover_penalty=0.0,
            use_excess_returns=True,
            seed=seed + i,
            rebalance_schedule=schedule,
        )
        envs.append(AllocatorEnv(inner, signal, cfg, seed=seed + i))
    return envs


def load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    payload = yaml.safe_load(Path(path).read_text())
    return dict(payload) if isinstance(payload, dict) else {}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--signal-dir", type=Path, required=True, help="data/signal/<tag>")
    p.add_argument("--panel", type=Path, required=True, help="feature panel parquet")
    p.add_argument("--env-config", type=Path, default=Path("configs/env/allocator.yaml"))
    p.add_argument("--train-config", type=Path, default=Path("configs/train/ppo_allocator.yaml"))
    p.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/allocator_rl"))
    p.add_argument("--device", default="cpu", help="cpu | mps | cuda (default cpu)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mlflow-experiment", default="allocator_rl")
    p.add_argument("--no-mlflow", action="store_true")
    args = p.parse_args(argv)

    # ── the gate, before anything expensive is imported or loaded ────────────
    require_passing_gate(args.signal_dir)

    from trader.models.allocator_policy import AllocatorPolicy, AllocatorPolicyConfig
    from trader.training.ppo_allocator import PPOAllocatorConfig, PPOAllocatorTrainer
    from trader.utils.seeding import seed_everything

    seed_everything(args.seed)
    env_cfg = load_yaml(args.env_config)
    train_cfg = load_yaml(args.train_config)

    ppo_cfg = PPOAllocatorConfig(
        total_steps=int(train_cfg.get("total_steps", 20_000)),
        n_envs=int(train_cfg.get("n_envs", 8)),
        n_steps=int(train_cfg.get("n_steps", 24)),
        n_epochs=int(train_cfg.get("n_epochs", 10)),
        n_minibatches=int(train_cfg.get("n_minibatches", 4)),
        gamma=float(train_cfg.get("gamma", 0.99)),
        gae_lambda=float(train_cfg.get("gae_lambda", 0.95)),
        clip_coef=float(train_cfg.get("clip_coef", 0.2)),
        ent_coef=float(train_cfg.get("ent_coef", 0.003)),
        vf_coef=float(train_cfg.get("vf_coef", 0.5)),
        max_grad_norm=float(train_cfg.get("max_grad_norm", 0.5)),
        learning_rate=float(train_cfg.get("learning_rate", 3e-4)),
        anneal_lr=bool(train_cfg.get("anneal_lr", True)),
        target_kl=train_cfg.get("target_kl", 0.02),
        normalize_advantage=bool(train_cfg.get("normalize_advantage", True)),
        normalize_rewards=bool(train_cfg.get("normalize_rewards", True)),
        checkpoint_dir=args.checkpoint_dir,
        log_interval=int(train_cfg.get("log_interval", 10)),
        checkpoint_interval=int(train_cfg.get("checkpoint_interval", 50)),
    )

    envs = build_envs(
        signal_dir=args.signal_dir,
        panel_path=args.panel,
        n_envs=ppo_cfg.n_envs,
        env_cfg=env_cfg,
        seed=args.seed,
    )
    covered = envs[0].n_signal_dates
    logger.info(
        f"{len(envs)} envs over {len(envs[0].inner.dates)} trading days, "
        f"{covered} of them carrying an OOS prediction; obs_dim={envs[0].obs_dim}, "
        f"action_dim={envs[0].ranges.action_dim}"
    )
    if covered == 0:
        raise SignalGateNotPassed(
            "no date in the panel has an OOS prediction — the signal artefact and "
            "the panel do not overlap. Check that --panel is the split the "
            "predictions were generated for."
        )

    model = AllocatorPolicy(
        AllocatorPolicyConfig(obs_dim=envs[0].obs_dim, action_dim=envs[0].ranges.action_dim)
    )
    trainer = PPOAllocatorTrainer(
        envs, model, ppo_cfg, torch.device(args.device), seed=args.seed
    )

    run_id: str | None = None
    mlflow = None
    if not args.no_mlflow:
        try:
            import mlflow as _mlflow

            mlflow = _mlflow
            mlflow.set_experiment(args.mlflow_experiment)
            mlflow.start_run(run_name=f"r6_allocator_seed{args.seed}_{date.today().isoformat()}")
            mlflow.log_params(
                {
                    "signal_dir": str(args.signal_dir),
                    "panel": str(args.panel),
                    "seed": args.seed,
                    **{f"ppo.{k}": v for k, v in vars(ppo_cfg).items()},
                }
            )
            active = mlflow.active_run()
            run_id = active.info.run_id if active is not None else None
        except Exception as exc:  # pragma: no cover - MLflow is optional here
            logger.warning(f"MLflow disabled: {exc}")
            mlflow = None

    try:
        episodes = trainer.train()
    finally:
        if mlflow is not None:
            mlflow.end_run()

    if episodes:
        tail = episodes[-ppo_cfg.n_envs :]
        mean_excess = sum(e.excess_log_return for e in tail) / len(tail)
        mean_turnover = sum(e.mean_turnover for e in tail) / len(tail)
        mean_k = sum(e.mean_k for e in tail) / len(tail)
        logger.info(
            f"last {len(tail)} episodes: mean excess log return {mean_excess:.4f}, "
            f"mean turnover/period {mean_turnover:.4f}, mean k {mean_k:.1f}"
        )
    # CLAUDE.md rule 2: nothing here is a "winner" without this id.
    logger.info(f"MLflow run id: {run_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
