#!/usr/bin/env python
"""Run a trained R6 allocator policy over a panel, deterministically.

`train_allocator_rl.py` fits a policy and checkpoints it; nothing existed to
run one afterwards. This does, and it is the only way to answer the question
the whole R6 stream exists for: does PPO over the allocator's parameters beat
the fixed parameters it was given?

    uv run python scripts/eval_allocator_rl.py \\
        --checkpoint checkpoints/allocator_rl_r6/allocator_policy_000100.pt \\
        --signal-dir data/signal/r4_v2_holdout \\
        --panel data/panels_kite/test.parquet

Actions are taken at the distribution **mean**, not sampled. Sampling is an
exploration device; a deployed policy would act on its mean, and reporting a
sampled rollout would mix policy quality with exploration noise — the exact
confusion that made the retired 505-dim policy churn 29% of NAV a day
(`10_architecture_revamp.md` §1.1).

Prints the fixed-parameter allocator over the identical env as a control, so
the comparison is like for like rather than against a number from another run.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from loguru import logger


def _load_yaml(p: Path) -> dict:  # type: ignore[type-arg]
    import yaml

    return dict(yaml.safe_load(p.read_text()) or {})


def _metrics(navs: list[float], turns: list[float], periods_per_year: float) -> dict[str, float]:
    nav = np.asarray(navs, dtype=np.float64)
    if nav.size < 3:
        return {"sharpe": float("nan"), "cagr": float("nan"), "mdd": float("nan"),
                "turnover_ann": float("nan"), "total_return": float("nan")}
    r = np.diff(np.log(np.maximum(nav, 1e-12)))
    sd = float(np.std(r, ddof=1))
    sharpe = float(np.mean(r) / sd * np.sqrt(periods_per_year)) if sd > 0 else float("nan")
    years = max(r.size / periods_per_year, 1e-9)
    cagr = float((nav[-1] / nav[0]) ** (1.0 / years) - 1.0)
    peak = np.maximum.accumulate(nav)
    mdd = float(np.max(1.0 - nav / np.maximum(peak, 1e-12)))
    return {
        "sharpe": sharpe,
        "cagr": cagr,
        "mdd": -mdd,
        "turnover_ann": float(np.sum(turns) / years) if turns else float("nan"),
        "total_return": float(nav[-1] / nav[0] - 1.0),
    }


def _rollout(env, policy, device, deterministic: bool) -> tuple[list[float], list[float]]:  # type: ignore[no-untyped-def]
    obs, _ = env.reset(seed=0)
    navs = [float(env.inner.nav)] if hasattr(env.inner, "nav") else []
    turns: list[float] = []
    done = False
    while not done:
        if policy is None:
            # Control: the env's own default action, i.e. the fixed parameters
            # the policy was asked to improve on. 0.5 in every bounded dim is
            # the midpoint of each ActionRanges span.
            action = np.full(env.action_space.shape, 0.5, dtype=np.float32)
        else:
            with torch.no_grad():
                t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                act, _, _, _ = policy.get_action_and_value(
                    t, deterministic=deterministic
                )
                action = act.squeeze(0).cpu().numpy()
        obs, _, term, trunc, info = env.step(action)
        done = bool(term or trunc)
        if "nav" in info:
            navs.append(float(info["nav"]))
        if "turnover" in info:
            turns.append(float(info["turnover"]))
    return navs, turns


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--signal-dir", type=Path, required=True)
    ap.add_argument("--panel", type=Path, required=True)
    ap.add_argument("--env-config", type=Path, default=Path("configs/env/allocator.yaml"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sampled", action="store_true",
                    help="sample actions instead of using the distribution mean")
    args = ap.parse_args()

    import importlib.util

    from trader.models.allocator_policy import AllocatorPolicy

    spec = importlib.util.spec_from_file_location(
        "_train_rl", Path(__file__).with_name("train_allocator_rl.py")
    )
    assert spec and spec.loader
    tr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tr)

    env_cfg = _load_yaml(args.env_config)
    envs = tr.build_envs(
        signal_dir=args.signal_dir, panel_path=args.panel,
        n_envs=1, env_cfg=env_cfg, seed=args.seed,
    )
    env = envs[0]
    ppy = float(env_cfg.get("periods_per_year", 12))

    # torch 2.6 defaults to weights_only=True and refuses the AllocatorPolicyConfig
    # dataclass the trainer stores alongside the weights. Allowlist that ONE class
    # rather than passing weights_only=False, which would disable the check for
    # every global in the file.
    from trader.models.allocator_policy import AllocatorPolicyConfig

    torch.serialization.add_safe_globals([AllocatorPolicyConfig])
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    device = torch.device(args.device)
    policy = AllocatorPolicy(ckpt["policy_config"]).to(device)
    policy.load_state_dict(ckpt["model_state"])
    policy.eval()
    logger.info(
        f"checkpoint {args.checkpoint.name}: update {ckpt.get('update')}, "
        f"metrics {json.dumps(ckpt.get('metrics', {}), default=float)[:200]}"
    )

    rows = []
    navs, turns = _rollout(env, None, device, True)
    rows.append(("fixed params (control)", _metrics(navs, turns, ppy)))
    navs, turns = _rollout(env, policy, device, not args.sampled)
    rows.append((f"RL policy ({'sampled' if args.sampled else 'mean'})",
                 _metrics(navs, turns, ppy)))

    print(f"\n{'arm':<26}{'Sharpe':>9}{'CAGR':>9}{'MDD':>9}{'Turn':>9}{'TotRet':>10}")
    print("-" * 72)
    for name, m in rows:
        print(f"{name:<26}{m['sharpe']:>9.3f}{m['cagr']:>9.3f}{m['mdd']:>9.3f}"
              f"{m['turnover_ann']:>9.3f}{m['total_return']:>10.3f}")
    print("-" * 72)
    print("A row here is a measurement, not a verdict (CLAUDE.md rule 2).")


if __name__ == "__main__":
    main()
