#!/usr/bin/env python
"""R2: every baseline, every cadence, net of cost and tax.

`PROGRESS.md` has carried R2 as NOT_STARTED and then PARTIAL since session 1,
while R5's gate names it:

> R2 — 5 baselines × 3 frequencies × 4 benchmarks, net of cost **and tax**
> R5 — beats **best R2 baseline** net of cost+tax, paired bootstrap CI …

`audit/R5_GATE.md` shows the allocator clearing a paired interval against
`EqualWeightRebalanced` on every arm. That is the strongest baseline *measured*;
until the other four run, it is not shown to be the strongest that *exists*, and
R5 stays conditional. All five agents already exist in `trader/env/baselines.py`
— `run_allocator.py` simply only ever called one of them.

    uv run python scripts/run_baselines.py data=kite_v1 +split=oos_r4_v2 \
        +apply_tax=true '++baselines.freq_grid=[monthly,weekly,daily]'

Each baseline runs through the SAME `PanelTradingEnv` at the same cadence as the
allocator, with the same costs, the same `min_trade_value`, and tax when asked
for — so a row here is directly comparable with a row in the allocator table
rather than approximately so.

WHY `RandomPolicy` IS IN THE GRID. It is not a contender; it is the floor. If a
baseline cannot beat uniformly random logits over the same tradeable set, the
comparison is measuring the cadence and the cost model rather than any
selection. The project has been here before: four milestones were spent tuning a
policy whose whole measured effect turned out to be turnover.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import hydra
from loguru import logger
from omegaconf import DictConfig

if TYPE_CHECKING:
    pass

_DEFAULT_FREQ = ("monthly", "weekly", "daily")


def _fail(msg: str) -> None:
    logger.error(msg)
    sys.exit(1)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    import hydra.utils
    import mlflow
    import polars as pl

    from trader.allocator.rebalance import RebalanceSchedule
    from trader.data.features import FEATURE_COLS, resolve_panels_root
    from trader.data.universe import resolve_traded_universe
    from trader.env.baselines import (
        EqualWeightFrozenUniverse,
        EqualWeightRebalanced,
        MomentumTopK,
        RandomPolicy,
        SixtyFortyCash,
        run_baseline_episode,
    )
    from trader.env.costs import DEFAULT_MIN_TRADE_VALUE
    from trader.env.panel_env import PanelTradingEnv
    from trader.training.eval_metrics import compute_episode_metrics

    orig = Path(hydra.utils.get_original_cwd())
    panels_root = resolve_panels_root(cfg, orig)
    seed = int(cfg.get("seed", 42))
    split = str(cfg.get("split", "val"))
    panel_path = panels_root / f"{split}.parquet"
    if not panel_path.exists():
        _fail(f"{panel_path} does not exist")

    bcfg = cfg.get("baselines", {})
    freq_grid = [str(f) for f in bcfg.get("freq_grid", _DEFAULT_FREQ)]
    momentum_k = int(bcfg.get("momentum_k", 20))
    nav_dir = bcfg.get("nav_dir", None)

    # WHERE THE UNIVERSE COMES FROM, and why this is not a detail.
    #
    # The env is built over `universe`, so it can only ever see those columns.
    # Pointing this script at the point-in-time panel while taking the universe
    # from `active_tickers()` would run every baseline over the SAME fixed 504
    # names the rebuild exists to escape — and print a table that looks
    # entirely normal while measuring nothing new.
    #
    # Default stays `active_tickers()` so every number already recorded against
    # the Kite panels reproduces exactly.
    universe, provenance = resolve_traded_universe(
        panel_path, from_panel=bool(bcfg.get("universe_from_panel", False))
    )
    logger.info(f"universe from {provenance}")
    lookback = int(cfg.env.lookback_days)
    dates = sorted(pl.read_parquet(panel_path, columns=["date"])["date"].unique().to_list())
    episode_length = len(dates) - lookback - 1
    if episode_length < 2:
        _fail(f"{panel_path} has {len(dates)} dates; too short for lookback={lookback}")

    capital = float(cfg.env.initial_cash)
    env_base: dict[str, Any] = dict(
        panel_path=panel_path,
        universe=universe,
        feature_columns=FEATURE_COLS,
        lookback=lookback,
        episode_length=episode_length,
        initial_cash=capital,
        min_trade_value=float(cfg.env.get("min_trade_value", DEFAULT_MIN_TRADE_VALUE)),
        apply_tax=bool(cfg.get("apply_tax", False)),
        seed=seed,
    )

    def agents() -> list[tuple[str, Any]]:
        # Rebuilt per cadence: RandomPolicy carries RNG state, and MomentumTopK
        # freezes a selection, so a single instance reused across cadences would
        # make the later ones depend on the earlier.
        return [
            ("equal_weight", EqualWeightRebalanced()),
            ("equal_weight_frozen", EqualWeightFrozenUniverse()),
            ("momentum_topk", MomentumTopK(k=momentum_k, feature_columns=FEATURE_COLS)),
            ("sixty_forty", SixtyFortyCash()),
            ("random", RandomPolicy(seed=seed)),
        ]

    mlflow.set_tracking_uri(f"http://localhost:{cfg.get('mlflow_port', 5555)}")
    mlflow.set_experiment("baselines")
    apply_tax = bool(cfg.get("apply_tax", False))
    logger.info(
        f"R2 grid: {len(agents())} baselines x {len(freq_grid)} cadence(s) on "
        f"{split}, initial_cash={capital:,.0f}, apply_tax={apply_tax}"
    )

    header = (f"{'baseline':<24}{'freq':>9}{'Sharpe':>9}{'CAGR':>9}{'MDD':>9}"
              f"{'Turn':>9}{'sold/reb':>10}{'DP ₹':>11}")
    rule = "-" * len(header)
    lines: list[str] = [rule, header, rule]
    best: dict[str, tuple[str, float]] = {}

    for freq in freq_grid:
        schedule = None if freq == "daily" else RebalanceSchedule(freq)  # type: ignore[arg-type]
        for name, agent in agents():
            env = PanelTradingEnv(rebalance_schedule=schedule, **env_base)
            navs, turns, diag = run_baseline_episode(env, agent, seed)
            m = compute_episode_metrics(navs, turns)
            if nav_dir:
                out = orig / str(nav_dir) / f"nav_{name}_{freq}_{split}.parquet"
                out.parent.mkdir(parents=True, exist_ok=True)
                n = len(navs)
                d = list(dates[lookback:][:n])
                d += [None] * (n - len(d))
                pl.DataFrame({"date": d, "nav": navs}).write_parquet(out)
            lines.append(
                f"{name:<24}{freq:>9}{m.sharpe:>9.3f}{m.cagr:>9.3f}"
                f"{m.max_drawdown:>9.3f}{m.turnover_ann:>9.3f}"
                f"{diag['scrip_sell_days_per_rebalance']:>10.1f}"
                f"{diag['dp_charges_paid']:>11,.0f}"
            )
            if freq not in best or m.cagr > best[freq][1]:
                best[freq] = (name, m.cagr)
            with mlflow.start_run(run_name=f"{name}_{freq}_{split}"):
                mlflow.log_params(
                    {"strategy": name, "freq": freq, "split": split,
                     "universe_size": len(universe), "initial_cash": capital,
                     "apply_tax": apply_tax, "seed": seed,
                     "momentum_k": momentum_k}
                )
                mlflow.log_metrics(
                    {f"{split}/sharpe": m.sharpe, f"{split}/cagr": m.cagr,
                     f"{split}/max_drawdown": m.max_drawdown,
                     f"{split}/turnover_ann": m.turnover_ann,
                     f"{split}/total_return": m.total_return,
                     **{f"{split}/{k}": v for k, v in diag.items()}}
                )
        lines.append(rule)

    print("\n".join(lines))
    print("\nStrongest baseline by CAGR, per cadence:")
    for freq, (name, cagr) in best.items():
        marker = "" if name == "equal_weight" else "   <-- NOT equal_weight"
        print(f"  {freq:<10}{name:<24}{cagr:>8.3f}{marker}")
    if any(n != "equal_weight" for n, _ in best.values()):
        print("\nR5's gate compares against equal_weight. A cadence where another")
        print("baseline wins means that comparison is against the wrong bar and")
        print("audit/R5_GATE.md must be re-run against the winner.")
    else:
        print("\nequal_weight is the strongest baseline at every cadence measured,")
        print("which is the bar audit/R5_GATE.md already tests against.")
    print("\nA row here is a measurement, not a verdict: nothing is a winner "
          "without a run ID (CLAUDE.md rule 2).")


if __name__ == "__main__":
    main()
