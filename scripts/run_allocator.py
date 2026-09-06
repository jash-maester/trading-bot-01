#!/usr/bin/env python
"""Backtest the R5 deterministic allocator over an R4 signal, on a K × cadence grid.

`10_architecture_revamp.md` §4 ranks rebalance frequency as the second-biggest
profit lever in the system — delivery STT is 0.1% on **both** legs and STCG is
20% under twelve months — and §5 replaces the Gaussian-over-logits policy with
an allocator that has no learned parameters.  This script measures both at
once: K ∈ {20, 30, 40} × freq ∈ {daily, weekly, monthly} × horizon ∈ {5d, 20d},
each against `EqualWeightRebalanced` **at the same cadence**, which is the only
comparison that isolates selection from cadence.

`+allocator.null_control=true` adds a third arm per cadence: the same allocator
driven by **seeded white noise** instead of the signal.  It is the control that
says whether a row above beat equal-weight because the signal selects or merely
because the cadence stopped it trading — a distinction the old project never
drew, and the reason it spent four milestones tuning a policy whose whole
measured effect was turnover (`10` §1.1).

Nothing here is a gate and nothing here is a winner.  It prints a table and
writes MLflow runs under the experiment ``allocator``; a run ID is what makes a
number quotable (CLAUDE.md rule 2).

Usage
-----
    uv run python scripts/run_allocator.py                        # val split
    uv run python scripts/run_allocator.py +split=test
    uv run python scripts/run_allocator.py +signal_tag=r4_v2
    uv run python scripts/run_allocator.py \
        +allocator.k_grid=[30] +allocator.freq_grid=[monthly]

Note the leading ``+`` on every override above.  ``configs/config.yaml`` is a
struct, and ``split``, ``signal_tag`` and ``allocator`` are not keys in it, so
Hydra rejects a bare ``split=test`` with *"Key 'split' is not in struct"*.
(`scripts/evaluate.py`'s usage block documents ``split=val`` without the
``+``; that form does not work — reported, not silently corrected, because
that file belongs to another stream.)

Requires `data/signal/<signal_tag>/predictions.parquet` from R4 (`09` §10, Q4).
If it is not there this script says so and stops; it never invents a signal.
"""
from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

import hydra
from loguru import logger
from omegaconf import DictConfig

if TYPE_CHECKING:                                   # pragma: no cover
    from pathlib import Path

    import numpy as np

    from trader.allocator import AllocatorParams
    from trader.env.panel_env import PanelTradingEnv

# Grid defaults.  Overridable from the CLI; see the usage block above.
_K_GRID = (20, 30, 40)
_FREQ_GRID = ("daily", "weekly", "monthly")
_HORIZON_GRID = ("5d", "20d")

# The column the allocator sizes on.  Annualised in the panel (`features.py`
# multiplies the rolling std by √252); inverse-vol weights are renormalised so
# the annualisation cancels, but the column has to be the *same* one for every
# name, which is why it is read from the panel rather than from `obs`.
# Default only. The column actually read is derived from the configured
# `vol_lookback` (see `_vol_lookback` in main), so the number the allocator is
# told it is sizing on and the column it is handed cannot disagree.
_DEFAULT_VOL_LOOKBACK = 20


def _fail(message: str) -> None:
    """Stop with a message that names what is missing and who produces it."""
    logger.error(message)
    sys.exit(1)


def _dense_signal(
    pred_path: Path,
    dates: list[Any],
    tickers: list[str],
    horizons: tuple[str, ...],
) -> dict[str, np.ndarray]:
    """Scatter `predictions.parquet` into ``{horizon: [T, N] float64}``, NaN-filled.

    A missing (date, ticker) stays NaN and the allocator drops that name — the
    right behaviour for an OOS prediction file, which by construction covers
    only the walk-forward's out-of-sample windows and only the tickers that
    were tradeable then.  Filling with 0.0 instead would rank a name with no
    prediction above every name the model actually dislikes.
    """
    import numpy as np
    import polars as pl

    preds = pl.read_parquet(pred_path)
    required = {"date", "ticker", "r_hat_5d", "r_hat_20d"}
    missing = required - set(preds.columns)
    if missing:
        _fail(
            f"{pred_path} is missing column(s) {sorted(missing)}. "
            "The R4 contract is date, ticker, r_hat_5d, r_hat_20d."
        )

    date_idx = {d: i for i, d in enumerate(dates)}
    tick_idx = {t: i for i, t in enumerate(tickers)}
    rows = preds.filter(
        pl.col("date").is_in(list(date_idx)) & pl.col("ticker").is_in(list(tick_idx))
    )
    if rows.height == 0:
        _fail(
            f"{pred_path} has {preds.height} rows but none of them land on this "
            "split's (date, ticker) grid — wrong split, or a stale panel."
        )

    di = np.array([date_idx[d] for d in rows["date"].to_list()], dtype=np.int64)
    ti = np.array([tick_idx[t] for t in rows["ticker"].to_list()], dtype=np.int64)

    out: dict[str, np.ndarray] = {}
    for h in horizons:
        arr = np.full((len(dates), len(tickers)), np.nan, dtype=np.float64)
        arr[di, ti] = rows[f"r_hat_{h}"].to_numpy().astype(np.float64)
        out[h] = arr

    covered = float(np.isfinite(out[horizons[0]]).mean())
    logger.info(
        f"Signal {pred_path}: {preds.height} rows, {rows.height} on-grid, "
        f"{covered:.1%} of the [T={len(dates)}, N={len(tickers)}] grid populated."
    )
    return out


def _dense_column(
    panel_path: Path, column: str, dates: list[Any], tickers: list[str]
) -> np.ndarray:
    """One panel column as a dense ``[T, N] float64``, NaN where absent."""
    import numpy as np
    import polars as pl

    frame = pl.read_parquet(panel_path, columns=["date", "ticker", column])
    date_idx = {d: i for i, d in enumerate(dates)}
    tick_idx = {t: i for i, t in enumerate(tickers)}
    frame = frame.filter(pl.col("ticker").is_in(list(tick_idx)))
    di = np.array([date_idx[d] for d in frame["date"].to_list()], dtype=np.int64)
    ti = np.array([tick_idx[t] for t in frame["ticker"].to_list()], dtype=np.int64)
    arr = np.full((len(dates), len(tickers)), np.nan, dtype=np.float64)
    arr[di, ti] = frame[column].to_numpy().astype(np.float64)
    return arr


def _run_allocator(
    env: PanelTradingEnv,
    r_hat: np.ndarray,
    vol: np.ndarray,
    params: AllocatorParams,
    seed: int,
) -> tuple[list[float], list[float]]:
    """Drive one full pass of the env from the allocator.  Returns (navs, turnovers).

    Information set
    ---------------
    The env trades day ``d`` at its **open** from an observation whose feature
    window ends at ``d-1``'s close (`panel_env._build_obs` slices
    ``[start:day_idx]``).  So the signal row used to trade day ``d`` is the one
    dated ``d-1``.  Indexing ``r_hat[day_idx]`` instead would trade on a
    prediction made from data that includes the day being traded — a one-day
    lookahead worth more than the entire strategy.
    """
    import numpy as np

    from trader.allocator import allocate

    obs, _ = env.reset(seed=seed)
    navs = [float(obs["nav"])]
    turnovers: list[float] = []
    done = False
    while not done:
        if env.is_rebalance_step():
            sig = env.day_index - 1
            target = allocate(
                r_hat[sig],
                vol[sig],
                obs["mask"].astype(bool),
                obs["sector_ids"].astype(np.int64),
                obs["portfolio"].astype(np.float64),
                params,
            )
            obs, _, terminated, truncated, info = env.step_weights(target)
        else:
            # Hold day: the env ignores the action, so its content is
            # irrelevant — but `step` is the cheaper call.
            obs, _, terminated, truncated, info = env.step(
                np.zeros(len(env.universe) + 1, dtype=np.float32)
            )
        navs.append(float(info["nav"]))
        turnovers.append(float(info["turnover"]))
        done = terminated or truncated
    return navs, turnovers


def _read_gate(signal_dir: Path) -> tuple[str, str]:
    """``(verdict, detail)`` from ``<signal_dir>/gate.json``.

    A missing or unparseable gate is treated as **not** a pass: the pinned R4
    artefact set includes gate.json, so its absence means the signal was not
    produced by a completed walk-forward and nothing is known about it.
    """
    import json
    from pathlib import Path as _Path

    path = _Path(signal_dir) / "gate.json"
    if not path.exists():
        return "MISSING", f"no gate.json at {path}"
    try:
        payload = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        return "UNREADABLE", f"{path}: {exc}"
    verdict = str(payload.get("verdict", "MISSING"))
    detail = (
        f"mean_ic_5d={payload.get('mean_ic_5d')} "
        f"mean_ic_20d={payload.get('mean_ic_20d')} "
        f"ic_ci_low={payload.get('ic_ci_low')} "
        f"n_windows={payload.get('n_windows')}"
    )
    reasons = payload.get("reasons")
    if verdict != "PASS" and reasons:
        detail += f" reasons={list(reasons)[:3]}"
    return verdict, detail


def _null_signal(
    shape: tuple[int, int], seed: int, support: np.ndarray | None = None
) -> np.ndarray:
    """Seeded white noise with the shape of a signal — the control arm.

    Deliberately not "no trading": a null *signal* still ranks, still picks K
    names and still pays the cadence's turnover bill, so it prices the cadence
    with the selection held at zero information.

    ``support`` masks the noise to the *real* signal's finite cells.  R4's
    predictions are OOS-only and tradeable-only, so they are NaN over much of
    the [T, N] grid (measured 28.4% populated against the real val panel).
    Dense noise would hand the control arm a candidate set the arm it controls
    never sees — a bigger, differently-shaped universe on every date — and the
    comparison would price the universe, not the signal.
    """
    import numpy as np

    out = np.random.default_rng(seed).normal(0.0, 0.02, shape)
    if support is not None:
        out = np.where(np.isfinite(support), out, np.nan)
    return out


def _run_baseline(env: PanelTradingEnv, seed: int) -> tuple[list[float], list[float]]:
    """`EqualWeightRebalanced` through the same env, so the cadence is identical."""
    from trader.env.baselines import EqualWeightRebalanced

    agent = EqualWeightRebalanced()
    obs, _ = env.reset(seed=seed)
    agent.reset()
    navs = [float(obs["nav"])]
    turnovers: list[float] = []
    done = False
    while not done:
        obs, _, terminated, truncated, info = env.step(agent.act(obs))
        navs.append(float(info["nav"]))
        turnovers.append(float(info["turnover"]))
        done = terminated or truncated
    return navs, turnovers


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    from pathlib import Path

    import hydra.utils
    import mlflow
    import polars as pl

    from trader.allocator import AllocatorParams, RebalanceSchedule
    from trader.data.features import FEATURE_COLS, resolve_panels_root
    from trader.data.universe import active_tickers
    from trader.env.costs import DEFAULT_MIN_TRADE_VALUE
    from trader.env.panel_env import PanelTradingEnv
    from trader.training.eval_metrics import compute_episode_metrics
    from trader.training.quantstats_report import compute_quantstats_metrics

    orig_cwd = Path(hydra.utils.get_original_cwd())
    panels_root = resolve_panels_root(cfg, orig_cwd)
    universe = active_tickers()
    seed = int(cfg.get("seed", 42))

    split = str(cfg.get("split", "val"))
    panel_path = panels_root / f"{split}.parquet"
    if not panel_path.exists():
        _fail(f"No panel at {panel_path}. Run scripts/build_features.py first.")

    tag = str(cfg.get("signal_tag", "default"))
    signal_dir = orig_cwd / "data" / "signal" / tag
    pred_path = signal_dir / "predictions.parquet"
    if not pred_path.exists():
        _fail(
            f"No signal at {pred_path}.\n"
            f"  The allocator has no opinion of its own — it ranks r_hat, so with no\n"
            f"  predictions there is nothing to rank and this script will not invent\n"
            f"  any.  Produce them with the R4 supervised cross-sectional model\n"
            f"  (`09_revamp_and_audit.md` §10, run Q4), which writes\n"
            f"  data/signal/<tag>/{{predictions.parquet, embeddings.npy, index.json,\n"
            f"  gate.json}}.  Then re-run with `+signal_tag={tag}`.\n"
            f"  Available tags: "
            f"{sorted(p.name for p in (orig_cwd / 'data' / 'signal').glob('*')) or 'none'}"
        )

    # Read the R4 gate BEFORE running anything.  Without this the script
    # happily prints a "vs EW Δ CAGR" for a signal whose own gate said FAIL,
    # with no trace of it anywhere in the output or in MLflow — exactly the
    # CLAUDE.md rule 1 failure ("never call a result validated without a run
    # ID", and never let a failed gate quietly become a headline number).
    gate_verdict, gate_detail = _read_gate(signal_dir)
    require_pass = bool(cfg.get("require_gate_pass", True))
    if gate_verdict != "PASS":
        message = (
            f"R4 gate for signal tag {tag!r} is {gate_verdict}: {gate_detail}\n"
            f"  A backtest of a failed signal is not evidence, and its 'vs EW Δ\n"
            f"  CAGR' must not be quoted as one.  Re-run with\n"
            f"  `+require_gate_pass=false` to measure it anyway — the verdict is\n"
            f"  then stamped into every MLflow run and into the printed header."
        )
        if require_pass:
            _fail(message)
        logger.warning(message)

    alloc_cfg = cfg.get("allocator", {})
    k_grid = [int(k) for k in alloc_cfg.get("k_grid", _K_GRID)]
    freq_grid = [str(f) for f in alloc_cfg.get("freq_grid", _FREQ_GRID)]
    horizon_grid = [str(h) for h in alloc_cfg.get("horizon_grid", _HORIZON_GRID)]

    # Env span: one deterministic full-length pass over the split.  With
    # `episode_length == n_dates - lookback - 1`, `reset` samples `start_idx`
    # from a single-element range, so the backtest is the whole split and the
    # seed cannot move it.
    lookback = int(cfg.env.lookback_days)
    dates = sorted(pl.read_parquet(panel_path, columns=["date"])["date"].unique().to_list())
    episode_length = len(dates) - lookback - 1
    if episode_length < 2:
        _fail(f"{panel_path} has {len(dates)} dates; too short for lookback={lookback}.")

    r_hat_all = _dense_signal(pred_path, dates, universe, tuple(horizon_grid))
    from trader.env.allocator_env import vol_column_for

    vol_lookback = int(alloc_cfg.get("vol_lookback", _DEFAULT_VOL_LOOKBACK))
    vol_col = vol_column_for(vol_lookback)
    vol_all = _dense_column(panel_path, vol_col, dates, universe)
    if not bool((vol_all > 0).any()):
        _fail(
            f"`{vol_col}` is non-positive everywhere in {panel_path}. "
            "A dead vol channel makes every name uninvestable (CLAUDE.md, "
            "feature liveness)."
        )

    env_base: dict[str, Any] = dict(
        panel_path=panel_path,
        universe=universe,
        feature_columns=FEATURE_COLS,
        lookback=lookback,
        episode_length=episode_length,
        initial_cash=float(cfg.env.initial_cash),
        # Reads the `min_trade_value: 500` key that CLAUDE.md lists as declared
        # and never read. The env now gates execution on it, matching the paper
        # broker; leaving it unread is what produced the 11% NAV divergence.
        min_trade_value=float(cfg.env.get("min_trade_value", DEFAULT_MIN_TRADE_VALUE)),
        seed=seed,
    )

    mlflow.set_tracking_uri(f"http://localhost:{cfg.get('mlflow_port', 5555)}")
    mlflow.set_experiment("allocator")

    header = (
        f"{'strategy':<26}{'K':>4}{'freq':>9}{'hor':>5}"
        f"{'Sharpe':>9}{'CAGR':>9}{'MDD':>9}{'Turn':>9}{'vs EW Δ CAGR':>14}"
    )
    rule = "-" * len(header)
    # The verdict rides on the printed table, so a number lifted out of this
    # output carries its provenance with it.
    banner = f"R4 signal gate [{tag}]: {gate_verdict} — {gate_detail}"
    if gate_verdict != "PASS":
        banner += "\n*** SIGNAL GATE DID NOT PASS: these numbers are NOT evidence ***"
    lines: list[str] = [rule, banner, rule, header, rule]

    for freq in freq_grid:
        schedule = None if freq == "daily" else RebalanceSchedule(freq)  # type: ignore[arg-type]

        # The bar, at this cadence.  Recomputed per freq on purpose: an
        # allocator that only beats *daily* equal-weight has beaten the cadence,
        # not the universe.
        bl_env = PanelTradingEnv(rebalance_schedule=schedule, **env_base)
        bl_navs, bl_turns = _run_baseline(bl_env, seed)
        bl = compute_episode_metrics(bl_navs, bl_turns)
        lines.append(
            f"{'equal_weight':<26}{'-':>4}{freq:>9}{'-':>5}"
            f"{bl.sharpe:>9.3f}{bl.cagr:>9.3f}{bl.max_drawdown:>9.3f}"
            f"{bl.turnover_ann:>9.3f}{'—':>14}"
        )
        with mlflow.start_run(run_name=f"equal_weight_{freq}_{split}"):
            mlflow.log_params(
                {"strategy": "equal_weight", "freq": freq, "split": split,
                 "signal_tag": tag, "universe_size": len(universe), "seed": seed,
                 "signal_gate_verdict": gate_verdict}
            )
            mlflow.log_metrics(
                {f"{split}/sharpe": bl.sharpe, f"{split}/cagr": bl.cagr,
                 f"{split}/max_drawdown": bl.max_drawdown,
                 f"{split}/turnover_ann": bl.turnover_ann,
                 f"{split}/total_return": bl.total_return}
            )
            for name, value in compute_quantstats_metrics(bl.daily_returns).items():
                mlflow.log_metric(f"qs/{split}/{name}", value)

        if bool(alloc_cfg.get("null_control", False)):
            nk = k_grid[len(k_grid) // 2]
            env = PanelTradingEnv(rebalance_schedule=schedule, **env_base)
            nm = compute_episode_metrics(*_run_allocator(
                env,
                # Masked to the real signal's support so the control arm and
                # the arm it controls face the same candidate set each day.
                _null_signal(
                    (len(dates), len(universe)), seed,
                    support=r_hat_all[horizon_grid[0]],
                ),
                vol_all,
                AllocatorParams(k=nk, vol_lookback=vol_lookback),
                seed,
            ))
            lines.append(
                f"{'null_signal (control)':<26}{nk:>4}{freq:>9}{'-':>5}"
                f"{nm.sharpe:>9.3f}{nm.cagr:>9.3f}{nm.max_drawdown:>9.3f}"
                f"{nm.turnover_ann:>9.3f}{nm.cagr - bl.cagr:>+14.4f}"
            )
            with mlflow.start_run(run_name=f"null_signal_k{nk}_{freq}_{split}"):
                mlflow.log_params(
                    {"strategy": "null_signal", "k": nk, "freq": freq,
                     "split": split, "signal_tag": tag, "seed": seed,
                     "signal_gate_verdict": gate_verdict}
                )
                mlflow.log_metrics(
                    {f"{split}/sharpe": nm.sharpe, f"{split}/cagr": nm.cagr,
                     f"{split}/max_drawdown": nm.max_drawdown,
                     f"{split}/turnover_ann": nm.turnover_ann,
                     f"{split}/cagr_minus_equal_weight": nm.cagr - bl.cagr}
                )

        for horizon in horizon_grid:
            for k in k_grid:
                params = AllocatorParams(
                    k=k,
                    max_name_weight=float(alloc_cfg.get("max_name_weight", 0.10)),
                    max_sector_weight=float(alloc_cfg.get("max_sector_weight", 0.25)),
                    turnover_budget=float(alloc_cfg.get("turnover_budget", 0.30)),
                    cash_floor=float(alloc_cfg.get("cash_floor", 0.0)),
                    vol_lookback=vol_lookback,
                )
                env = PanelTradingEnv(rebalance_schedule=schedule, **env_base)
                navs, turns = _run_allocator(
                    env, r_hat_all[horizon], vol_all, params, seed
                )
                m = compute_episode_metrics(navs, turns)
                lines.append(
                    f"{'allocator':<26}{k:>4}{freq:>9}{horizon:>5}"
                    f"{m.sharpe:>9.3f}{m.cagr:>9.3f}{m.max_drawdown:>9.3f}"
                    f"{m.turnover_ann:>9.3f}{m.cagr - bl.cagr:>+14.4f}"
                )
                with mlflow.start_run(
                    run_name=f"allocator_k{k}_{freq}_{horizon}_{split}"
                ):
                    mlflow.log_params(
                        {"strategy": "allocator", "k": k, "freq": freq,
                         "horizon": horizon, "split": split, "signal_tag": tag,
                         "max_name_weight": params.max_name_weight,
                         "max_sector_weight": params.max_sector_weight,
                         "turnover_budget": params.turnover_budget,
                         "cash_floor": params.cash_floor,
                         "universe_size": len(universe), "seed": seed,
                         "signal_gate_verdict": gate_verdict}
                    )
                    mlflow.log_metrics(
                        {f"{split}/sharpe": m.sharpe, f"{split}/cagr": m.cagr,
                         f"{split}/max_drawdown": m.max_drawdown,
                         f"{split}/turnover_ann": m.turnover_ann,
                         f"{split}/total_return": m.total_return,
                         f"{split}/cagr_minus_equal_weight": m.cagr - bl.cagr,
                         f"{split}/sharpe_minus_equal_weight": m.sharpe - bl.sharpe}
                    )
                    for name, value in compute_quantstats_metrics(
                        m.daily_returns
                    ).items():
                        mlflow.log_metric(f"qs/{split}/{name}", value)

        lines.append(rule)

    logger.info("\n" + "\n".join(lines))
    logger.info(
        f"{len(freq_grid) * (1 + len(horizon_grid) * len(k_grid))} runs logged to "
        f"MLflow experiment 'allocator' (split={split}, signal_tag={tag}). "
        "A row here is a measurement, not a verdict: nothing is a winner without "
        "a walk-forward and a run ID (CLAUDE.md rule 2)."
    )


if __name__ == "__main__":
    main()
