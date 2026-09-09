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

Every arm also reports the number the flat demat debit is actually billed on —
distinct scrips sold per rebalance, and the rupees that costs — because turnover
is the wrong instrument for a flat per-scrip fee and reading only the turnover
column is how the daily arm's 17%/yr fee bill went unnoticed
(`11_cost_defect_and_fix_plan.md`).  `+allocator.band_grid=` sweeps P3's
per-name no-trade band, which is the only control in the system on that count.

Usage
-----
    uv run python scripts/run_allocator.py                        # val split
    uv run python scripts/run_allocator.py +split=test
    uv run python scripts/run_allocator.py +signal_tag=r4_v2
    uv run python scripts/run_allocator.py \
        +allocator.k_grid=[30] +allocator.freq_grid=[monthly]
    uv run python scripts/run_allocator.py \
        +allocator.band_grid=[0.0,0.002,0.005,0.010]   # P3 band sweep
    uv run python scripts/run_allocator.py +allocator.no_trade_band=0.005

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
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import hydra
from loguru import logger
from omegaconf import DictConfig

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np

    from trader.allocator import AllocatorParams
    from trader.allocator.risk import RiskOverlay  # pragma: no cover
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


#: Pinned column order and dtype names for the stop-event log. Kept as strings
#: because polars is imported lazily in this module; `_stop_event_schema()`
#: turns them into a real schema at the one place that needs one. An arm can
#: legitimately fire no stops, and an empty frame with no schema is unreadable.
_STOP_EVENT_COLUMNS: Final[tuple[tuple[str, str], ...]] = (
    ("day_index", "Int64"),
    ("name_index", "Int64"),
    ("ticker", "Utf8"),
    ("entry_price", "Float64"),
    ("close", "Float64"),
    ("loss", "Float64"),
    ("threshold", "Float64"),
    ("weight", "Float64"),
    ("nav", "Float64"),
)


def alloc_cfg_universe_from_panel(cfg: DictConfig) -> bool:
    """Whether to take the traded universe from the panel rather than the list.

    Read through a helper so the flag is greppable: the same defaulted-wrong
    universe cost `scripts/run_baselines.py` a silent no-op, and a bare
    ``cfg.allocator.get(...)`` buried in `main` is easy to miss when auditing
    which script measures which universe.
    """
    alloc = cfg.get("allocator", {})
    return bool(alloc.get("universe_from_panel", False))


def _stop_event_schema() -> dict[str, Any]:
    import polars as pl

    return {name: getattr(pl, dtype)() for name, dtype in _STOP_EVENT_COLUMNS}


def _write_navs(nav_dir: str, tag: str, navs: list[float], dates: list[Any]) -> None:
    """Dump one arm's NAV path beside its dates.

    Written per arm rather than as one wide frame because arms are produced in
    a nested loop and a partial run should still leave usable files.
    """
    import polars as pl

    out = Path(nav_dir) / f"nav_{tag}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    # navs carries the opening NAV plus one per step, so it is one longer than
    # the stepped dates. Trimming the dates rather than the NAVs keeps the
    # opening value, which every return series has to start from.
    n = len(navs)
    d = list(dates[:n]) if len(dates) >= n else list(dates) + [None] * (n - len(dates))
    pl.DataFrame({"date": d, "nav": navs}).write_parquet(out)


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
    risk: RiskOverlay | None = None,
    stop_events: list[dict[str, Any]] | None = None,
) -> tuple[list[float], list[float], dict[str, float]]:
    """Drive one full pass of the env from the allocator.

    Returns ``(navs, turnovers, diagnostics)``.

    Information set
    ---------------
    The env trades day ``d`` at its **open** from an observation whose feature
    window ends at ``d-1``'s close (`panel_env._build_obs` slices
    ``[start:day_idx]``).  So the signal row used to trade day ``d`` is the one
    dated ``d-1``.  Indexing ``r_hat[day_idx]`` instead would trade on a
    prediction made from data that includes the day being traded — a one-day
    lookahead worth more than the entire strategy.

    Diagnostics
    -----------
    Turnover is the wrong instrument for this system's dominant cost.  The demat
    debit is **flat** — ₹15.34 per distinct scrip per selling day, 83-98% of all
    cost measured (`11_cost_defect_and_fix_plan.md`) — so the quantity that sets
    the bill is the *count of names sold*, which no turnover figure reveals.
    ``scrip_sell_days`` is that count, taken from ``info["n_scrips_sold"]``,
    i.e. from the orders the env actually executed rather than from a
    weight-space estimate of them.

    ``names_suppressed`` is the band's counterfactual: how many orders
    ``no_trade_band`` kept out of the list, from
    :func:`trader.allocator.band_suppression`, which re-runs ``allocate`` with
    the band forced to zero.  It costs two extra allocator calls per rebalance
    day and is therefore computed **only when the band is on**.  It is an upper
    bound on orders removed — the env can still emit a one-share order on a
    pinned name when an overnight gap moves its weight by more than half a share
    (`tests/unit/test_weight_to_share_orders.py`) — so ``scrip_sell_days``, not
    this, is what a fee argument must be made on.
    """
    import numpy as np

    from trader.allocator import allocate, band_suppression
    from trader.env.costs import _DP_CHARGE

    track_band = params.no_trade_band > 0.0
    stops_fired = 0
    forced_days = 0
    obs, _ = env.reset(seed=seed)
    navs = [float(obs["nav"])]
    turnovers: list[float] = []
    rebalances = 0
    scrip_sell_days = 0
    legs = 0
    suppressed = 0
    traded_unbanded = 0
    done = False
    while not done:
        if env.is_rebalance_step():
            sig = env.day_index - 1
            mask = obs["mask"].astype(bool)
            sids = obs["sector_ids"].astype(np.int64)
            cur = obs["portfolio"].astype(np.float64)
            target = allocate(r_hat[sig], vol[sig], mask, sids, cur, params)
            if track_band:
                rep = band_suppression(r_hat[sig], vol[sig], mask, sids, cur, params)
                suppressed += rep.n_suppressed
                traded_unbanded += rep.n_traded_unbanded
            if risk is not None:
                target = risk.apply(target)
                risk.register_stops()
            rebalances += 1
            obs, _, terminated, truncated, info = env.step_weights(target)
        elif risk is not None and risk.stops_to_execute().any():
            # A stop must act NOW. Waiting for the next scheduled rebalance is
            # not a stop-loss, so this is the one caller allowed to force a
            # trade on a hold day. Everything else about the cadence is
            # unchanged: the book is carried, only the stopped names are sold.
            # `obs["portfolio"]` is float32 and the env's Box(0,1) clip lets it
            # sum to ~1.0098 when cash has gone negative, so the surviving block
            # must be renormalised before it can be handed back as a target —
            # `step_weights` requires an exact sum of 1 and rightly rejects
            # anything else.
            cur = obs["portfolio"].astype(np.float64)
            eq = np.maximum(np.where(risk.stops_to_execute(), 0.0, cur[1:]), 0.0)
            tot = float(eq.sum())
            if tot > 1.0:
                eq = eq / tot
                tot = 1.0
            target = np.empty(len(cur), dtype=np.float64)
            target[1:] = eq
            target[0] = 1.0 - tot
            if stop_events is not None:
                # Recorded before register_stops() clears the mask and before
                # the sale releases the entry price. `day_index` is the row
                # about to be stepped, so the event is stamped with the day the
                # stop acts on, not the day it was detected.
                fired = np.flatnonzero(risk.stops_to_execute())
                entry = risk.entry_prices()
                closes = env.closes_today()
                thr = risk.stop_thresholds()
                for j in fired:
                    stop_events.append({
                        "day_index": int(env.day_index),
                        "name_index": int(j),
                        "ticker": str(env.universe[j]),
                        "entry_price": float(entry[j]),
                        "close": float(closes[j]),
                        "loss": float(closes[j] / max(entry[j], 1e-12) - 1.0),
                        "threshold": float(thr[j]),
                        "weight": float(cur[1 + j]),
                        "nav": float(obs["nav"]),
                    })
            stops_fired += risk.register_stops()
            forced_days += 1
            obs, _, terminated, truncated, info = env.step_weights(target, force=True)
        else:
            # Hold day: the env ignores the action, so its content is
            # irrelevant — but `step` is the cheaper call.
            obs, _, terminated, truncated, info = env.step(
                np.zeros(len(env.universe) + 1, dtype=np.float32)
            )
        if risk is not None:
            risk.update(
                float(info["nav"]),
                env.closes_today(),
                obs["portfolio"].astype(np.float64)[1:],
                fill_prices=env.last_fill_prices(),
                # Per-name annualised vol for a volatility-scaled stop. Taken at
                # the day just stepped, never the day ahead.
                vol_ann=vol[max(env.day_index - 1, 0)],
            )
        navs.append(float(info["nav"]))
        turnovers.append(float(info["turnover"]))
        scrip_sell_days += int(info["n_scrips_sold"])
        legs += int(info["n_legs"])
        done = terminated or truncated

    per = max(rebalances, 1)
    diag = {
        "rebalances": float(rebalances),
        "scrip_sell_days": float(scrip_sell_days),
        "scrip_sell_days_per_rebalance": scrip_sell_days / per,
        "legs": float(legs),
        "dp_charges_paid": scrip_sell_days * _DP_CHARGE,
    }
    if track_band:
        diag["names_suppressed_per_rebalance"] = suppressed / per
        diag["names_traded_unbanded_per_rebalance"] = traded_unbanded / per
    diag["tax_paid"] = float(getattr(env, "tax_paid", 0.0))
    diag["tax_accrued_unpaid"] = float(
        env.tax_accrued_unpaid() if hasattr(env, "tax_accrued_unpaid") else 0.0
    )
    if risk is not None:
        diag["stops_fired"] = float(stops_fired)
        diag["forced_stop_days"] = float(forced_days)
        diag["final_exposure_scale"] = float(risk.state.exposure_scale)
        diag["final_realised_vol"] = float(risk.state.realised_vol)
    return navs, turnovers, diag


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


def _run_baseline(
    env: PanelTradingEnv, seed: int
) -> tuple[list[float], list[float], dict[str, float]]:
    """`EqualWeightRebalanced` through the same env, so the cadence is identical.

    Same diagnostics as :func:`_run_allocator`: the baseline's scrip-sell-day
    count is the number the allocator's has to be compared against, and it is
    the one `11_cost_defect_and_fix_plan.md` shows is 97.7% of the daily arm's
    entire cost.  It has no band, so no suppression counterfactual.
    """
    from trader.env.baselines import EqualWeightRebalanced, run_baseline_episode

    # Delegates to the library so this baseline and the same one in
    # `scripts/run_baselines.py` are literally the same computation.
    return run_baseline_episode(env, EqualWeightRebalanced(), seed)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    from pathlib import Path

    import hydra.utils
    import mlflow
    import polars as pl

    from trader.allocator import (
        AllocatorParams,
        RebalanceSchedule,
        fee_drag_estimate,
        max_supportable_k,
    )
    from trader.allocator.risk import RiskOverlay, RiskParams
    from trader.allocator.sizing import REBALANCES_PER_YEAR
    from trader.data.features import FEATURE_COLS, resolve_panels_root
    from trader.data.universe import resolve_traded_universe
    from trader.env.costs import DEFAULT_MIN_TRADE_VALUE
    from trader.env.panel_env import PanelTradingEnv
    from trader.training.eval_metrics import compute_episode_metrics
    from trader.training.quantstats_report import compute_quantstats_metrics

    orig_cwd = Path(hydra.utils.get_original_cwd())
    panels_root = resolve_panels_root(cfg, orig_cwd)
    seed = int(cfg.get("seed", 42))

    split = str(cfg.get("split", "val"))
    panel_path = panels_root / f"{split}.parquet"
    if not panel_path.exists():
        _fail(f"No panel at {panel_path}. Run scripts/build_features.py first.")

    # WHERE THE UNIVERSE COMES FROM, and why the default is not safe everywhere.
    #
    # The env is built over `universe` and can only ever see those columns.
    # Running this against the point-in-time panel while taking the universe
    # from `active_tickers()` would measure the OLD fixed 504 names on the new
    # bars — and print a table that looks entirely normal while answering a
    # question nobody asked. `scripts/run_baselines.py` had the identical bug
    # and it would have wasted Phase 1.
    #
    # Default stays `active_tickers()` so every number already recorded against
    # the Kite panels reproduces exactly.
    universe, provenance = resolve_traded_universe(
        panel_path, from_panel=alloc_cfg_universe_from_panel(cfg)
    )
    logger.info(f"universe from {provenance}")

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
    # P3's per-name no-trade band.  `band_grid` sweeps it; `no_trade_band` sets
    # a single value and is the default for the grid, so the two cannot
    # disagree about what is being run (the `vol_column` / `vol_lookback` trap).
    #     +allocator.no_trade_band=0.005
    #     +allocator.band_grid=[0.0,0.002,0.005,0.010]
    # Run 1's risk sweep. Each entry is one ARM, given as a name plus the
    # RiskParams kwargs that define it, so a run's arms are self-describing in
    # MLflow rather than being three anonymous booleans. The default is the
    # single "none" arm, which `RiskOverlay.apply` short-circuits to the
    # identity, so a run that does not ask for risk control is bit-identical to
    # one built before this existed.
    #   +allocator.risk_grid='[{name: none}, {name: vol, vol_target: 0.15}]'
    risk_grid: list[dict[str, Any]] = [
        dict(r) for r in alloc_cfg.get("risk_grid", [{"name": "none"}])
    ]

    band_grid = [
        float(b)
        for b in alloc_cfg.get(
            "band_grid", [float(alloc_cfg.get("no_trade_band", 0.0))]
        )
    ]

    # Env span: one deterministic full-length pass over the split.  With
    # `episode_length == n_dates - lookback - 1`, `reset` samples `start_idx`
    # from a single-element range, so the backtest is the whole split and the
    # seed cannot move it.
    lookback = int(cfg.env.lookback_days)
    dates = sorted(pl.read_parquet(panel_path, columns=["date"])["date"].unique().to_list())
    episode_length = len(dates) - lookback - 1
    if episode_length < 2:
        _fail(f"{panel_path} has {len(dates)} dates; too short for lookback={lookback}.")

    # How many of the env's universe this split can actually trade.  `universe`
    # is always `active_tickers()` (504) regardless of split, so logging only
    # `universe_size` tagged a restricted P4 arm identically to an unrestricted
    # run — a run ID that does not say what it ran (CLAUDE.md rule 2).  A
    # universe ticker with no rows here is an all-zero column, mask False on
    # every date, so this is the width both arms really see.
    panel_tickers = set(
        pl.read_parquet(panel_path, columns=["ticker"])["ticker"].unique().to_list()
    )
    n_effective = len(panel_tickers & set(universe))
    logger.info(
        f"universe {len(universe)} from active_tickers(); {n_effective} of them have "
        f"rows in {panel_path} — the rest are all-zero columns, mask False."
    )

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

    # Read ONCE and reused by the env, by P5's capacity check and by the MLflow
    # params, so the three cannot disagree about what the run was configured
    # with.  `min_trade_value: 500` is the key CLAUDE.md lists as declared and
    # never read; the env now gates execution on it, matching the paper broker.
    capital = float(cfg.env.initial_cash)
    min_trade_value = float(cfg.env.get("min_trade_value", DEFAULT_MIN_TRADE_VALUE))
    cash_floor = float(alloc_cfg.get("cash_floor", 0.0))
    # Where to write per-stop diagnostics. Unset = do not record.
    stop_events_dir = alloc_cfg.get("stop_events_dir", None)
    # Where to dump each arm's NAV path. Unset = do not write. R5's gate asks
    # for a PAIRED bootstrap CI against the baseline, and a paired test needs
    # both NAV series day by day -- a summary CAGR cannot produce one.
    nav_dir = alloc_cfg.get("nav_dir", None)

    env_base: dict[str, Any] = dict(
        panel_path=panel_path,
        universe=universe,
        feature_columns=FEATURE_COLS,
        lookback=lookback,
        episode_length=episode_length,
        initial_cash=capital,
        min_trade_value=min_trade_value,
        # Capital-gains tax inside the loop. Off by default so every result
        # predating 2026-09-07 still reproduces; +apply_tax=true turns it on.
        apply_tax=bool(cfg.get("apply_tax", False)),
        seed=seed,
    )

    # ── P5: does this capital support the K being measured? ───────────────────
    # A WARNING, never a refusal.  Measuring an unsupportable K is a legitimate
    # thing to want — it is how you show what the constraint costs — and a
    # script that refused would make that measurement impossible.  What is not
    # legitimate is measuring one without knowing, so the warning is emitted
    # once per K here and the bound is stamped into every MLflow run below.
    supported_k = max_supportable_k(
        capital,
        min_trade_value=min_trade_value,
        smallest_position_share=1.0 - cash_floor,
    )
    for k in k_grid:
        if k > supported_k:
            logger.warning(
                f"k={k} exceeds max_supportable_k={supported_k} at "
                f"initial_cash={capital:,.0f} (min_trade_value={min_trade_value:,.0f}, "
                f"cash_floor={cash_floor}): a 20% trim of one position would not "
                f"clear min_trade_value, so those names can only be held or fully "
                f"exited and the allocator's targets stop being reachable. "
                f"Measuring it anyway — see P5, src/trader/allocator/sizing.py."
            )

    mlflow.set_tracking_uri(f"http://localhost:{cfg.get('mlflow_port', 5555)}")
    mlflow.set_experiment("allocator")

    header = (
        f"{'strategy':<26}{'K':>4}{'band':>7}{'freq':>9}{'hor':>5}"
        f"{'Sharpe':>9}{'CAGR':>9}{'MDD':>9}{'Turn':>9}"
        f"{'sold/reb':>10}{'DP ₹':>11}{'vs EW Δ CAGR':>14}"
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
        bl_navs, bl_turns, bl_diag = _run_baseline(bl_env, seed)
        if nav_dir:
            _write_navs(nav_dir, f"equal_weight_{freq}_{split}", bl_navs, dates[lookback:])
        bl = compute_episode_metrics(bl_navs, bl_turns)
        lines.append(
            f"{'equal_weight':<26}{'-':>4}{'-':>7}{freq:>9}{'-':>5}"
            f"{bl.sharpe:>9.3f}{bl.cagr:>9.3f}{bl.max_drawdown:>9.3f}"
            f"{bl.turnover_ann:>9.3f}"
            f"{bl_diag['scrip_sell_days_per_rebalance']:>10.1f}"
            f"{bl_diag['dp_charges_paid']:>11,.0f}{'—':>14}"
        )
        with mlflow.start_run(run_name=f"equal_weight_{freq}_{split}"):
            mlflow.log_params(
                {"strategy": "equal_weight", "freq": freq, "split": split,
                 "signal_tag": tag, "universe_size": len(universe),
                 "universe_effective": n_effective,
                 "initial_cash": capital, "min_trade_value": min_trade_value,
                 "max_supportable_k": supported_k, "seed": seed,
                 "signal_gate_verdict": gate_verdict}
            )
            mlflow.log_metrics(
                {f"{split}/sharpe": bl.sharpe, f"{split}/cagr": bl.cagr,
                 f"{split}/max_drawdown": bl.max_drawdown,
                 f"{split}/turnover_ann": bl.turnover_ann,
                 f"{split}/total_return": bl.total_return,
                 **{f"{split}/{n}": v for n, v in bl_diag.items()}}
            )
            for name, value in compute_quantstats_metrics(bl.daily_returns).items():
                mlflow.log_metric(f"qs/{split}/{name}", value)

        if bool(alloc_cfg.get("null_control", False)):
            # The control must run the SAME machinery as the arms it controls.
            # Until 2026-09-09 it ran one arm at band_grid[0] with NO risk
            # overlay, while every real arm carried a band and a stop -- so the
            # comparison priced the signal PLUS the band PLUS the stop against
            # nothing, and the risk numbers in audit/navs_long had no matched
            # control at all. It now sweeps the same risk_grid x band_grid.
            nk = k_grid[len(k_grid) // 2]
            nhor = horizon_grid[0]
            for nrisk in risk_grid:
                for nband in band_grid:
                    nrisk_name = str(nrisk.get("name", "none"))
                    nrisk_kw = {k2: v for k2, v in nrisk.items() if k2 != "name"}
                    noverlay = (
                        RiskOverlay(RiskParams(**nrisk_kw), len(universe))
                        if nrisk_kw
                        else None
                    )
                    env = PanelTradingEnv(rebalance_schedule=schedule, **env_base)
                    n_navs, n_turns, n_diag = _run_allocator(
                        env,
                        # Masked to the real signal's support so the control arm
                        # and the arm it controls face the same candidate set.
                        _null_signal(
                            (len(dates), len(universe)), seed,
                            support=r_hat_all[nhor],
                        ),
                        vol_all,
                        AllocatorParams(
                            k=nk, no_trade_band=nband, vol_lookback=vol_lookback
                        ),
                        seed,
                        risk=noverlay,
                    )
                    nm = compute_episode_metrics(n_navs, n_turns)
                    if nav_dir:
                        _write_navs(
                            nav_dir,
                            f"null_signal_k{nk}_b{nband}_r{nrisk_name}"
                            f"_{freq}_{nhor}_{split}",
                            n_navs, dates[lookback:],
                        )
                    lines.append(
                        f"{'null/' + nrisk_name + ' (control)':<26}{nk:>4}"
                        f"{nband:>7.3f}{freq:>9}{'-':>5}"
                        f"{nm.sharpe:>9.3f}{nm.cagr:>9.3f}{nm.max_drawdown:>9.3f}"
                        f"{nm.turnover_ann:>9.3f}"
                        f"{n_diag['scrip_sell_days_per_rebalance']:>10.1f}"
                        f"{n_diag['dp_charges_paid']:>11,.0f}"
                        f"{nm.cagr - bl.cagr:>+14.4f}"
                    )
                    with mlflow.start_run(
                        run_name=f"null_signal_k{nk}_b{nband}_r{nrisk_name}"
                                 f"_{freq}_{split}"
                    ):
                        mlflow.log_params(
                            {"strategy": "null_signal", "k": nk, "freq": freq,
                             "no_trade_band": nband, "risk_arm": nrisk_name,
                             "split": split, "signal_tag": tag, "seed": seed,
                             "universe_size": len(universe),
                             "universe_effective": n_effective,
                             "initial_cash": capital,
                             "min_trade_value": min_trade_value,
                             "max_supportable_k": supported_k,
                             "signal_gate_verdict": gate_verdict}
                        )
                        mlflow.log_metrics(
                            {f"{split}/sharpe": nm.sharpe, f"{split}/cagr": nm.cagr,
                             f"{split}/max_drawdown": nm.max_drawdown,
                             f"{split}/turnover_ann": nm.turnover_ann,
                             f"{split}/cagr_minus_equal_weight": nm.cagr - bl.cagr,
                             **{f"{split}/{n}": v for n, v in n_diag.items()}}
                        )

        for horizon in horizon_grid:
            for k in k_grid:
              for risk_arm in risk_grid:
                for band in band_grid:
                    params = AllocatorParams(
                        k=k,
                        max_name_weight=float(alloc_cfg.get("max_name_weight", 0.10)),
                        max_sector_weight=float(
                            alloc_cfg.get("max_sector_weight", 0.25)
                        ),
                        turnover_budget=float(alloc_cfg.get("turnover_budget", 0.30)),
                        no_trade_band=band,
                        cash_floor=cash_floor,
                        vol_lookback=vol_lookback,
                    )
                    env = PanelTradingEnv(rebalance_schedule=schedule, **env_base)
                    risk_name = str(risk_arm.get("name", "none"))
                    risk_kw = {k2: v for k2, v in risk_arm.items() if k2 != "name"}
                    overlay = (
                        RiskOverlay(RiskParams(**risk_kw), len(universe))
                        if risk_kw
                        else None
                    )
                    # Stop-event recording is opt-in and off by default: it
                    # costs a dict per stopped name and only makes sense for the
                    # arms that have a stop at all.
                    events: list[dict[str, Any]] | None = (
                        [] if (stop_events_dir and overlay is not None) else None
                    )
                    navs, turns, diag = _run_allocator(
                        env, r_hat_all[horizon], vol_all, params, seed, risk=overlay,
                        stop_events=events,
                    )
                    if events is not None:
                        tag = f"k{k}_b{band}_r{risk_name}_{freq}_{horizon}_{split}"
                        out = Path(stop_events_dir) / f"stop_events_{tag}.parquet"
                        out.parent.mkdir(parents=True, exist_ok=True)
                        # An arm can legitimately fire no stops; write the empty
                        # frame with its schema anyway so a downstream reader
                        # sees "none fired" rather than "file missing".
                        pl.DataFrame(events, schema=_stop_event_schema()).write_parquet(out)
                        logger.info(f"{len(events):,} stop event(s) -> {out}")
                    if nav_dir:
                        _write_navs(
                            nav_dir,
                            f"allocator_k{k}_b{band}_r{risk_name}_{freq}_{horizon}_{split}",
                            navs, dates[lookback:],
                        )
                    m = compute_episode_metrics(navs, turns)
                    # P5's closed form against this run's OWN measured turnover,
                    # so the estimate and the bill are comparable rather than
                    # two unrelated numbers.  It models sells as full exits, so
                    # it is a FLOOR on the flat-fee drag; `dp_charges_paid` is
                    # what the ledger actually paid.
                    drag = fee_drag_estimate(
                        capital, k, REBALANCES_PER_YEAR[freq], m.turnover_ann
                    )
                    lines.append(
                        f"{'allocator/' + risk_name:<26}{k:>4}{band:>7.3f}"
                        f"{freq:>9}{horizon:>5}"
                        f"{m.sharpe:>9.3f}{m.cagr:>9.3f}{m.max_drawdown:>9.3f}"
                        f"{m.turnover_ann:>9.3f}"
                        f"{diag['scrip_sell_days_per_rebalance']:>10.1f}"
                        f"{diag['dp_charges_paid']:>11,.0f}"
                        f"{m.cagr - bl.cagr:>+14.4f}"
                    )
                    with mlflow.start_run(
                        run_name=(
                            f"allocator_k{k}_b{band}_r{risk_name}_"
                            f"{freq}_{horizon}_{split}"
                        )
                    ):
                        mlflow.log_params(
                            {"strategy": "allocator", "k": k, "freq": freq,
                             "horizon": horizon, "split": split, "signal_tag": tag,
                             "max_name_weight": params.max_name_weight,
                             "max_sector_weight": params.max_sector_weight,
                             "turnover_budget": params.turnover_budget,
                             "no_trade_band": params.no_trade_band,
                             "risk_arm": risk_name,
                             "cash_floor": params.cash_floor,
                             "universe_size": len(universe),
                             "universe_effective": n_effective,
                             "initial_cash": capital,
                             "min_trade_value": min_trade_value,
                             "max_supportable_k": supported_k,
                             "k_is_supported": k <= supported_k,
                             "seed": seed,
                             "signal_gate_verdict": gate_verdict}
                        )
                        mlflow.log_metrics(
                            {f"{split}/sharpe": m.sharpe, f"{split}/cagr": m.cagr,
                             f"{split}/max_drawdown": m.max_drawdown,
                             f"{split}/turnover_ann": m.turnover_ann,
                             f"{split}/total_return": m.total_return,
                             f"{split}/cagr_minus_equal_weight": m.cagr - bl.cagr,
                             f"{split}/sharpe_minus_equal_weight": m.sharpe - bl.sharpe,
                             f"{split}/fee_drag_estimate": drag,
                             f"{split}/dp_charges_minus_equal_weight":
                                 diag["dp_charges_paid"] - bl_diag["dp_charges_paid"],
                             **{f"{split}/{n}": v for n, v in diag.items()}}
                        )
                        for name, value in compute_quantstats_metrics(
                            m.daily_returns
                        ).items():
                            mlflow.log_metric(f"qs/{split}/{name}", value)

        lines.append(rule)

    logger.info("\n" + "\n".join(lines))
    logger.info(
        f"{len(freq_grid) * (1 + len(horizon_grid) * len(k_grid) * len(band_grid))} "
        "runs logged to "
        f"MLflow experiment 'allocator' (split={split}, signal_tag={tag}). "
        "A row here is a measurement, not a verdict: nothing is a winner without "
        "a walk-forward and a run ID (CLAUDE.md rule 2)."
    )


if __name__ == "__main__":
    main()
