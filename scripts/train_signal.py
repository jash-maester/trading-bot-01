#!/usr/bin/env python
"""R4 — train the supervised cross-sectional signal model, walk-forward.

Hydra wrapper around :func:`trader.training.supervised.run_signal_walk_forward`,
styled on ``scripts/train.py``.  No PPO, no env, no reward: the model predicts
each stock's cross-sectionally standardised forward log return over 5 and 20
days, and the go/no-go metric is out-of-sample rank IC.
``10_architecture_revamp.md`` §5 is why.

Usage
-----
    # The real thing (needs `full.parquet` under data.panels_root):
    uv run python scripts/train_signal.py model=signal train=supervised

    # The Kite panel, offline (no MLflow), capped for a smoke run:
    uv run python scripts/train_signal.py model=signal train=supervised \
        data=kite_v1 train.mlflow_port=null train.max_steps=20 train.max_epochs=1

Both group overrides are mandatory.  ``configs/config.yaml`` defaults to the RL
stack (``model=mlp_regime train=ppo_baseline``) and this script refuses to run
against it rather than quietly reinterpreting a PPO config as a signal config.

Outputs, under ``<train.out_root>/<train.tag>/`` — the pinned R4 contract:

    predictions.parquet   date, ticker, r_hat_5d, r_hat_20d — OOS rows only
    embeddings.npy        float16 [T, N, D] from the FROZEN encoder
    index.json            dates, tickers, embed_dim, feature_cols, encoder sha256
    gate.json             verdict, pooled OOS mean IC per horizon, CI, ICIR
    signal_model.pt       state_dict behind embeddings.npy
    summary.json          per-window metrics and MLflow run ids

The gate is OOS mean rank IC > 0.02 with the bootstrap CI excluding zero, on
every window.  **A FAIL is a legitimate outcome and is reported as one**
(CLAUDE.md): the script exits 0 either way, and the verdict lives in
``gate.json`` — not in whether the process crashed.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import hydra
from loguru import logger
from omegaconf import DictConfig, OmegaConf


def _as_date(value: object) -> date:
    return value if isinstance(value, date) else date.fromisoformat(str(value))


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    import hydra.utils
    import polars as pl

    from trader.data.features import FEATURE_COLS, resolve_panels_root
    from trader.data.universe import active_tickers
    from trader.models.signal import SignalConfig
    from trader.training.supervised import (
        SupervisedConfig,
        format_verdict,
        run_signal_walk_forward,
    )
    from trader.training.walk_forward import (
        DEFAULT_PURGE_MONTHS,
        compute_windows,
        find_calendar_gaps,
    )

    # ── Refuse the wrong config groups ───────────────────────────────────────
    # `configs/config.yaml` defaults to model=mlp_regime train=ppo_baseline. An
    # RL model config has an actor, a critic and a graph and no `horizons`; read
    # loosely it would still produce *a* model, trained on *a* target, and the
    # run would look fine. Name the mistake instead.
    model_name = str(cfg.model.get("name", ""))
    train_kind = str(cfg.train.get("kind", ""))
    if model_name != "signal" or train_kind != "supervised":
        raise SystemExit(
            f"train_signal.py needs the R4 config groups, got model={model_name!r} "
            f"train.kind={train_kind!r}. Run:\n"
            "  uv run python scripts/train_signal.py model=signal train=supervised"
        )

    orig_cwd = Path(hydra.utils.get_original_cwd())

    # ── The panel, said out loud ─────────────────────────────────────────────
    # Never hardcode data/panels: `data.panels_root` selects the dataset, and the
    # Kite panel was written and orphaned for weeks because entrypoints ignored
    # it. resolve_panels_root logs the resolved path, the splits present, the
    # ticker count and the date span of each.
    panels_root = resolve_panels_root(cfg, orig_cwd)
    full_path = panels_root / "full.parquet"
    if not full_path.exists():
        raise SystemExit(
            f"{full_path} not found. Walk-forward windows are cut from the "
            "un-split panel; the three split parquets do not tile the history "
            "(build_features drops the purge months between them), so "
            "concatenating them slices windows against a calendar with a hole at "
            "every build-time boundary. Run scripts/build_features.py."
        )
    full_panel = pl.read_parquet(full_path).sort(["date", "ticker"])

    gaps = find_calendar_gaps(full_panel)
    if gaps:
        detail = ", ".join(f"{a} → {b} ({n} days)" for a, b, n in gaps)
        raise SystemExit(
            f"{full_path} has {len(gaps)} calendar gap(s): {detail}. Every window "
            "spanning a gap is silently short. Rebuild the panel."
        )

    universe = active_tickers()
    panel_tickers = sorted(full_panel["ticker"].unique().to_list())
    logger.info(
        f"Resolved panel: {full_path}  ({full_panel.shape[0]:,} rows, "
        f"{len(panel_tickers)} tickers, "
        f"{full_panel['date'].min()}..{full_panel['date'].max()})"
    )
    logger.info(
        f"Universe: active_tickers() = {len(universe)} tickers; the model's "
        f"ticker axis is that order, and index.json['tickers'] matches it."
    )
    missing = [t for t in universe if t not in set(panel_tickers)]
    if missing:
        # Not fatal, and not silent. The panels on disk are stale: 163 tickers
        # against 504 in active_tickers(). Absent tickers become all-False
        # is_tradeable columns, so they never enter a loss or a metric — but the
        # run is then measuring a smaller cross-section than the name suggests.
        logger.warning(
            f"{len(missing)} of {len(universe)} universe tickers are absent from "
            f"the panel and will be untradeable everywhere (e.g. "
            f"{', '.join(missing[:8])}{' …' if len(missing) > 8 else ''}). "
            "The effective cross-section is "
            f"{len(universe) - len(missing)} names, not {len(universe)}."
        )

    cfg_cols = cfg.train.get("feature_cols", None)
    feature_cols = list(cfg_cols) if cfg_cols else list(FEATURE_COLS)
    logger.info(f"Features ({len(feature_cols)}): {', '.join(feature_cols)}")

    # ── Walk-forward windows ─────────────────────────────────────────────────
    # compute_windows() raises if the purge is shorter than the longest feature
    # lookback. Do not widen the guard; a shorter purge means the train and val
    # feature windows physically overlap.
    walk_cfg = cfg.get("walk", {})
    # `walk.data_start` / `walk.data_end` bound the walk-forward span independently
    # of the panel span, and default to it. Two reasons they exist:
    #   * the panel opens in 2005, and windows anchored there test 2011-2015 —
    #     eight years of OOS evidence about a market that no longer exists;
    #   * without an upper bound, enough windows walk into the 2025+ modern
    #     holdout that `configs/data/kite_v1.yaml` exists to protect. Capping the
    #     span is what keeps that holdout unseen.
    panel_start = _as_date(full_panel["date"].min())
    panel_end = _as_date(full_panel["date"].max())
    walk_start = _as_date(walk_cfg["data_start"]) if walk_cfg.get("data_start") else panel_start
    walk_end = _as_date(walk_cfg["data_end"]) if walk_cfg.get("data_end") else panel_end
    if walk_start < panel_start or walk_end > panel_end:
        raise SystemExit(
            f"walk.data_start/data_end ({walk_start}..{walk_end}) fall outside the "
            f"panel span ({panel_start}..{panel_end})."
        )
    if walk_start != panel_start or walk_end != panel_end:
        logger.info(
            f"Walk-forward span bounded to {walk_start}..{walk_end} "
            f"(panel is {panel_start}..{panel_end})."
        )
    windows = compute_windows(
        data_start=walk_start,
        data_end=walk_end,
        train_years=int(walk_cfg.get("train_years", 5)),
        val_months=int(walk_cfg.get("val_months", 12)),
        test_months=int(walk_cfg.get("test_months", 12)),
        purge_months=int(walk_cfg.get("purge_months", DEFAULT_PURGE_MONTHS)),
        n_windows=int(walk_cfg.get("n_windows", 4)),
        step_months=int(walk_cfg.get("step_months", 12)),
    )
    if not windows:
        raise SystemExit(
            "No walk-forward window fits in the panel's date range — check "
            "walk.train_years / walk.test_months against the panel span above."
        )
    logger.info(f"{len(windows)} walk-forward window(s):")
    for w in windows:
        logger.info(f"  {w.name}: {w.asdict_iso()}")

    # ── Model and training config ────────────────────────────────────────────
    horizons = tuple(int(h) for h in cfg.train.horizons)
    tcn = cfg.model.get("tcn", {})
    model_cfg = SignalConfig(
        in_features=len(feature_cols),
        embed_dim=int(cfg.model.get("embed_dim", 128)),
        num_channels=[int(c) for c in tcn.get("num_channels", [])] or None,
        kernel_size=int(tcn.get("kernel_size", 3)),
        dropout=float(tcn.get("dropout", 0.1)),
        head_hidden=int(cfg.model.get("head_hidden", 32)),
        horizons=horizons,
    )
    max_steps = cfg.train.get("max_steps", None)
    train_cfg = SupervisedConfig(
        lookback=int(cfg.train.lookback),
        horizons=horizons,
        batch_days=int(cfg.train.batch_days),
        eval_batch_days=int(cfg.train.eval_batch_days),
        max_epochs=int(cfg.train.max_epochs),
        max_steps=None if max_steps is None else int(max_steps),
        learning_rate=float(cfg.train.learning_rate),
        weight_decay=float(cfg.train.weight_decay),
        lr_schedule=str(cfg.train.lr_schedule),
        patience=int(cfg.train.patience),
        max_grad_norm=float(cfg.train.max_grad_norm),
        min_cross_section=int(cfg.train.min_cross_section),
        n_boot=int(cfg.train.n_boot),
        seed=int(cfg.seed),
        device=str(cfg.train.get("device", "auto")),
        xs_normalise=(
            None if cfg.train.get("xs_normalise", None) in (None, "null", "none", "")
            else str(cfg.train.xs_normalise)
        ),
    )

    tag = str(cfg.train.get("tag", "v1"))
    out_dir = orig_cwd / str(cfg.train.get("out_root", "data/signal")) / tag
    mlflow_port_raw = cfg.train.get("mlflow_port", 5555)
    mlflow_port = None if mlflow_port_raw is None else int(mlflow_port_raw)
    logger.info(f"Artefacts → {out_dir}  (MLflow port: {mlflow_port})")

    # ── point-in-time universe, per window ──────────────────────────────────
    # Off unless `train.pit_universe` is configured, so every existing run
    # reproduces. When on, each window picks its own names from bars strictly
    # before its TEST span begins — a decision taken once, at the boundary, the
    # way an index reconstitutes. Not lookahead, and the width stays fixed so
    # the observation cost per run does not move with the universe.
    universe_fn = None
    pit_cfg = cfg.train.get("pit_universe", None)
    if pit_cfg is not None:
        from trader.data.pit_universe import LiquidityRule, eligible_on

        bars_path = orig_cwd / str(pit_cfg.get("bars", "data/ext/bhavcopy.parquet"))
        if not bars_path.exists():
            logger.error(
                f"train.pit_universe is set but {bars_path} does not exist. "
                "Run scripts/fetch_bhavcopy.py first; falling back to the fixed "
                "universe here would silently train the thing this config exists "
                "to avoid."
            )
            raise SystemExit(1)
        pit_bars = pl.read_parquet(
            bars_path, columns=["date", "ticker", "series", "close", "turnover"]
        )
        pit_rule = LiquidityRule(
            min_median_turnover=float(pit_cfg.get("min_median_turnover", 5e7)),
            lookback_days=int(pit_cfg.get("lookback_days", 365)),
            min_sessions=int(pit_cfg.get("min_sessions", 100)),
            # Zero for adjusted bars: see LiquidityRule.min_price.
            min_price=float(pit_cfg.get("min_price", 0.0)),
            max_names=(int(pit_cfg["max_names"]) if pit_cfg.get("max_names") else None),
        )
        panel_set = set(panel_tickers)
        logger.info(f"Point-in-time universe per window: {pit_rule.describe()}")

        def universe_fn(window: object) -> list[str]:  # noqa: ANN401
            asof = window.test_start  # type: ignore[attr-defined]
            names = eligible_on(pit_bars, asof, pit_rule)
            # A name the rule admits but the panel lacks would become an
            # all-False column: present in the action space, never tradeable.
            return [t for t in names if t in panel_set]

    summary = run_signal_walk_forward(
        full_panel=full_panel,
        windows=windows,
        tickers=universe,
        universe_fn=universe_fn,
        feature_cols=feature_cols,
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        out_dir=out_dir,
        tag=tag,
        mlflow_port=mlflow_port,
        mlflow_experiment=str(cfg.train.get("mlflow_experiment", "signal")),
        mlflow_params={
            "panels_root": str(panels_root),
            "n_panel_tickers": len(panel_tickers),
            "walk": OmegaConf.to_yaml(walk_cfg).strip() if walk_cfg else "",
        },
    )

    logger.info(format_verdict(summary.gate))
    for name, path in sorted(summary.artefacts.items()):
        logger.info(f"  {name}: {path}")
    run_ids = [w.mlflow_run_id for w in summary.windows if w.mlflow_run_id]
    logger.info(f"MLflow run ids: {run_ids if run_ids else 'none (MLflow disabled)'}")


if __name__ == "__main__":
    main()
