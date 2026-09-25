#!/usr/bin/env python
"""Walk-forward training driver — M7.

Runs the configured model across multiple sliding windows × seeds and
reports per-window metrics + aggregated bootstrap CI vs equal-weight.

Usage:
    # Default 4 windows × 3 seeds for one architecture:
    uv run python scripts/walk_forward.py model=gnn_intra_only

    # Custom seeds / windows:
    uv run python scripts/walk_forward.py model=gnn_v1 \
        walk.seeds=[42,43,44,45,46] walk.n_windows=4

    # With the shuffled-ticker-label leak check (off by default).  No leading
    # '+': `shuffle_check` IS a key in configs/walk/default.yaml, and Hydra
    # errors on '+' for a key that already exists.
    uv run python scripts/walk_forward.py model=mlp_regime walk.shuffle_check=true

Reported at the end, in order of importance:
  1. corr(val_sharpe, test_sharpe), Pearson AND Spearman — the metric the
     Phase-1/Phase-2 A/B is decided on (baseline: Pearson -0.86).
  2. Paired bootstrap CI of the agent's test Sharpe minus equal-weight's, on
     identical episode windows.
  3. The usual aggregated test metrics.
All of the above also go to MLflow as a single `<model>_walk_summary` run so
they can be diffed across configs.

The full historical panel is read from ``<data.panels_root>/full.parquet`` and
re-sliced into walk-forward segments.  Per-window panels are written under
``data/walks/<W>/{train,val,test}.parquet``.

``full.parquet`` matters: the train/val/test parquets do NOT tile the history,
because build_features cuts the purge months out of each later split and never
writes them.  Re-splitting the concatenation therefore slices windows against a
calendar with a month-long hole at each build-time boundary — on data/panels
that truncated 4 of 12 segments in silence.  The driver falls back to the
concatenation for panels built before full.parquet existed, but refuses to run
once it finds gaps (override: ``walk.allow_panel_gaps=true``).
"""
from __future__ import annotations

import json
import math
from datetime import date
from pathlib import Path

import hydra
from loguru import logger
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    import hydra.utils
    import polars as pl

    from trader.data.features import resolve_panels_root
    from trader.training.walk_forward import (
        DEFAULT_PURGE_MONTHS,
        aggregate_walk_forward,
        compute_windows,
        find_calendar_gaps,
        paired_bootstrap_ci,
        paired_test_values,
        run_walk_forward,
        val_test_correlation,
    )

    orig_cwd = Path(hydra.utils.get_original_cwd())
    panels_root = resolve_panels_root(cfg, orig_cwd)
    walks_root = orig_cwd / "data" / "walks"
    walks_root.mkdir(parents=True, exist_ok=True)

    # ── Load the full historical panel ────────────────────────────────────────
    # `full.parquet` is the un-split panel; prefer it. The three split parquets
    # do not tile the history — build_features drops the purge months between
    # them entirely — so concatenating them yields a calendar with a month-long
    # hole at each build-time boundary, and windows get sliced against it
    # without a word. Fall back to the concatenation only when the panel predates
    # full.parquet, and refuse to run on it if the holes are actually there.
    full_path = panels_root / "full.parquet"
    if full_path.exists():
        full_panel = pl.read_parquet(full_path).sort(["date", "ticker"])
    else:
        panel_paths = [
            panels_root / "train.parquet",
            panels_root / "val.parquet",
            panels_root / "test.parquet",
        ]
        available = [p for p in panel_paths if p.exists()]
        if not available:
            logger.error(
                f"No panels found under {panels_root}. Run build_features.py first."
            )
            return
        logger.warning(
            f"{full_path} not found — falling back to concat(train, val, test), "
            "which is missing the build-time purge months. Rebuild with "
            "scripts/build_features.py to emit full.parquet."
        )
        full_panel = pl.concat([pl.read_parquet(p) for p in available]).unique(
            subset=["date", "ticker"], keep="first"
        ).sort(["date", "ticker"])
    logger.info(
        f"Full panel: {full_panel.shape[0]:,} rows, "
        f"{full_panel['date'].min()}..{full_panel['date'].max()}"
    )

    # ── Walk-forward config (with sensible defaults if not in YAML) ───────────
    walk_cfg = cfg.get("walk", {})
    n_windows = int(walk_cfg.get("n_windows", 4))
    train_years = int(walk_cfg.get("train_years", 5))
    val_months = int(walk_cfg.get("val_months", 12))
    test_months = int(walk_cfg.get("test_months", 12))
    purge_months = int(walk_cfg.get("purge_months", DEFAULT_PURGE_MONTHS))
    step_months = int(walk_cfg.get("step_months", 12))
    seeds_list = list(walk_cfg.get("seeds", [42, 43, 44]))

    # ── The panel must be a contiguous calendar before it is re-split ─────────
    gaps = find_calendar_gaps(full_panel)
    if gaps:
        detail = ", ".join(f"{a} → {b} ({n} days)" for a, b, n in gaps)
        msg = (
            f"Full panel has {len(gaps)} calendar gap(s): {detail}. Walk-forward "
            "windows are cut from this calendar, so every segment spanning a gap "
            "is silently short — on data/panels that truncated 4 of 12 segments. "
            f"Rebuild the panel so {full_path} exists."
        )
        if bool(walk_cfg.get("allow_panel_gaps", False)):
            logger.error(msg + " Continuing anyway: walk.allow_panel_gaps=true.")
        else:
            logger.error(msg)
            raise SystemExit(
                "Refusing to run walk-forward on a panel with calendar gaps. "
                "Rebuild with scripts/build_features.py, or set "
                "walk.allow_panel_gaps=true to override (results will be wrong)."
            )

    data_start = full_panel["date"].min()
    data_end = full_panel["date"].max()
    if not isinstance(data_start, date):
        data_start = date.fromisoformat(str(data_start))
    if not isinstance(data_end, date):
        data_end = date.fromisoformat(str(data_end))

    windows = compute_windows(
        data_start=data_start,
        data_end=data_end,
        train_years=train_years,
        val_months=val_months,
        test_months=test_months,
        purge_months=purge_months,
        n_windows=n_windows,
        step_months=step_months,
    )
    logger.info(f"Computed {len(windows)} windows:")
    for w in windows:
        logger.info(f"  {w.name}: {w.asdict_iso()}")

    if not windows:
        logger.error("No windows fit in the available data — check date ranges.")
        return

    # ── Drive ─────────────────────────────────────────────────────────────────
    model_name = str(cfg.model.get("name", "ppo"))
    results = run_walk_forward(
        cfg,
        full_panel=full_panel,
        windows=windows,
        seeds=seeds_list,
        walks_root=walks_root,
        mlflow_experiment=f"walk_forward_{model_name}",
    )

    # ── Aggregate + dump JSON for downstream analysis ─────────────────────────
    rl_sharpes, bl_sharpes = paired_test_values(results, "mean_sharpe")
    summary = {
        "model": model_name,
        "n_windows": len(windows),
        "n_seeds": len(seeds_list),
        "windows": [w.asdict_iso() for w in windows],
        # THE decision metric for the Phase-1/Phase-2 A/B.  Baseline is
        # Pearson −0.86 on mlp_regime's control; success is this moving toward
        # zero, NOT a higher Sharpe.  See HANDOFF.md §6.
        "val_test_corr": val_test_correlation(results, "mean_sharpe"),
        "val_test_corr_cagr": val_test_correlation(results, "mean_cagr"),
        # M7 acceptance criterion: paired bootstrap CI of RL vs equal-weight.
        "bootstrap_vs_equal_weight": paired_bootstrap_ci(rl_sharpes, bl_sharpes),
        "test_mean_sharpe": aggregate_walk_forward(results, "mean_sharpe", "test"),
        "test_mean_cagr": aggregate_walk_forward(results, "mean_cagr", "test"),
        "test_mean_max_drawdown": aggregate_walk_forward(
            results, "mean_max_drawdown", "test"
        ),
        "val_mean_sharpe": aggregate_walk_forward(results, "mean_sharpe", "val"),
        "baseline_test_mean_sharpe": aggregate_walk_forward(
            results, "mean_sharpe", "baseline_test"
        ),
        "shuffled_test_mean_sharpe": aggregate_walk_forward(
            results, "mean_sharpe", "shuffled_test"
        ),
        "per_run": [
            {
                "window": r["window"],
                "seed": r["seed"],
                "train_sharpe": (r["train"] or {}).get("mean_sharpe"),
                "val_sharpe": (r["val"] or {}).get("mean_sharpe"),
                "test_sharpe": (r["test"] or {}).get("mean_sharpe"),
                "baseline_test_sharpe": (r.get("baseline_test") or {}).get(
                    "mean_sharpe"
                ),
                "shuffled_test_sharpe": (r.get("shuffled_test") or {}).get(
                    "mean_sharpe"
                ),
                "run_id": r.get("run_id"),
            }
            for r in results
        ],
    }

    out_path = walks_root / f"summary_{model_name}.json"
    with out_path.open("w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info("\n" + "=" * 72)
    logger.info(f"=== Walk-forward summary — {model_name} ===")
    logger.info("=" * 72)

    # ── The number that decides the experiment ────────────────────────────────
    corr = summary["val_test_corr"]
    logger.info("")
    logger.info("  *** corr(val_sharpe, test_sharpe) — THE decision metric ***")
    if corr:
        logger.info(
            f"      Pearson  r   = {corr['pearson_r']:+.4f}   "
            f"(p ≈ {corr['pearson_p']:.4f})"
        )
        logger.info(
            f"      Spearman rho = {corr['spearman_rho']:+.4f}   "
            f"(p ≈ {corr['spearman_p']:.4f})"
        )
        logger.info(f"      n pairs      = {corr['n']:.0f}  (window × seed runs)")
        logger.info(
            "      Reference: mlp_baseline M7 Pearson r = -0.86 (val was "
            "ANTI-predictive)."
        )
        logger.info(
            "      Success = movement toward 0 / positive, NOT a higher Sharpe."
        )
        if corr.get("degenerate"):
            logger.warning(
                "      DEGENERATE: a series had zero variance — correlation is "
                "undefined and reported as 0.0."
            )
    else:
        logger.warning(
            "      Not computed: fewer than 3 runs with both val and test metrics."
        )

    # ── Significance vs the equal-weight benchmark ────────────────────────────
    boot = summary["bootstrap_vs_equal_weight"]
    logger.info("")
    logger.info("  Paired bootstrap — RL minus equal-weight (test mean_sharpe):")
    if rl_sharpes:
        logger.info(
            f"      mean diff = {boot['mean_diff']:+.4f}   "
            f"95% CI [{boot['ci_lo']:+.4f}, {boot['ci_hi']:+.4f}]   "
            f"P(diff>0) = {boot.get('p_above_zero', float('nan')):.3f}"
            f"   n = {len(rl_sharpes)}"
        )
    else:
        logger.warning("      Not computed: no paired (RL, equal-weight) runs.")

    # ── Headline metrics ──────────────────────────────────────────────────────
    logger.info("")
    logger.info(
        f"  Test mean_sharpe (across "
        f"{summary['test_mean_sharpe'].get('n', 0):.0f} runs):"
    )
    for k, v in summary["test_mean_sharpe"].items():
        logger.info(f"      {k}: {v:.4f}")
    if summary["baseline_test_mean_sharpe"]:
        logger.info(
            f"  Equal-weight test mean_sharpe: "
            f"{summary['baseline_test_mean_sharpe']['mean']:.4f}"
        )
    if summary["shuffled_test_mean_sharpe"]:
        logger.info(
            f"  Shuffled-ticker test mean_sharpe: "
            f"{summary['shuffled_test_mean_sharpe']['mean']:.4f}  "
            "(≈ real Sharpe ⇒ no cross-sectional edge; ≫ equal-weight ⇒ "
            "suspect a leak)"
        )
    logger.info(f"  Saved: {out_path}")

    # Last, deliberately: an unreachable tracking server retries for a while,
    # and neither the on-disk summary nor the printed correlation should have
    # to wait on the network.
    _log_summary_to_mlflow(cfg, model_name, summary)


# Summary blocks that are dicts of scalars and belong in MLflow.  Order is the
# order they appear in the UI.
_METRIC_GROUPS = (
    "val_test_corr",
    "val_test_corr_cagr",
    "bootstrap_vs_equal_weight",
    "test_mean_sharpe",
    "test_mean_cagr",
    "test_mean_max_drawdown",
    "val_mean_sharpe",
    "baseline_test_mean_sharpe",
    "shuffled_test_mean_sharpe",
)


def _flatten_summary_metrics(summary: dict[str, object]) -> dict[str, float]:
    """Flatten the nested summary into MLflow's flat `key -> float` metric space.

    Keys are namespaced with '/' (MLflow permits it) so the UI groups them and
    so `val_test_corr/pearson_r` is directly comparable across configs — which
    is the whole point of logging it: the A/B is decided by comparing that one
    number between two sweeps.  Non-numeric and non-finite entries are dropped
    rather than sent as NaN, which MLflow stores but nobody can chart.
    """
    metrics: dict[str, float] = {}
    for group in _METRIC_GROUPS:
        block = summary.get(group)
        if not isinstance(block, dict):
            continue
        for k, v in block.items():
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                continue
            if not math.isfinite(float(v)):
                continue
            metrics[f"{group}/{k}"] = float(v)
    return metrics


def _log_summary_to_mlflow(
    cfg: DictConfig,
    model_name: str,
    summary: dict[str, object],
) -> None:
    """Log walk-forward-level metrics to a single MLflow 'summary' run.

    The per-(window × seed) runs each get their own MLflow run from
    `train_one_run`; the correlation and the bootstrap CI are properties of the
    *whole sweep*, so they need a run of their own to be comparable across
    configs.  Best-effort: a tracking-server hiccup must not lose the on-disk
    summary that was already written.
    """
    import mlflow

    metrics = _flatten_summary_metrics(summary)

    try:
        from trader.tracking import tracking_uri  # noqa: PLC0415
        mlflow.set_tracking_uri(tracking_uri(cfg.get('mlflow_port', 5555)))
        mlflow.set_experiment(f"walk_forward_{model_name}")
        with mlflow.start_run(run_name=f"{model_name}_walk_summary"):
            mlflow.set_tags(
                {
                    "summary": "true",
                    "model": model_name,
                    "n_windows": str(summary.get("n_windows")),
                    "n_seeds": str(summary.get("n_seeds")),
                }
            )
            if metrics:
                mlflow.log_metrics(metrics)
            mlflow.log_dict(summary, "walk_forward_summary.json")
        logger.info(f"MLflow: logged {len(metrics)} summary metrics")
    except Exception as e:  # noqa: BLE001 — reporting must not fail the sweep
        logger.warning(f"MLflow summary logging failed, continuing: {e}")


if __name__ == "__main__":
    main()
