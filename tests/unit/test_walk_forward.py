"""Unit tests for the M7 walk-forward window logic."""
from __future__ import annotations

import pathlib
from datetime import date

import numpy as np
import polars as pl
import pytest


def test_compute_windows_canonical() -> None:
    """The 4-window protocol from 05_training.md §Walk-forward."""
    from trader.training.walk_forward import compute_windows

    windows = compute_windows(
        data_start=date(2014, 1, 1),
        data_end=date(2024, 12, 31),
        train_years=5,
        val_months=12,
        test_months=12,
        purge_months=1,
        n_windows=4,
        step_months=12,
    )
    assert len(windows) == 4
    assert [w.name for w in windows] == ["W1", "W2", "W3", "W4"]
    # W1: train 2014-01..2018-12, val 2020-02..2021-01, test 2022-03..2023-02
    assert windows[0].train_start == date(2014, 1, 1)
    assert windows[0].train_end == date(2018, 12, 31)
    # W2..W4 must each advance the train_start by 12 months
    for prev, nxt in zip(windows[:-1], windows[1:]):
        assert (nxt.train_start.year, nxt.train_start.month) == (
            prev.train_start.year + 1,
            prev.train_start.month,
        )


def test_compute_windows_skips_when_test_overflows() -> None:
    """Last window must not extend past data_end."""
    from trader.training.walk_forward import compute_windows

    # data ends well before any 4th window's test segment can fit
    windows = compute_windows(
        data_start=date(2014, 1, 1),
        data_end=date(2021, 12, 31),
        train_years=5,
        val_months=12,
        test_months=12,
        n_windows=10,
    )
    # No window should have test_end > data_end
    assert all(w.test_end <= date(2021, 12, 31) for w in windows)


def test_compute_windows_purge_gaps() -> None:
    """Each segment is separated by `purge_months` from the next."""
    from trader.training.walk_forward import compute_windows

    windows = compute_windows(
        data_start=date(2014, 1, 1),
        data_end=date(2024, 12, 31),
        train_years=5, val_months=12, test_months=12,
        purge_months=1,
        n_windows=1,
    )
    w = windows[0]
    # train_end → val_start gap ≥ ~1 month
    gap_train_val = (w.val_start - w.train_end).days
    gap_val_test = (w.test_start - w.val_end).days
    assert 28 <= gap_train_val <= 32
    assert 28 <= gap_val_test <= 32


def test_slice_panel_inclusive_bounds() -> None:
    """Boundary dates are included; rows outside [start, end] are excluded."""
    from trader.training.walk_forward import slice_panel

    df = pl.DataFrame(
        {
            "date":   [date(2020, 1, 1), date(2020, 6, 15), date(2020, 12, 31)],
            "ticker": ["A", "A", "A"],
            "value":  [1.0, 2.0, 3.0],
        }
    )
    out = slice_panel(df, date(2020, 1, 1), date(2020, 12, 31))
    assert out.shape[0] == 3
    out2 = slice_panel(df, date(2020, 6, 1), date(2020, 6, 30))
    assert out2.shape[0] == 1
    assert out2["value"][0] == 2.0


def test_paired_bootstrap_ci_significance() -> None:
    """Reliably positive RL > baseline → CI strictly above zero."""
    from trader.training.walk_forward import paired_bootstrap_ci

    rl = [1.0, 1.1, 1.2, 0.9, 1.05, 1.15, 0.95, 1.0]
    bl = [0.0] * len(rl)
    res = paired_bootstrap_ci(rl, bl, n_boot=2000, rng_seed=0)
    assert res["ci_lo"] > 0.5
    assert res["ci_hi"] < 1.5
    assert res["p_above_zero"] > 0.99


def test_paired_bootstrap_ci_no_signal() -> None:
    """No signal → CI straddles zero."""
    from trader.training.walk_forward import paired_bootstrap_ci

    rng = np.random.default_rng(0)
    rl = rng.normal(0, 1, size=20).tolist()
    bl = rng.normal(0, 1, size=20).tolist()
    res = paired_bootstrap_ci(rl, bl, n_boot=2000, rng_seed=0)
    assert res["ci_lo"] < 0 < res["ci_hi"]


def test_aggregate_walk_forward_mean_std() -> None:
    """Aggregator pulls per-run metrics correctly."""
    from trader.training.walk_forward import aggregate_walk_forward

    results = [
        {"window": "W1", "seed": 42, "test": {"mean_sharpe": 0.5}},
        {"window": "W1", "seed": 43, "test": {"mean_sharpe": 0.7}},
        {"window": "W2", "seed": 42, "test": {"mean_sharpe": 0.3}},
        {"window": "W2", "seed": 43, "test": {"mean_sharpe": 0.9}},
    ]
    agg = aggregate_walk_forward(results, metric="mean_sharpe", split="test")
    assert agg["n"] == 4
    assert agg["mean"] == pytest.approx(0.6)
    assert agg["min"] == pytest.approx(0.3)
    assert agg["max"] == pytest.approx(0.9)


def test_materialise_window_writes_three_parquets(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """End-to-end: slicing + parquet I/O for a small synthetic panel."""
    from trader.training.walk_forward import (
        WindowConfig,
        materialise_window,
    )

    # Build a tiny panel that covers all three segments
    dates = [date(2014, 1, 1) + __import__("datetime").timedelta(days=i)
             for i in range(0, 3650, 30)]
    df = pl.DataFrame(
        {
            "date":   dates * 2,
            "ticker": ["A"] * len(dates) + ["B"] * len(dates),
            "value":  list(range(2 * len(dates))),
        }
    ).sort("date")

    win = WindowConfig(
        name="W1",
        train_start=date(2014, 1, 1), train_end=date(2018, 12, 31),
        val_start=date(2019, 2, 1), val_end=date(2019, 12, 31),
        test_start=date(2020, 2, 1), test_end=date(2020, 12, 31),
    )
    paths = materialise_window(df, win, tmp_path / "W1")
    assert set(paths.keys()) == {"train", "val", "test"}
    for p in paths.values():
        assert p.exists()
        sub = pl.read_parquet(p)
        assert sub.shape[0] > 0


# ── val/test correlation — the project's headline diagnostic ──────────────────


def _runs(pairs: list[tuple[float | None, float | None]],
          metric: str = "mean_sharpe") -> list[dict]:  # type: ignore[type-arg]
    """Build synthetic `run_walk_forward` results from (val, test) pairs.

    `None` on either side stands for a run whose split never produced metrics
    (evaluation panel too short, run crashed) — those must be skipped, not
    crash the aggregation.
    """
    out: list[dict] = []  # type: ignore[type-arg]
    for i, (v, t) in enumerate(pairs):
        out.append(
            {
                "window": f"W{i // 3 + 1}",
                "seed": 42 + (i % 3),
                "train": {metric: 0.0},
                "val": None if v is None else {metric: v},
                "test": None if t is None else {metric: t},
                "run_id": f"run{i}",
            }
        )
    return out


def test_val_test_correlation_perfect_anticorrelation() -> None:
    """The pathology this function exists to measure: val ranks test backwards."""
    from trader.training.walk_forward import val_test_correlation

    res = val_test_correlation(_runs([(float(i), -float(i)) for i in range(1, 9)]))
    assert res["n"] == 8
    assert res["pearson_r"] == pytest.approx(-1.0)
    assert res["spearman_rho"] == pytest.approx(-1.0)
    assert res["pearson_p"] == pytest.approx(0.0, abs=1e-12)
    assert res["degenerate"] == 0.0


def test_val_test_correlation_independent_is_near_zero() -> None:
    """Independent series → both coefficients ≈ 0 (deterministic + noisy case)."""
    from trader.training.walk_forward import val_test_correlation

    # Constructed so the covariance is *exactly* zero — no tolerance needed.
    exact = val_test_correlation(_runs([(1.0, 1.0), (2.0, -1.0), (3.0, -1.0), (4.0, 1.0)]))
    assert exact["pearson_r"] == pytest.approx(0.0, abs=1e-12)
    assert exact["spearman_rho"] == pytest.approx(0.0, abs=1e-12)

    rng = np.random.default_rng(7)
    a = rng.normal(size=300)
    b = rng.normal(size=300)
    noisy = val_test_correlation(_runs(list(zip(a.tolist(), b.tolist(), strict=True))))
    assert abs(noisy["pearson_r"]) < 0.15
    assert abs(noisy["spearman_rho"]) < 0.15
    assert noisy["pearson_p"] > 0.05          # nowhere near significant


def test_val_test_correlation_monotonic_nonlinear_splits_the_two() -> None:
    """Exactly why both coefficients are reported.

    A perfectly monotonic but convex relationship is *fully* rank-predictive —
    selecting the best model on val would pick the best model on test — yet
    Pearson understates it badly.  Reporting only Pearson would call a usable
    validation signal broken.
    """
    from trader.training.walk_forward import val_test_correlation

    res = val_test_correlation(
        _runs([(float(x), float(np.exp(x))) for x in range(8)])
    )
    assert res["spearman_rho"] == pytest.approx(1.0)
    assert res["pearson_r"] < 0.9
    assert res["pearson_r"] > 0.5


def test_val_test_correlation_too_few_pairs_returns_empty() -> None:
    """Fewer than 3 pairs → empty dict, never an exception or a fake ±1."""
    from trader.training.walk_forward import val_test_correlation

    assert val_test_correlation([]) == {}
    assert val_test_correlation(_runs([(0.5, 0.4)])) == {}
    assert val_test_correlation(_runs([(0.5, 0.4), (0.2, 0.9)])) == {}
    # Three rows but only two usable pairs is still too few.
    assert val_test_correlation(_runs([(0.5, 0.4), (0.2, 0.9), (None, 0.3)])) == {}


def test_val_test_correlation_skips_runs_with_missing_splits() -> None:
    """Runs missing val or test are dropped; the rest still correlate cleanly."""
    from trader.training.walk_forward import val_test_correlation

    res = val_test_correlation(
        _runs(
            [
                (1.0, 1.0),
                (None, 2.0),      # val eval never ran
                (2.0, 2.0),
                (3.0, None),      # test eval never ran
                (4.0, 4.0),
                (5.0, 5.0),
            ]
        )
    )
    assert res["n"] == 4
    assert res["pearson_r"] == pytest.approx(1.0)


def test_val_test_correlation_constant_series_does_not_crash() -> None:
    """Zero variance → correlation undefined; report 0.0 + a flag, not NaN."""
    from trader.training.walk_forward import val_test_correlation

    res = val_test_correlation(_runs([(1.0, t) for t in (0.1, 0.2, 0.3, 0.4)]))
    assert res["degenerate"] == 1.0
    assert res["pearson_r"] == 0.0
    assert res["spearman_rho"] == 0.0
    assert res["pearson_p"] == 1.0
    for v in res.values():
        assert np.isfinite(v)

    both_constant = val_test_correlation(_runs([(1.0, 2.0)] * 5))
    assert both_constant["degenerate"] == 1.0
    assert np.isfinite(both_constant["spearman_rho"])


def test_val_test_correlation_ties_use_average_ranks() -> None:
    """Tied Sharpes must not leak input ordering into rho."""
    from trader.training.walk_forward import val_test_correlation

    forward = val_test_correlation(
        _runs([(1.0, 1.0), (2.0, 2.0), (2.0, 3.0), (3.0, 4.0)])
    )
    # Same data, tied pair swapped: a tie-blind ranker would change rho here.
    swapped = val_test_correlation(
        _runs([(1.0, 1.0), (2.0, 3.0), (2.0, 2.0), (3.0, 4.0)])
    )
    assert forward["spearman_rho"] == pytest.approx(swapped["spearman_rho"])


def test_val_test_correlation_metric_is_a_parameter() -> None:
    """Works for cagr (or anything else) as well as Sharpe."""
    from trader.training.walk_forward import val_test_correlation

    runs = _runs([(0.1, -0.1), (0.2, -0.2), (0.3, -0.3), (0.4, -0.4)],
                 metric="mean_cagr")
    assert val_test_correlation(runs, "mean_cagr")["pearson_r"] == pytest.approx(-1.0)
    # Asking for a metric nobody recorded is not an error — just no pairs.
    assert val_test_correlation(runs, "mean_sharpe") == {}


def test_corr_p_value_matches_the_t_table() -> None:
    """The scipy-free t-approximation must agree with published critical values."""
    from trader.training.walk_forward import _corr_p_value

    # df = 10, t = 2.228 → two-sided p = 0.05.  r = t / sqrt(t² + df).
    assert _corr_p_value(2.228 / np.sqrt(2.228**2 + 10), 12) == pytest.approx(
        0.05, abs=1e-3
    )
    # df = 8, t = 2.306 → p = 0.05.
    assert _corr_p_value(2.306 / np.sqrt(2.306**2 + 8), 10) == pytest.approx(
        0.05, abs=1e-3
    )
    # df = 1 is Cauchy: t = 1 → p = 0.5 exactly.
    assert _corr_p_value(1 / np.sqrt(2.0), 3) == pytest.approx(0.5, abs=1e-6)
    assert _corr_p_value(0.0, 12) == pytest.approx(1.0)
    assert _corr_p_value(1.0, 12) == 0.0        # |r| = 1 must not divide by zero
    assert _corr_p_value(0.9, 2) == 1.0         # df = 0 → undefined, not a crash


# ── paired RL-vs-equal-weight arm ─────────────────────────────────────────────


def test_paired_test_values_keeps_runs_aligned() -> None:
    """Pairing is by run: a run missing either arm drops out of *both* lists."""
    from trader.training.walk_forward import paired_test_values

    results = [
        {"test": {"mean_sharpe": 1.0}, "baseline_test": {"mean_sharpe": 0.4}},
        {"test": {"mean_sharpe": 2.0}, "baseline_test": None},        # no baseline
        {"test": None, "baseline_test": {"mean_sharpe": 0.5}},        # no agent
        {"test": {"mean_sharpe": 3.0}, "baseline_test": {"mean_sharpe": 0.6}},
    ]
    rl, bl = paired_test_values(results)
    assert rl == [1.0, 3.0]
    assert bl == [0.4, 0.6]


def test_paired_test_values_feeds_the_bootstrap() -> None:
    """End-to-end: the arm wires straight into the (previously uncalled) CI."""
    from trader.training.walk_forward import paired_bootstrap_ci, paired_test_values

    results = [
        {"test": {"mean_sharpe": 0.9 + 0.05 * i},
         "baseline_test": {"mean_sharpe": 0.2}}
        for i in range(10)
    ]
    ci = paired_bootstrap_ci(*paired_test_values(results), n_boot=2000, rng_seed=0)
    assert ci["ci_lo"] > 0.0        # agent reliably ahead of equal weight
    assert ci["mean_diff"] == pytest.approx(0.925, abs=1e-9)


def test_paired_test_values_empty_when_no_baseline_arm() -> None:
    """Pre-baseline result dicts degrade to an empty (not mismatched) pairing."""
    from trader.training.walk_forward import paired_bootstrap_ci, paired_test_values

    rl, bl = paired_test_values([{"test": {"mean_sharpe": 1.0}}] * 4)
    assert rl == [] and bl == []
    assert paired_bootstrap_ci(rl, bl)["mean_diff"] == 0.0


# ── shuffled-ticker leak check ────────────────────────────────────────────────


def test_make_ticker_permutation_only_moves_tradeable_slots() -> None:
    """Untradeable (padded) names stay put so the shuffle isn't detectable."""
    from trader.training.walk_forward import make_ticker_permutation

    mask = np.array([1, 0, 1, 1, 0, 1], dtype=np.int8)
    perm = make_ticker_permutation(mask, np.random.default_rng(0))

    assert sorted(perm.tolist()) == list(range(6))       # a real permutation
    assert perm[1] == 1 and perm[4] == 4                 # untradeable fixed
    assert set(perm[[0, 2, 3, 5]].tolist()) == {0, 2, 3, 5}  # tradeable stay tradeable


def test_make_ticker_permutation_degenerate_masks() -> None:
    """Zero or one tradeable name → identity, never an error."""
    from trader.training.walk_forward import make_ticker_permutation

    rng = np.random.default_rng(0)
    assert make_ticker_permutation(np.zeros(5, dtype=np.int8), rng).tolist() == list(
        range(5)
    )
    one = np.array([0, 1, 0], dtype=np.int8)
    assert make_ticker_permutation(one, rng).tolist() == [0, 1, 2]


def test_apply_ticker_permutation_touches_only_feature_keys() -> None:
    """Features and sector_ids move together; everything realisation-side doesn't.

    That asymmetry *is* the test: after this, slot j shows another stock's
    history while slot j's P&L is still its own.
    """
    from trader.training.walk_forward import apply_ticker_permutation

    n, lookback, n_feat = 4, 3, 2
    obs = {
        "features": np.arange(lookback * n * n_feat, dtype=np.float32).reshape(
            lookback, n, n_feat
        ),
        "sector_ids": np.array([10, 11, 12, 13], dtype=np.int32),
        "mask": np.array([1, 1, 1, 1], dtype=np.int8),
        "portfolio": np.array([0.1, 0.2, 0.3, 0.2, 0.2], dtype=np.float32),
        "nav": np.array(1000.0, dtype=np.float32),
        "regime": np.array([0.5, -0.5], dtype=np.float32),
        "next_day_returns": np.array([0.01, 0.02, 0.03, 0.04], dtype=np.float32),
    }
    perm = np.array([2, 3, 0, 1], dtype=np.int64)
    out = apply_ticker_permutation(obs, perm)

    assert np.array_equal(out["features"], obs["features"][:, perm, :])
    assert out["sector_ids"].tolist() == [12, 13, 10, 11]
    for key in ("mask", "portfolio", "nav", "regime", "next_day_returns"):
        assert np.array_equal(out[key], obs[key]), f"{key} must not be permuted"
    # The caller's obs is left intact (the env keeps stepping from the real one).
    assert obs["sector_ids"].tolist() == [10, 11, 12, 13]


# ── guards on the code this module deliberately duplicates ────────────────────


def _modelconfig_kwargs(path: str) -> set[str]:
    """Keyword names passed to the single `ModelConfig(...)` call in a file."""
    import ast

    tree = ast.parse(pathlib.Path(path).read_text())
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "ModelConfig"
        ):
            return {kw.arg for kw in node.keywords if kw.arg is not None}
    raise AssertionError(f"no ModelConfig(...) call found in {path}")


def test_shuffle_check_model_builder_matches_runner() -> None:
    """`_rebuild_eval_model` duplicates runner's model construction — keep it honest.

    The shuffled-ticker check has to instantiate the same architecture the
    checkpoint was trained with, but runner.py exposes no shared builder and is
    not ours to edit, so the construction is duplicated.  A silent divergence
    here would evaluate a *different* model and quietly invalidate the leak
    check, so this pins the two ModelConfig call sites together.

    If this fails: reconcile the two, or (better) extract a shared
    `build_model()` in runner.py and delete the duplicate.
    """
    runner_kwargs = _modelconfig_kwargs("src/trader/training/runner.py")
    walk_kwargs = _modelconfig_kwargs("src/trader/training/walk_forward.py")
    assert walk_kwargs == runner_kwargs, (
        "ModelConfig arguments have diverged between runner.train_one_run and "
        f"walk_forward._rebuild_eval_model: only in runner={runner_kwargs - walk_kwargs}, "
        f"only in walk_forward={walk_kwargs - runner_kwargs}"
    )


def test_baseline_arm_seeding_matches_runner_eval() -> None:
    """The baseline arm is only *paired* while its seeds match runner's eval.

    Episode windows are a pure function of the reset seed.  runner evaluates
    the test split with `seed + 1` and resets with `seed + 999 + episode`; if
    either offset moves, the equal-weight arm silently starts scoring different
    episodes and `paired_bootstrap_ci` stops being a paired test.
    """
    import inspect

    from trader.training import runner
    from trader.training.walk_forward import (
        _ENV_SEED_OFFSET,
        _EVAL_N_EPISODES,
        _TEST_SEED_OFFSET,
    )

    n_ep_default = inspect.signature(runner._evaluate_split).parameters["n_episodes"].default
    assert n_ep_default == _EVAL_N_EPISODES, (
        "runner._evaluate_split's episode count changed; update _EVAL_N_EPISODES"
    )

    eval_src = inspect.getsource(runner._evaluate_split)
    assert f"seed + {_ENV_SEED_OFFSET} + ep" in eval_src, (
        "runner._evaluate_split's reset-seed formula changed; update _ENV_SEED_OFFSET"
    )
    assert f"seed + {_TEST_SEED_OFFSET}," in inspect.getsource(runner.train_one_run), (
        "runner's test-split seed offset changed; update _TEST_SEED_OFFSET"
    )


# ── driver: MLflow summary metrics ────────────────────────────────────────────


def _load_driver():  # type: ignore[no-untyped-def]
    """Import `scripts/walk_forward.py` (not a package) by path."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_wf_driver", pathlib.Path("scripts/walk_forward.py")
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_summary_metrics_are_flat_namespaced_floats() -> None:
    """The A/B is decided by comparing `val_test_corr/pearson_r` across sweeps.

    So it has to reach MLflow as a real, chartable metric — namespaced, finite,
    and float-typed.
    """
    driver = _load_driver()
    summary = {
        "model": "mlp_regime",                      # non-dict → ignored
        "n_windows": 4,                             # non-dict → ignored
        "val_test_corr": {
            "n": 12.0, "pearson_r": -0.86, "pearson_p": 0.0003,
            "spearman_rho": -0.79, "spearman_p": 0.002, "degenerate": 0.0,
        },
        "bootstrap_vs_equal_weight": {
            "mean_diff": 0.21, "ci_lo": -0.05, "ci_hi": 0.47, "p_above_zero": 0.93,
        },
        "test_mean_sharpe": {"n": 12.0, "mean": 0.42, "std": float("nan")},
        "shuffled_test_mean_sharpe": {},            # feature off → nothing logged
    }
    metrics = driver._flatten_summary_metrics(summary)

    assert metrics["val_test_corr/pearson_r"] == pytest.approx(-0.86)
    assert metrics["val_test_corr/spearman_rho"] == pytest.approx(-0.79)
    assert metrics["bootstrap_vs_equal_weight/ci_lo"] == pytest.approx(-0.05)
    assert "test_mean_sharpe/std" not in metrics      # NaN dropped, not logged
    assert not any(k.startswith("shuffled_") for k in metrics)
    assert all(isinstance(v, float) for v in metrics.values())
    assert all("/" in k for k in metrics)
