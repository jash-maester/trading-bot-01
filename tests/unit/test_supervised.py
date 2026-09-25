"""R4 — unit tests for :mod:`trader.training.supervised`.

Four things are being defended here, in descending order of how expensive they
would be to get wrong:

1. **Leakage.**  A prediction dated ``t`` must be a function of feature rows
   ``t-L+1 .. t`` and of train-split normalisation stats, and of nothing else.
   Two tests perturb the future and demand bit-identical output.
2. **Labels are never fabricated.**  A row whose forward window runs off the end
   of a split, or whose stock is suspended inside that window, is *unlabelled* —
   NaN — not zero.  A zero label is a claim that the stock returned exactly the
   cross-sectional mean, and MSE would happily learn it.
3. **The gate is a gate.**  PASS requires mean IC above the bar *and* a CI that
   excludes zero, on every window.  A FAIL must come out as a FAIL.
4. **The pinned artefact contract** — column names, dtypes, ticker order,
   `gate.json` keys — because three other streams read it.
"""
from __future__ import annotations

import json
import math
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import torch

from trader.models.signal import SignalConfig, SignalModel
from trader.training.supervised import (
    GATE_MIN_MEAN_IC,
    GATE_RULE,
    DailyScores,
    DeadFeatureError,
    HorizonMetrics,
    PanelTensors,
    SupervisedConfig,
    assert_feature_liveness,
    bootstrap_mean_ci,
    build_panel_tensors,
    cross_sectional_zscore,
    daily_scores,
    empty_predictions_frame,
    evaluate_predictions,
    forward_returns,
    gate_json_payload,
    gate_verdict,
    predict_panel,
    predictions_frame,
    predictions_schema,
    run_signal_walk_forward,
    spearman,
    summarise_daily,
    train_signal_window,
)
from trader.training.walk_forward import WindowConfig, compute_windows

FEATS = ["log_return_1d", "sig", "noise_a", "noise_b"]


# ── Synthetic panel ───────────────────────────────────────────────────────────


def _business_days(start: date, n: int) -> list[date]:
    out: list[date] = []
    d = start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def _synthetic_panel(
    n_days: int,
    tickers: list[str],
    *,
    start: date = date(2018, 1, 1),
    seed: int = 0,
    signal: float = 0.0,
    untradeable: set[tuple[int, str]] | None = None,
) -> pl.DataFrame:
    """A panel with a *known* cross-sectional signal and no other structure.

    ``sig[t, i]`` predicts ``log_return_1d[t+1, i]`` with strength `signal`.
    Everything else is noise.  This is a test fixture, not data: any IC measured
    on it says something about the plumbing and nothing about markets.
    """
    rng = np.random.default_rng(seed)
    n = len(tickers)
    dates = _business_days(start, n_days)
    sig = rng.standard_normal((n_days, n))
    ret = np.zeros((n_days, n))
    ret[1:] = signal * sig[:-1] + 0.01 * rng.standard_normal((n_days - 1, n))
    ret[0] = 0.01 * rng.standard_normal(n)
    dead = untradeable or set()
    rows: dict[str, list[object]] = {c: [] for c in ("date", "ticker", "is_tradeable", *FEATS)}
    for ti, d in enumerate(dates):
        for ni, tk in enumerate(tickers):
            trd = (ti, tk) not in dead
            rows["date"].append(d)
            rows["ticker"].append(tk)
            rows["is_tradeable"].append(trd)
            rows["log_return_1d"].append(float(ret[ti, ni]))
            rows["sig"].append(float(sig[ti, ni]))
            rows["noise_a"].append(float(rng.standard_normal()))
            rows["noise_b"].append(float(rng.standard_normal()))
    return pl.DataFrame(
        rows,
        schema={
            "date": pl.Date,
            "ticker": pl.Utf8,
            "is_tradeable": pl.Boolean,
            **{c: pl.Float64 for c in FEATS},
        },
    )


def _tiny_model(n_features: int, horizons: tuple[int, ...] = (2,)) -> SignalModel:
    torch.manual_seed(0)
    return SignalModel(
        SignalConfig(
            in_features=n_features,
            embed_dim=8,
            num_channels=[8, 8],
            kernel_size=2,
            dropout=0.0,
            head_hidden=4,
            horizons=horizons,
        )
    )


# ── forward_returns: the label ────────────────────────────────────────────────


def test_forward_returns_sums_the_right_window() -> None:
    r = np.arange(1, 13, dtype=np.float32).reshape(6, 2) / 100.0
    m = np.ones((6, 2), dtype=bool)
    fwd = forward_returns(r, m, 2)
    # fwd[0] = r[1] + r[2]
    assert fwd[0, 0] == pytest.approx(r[1, 0] + r[2, 0])
    assert fwd[3, 1] == pytest.approx(r[4, 1] + r[5, 1])


def test_forward_returns_tail_is_unlabelled_not_zero() -> None:
    """The last h rows have no forward window. They must be NaN, never 0.0."""
    r = np.full((6, 2), 0.01, dtype=np.float32)
    fwd = forward_returns(r, np.ones((6, 2), dtype=bool), 2)
    assert np.all(np.isnan(fwd[-2:]))
    assert np.all(np.isfinite(fwd[:-2]))
    assert not np.any(fwd[-2:] == 0.0)


def test_forward_returns_unlabelled_when_suspended_inside_the_window() -> None:
    """A run of sentinel zeros through a suspension is not a holding-period return."""
    r = np.full((6, 1), 0.01, dtype=np.float32)
    m = np.ones((6, 1), dtype=bool)
    m[3, 0] = False                       # suspended on day 3
    fwd = forward_returns(r, m, 2)
    assert np.isnan(fwd[1, 0])            # window t+1..t+2 = days 2,3 → touches it
    assert np.isnan(fwd[2, 0])            # days 3,4
    assert np.isnan(fwd[3, 0])            # untradeable at t itself
    assert np.isfinite(fwd[0, 0])         # days 1,2 → clean


def test_forward_returns_short_panel_is_all_nan() -> None:
    fwd = forward_returns(np.zeros((2, 3), np.float32), np.ones((2, 3), bool), 5)
    assert np.all(np.isnan(fwd))


@pytest.mark.parametrize("horizon", [0, -3])
def test_forward_returns_rejects_bad_horizon(horizon: int) -> None:
    with pytest.raises(ValueError, match="horizon must be positive"):
        forward_returns(np.zeros((4, 2), np.float32), np.ones((4, 2), bool), horizon)


def test_forward_returns_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        forward_returns(np.zeros((4, 2), np.float32), np.ones((4, 3), bool), 1)


# ── cross_sectional_zscore: the target ────────────────────────────────────────


def test_zscore_is_per_date_over_the_labelled_cross_section() -> None:
    rng = np.random.default_rng(3)
    raw = rng.standard_normal((4, 30)) * 0.05 + 0.2   # a big common (market) move
    z = cross_sectional_zscore(raw, min_cross_section=10)
    for t in range(4):
        assert z[t].mean() == pytest.approx(0.0, abs=1e-5)
        assert z[t].std() == pytest.approx(1.0, abs=1e-5)


def test_zscore_removes_the_market_move() -> None:
    """The target is *relative*: adding a constant to every stock changes nothing.

    This is the whole reason the target is standardised — §1.3 of the revamp:
    an agent paid for raw return is paid for being long.
    """
    rng = np.random.default_rng(4)
    raw = rng.standard_normal((3, 20)) * 0.05
    a = cross_sectional_zscore(raw, min_cross_section=10)
    b = cross_sectional_zscore(raw + 0.10, min_cross_section=10)
    np.testing.assert_allclose(a, b, atol=1e-5)


def test_zscore_preserves_unlabelled_positions() -> None:
    rng = np.random.default_rng(5)
    raw = rng.standard_normal((2, 20))
    raw[0, 3] = np.nan
    z = cross_sectional_zscore(raw, min_cross_section=5)
    assert np.isnan(z[0, 3])
    assert np.isfinite(z[0, 4])


def test_zscore_drops_thin_and_degenerate_dates() -> None:
    raw = np.full((3, 20), np.nan)
    raw[0, :4] = [0.1, 0.2, 0.3, 0.4]          # only 4 labelled → too thin
    raw[1, :] = 0.05                            # no spread → degenerate
    raw[2, :] = np.linspace(0.0, 1.0, 20)       # fine
    z = cross_sectional_zscore(raw, min_cross_section=10)
    assert np.all(np.isnan(z[0]))
    assert np.all(np.isnan(z[1]))
    assert np.all(np.isfinite(z[2]))


# ── Feature liveness (rule: a dead channel is a bias term) ────────────────────


def test_liveness_passes_and_reports_stds() -> None:
    panel = _synthetic_panel(30, ["A", "B", "C"], seed=1)
    stds = assert_feature_liveness(panel, FEATS)
    assert set(stds) == set(FEATS)
    assert all(v > 0 for v in stds.values())


def test_liveness_names_the_dead_column() -> None:
    """`beta_nifty_60d` was constant 1.0 across 276,005 rows. This is the catch."""
    panel = _synthetic_panel(30, ["A", "B", "C"], seed=1).with_columns(
        pl.lit(1.0).alias("beta_nifty_60d")
    )
    with pytest.raises(DeadFeatureError, match="beta_nifty_60d"):
        assert_feature_liveness(panel, [*FEATS, "beta_nifty_60d"])


def test_liveness_names_a_missing_column() -> None:
    panel = _synthetic_panel(20, ["A", "B"], seed=1)
    with pytest.raises(DeadFeatureError, match=r"nope \(missing\)"):
        assert_feature_liveness(panel, [*FEATS, "nope"])


def test_liveness_only_counts_tradeable_rows() -> None:
    """Variance that exists only on masked rows is not variance the model sees."""
    panel = _synthetic_panel(20, ["A", "B"], seed=1).with_columns(
        pl.when(pl.col("is_tradeable"))
        .then(pl.lit(7.0))
        .otherwise(pl.col("sig"))
        .alias("half_dead")
    )
    with pytest.raises(DeadFeatureError, match="half_dead"):
        assert_feature_liveness(panel, ["half_dead"])


def test_liveness_rejects_a_panel_with_no_tradeable_rows() -> None:
    panel = _synthetic_panel(10, ["A"], seed=1).with_columns(
        pl.lit(False).alias("is_tradeable")
    )
    with pytest.raises(DeadFeatureError, match="no tradeable rows"):
        assert_feature_liveness(panel, FEATS)


def test_liveness_requires_the_mask_column() -> None:
    panel = _synthetic_panel(10, ["A"], seed=1).drop("is_tradeable")
    with pytest.raises(ValueError, match="is_tradeable"):
        assert_feature_liveness(panel, FEATS)


# ── build_panel_tensors ───────────────────────────────────────────────────────


def test_build_panel_tensors_shapes_and_missing_tickers() -> None:
    """A ticker absent from a stale panel is untradeable everywhere, not absent."""
    panel = _synthetic_panel(40, ["A", "B"], seed=2)
    t = build_panel_tensors(panel, ["A", "B", "GHOST"], FEATS, (2,), min_cross_section=2)
    assert t.features.shape == (40, 3, len(FEATS))
    assert t.mask.shape == (40, 3)
    assert t.tickers == ["A", "B", "GHOST"]
    assert not t.mask[:, 2].any()
    assert np.all(np.isnan(t.targets[2][:, 2]))


def test_build_panel_tensors_refuses_a_nan_on_a_tradeable_row() -> None:
    panel = _synthetic_panel(20, ["A", "B"], seed=2).with_columns(
        pl.when((pl.col("ticker") == "A") & (pl.col("date") == pl.col("date").min()))
        .then(None)
        .otherwise(pl.col("sig"))
        .alias("sig")
    )
    with pytest.raises(ValueError, match="non-finite"):
        build_panel_tensors(panel, ["A", "B"], FEATS, (2,), min_cross_section=2)


def test_build_panel_tensors_requires_the_contract_columns() -> None:
    panel = _synthetic_panel(10, ["A"], seed=2).drop("log_return_1d")
    with pytest.raises(ValueError, match="log_return_1d"):
        build_panel_tensors(panel, ["A"], ["sig"], (1,))


def test_build_panel_tensors_targets_stop_at_the_split_boundary() -> None:
    """A train label may not reach across the purge into val.

    The tensors are built per split, so the last `h` rows of the train split are
    unlabelled — which is the only way a purge gap can actually mean anything.
    """
    panel = _synthetic_panel(30, [f"T{i}" for i in range(12)], seed=6)
    t = build_panel_tensors(panel, [f"T{i}" for i in range(12)], FEATS, (3,), min_cross_section=5)
    assert np.all(np.isnan(t.targets[3][-3:]))
    assert np.isfinite(t.targets[3][:-3]).any()


# ── Metrics ───────────────────────────────────────────────────────────────────


def test_spearman_is_rank_based() -> None:
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert spearman(x, np.array([10.0, 20.0, 30.0, 40.0])) == pytest.approx(1.0)
    assert spearman(x, np.array([1.0, 8.0, 27.0, 64.0])) == pytest.approx(1.0)
    assert spearman(x, np.array([4.0, 3.0, 2.0, 1.0])) == pytest.approx(-1.0)


def test_spearman_returns_none_when_undefined() -> None:
    assert spearman(np.array([1.0, 2.0]), np.array([1.0, 2.0])) is None
    assert spearman(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0])) is None


def test_daily_scores_perfect_and_inverted() -> None:
    rng = np.random.default_rng(7)
    target = rng.standard_normal((5, 20))
    d = daily_scores(target.copy(), target, target, min_cross_section=10)
    assert d.ic.size == 5
    np.testing.assert_allclose(d.ic, 1.0)
    assert np.all(d.decile_spread > 0)

    inv = daily_scores(-target, target, target, min_cross_section=10)
    np.testing.assert_allclose(inv.ic, -1.0)
    assert np.all(inv.decile_spread < 0)


def test_daily_scores_skips_thin_days() -> None:
    pred = np.full((3, 20), np.nan)
    target = np.full((3, 20), np.nan)
    rng = np.random.default_rng(8)
    pred[1] = rng.standard_normal(20)
    target[1] = rng.standard_normal(20)
    pred[2, :4] = rng.standard_normal(4)
    target[2, :4] = rng.standard_normal(4)
    d = daily_scores(pred, target, target, min_cross_section=10)
    assert d.day_index.tolist() == [1]


def test_summarise_daily_and_bootstrap() -> None:
    ic = np.array([0.05, 0.04, 0.06, 0.05, 0.03, 0.07])
    m = summarise_daily(5, DailyScores(np.arange(6), ic, ic * 2))
    assert m.mean_ic == pytest.approx(ic.mean())
    assert m.hit_rate == 1.0
    assert m.icir == pytest.approx(ic.mean() / ic.std(ddof=1))
    assert m.ci_lo < m.mean_ic < m.ci_hi


def test_summarise_daily_on_no_days() -> None:
    m = summarise_daily(5, DailyScores(np.zeros(0, np.int64), np.zeros(0), np.zeros(0)))
    assert m.n_days == 0
    assert math.isnan(m.mean_ic)


def test_bootstrap_ci_is_seeded_and_brackets_the_mean() -> None:
    rng = np.random.default_rng(9)
    vals = rng.standard_normal(300) * 0.1 + 0.05
    a = bootstrap_mean_ci(vals, n_boot=500, rng_seed=1)
    b = bootstrap_mean_ci(vals, n_boot=500, rng_seed=1)
    assert a == b
    assert a[0] < vals.mean() < a[1]
    assert all(math.isnan(v) for v in bootstrap_mean_ci(np.zeros(0)))


# ── The gate ──────────────────────────────────────────────────────────────────


def _metrics(mean_ic: float, ci_lo: float, ci_hi: float, h: int = 5) -> HorizonMetrics:
    return HorizonMetrics(
        horizon=h, n_days=100, mean_ic=mean_ic, std_ic=0.1, icir=mean_ic / 0.1,
        t_stat=1.0, hit_rate=0.55, ci_lo=ci_lo, ci_hi=ci_hi, decile_spread=0.001,
        daily=DailyScores(np.arange(100), np.full(100, mean_ic), np.zeros(100)),
    )


def _windows(
    *ics: float, h: int = 5, ci_lo: float | None = None
) -> dict[str, dict[int, HorizonMetrics]]:
    """N windows with the given OOS mean ICs; per-window CI is a diagnostic only."""
    out: dict[str, dict[int, HorizonMetrics]] = {}
    for i, ic in enumerate(ics, 1):
        lo = ci_lo if ci_lo is not None else ic - 0.01
        out[f"W{i}"] = {h: _metrics(ic, lo, ic + 0.03, h)}
    return out


# The r4_v2 run's 5d window ICs (audit/r4_v2/summary.json, 2026-09-06): the
# case the rule was redesigned on.  Eight positive windows, two of which the
# every-window rule failed by a hair, and a window-level t of 7.5.
_R4_V2_5D = (0.0499, 0.0625, 0.0229, 0.0295, 0.0413, 0.0458, 0.0433, 0.0180)


def test_gate_passes_when_the_windows_agree() -> None:
    g = gate_verdict(_windows(0.05, 0.04, 0.045, 0.055))
    assert g.passed and g.per_horizon[5]
    assert g.reasons == []
    assert g.evidence and "t " in g.evidence[0]


def test_gate_passes_the_r4_v2_shape_that_the_every_window_rule_failed() -> None:
    """Two windows individually miss; the eight together are unambiguous."""
    per_window = _windows(*_R4_V2_5D)
    # Reproduce the strict failures: W3's interval spans zero, W8 is under the floor.
    per_window["W3"][5] = _metrics(0.0229, -0.0066, 0.0527)
    g = gate_verdict(per_window)
    assert g.passed
    w = g.window_level[5]
    assert w.n_windows == 8 and w.n_positive == 8
    assert w.t_stat == pytest.approx(7.50, abs=0.05)
    assert w.t_crit == pytest.approx(2.365)
    # The retired rule is still reported, and still says what it said.
    assert g.strict_every_window[5] is False
    assert any("W3" in r and "includes zero" in r for r in g.strict_reasons)
    assert any("W8" in r and "mean IC" in r for r in g.strict_reasons)


def test_gate_fails_when_the_mean_is_below_the_materiality_floor() -> None:
    g = gate_verdict(_windows(0.015, 0.012, 0.018, 0.016, 0.014))
    assert not g.passed
    assert any("mean of window OOS ICs" in r for r in g.reasons)


def test_gate_fails_when_windows_disagree_beyond_chance() -> None:
    """Mean above the floor, but the windows scatter so widely the t is weak."""
    g = gate_verdict(_windows(0.12, -0.06, 0.09, -0.05, 0.03))
    assert not g.passed
    assert any("window-level t" in r for r in g.reasons)


def test_gate_fails_on_sign_disagreement_even_with_a_high_mean() -> None:
    """One enormous era must not carry three negative ones."""
    g = gate_verdict(_windows(0.40, -0.01, -0.01, -0.01, 0.30, -0.02, 0.35, -0.01))
    assert not g.passed
    assert any("windows positive" in r for r in g.reasons)


def test_gate_needs_enough_windows_for_a_window_level_test() -> None:
    g = gate_verdict(_windows(0.05, 0.06, 0.07))
    assert not g.passed
    assert any("need >= 4" in r for r in g.reasons)


def test_gate_bar_is_strict_inequality_at_the_threshold() -> None:
    g = gate_verdict(_windows(*([GATE_MIN_MEAN_IC] * 6)))
    assert not g.passed


def test_gate_passes_on_one_clean_horizon() -> None:
    """The allocator consumes one horizon; one clean horizon is a usable signal."""
    per_window = _windows(0.05, 0.04, 0.045, 0.055)
    for w in per_window.values():
        w[20] = _metrics(0.00, -0.02, 0.02, 20)
    g = gate_verdict(per_window)
    assert g.passed
    assert g.per_horizon == {5: True, 20: False}


def test_gate_with_no_windows_is_a_fail_not_a_vacuous_pass() -> None:
    g = gate_verdict({})
    assert not g.passed
    assert g.reasons == ["no windows evaluated"]


def test_gate_fails_a_horizon_with_no_scorable_days() -> None:
    empty = summarise_daily(20, DailyScores(np.zeros(0, np.int64), np.zeros(0), np.zeros(0)))
    per_window = _windows(0.05, 0.04, 0.045, 0.055, h=20)
    per_window["W2"][20] = empty
    g = gate_verdict(per_window)
    assert not g.passed
    assert any("no scorable OOS days" in r for r in g.reasons)


def test_gate_fails_a_window_below_the_day_floor() -> None:
    per_window = _windows(0.05, 0.04, 0.045, 0.055)
    thin = _metrics(0.9, 0.5, 1.0)
    thin.n_days = 3
    per_window["W1"][5] = thin
    g = gate_verdict(per_window)
    assert not g.passed
    assert any("only 3 scorable OOS day" in r for r in g.reasons)


def test_t_critical_table_is_sane() -> None:
    from trader.training.supervised import t_critical_95

    assert t_critical_95(7) == pytest.approx(2.365)
    assert t_critical_95(1) > t_critical_95(2) > t_critical_95(7) > t_critical_95(30)
    assert t_critical_95(100) == t_critical_95(30)   # conservative beyond the table
    with pytest.raises(ValueError):
        t_critical_95(0)


def test_gate_json_carries_the_rule_and_the_window_level_test() -> None:
    per_window = _windows(*_R4_V2_5D)
    g = gate_verdict(per_window)
    pooled = {5: _metrics(0.0392, 0.0317, 0.0469)}
    payload = gate_json_payload(g, pooled, n_windows=8)
    assert payload["gate_rule"] == GATE_RULE
    assert payload["window_level"]["5"]["n_windows"] == 8
    assert payload["strict_every_window"]["5"] is False
    assert payload["evidence"]
    assert "NaN" not in json.dumps(payload)


# ── gate.json (pinned schema) ─────────────────────────────────────────────────

PINNED_GATE_KEYS = {
    "verdict", "mean_ic_5d", "mean_ic_20d", "ic_ci_low", "ic_ci_high", "icir", "n_windows",
}


def test_gate_json_payload_matches_the_pinned_schema() -> None:
    pooled = {5: _metrics(0.05, 0.02, 0.08, 5), 20: _metrics(0.01, -0.01, 0.03, 20)}
    per_window = {
        "W1": {5: _metrics(0.05, 0.02, 0.08, 5), 20: _metrics(0.01, -0.01, 0.03, 20)},
        "W2": {5: _metrics(0.04, 0.01, 0.07, 5), 20: _metrics(0.00, -0.02, 0.02, 20)},
        "W3": {5: _metrics(0.06, 0.03, 0.09, 5), 20: _metrics(0.02, -0.01, 0.05, 20)},
        "W4": {5: _metrics(0.05, 0.02, 0.08, 5), 20: _metrics(0.01, -0.01, 0.03, 20)},
    }
    gate = gate_verdict(per_window)
    payload = gate_json_payload(gate, pooled, n_windows=4)
    assert PINNED_GATE_KEYS <= set(payload)
    assert payload["verdict"] == "PASS"
    assert payload["mean_ic_5d"] == pytest.approx(0.05)
    assert payload["mean_ic_20d"] == pytest.approx(0.01)
    assert payload["n_windows"] == 4
    # The single CI/ICIR triple describes the horizon the verdict rests on.
    assert payload["gate_horizon"] == 5
    assert payload["ic_ci_low"] == pytest.approx(0.02)
    assert payload["ic_ci_high"] == pytest.approx(0.08)
    assert payload["icir"] == pytest.approx(0.5)


def test_gate_json_fail_is_reported_as_fail() -> None:
    pooled = {5: _metrics(0.001, -0.01, 0.02, 5), 20: _metrics(-0.02, -0.05, 0.01, 20)}
    payload = gate_json_payload(gate_verdict({"W1": pooled}), pooled, n_windows=1)
    assert payload["verdict"] == "FAIL"
    assert payload["gate_horizon"] == 5          # least bad, so the shortfall is visible
    assert payload["reasons"]


def test_gate_json_is_parseable_json_even_with_no_scorable_days() -> None:
    """`NaN` is not valid JSON. A metric that does not exist is `null`."""
    empty = summarise_daily(5, DailyScores(np.zeros(0, np.int64), np.zeros(0), np.zeros(0)))
    pooled = {5: empty}
    payload = gate_json_payload(gate_verdict({"W1": pooled}), pooled, n_windows=1)
    text = json.dumps(payload)
    assert "NaN" not in text
    assert json.loads(text)["mean_ic_5d"] is None
    assert json.loads(text)["verdict"] == "FAIL"


# ── predictions.parquet (pinned schema) ───────────────────────────────────────


def test_predictions_schema_is_the_pinned_contract() -> None:
    assert predictions_schema((5, 20)) == {
        "date": pl.Date, "ticker": pl.Utf8, "r_hat_5d": pl.Float64, "r_hat_20d": pl.Float64,
    }
    assert empty_predictions_frame((5, 20)).schema == predictions_schema((5, 20))
    assert empty_predictions_frame((5, 20)).height == 0


def test_predictions_frame_one_row_per_date_tradeable_ticker() -> None:
    panel = _synthetic_panel(12, ["A", "B", "C"], seed=10)
    t = build_panel_tensors(panel, ["A", "B", "C"], FEATS, (2,), min_cross_section=2)
    preds = {2: np.full((12, 3), np.nan, dtype=np.float32)}
    preds[2][5:] = 0.5
    t.mask[6, 1] = False                       # B untradeable on day 6
    df = predictions_frame(preds, t, (2,))
    assert df.schema == predictions_schema((2,))
    assert df.height == 7 * 3 - 1
    assert df.filter((pl.col("date") == t.dates[6]) & (pl.col("ticker") == "B")).is_empty()


# ── Leakage: the whole game ───────────────────────────────────────────────────


def test_future_return_perturbation_leaves_the_prediction_at_t_bit_identical() -> None:
    """Perturb a return at t' > t; the prediction dated t must not move one bit.

    `log_return_1d` is both a feature and the raw material of the label, so this
    is the sharpest available probe: if any part of the window construction were
    off by one — an unshifted target, an inclusive slice — this test moves.
    """
    tickers = [f"T{i}" for i in range(12)]
    base = _synthetic_panel(60, tickers, seed=11, signal=0.4)
    t_pred = 30
    dates = sorted(base["date"].unique().to_list())

    perturbed = base.with_columns(
        pl.when(pl.col("date") > dates[t_pred])
        .then(pl.col("log_return_1d") + 1.0)     # an absurd, unmissable future shock
        .otherwise(pl.col("log_return_1d"))
        .alias("log_return_1d")
    )
    assert not perturbed.equals(base)

    model = _tiny_model(len(FEATS), (2,))
    out = []
    for panel in (base, perturbed):
        tensors = build_panel_tensors(panel, tickers, FEATS, (2,), min_cross_section=5)
        out.append(
            predict_panel(model, tensors, lookback=5, device=torch.device("cpu"))[2]
        )
    a, b = out
    assert np.isfinite(a[t_pred]).any()
    # Bit-identical, not "close": same bytes.
    assert a[: t_pred + 1].tobytes() == b[: t_pred + 1].tobytes()
    # And the probe is live — the perturbed feature does move later predictions.
    assert a[t_pred + 6].tobytes() != b[t_pred + 6].tobytes()


def test_training_never_sees_the_test_split() -> None:
    """Perturb the OOS segment; the trained weights must be byte-identical.

    Normalisation stats, targets and gradients all come from the train split
    (`runner.py:111` convention). If any of them reached into val or test, this
    changes the encoder hash.
    """
    tickers = [f"T{i}" for i in range(10)]
    window = WindowConfig(
        name="W1",
        train_start=date(2018, 1, 1), train_end=date(2018, 6, 30),
        val_start=date(2018, 8, 1), val_end=date(2018, 9, 30),
        test_start=date(2018, 11, 1), test_end=date(2018, 12, 31),
    )
    base = _synthetic_panel(260, tickers, start=date(2018, 1, 1), seed=12, signal=0.5)
    shocked = base.with_columns(
        pl.when(pl.col("date") >= window.test_start)
        .then(pl.col("sig") * 100.0 + 7.0)
        .otherwise(pl.col("sig"))
        .alias("sig")
    )

    cfg = SupervisedConfig(
        lookback=5, horizons=(2,), batch_days=16, eval_batch_days=32,
        max_epochs=2, max_steps=6, n_boot=20, min_cross_section=5, seed=3,
        device="cpu", lr_schedule="constant",
    )
    hashes = []
    for panel in (base, shocked):
        train_t = build_panel_tensors(
            panel.filter(
                pl.col("date").is_between(window.train_start, window.train_end)
            ),
            tickers, FEATS, (2,), min_cross_section=5,
        )
        val_t = build_panel_tensors(
            panel.filter(pl.col("date").is_between(window.val_start, window.val_end)),
            tickers, FEATS, (2,), min_cross_section=5,
        )
        model = _tiny_model(len(FEATS), (2,))
        train_signal_window(model, train_t, val_t, cfg, device=torch.device("cpu"))
        hashes.append(model.encoder_state_sha256())
    assert hashes[0] == hashes[1]


# ── Training loop ─────────────────────────────────────────────────────────────


def test_training_recovers_a_planted_signal_in_sample() -> None:
    """Not evidence of market signal — evidence the loop optimises the right thing.

    The panel is synthetic and the signal was planted by the fixture, so the IC
    below says the plumbing works and nothing whatsoever about markets.
    """
    tickers = [f"T{i}" for i in range(24)]
    panel = _synthetic_panel(220, tickers, seed=13, signal=1.5)
    tensors = build_panel_tensors(panel, tickers, FEATS, (1,), min_cross_section=10)
    cfg = SupervisedConfig(
        lookback=5, horizons=(1,), batch_days=32, max_epochs=12, learning_rate=3e-3,
        n_boot=50, min_cross_section=10, seed=1, device="cpu", lr_schedule="constant",
        patience=12,
    )
    model = _tiny_model(len(FEATS), (1,))
    hist = train_signal_window(model, tensors, tensors, cfg, device=torch.device("cpu"))
    assert hist.n_steps > 0
    preds = predict_panel(model, tensors, lookback=5, device=torch.device("cpu"))
    metrics = evaluate_predictions(preds, tensors, min_cross_section=10, n_boot=50)
    assert metrics[1].mean_ic > 0.2, metrics[1].mean_ic


def test_train_refuses_a_split_with_no_labelled_lookback_day() -> None:
    tickers = [f"T{i}" for i in range(12)]
    tensors = build_panel_tensors(
        _synthetic_panel(6, tickers, seed=14), tickers, FEATS, (2,), min_cross_section=5
    )
    cfg = SupervisedConfig(lookback=30, horizons=(2,), device="cpu")
    with pytest.raises(ValueError, match="no labelled day"):
        train_signal_window(
            _tiny_model(len(FEATS), (2,)), tensors, tensors, cfg, device=torch.device("cpu")
        )


def test_supervised_config_rejects_an_unknown_schedule() -> None:
    with pytest.raises(ValueError, match="lr_schedule"):
        SupervisedConfig(lr_schedule="linear")


def test_predict_panel_leaves_warmup_days_unpredicted() -> None:
    tickers = [f"T{i}" for i in range(8)]
    tensors = build_panel_tensors(
        _synthetic_panel(30, tickers, seed=15), tickers, FEATS, (2,), min_cross_section=4
    )
    preds = predict_panel(
        _tiny_model(len(FEATS), (2,)), tensors, lookback=7, device=torch.device("cpu")
    )
    assert np.all(np.isnan(preds[2][:6]))
    assert np.isfinite(preds[2][6]).any()


# ── End to end: the artefact contract ─────────────────────────────────────────


def _run_end_to_end(tmp_path: Path, tickers: list[str]) -> tuple[object, Path]:
    panel = _synthetic_panel(560, tickers, start=date(2018, 1, 1), seed=16, signal=0.8)
    windows = compute_windows(
        data_start=min(panel["date"].to_list()),
        data_end=max(panel["date"].to_list()),
        train_years=1, val_months=3, test_months=3, purge_months=3,
        n_windows=1, step_months=12,
    )
    assert windows, "fixture must produce at least one window"
    out_dir = tmp_path / "signal" / "test_tag"
    summary = run_signal_walk_forward(
        full_panel=panel,
        windows=windows,
        tickers=tickers,
        feature_cols=FEATS,
        model_cfg=SignalConfig(
            in_features=len(FEATS), embed_dim=8, num_channels=[8, 8],
            kernel_size=2, dropout=0.0, head_hidden=4, horizons=(5, 20),
        ),
        train_cfg=SupervisedConfig(
            lookback=10, horizons=(5, 20), batch_days=32, eval_batch_days=64,
            max_epochs=1, max_steps=4, n_boot=25, min_cross_section=5, seed=7,
            device="cpu", lr_schedule="constant",
        ),
        out_dir=out_dir,
        tag="test_tag",
        mlflow_port=None,            # no tracking server in unit tests
    )
    return summary, out_dir


def test_end_to_end_writes_the_pinned_artefact_set(tmp_path: Path) -> None:
    tickers = [f"T{i}" for i in range(14)]
    summary, out_dir = _run_end_to_end(tmp_path, tickers)

    for name in ("predictions.parquet", "embeddings.npy", "index.json", "gate.json"):
        assert (out_dir / name).exists(), name

    preds = pl.read_parquet(out_dir / "predictions.parquet")
    assert preds.schema == predictions_schema((5, 20))
    assert preds.height > 0

    index = json.loads((out_dir / "index.json").read_text())
    assert index["tickers"] == tickers            # active_tickers() order, exactly
    assert index["feature_cols"] == FEATS
    assert index["embed_dim"] == 8
    assert len(index["encoder_state_sha256"]) == 64

    emb = np.load(out_dir / "embeddings.npy", mmap_mode="r")
    assert emb.dtype == np.float16
    assert emb.shape == (len(index["dates"]), len(tickers), index["embed_dim"])

    gate = json.loads((out_dir / "gate.json").read_text())
    assert PINNED_GATE_KEYS <= set(gate)
    assert gate["verdict"] in ("PASS", "FAIL")
    assert gate["n_windows"] == 1


def test_end_to_end_oos_predictions_are_confined_to_the_test_segment(
    tmp_path: Path,
) -> None:
    """`predictions.parquet` is OOS only — nothing in it may predate test_start."""
    tickers = [f"T{i}" for i in range(14)]
    summary, out_dir = _run_end_to_end(tmp_path, tickers)
    windows = [w.window for w in summary.windows]  # type: ignore[attr-defined]
    preds = pl.read_parquet(out_dir / "predictions.parquet")
    earliest = min(w.test_start for w in windows)
    assert preds["date"].min() >= earliest


def test_end_to_end_embeddings_come_from_the_saved_encoder(tmp_path: Path) -> None:
    """`index.json` must stamp the encoder that actually produced embeddings.npy."""
    tickers = [f"T{i}" for i in range(14)]
    summary, out_dir = _run_end_to_end(tmp_path, tickers)
    index = json.loads((out_dir / "index.json").read_text())
    reloaded = SignalModel(
        SignalConfig(
            in_features=len(FEATS), embed_dim=8, num_channels=[8, 8],
            kernel_size=2, dropout=0.0, head_hidden=4, horizons=(5, 20),
        ),
        # The saved encoder carries its train-split normaliser as buffers, so a
        # reload has to have the same slots to put them in.
        feat_mean=torch.zeros(len(FEATS)),
        feat_std=torch.ones(len(FEATS)),
    )
    reloaded.load_state_dict(torch.load(out_dir / "signal_model.pt", weights_only=True))
    assert reloaded.encoder_state_sha256() == index["encoder_state_sha256"]


def test_end_to_end_refuses_a_dead_feature(tmp_path: Path) -> None:
    """Gate 0 fires before a single optimiser step is taken."""
    tickers = [f"T{i}" for i in range(12)]
    panel = _synthetic_panel(560, tickers, start=date(2018, 1, 1), seed=17).with_columns(
        pl.lit(1.0).alias("beta_nifty_60d")
    )
    windows = compute_windows(
        data_start=min(panel["date"].to_list()),
        data_end=max(panel["date"].to_list()),
        train_years=1, val_months=3, test_months=3, purge_months=3,
        n_windows=1, step_months=12,
    )
    with pytest.raises(DeadFeatureError, match="beta_nifty_60d"):
        run_signal_walk_forward(
            full_panel=panel,
            windows=windows,
            tickers=tickers,
            feature_cols=[*FEATS, "beta_nifty_60d"],
            model_cfg=SignalConfig(
                in_features=len(FEATS) + 1, embed_dim=8, num_channels=[8],
                kernel_size=2, dropout=0.0, head_hidden=4, horizons=(5,),
            ),
            train_cfg=SupervisedConfig(
                lookback=10, horizons=(5,), max_epochs=1, max_steps=1, n_boot=10,
                min_cross_section=5, device="cpu",
            ),
            out_dir=tmp_path / "dead",
            tag="dead",
            mlflow_port=None,
        )


def test_run_rejects_mismatched_horizons(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="model horizons"):
        run_signal_walk_forward(
            full_panel=_synthetic_panel(30, ["A"], seed=18),
            windows=[
                WindowConfig(
                    name="W1",
                    train_start=date(2018, 1, 1), train_end=date(2018, 1, 20),
                    val_start=date(2018, 1, 21), val_end=date(2018, 1, 25),
                    test_start=date(2018, 1, 26), test_end=date(2018, 2, 1),
                )
            ],
            tickers=["A"],
            feature_cols=FEATS,
            model_cfg=SignalConfig(in_features=len(FEATS), horizons=(5,)),
            train_cfg=SupervisedConfig(horizons=(5, 20), device="cpu"),
            out_dir=tmp_path / "mismatch",
            tag="mismatch",
            mlflow_port=None,
        )


def test_run_rejects_no_windows(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no walk-forward windows"):
        run_signal_walk_forward(
            full_panel=_synthetic_panel(10, ["A"], seed=19),
            windows=[],
            tickers=["A"],
            feature_cols=FEATS,
            model_cfg=SignalConfig(in_features=len(FEATS)),
            train_cfg=SupervisedConfig(device="cpu"),
            out_dir=tmp_path / "none",
            tag="none",
            mlflow_port=None,
        )


def test_panel_tensors_dataclass_accessors() -> None:
    t = PanelTensors(
        dates=[date(2020, 1, 1)], tickers=["A", "B"], feature_cols=["x"],
        features=np.zeros((1, 2, 1), np.float32), mask=np.ones((1, 2), bool),
        targets={1: np.zeros((1, 2), np.float32)},
        fwd_raw={1: np.zeros((1, 2), np.float32)},
    )
    assert (t.n_days, t.n_tickers) == (1, 2)


def test_listnet_loss_prefers_the_correct_top() -> None:
    """ListNet is lower when the prediction ranks the target's top name first."""
    import torch

    from trader.models.heads import listnet_loss

    tgt = torch.tensor([[2.0, 0.0, -2.0, 0.5]])
    m = torch.tensor([[True, True, True, False]])
    good = listnet_loss(torch.tensor([[3.0, 0.0, -3.0, 99.0]]), tgt, m)
    bad = listnet_loss(torch.tensor([[-3.0, 0.0, 3.0, 99.0]]), tgt, m)
    assert torch.isfinite(good) and good < bad
    # a masked name's prediction (99.0) must not matter
    other = listnet_loss(torch.tensor([[3.0, 0.0, -3.0, -99.0]]), tgt, m)
    torch.testing.assert_close(good, other)


def test_listnet_loss_degenerate_rows_are_zero_not_nan() -> None:
    import torch

    from trader.models.heads import listnet_loss

    p = torch.randn(2, 3, requires_grad=True)
    m = torch.tensor([[True, False, False], [False, False, False]])
    out = listnet_loss(p, torch.randn(2, 3), m)
    out.backward()
    assert out.item() == 0.0 and torch.isfinite(p.grad).all()


def test_supervised_config_rejects_unknown_loss() -> None:
    import pytest

    from trader.training.supervised import SupervisedConfig

    with pytest.raises(ValueError, match="loss must be"):
        SupervisedConfig(loss="huber")
