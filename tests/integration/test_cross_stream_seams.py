"""End-to-end tests of the seams BETWEEN the four streams.

Each stream was built and verified in isolation on a disjoint file set, so
every defect that lived in the *handoff* between two of them was invisible to
both.  These tests exercise the handoffs on the real code with small synthetic
data — no GPU, no real panel, no training run.

The seams, in the order the pipeline runs them:

a. R4 writes the pinned signal artefact set; R5's ``run_allocator.py`` readers
   consume it and produce weights.
b. R6's ``EmbeddingCache`` opens what R4's writer produced, and its sha guard
   actually refuses a mismatched encoder.
c. R6's ``AllocatorEnv`` drives R5's real ``allocate()`` and ``RebalanceSchedule``
   -- not a test double -- and one env step advances exactly one rebalance period.
d. R6's ``train_allocator_rl.py`` refuses to start on a FAIL gate and proceeds
   on a PASS.  Both directions, because a gate that never fires is not a gate.
e. R8's ext feature group is genuinely opt-in: ``FEATURE_COLS`` is untouched and
   R4 produces bit-identical output with the group disabled.

The panel generator is ``tests/fixtures/synthetic_panel.py``, committed so the
evidence outlives the session that produced it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest
import torch

from tests.fixtures.synthetic_panel import make_signal_panel
from trader.allocator.rebalance import RebalanceSchedule
from trader.data.features import FEATURE_COLS
from trader.models.signal import SignalConfig
from trader.training.supervised import (
    SupervisedConfig,
    run_signal_walk_forward,
)
from trader.training.walk_forward import WindowConfig

REPO_ROOT = Path(__file__).resolve().parents[2]

# Small enough to run in seconds, large enough that every code path is real:
# the lookback, the purge and the three splits all have to fit.
_N_DATES = 340
_N_TICKERS = 12
_LOOKBACK = 20


def _windows(dates: list[Any]) -> list[WindowConfig]:
    """Two non-overlapping walk-forward windows over the fixture's calendar."""
    return [
        WindowConfig(
            name="w1",
            train_start=dates[0],
            train_end=dates[149],
            val_start=dates[160],
            val_end=dates[199],
            test_start=dates[210],
            test_end=dates[269],
        ),
    ]


def _run_r4(out_dir: Path, panel: pl.DataFrame, tickers: list[str]) -> Any:
    """The real R4 orchestrator, capped to a handful of optimiser steps."""
    dates = sorted(panel["date"].unique().to_list())
    train_cfg = SupervisedConfig(
        lookback=_LOOKBACK,
        horizons=(5, 20),
        batch_days=8,
        eval_batch_days=16,
        max_epochs=1,
        max_steps=3,          # a smoke run: this proves the loop, not the model
        n_boot=64,
        min_cross_section=5,
        seed=0,
        device="cpu",         # never touch the GPU from a test
    )
    model_cfg = SignalConfig(
        in_features=len(FEATURE_COLS),
        embed_dim=8,
        num_channels=[8, 8],
        head_hidden=8,
        horizons=(5, 20),
    )
    return run_signal_walk_forward(
        full_panel=panel,
        windows=_windows(dates),
        tickers=tickers,
        feature_cols=list(FEATURE_COLS),
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        out_dir=out_dir,
        tag="seam",
        mlflow_port=None,     # no MLflow server in a test
    )


@pytest.fixture(scope="module")
def r4_artefacts(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Run R4 ONCE and share the artefact directory across the seam tests."""
    tmp = tmp_path_factory.mktemp("r4")
    panel = make_signal_panel(
        n_dates=_N_DATES, n_tickers=_N_TICKERS, seed=0, signal_strength=0.0
    )
    tickers = sorted(panel["ticker"].unique().to_list())
    out_dir = tmp / "signal" / "seam"
    summary = _run_r4(out_dir, panel, tickers)
    panel_path = tmp / "panel.parquet"
    panel.write_parquet(panel_path)

    # A second panel sliced to the OOS span plus `lookback` days of run-up.
    # The env needs `lookback` rows of history before its first tradeable day,
    # and R4's predictions are OOS-only -- so an env built over the FULL panel
    # starts far before any prediction exists and (correctly) holds on every
    # step, exercising nothing. Seam (c) needs the two calendars to overlap.
    oos = sorted(
        pl.read_parquet(out_dir / "predictions.parquet")["date"].unique().to_list()
    )
    all_dates = sorted(panel["date"].unique().to_list())
    first = all_dates.index(oos[0])
    span = all_dates[max(0, first - _LOOKBACK - 1) : all_dates.index(oos[-1]) + 1]
    oos_panel_path = tmp / "panel_oos.parquet"
    panel.filter(pl.col("date").is_in(span)).write_parquet(oos_panel_path)
    return {
        "dir": out_dir,
        "panel": panel,
        "panel_path": panel_path,
        "oos_panel_path": oos_panel_path,
        "oos_dates": oos,
        "tickers": tickers,
        "summary": summary,
    }


# ── seam (a): R4 artefacts -> R5's allocator readers ─────────────────────────


def test_seam_a_r4_writes_the_pinned_artefact_set(r4_artefacts: dict[str, Any]) -> None:
    """All four pinned files exist with the pinned schema."""
    d = r4_artefacts["dir"]
    for name in ("predictions.parquet", "embeddings.npy", "index.json", "gate.json"):
        assert (d / name).exists(), f"{name} missing from the pinned artefact set"

    preds = pl.read_parquet(d / "predictions.parquet")
    assert preds.schema["date"] == pl.Date
    assert preds.schema["ticker"] == pl.String
    # f64 is pinned. R6's embedding_cache wrote f32 to this filename once.
    assert preds.schema["r_hat_5d"] == pl.Float64
    assert preds.schema["r_hat_20d"] == pl.Float64
    # One row per (date, ticker) -- the contract the concat used to break.
    assert preds.select(["date", "ticker"]).unique().height == preds.height

    index = json.loads((d / "index.json").read_text())
    for key in ("dates", "tickers", "embed_dim", "feature_cols", "encoder_state_sha256"):
        assert key in index, f"index.json missing pinned key {key!r}"
    assert index["feature_cols"] == list(FEATURE_COLS)

    emb = np.load(d / "embeddings.npy", mmap_mode="r")
    assert emb.dtype == np.float16
    assert emb.shape == (len(index["dates"]), len(index["tickers"]), index["embed_dim"])
    # Axis 1 of embeddings.npy MUST equal index.json["tickers"].
    assert index["tickers"] == r4_artefacts["tickers"]

    gate = json.loads((d / "gate.json").read_text())
    for key in ("verdict", "mean_ic_5d", "mean_ic_20d", "ic_ci_low", "ic_ci_high",
                "icir", "n_windows"):
        assert key in gate, f"gate.json missing pinned key {key!r}"
    assert gate["verdict"] in ("PASS", "FAIL")


def test_seam_a_r5_readers_turn_r4_predictions_into_weights(
    r4_artefacts: dict[str, Any],
) -> None:
    """R5's real readers + real `allocate()` over R4's real artefacts.

    This is the seam that no single stream could test: R5's `_dense_signal`
    scatters R4's long-format parquet onto the [T, N] grid the allocator wants,
    and a shape or ordering disagreement between the two would surface here and
    nowhere else.
    """
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    try:
        from run_allocator import _dense_column, _dense_signal
    finally:
        sys.path.pop(0)
    from trader.allocator import AllocatorParams, allocate

    d = r4_artefacts["dir"]
    tickers = r4_artefacts["tickers"]
    dates = sorted(r4_artefacts["panel"]["date"].unique().to_list())

    r_hat_all = _dense_signal(d / "predictions.parquet", dates, tickers, ("5d", "20d"))
    vol_all = _dense_column(r4_artefacts["panel_path"], "realized_vol_20d", dates, tickers)
    assert set(r_hat_all) == {"5d", "20d"}
    for arr in r_hat_all.values():
        assert arr.shape == (len(dates), len(tickers))
    assert vol_all.shape == (len(dates), len(tickers))

    r_hat = r_hat_all["20d"]
    populated = np.isfinite(r_hat).any(axis=1)
    assert populated.any(), "R5 read no predictions at all out of R4's artefact"

    sector_ids = np.array([1 + (i % 5) for i in range(len(tickers))], dtype=np.int64)
    params = AllocatorParams()
    current = np.r_[1.0, np.zeros(len(tickers))]
    n_invested = 0
    for t in np.flatnonzero(populated)[:20]:
        w = allocate(
            r_hat[t],
            vol_all[t],
            np.isfinite(vol_all[t]) & (vol_all[t] > 0),
            sector_ids,
            current,
            params,
        )
        assert w.shape == (len(tickers) + 1,)
        assert w.min() >= -1e-12
        assert w.sum() == pytest.approx(1.0, abs=1e-9)
        assert w[1:].max() <= params.max_name_weight + 1e-9
        sec_w = np.bincount(sector_ids, weights=w[1:], minlength=6)
        assert sec_w[1:].max() <= params.max_sector_weight + 1e-9
        if w[1:].sum() > 0:
            n_invested += 1
        current = w
    assert n_invested > 0, "R4's predictions produced an all-cash book on every day"


# ── seam (b): R4 artefacts -> R6's EmbeddingCache, and the sha guard ─────────


def test_seam_b_embedding_cache_opens_r4s_artefact_directory(
    r4_artefacts: dict[str, Any],
) -> None:
    """R6's reader must open R4's writer's output without a conversion step.

    Both sides implement the same ``index.json`` contract, but they are
    different modules written by different streams -- the field names
    (``feature_cols``, ``encoder_state_sha256``) are the seam, and a rename on
    either side is invisible until something opens the other's directory.
    """
    from trader.training.embedding_cache import EmbeddingCache

    d = r4_artefacts["dir"]
    index = json.loads((d / "index.json").read_text())

    cache = EmbeddingCache(d, expected_sha256=index["encoder_state_sha256"])
    assert cache.tickers == r4_artefacts["tickers"]
    assert cache.feature_cols == list(FEATURE_COLS)
    assert cache.embed_dim == index["embed_dim"]
    assert len(cache.dates) == len(index["dates"])

    block = cache.window(cache.dates[0], cache.dates[min(3, len(cache.dates) - 1)])
    assert block.shape[0] >= 1
    assert block.shape[1] == len(cache.tickers)
    assert block.shape[2] == cache.embed_dim
    assert np.isfinite(np.asarray(block, dtype=np.float32)).all()


def test_seam_b_the_sha_guard_refuses_a_mismatched_encoder(
    r4_artefacts: dict[str, Any],
) -> None:
    """The guard must actually fire -- an unchecked cache is the whole hazard.

    An embedding cache silently paired with a different encoder produces
    embeddings that mean nothing, and nothing downstream can detect it.
    """
    from trader.training.embedding_cache import EmbeddingCache, EncoderHashMismatch

    d = r4_artefacts["dir"]
    with pytest.raises(EncoderHashMismatch):
        EmbeddingCache(d, expected_sha256="0" * 64)

    # And a real, differently-initialised encoder must be refused too -- not
    # just a made-up string.
    from trader.models.signal import SignalConfig, SignalModel

    other = SignalModel(
        SignalConfig(
            in_features=len(FEATURE_COLS), embed_dim=8, num_channels=[8, 8],
            head_hidden=8, horizons=(5, 20),
        ),
        feat_mean=torch.zeros(len(FEATURE_COLS)),
        feat_std=torch.ones(len(FEATURE_COLS)),
    )
    index = json.loads((d / "index.json").read_text())
    assert other.encoder_state_sha256() != index["encoder_state_sha256"], (
        "the freshly built encoder happens to hash identically; this test is inert"
    )
    with pytest.raises(EncoderHashMismatch):
        EmbeddingCache(d, encoder=other)


# ── seam (c): R6's AllocatorEnv drives R5's real allocate + schedule ─────────


def _inner_env(panel_path: Path, tickers: list[str], freq: str, episode_length: int = 40) -> Any:
    from trader.allocator.rebalance import RebalanceSchedule
    from trader.env.panel_env import PanelTradingEnv
    from trader.env.reward import LogReturn

    return PanelTradingEnv(
        panel_path=panel_path,
        universe=tickers,
        feature_columns=["log_return_1d"],
        lookback=_LOOKBACK,
        episode_length=episode_length,
        rebalance_schedule=RebalanceSchedule(freq=freq),  # type: ignore[arg-type]
        use_excess_returns=True,
        turnover_penalty=0.0,
        reward_fn=LogReturn(),
        seed=0,
    )


def test_seam_c_one_env_step_advances_exactly_one_rebalance_period(
    r4_artefacts: dict[str, Any],
) -> None:
    """R6's env, R5's real schedule, R5's real `allocate()` -- no test double.

    `AllocatorEnv.step` sums the inner env's daily rewards over a period and
    calls the sum the period's excess log return.  That identity requires the
    step boundary to be a REBALANCE boundary; if it is not, `periods_per_episode`
    (read as months) and the x12 annualisation are silently wrong.
    """
    from trader.env.allocator_env import AllocatorEnv, SignalPanel

    d = r4_artefacts["dir"]
    tickers = r4_artefacts["tickers"]
    signal = SignalPanel.from_artifacts(
        d, r4_artefacts["oos_panel_path"], tickers, horizon="r_hat_20d", vol_lookback=20
    )
    inner = _inner_env(r4_artefacts["oos_panel_path"], tickers, "monthly")
    env = AllocatorEnv(inner, signal, seed=0)
    env.reset(seed=0)

    schedule = inner.rebalance_schedule
    assert schedule is not None
    n_days_seen: list[int] = []
    for _ in range(4):
        _, _, term, trunc, info = env.step(env.action_space.sample())
        n_days_seen.append(int(info["n_days"]))
        if term or trunc:
            break

    assert n_days_seen, "the env terminated before a single period completed"
    # A monthly period is many days, never one. The `rebalance_schedule=None`
    # bug made every period exactly one day, with no error and no log line.
    assert min(n_days_seen) > 1, (
        f"a 'rebalance period' was {min(n_days_seen)} day(s): the step boundary "
        f"is not a rebalance boundary and the x12 annualisation is invalid"
    )
    assert max(n_days_seen) <= 31, n_days_seen


def test_seam_c_allocator_env_refuses_an_inner_env_with_no_schedule(
    r4_artefacts: dict[str, Any],
) -> None:
    """Without a schedule a 'period' is one day; that must be an error, not a default."""
    from trader.env.allocator_env import AllocatorEnv, InnerEnvMisconfigured, SignalPanel
    from trader.env.panel_env import PanelTradingEnv
    from trader.env.reward import LogReturn

    d = r4_artefacts["dir"]
    tickers = r4_artefacts["tickers"]
    signal = SignalPanel.from_artifacts(
        d, r4_artefacts["panel_path"], tickers, horizon="r_hat_20d", vol_lookback=20
    )
    inner = PanelTradingEnv(
        panel_path=r4_artefacts["oos_panel_path"],
        universe=tickers,
        feature_columns=["log_return_1d"],
        lookback=_LOOKBACK,
        episode_length=40,
        rebalance_schedule=None,
        use_excess_returns=True,
        turnover_penalty=0.0,
        reward_fn=LogReturn(),
        seed=0,
    )
    with pytest.raises(InnerEnvMisconfigured, match="rebalance_schedule"):
        AllocatorEnv(inner, signal, seed=0)


def test_seam_c_the_env_uses_r5s_real_allocate_not_a_double(
    r4_artefacts: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break R5's `allocate` and R6's env must break with it.

    If the env had its own copy of the sizing logic, or imported a double, this
    would pass unchanged -- which is exactly the seam this file exists to test.
    """
    import trader.env.allocator_env as ae
    from trader.env.allocator_env import AllocatorEnv, SignalPanel

    calls: list[int] = []
    real = ae.allocate

    def counting_allocate(*args: Any, **kwargs: Any) -> Any:
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(ae, "allocate", counting_allocate)

    d = r4_artefacts["dir"]
    tickers = r4_artefacts["tickers"]
    signal = SignalPanel.from_artifacts(
        d, r4_artefacts["oos_panel_path"], tickers, horizon="r_hat_20d", vol_lookback=20
    )
    env = AllocatorEnv(
        _inner_env(r4_artefacts["oos_panel_path"], tickers, "monthly"), signal, seed=0
    )
    env.reset(seed=0)
    for _ in range(2):
        _, _, term, trunc, _ = env.step(env.action_space.sample())
        if term or trunc:
            break
    assert calls, "AllocatorEnv never called trader.allocator.deterministic.allocate"


# ── seam (d): the R4 gate actually gates the R6 training run ────────────────


def _gate_dir(tmp_path: Path, src: Path, verdict: str) -> Path:
    """Copy an artefact directory, overwriting only ``gate.json``'s verdict."""
    import shutil

    dst = tmp_path / f"signal_{verdict.lower()}"
    shutil.copytree(src, dst)
    payload = json.loads((dst / "gate.json").read_text())
    payload["verdict"] = verdict
    (dst / "gate.json").write_text(json.dumps(payload, indent=2))
    return dst


def _require_passing_gate() -> Any:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    try:
        from train_allocator_rl import SignalGateNotPassed, require_passing_gate

        return require_passing_gate, SignalGateNotPassed
    finally:
        sys.path.pop(0)


def test_seam_d_training_refuses_to_start_on_a_failed_gate(
    r4_artefacts: dict[str, Any], tmp_path: Path
) -> None:
    """A FAIL gate must stop the RL run. This is CLAUDE.md rule 1 in code."""
    require_passing_gate, SignalGateNotPassed = _require_passing_gate()
    failed = _gate_dir(tmp_path, r4_artefacts["dir"], "FAIL")
    with pytest.raises(SignalGateNotPassed) as exc:
        require_passing_gate(failed)
    assert "FAIL" in str(exc.value)


def test_seam_d_training_proceeds_on_a_passing_gate(
    r4_artefacts: dict[str, Any], tmp_path: Path
) -> None:
    """And it must PROCEED on a PASS -- a gate that always refuses is not a gate.

    Testing only the refusing direction is how a permanently-closed gate ships
    looking like a working one.
    """
    require_passing_gate, _ = _require_passing_gate()
    passed = _gate_dir(tmp_path, r4_artefacts["dir"], "PASS")
    payload = require_passing_gate(passed)
    assert payload["verdict"] == "PASS"


def test_seam_d_a_missing_or_incomplete_artefact_set_is_also_refused(
    r4_artefacts: dict[str, Any], tmp_path: Path
) -> None:
    """PASS with no predictions.parquet is an incomplete directory, not a pass."""
    require_passing_gate, SignalGateNotPassed = _require_passing_gate()

    empty = tmp_path / "no_gate"
    empty.mkdir()
    with pytest.raises((SignalGateNotPassed, FileNotFoundError)):
        require_passing_gate(empty)

    truncated = _gate_dir(tmp_path, r4_artefacts["dir"], "PASS")
    (truncated / "predictions.parquet").unlink()
    with pytest.raises(SignalGateNotPassed, match="predictions"):
        require_passing_gate(truncated)


def test_seam_d_run_allocator_also_refuses_a_failed_gate(
    r4_artefacts: dict[str, Any], tmp_path: Path
) -> None:
    """R5's backtest script must not print a 'vs EW Δ CAGR' for a failed signal."""
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    try:
        from run_allocator import _read_gate
    finally:
        sys.path.pop(0)

    failed = _gate_dir(tmp_path, r4_artefacts["dir"], "FAIL")
    verdict, detail = _read_gate(failed)
    assert verdict == "FAIL"
    assert detail

    passed = _gate_dir(tmp_path, r4_artefacts["dir"], "PASS")
    assert _read_gate(passed)[0] == "PASS"

    # A missing gate.json is NOT a pass.
    bare = tmp_path / "bare"
    bare.mkdir()
    assert _read_gate(bare)[0] != "PASS"


# ── seam (e): R8's ext features are genuinely opt-in ────────────────────────


def test_seam_e_ext_features_are_disjoint_from_the_base_feature_set() -> None:
    """`FEATURE_COLS` is what R4 trains on; the ext group must not enter it."""
    from trader.data.features_ext import EXT_FEATURE_COLS, assert_disjoint_from_base

    assert_disjoint_from_base()
    assert set(EXT_FEATURE_COLS).isdisjoint(set(FEATURE_COLS))
    # R4's own artefact records the columns it used; nothing from R8 is there.
    assert len(FEATURE_COLS) == 15, (
        f"FEATURE_COLS changed to {len(FEATURE_COLS)} columns; the ext group is "
        f"supposed to be additive and opt-in"
    )


def test_seam_e_r4_is_bit_identical_with_the_ext_group_present_but_disabled(
    r4_artefacts: dict[str, Any], tmp_path: Path
) -> None:
    """Adding ext columns to the panel must not change R4's output at all.

    "Opt-in" has to mean the disabled path is untouched, not merely that a flag
    exists. So: compute the ext columns onto the panel, run the SAME R4
    walk-forward over it with `feature_cols=FEATURE_COLS`, and require the
    predictions to match the run that never saw those columns.
    """
    panel = r4_artefacts["panel"]
    tickers = r4_artefacts["tickers"]

    # Extra columns the ext group would add, with real variance so they cannot
    # be ignored by accident.
    rng = np.random.default_rng(7)
    widened = panel.with_columns(
        [
            pl.Series(name, rng.normal(size=panel.height))
            for name in ("delivery_pct_20d", "delivery_pct_z_60d", "fii_net_5d")
        ]
    )
    out = tmp_path / "signal_widened"
    _run_r4(out, widened, tickers)

    base = pl.read_parquet(r4_artefacts["dir"] / "predictions.parquet").sort(
        ["date", "ticker"]
    )
    widened_preds = pl.read_parquet(out / "predictions.parquet").sort(["date", "ticker"])
    assert widened_preds.schema == base.schema
    assert widened_preds.select(["date", "ticker"]).equals(
        base.select(["date", "ticker"])
    )
    for col in ("r_hat_5d", "r_hat_20d"):
        np.testing.assert_array_equal(
            widened_preds[col].to_numpy(), base[col].to_numpy(),
            err_msg=f"{col} changed when unused ext columns were added to the panel",
        )

    # And the recorded feature list is still exactly the base set.
    index = json.loads((out / "index.json").read_text())
    assert index["feature_cols"] == list(FEATURE_COLS)


def test_seam_e_the_ext_group_widens_the_purge_only_when_asked(
    r4_artefacts: dict[str, Any],
) -> None:
    """The purge guard must be opt-in too, and must actually widen when opted in."""
    from trader.data.features_ext import combined_max_lookback_days
    from trader.training.walk_forward import assert_purge_clears_feature_lookback

    base = combined_max_lookback_days(include_ext=False)
    with_ext = combined_max_lookback_days(include_ext=True)
    assert with_ext >= base

    # At the shipped purge_months=3 (~63 trading days) both clear, so enabling
    # the group moves no window today -- the guard is for the NEXT feature.
    assert_purge_clears_feature_lookback(3, include_ext=False)
    assert_purge_clears_feature_lookback(3, include_ext=True)


# ── the dead-config trap: vol_lookback must be load-bearing ─────────────────


def test_vol_lookback_names_the_column_and_cannot_silently_disagree(
    r4_artefacts: dict[str, Any],
) -> None:
    """`vol_lookback` was declared in four layers and read by none of them.

    A second key, `vol_column`, decided the actual column, so the number the
    allocator was told it sized on and the column it was handed were free to
    disagree -- CLAUDE.md's `min_trade_value: 500` trap. The column is now
    derived from the lookback, so there is only one knob.
    """
    from trader.env.allocator_env import SignalPanel, vol_column_for

    assert vol_column_for(20) == "realized_vol_20d"
    assert vol_column_for(60) == "realized_vol_60d"
    with pytest.raises(ValueError):
        vol_column_for(0)

    d = r4_artefacts["dir"]
    tickers = r4_artefacts["tickers"]
    # A different lookback must read a DIFFERENT column, not silently the same
    # one. Both exist in the fixture panel and carry different values.
    a = SignalPanel.from_artifacts(
        d, r4_artefacts["oos_panel_path"], tickers, horizon="r_hat_20d", vol_lookback=20
    )
    b = SignalPanel.from_artifacts(
        d, r4_artefacts["oos_panel_path"], tickers, horizon="r_hat_20d", vol_lookback=60
    )
    finite = np.isfinite(a.vol) & np.isfinite(b.vol)
    assert finite.any()
    assert not np.allclose(a.vol[finite], b.vol[finite]), (
        "vol_lookback=20 and vol_lookback=60 read the same values: the key is "
        "not actually selecting the column"
    )

    # And the config must not reintroduce a second, independent key.
    cfg_text = (REPO_ROOT / "configs" / "env" / "allocator.yaml").read_text()
    assert "vol_lookback:" in cfg_text
    assert "\nvol_column:" not in cfg_text, (
        "configs/env/allocator.yaml declares vol_column again; two keys for one "
        "thing is the dead-config trap this test exists to prevent"
    )


# ── seam (f): P3 x P4 x P5 compose — band 0, restricted universe, supported K ─


def _allocator_pass(
    panel_path: Path,
    universe: list[str],
    r_hat: np.ndarray,
    vol_path: Path,
    dates: list[Any],
    params: Any,
    *,
    capital: float,
) -> tuple[list[np.ndarray], list[float], list[int]]:
    """Drive `PanelTradingEnv` from `allocate` and record what came out.

    Returns the target weight vector of every rebalance day, the NAV path, and
    the per-day count of distinct scrips sold — the quantity the flat demat fee
    is billed on, read from the env rather than estimated from weights.
    """
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    try:
        from run_allocator import _dense_column
    finally:
        sys.path.pop(0)
    from trader.allocator import allocate
    from trader.env.panel_env import PanelTradingEnv

    vol = _dense_column(vol_path, "realized_vol_20d", dates, universe)
    env = PanelTradingEnv(
        panel_path=panel_path,
        universe=universe,
        feature_columns=["log_return_1d"],
        lookback=_LOOKBACK,
        episode_length=40,
        initial_cash=capital,
        rebalance_schedule=RebalanceSchedule(freq="monthly"),  # type: ignore[arg-type]
        seed=0,
    )
    obs, _ = env.reset(seed=0)
    targets: list[np.ndarray] = []
    navs = [float(obs["nav"])]
    sold: list[int] = []
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
            targets.append(target.copy())
            obs, _, term, trunc, info = env.step_weights(target)
        else:
            obs, _, term, trunc, info = env.step(
                np.zeros(len(universe) + 1, dtype=np.float32)
            )
        navs.append(float(info["nav"]))
        sold.append(int(info["n_scrips_sold"]))
        done = bool(term or trunc)
    return targets, navs, sold


def test_seam_f_band_zero_supported_k_and_a_restricted_universe_compose(
    r4_artefacts: dict[str, Any], tmp_path: Path
) -> None:
    """P3, P4 and P5 in one run, and each of the three is load-bearing here.

    * **P4** restricts the universe by dropping a ticker's rows from the split —
      there is no enforcement code path, the name simply becomes an all-zero
      column with ``mask == False`` (`scripts/survivorship_arm.py`).  If that is
      all it does, a restriction that keeps every name the allocator would have
      picked must reproduce the unrestricted run **exactly**: same weights, same
      NAV, same scrip-sell-day count.  A restriction that also perturbed
      normalisation, ordering or the cash leg would show up here as a diff.
    * **P3** contributes ``no_trade_band=0.0``, which must take the pre-band code
      path and change nothing.  The band is shown to be live by flipping it on
      at the end: the same run then produces a *different* weight path, so the
      equality above is not "the band never does anything".
    * **P5** contributes the bound: ``k`` is asserted inside
      ``max_supportable_k`` at this capital, so the run being compared is one
      the account can actually hold — a K above it makes the allocator's targets
      unreachable and the comparison meaningless.

    Negative control at the end: dropping a name the allocator DID pick changes
    the weights.  Without it this test would pass with the restriction doing
    nothing at all.
    """
    from trader.allocator import AllocatorParams, max_supportable_k
    from trader.env.costs import DEFAULT_MIN_TRADE_VALUE

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    try:
        from run_allocator import _dense_signal
    finally:
        sys.path.pop(0)

    tickers = r4_artefacts["tickers"]
    full_path = r4_artefacts["oos_panel_path"]
    full = pl.read_parquet(full_path)
    dates = sorted(full["date"].unique().to_list())
    r_hat = _dense_signal(
        r4_artefacts["dir"] / "predictions.parquet", dates, tickers, ("20d",)
    )["20d"]

    capital = 1_000_000.0
    k = 5
    supported = max_supportable_k(
        capital, min_trade_value=DEFAULT_MIN_TRADE_VALUE, smallest_position_share=1.0
    )
    assert k <= supported, (
        f"k={k} is above max_supportable_k={supported} at ₹{capital:,.0f}: this "
        "test would be comparing two runs whose targets are unreachable"
    )
    params = AllocatorParams(
        k=k, max_name_weight=0.30, max_sector_weight=1.0, turnover_budget=0.30
    )
    assert params.no_trade_band == 0.0

    w_full, nav_full, sold_full = _allocator_pass(
        full_path, tickers, r_hat, full_path, dates, params, capital=capital
    )
    assert w_full, "the unrestricted arm never reached a rebalance day"
    picked = sorted(
        {tickers[i] for w in w_full for i in np.flatnonzero(w[1:] > 0.0)}
    )
    assert picked, "the unrestricted arm never bought anything"
    assert len(picked) < len(tickers), (
        "every ticker was picked, so no restriction can be a superset — the "
        "fixture is too small for this test to mean anything"
    )

    # Restriction that keeps every picked name: must reproduce exactly.
    kept_path = tmp_path / "oos_restricted_superset.parquet"
    full.filter(pl.col("ticker").is_in(picked)).write_parquet(kept_path)
    assert (
        sorted(pl.read_parquet(kept_path)["date"].unique().to_list()) == dates
    ), "the restriction changed the calendar, not just the cross-section"

    w_kept, nav_kept, sold_kept = _allocator_pass(
        kept_path, tickers, r_hat, kept_path, dates, params, capital=capital
    )
    assert len(w_kept) == len(w_full)
    for i, (a, b) in enumerate(zip(w_full, w_kept, strict=True)):
        np.testing.assert_array_equal(a, b, err_msg=f"rebalance {i} diverged")
    assert nav_kept == nav_full
    assert sold_kept == sold_full

    # Negative control 1: drop a name that WAS picked and the path must move.
    dropped_path = tmp_path / "oos_restricted_strict.parquet"
    full.filter(pl.col("ticker").is_in(picked[1:])).write_parquet(dropped_path)
    w_drop, nav_drop, _ = _allocator_pass(
        dropped_path, tickers, r_hat, dropped_path, dates, params, capital=capital
    )
    assert any(
        not np.array_equal(a, b) for a, b in zip(w_full, w_drop, strict=True)
    ), "dropping a selected name changed nothing — the restriction is not binding"

    # Negative control 2: the band is live, so `no_trade_band=0.0` above is a
    # real statement about the code path and not about an inert parameter.
    banded = AllocatorParams(
        k=k, max_name_weight=0.30, max_sector_weight=1.0, turnover_budget=0.30,
        no_trade_band=0.05,
    )
    w_band, _, sold_band = _allocator_pass(
        full_path, tickers, r_hat, full_path, dates, banded, capital=capital
    )
    assert any(
        not np.array_equal(a, b) for a, b in zip(w_full, w_band, strict=True)
    ), "no_trade_band=0.05 produced the same weights as 0.0 — the band is inert"
    assert sum(sold_band) <= sum(sold_full), (
        "the band increased the number of scrip-sell-days, which is the one "
        "quantity it exists to reduce"
    )
