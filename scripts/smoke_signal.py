#!/usr/bin/env python
"""Synthetic-data smoke run of the R4 signal walk-forward.

Proves the loop executes and writes the pinned artefact set.  It says NOTHING
about the model's skill: the panel is synthetic (`tests/fixtures/synthetic_panel.py`)
and the run is capped at a handful of optimiser steps, so the IC it reports is
an artefact of the fixture, not a measurement of anything.  Per CLAUDE.md rule
2, no output of this script may be called a winner, a baseline-beating result,
or validated.

It exists because the original smoke evidence lived in an ephemeral session
scratchpad, so the numbers reported from it cited nothing that survived.

    uv run python scripts/smoke_signal.py --out audit/smoke_signal

Runs on CPU by default and never touches the GPU.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tests.fixtures.synthetic_panel import make_signal_panel  # noqa: E402
from trader.data.features import FEATURE_COLS  # noqa: E402
from trader.models.signal import SignalConfig  # noqa: E402
from trader.training.supervised import SupervisedConfig, run_signal_walk_forward  # noqa: E402
from trader.training.walk_forward import WindowConfig  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "audit" / "smoke_signal")
    ap.add_argument("--device", default="cpu", help="cpu (default); do not use a GPU here")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    panel = make_signal_panel(n_dates=340, n_tickers=12, seed=args.seed, signal_strength=0.0)
    tickers = sorted(panel["ticker"].unique().to_list())
    dates = sorted(panel["date"].unique().to_list())

    windows = [
        WindowConfig(
            name="w1",
            train_start=dates[0], train_end=dates[149],
            val_start=dates[160], val_end=dates[199],
            test_start=dates[210], test_end=dates[269],
        )
    ]
    summary = run_signal_walk_forward(
        full_panel=panel,
        windows=windows,
        tickers=tickers,
        feature_cols=list(FEATURE_COLS),
        model_cfg=SignalConfig(
            in_features=len(FEATURE_COLS), embed_dim=8, num_channels=[8, 8],
            head_hidden=8, horizons=(5, 20),
        ),
        train_cfg=SupervisedConfig(
            lookback=20, horizons=(5, 20), batch_days=8, eval_batch_days=16,
            max_epochs=1, max_steps=3, n_boot=64, min_cross_section=5,
            seed=args.seed, device=args.device,
        ),
        out_dir=args.out,
        tag="smoke",
        mlflow_port=None,
    )
    gate = json.loads((args.out / "gate.json").read_text())
    print(json.dumps(gate, indent=2))
    print(f"\nartefacts: {sorted(p.name for p in args.out.glob('*'))}")
    print(f"verdict:   {gate['verdict']}  (synthetic fixture — not a result)")
    return 0 if summary is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
