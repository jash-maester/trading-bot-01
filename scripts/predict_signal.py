#!/usr/bin/env python
"""Run a frozen R4 signal model over a panel it has never seen.

`train_signal.py` writes predictions only for the walk-forward windows it fits,
so `data/signal/<tag>/predictions.parquet` stops at the last window's test end
(2024-06-28 for r4_v2). Trading a later span — the 2025+ holdout — needs the
saved encoder run forward over that panel. That is what this does.

    uv run python scripts/predict_signal.py \\
        --signal-dir data/signal/r4_v2 \\
        --panel data/panels_kite/test.parquet \\
        --out-tag r4_v2_holdout

It is deliberately inference ONLY: nothing is fitted, no normalisation
statistic is recomputed, and the feature means and standard deviations ride
inside the saved `state_dict` as buffers, so the panel cannot leak into them.
The output directory carries the same pinned artefact contract as a training
run, minus the gate — a gate belongs to the fit that produced the model, and
copying it here would let a holdout run masquerade as a validated one.

**The model is as stale as its last fit.** r4_v2's final window trained through
2021-12, so predicting 2025 asks it to extrapolate three years past anything it
saw. That is a real and reportable weakness of a frozen-model holdout, not a
bug; a deployment would refit first. `staleness_years` is written into
`index.json` so a reader cannot miss it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import torch
from loguru import logger


def _as_date(v: object) -> date:
    return v if isinstance(v, date) else date.fromisoformat(str(v))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--signal-dir", type=Path, required=True, help="data/signal/<tag>")
    ap.add_argument("--panel", type=Path, required=True)
    ap.add_argument("--out-tag", required=True, help="written to data/signal/<out-tag>")
    ap.add_argument("--out-root", type=Path, default=Path("data/signal"))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch-days", type=int, default=24)
    args = ap.parse_args()

    from trader.data.features import FEATURE_COLS
    from trader.data.universe import active_tickers
    from trader.models.signal import SignalConfig, SignalModel
    from trader.training.supervised import (
        build_panel_tensors,
        predict_panel,
        predictions_frame,
    )
    from trader.utils.seeding import get_device

    summary = json.loads((args.signal_dir / "summary.json").read_text())
    index = json.loads((args.signal_dir / "index.json").read_text())
    mcfg = summary["model_cfg"]
    tcfg = summary["train_cfg"]
    horizons = tuple(int(h) for h in mcfg["horizons"])
    lookback = int(tcfg["lookback"])
    feature_cols = list(index.get("feature_cols") or FEATURE_COLS)

    # The ticker axis is the artefact contract; reusing the source run's order
    # keeps embeddings, predictions and the env universe aligned.
    tickers = list(index["tickers"])
    if tickers != active_tickers():
        logger.warning(
            f"{args.signal_dir}/index.json ticker order differs from "
            "active_tickers(); using the artefact's order, which is the contract."
        )

    device = get_device(args.device)
    # feat_mean/feat_std must be passed so the encoder REGISTERS its normaliser
    # buffers; their values here are placeholders that `load_state_dict`
    # overwrites with the ones saved from the fit. That is the point: the
    # normalisation comes from the training split, and this panel can never
    # enter it.
    n_feat = len(feature_cols)
    model = SignalModel(
        SignalConfig(
            in_features=n_feat,
            embed_dim=int(mcfg["embed_dim"]),
            num_channels=list(mcfg["num_channels"]),
            kernel_size=int(mcfg["kernel_size"]),
            dropout=float(mcfg["dropout"]),
            head_hidden=int(mcfg["head_hidden"]),
            horizons=horizons,
        ),
        feat_mean=torch.zeros(n_feat),
        feat_std=torch.ones(n_feat),
    )
    state = torch.load(args.signal_dir / "signal_model.pt", map_location="cpu")
    loaded = state["model_state"] if "model_state" in state else state
    model.load_state_dict(loaded, strict=True)
    logger.info(
        "normaliser buffers loaded from the checkpoint: "
        f"mean[0]={float(model.encoder.input_norm.mean[0]):+.4f} "
        f"std[0]={float(model.encoder.input_norm.std[0]):.4f}"
    )
    model.to(device).eval()

    sha = model.encoder_state_sha256()
    if sha != index.get("encoder_state_sha256"):
        raise SystemExit(
            f"encoder sha mismatch: loaded {sha[:16]}... but index.json records "
            f"{str(index.get('encoder_state_sha256'))[:16]}... — wrong checkpoint."
        )

    panel = pl.read_parquet(args.panel)
    # The input representation is part of the checkpoint's contract, not a
    # property of this panel. A model fitted on per-date rank scores that is
    # handed raw features here would still run, still emit finite numbers, and
    # be silently meaningless -- `dollar_volume_20` alone would arrive ~1e9
    # standard deviations from anything the encoder ever saw. Artefacts written
    # before this key existed carry no entry and get None, which is exactly the
    # behaviour they were fitted with.
    xs_normalise = tcfg.get("xs_normalise") or None
    logger.info(
        f"input representation: xs_normalise={xs_normalise!r} "
        f"(from {args.signal_dir.name}/summary.json train_cfg)"
    )
    tensors = build_panel_tensors(
        panel, tickers, feature_cols, horizons,
        min_cross_section=int(tcfg.get("min_cross_section", 10)),
        xs_normalise=xs_normalise,
    )
    logger.info(
        f"{args.panel.name}: {tensors.n_days} days "
        f"{tensors.dates[0]}..{tensors.dates[-1]}, {len(tickers)} tickers; "
        f"first {lookback - 1} days have no full lookback and are not predicted"
    )

    preds = predict_panel(
        model, tensors, lookback=lookback, device=device, batch_days=args.batch_days
    )
    frame = predictions_frame(preds, tensors, horizons)

    out_dir = args.out_root / args.out_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "predictions.parquet"
    frame.write_parquet(pred_path)

    # How far past its last training data this model is being asked to see.
    trained_to = _as_date(summary["windows"][-1]["window"]["train_end"])
    staleness = (tensors.dates[-1] - trained_to).days / 365.25
    (out_dir / "index.json").write_text(
        json.dumps(
            {
                "dates": [d.isoformat() for d in tensors.dates],
                "tickers": tickers,
                "embed_dim": int(mcfg["embed_dim"]),
                "feature_cols": feature_cols,
                "encoder_state_sha256": sha,
                "source_signal_dir": str(args.signal_dir),
                "source_trained_to": trained_to.isoformat(),
                "staleness_years": round(staleness, 2),
                "inference_only": True,
            },
            indent=2,
        )
        + "\n"
    )
    digest = hashlib.sha256(pred_path.read_bytes()).hexdigest()
    logger.info(
        f"{pred_path}: {frame.height:,} rows, "
        f"{frame['date'].min()}..{frame['date'].max()}, SHA256={digest[:16]}..."
    )
    logger.info(
        f"model last trained to {trained_to}; this panel ends "
        f"{tensors.dates[-1]} — {staleness:.2f} years of extrapolation."
    )
    logger.info("No gate.json written: a gate belongs to the fit, not to an inference run.")
    for h in horizons:
        col = f"r_hat_{h}d"
        v = frame[col].to_numpy()
        logger.info(f"  {col}: mean {np.nanmean(v):+.4f} sd {np.nanstd(v):.4f}")


if __name__ == "__main__":
    main()
