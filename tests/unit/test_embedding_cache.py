"""R6 — the frozen-encoder embedding cache.

The cache is the 47× from ``audit/A2_compute.md``: the TCN is 97.9% of all the
arithmetic in an update and, once frozen, a pure function of the panel.  These
tests pin the two properties that make it safe to rely on — the cached
embedding equals what the encoder would have produced, and a cache written by a
*different* encoder is refused rather than silently used.
"""
from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import torch

from trader.models.encoders import TCNEncoder
from trader.training.embedding_cache import (
    EMBEDDINGS_FILE,
    INDEX_FILE,
    PREDICTIONS_FILE,
    EmbeddingCache,
    EncoderHashMismatch,
    encoder_state_sha256,
    write_cache,
)

FEATURES = ["f0", "f1", "f2"]
LOOKBACK = 8


def _panel(T: int = 40, N: int = 5, seed: int = 0) -> tuple[pl.DataFrame, list[str]]:
    rng = np.random.default_rng(seed)
    dates = [date(2021, 1, 4) + timedelta(days=i) for i in range(T)]
    tickers = [f"T{i}" for i in range(N)]
    rows: list[dict[str, object]] = []
    for d in dates:
        for t in tickers:
            row: dict[str, object] = {"date": d, "ticker": t}
            for c in FEATURES:
                row[c] = float(rng.normal())
            rows.append(row)
    return pl.DataFrame(rows), tickers


def _encoder(seed: int = 0) -> TCNEncoder:
    torch.manual_seed(seed)
    return TCNEncoder(in_features=len(FEATURES), embed_dim=6, num_channels=[6, 6], kernel_size=2)


# ── round trip ────────────────────────────────────────────────────────────────


def test_write_then_read_round_trip(tmp_path: Path) -> None:
    panel, tickers = _panel()
    enc = _encoder()
    out = write_cache(
        enc, panel, tickers, tmp_path / "cache", feature_columns=FEATURES, lookback=LOOKBACK
    )
    cache = EmbeddingCache(out, encoder=enc)

    assert len(cache) == 40 - LOOKBACK
    assert cache.tickers == tickers
    assert cache.embed_dim == 6
    assert cache.at(cache.dates[0]).shape == (len(tickers), 6)
    assert cache.as_array().shape == (40 - LOOKBACK, len(tickers), 6)


def test_cached_embedding_equals_a_live_forward_pass(tmp_path: Path) -> None:
    """The whole premise: reading the cache must equal re-running the encoder."""
    panel, tickers = _panel(seed=3)
    enc = _encoder(seed=3)
    out = write_cache(
        enc, panel, tickers, tmp_path / "c", feature_columns=FEATURES, lookback=LOOKBACK
    )
    cache = EmbeddingCache(out, encoder=enc)

    from trader.training.embedding_cache import stack_panel_features

    feats, _ = stack_panel_features(panel, tickers, FEATURES)
    i = 5
    window = feats[i : i + LOOKBACK]                                  # [L, N, F]
    x = torch.from_numpy(window).permute(1, 0, 2).unsqueeze(0)        # [1, N, L, F]
    enc.eval()
    with torch.no_grad():
        expected = enc(x)[0].numpy()
    got = cache.at(cache.dates[i]).astype(np.float32)
    # float16 storage: ~3 decimal digits.
    np.testing.assert_allclose(got, expected, rtol=2e-2, atol=2e-3)


def test_window_is_inclusive_at_both_ends(tmp_path: Path) -> None:
    panel, tickers = _panel()
    enc = _encoder()
    out = write_cache(
        enc, panel, tickers, tmp_path / "c", feature_columns=FEATURES, lookback=LOOKBACK
    )
    cache = EmbeddingCache(out, encoder=enc)
    lo, hi = cache.dates[2], cache.dates[9]
    assert cache.window(lo, hi).shape[0] == 8
    # A range entirely outside the cache is empty, not an error.
    assert cache.window(date(1990, 1, 1), date(1990, 2, 1)).shape[0] == 0


def test_index_json_uses_the_pinned_field_names(tmp_path: Path) -> None:
    """`index.json` is a cross-stream contract; R4 and R6 must agree on keys."""
    panel, tickers = _panel()
    enc = _encoder()
    out = write_cache(
        enc, panel, tickers, tmp_path / "c", feature_columns=FEATURES, lookback=LOOKBACK
    )
    meta = json.loads((out / INDEX_FILE).read_text())
    assert meta["tickers"] == tickers
    assert meta["feature_cols"] == FEATURES
    assert meta["embed_dim"] == 6
    assert meta["encoder_state_sha256"] == encoder_state_sha256(enc)
    # ISO dates, not datetimes.
    assert meta["dates"][0] == date.fromisoformat(meta["dates"][0]).isoformat()
    # Aliases stay in sync with the pinned names.
    assert meta["encoder_sha256"] == meta["encoder_state_sha256"]
    assert meta["feature_columns"] == meta["feature_cols"]


def test_hash_matches_the_signal_model_definition(tmp_path: Path) -> None:
    """One hash function in the repo, or every cross-stream cache looks stale."""
    from trader.models.signal import SignalConfig, SignalModel

    model = SignalModel(SignalConfig(in_features=len(FEATURES), embed_dim=6))
    assert encoder_state_sha256(model.encoder) == model.encoder_state_sha256()


# ── refusals ──────────────────────────────────────────────────────────────────


def test_a_different_encoder_is_refused(tmp_path: Path) -> None:
    panel, tickers = _panel()
    enc = _encoder(seed=0)
    out = write_cache(
        enc, panel, tickers, tmp_path / "c", feature_columns=FEATURES, lookback=LOOKBACK
    )
    other = _encoder(seed=1)
    with pytest.raises(EncoderHashMismatch):
        EmbeddingCache(out, encoder=other)


def test_a_single_perturbed_weight_is_refused(tmp_path: Path) -> None:
    """Shape alone cannot tell a stale cache from a live one — the hash must."""
    panel, tickers = _panel()
    enc = _encoder()
    out = write_cache(
        enc, panel, tickers, tmp_path / "c", feature_columns=FEATURES, lookback=LOOKBACK
    )
    with torch.no_grad():
        next(iter(enc.parameters())).add_(1e-3)
    with pytest.raises(EncoderHashMismatch):
        EmbeddingCache(out, encoder=enc)


def test_index_without_a_hash_is_refused(tmp_path: Path) -> None:
    panel, tickers = _panel()
    enc = _encoder()
    out = write_cache(
        enc, panel, tickers, tmp_path / "c", feature_columns=FEATURES, lookback=LOOKBACK
    )
    meta = json.loads((out / INDEX_FILE).read_text())
    del meta["encoder_state_sha256"]
    del meta["encoder_sha256"]
    (out / INDEX_FILE).write_text(json.dumps(meta))
    with pytest.raises(EncoderHashMismatch):
        EmbeddingCache(out, encoder=enc)


def test_unchecked_open_is_possible_but_explicit(tmp_path: Path) -> None:
    """Inspection needs no encoder; training passes one.  Both must be reachable."""
    panel, tickers = _panel()
    enc = _encoder()
    out = write_cache(
        enc, panel, tickers, tmp_path / "c", feature_columns=FEATURES, lookback=LOOKBACK
    )
    cache = EmbeddingCache(out)
    assert cache.encoder_sha256 == encoder_state_sha256(enc)
    with pytest.raises(ValueError, match="not both"):
        EmbeddingCache(out, encoder=enc, expected_sha256="deadbeef")


def test_panel_shorter_than_lookback_raises(tmp_path: Path) -> None:
    panel, tickers = _panel(T=LOOKBACK)
    with pytest.raises(ValueError, match="lookback"):
        write_cache(
            _encoder(), panel, tickers, tmp_path / "c",
            feature_columns=FEATURES, lookback=LOOKBACK,
        )


# ── the encoder is left alone ─────────────────────────────────────────────────


def test_write_cache_freezes_nothing_and_restores_the_training_flag(tmp_path: Path) -> None:
    panel, tickers = _panel()
    enc = _encoder()
    enc.train()
    before = {k: v.clone() for k, v in enc.state_dict().items()}
    write_cache(
        enc, panel, tickers, tmp_path / "c", feature_columns=FEATURES, lookback=LOOKBACK
    )
    assert enc.training, "write_cache must restore the caller's train/eval mode"
    for k, v in enc.state_dict().items():
        assert torch.equal(v, before[k]), f"write_cache mutated {k}"


def test_dropout_is_off_while_caching(tmp_path: Path) -> None:
    """A cache written under dropout is a random draw, not a function of the panel."""
    panel, tickers = _panel(seed=7)
    torch.manual_seed(11)
    enc = TCNEncoder(
        in_features=len(FEATURES), embed_dim=6, num_channels=[6, 6],
        kernel_size=2, dropout=0.5,
    )
    enc.train()
    a = write_cache(
        enc, panel, tickers, tmp_path / "a", feature_columns=FEATURES, lookback=LOOKBACK
    )
    b = write_cache(
        enc, panel, tickers, tmp_path / "b", feature_columns=FEATURES, lookback=LOOKBACK
    )
    np.testing.assert_array_equal(
        np.load(a / EMBEDDINGS_FILE), np.load(b / EMBEDDINGS_FILE)
    )


# ── predictions.parquet ───────────────────────────────────────────────────────


def test_predictions_parquet_has_the_pinned_columns(tmp_path: Path) -> None:
    panel, tickers = _panel()
    enc = _encoder()

    def predict(z: torch.Tensor) -> torch.Tensor:
        return torch.stack([z.mean(-1), z.sum(-1)], dim=-1)      # [B, N, 2]

    out = write_cache(
        enc, panel, tickers, tmp_path / "c",
        feature_columns=FEATURES, lookback=LOOKBACK, predict_fn=predict,
    )
    frame = pl.read_parquet(out / PREDICTIONS_FILE)
    assert frame.columns == ["date", "ticker", "r_hat_5d", "r_hat_20d"]
    assert frame.height == (40 - LOOKBACK) * len(tickers)
    assert set(frame["ticker"].unique().to_list()) == set(tickers)
