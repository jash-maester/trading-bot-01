"""R4 — unit tests for :mod:`trader.models.signal`.

The point of this module is that it is *composition*, not new modelling: one
existing `TCNEncoder`, one existing `ReturnPredictionHead` per horizon, sharing
that encoder.  These tests pin exactly that, plus the two contracts other
streams read off it — `encode()` as the embedding source for R6, and
`encoder_state_sha256()` as the provenance stamp in `index.json`.
"""
from __future__ import annotations

import pytest
import torch

from trader.models.encoders import TCNEncoder
from trader.models.heads import ReturnPredictionHead
from trader.models.signal import (
    DEFAULT_HORIZONS,
    SignalConfig,
    SignalModel,
    horizon_key,
    state_dict_sha256,
)

F = 6
B, N, L = 2, 7, 12


def _model(**kw: object) -> SignalModel:
    cfg = SignalConfig(
        in_features=F,
        embed_dim=kw.pop("embed_dim", 16),  # type: ignore[arg-type]
        num_channels=kw.pop("num_channels", [16, 16]),  # type: ignore[arg-type]
        kernel_size=3,
        dropout=0.0,
        head_hidden=8,
        **kw,  # type: ignore[arg-type]
    )
    torch.manual_seed(0)
    return SignalModel(cfg)


def _features() -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randn(B, L, N, F)


# ── Column naming (a pinned contract: predictions.parquet columns) ────────────


def test_horizon_key_matches_pinned_contract() -> None:
    assert horizon_key(5) == "r_hat_5d"
    assert horizon_key(20) == "r_hat_20d"


def test_default_horizons() -> None:
    assert DEFAULT_HORIZONS == (5, 20)


# ── Config validation ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "horizons",
    [(), (0, 5), (-1,), (5, 5)],
    ids=["empty", "zero", "negative", "duplicate"],
)
def test_signal_config_rejects_bad_horizons(horizons: tuple[int, ...]) -> None:
    with pytest.raises(ValueError):
        SignalConfig(in_features=F, horizons=horizons)


# ── Composition: the existing encoder and the existing head ──────────────────


def test_model_composes_existing_encoder_and_one_head_per_horizon() -> None:
    m = _model(horizons=(5, 20))
    assert isinstance(m.encoder, TCNEncoder)
    assert set(m.heads.keys()) == {"r_hat_5d", "r_hat_20d"}
    for head in m.heads.values():
        assert isinstance(head, ReturnPredictionHead)


def test_heads_share_one_encoder() -> None:
    """One encoder, two heads — not two encoders.

    If each head had its own trunk the embedding cache R6 reads would be
    ambiguous (which encoder produced `embeddings.npy`?) and the whole point of
    the dense supervision — 504 labels pushing on *one* representation — would
    be halved.
    """
    m = _model(horizons=(5, 20))
    encoders = [mod for mod in m.modules() if isinstance(mod, TCNEncoder)]
    assert len(encoders) == 1
    assert encoders[0] is m.encoder


def test_head_parameters_are_distinct_per_horizon() -> None:
    m = _model(horizons=(5, 20))
    p5 = dict(m.heads["r_hat_5d"].named_parameters())
    p20 = dict(m.heads["r_hat_20d"].named_parameters())
    assert p5.keys() == p20.keys()
    assert all(p5[k] is not p20[k] for k in p5)


# ── encode(): the R6 embedding contract ──────────────────────────────────────


def test_encode_shape_and_dim() -> None:
    m = _model()
    z = m.encode(_features())
    assert z.shape == (B, N, m.embed_dim)
    assert m.embed_dim == 16


def test_encode_rejects_wrong_rank() -> None:
    m = _model()
    with pytest.raises(ValueError, match=r"\[B, L, N, F\]"):
        m.encode(torch.randn(B, L, N))


def test_encode_is_per_stock_only() -> None:
    """No cross-stock mixing: stock i's embedding must not move when stock j does.

    This is what keeps a cached embedding valid regardless of which subset of
    the universe happens to be tradeable that day.  `CrossStockAttention` is
    deliberately absent from this model; if someone adds it, this fails.
    """
    m = _model()
    m.eval()
    x = _features()
    y = x.clone()
    y[:, :, 1, :] += 5.0                       # perturb stock 1 only
    with torch.no_grad():
        z0, z1 = m.encode(x), m.encode(y)
    assert torch.equal(z0[:, 0], z1[:, 0])     # stock 0 untouched
    assert not torch.equal(z0[:, 1], z1[:, 1])  # stock 1 did move


def test_forward_equals_predict_from_embeddings() -> None:
    m = _model()
    m.eval()
    x = _features()
    with torch.no_grad():
        direct = m(x)
        staged = m.predict_from_embeddings(m.encode(x))
    assert direct.keys() == staged.keys()
    for k in direct:
        assert torch.equal(direct[k], staged[k])


def test_forward_shapes_and_keys() -> None:
    m = _model(horizons=(5, 20))
    out = m(_features())
    assert set(out) == {"r_hat_5d", "r_hat_20d"}
    for v in out.values():
        assert v.shape == (B, N)


def test_mask_zeroes_untradeable_predictions() -> None:
    """A consumer that forgets to mask must not read a real-looking number."""
    m = _model()
    m.eval()
    mask = torch.ones(B, N, dtype=torch.bool)
    mask[:, 2] = False
    with torch.no_grad():
        out = m(_features(), mask)
    for v in out.values():
        assert torch.all(v[:, 2] == 0.0)
        assert torch.any(v[:, 0] != 0.0)


def test_single_horizon_model() -> None:
    m = _model(horizons=(1,))
    assert set(m(_features())) == {"r_hat_1d"}


# ── Provenance ───────────────────────────────────────────────────────────────


def test_encoder_sha256_is_stable_across_calls() -> None:
    m = _model()
    assert m.encoder_state_sha256() == m.encoder_state_sha256()


def test_encoder_sha256_changes_with_encoder_weights() -> None:
    m = _model()
    before = m.encoder_state_sha256()
    with torch.no_grad():
        next(iter(m.encoder.parameters())).add_(1.0)
    assert m.encoder_state_sha256() != before


def test_encoder_sha256_ignores_head_weights() -> None:
    """The stamp describes what produced `embeddings.npy` — the encoder alone."""
    m = _model()
    before = m.encoder_state_sha256()
    with torch.no_grad():
        next(iter(m.heads["r_hat_5d"].parameters())).add_(1.0)
    assert m.encoder_state_sha256() == before
    assert state_dict_sha256(m.state_dict()) != state_dict_sha256(
        {k: v for k, v in m.state_dict().items() if k.startswith("encoder.")}
    )


def test_encoder_sha256_is_key_order_independent() -> None:
    m = _model()
    state = m.encoder.state_dict()
    reversed_state = {k: state[k] for k in reversed(list(state))}
    assert state_dict_sha256(state) == state_dict_sha256(reversed_state)


def test_normalisation_buffers_ride_in_the_state_dict() -> None:
    """Train-split stats must travel with the weights, or a reload is a leak.

    If the normaliser were re-derived at load time from whatever panel was to
    hand, an OOS reload would silently normalise with OOS statistics.
    """
    mean = torch.arange(F, dtype=torch.float32)
    std = torch.full((F,), 2.0)
    cfg = SignalConfig(in_features=F, embed_dim=16, num_channels=[16, 16], dropout=0.0)
    m = SignalModel(cfg, feat_mean=mean, feat_std=std)
    keys = list(m.encoder.state_dict())
    assert any("input_norm" in k for k in keys), keys
    m2 = SignalModel(cfg, feat_mean=torch.zeros(F), feat_std=torch.ones(F))
    assert m2.encoder_state_sha256() != m.encoder_state_sha256()
    m2.load_state_dict(m.state_dict())
    assert m2.encoder_state_sha256() == m.encoder_state_sha256()
