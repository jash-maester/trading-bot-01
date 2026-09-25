"""The pruned TCN forward computes the same function as the full one.

The encoder returns only the last timestep, so `TCNEncoder._forward_pruned`
skips the ~69% of conv positions that cannot reach it. It must agree with the
full forward -- outputs AND gradients -- and load the same checkpoints.
"""
from __future__ import annotations

import torch

from trader.models.encoders import TCNEncoder, _live_positions
from trader.models.signal import SignalConfig, SignalModel


def _pair(dropout: float = 0.0) -> tuple[TCNEncoder, TCNEncoder]:
    torch.manual_seed(0)
    full = TCNEncoder(15, embed_dim=32, dropout=dropout)
    pruned = TCNEncoder(15, embed_dim=32, dropout=dropout)
    pruned.load_state_dict(full.state_dict())
    full.prune_to_last, pruned.prune_to_last = False, True
    return full, pruned


def test_outputs_match() -> None:
    full, pruned = _pair()
    x = torch.randn(3, 7, 60, 15)
    torch.testing.assert_close(pruned(x), full(x), rtol=1e-5, atol=1e-5)


def test_gradients_match() -> None:
    """Exact in float64 (~1e-15); in float32 the two differ only by summation
    order, ~1e-6 relative, so the float64 check is the meaningful one."""
    full, pruned = _pair()
    full, pruned = full.double(), pruned.double()
    x = torch.randn(2, 5, 60, 15, dtype=torch.float64)
    full(x).square().sum().backward()
    pruned(x).square().sum().backward()
    for (n, a), (_, b) in zip(full.named_parameters(), pruned.named_parameters(), strict=True):
        torch.testing.assert_close(b.grad, a.grad, rtol=1e-10, atol=1e-12, msg=n)


def test_other_lookbacks() -> None:
    full, pruned = _pair()
    for length in (30, 45, 90):
        x = torch.randn(2, 3, length, 15)
        torch.testing.assert_close(pruned(x), full(x), rtol=1e-5, atol=1e-5)


def test_state_dict_keys_unchanged() -> None:
    full, pruned = _pair()
    assert list(full.state_dict()) == list(pruned.state_dict())


def test_signal_model_end_to_end() -> None:
    torch.manual_seed(1)
    cfg = SignalConfig(in_features=15, embed_dim=32)
    m = SignalModel(cfg, feat_mean=torch.zeros(15), feat_std=torch.ones(15)).eval()
    x = torch.randn(2, 60, 9, 15)
    m.encoder.prune_to_last = True
    a = m(x)
    m.encoder.prune_to_last = False
    b = m(x)
    for k in a:
        torch.testing.assert_close(a[k], b[k], rtol=1e-5, atol=1e-5)


def test_dead_work_is_what_the_docstring_says() -> None:
    plan = _live_positions(60, 3, [1, 2, 4, 8])
    computed = sum(len(mid) + len(out) for _, mid, out in plan)
    assert computed == 148          # of 480 conv output positions
    assert plan[-1][2] == [59]
