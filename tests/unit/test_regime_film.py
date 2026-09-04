"""Unit tests for FiLM regime conditioning + ActorCritic integration.

Covers:

1. `RegimeNormalizer` — identity when no stats, z-score with stats.
2. `FiLM` module:
   - shape preservation
   - identity at initialisation (γ=1, β=0 → out=x)
   - different conditioning vectors produce different outputs
   - same conditioning vector produces same output (deterministic)
   - gradients flow into the conditioning MLP
3. `ActorCritic` with regime conditioning:
   - Param-count strictly increases when FiLM is enabled
   - Forward pass is identical to no-FiLM at step 0 (identity init)
   - With trained-style perturbed FiLM weights, outputs change with regime
   - Backward compatibility: regime_dim=0 model identical to legacy model
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


# ── RegimeNormalizer ───────────────────────────────────────────────────────────


def test_regime_normalizer_identity_without_stats() -> None:
    from trader.models.encoders import RegimeNormalizer

    norm = RegimeNormalizer()
    x = torch.randn(4, 6)
    out = norm(x)
    assert torch.allclose(out, x)


def test_regime_normalizer_zscore_with_stats() -> None:
    from trader.models.encoders import RegimeNormalizer

    mean = torch.tensor([1.0, 2.0, 3.0])
    std = torch.tensor([0.5, 1.0, 2.0])
    norm = RegimeNormalizer(mean, std)
    x = torch.tensor([[1.0, 2.0, 3.0], [2.0, 4.0, 9.0]])  # [2, 3]
    out = norm(x)
    expected = torch.tensor([[0.0, 0.0, 0.0], [2.0, 2.0, 3.0]])
    assert torch.allclose(out, expected, atol=1e-6)


def test_regime_normalizer_clamps_zero_std() -> None:
    """A degenerate std=0 must be clamped, not raise / NaN."""
    from trader.models.encoders import RegimeNormalizer

    mean = torch.tensor([0.0, 0.0])
    std = torch.tensor([0.0, 1.0])
    norm = RegimeNormalizer(mean, std)
    out = norm(torch.tensor([[1.0, 1.0]]))
    assert torch.isfinite(out).all()


# ── FiLM module ────────────────────────────────────────────────────────────────


def test_film_output_shape() -> None:
    from trader.models.encoders import FiLM

    B, N, D, R = 3, 8, 32, 6
    film = FiLM(cond_dim=R, embed_dim=D, hidden=16)
    x = torch.randn(B, N, D)
    cond = torch.randn(B, R)
    out = film(x, cond)
    assert out.shape == (B, N, D)


def test_film_is_identity_at_init() -> None:
    """Zero-init of the last linear layer ⇒ γ=1, β=0 ⇒ out=x at step 0."""
    from trader.models.encoders import FiLM

    B, N, D, R = 2, 5, 16, 4
    film = FiLM(cond_dim=R, embed_dim=D, hidden=8)
    film.eval()
    x = torch.randn(B, N, D)
    cond = torch.randn(B, R)
    out = film(x, cond)
    assert torch.allclose(out, x, atol=1e-6), (
        "FiLM should be identity at init — got max diff "
        f"{(out - x).abs().max().item():.2e}"
    )


def test_film_responds_to_conditioning_after_training_step() -> None:
    """After we kick the FiLM MLP off zero, different conds produce different outputs."""
    from trader.models.encoders import FiLM

    B, N, D, R = 2, 5, 16, 4
    film = FiLM(cond_dim=R, embed_dim=D, hidden=8)
    # Manually perturb the last layer's weights (simulate a training step).
    last = film.net[-1]
    with torch.no_grad():
        last.weight.normal_(0.0, 0.5)
        last.bias.normal_(0.0, 0.1)

    film.eval()
    x = torch.randn(B, N, D)
    cond_a = torch.randn(B, R)
    cond_b = torch.randn(B, R)
    out_a = film(x, cond_a)
    out_b = film(x, cond_b)
    diff = (out_a - out_b).abs().max().item()
    assert diff > 1e-3, "FiLM not responding to conditioning vector"


def test_film_is_deterministic() -> None:
    from trader.models.encoders import FiLM

    film = FiLM(cond_dim=4, embed_dim=8, hidden=4)
    film.eval()
    x = torch.randn(1, 3, 8)
    cond = torch.randn(1, 4)
    out1 = film(x, cond)
    out2 = film(x, cond)
    assert torch.allclose(out1, out2)


def test_film_gradients_flow() -> None:
    """Backprop must produce non-zero grads on the FiLM conditioning MLP."""
    from trader.models.encoders import FiLM

    film = FiLM(cond_dim=4, embed_dim=8, hidden=4)
    x = torch.randn(2, 3, 8, requires_grad=False)
    cond = torch.randn(2, 4)
    out = film(x, cond)
    out.sum().backward()
    # Final-layer weights are zero-init but should still receive gradient.
    last = film.net[-1]
    assert last.weight.grad is not None
    assert last.weight.grad.abs().max().item() > 0.0


def test_film_handles_unbatched_cond() -> None:
    """A 1-D cond vector should be auto-expanded to [1, R]."""
    from trader.models.encoders import FiLM

    film = FiLM(cond_dim=4, embed_dim=8, hidden=4)
    x = torch.randn(1, 3, 8)
    cond_1d = torch.randn(4)
    out = film(x, cond_1d)
    assert out.shape == (1, 3, 8)


# ── ActorCritic integration ────────────────────────────────────────────────────


def _make_obs_with_regime(
    B: int = 2, N: int = 5, L: int = 60, F: int = 15, R: int = 6
) -> dict[str, torch.Tensor]:
    return {
        "features": torch.randn(B, L, N, F),
        "mask": torch.ones(B, N, dtype=torch.int8),
        "sector_ids": torch.zeros(B, N, dtype=torch.int32),
        "portfolio": torch.softmax(torch.randn(B, N + 1), dim=-1),
        "cash": torch.full((B,), 500_000.0),
        "nav": torch.full((B,), 1_000_000.0),
        "t_frac": torch.full((B,), 0.5),
        "recent_return_1d": torch.zeros(B),
        "recent_vol_20d": torch.zeros(B),
        "nav_log_progress": torch.zeros(B),
        "regime": torch.randn(B, R),
    }


def test_actor_critic_with_regime_film_forward() -> None:
    from trader.models.actor_critic import ActorCritic, ModelConfig

    N, F, R = 5, 15, 6
    cfg = ModelConfig(
        in_features=F, n_tickers=N, embed_dim=32,
        regime_dim=R,
        regime_film_encoder=True,
        regime_film_attn=True,
        regime_in_critic=True,
    )
    model = ActorCritic(cfg)
    obs = _make_obs_with_regime(B=3, N=N, F=F, R=R)
    action_mean, value, log_std = model(obs)
    assert action_mean.shape == (3, N + 1)
    assert value.shape == (3,)
    assert log_std.shape == (N + 1,)
    assert torch.isfinite(action_mean).all()
    assert torch.isfinite(value).all()


def test_film_enabled_increases_param_count() -> None:
    from trader.models.actor_critic import ActorCritic, ModelConfig

    base = ActorCritic(
        ModelConfig(in_features=15, n_tickers=10, regime_dim=6,
                    regime_film_encoder=False, regime_film_attn=False,
                    regime_in_critic=False)
    )
    with_film = ActorCritic(
        ModelConfig(in_features=15, n_tickers=10, regime_dim=6,
                    regime_film_encoder=True, regime_film_attn=True,
                    regime_in_critic=True)
    )
    base_p = sum(p.numel() for p in base.parameters())
    film_p = sum(p.numel() for p in with_film.parameters())
    assert film_p > base_p, (
        f"FiLM should add params: base={base_p}, film={film_p}"
    )


def test_film_identity_init_ignores_regime() -> None:
    """At identity init the FiLM blocks reduce to ``out = x`` regardless of
    the conditioning vector — so two different regime vectors must produce
    *bit-identical* actor and critic outputs.  This is the cleanest test of
    the identity-init property; comparing two separately-constructed models
    is unreliable because FiLM creation consumes RNG state.
    """
    from trader.models.actor_critic import ActorCritic, ModelConfig

    N, F, R = 4, 15, 6
    cfg = ModelConfig(
        in_features=F, n_tickers=N, embed_dim=32, regime_dim=R,
        regime_film_encoder=True, regime_film_attn=True,
        regime_in_critic=False,    # exclude critic concat from this property test
    )
    model = ActorCritic(cfg)
    model.eval()

    obs_a = _make_obs_with_regime(B=2, N=N, F=F, R=R)
    obs_b = {k: v.clone() for k, v in obs_a.items()}
    obs_b["regime"] = torch.randn(2, R) * 5.0   # wildly different regime

    a_a, v_a, _ = model(obs_a)
    a_b, v_b, _ = model(obs_b)
    assert torch.allclose(a_a, a_b, atol=1e-6), (
        "Identity-init FiLM leaked regime into actor: max diff "
        f"{(a_a - a_b).abs().max().item():.2e}"
    )
    assert torch.allclose(v_a, v_b, atol=1e-6), (
        "Identity-init FiLM leaked regime into critic: max diff "
        f"{(v_a - v_b).abs().max().item():.2e}"
    )


def test_actor_critic_responds_to_regime_after_perturbation() -> None:
    """Once FiLM weights are non-zero, swapping the regime vector must change
    both the actor and the critic outputs."""
    from trader.models.actor_critic import ActorCritic, ModelConfig

    N, F, R = 4, 15, 6
    cfg = ModelConfig(
        in_features=F, n_tickers=N, embed_dim=32, regime_dim=R,
        regime_film_encoder=True, regime_film_attn=True, regime_in_critic=True,
    )
    model = ActorCritic(cfg)
    # Kick FiLM out of identity init
    for film in (model.film_encoder, model.film_attn):
        if film is None:
            continue
        last = film.net[-1]
        with torch.no_grad():
            last.weight.normal_(0.0, 0.3)
            last.bias.normal_(0.0, 0.1)
    model.eval()

    obs_a = _make_obs_with_regime(B=2, N=N, F=F, R=R)
    obs_b = {k: v.clone() for k, v in obs_a.items()}
    obs_b["regime"] = torch.randn(2, R) * 2.0

    a_a, v_a, _ = model(obs_a)
    a_b, v_b, _ = model(obs_b)
    assert (a_a - a_b).abs().max().item() > 1e-3, "Actor blind to regime"
    assert (v_a - v_b).abs().max().item() > 1e-3, "Critic blind to regime"


def test_actor_critic_regime_dim_zero_is_legacy_path() -> None:
    """With regime_dim=0 the model must construct cleanly without FiLM and
    not require a 'regime' key in obs."""
    from trader.models.actor_critic import ActorCritic, ModelConfig

    cfg = ModelConfig(
        in_features=15, n_tickers=5, embed_dim=32, regime_dim=0,
        regime_film_encoder=True, regime_film_attn=True, regime_in_critic=True,
    )
    model = ActorCritic(cfg)
    # When regime_dim=0, the FiLM blocks must be silently disabled.
    assert model.film_encoder is None
    assert model.film_attn is None

    # And the obs dict without "regime" must work.
    obs = {
        "features": torch.randn(2, 60, 5, 15),
        "mask": torch.ones(2, 5, dtype=torch.int8),
        "sector_ids": torch.zeros(2, 5, dtype=torch.int32),
        "portfolio": torch.softmax(torch.randn(2, 6), dim=-1),
        "cash": torch.full((2,), 500_000.0),
        "nav": torch.full((2,), 1_000_000.0),
        "t_frac": torch.full((2,), 0.5),
        "recent_return_1d": torch.zeros(2),
        "recent_vol_20d": torch.zeros(2),
        "nav_log_progress": torch.zeros(2),
    }
    a, v, _ = model(obs)
    assert a.shape == (2, 6)
    assert v.shape == (2,)


def test_critic_head_with_regime_dim_concat_shape() -> None:
    """CriticHead with regime_dim=R must accept and use the regime vector."""
    from trader.models.heads import CriticHead

    B, N, d, S, R = 3, 6, 16, 4, 5
    head = CriticHead(embed_dim=d, num_sectors=S, regime_dim=R)
    z = torch.randn(B, N, d)
    portfolio = torch.softmax(torch.randn(B, N + 1), dim=-1)
    t_frac = torch.rand(B)
    sector_ids = torch.randint(1, S + 1, (B, N), dtype=torch.long)
    z_b = torch.zeros(B)
    regime = torch.randn(B, R)
    v = head(z, portfolio, t_frac, sector_ids, z_b, z_b, z_b, regime=regime)
    assert v.shape == (B,)
    assert torch.isfinite(v).all()


def test_critic_head_regime_changes_value() -> None:
    """Different regime vectors → different value estimates (when regime_dim>0)."""
    from trader.models.heads import CriticHead

    B, N, d, S, R = 2, 5, 16, 4, 5
    head = CriticHead(embed_dim=d, num_sectors=S, regime_dim=R)
    head.eval()
    z = torch.randn(B, N, d)
    portfolio = torch.softmax(torch.randn(B, N + 1), dim=-1)
    t_frac = torch.rand(B)
    sector_ids = torch.randint(1, S + 1, (B, N), dtype=torch.long)
    z_b = torch.zeros(B)
    r_a = torch.randn(B, R)
    r_b = torch.randn(B, R) * 5.0
    v_a = head(z, portfolio, t_frac, sector_ids, z_b, z_b, z_b, regime=r_a)
    v_b = head(z, portfolio, t_frac, sector_ids, z_b, z_b, z_b, regime=r_b)
    assert (v_a - v_b).abs().max().item() > 1e-4
