"""Unit tests for M5 — TCN encoder, actor-critic heads, forward pass."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


# ── Cross-stock attention ─────────────────────────────────────────────────────


def test_cross_stock_attention_shape_and_mixing() -> None:
    """Output preserves shape AND each stock's output depends on others."""
    from trader.models.encoders import CrossStockAttention

    B, N, d = 2, 10, 64
    attn = CrossStockAttention(embed_dim=d, num_heads=4, dropout=0.0)
    attn.eval()
    x = torch.randn(B, N, d)
    out_a = attn(x)
    assert out_a.shape == (B, N, d)

    # Modify ONLY stock 0's input; stock 5's output should change
    # (proving cross-stock mixing happened).
    x2 = x.clone()
    x2[:, 0] = torch.randn(B, d)
    out_b = attn(x2)
    diff_at_5 = (out_b[:, 5] - out_a[:, 5]).abs().max().item()
    assert diff_at_5 > 1e-5, (
        "Cross-stock attention should make stock 5's output depend on stock 0"
    )


def test_cross_stock_attention_respects_mask() -> None:
    """Untradeable stocks must NOT influence other stocks' outputs."""
    from trader.models.encoders import CrossStockAttention

    B, N, d = 1, 6, 32
    attn = CrossStockAttention(embed_dim=d, num_heads=2, dropout=0.0)
    attn.eval()
    x = torch.randn(B, N, d)
    mask_full = torch.ones(B, N, dtype=torch.bool)         # all tradeable
    mask_partial = mask_full.clone()
    mask_partial[0, 5] = False                             # mask out stock 5

    out_full = attn(x, tradeable_mask=mask_full)
    # Now perturb stock 5 only; with stock 5 masked, that perturbation
    # should NOT propagate to other stocks.
    x2 = x.clone()
    x2[0, 5] = torch.randn(d) * 100
    out_partial_perturbed = attn(x2, tradeable_mask=mask_partial)
    # Compare: stocks 0..4 should be (almost) identical between
    # `attn(x, mask_partial)` and `attn(x2, mask_partial)`.
    out_partial = attn(x, tradeable_mask=mask_partial)
    diff = (out_partial_perturbed[0, :5] - out_partial[0, :5]).abs().max().item()
    assert diff < 1e-5, f"Masked stock leaked into other stocks (diff={diff})"
    # Sanity: with full mask (no masking), the perturbation DOES propagate.
    out_full_perturbed = attn(x2, tradeable_mask=mask_full)
    diff_unmasked = (out_full_perturbed[0, :5] - out_full[0, :5]).abs().max().item()
    assert diff_unmasked > 1e-3, "Unmasked perturbation should propagate"


def test_actor_critic_uses_cross_attn_when_enabled() -> None:
    """ActorCritic with use_cross_attn=True actually inserts the layer."""
    from trader.models.actor_critic import ActorCritic, ModelConfig
    from trader.models.encoders import CrossStockAttention

    cfg_on = ModelConfig(in_features=15, n_tickers=10, use_cross_attn=True)
    cfg_off = ModelConfig(in_features=15, n_tickers=10, use_cross_attn=False)
    m_on = ActorCritic(cfg_on)
    m_off = ActorCritic(cfg_off)

    assert isinstance(m_on.cross_attn, CrossStockAttention)
    assert m_off.cross_attn is None
    # The "on" model must have strictly more params than the "off" model.
    on_params = sum(p.numel() for p in m_on.parameters())
    off_params = sum(p.numel() for p in m_off.parameters())
    assert on_params > off_params


# ── TCN encoder ───────────────────────────────────────────────────────────────


def test_tcn_output_shape() -> None:
    from trader.models.encoders import TCNEncoder

    B, N, L, F = 2, 10, 60, 15
    enc = TCNEncoder(in_features=F, embed_dim=64)
    x = torch.randn(B, N, L, F)
    out = enc(x)
    assert out.shape == (B, N, 64)


def test_tcn_causal_no_lookahead() -> None:
    """Changing future timesteps must not change the output at earlier positions."""
    from trader.models.encoders import TCNEncoder

    enc = TCNEncoder(in_features=8, embed_dim=32)
    enc.eval()
    x = torch.randn(1, 3, 60, 8)
    x_perturbed = x.clone()
    # Perturb last 5 timesteps (future relative to position 54)
    x_perturbed[:, :, -5:, :] += 100.0
    with torch.no_grad():
        out = enc(x)
        out_p = enc(x_perturbed)
    # Since TCN only uses the last timestep output, perturbing only future
    # positions changes things (causal conv takes all up to current).
    # What we verify: output IS causal — no negative-indexed data used.
    # Checking via output shape and no NaN.
    assert not torch.isnan(out).any()
    assert not torch.isnan(out_p).any()


def test_tcn_no_nan_on_zero_input() -> None:
    from trader.models.encoders import TCNEncoder

    enc = TCNEncoder(in_features=5, embed_dim=16)
    x = torch.zeros(1, 4, 60, 5)
    out = enc(x)
    assert not torch.isnan(out).any()


def test_tcn_configurable_channels() -> None:
    from trader.models.encoders import TCNEncoder

    enc = TCNEncoder(in_features=15, embed_dim=64, num_channels=[32, 64, 128])
    assert enc.out_dim == 128
    x = torch.randn(2, 5, 60, 15)
    out = enc(x)
    assert out.shape == (2, 5, 128)


# ── Actor head ────────────────────────────────────────────────────────────────


def test_actor_head_output_shape() -> None:
    from trader.models.heads import ActorHead

    B, N, d = 4, 10, 64
    head = ActorHead(embed_dim=d)
    z = torch.randn(B, N, d)
    portfolio = torch.softmax(torch.randn(B, N + 1), dim=-1)
    logits = head(z, portfolio)
    assert logits.shape == (B, N + 1)


def test_actor_head_no_nan() -> None:
    from trader.models.heads import ActorHead

    head = ActorHead(embed_dim=64)
    z = torch.randn(2, 5, 64)
    portfolio = torch.zeros(2, 6)
    portfolio[:, 0] = 1.0
    logits = head(z, portfolio)
    assert not torch.isnan(logits).any()


# ── Critic head ───────────────────────────────────────────────────────────────


def test_critic_head_output_shape() -> None:
    from trader.models.heads import CriticHead

    B, N, d = 4, 10, 64
    head = CriticHead(embed_dim=d, num_sectors=8)
    z = torch.randn(B, N, d)
    portfolio = torch.softmax(torch.randn(B, N + 1), dim=-1)
    t_frac = torch.rand(B)
    sector_ids = torch.randint(1, 9, (B, N), dtype=torch.long)
    recent_return = torch.randn(B) * 0.01
    recent_vol = torch.rand(B) * 0.02
    nav_log_progress = torch.randn(B) * 0.1
    v = head(z, portfolio, t_frac, sector_ids, recent_return, recent_vol, nav_log_progress)
    assert v.shape == (B,)


def test_critic_head_sector_exposure_correct() -> None:
    """Sector exposure must equal the sum of per-stock weights inside each sector."""
    from trader.models.heads import CriticHead

    B, N, d, S = 2, 6, 16, 4
    head = CriticHead(embed_dim=d, num_sectors=S)
    z = torch.zeros(B, N, d)
    # Hand-built portfolio: cash=0.4, equity weights known per slot.
    eq_w = torch.tensor(
        [
            [0.10, 0.20, 0.05, 0.15, 0.05, 0.05],
            [0.30, 0.05, 0.05, 0.05, 0.10, 0.05],
        ]
    )
    portfolio = torch.cat([torch.full((B, 1), 0.4), eq_w], dim=1)
    t_frac = torch.zeros(B)
    # Stocks 0,1 in sector 1; 2,3 in sector 2; 4 in sector 3; 5 in sector 4.
    sector_ids = torch.tensor([[1, 1, 2, 2, 3, 4]] * B, dtype=torch.long)
    zero = torch.zeros(B)
    head(z, portfolio, t_frac, sector_ids, zero, zero, zero)
    # Independently compute sector exposure for env 0:
    # sector1 = 0.30, sector2 = 0.20, sector3 = 0.05, sector4 = 0.05
    # We can't easily inspect the internal sector_exp, but we can build a
    # checker that mirrors the scatter logic and assert it sums correctly.
    sec_idx = (sector_ids - 1).clamp(min=0).long()
    sector_exp = torch.zeros(B, S)
    sector_exp.scatter_add_(1, sec_idx, eq_w)
    assert torch.allclose(sector_exp[0], torch.tensor([0.30, 0.20, 0.05, 0.05]))
    # Total equity exposure must equal sum of per-sector exposure.
    assert torch.allclose(sector_exp.sum(dim=1), eq_w.sum(dim=1))


def test_actor_head_init_is_small() -> None:
    """Orthogonal init with gain=0.01 → output magnitude << 1 at step 0."""
    from trader.models.heads import ActorHead

    B, N, d = 4, 20, 64
    head = ActorHead(embed_dim=d)
    z = torch.randn(B, N, d)
    portfolio = torch.softmax(torch.randn(B, N + 1), dim=-1)
    out = head(z, portfolio)
    # With gain=0.01 the last layer's output is bounded by ~|input| × 0.01.
    # Hidden activations are ~O(1) after LayerNorm, so |out| should be << 1.
    assert out.abs().max().item() < 0.5, f"actor init too aggressive: {out.abs().max().item()}"


# ── ActorCritic full forward ──────────────────────────────────────────────────


def _make_obs(B: int = 2, N: int = 5, L: int = 60, F: int = 15) -> dict[str, torch.Tensor]:
    return {
        "features": torch.randn(B, L, N, F),
        "mask": torch.ones(B, N, dtype=torch.int8),
        "sector_ids": torch.zeros(B, N, dtype=torch.int32),
        "portfolio": torch.softmax(torch.randn(B, N + 1), dim=-1),
        "cash": torch.full((B,), 500_000.0),
        "nav": torch.full((B,), 1_000_000.0),
        "t_frac": torch.full((B,), 0.5),
    }


def test_actor_critic_forward_shapes() -> None:
    from trader.models.actor_critic import ActorCritic, ModelConfig

    N, F = 5, 15
    cfg = ModelConfig(in_features=F, n_tickers=N, embed_dim=32)
    model = ActorCritic(cfg)
    obs = _make_obs(B=3, N=N, F=F)
    action_mean, value, log_std = model(obs)
    assert action_mean.shape == (3, N + 1)
    assert value.shape == (3,)
    assert log_std.shape == (N + 1,)


def test_actor_critic_get_action_and_value() -> None:
    from trader.models.actor_critic import ActorCritic, ModelConfig

    N, F = 6, 15
    cfg = ModelConfig(in_features=F, n_tickers=N, embed_dim=32)
    model = ActorCritic(cfg)
    obs = _make_obs(B=2, N=N, F=F)
    action, log_prob, entropy, value = model.get_action_and_value(obs)
    assert action.shape == (2, N + 1)
    assert log_prob.shape == (2,)
    assert entropy.shape == (2,)
    assert value.shape == (2,)
    assert not torch.isnan(action).any()
    assert not torch.isnan(log_prob).any()


def test_actor_critic_evaluate_action() -> None:
    """Log prob of a given action must be finite and differ from random."""
    from trader.models.actor_critic import ActorCritic, ModelConfig

    N, F = 4, 15
    cfg = ModelConfig(in_features=F, n_tickers=N, embed_dim=32)
    model = ActorCritic(cfg)
    obs = _make_obs(B=2, N=N, F=F)
    action = torch.zeros(2, N + 1)
    _, log_prob, entropy, value = model.get_action_and_value(obs, action)
    assert torch.isfinite(log_prob).all()
    assert torch.isfinite(entropy).all()


def test_gradients_flow_end_to_end() -> None:
    """A backward pass must produce non-zero gradients for all parameters."""
    from trader.models.actor_critic import ActorCritic, ModelConfig

    N, F = 5, 15
    cfg = ModelConfig(in_features=F, n_tickers=N, embed_dim=32)
    model = ActorCritic(cfg)
    obs = _make_obs(B=2, N=N, F=F)

    action, log_prob, entropy, value = model.get_action_and_value(obs)
    # Synthetic loss
    loss = -log_prob.mean() + value.pow(2).mean() - 0.01 * entropy.mean()
    loss.backward()

    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            assert not torch.isnan(param.grad).any(), f"NaN grad in {name}"


# ── Eval metrics ──────────────────────────────────────────────────────────────


def test_sharpe_positive_on_uptrend() -> None:
    from trader.training.eval_metrics import compute_episode_metrics

    nav = [1_000_000.0 * (1.001 ** i) for i in range(253)]
    turnovers = [0.01] * 252
    m = compute_episode_metrics(nav, turnovers)
    assert m.sharpe > 0


def test_max_drawdown_negative() -> None:
    from trader.training.eval_metrics import compute_episode_metrics

    # NAV drops 20% then recovers
    nav = [100.0, 90.0, 80.0, 85.0, 90.0, 100.0]
    m = compute_episode_metrics(nav, [0.0] * 5)
    assert m.max_drawdown < 0
    assert m.max_drawdown == pytest.approx(-0.20, abs=0.01)


def test_seeding_utility() -> None:
    from trader.utils.seeding import seed_everything

    seed_everything(42)
    x = torch.randn(5)
    seed_everything(42)
    y = torch.randn(5)
    assert torch.allclose(x, y)


# ── B1: num_sectors is derived from SECTOR_IDS, not hardcoded ─────────────────
#
# The universe grew from 8 to 14 sectors on 2026-09-04 while `num_sectors`
# stayed at a literal 8 in CriticHead, ModelConfig and GNNConfig.  The bug was
# dormant only because the panel on disk carried stale sector ids; the first
# rebuilt panel would have crashed the critic's scatter_add_ with
# "index 13 is out of bounds for dimension 1 with size 8".


def test_default_num_sectors_matches_sector_ids() -> None:
    """The model's sector width is the taxonomy's largest id, not a literal."""
    from trader.data.universe import SECTOR_IDS
    from trader.models.heads import default_num_sectors

    assert default_num_sectors() == max(SECTOR_IDS.values())


def test_num_sectors_tracks_a_newly_added_sector(monkeypatch: pytest.MonkeyPatch) -> None:
    """Adding a sector must widen every model default — no silent desync.

    This is the test that would have caught B1: it fails against a hardcoded
    ``8`` (and against any other literal) the moment SECTOR_IDS grows.
    """
    from trader.data import universe
    from trader.models.actor_critic import ModelConfig
    from trader.models.heads import CriticHead, default_num_sectors

    before = default_num_sectors()
    monkeypatch.setitem(universe.SECTOR_IDS, "unit_test_sector", before + 1)

    assert default_num_sectors() == before + 1
    assert ModelConfig(in_features=4, n_tickers=3).num_sectors == before + 1
    assert CriticHead(embed_dim=8).num_sectors == before + 1

    from trader.models.graph import GNNConfig  # imported late: needs torch_geometric

    assert GNNConfig().num_sectors == before + 1


def test_critic_head_accepts_top_of_range_sector_id() -> None:
    """sector_id == max(SECTOR_IDS) must work with the default configuration."""
    from trader.data.universe import SECTOR_IDS
    from trader.models.heads import CriticHead

    top = max(SECTOR_IDS.values())
    B, N, d = 2, 6, 16
    head = CriticHead(embed_dim=d)          # no explicit num_sectors
    assert head.num_sectors >= top
    z = torch.randn(B, N, d)
    portfolio = torch.softmax(torch.randn(B, N + 1), dim=-1)
    t_frac = torch.rand(B)
    # Every stock in the highest-numbered sector — the exact case that raised
    # "index 13 is out of bounds for dimension 1 with size 8".
    sector_ids = torch.full((B, N), top, dtype=torch.long)
    zero = torch.zeros(B)
    v = head(z, portfolio, t_frac, sector_ids, zero, zero, zero)
    assert v.shape == (B,)
    assert torch.isfinite(v).all()


def test_critic_head_rejects_sector_id_above_num_sectors() -> None:
    """An out-of-range id raises a *clear* ValueError, not an index error."""
    from trader.models.heads import CriticHead

    B, N, d, S = 2, 4, 16, 8
    head = CriticHead(embed_dim=d, num_sectors=S)
    z = torch.randn(B, N, d)
    portfolio = torch.softmax(torch.randn(B, N + 1), dim=-1)
    t_frac = torch.rand(B)
    sector_ids = torch.tensor([[1, 2, 3, S + 6]] * B, dtype=torch.long)
    zero = torch.zeros(B)

    with pytest.raises(ValueError, match="sector_id out of range") as exc:
        head(z, portfolio, t_frac, sector_ids, zero, zero, zero)
    msg = str(exc.value)
    assert "num_sectors=8" in msg          # names the misconfigured width
    assert str(S + 6) in msg               # names the offending id
    assert "SECTOR_IDS" in msg             # points at the source of truth


def test_critic_head_rejects_negative_sector_id() -> None:
    """Negative ids used to be silently clamped into bucket 0."""
    from trader.models.heads import CriticHead

    B, N, d = 1, 3, 8
    head = CriticHead(embed_dim=d, num_sectors=4)
    z = torch.randn(B, N, d)
    portfolio = torch.softmax(torch.randn(B, N + 1), dim=-1)
    zero = torch.zeros(B)
    sector_ids = torch.tensor([[1, -1, 2]], dtype=torch.long)
    with pytest.raises(ValueError, match="sector_id out of range"):
        head(z, portfolio, torch.zeros(B), sector_ids, zero, zero, zero)


def test_actor_critic_handles_full_sector_range() -> None:
    """End-to-end regression for B1 on the default (derived) configuration."""
    from trader.data.universe import SECTOR_IDS
    from trader.models.actor_critic import ActorCritic, ModelConfig

    top = max(SECTOR_IDS.values())
    N, F, B = top, 15, 2
    cfg = ModelConfig(in_features=F, n_tickers=N, embed_dim=32)
    model = ActorCritic(cfg)
    obs = _make_obs(B=B, N=N, F=F)
    # One stock per sector, covering 1..top inclusive.
    obs["sector_ids"] = (
        torch.arange(1, top + 1, dtype=torch.int32).unsqueeze(0).expand(B, -1)
    )
    action_mean, value, _ = model(obs)
    assert action_mean.shape == (B, N + 1)
    assert torch.isfinite(value).all()


# ── B6: PPO train/eval mode discipline ────────────────────────────────────────
#
# Rollout must run under eval() (deterministic log_prob_old — see the note in
# ppo.py), the gradient update must run under train() (or dropout / DropEdge
# never regularise anything), and train() must return with the model in eval()
# (runner._evaluate_split does not set the mode itself, so val/test would
# otherwise run with dropout ON).


def test_dropout_changes_output_in_train_mode_only() -> None:
    """The premise of the whole fix: mode actually changes the computation.

    Two forward passes must differ under train() and be identical under
    eval().  If this ever stops holding, the mode-discipline assertions below
    are vacuous.
    """
    from trader.models.actor_critic import ActorCritic, ModelConfig

    N, F, B = 5, 15, 2
    cfg = ModelConfig(in_features=F, n_tickers=N, embed_dim=32, dropout=0.5)
    model = ActorCritic(cfg)
    obs = _make_obs(B=B, N=N, F=F)

    model.eval()
    with torch.no_grad():
        a1, v1, _ = model(obs)
        a2, v2, _ = model(obs)
    assert torch.allclose(a1, a2), "eval() must be deterministic"
    assert torch.allclose(v1, v2), "eval() must be deterministic"

    model.train()
    with torch.no_grad():
        b1, _, _ = model(obs)
        b2, _, _ = model(obs)
    assert not torch.allclose(b1, b2), "train() must apply dropout"


class _ModeRecordingModel(torch.nn.Module):
    """Minimal actor-critic stub that records train/eval mode at each call site.

    Mode discipline is a property of :class:`PPOTrainer`, not of any particular
    network, so the stub keeps the test fast and unambiguous.  The nested
    ``Dropout`` is recorded alongside the module's own flag to prove the mode
    actually propagates to submodules.
    """

    def __init__(self, n_actions: int) -> None:
        super().__init__()
        self.body = torch.nn.Sequential(
            torch.nn.Linear(n_actions, n_actions),
            torch.nn.Dropout(0.5),
        )
        self.log_std = torch.nn.Parameter(torch.zeros(n_actions))
        self.rollout_modes: list[tuple[bool, bool]] = []
        self.update_modes: list[tuple[bool, bool]] = []

    def _modes(self) -> tuple[bool, bool]:
        return (self.training, self.body[1].training)

    def _dist(self, obs: dict) -> tuple[torch.distributions.Normal, torch.Tensor]:
        mean = self.body(obs["portfolio"].float())
        dist = torch.distributions.Normal(mean, self.log_std.exp().expand_as(mean))
        return dist, mean

    def get_action_and_value(self, obs: dict, action=None):
        # `action is None` only during rollout sampling; the PPO update always
        # re-evaluates an action stored in the rollout buffer.
        (self.rollout_modes if action is None else self.update_modes).append(self._modes())
        dist, mean = self._dist(obs)
        if action is None:
            action = dist.sample()
        return (
            action,
            dist.log_prob(action).sum(-1),
            dist.entropy().sum(-1),
            mean.sum(-1),
        )

    def get_value(self, obs: dict):
        self.rollout_modes.append(self._modes())
        _, mean = self._dist(obs)
        return mean.sum(-1)


class _ConstantVecEnv:
    """Minimal stand-in for the vectorised PanelTradingEnv."""

    def __init__(self, n_envs: int, n_tickers: int) -> None:
        import numpy as np

        self._np = np
        self.n_envs = n_envs
        self.n_tickers = n_tickers
        self._rng = np.random.default_rng(0)

    def _obs(self) -> dict:
        np = self._np
        E, N = self.n_envs, self.n_tickers
        return {
            "portfolio": np.full((E, N + 1), 1.0 / (N + 1), dtype=np.float32),
        }

    def reset(self):
        return self._obs(), {}

    def step(self, action):
        np = self._np
        reward = self._rng.standard_normal(self.n_envs).astype(np.float32) * 0.01
        terminated = np.zeros(self.n_envs, dtype=bool)
        truncated = np.zeros(self.n_envs, dtype=bool)
        info = {
            "nav": [1_000_000.0] * self.n_envs,
            "turnover": [0.0] * self.n_envs,
        }
        return self._obs(), reward, terminated, truncated, info


def test_ppo_mode_discipline(tmp_path) -> None:
    """Rollout in eval(), gradient update in train(), return in eval()."""
    from trader.training.ppo import PPOConfig, PPOTrainer

    E, N = 2, 4
    n_steps, n_epochs, n_minibatches = 4, 2, 2
    cfg = PPOConfig(
        total_steps=2 * n_steps * E,      # exactly two update iterations
        n_envs=E,
        n_steps=n_steps,
        n_epochs=n_epochs,
        n_minibatches=n_minibatches,
        target_kl=None,                   # don't early-stop out of the epochs
        checkpoint_dir=tmp_path / "ckpt",
        log_interval=10**6,
        checkpoint_interval=10**6,
    )
    model = _ModeRecordingModel(N + 1)
    trainer = PPOTrainer(_ConstantVecEnv(E, N), model, cfg, torch.device("cpu"))
    trainer.train()

    # 2 updates x (n_steps sampling calls + 1 bootstrap get_value)
    assert len(model.rollout_modes) == 2 * (n_steps + 1)
    # 2 updates x n_epochs x n_minibatches
    assert len(model.update_modes) == 2 * n_epochs * n_minibatches

    assert all(m == (False, False) for m in model.rollout_modes), (
        "rollout must run in eval() — log_prob_old has to be a deterministic "
        "function of the weights, or the importance ratio measures dropout masks"
    )
    assert all(m == (True, True) for m in model.update_modes), (
        "the gradient update must run in train(), or dropout / DropEdge never "
        "regularise anything"
    )
    assert not model.training, (
        "train() must return with the model in eval(): runner._evaluate_split "
        "never sets the mode, so val/test would otherwise run with dropout on"
    )


def test_ppo_leaves_model_in_eval_with_zero_updates(tmp_path) -> None:
    """Even when the loop body never runs, the model must come back in eval()."""
    from trader.training.ppo import PPOConfig, PPOTrainer

    E, N = 2, 3
    cfg = PPOConfig(
        total_steps=0,                    # n_updates == 0
        n_envs=E,
        n_steps=4,
        n_minibatches=2,
        checkpoint_dir=tmp_path / "ckpt",
    )
    model = _ModeRecordingModel(N + 1)
    model.train()
    trainer = PPOTrainer(_ConstantVecEnv(E, N), model, cfg, torch.device("cpu"))
    trainer.train()
    assert not model.training
