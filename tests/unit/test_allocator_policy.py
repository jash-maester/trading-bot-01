"""R6 — the Beta policy over the allocator's bounded scalars.

The test that matters most is at the bottom:
``test_turnover_from_sampling_is_far_below_the_old_policy``.  It measures the
one number that justifies this whole stream — how much of NAV a *sampled*
action moves relative to what the policy actually intends — against the 0.289
per day that ``10_architecture_revamp.md`` §1.1 measured for the retired
505-logit Gaussian-through-softmax policy.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch
from torch.distributions import Beta

from trader.allocator.deterministic import allocate
from trader.env.allocator_env import N_SECTORS, ActionRanges
from trader.models.allocator_policy import AllocatorPolicy, AllocatorPolicyConfig

# ── numbers this file is measured against, with their sources ────────────────
# 10_architecture_revamp.md §1.1, simulated with the real masked_softmax over
# 504 tickers: "sigma=0.368: sampled vs mean L1 dist mean=0.289", per DAY.
OLD_POLICY_L1_PER_DAY = 0.289
# §1.1 again: 252 trading days x 0.289 x the verified delivery round trip
# (Rs 237.82 per Rs 2L traded = 0.11891% of traded value, costs.py @ 9019892)
# = 8.7%/yr.  Reproduced in the test below as an arithmetic check.
ROUND_TRIP_COST_FRACTION = 237.82 / 200_000.0
TRADING_DAYS = 252
MONTHS = 12


def _obs_dim(ranges: ActionRanges) -> int:
    from trader.data.regime_features import REGIME_DIM

    return ranges.action_dim + 5 + ranges.n_sectors + REGIME_DIM


def _policy(**kw: float) -> tuple[AllocatorPolicy, ActionRanges]:
    ranges = ActionRanges()
    torch.manual_seed(0)
    cfg = AllocatorPolicyConfig(
        obs_dim=_obs_dim(ranges), action_dim=ranges.action_dim, **kw  # type: ignore[arg-type]
    )
    return AllocatorPolicy(cfg), ranges


# ── shape and support ─────────────────────────────────────────────────────────


def test_actions_are_inside_the_unit_box() -> None:
    policy, ranges = _policy()
    obs = torch.randn(64, policy.obs_dim)
    action, log_prob, entropy, value = policy.get_action_and_value(obs)
    assert action.shape == (64, ranges.action_dim)
    assert log_prob.shape == (64,)
    assert entropy.shape == (64,)
    assert value.shape == (64,)
    assert bool((action > 0.0).all()) and bool((action < 1.0).all())


def test_action_dim_is_three_plus_one_per_sector() -> None:
    _, ranges = _policy()
    assert ranges.action_dim == 3 + N_SECTORS == 3 + 14


def test_every_marginal_is_unimodal() -> None:
    """alpha, beta >= 1 — a U-shaped Beta on K would flip between 10 and 60."""
    policy, _ = _policy()
    with torch.no_grad():
        alpha, beta = policy.concentrations(torch.randn(128, policy.obs_dim) * 5.0)
    assert float(alpha.min()) >= policy.cfg.min_concentration
    assert float(beta.min()) >= policy.cfg.min_concentration


def test_initial_policy_is_symmetric_at_the_requested_concentration() -> None:
    policy, _ = _policy(init_concentration=25.0)
    with torch.no_grad():
        alpha, beta = policy.concentrations(torch.zeros(1, policy.obs_dim))
    assert float(alpha.mean()) == pytest.approx(25.0, rel=1e-3)
    assert float(beta.mean()) == pytest.approx(25.0, rel=1e-3)
    # Beta(c, c) has mean 0.5 and sd sqrt(1 / (4(2c+1))).
    sd = float(Beta(alpha, beta).stddev.mean())
    assert sd == pytest.approx((1.0 / (4.0 * (2 * 25.0 + 1))) ** 0.5, rel=1e-3)


def test_deterministic_action_is_the_distribution_mean() -> None:
    policy, _ = _policy()
    obs = torch.randn(8, policy.obs_dim)
    with torch.no_grad():
        action, _, _, _ = policy.get_action_and_value(obs, deterministic=True)
        alpha, beta = policy.concentrations(obs)
    torch.testing.assert_close(action, alpha / (alpha + beta), rtol=1e-5, atol=1e-6)


# ── entropy is in ACTION space ────────────────────────────────────────────────


def test_entropy_is_the_analytic_beta_entropy_of_the_actions() -> None:
    """Not a logit-space proxy, not a pre-squash Gaussian entropy."""
    policy, _ = _policy()
    obs = torch.randn(16, policy.obs_dim)
    with torch.no_grad():
        _, _, entropy, _ = policy.get_action_and_value(obs)
        alpha, beta = policy.concentrations(obs)
        expected = Beta(alpha, beta).entropy().sum(-1)  # type: ignore[no-untyped-call]
    torch.testing.assert_close(entropy, expected, rtol=1e-5, atol=1e-6)


def test_entropy_responds_to_concentration_not_just_to_dimension_count() -> None:
    """§1.1: the old bonus was exactly (N+1)*H(sigma) and said nothing about
    the portfolio — a book in one stock and one in 500 scored identically.
    Here a sharper policy must score strictly lower entropy at the same width."""
    broad, _ = _policy(init_concentration=2.0)
    sharp, _ = _policy(init_concentration=50.0)
    obs = torch.zeros(1, broad.obs_dim)
    with torch.no_grad():
        _, _, h_broad, _ = broad.get_action_and_value(obs)
        _, _, h_sharp, _ = sharp.get_action_and_value(obs)
    assert float(h_sharp) < float(h_broad)
    # And both are finite — the min_concentration >= 1 guard.
    assert np.isfinite(float(h_broad)) and np.isfinite(float(h_sharp))


def test_log_prob_is_finite_at_the_bounds() -> None:
    """Beta's density is +-inf at exactly 0 and 1; the clamp must absorb that."""
    policy, ranges = _policy()
    obs = torch.zeros(2, policy.obs_dim)
    edges = torch.zeros(2, ranges.action_dim)
    edges[1] = 1.0
    with torch.no_grad():
        _, log_prob, _, _ = policy.get_action_and_value(obs, edges)
    assert bool(torch.isfinite(log_prob).all())


def test_re_evaluating_a_sampled_action_reproduces_its_log_prob() -> None:
    """The PPO ratio is exp(new_lp - old_lp); at unchanged weights it must be 1."""
    policy, _ = _policy()
    policy.eval()
    obs = torch.randn(32, policy.obs_dim)
    with torch.no_grad():
        action, old_lp, _, _ = policy.get_action_and_value(obs)
        _, new_lp, _, _ = policy.get_action_and_value(obs, action)
    torch.testing.assert_close((new_lp - old_lp).exp(), torch.ones(32), rtol=1e-5, atol=1e-6)


def test_gradients_reach_both_heads() -> None:
    policy, _ = _policy()
    policy.train()
    obs = torch.randn(16, policy.obs_dim)
    _, log_prob, entropy, value = policy.get_action_and_value(obs)
    (log_prob.mean() + entropy.mean() + value.mean()).backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in policy.actor.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in policy.critic.parameters())


def test_config_rejects_a_bimodal_parameterisation() -> None:
    with pytest.raises(ValueError, match="unimodal"):
        AllocatorPolicyConfig(obs_dim=10, action_dim=3, min_concentration=0.5)


# ── THE test ──────────────────────────────────────────────────────────────────


def _synthetic_cross_section(n: int = 200, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        "r_hat": rng.normal(0.0, 1.0, n),          # cross-sectionally standardised
        "vol": np.exp(rng.normal(np.log(0.25), 0.3, n)),
        "mask": np.ones(n, dtype=bool),
        "sector_ids": 1 + rng.integers(0, N_SECTORS, n),
    }


def _weights_for(
    action: np.ndarray,
    ranges: ActionRanges,
    cross: dict[str, np.ndarray],
    current_w: np.ndarray,
    *,
    force_budget: float | None = None,
) -> tuple[np.ndarray, float, int]:
    params, tilt = ranges.decode(
        action, max_name_weight=0.10, max_sector_weight=0.25, vol_lookback=20
    )
    budget = params.turnover_budget
    if force_budget is not None:
        params = replace(params, turnover_budget=force_budget)
    padded = np.concatenate([[0.0], tilt])
    tilted = cross["r_hat"] * (1.0 + padded[cross["sector_ids"]])
    w = allocate(
        tilted, cross["vol"], cross["mask"], cross["sector_ids"], current_w, params
    )
    return w, budget, params.k


def measure_turnover_from_sampling(
    n_samples: int = 512, n_names: int = 200, seed: int = 0, *, flip: bool = False
) -> dict[str, float]:
    """L1 distance between the SAMPLED portfolio and the policy's INTENDED one.

    This is the same quantity §1.1 measured for the old policy ("sampled vs
    mean L1 dist"), computed the same way: hold the state fixed, take the
    distribution mean as the intent, and ask how far a draw moves the book away
    from it.  The incumbent book is set to the intended portfolio, so every unit
    of L1 is turnover the *sampling* caused and nothing else.

    Two variants are reported:

    ``deployed``
        the sampled action in full, including the sampled ``turnover_budget``
        — what the env actually executes.
    ``unclamped``
        the same action with the budget disabled, isolating how far the
        allocator's *target* moves.  The gap between the two is what the budget
        is buying.
    """
    torch.manual_seed(seed)
    policy, ranges = _policy()
    policy.eval()
    cross = _synthetic_cross_section(n_names, seed)
    obs = torch.zeros(1, policy.obs_dim)

    with torch.no_grad():
        mean_action, _, _, _ = policy.get_action_and_value(obs, deterministic=True)
        samples, _, _, _ = policy.get_action_and_value(obs.expand(n_samples, -1))

    all_cash = np.zeros(n_names + 1)
    all_cash[0] = 1.0
    intended, _, k_mean = _weights_for(
        mean_action.numpy()[0], ranges, cross, all_cash, force_budget=2.0
    )
    if flip:
        # Start from a book built on the INVERTED signal, so every sampled
        # action wants to rotate most of the portfolio and the sampled budget
        # (ActionRanges' low end is ~0.05) genuinely binds.  Without this the
        # budget never touches the trade and the test is inert.
        flipped_cross = dict(cross, r_hat=-cross["r_hat"])
        intended, _, _ = _weights_for(
            mean_action.numpy()[0], ranges, flipped_cross, all_cash, force_budget=2.0
        )

    deployed: list[float] = []
    unclamped: list[float] = []
    overruns: list[float] = []
    ks: list[int] = []
    n_binding = 0
    for a in samples.numpy():
        w_dep, budget, k = _weights_for(a, ranges, cross, intended)
        w_unc, _, _ = _weights_for(a, ranges, cross, intended, force_budget=2.0)
        gross = float(np.abs(w_dep[1:] - intended[1:]).sum())
        deployed.append(gross)
        unc_gross = float(np.abs(w_unc[1:] - intended[1:]).sum())
        unclamped.append(unc_gross)
        overruns.append(gross - budget)
        # Did the clamp actually do anything on THIS draw?  Without this count
        # a budget test can pass because the sampled budget happened to exceed
        # every unclamped move -- see test_the_turnover_budget_clamp_is_live.
        if unc_gross > budget + 1e-9:
            n_binding += 1
        ks.append(k)
    return {
        "deployed_mean": float(np.mean(deployed)),
        "deployed_max": float(np.max(deployed)),
        "unclamped_mean": float(np.mean(unclamped)),
        "unclamped_max": float(np.max(unclamped)),
        "worst_budget_overrun": float(np.max(overruns)),
        "n_binding": float(n_binding),
        "n_samples": float(len(deployed)),
        "min_budget": float(np.min([_weights_for(a, ranges, cross, intended)[1]
                                    for a in samples.numpy()])),
        "k_intended": float(k_mean),
        "k_mean": float(np.mean(ks)),
        "k_std": float(np.std(ks)),
    }


def test_turnover_from_sampling_is_far_below_the_old_policy() -> None:
    """THE number this stream exists to produce.

    Old policy (§1.1, 505 logits through a masked softmax, sigma=0.368):
    L1 0.289 **per day** -> 72.8 gross NAV/yr -> 8.7%/yr of cost, against an
    equal-weight CAGR of 16.4%.  New policy: a handful of bounded scalars read
    by a deterministic allocator, at monthly cadence.
    """
    m = measure_turnover_from_sampling()
    old_annual = OLD_POLICY_L1_PER_DAY * TRADING_DAYS
    new_annual = m["deployed_mean"] * MONTHS

    print(
        "\nturnover-from-sampling (512 draws, 200 names, seed 0)\n"
        f"  intended k                 : {m['k_intended']:.0f}\n"
        f"  sampled  k                 : {m['k_mean']:.1f} +- {m['k_std']:.1f}\n"
        f"  L1 per decision, deployed  : {m['deployed_mean']:.4f} "
        f"(max {m['deployed_max']:.4f})\n"
        f"  L1 per decision, no budget : {m['unclamped_mean']:.4f} "
        f"(max {m['unclamped_max']:.4f})\n"
        f"  annualised, monthly (x12)  : {new_annual:.2f} gross NAV/yr "
        f"= {new_annual * ROUND_TRIP_COST_FRACTION * 100:.2f}%/yr\n"
        f"  old policy, daily (x252)   : {old_annual:.2f} gross NAV/yr "
        f"= {old_annual * ROUND_TRIP_COST_FRACTION * 100:.2f}%/yr\n"
        f"  reduction                  : {old_annual / max(new_annual, 1e-9):.1f}x"
    )

    # A sampled action moves K by a few names, not by tens.
    assert m["k_std"] < 5.0, f"K sampling spread {m['k_std']:.1f} names is not 'a few'"
    # Per decision, below the old policy's per-DAY figure.
    assert m["deployed_mean"] < OLD_POLICY_L1_PER_DAY
    # And the comparison that decides the money: annualised churn from noise.
    assert new_annual < old_annual / 20.0, (
        f"annualised sampling churn {new_annual:.2f} is not >=20x below the old "
        f"policy's {old_annual:.2f}"
    )


def test_a_sampled_action_never_exceeds_its_own_turnover_budget() -> None:
    """The bound is structural: the sampled budget IS the cap on the trade.

    The old policy had no such bound — the size of the trade was whatever the
    softmax of a Gaussian draw happened to be.  Here the same draw that picks
    an aggressive K also picks the budget that constrains it.

    This measures the property in a regime where the clamp actually **binds**.
    The previous version ran only the near-cash starting book at
    ``init_concentration=25``, where the sampled budgets (min 0.3289, mean
    0.5293) never overlapped the unclamped moves (max gross 0.3756): 0 of 512
    draws activated the clamp, and the assertion still passed with the clamp
    deleted entirely (verified by monkeypatching the budget to 1e9 — worst
    overrun -3.245e-02, test still green).  It certified a coincidence of the
    joint distribution, not the structural bound.
    """
    m = measure_turnover_from_sampling(flip=True)
    assert m["n_binding"] > 0.5 * m["n_samples"], (
        f"only {m['n_binding']:.0f}/{m['n_samples']:.0f} draws activate the "
        f"turnover clamp — this test would pass with the clamp deleted"
    )
    assert m["worst_budget_overrun"] <= 1e-9, (
        f"a sampled allocation exceeded its own turnover_budget by "
        f"{m['worst_budget_overrun']:.2e}"
    )


def test_the_turnover_budget_clamp_is_live_not_a_distributional_coincidence() -> None:
    """Deleting the clamp must change the measured turnover.

    The guard against this whole test family going inert again: with the budget
    forced to a no-op the deployed turnover must rise materially.  If it does
    not, the clamp is not doing anything and every 'hard bound' claim in this
    module is unsupported whatever the other assertions say.
    """
    clamped = measure_turnover_from_sampling(flip=True)
    assert clamped["deployed_mean"] < 0.9 * clamped["unclamped_mean"], (
        f"deployed turnover {clamped['deployed_mean']:.4f} is not materially "
        f"below the unclamped {clamped['unclamped_mean']:.4f}: the clamp is inert"
    )


def test_sampling_churn_falls_as_the_policy_sharpens() -> None:
    """PPO concentrating the Beta must reduce churn, not leave it fixed.

    Under the old policy sigma was a free parameter that the entropy bonus
    actively pushed *up* (r7 config comment: "entropy crept sigma from 0.37 to
    0.68"), and churn rose with it.  Here the same mechanism runs the right way.
    """
    def churn(c: float) -> float:
        torch.manual_seed(0)
        ranges = ActionRanges()
        policy = AllocatorPolicy(
            AllocatorPolicyConfig(
                obs_dim=_obs_dim(ranges),
                action_dim=ranges.action_dim,
                init_concentration=c,
            )
        )
        policy.eval()
        cross = _synthetic_cross_section(120, 1)
        obs = torch.zeros(1, policy.obs_dim)
        with torch.no_grad():
            mean_a, _, _, _ = policy.get_action_and_value(obs, deterministic=True)
            draws, _, _, _ = policy.get_action_and_value(obs.expand(128, -1))
        cash = np.zeros(121)
        cash[0] = 1.0
        intended, _, _ = _weights_for(mean_a.numpy()[0], ranges, cross, cash, force_budget=2.0)
        return float(
            np.mean(
                [
                    np.abs(
                        _weights_for(a, ranges, cross, intended, force_budget=2.0)[0][1:]
                        - intended[1:]
                    ).sum()
                    for a in draws.numpy()
                ]
            )
        )

    assert churn(100.0) < churn(25.0) < churn(4.0)


# ── PPO mode discipline ───────────────────────────────────────────────────────


class _ModeRecordingPolicy(AllocatorPolicy):
    """Records ``self.training`` at every call, so the mode order is assertable."""

    def __init__(self, obs_dim: int, action_dim: int) -> None:
        super().__init__(
            AllocatorPolicyConfig(obs_dim=obs_dim, action_dim=action_dim, dropout=0.1)
        )
        self.rollout_modes: list[bool] = []
        self.update_modes: list[bool] = []

    def get_action_and_value(  # type: ignore[override]
        self,
        obs: torch.Tensor,
        action: torch.Tensor | None = None,
        *,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        (self.update_modes if action is not None else self.rollout_modes).append(self.training)
        return super().get_action_and_value(obs, action, deterministic=deterministic)


class _StubAllocatorEnv:
    """Minimal stand-in: constant obs, constant reward, fixed episode length."""

    def __init__(self, obs_dim: int, action_dim: int, ep_len: int = 4) -> None:
        self.obs_dim = obs_dim
        self.ranges = ActionRanges()
        self._action_dim = action_dim
        self._ep_len = ep_len
        self._t = 0

    def reset(self, *, seed: int | None = None, options: object = None):  # type: ignore[no-untyped-def]
        self._t = 0
        return np.zeros(self.obs_dim, dtype=np.float32), {}

    def step(self, action: np.ndarray):  # type: ignore[no-untyped-def]
        self._t += 1
        info = {"excess_log_return": 0.01, "turnover": 0.05, "drawdown": 0.0,
                "k": 30, "nav": 1_000_000.0}
        return (
            np.zeros(self.obs_dim, dtype=np.float32),
            0.01,
            False,
            self._t >= self._ep_len,
            info,
        )


def test_ppo_allocator_mode_discipline(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Rollout in eval(), gradient update in train(), return in eval().

    ``ppo.py`` had this exactly inverted until it was fixed: dropout never
    fired in any backward pass, and evaluation then ran *with* dropout on.
    """
    from pathlib import Path

    from trader.training.ppo_allocator import PPOAllocatorConfig, PPOAllocatorTrainer

    ranges = ActionRanges()
    obs_dim, A = _obs_dim(ranges), ranges.action_dim
    n_envs, n_steps, n_epochs, n_mb = 2, 4, 2, 2
    cfg = PPOAllocatorConfig(
        total_steps=2 * n_steps * n_envs,     # exactly two update iterations
        n_envs=n_envs, n_steps=n_steps, n_epochs=n_epochs, n_minibatches=n_mb,
        target_kl=None,                        # do not early-stop out of the epochs
        checkpoint_dir=Path(tmp_path) / "ckpt",
        log_interval=10**6, checkpoint_interval=10**6,
    )
    model = _ModeRecordingPolicy(obs_dim, A)
    envs = [_StubAllocatorEnv(obs_dim, A) for _ in range(n_envs)]
    trainer = PPOAllocatorTrainer(envs, model, cfg, torch.device("cpu"))  # type: ignore[arg-type]
    trainer.train()

    assert len(model.rollout_modes) == 2 * n_steps, model.rollout_modes
    assert len(model.update_modes) == 2 * n_epochs * n_mb, len(model.update_modes)
    assert all(m is False for m in model.rollout_modes), (
        "rollout must run in eval() — log_prob_old has to be a deterministic "
        "function of the weights, or the importance ratio measures dropout masks"
    )
    assert all(m is True for m in model.update_modes), (
        "the gradient update must run in train(), or dropout never regularises"
    )
    assert not model.training, "train() must return with the model in eval()"


def test_ppo_allocator_returns_in_eval_with_zero_updates(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from pathlib import Path

    from trader.training.ppo_allocator import PPOAllocatorConfig, PPOAllocatorTrainer

    ranges = ActionRanges()
    cfg = PPOAllocatorConfig(
        total_steps=0, n_envs=2, n_steps=4, n_minibatches=2,
        checkpoint_dir=Path(tmp_path) / "ckpt",
    )
    model = _ModeRecordingPolicy(_obs_dim(ranges), ranges.action_dim)
    model.train()
    envs = [_StubAllocatorEnv(_obs_dim(ranges), ranges.action_dim) for _ in range(2)]
    PPOAllocatorTrainer(envs, model, cfg, torch.device("cpu")).train()  # type: ignore[arg-type]
    assert not model.training


def test_trainer_rejects_a_shape_mismatch(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from pathlib import Path

    from trader.training.ppo_allocator import PPOAllocatorConfig, PPOAllocatorTrainer

    ranges = ActionRanges()
    cfg = PPOAllocatorConfig(
        total_steps=8, n_envs=1, n_steps=4, n_minibatches=2,
        checkpoint_dir=Path(tmp_path) / "ckpt",
    )
    model = AllocatorPolicy(AllocatorPolicyConfig(obs_dim=7, action_dim=ranges.action_dim))
    envs = [_StubAllocatorEnv(_obs_dim(ranges), ranges.action_dim)]
    with pytest.raises(ValueError, match="obs_dim"):
        PPOAllocatorTrainer(envs, model, cfg, torch.device("cpu"))  # type: ignore[arg-type]
