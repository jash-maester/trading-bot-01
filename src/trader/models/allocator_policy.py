"""Beta-distributed policy over the allocator's bounded scalars (R6).

Read ``10_architecture_revamp.md`` §1.1 before changing anything here.  The old
policy was a 505-dimensional independent Gaussian over logits pushed through a
masked softmax, and the *sampling alone* moved 28.9% of NAV per day (measured,
§1.1) — 3–9 percentage points of annual return spent on exploration noise before
the policy intended a single trade.  The churn was never a property of RL.  It
was a property of what was being sampled: a sampled action was "a slightly
different 505-vector of portfolio weights".

Here a sampled action is "a slightly different K", "a slightly different
turnover budget", "a slightly tilted sector preference".  Those are read by a
*deterministic* allocator (`trader.allocator.deterministic.allocate`) whose
turnover budget constrains the trade, so exploration cannot manufacture
turnover: the sampled ``turnover_budget`` is itself the bound.

Precisely: the budget binds on ``|Δw|`` measured against a book marked at the
previous close, and the env fills at the open, so realised turnover can exceed
the sampled budget by the overnight gap on the traded names — measured +0.0006
/ +0.0037 / +0.0107 at overnight-gap sd 0 / 1% / 3%.  It is a bound up to the
gap, not a hard one; see the note in `trader.env.allocator_env`.

Why Beta and not a squashed Gaussian
------------------------------------
Both are bounded.  Beta wins on three counts that matter given §1.1:

1. **Bounded by construction, with no change of variables.**  ``tanh``-squashed
   Gaussians need the ``log(1 - tanh(u)²)`` log-determinant correction in
   ``log_prob``; forgetting it (or getting its sign wrong) is a classic silent
   bug that biases the importance ratio, and the ratio is the one quantity PPO
   cannot be wrong about.  Beta's support *is* ``[0, 1]``.
2. **Entropy is analytic in ACTION space.**  ``Beta.entropy()`` is the entropy
   of the distribution over the actions actually executed.  The old entropy
   bonus summed Gaussian entropies over *logits* and was exactly
   ``(N+1)·H(σ)`` — a function of the action count and σ, identical for a
   portfolio concentrated in one stock and one spread over 500 (§1.1).  A
   squashed Gaussian has no closed-form entropy at all, so implementations
   report the pre-squash Gaussian entropy: the same class of lie.
3. **It can be asymmetric.**  A squashed Gaussian near a bound has almost all
   its mass pinned at the bound with a mean that never gets there.  Beta
   represents "K should be near its maximum" honestly.

The cost is that Beta cannot express a point mass, and that ``log_prob`` is
``±inf`` exactly at 0 and 1 — handled by :data:`_ACTION_EPS`.

Concentration parameterisation
------------------------------
``α, β = min_concentration + softplus(·)`` with ``min_concentration >= 1``.
Holding both above 1 keeps every marginal **unimodal**: below 1 the Beta density
is U-shaped (mass piled on both bounds) or unbounded at an endpoint, which for
"K in [10, 60]" would mean a policy that flips between 10 and 60 names and calls
the average its intent — the discrete cousin of the churn this stream exists to
remove.  Entropy stays finite everywhere in the parameterised family.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.distributions import Beta

# `log_prob` of a Beta is -inf at exactly 0 or 1, and `rsample` can return
# those after rounding in float32.  Actions are clamped into the open interval
# before any density is evaluated.  1e-6 on a [0, 1] action is 5e-5 of one name
# on the K axis — far below the integer rounding K goes through anyway.
_ACTION_EPS = 1e-6


def _softplus_inv(y: float) -> float:
    """Inverse of ``softplus``: the pre-activation that yields ``y > 0``."""
    if y <= 0.0:
        raise ValueError(f"softplus is strictly positive; got target {y}")
    return math.log(math.expm1(y))


@dataclass(frozen=True)
class AllocatorPolicyConfig:
    """Shape and initialisation of :class:`AllocatorPolicy`.

    Parameters
    ----------
    obs_dim, action_dim:
        Flat observation width and number of bounded scalars.  Both come from
        :class:`trader.env.allocator_env.AllocatorEnv` — ``action_dim`` is
        ``3 + n_sectors`` (K, turnover budget, cash floor, one tilt per sector).
    hidden:
        Widths of the shared-shape but *separately parameterised* actor and
        critic trunks.  They are separate on purpose: the critic's job is to
        predict one scalar from the whole allocator state, and §1.2 is what
        happens when the critic's input is starved.  Nothing is shared, so no
        actor gradient can wash out the value features.
    dropout:
        Applied inside both trunks.  Defaults to 0.0: the observation is ~30
        floats, not a 504-stock tensor, so there is very little to regularise —
        and dropout in a PPO actor makes ``log_prob_old`` and ``log_prob_new``
        incomparable unless the mode discipline is exactly right
        (``ppo_allocator.py``).  Turn it on deliberately or not at all.
    min_concentration:
        Lower bound on both Beta concentrations; ``>= 1`` keeps marginals
        unimodal (see module docstring).
    init_concentration:
        Concentration both parameters start at, so the initial policy is
        ``Beta(c, c)`` on every axis: symmetric, centred on the middle of each
        range, with standard deviation ``sqrt(1 / (4·(2c + 1)))``.

        The default 25.0 gives 0.0666 on a unit action — K sampled at 35 ± 3.3
        names on a range of [10, 60], and a measured
        ``turnover-from-sampling`` of **0.179 gross NAV per rebalance decision**
        (``tests/unit/test_allocator_policy.py::
        test_turnover_from_sampling_is_far_below_the_old_policy``, which prints
        the number and the decomposition).  At monthly cadence that is 2.14
        gross NAV per year, against the old 505-logit policy's 0.289 *per day*
        = 72.8 per year (``10_architecture_revamp.md`` §1.1): **34× less
        churn from exploration**, or 0.26%/yr against 8.7%/yr priced at the
        verified delivery round trip.

        The knob is not free.  ``init_concentration=4.0`` samples K at 35 ± 8.1
        and measures 0.425 per decision; the sampling spread is roughly linear
        in ``1/sqrt(c)`` and splits about evenly between the K axis and the
        sector tilts.  25.0 is chosen so a *sampled* action differs from the
        policy's intent by a few names, which is the property this whole stream
        exists to obtain — not because 25 is optimal for learning, which is
        **UNVERIFIED and needs a sweep**.  PPO can widen or sharpen it from
        there: the concentration is a learned function of the observation and
        the parameterisation is unbounded above.
    """

    obs_dim: int
    action_dim: int
    hidden: tuple[int, ...] = (128, 128)
    dropout: float = 0.0
    min_concentration: float = 1.0
    init_concentration: float = 25.0

    def __post_init__(self) -> None:
        if self.obs_dim < 1:
            raise ValueError(f"obs_dim must be >= 1, got {self.obs_dim}")
        if self.action_dim < 1:
            raise ValueError(f"action_dim must be >= 1, got {self.action_dim}")
        if not self.hidden:
            raise ValueError("hidden must have at least one layer")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if self.min_concentration < 1.0:
            raise ValueError(
                "min_concentration must be >= 1 so every marginal stays unimodal; "
                f"got {self.min_concentration}"
            )
        if self.init_concentration <= self.min_concentration:
            raise ValueError(
                f"init_concentration ({self.init_concentration}) must exceed "
                f"min_concentration ({self.min_concentration})"
            )


def _mlp(in_dim: int, hidden: tuple[int, ...], out_dim: int, dropout: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    d = in_dim
    for h in hidden:
        layers.append(nn.Linear(d, h))
        layers.append(nn.Tanh())
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        d = h
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


def _orthogonal_init(module: nn.Sequential, final_gain: float) -> None:
    """Orthogonal init, small gain on the head — the CleanRL default."""
    linears = [m for m in module if isinstance(m, nn.Linear)]
    for i, lin in enumerate(linears):
        gain = final_gain if i == len(linears) - 1 else math.sqrt(2.0)
        nn.init.orthogonal_(lin.weight, gain)
        nn.init.zeros_(lin.bias)


class AllocatorPolicy(nn.Module):
    """Actor (Beta over ``[0, 1]^A``) + critic (scalar), both over the same obs.

    The observation is the *allocator state* — current parameters, weight
    summary, per-sector exposure, realised turnover, drawdown, regime, time —
    and both heads see all of it.  Contrast §1.2: the old critic conditioned
    ``V(s)`` on the mean of 504 per-stock embeddings, which is near-constant
    across states, so its advantages were noise and PPO learned the noise.
    Here there is no pooling to hide behind; the critic's input *is* the state.

    Interface mirrors :class:`trader.training.ppo.PPOTrainer`'s model contract
    (``get_action_and_value`` / ``get_value``) so the two trainers read the
    same way.
    """

    def __init__(self, cfg: AllocatorPolicyConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.action_dim = cfg.action_dim
        self.obs_dim = cfg.obs_dim
        self._min_conc = float(cfg.min_concentration)

        self.actor = _mlp(cfg.obs_dim, cfg.hidden, 2 * cfg.action_dim, cfg.dropout)
        self.critic = _mlp(cfg.obs_dim, cfg.hidden, 1, cfg.dropout)
        _orthogonal_init(self.actor, final_gain=0.01)
        _orthogonal_init(self.critic, final_gain=1.0)

        # Bias the actor head so the initial policy is Beta(c, c) on every axis
        # regardless of obs scale: with an orthogonal weight of gain 0.01 the
        # pre-activation is dominated by the bias at init.
        final = [m for m in self.actor if isinstance(m, nn.Linear)][-1]
        with torch.no_grad():
            final.bias.fill_(_softplus_inv(cfg.init_concentration - self._min_conc))

    # ── distribution ─────────────────────────────────────────────────────────

    def concentrations(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``(alpha, beta)``, each ``[B, A]`` and each ``>= min_concentration``."""
        raw = self.actor(obs)
        alpha_raw, beta_raw = raw.chunk(2, dim=-1)
        alpha = self._min_conc + nn.functional.softplus(alpha_raw)
        beta = self._min_conc + nn.functional.softplus(beta_raw)
        return alpha, beta

    def distribution(self, obs: torch.Tensor) -> Beta:
        """The action distribution — independent Beta per dimension."""
        alpha, beta = self.concentrations(obs)
        return Beta(alpha, beta)

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """``V(s)``, shape ``[B]``."""
        return self.critic(obs).squeeze(-1)  # type: ignore[no-any-return]

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: torch.Tensor | None = None,
        *,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(action, log_prob, entropy, value)``.

        ``log_prob`` and ``entropy`` are summed over the action dimensions, so
        both are ``[B]``.  **Entropy is the Beta's analytic entropy in action
        space** — the entropy of the distribution over the scalars the allocator
        actually consumes.  It is not a logit-space proxy, and it is not the
        pre-squash entropy of some other distribution.  See §1.1 for the bonus
        this replaces, which was ``(N+1)·H(σ)`` and said nothing about the
        portfolio.

        ``deterministic=True`` returns the distribution mean
        ``α / (α + β)`` — used for evaluation, never for rollout collection
        (a PPO ratio needs the sampled action).
        """
        dist = self.distribution(obs)
        if action is None:
            action = dist.mean if deterministic else dist.rsample()
        act = action.clamp(_ACTION_EPS, 1.0 - _ACTION_EPS)
        log_prob = dist.log_prob(act).sum(-1)   # type: ignore[no-untyped-call]
        entropy = dist.entropy().sum(-1)       # type: ignore[no-untyped-call]
        return act, log_prob, entropy, self.get_value(obs)

    def forward(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.get_action_and_value(obs)
