"""A2 — compute forensics for one PPO update.

Drives the *unmodified* `PPOTrainer.train()` for a whole number of updates and
instruments it entirely from the outside (monkeypatches applied in this file,
never in `src/`).  Nothing in `src/trader/**` is edited or imported in a
modified form.

What is instrumented
--------------------
  env.step / env.reset            -> environment time
  model.get_action_and_value      -> forward, bucketed by torch.is_grad_enabled()
                                     (rollout = no_grad, update = grad)
  torch.Tensor.backward           -> backward time
  optimizer.step + clip_grad_norm_-> optimiser time
  torch.Tensor.to / torch.tensor  -> exact host->device and device->host bytes
  ppo._batch_obs                  -> rollout-buffer flatten time
  model.encoder forward pre-hook  -> encoder invocation count and sequence count

Synthetic vs end-to-end
-----------------------
`--panel synthetic` builds a fake vectorised env that emits obs dicts with the
exact dtypes/shapes of `PanelTradingEnv` at an arbitrary ticker count.  It is
the only way to see the intended 504-ticker workload, because both shipped
panels hold 163 tickers.  `--panel real` uses the real `PanelTradingEnv` over
`data/panels/train.parquet` (163 tickers) and calibrates the synthetic path.

Subcommands
-----------
  update   one (or more) full PPO updates: phase split, H2D/D2H, peak memory
  profile  torch.profiler over N gradient steps: kernels, shapes, memcpy bytes
  scale    per-gradient-step time vs ticker count and vs lookback
  ceiling  OOM sweep over (ticker count, minibatch size)
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from trader.data.features import FEATURE_COLS  # noqa: E402
from trader.data.regime_features import REGIME_DIM  # noqa: E402
from trader.models.actor_critic import ActorCritic, ModelConfig  # noqa: E402
from trader.training import ppo as ppo_mod  # noqa: E402
from trader.training.ppo import PPOConfig, PPOTrainer  # noqa: E402

F = len(FEATURE_COLS)          # 15
NSECTORS = 8


def pick_device() -> torch.device:
    """Same priority as `trader.utils.seeding.get_device`: cuda -> mps -> cpu."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sync(dev: torch.device) -> None:
    if dev.type == "cuda":
        torch.cuda.synchronize()
    elif dev.type == "mps":
        torch.mps.synchronize()


# ══════════════════════════════════════════════════════════════════════════════
# Synthetic vectorised env — mirrors PanelTradingEnv._build_obs exactly
# ══════════════════════════════════════════════════════════════════════════════
class SyntheticVecEnv:
    """Emits the obs dict of `PanelTradingEnv` at an arbitrary ticker count.

    Memory traffic is faithful: obs["features"] is a real [L, N, F] slice out
    of a shared [T, N, F] float32 panel, stacked across envs exactly the way
    `gymnasium.vector.SyncVectorEnv` stacks per-env dicts.  Trading logic is
    replaced by arithmetic of negligible cost, so env time from this env is a
    LOWER BOUND on the real env; see `--panel real` for the true figure.
    """

    def __init__(
        self, n_envs: int, n_tickers: int, lookback: int, n_dates: int = 1972,
        episode_length: int = 252, seed: int = 0,
    ) -> None:
        rng = np.random.default_rng(seed)
        self.N, self.L, self.E = n_tickers, lookback, n_envs
        self.T = n_dates
        self.episode_length = episode_length
        self.panel = rng.standard_normal((n_dates, n_tickers, F)).astype(np.float32)
        self.mask = np.ones((n_dates, n_tickers), dtype=np.int8)
        self.sector_ids = rng.integers(0, NSECTORS, (n_tickers,)).astype(np.int32)
        self.regime = rng.standard_normal((n_dates, REGIME_DIM)).astype(np.float32)
        self.logret = rng.standard_normal((n_dates, n_tickers)).astype(np.float32) * 0.01
        hi = n_dates - episode_length - 2
        self.start = rng.integers(lookback, max(lookback + 1, hi), size=n_envs)
        self.t = np.zeros(n_envs, dtype=np.int64)
        self.rng = rng
        self.visited_dates: set[int] = set()

    def _obs_one(self, e: int) -> dict[str, np.ndarray]:
        day = int(self.start[e] + self.t[e])
        self.visited_dates.add(day)
        return {
            "features": self.panel[day - self.L: day],
            "mask": self.mask[day],
            "sector_ids": self.sector_ids,
            "portfolio": np.full(self.N + 1, 1.0 / (self.N + 1), dtype=np.float32),
            "cash": np.float32(1e6),
            "nav": np.float32(1e6),
            "t_frac": np.float32(self.t[e] / self.episode_length),
            "recent_return_1d": np.float32(0.0),
            "recent_vol_20d": np.float32(0.0),
            "nav_log_progress": np.float32(0.0),
            "regime": self.regime[day],
            "next_day_returns": self.logret[day],
        }

    def _stack(self) -> dict[str, np.ndarray]:
        per = [self._obs_one(e) for e in range(self.E)]
        return {k: np.stack([p[k] for p in per]) for k in per[0]}

    def reset(self, **_: Any) -> tuple[dict[str, np.ndarray], dict]:
        self.t[:] = 0
        return self._stack(), {}

    def step(self, action: np.ndarray) -> tuple:
        self.t += 1
        term = self.t >= self.episode_length
        for e in np.flatnonzero(term):
            self.t[e] = 0
            self.start[e] = self.rng.integers(
                self.L, max(self.L + 1, self.T - self.episode_length - 2)
            )
        obs = self._stack()
        rew = self.rng.standard_normal(self.E).astype(np.float32) * 1e-3
        info = {"nav": [1e6] * self.E, "turnover": [0.0] * self.E}
        return obs, rew, term.copy(), np.zeros(self.E, bool), info

    def close(self) -> None:
        pass


def make_real_envs(n_envs: int, lookback: int, episode_length: int, seed: int) -> Any:
    import polars as pl
    from gymnasium.vector import SyncVectorEnv

    from trader.data.universe import active_tickers
    from trader.env.panel_env import PanelTradingEnv
    from trader.env.reward import LogReturn

    panel = Path("data/panels/train.parquet")
    have = set(pl.read_parquet(panel, columns=["ticker"])["ticker"].unique().to_list())
    universe = [t for t in active_tickers() if t in have]

    def mk(s: int) -> Any:
        def _init() -> Any:
            return PanelTradingEnv(
                panel_path=panel, universe=universe, feature_columns=FEATURE_COLS,
                lookback=lookback, episode_length=episode_length,
                initial_cash=1e6, turnover_penalty=0.001, reward_fn=LogReturn(),
                use_excess_returns=False, seed=s,
            )
        return _init

    envs = SyncVectorEnv([mk(seed + i) for i in range(n_envs)])
    return envs, len(universe)


# ══════════════════════════════════════════════════════════════════════════════
# Model
# ══════════════════════════════════════════════════════════════════════════════
def build_model(n_tickers: int, compile_: bool = False) -> ActorCritic:
    """Production `model=mlp_regime`: TCN[64,64,64] k=3 -> FiLM -> attn -> FiLM."""
    cfg = ModelConfig(
        in_features=F, n_tickers=n_tickers, embed_dim=128,
        num_channels=[64, 64, 64], kernel_size=3, dropout=0.1,
        use_cross_attn=True, cross_attn_heads=4,
        regime_dim=REGIME_DIM, regime_film_encoder=True,
        regime_film_attn=True, regime_in_critic=True, regime_film_hidden=32,
        use_aux_return_head=False,
    )
    m = ActorCritic(
        cfg,
        feat_mean=torch.zeros(F), feat_std=torch.ones(F),
        regime_mean=torch.zeros(REGIME_DIM), regime_std=torch.ones(REGIME_DIM),
    )
    if compile_:
        m.forward = torch.compile(  # type: ignore[method-assign]
            m.forward, mode="reduce-overhead", dynamic=False, fullgraph=False
        )
    return m


# ══════════════════════════════════════════════════════════════════════════════
# Instrumentation (all monkeypatches — production code untouched)
# ══════════════════════════════════════════════════════════════════════════════
class Probe:
    def __init__(self, dev: torch.device) -> None:
        self.dev = dev
        self.t: dict[str, float] = defaultdict(float)
        self.n: dict[str, int] = defaultdict(int)
        self.h2d_bytes = 0
        self.d2h_bytes = 0
        self.h2d_calls = 0
        self.d2h_calls = 0
        self.enc_calls = 0
        self.enc_seqs = 0
        self._orig: dict[str, Any] = {}

    def sync(self) -> None:
        sync(self.dev)

    def _timed(self, key: str, fn: Any) -> Any:
        def wrap(*a: Any, **k: Any) -> Any:
            self.sync()
            t0 = time.perf_counter()
            out = fn(*a, **k)
            self.sync()
            self.t[key] += time.perf_counter() - t0
            self.n[key] += 1
            return out
        return wrap

    # ── install / remove ──────────────────────────────────────────────────
    def install(self, trainer: PPOTrainer, envs: Any, model: torch.nn.Module) -> None:
        p = self
        dev_t = self.dev.type

        # env
        envs.step = self._timed("env", envs.step)          # type: ignore[method-assign]

        # forward, bucketed by grad mode
        for name in ("get_action_and_value", "get_value", "get_action_value_and_aux"):
            if not hasattr(model, name):
                continue
            orig = getattr(model, name)

            def mk(orig: Any = orig) -> Any:
                def wrap(*a: Any, **k: Any) -> Any:
                    key = "fwd_update" if torch.is_grad_enabled() else "fwd_rollout"
                    p.sync()
                    t0 = time.perf_counter()
                    out = orig(*a, **k)
                    p.sync()
                    p.t[key] += time.perf_counter() - t0
                    p.n[key] += 1
                    return out
                return wrap
            setattr(model, name, mk())

        # backward
        self._orig["backward"] = torch.Tensor.backward

        def bwd(self_t: torch.Tensor, *a: Any, **k: Any) -> Any:
            p.sync()
            t0 = time.perf_counter()
            out = p._orig["backward"](self_t, *a, **k)
            p.sync()
            p.t["backward"] += time.perf_counter() - t0
            p.n["backward"] += 1
            return out
        torch.Tensor.backward = bwd                        # type: ignore[assignment]

        # optimiser + grad clip
        trainer.optimizer.step = self._timed(              # type: ignore[method-assign]
            "optim", trainer.optimizer.step
        )
        self._orig["clip"] = torch.nn.utils.clip_grad_norm_
        torch.nn.utils.clip_grad_norm_ = self._timed(      # type: ignore[assignment]
            "optim", self._orig["clip"]
        )

        # rollout-buffer flatten
        self._orig["batch_obs"] = ppo_mod._batch_obs
        ppo_mod._batch_obs = self._timed("batch_obs", self._orig["batch_obs"])

        # ── transfers: Tensor.to and torch.tensor(device=) ────────────────
        # If PyTorch refuses the patch we still get memcpy bytes from the
        # profiler trace in `cmd_profile`; nothing silently reports zero.
        self._orig["to"] = torch.Tensor.to

        def to_wrap(self_t: torch.Tensor, *a: Any, **k: Any) -> Any:
            tgt = None
            for x in a:
                if isinstance(x, (str, torch.device)):
                    tgt = torch.device(x)
            if "device" in k and k["device"] is not None:
                tgt = torch.device(k["device"])
            counted = False
            if tgt is not None:
                if self_t.device.type == "cpu" and tgt.type == dev_t:
                    p.h2d_bytes += self_t.numel() * self_t.element_size()
                    p.h2d_calls += 1
                    counted = True
                elif self_t.device.type == dev_t and tgt.type == "cpu":
                    p.d2h_bytes += self_t.numel() * self_t.element_size()
                    p.d2h_calls += 1
                    counted = True
            if not counted:
                return p._orig["to"](self_t, *a, **k)
            p.sync()
            t0 = time.perf_counter()
            out = p._orig["to"](self_t, *a, **k)
            p.sync()
            p.t["h2d" if tgt.type == dev_t else "d2h"] += time.perf_counter() - t0
            return out
        torch.Tensor.to = to_wrap                          # type: ignore[assignment]

        self._orig["cpu"] = torch.Tensor.cpu

        def cpu_wrap(self_t: torch.Tensor, *a: Any, **k: Any) -> Any:
            if self_t.device.type == dev_t:
                p.d2h_bytes += self_t.numel() * self_t.element_size()
                p.d2h_calls += 1
                p.sync()
                t0 = time.perf_counter()
                out = p._orig["cpu"](self_t, *a, **k)
                p.sync()
                p.t["d2h"] += time.perf_counter() - t0
                return out
            return p._orig["cpu"](self_t, *a, **k)
        torch.Tensor.cpu = cpu_wrap                        # type: ignore[assignment]

        self._orig["tensor"] = torch.tensor

        def tensor_wrap(data: Any, *a: Any, **k: Any) -> Any:
            out = p._orig["tensor"](data, *a, **k)
            if out.device.type == dev_t:
                p.h2d_bytes += out.numel() * out.element_size()
                p.h2d_calls += 1
            return out
        torch.tensor = tensor_wrap                         # type: ignore[assignment]

        # encoder invocation count
        enc = getattr(model, "encoder", None)
        if enc is not None:
            def hook(_m: Any, inp: Any) -> None:
                x = inp[0]
                p.enc_calls += 1
                p.enc_seqs += int(x.shape[0]) * int(x.shape[1])
            enc.register_forward_pre_hook(hook)

    def remove(self) -> None:
        if "backward" in self._orig:
            torch.Tensor.backward = self._orig["backward"]   # type: ignore[assignment]
        if "to" in self._orig:
            torch.Tensor.to = self._orig["to"]               # type: ignore[assignment]
        if "cpu" in self._orig:
            torch.Tensor.cpu = self._orig["cpu"]             # type: ignore[assignment]
        if "tensor" in self._orig:
            torch.tensor = self._orig["tensor"]              # type: ignore[assignment]
        if "clip" in self._orig:
            torch.nn.utils.clip_grad_norm_ = self._orig["clip"]  # type: ignore[assignment]
        if "batch_obs" in self._orig:
            ppo_mod._batch_obs = self._orig["batch_obs"]


def emit(rep: dict, a: argparse.Namespace, name: str) -> None:
    """Print AND persist.  Long remote runs lose their stdout pipe when the
    caller detaches, so every result is also written to `--outdir`."""
    txt = json.dumps(rep, indent=2)
    print(txt, flush=True)
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.json").write_text(txt)


def host_peak_gib() -> float:
    import resource
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KiB, macOS bytes.
    return ru / 2**20 if sys.platform.startswith("linux") else ru / 2**30


# ══════════════════════════════════════════════════════════════════════════════
# cmd: update
# ══════════════════════════════════════════════════════════════════════════════
def cmd_update(a: argparse.Namespace) -> None:
    dev = pick_device()
    torch.manual_seed(0)

    if a.panel == "real":
        envs, n_tick = make_real_envs(a.n_envs, a.lookback, a.episode_length, 42)
    else:
        n_tick = a.n_tickers
        envs = SyntheticVecEnv(a.n_envs, n_tick, a.lookback,
                               episode_length=a.episode_length)

    model = build_model(n_tick, compile_=a.compile)
    cfg = PPOConfig(
        total_steps=a.n_envs * a.n_steps * a.updates, n_envs=a.n_envs,
        n_steps=a.n_steps, n_epochs=a.n_epochs, n_minibatches=a.n_minibatches,
        target_kl=None,  # never early-stop: we want every epoch measured
        log_interval=10**9, checkpoint_interval=10**9, eval_interval=10**9,
        checkpoint_dir=Path(a.outdir) / "ckpt",
    )
    trainer = PPOTrainer(envs, model, cfg, dev)
    probe = Probe(dev)
    # The probe patches torch.Tensor.to / .cpu / .backward at class level.
    # Under bf16 autocast every op does a dtype .to(), so the Python wrapper
    # is on a very hot path: expect a large wall-clock inflation.  Use
    # --no-probe for a clean total, --probe for the byte/phase accounting.
    if a.probe:
        probe.install(trainer, envs, model)

    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    trainer.train()
    probe.sync()
    wall = time.perf_counter() - t0
    probe.remove()

    mb_bytes = a.lookback * n_tick * F * 4 * (a.n_envs * a.n_steps // a.n_minibatches)
    buf_bytes = a.lookback * n_tick * F * 4 * a.n_envs * a.n_steps
    rep = {
        "panel": a.panel, "n_tickers": n_tick, "lookback": a.lookback,
        "n_envs": a.n_envs, "n_steps": a.n_steps, "n_epochs": a.n_epochs,
        "n_minibatches": a.n_minibatches, "compile": a.compile,
        "updates": a.updates, "instrumented": a.probe,
        "minibatch_transitions": a.n_envs * a.n_steps // a.n_minibatches,
        "grad_steps_per_update": a.n_epochs * a.n_minibatches,
        "wall_s_total": wall, "wall_s_per_update": wall / a.updates,
        "phase_s_per_update": {k: v / a.updates for k, v in sorted(probe.t.items())},
        "phase_calls_per_update": {
            k: v / a.updates for k, v in sorted(probe.n.items())
        },
        "h2d_GiB_per_update": probe.h2d_bytes / 2**30 / a.updates,
        "d2h_GiB_per_update": probe.d2h_bytes / 2**30 / a.updates,
        "h2d_calls_per_update": probe.h2d_calls / a.updates,
        "d2h_calls_per_update": probe.d2h_calls / a.updates,
        "analytic_minibatch_features_MiB": mb_bytes / 2**20,
        "analytic_rollout_buffer_GiB": buf_bytes / 2**30,
        "encoder_calls_per_update": probe.enc_calls / a.updates,
        "encoder_sequences_per_update": probe.enc_seqs / a.updates,
        "host_peak_rss_GiB": host_peak_gib(),
    }
    if isinstance(envs, SyntheticVecEnv):
        rep["distinct_dates_visited"] = len(envs.visited_dates)
        rep["distinct_date_stock_pairs"] = len(envs.visited_dates) * n_tick
    if dev.type == "cuda":
        rep["vram_peak_alloc_GiB"] = torch.cuda.max_memory_allocated() / 2**30
        rep["vram_peak_reserved_GiB"] = torch.cuda.max_memory_reserved() / 2**30
    tag = "probe" if a.probe else "clean"
    emit(rep, a, f"update_{a.panel}_n{n_tick}_L{a.lookback}_{tag}")


# ══════════════════════════════════════════════════════════════════════════════
# cmd: profile   (torch.profiler over N gradient steps)
# ══════════════════════════════════════════════════════════════════════════════
def _synth_minibatch(n_tick: int, lookback: int, mb: int, dev: torch.device) -> dict:
    """One CPU-resident minibatch, exactly as `_batch_obs` leaves it."""
    return {
        "features": torch.randn(mb, lookback, n_tick, F),
        "mask": torch.ones(mb, n_tick, dtype=torch.int8),
        "sector_ids": torch.randint(0, NSECTORS, (mb, n_tick), dtype=torch.int32),
        "portfolio": torch.rand(mb, n_tick + 1),
        "cash": torch.rand(mb), "nav": torch.rand(mb), "t_frac": torch.rand(mb),
        "recent_return_1d": torch.randn(mb), "recent_vol_20d": torch.rand(mb),
        "nav_log_progress": torch.randn(mb),
        "regime": torch.randn(mb, REGIME_DIM),
        "next_day_returns": torch.randn(mb, n_tick),
    }


def _grad_step(model: Any, opt: Any, mb_cpu: dict, dev: torch.device,
               amp: bool) -> None:
    """One PPO gradient step, mirroring ppo.py:330-380 (H2D inside the loop)."""
    mb_obs = {k: v.to(dev) for k, v in mb_cpu.items()}
    act = torch.randn(mb_obs["portfolio"].shape, device=dev)
    ctx = torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=amp)
    with ctx:
        _, lp, ent, val = model.get_action_and_value(mb_obs, act)
    loss = lp.mean() + val.mean() + ent.mean()
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    opt.step()


def cmd_profile(a: argparse.Namespace) -> None:
    from torch.profiler import ProfilerActivity, profile

    dev = pick_device()
    torch.manual_seed(0)
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)

    model = build_model(a.n_tickers, compile_=a.compile).to(dev)
    model.eval()
    opt = torch.optim.Adam(model.parameters(), lr=3e-4, eps=1e-5)
    mb = a.n_envs * a.n_steps // a.n_minibatches
    mb_cpu = _synth_minibatch(a.n_tickers, a.lookback, mb, dev)

    for _ in range(3):
        _grad_step(model, opt, mb_cpu, dev, a.amp)
    sync(dev)
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    for _ in range(a.steps):
        _grad_step(model, opt, mb_cpu, dev, a.amp)
    sync(dev)
    untimed = (time.perf_counter() - t0) / a.steps

    acts = [ProfilerActivity.CPU]
    if dev.type == "cuda":
        acts.append(ProfilerActivity.CUDA)
    with profile(activities=acts, record_shapes=True, profile_memory=True,
                 with_flops=True, with_stack=False) as prof:
        for _ in range(a.steps):
            _grad_step(model, opt, mb_cpu, dev, a.amp)
        sync(dev)

    trace = out / f"trace_n{a.n_tickers}_L{a.lookback}_mb{mb}.json"
    prof.export_chrome_trace(str(trace))

    # ── memcpy bytes from the trace ───────────────────────────────────────
    memcpy: dict[str, dict[str, float]] = defaultdict(
        lambda: {"bytes": 0.0, "count": 0.0, "us": 0.0}
    )
    with open(trace) as fh:
        ev = json.load(fh)
    events = ev["traceEvents"] if isinstance(ev, dict) else ev
    for e in events:
        if e.get("cat") not in ("gpu_memcpy", "Memcpy", "gpu_memset"):
            continue
        nm = e.get("name", "?")
        b = e.get("args", {}).get("bytes", 0) or 0
        memcpy[nm]["bytes"] += float(b)
        memcpy[nm]["count"] += 1
        memcpy[nm]["us"] += float(e.get("dur", 0) or 0)

    steps = a.steps
    gs_per_update = a.n_epochs * a.n_minibatches
    memcpy_rep = {
        k: {
            "GiB_per_grad_step": v["bytes"] / steps / 2**30,
            "GiB_per_update": v["bytes"] / steps * gs_per_update / 2**30,
            "calls_per_grad_step": v["count"] / steps,
            "ms_per_grad_step": v["us"] / steps / 1e3,
        }
        for k, v in sorted(memcpy.items(), key=lambda kv: -kv[1]["bytes"])
    }

    # ── top kernels ───────────────────────────────────────────────────────
    ka = prof.key_averages()
    keys = (["self_device_time_total", "self_cuda_time_total"]
            if dev.type == "cuda" else []) + ["self_cpu_time_total"]
    table = ""
    for k in keys:
        try:
            table = ka.table(sort_by=k, row_limit=25)
            break
        except Exception:
            continue

    # ── ops with a leading dim of 1, > 100 calls per update ───────────────
    leading_one: list[dict] = []
    for e in prof.key_averages(group_by_input_shape=True):
        shp = getattr(e, "input_shapes", None)
        if not shp:
            continue
        first = next((s for s in shp if isinstance(s, list) and s), None)
        if first and first[0] == 1 and len(first) > 1:
            per_update = e.count / steps * gs_per_update
            if per_update > 100:
                leading_one.append({
                    "op": e.key, "shapes": str(shp)[:160],
                    "calls_per_update": per_update,
                })

    # ── FLOPs reported by the profiler (conv + matmul only) ───────────────
    flops = sum(getattr(e, "flops", 0) or 0 for e in ka)
    dev_us = 0.0
    for e in ka:
        for attr in ("self_device_time_total", "self_cuda_time_total"):
            v = getattr(e, attr, None)
            if v:
                dev_us += float(v)
                break

    rep = {
        "device": str(dev), "n_tickers": a.n_tickers, "lookback": a.lookback,
        "minibatch": mb, "amp_bf16": a.amp, "compile": a.compile,
        "profiled_grad_steps": steps,
        "profiler_flops_per_grad_step": flops / steps,
        "profiler_self_device_ms_per_grad_step": dev_us / steps / 1e3,
        "grad_steps_per_update": gs_per_update,
        "untimed_s_per_grad_step": untimed,
        "untimed_s_per_update_gradonly": untimed * gs_per_update,
        "memcpy": memcpy_rep,
        "memcpy_total_h2d_GiB_per_update": sum(
            v["GiB_per_update"] for k, v in memcpy_rep.items() if "HtoD" in k
        ),
        "memcpy_total_d2h_GiB_per_update": sum(
            v["GiB_per_update"] for k, v in memcpy_rep.items() if "DtoH" in k
        ),
        "leading_dim_one_ops": leading_one,
        "trace": str(trace),
    }
    if dev.type == "cuda":
        rep["vram_peak_alloc_GiB"] = torch.cuda.max_memory_allocated() / 2**30
        rep["vram_peak_reserved_GiB"] = torch.cuda.max_memory_reserved() / 2**30
    emit(rep, a, f"profile_n{a.n_tickers}_L{a.lookback}")
    print("\n===== KEY AVERAGES =====")
    print(table)
    (out / f"kernels_n{a.n_tickers}_L{a.lookback}.txt").write_text(table)


# ══════════════════════════════════════════════════════════════════════════════
# cmd: scale
# ══════════════════════════════════════════════════════════════════════════════
def _time_grad_steps(n_tick: int, lookback: int, mb: int, dev: torch.device,
                     amp: bool, compile_: bool, iters: int) -> dict:
    torch.manual_seed(0)
    model = build_model(n_tick, compile_=compile_).to(dev)
    model.eval()
    opt = torch.optim.Adam(model.parameters(), lr=3e-4, eps=1e-5)
    mb_cpu = _synth_minibatch(n_tick, lookback, mb, dev)
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for _ in range(3):
        _grad_step(model, opt, mb_cpu, dev, amp)
    sync(dev)
    t0 = time.perf_counter()
    for _ in range(iters):
        _grad_step(model, opt, mb_cpu, dev, amp)
    sync(dev)
    dt = (time.perf_counter() - t0) / iters

    # H2D-only time for the same minibatch, measured separately
    sync(dev)
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = {k: v.to(dev) for k, v in mb_cpu.items()}
        sync(dev)
    h2d_s = (time.perf_counter() - t0) / iters

    res = {
        "n_tickers": n_tick, "lookback": lookback, "minibatch": mb,
        "s_per_grad_step": dt, "h2d_s_per_grad_step": h2d_s,
        "mb_features_MiB": mb * lookback * n_tick * F * 4 / 2**20,
    }
    if dev.type == "cuda":
        res["vram_peak_alloc_GiB"] = torch.cuda.max_memory_allocated() / 2**30
    del model, opt, mb_cpu
    gc.collect()
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return res


def cmd_scale(a: argparse.Namespace) -> None:
    dev = pick_device()
    mb = a.n_envs * a.n_steps // a.n_minibatches
    rows = []
    for n in [int(x) for x in a.tickers.split(",")]:
        for L in [int(x) for x in a.lookbacks.split(",")]:
            try:
                rows.append(_time_grad_steps(n, L, mb, dev, a.amp, a.compile, a.iters))
            except Exception as e:                     # OOM or anything else
                rows.append({"n_tickers": n, "lookback": L, "minibatch": mb,
                             "s_per_grad_step": None,
                             "error": f"{type(e).__name__}: {str(e)[:160]}"})
                gc.collect()
                if dev.type == "cuda":
                    torch.cuda.empty_cache()
            # Persist after every cell: a later cell may kill the process.
            emit({"amp_bf16": a.amp, "compile": a.compile, "rows": rows}, a, "scale")


# ══════════════════════════════════════════════════════════════════════════════
# cmd: ceiling
# ══════════════════════════════════════════════════════════════════════════════
def cmd_ceiling(a: argparse.Namespace) -> None:
    dev = pick_device()
    rows = []
    for n in [int(x) for x in a.tickers.split(",")]:
        for mb in [int(x) for x in a.minibatches.split(",")]:
            row: dict = {"n_tickers": n, "minibatch": mb, "lookback": a.lookback}
            row["mb_features_MiB"] = mb * a.lookback * n * F * 4 / 2**20
            try:
                r = _time_grad_steps(n, a.lookback, mb, dev, a.amp, False, 3)
                row.update({"ok": True, "s_per_grad_step": r["s_per_grad_step"],
                            "vram_peak_alloc_GiB": r.get("vram_peak_alloc_GiB")})
            except torch.cuda.OutOfMemoryError as e:
                row.update({"ok": False, "error": str(e).split("\n")[0][:180]})
                gc.collect()
                torch.cuda.empty_cache()
            except RuntimeError as e:
                row.update({"ok": False, "error": str(e).split("\n")[0][:180]})
                gc.collect()
                torch.cuda.empty_cache()
            row["host_peak_rss_GiB"] = host_peak_gib()
            rows.append(row)
            print(json.dumps(row), flush=True)
    emit({"amp_bf16": a.amp, "rows": rows}, a, "ceiling")


# ══════════════════════════════════════════════════════════════════════════════
# cmd: dates — encoder-invocation ratio, no model needed
# ══════════════════════════════════════════════════════════════════════════════
def cmd_dates(a: argparse.Namespace) -> None:
    """How many distinct (date, stock) pairs one production rollout touches.

    Steps the env sampler for a full rollout without building a model, so it
    is seconds rather than an hour.  `--panel real` uses PanelTradingEnv's own
    episode sampler; `--panel synthetic` mirrors it at any ticker count.
    """
    seeds = [int(x) for x in a.seeds.split(",")]
    rows = []
    for sd in seeds:
        if a.panel == "real":
            envs, n_tick = make_real_envs(a.n_envs, a.lookback, a.episode_length, sd)
            envs.reset(seed=sd)
            seen: set[int] = set()
            for e in envs.envs:
                seen.add(int(e._start_idx + e._t))
            for _ in range(a.n_steps):
                envs.step(np.zeros((a.n_envs, n_tick + 1), dtype=np.float32))
                for e in envs.envs:
                    seen.add(int(e._start_idx + e._t))
            n_dates_panel = len(envs.envs[0]._dates)
        else:
            n_tick = a.n_tickers
            env = SyntheticVecEnv(a.n_envs, n_tick, a.lookback,
                                  episode_length=a.episode_length, seed=sd)
            env.reset()
            for _ in range(a.n_steps):
                env.step(np.zeros((a.n_envs, n_tick + 1), dtype=np.float32))
            seen = env.visited_dates
            n_dates_panel = env.T
        rollout_calls = (a.n_steps + 1) * a.n_envs
        update_calls = a.n_epochs * a.n_steps * a.n_envs
        seqs = (rollout_calls + update_calls) * n_tick
        rows.append({
            "seed": sd, "n_tickers": n_tick,
            "panel_dates": n_dates_panel,
            "distinct_dates_per_update": len(seen),
            "distinct_date_stock_pairs_per_update": len(seen) * n_tick,
            "encoder_sequences_per_update": seqs,
            "redundancy_ratio_per_update": seqs / (len(seen) * n_tick),
            "sequences_if_cached_whole_panel_once": n_dates_panel * n_tick,
        })
        print(json.dumps(rows[-1]), flush=True)
    emit({"panel": a.panel, "rows": rows}, a, f"dates_{a.panel}_n{rows[0]['n_tickers']}")


# ══════════════════════════════════════════════════════════════════════════════
# cmd: analytic  — closed-form FLOP count of the production model
# ══════════════════════════════════════════════════════════════════════════════
def cmd_analytic(a: argparse.Namespace) -> None:
    """FLOPs of `model=mlp_regime` (TCN[64,64,64] k=3, d=64, 4-head attn).

    Counted as 2 FLOP per multiply-accumulate.  Causal padding makes every
    conv output length equal to L.
    """
    N, L, mb = a.n_tickers, a.lookback, a.n_envs * a.n_steps // a.n_minibatches
    ch, k = [64, 64, 64], 3
    d = ch[-1]

    # ── TCN, per stock-sequence ───────────────────────────────────────────
    macs = 0
    in_ch = F
    tcn: dict[str, int] = {}
    for i, out_ch in enumerate(ch):
        c1 = L * out_ch * in_ch * k
        c2 = L * out_ch * out_ch * k
        ds = L * out_ch * in_ch if in_ch != out_ch else 0
        tcn[f"block{i}"] = c1 + c2 + ds
        macs += c1 + c2 + ds
        in_ch = out_ch
    tcn_flops_per_seq = 2 * macs

    # ── Cross-stock attention, per batch element (all N stocks) ───────────
    attn_macs = 3 * N * d * d          # q,k,v projections
    attn_macs += N * N * d             # scores
    attn_macs += N * N * d             # weighted sum
    attn_macs += N * d * d             # output projection
    attn_flops_per_batch_elem = 2 * attn_macs

    # ── Heads, per batch element ──────────────────────────────────────────
    head_hidden = 64
    actor = N * ((d + 1) * head_hidden + head_hidden * 1)      # equity MLP
    actor += (d + 1) * head_hidden + head_hidden * 1           # cash MLP
    head_flops_per_batch_elem = 2 * actor

    fwd_per_step = (
        tcn_flops_per_seq * mb * N
        + (attn_flops_per_batch_elem + head_flops_per_batch_elem) * mb
    )
    rep = {
        "n_tickers": N, "lookback": L, "minibatch": mb,
        "tcn_MFLOP_per_stock_sequence": tcn_flops_per_seq / 1e6,
        "tcn_MACs_per_stock_sequence": macs / 1e6,
        "tcn_per_block_MACs": {k_: v / 1e6 for k_, v in tcn.items()},
        "attn_GFLOP_per_grad_step": attn_flops_per_batch_elem * mb / 1e9,
        "head_GFLOP_per_grad_step": head_flops_per_batch_elem * mb / 1e9,
        "sequences_per_grad_step": mb * N,
        "fwd_GFLOP_per_grad_step": fwd_per_step / 1e9,
        "fwd_bwd_GFLOP_per_grad_step_x3": 3 * fwd_per_step / 1e9,
        "grad_steps_per_update": a.n_epochs * a.n_minibatches,
        "fwd_bwd_TFLOP_per_update": 3 * fwd_per_step
        * a.n_epochs * a.n_minibatches / 1e12,
        "updates_for_2M_env_steps": 2_000_000 / (a.n_envs * a.n_steps),
        "PFLOP_for_2M_env_steps": 3 * fwd_per_step * a.n_epochs * a.n_minibatches
        * (2_000_000 / (a.n_envs * a.n_steps)) / 1e15,
    }
    emit(rep, a, f"analytic_n{N}_L{L}")


# ══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(q: argparse.ArgumentParser) -> None:
        q.add_argument("--n-tickers", type=int, default=504)
        q.add_argument("--lookback", type=int, default=60)
        q.add_argument("--n-envs", type=int, default=16)
        q.add_argument("--n-steps", type=int, default=256)
        q.add_argument("--n-epochs", type=int, default=4)
        q.add_argument("--n-minibatches", type=int, default=32)
        q.add_argument("--amp", action="store_true", default=True)
        q.add_argument("--no-amp", dest="amp", action="store_false")
        q.add_argument("--compile", action="store_true")
        q.add_argument("--outdir", default="outputs/a2")

    q = sub.add_parser("update")
    common(q)
    q.add_argument("--updates", type=int, default=1)
    q.add_argument("--episode-length", type=int, default=252)
    q.add_argument("--probe", action="store_true", default=True)
    q.add_argument("--no-probe", dest="probe", action="store_false")
    q.add_argument("--panel", choices=["synthetic", "real"], default="synthetic")
    q.set_defaults(fn=cmd_update)

    q = sub.add_parser("profile")
    common(q)
    q.add_argument("--steps", type=int, default=6)
    q.set_defaults(fn=cmd_profile)

    q = sub.add_parser("scale")
    common(q)
    q.add_argument("--tickers", default="163,300,504,645")
    q.add_argument("--lookbacks", default="60,30,20")
    q.add_argument("--iters", type=int, default=10)
    q.set_defaults(fn=cmd_scale)

    q = sub.add_parser("analytic")
    common(q)
    q.set_defaults(fn=cmd_analytic)

    q = sub.add_parser("dates")
    common(q)
    q.add_argument("--episode-length", type=int, default=252)
    q.add_argument("--panel", choices=["synthetic", "real"], default="synthetic")
    q.add_argument("--seeds", default="0,1,2")
    q.set_defaults(fn=cmd_dates)

    q = sub.add_parser("ceiling")
    common(q)
    q.add_argument("--tickers", default="163,300,504,645,1000")
    q.add_argument("--minibatches", default="64,128,256,512,1024")
    q.set_defaults(fn=cmd_ceiling)

    a = p.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    a.fn(a)


if __name__ == "__main__":
    main()
