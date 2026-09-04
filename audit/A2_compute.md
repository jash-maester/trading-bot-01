# A2 — Compute forensics

**Scope.** `09_revamp_and_audit.md` §6 (A2), against §5 (R3). Read-only with
respect to `src/`: no production file was edited. New files are this report and
`scripts/profiling/{a2_hw_probe.py, a2_ppo_update.py, a2_run.sh, a2_all.sh}`.

**Date.** 2026-09-04 / 05. Commit at start: `b89f156`.

---

## 0. Provenance — read this before using any number

Three machines appear below. Every table says which.

| Tag | Machine | Notes |
|---|---|---|
| **4060** | RTX 4060 Laptop, 8188 MiB, cc 8.9, 24 SM, torch 2.11.0+cu130 / CUDA 13.0, WSL2, 20 cores, 29 GiB RAM | the target |
| **M4-MPS** | M4 Mac mini, 24 GiB unified, MPS backend | calibration only |
| **M4-CPU** | same box, CPU backend | used where the measurement is device-independent |

Two workloads appear below.

| Tag | Meaning |
|---|---|
| **synthetic** | tensors generated at the correct shape for an arbitrary ticker count. Used because both shipped panels hold 163 tickers while `active_tickers()` returns 504 — a real 504-ticker end-to-end run is not constructible from the data on disk. |
| **real panel** | `data/panels/train.parquet` through the real `PanelTradingEnv`. |

Everything reproduces from the two scripts. Commands are printed inline.

### 0.1 Instrumentation caveat

`scripts/profiling/a2_ppo_update.py` measures the phase split by monkey-patching
`torch.Tensor.to`, `.cpu`, `.backward`, `optimizer.step`, `nn.utils.clip_grad_norm_`
and `ppo._batch_obs` **from the profiling script**, and by forcing a device
synchronise around each. Under bf16 autocast, `Tensor.to` is on an extremely hot
path, so the instrumented wall clock is inflated several-fold. **Instrumented runs
are used only for byte counts, call counts and phase *proportions*; every
wall-clock number is from an uninstrumented run** (`--no-probe`, or the `scale` /
`profile` subcommands, which never install the probe).

### 0.2 What could not be measured, and why

Partway through the session the training box began refusing SSH at the
authentication stage —

```
debug1: Server accepts key: /Users/jash/.ssh/id_ed25519 ED25519 SHA256:MXwpmn8R…
jashm@192.168.1.7: Permission denied (publickey,password,keyboard-interactive).
```

— from both the LAN address and the Tailscale name, for over two hours, while
port 22 stayed open and the host stayed pingable. That is the Windows-OpenSSH
signature of the server accepting the key and then failing to create the user
logon token; it is not a network or key problem and it is not fixable from this
side. Sessions had been opened at a high rate for two hours before it started.

Consequently the following A2 items are **NOT MEASURED**, and are marked as such
where they appear. They are not estimated:

- 4060 uninstrumented per-gradient-step time, and hence achieved FLOPS on the 4060
- 4060 `torch.profiler` kernel table and memcpy byte totals
- 4060 scaling curves (ticker count, lookback)
- 4060 OOM ceiling sweep

`scripts/profiling/a2_all.sh` runs all of them unattended in cost order and writes
one JSON per step under `outputs/a2/`, so a step that dies does not lose the
earlier ones. After `make win-push`, launch with

```
ssh -o BatchMode=yes jashm@192.168.1.7 \
  'wsl.exe -e bash /mnt/d/trading-bot-01/scripts/profiling/a2_all.sh'
ssh -o BatchMode=yes jashm@192.168.1.7 \
  'wsl.exe -e bash /mnt/d/trading-bot-01/scripts/profiling/a2_run.sh cat scale.json'
```

To clear the SSH state, restart the service from an elevated PowerShell on the
box (`Restart-Service sshd`) or reboot it; nothing on the Mac side will fix it.
**Operational note for the rest of the programme:** this failure followed ~two
hours of one-shot `ssh` invocations at a high rate. Long remote runs should use
one multiplexed connection (`ControlMaster`/`ControlPersist`, socket path under
104 bytes) and poll a file rather than reconnecting per check. Every long run
must also write its result to a file on the box — `a2_run.sh` exists because a
detached foreground `ssh … | tail` loses the pipe and, with it, an hour of GPU
time.

---

## 1. The configuration under test

Production defaults: `model=mlp_regime`, `train=ppo_baseline`, `env=panel_daily`.

| Quantity | Value | Source |
|---|---|---|
| tickers `N` | 504 | `active_tickers()`, `runner.py:107` |
| features `F` | 15 | `features.py:16` `FEATURE_COLS` |
| lookback `L` | 60 | `configs/env/panel_daily.yaml` |
| regime dim `R` | 6 | `regime_features.py:42` |
| envs × steps | 16 × 256 = 4096 per rollout | `configs/train/ppo_baseline.yaml` |
| minibatches × epochs | 32 × 4 = **128 gradient steps per update** | same |
| minibatch | 4096/32 = **128 transitions** | `ppo.py:163` |
| TCN | channels `[64,64,64]`, k=3, dilations 1/2/4 | `configs/model/mlp_regime.yaml` |
| model width `d` | **64**, not 128 | `encoders.py:273` — `embed_dim: 128` in the config is dead whenever `num_channels` is set |
| updates for 2M env steps | 488.28 | 2e6 / 4096 |
| updates for the configured 8M | 1953.1 | `total_steps: 8_000_000` |

### 1.1 The ticker axis is 72% empty

```
$ .venv/bin/python -c "…"   # see §11 for the full command
train panel tickers 163 | active_tickers 504 | overlap 143 | in panel not active 20
all_tickers 645
```

`PanelTradingEnv.__init__` allocates `self._stacked_features = np.zeros((T, N, F))`
with `N = len(universe) = 504` and fills only the columns it finds
(`panel_env.py:105-109`). **361 of 504 ticker columns (71.6%) are structurally
all-zero** in every observation, and `is_tradeable` is False for them. Every
convolution, every attention key, every byte moved over PCIe for those columns is
waste. This is a data-state fact, not an architecture fact — but at the current
data state it means **71.6% of the entire compute budget below is spent on zeros.**

---

## 2. Four corrections to the premises A2 was given

| Premise (from `09_…md` §5 R3 / §6 A2) | Verdict | Corrected value |
|---|---|---|
| ~3.3 MFLOP per stock-sequence; ~640 GFLOP fwd+bwd per gradient step; ~40 PFLOP for 2M steps | **understated 2.4×** | 7.83 MFLOP/seq; 1549 GFLOP/step; **96.8 PFLOP** |
| ~27.7 GB host→device per update | **confirmed for the update loop; incomplete as a total** | 27.79 GiB in the update loop **+ 6.97 GiB in the rollout = 34.76 GiB H2D**, plus **6.95 GiB D2H** |
| "Thousands of sub-millisecond `aten::conv1d` calls ⇒ a Python loop over the stock axis" | **refuted** | no per-stock loop exists; §4.3 |
| "The rollout buffer is already not on the GPU" | **confirmed** | `ppo.py:113` builds it with no `device=`; the measured host RSS is 15.8 GiB while VRAM holds only activations |
| "There is no VRAM overflow… **VRAM is not binding**" | half right, and the wrong half is load-bearing | no overflow *from the buffer* — but the **activations** put the process at **7762–7766 MiB of 8188 MiB** at 504 × mb 128, and the measured cost curve has a cliff exactly there; §7.2, §8 |

---

## 3. The 4060's actual envelope — measured, not assumed

```
ssh -o BatchMode=yes jashm@192.168.1.7 \
 'wsl.exe -e bash /mnt/d/trading-bot-01/scripts/win_bootstrap.sh run scripts/profiling/a2_hw_probe.py'
```

`§4.5` of the revamp plan budgets against "**~15 TFLOPS** FP32". That figure is
the desktop 4060's, and it is only reachable on the laptop part with TF32 enabled.

| Kernel | n=2048 | n=4096 | n=8192 |
|---|---|---|---|
| FP32 matmul, TF32 **off** | 8.19 | **9.05** | 8.65 TFLOPS |
| FP32 matmul, TF32 **on** | 12.64 | 15.20 | **15.34** TFLOPS |
| BF16 matmul | 30.34 | 30.81 | **30.96** TFLOPS |

**True FP32 peak is 9.05 TFLOPS, 40% below the planning figure.** The 15 TFLOPS in
the plan is the TF32 number.

Throughput at the shape the TCN actually runs — `Conv1d(C→64, k=3)` over
`[128·504, C, 60]`:

| in-channels | FP32 TFLOPS | % of FP32 peak |
|---|---|---|
| 15 (production) | **1.304** | **14.4%** |
| 16 (padded) | 1.392 | 15.4% |
| 64 (inner blocks) | 2.929 | 32.4% |

PCIe, measured at one production minibatch (232 MiB):

| direction | pageable | pinned |
|---|---|---|
| H2D | **8.605 GiB/s** | 9.761 GiB/s |
| D2H | **7.417 GiB/s** | 9.344 GiB/s |

The rollout buffer is plain (pageable) CPU memory, so the pageable column applies.

---

## 4. One PPO update

### 4.1 Wall-clock split

Instrumented, N=504, L=60, minibatch 128, `n_steps` and `n_minibatches` reduced 8×
(32 / 4) so the run was affordable — 16 gradient steps, 32 rollout steps. **M4-CPU**,
synthetic panel. Proportions only; see §0.1.

```
.venv/bin/python scripts/profiling/a2_ppo_update.py update \
  --n-tickers 504 --lookback 60 --n-envs 16 --n-steps 32 --n-minibatches 4 \
  --n-epochs 4 --updates 1 --no-amp --outdir <out>
```

| Phase | s | % of wall |
|---|---|---|
| update forward | 486.58 | 49.2 |
| backward | 408.21 | 41.3 |
| rollout forward | 86.55 | 8.7 |
| optimiser (`Adam.step` + `clip_grad_norm_`) | 0.127 | 0.013 |
| `_batch_obs` (rollout-buffer flatten) | 0.073 | 0.007 |
| env stepping (synthetic — lower bound) | 0.027 | 0.003 |
| host↔device transfers | 0.0012 | 0.0001 |
| **accounted** | 981.6 / 989.5 | **99.2** |

Forward+backward is 99.2% of the update; the optimiser, the buffer flatten and the
environment together are 0.02%. This reproduces the M4 figure quoted in
`CLAUDE.md` (96.2% update fwd+bwd) at 504 tickers rather than 163.

The reduced shape has 32 rollout steps to 16 gradient steps, where production has
256 to 128 — the same 2:1 ratio — so the proportions carry over directly. Per-call:
rollout forward 2.62 s at batch 16, update forward 30.4 s and backward 25.5 s at
batch 128. At the production 257 : 128 : 128 call counts that is 8.6% rollout,
49.7% update forward, 41.7% backward. **The split does not move with rollout
length.**

Two things this measurement does *not* cover, both flagged rather than guessed:

- Production sets `compile: true` on `mlp_regime`, so `model.forward` runs through
  TorchInductor with `mode="reduce-overhead"` (CUDA Graphs) — `runner.py:256-266`.
  Every timing here is eager. The workload is dominated by very large tensors
  rather than kernel launches, so the CUDA-Graph win should be small, but that is
  an expectation, not a measurement; `a2_all.sh` now times a compiled cell.
- `ppo.py:293` calls `torch.cuda.empty_cache()` once per update, immediately after
  the rollout. At §4.4's measured 95%-of-VRAM working point that releases and
  forces re-acquisition of ~7 GiB of allocator segments every update — the
  opposite of what is wanted under memory pressure. Its cost is inside the 0.8%
  unaccounted residual here and has not been isolated on CUDA.

**4060 equivalent: NOT MEASURED** (§0.2). The one 4060 attempt at the full
production shape — `update --n-tickers 504 --updates 1`, instrumented — was still
inside its 128 gradient steps after **76 minutes** when it was stopped, having
finished the rollout in under 4. That bounds the instrumented cost at
≥34 s/gradient step; it does not bound the clean cost.

### 4.2 Host→device and device→host bytes — the central claim

Per-transition observation payload at N=504, L=60 (`panel_env.py:374-390`):

| key | bytes |
|---|---|
| `features` `[60,504,15]` f32 | 1,814,400 |
| `sector_ids`, `portfolio`, `next_day_returns`, `mask`, `regime`, 6 scalars | 6,604 |
| **total** | **1,821,004** |

Measured against that formula on the reduced-shape instrumented run above
(`h2d_GiB_per_update` 5.238, of which 0.868 is an artefact of running on the CPU
device, where `_batch_obs`'s own `torch.tensor(flat)` counts as a transfer;
5.238 − 0.868 = **4.370 GiB measured vs 4.369 GiB predicted — 0.02%**). D2H
measured 0.8703 GiB vs 0.8693 predicted. The formula is therefore trusted at production shape:

| Traffic, per production update | GiB | at 4060 pageable rates |
|---|---|---|
| H2D — rollout (`_obs_to_tensor`, `ppo.py:266`), 257 × 16 obs | 6.97 | 0.81 s |
| H2D — update loop (`mb_obs = {… .to(self.device)}`, `ppo.py:334`), 4 epochs × 4096 | **27.79** | 3.23 s |
| **H2D total** | **34.76** | **4.04 s** |
| D2H — `obs_buf.append({k: v.cpu().numpy()})`, `ppo.py:228`, 256 × 16 obs | 6.95 | 0.94 s |
| **PCIe total** | **41.71** | **4.98 s** |

**The ~27.7 GB figure in R3 is confirmed — for the update loop alone.** Two things
it misses:

1. The rollout adds another **6.97 GiB up**.
2. `ppo.py:266` uploads each observation to the device, and `ppo.py:228` on the
   very next iteration copies the same tensor straight back to the host to store
   it in `obs_buf`. The environment produced that array on the host in the first
   place. **6.95 GiB per update is a pure round trip that never had to happen** —
   the data goes host → device → host, unchanged.

Scale: 488.28 updates × 4.98 s = **40.5 minutes of PCIe time per 2M-step run**.
Against a 2-hour gate that is 34% of the budget spent on transfers alone. It is
worth removing. It is not, however, the reason a run takes days — §4.1 puts
transfers at 0.0001% of update wall clock, and even at 4060 PCIe rates 4.98 s is
a rounding error against a multi-minute update. **R3's "the real cost is data
movement" is wrong; the real cost is arithmetic.**

### 4.3 Ops running >100×/update with a leading dim of 1

`torch.profiler`, `record_shapes=True`, `group_by_input_shape=True`, 3 gradient
steps at N=504, scaled to 128 gradient steps per update. **M4-MPS** — the aten
graph is the same on any backend, which is what this question is about.

```
.venv/bin/python scripts/profiling/a2_ppo_update.py profile \
  --n-tickers 504 --lookback 20 --steps 3 --no-amp --outdir <out>
```

16 op groups qualify. All 16 are one of two harmless things:

| calls/update | op | shape | what it is |
|---|---|---|---|
| 768 | `aten::mul_` | `[1,64]` | Adam's per-parameter loop over the LayerNorm weight/bias tensors |
| 384 each | `addcmul_`, `addcdiv_`, `lerp_`, `sqrt`, `div`, `add_`, `linalg_vector_norm`, `AccumulateGrad`, `detach` ×2 | `[1,64]` | same — 3 such parameters × 128 gradient steps |
| 256 / 128 | `as_strided`, `transpose`, `squeeze`, `UnsqueezeBackward0` | `[1,504,128,3,64]` | `nn.MultiheadAttention`'s internal packed-QKV reshape, metadata only |
| 128 | `aten::view` | `[1,505]` | actor-head concat |

**There is no per-stock Python loop.** `TCNEncoder.forward` flattens to
`[B·N, F, L]` in one `reshape` (`encoders.py:298-301`); `CriticHead` uses
`scatter_add_` rather than a sector loop (`heads.py:224`); `ActorHead` is a single
batched MLP. The signature R3's diagnosis step 2 told us to look for is absent.
The 384-call Adam groups are 3 gradient-step-sized ops, not a hot loop.

### 4.4 Memory

**Host.** The rollout buffer is `4096 × 1,821,004 B = 6.95 GiB` at 504 tickers (6.92 GiB
of it the `features` key alone), and
`ppo.py` holds three copies alive at once at the moment of the flatten:
`obs_buf` (numpy, still referenced), `np.stack(arrays)` inside `_batch_obs`, and
`torch.tensor(flat)`'s copy (`ppo.py:104-114`). Measured on the 4060 box during
the production-shape run: **RSS 15.83 GiB steady** during the update phase and
18 GiB of the 29 GiB WSL allowance in use — consistent with two copies resident
plus 16 per-env panel copies. Peak during the flatten itself is a third copy,
≈20.8 GiB.

**VRAM.** `nvidia-smi` during the same run, N=504 / L=60 / minibatch 128:

| metric | during run | idle baseline |
|---|---|---|
| `memory.used` | **7762–7766 MiB of 8188** | 359 MiB |
| `utilization.gpu` (sm) | 100% | 6% |
| `utilization.memory` (memory controller) | **1–4%** | — |
| `power.draw` | **33–38 W** | 4.8 W |
| `clocks.sm` | 2595 MHz | 210 MHz |

That combination — SM occupancy pinned at 100%, memory-controller utilisation at
1–4%, and only 34 W drawn on a part that will pull three times that when it is
actually computing — is not a compute-bound kernel and it is not a PCIe-bound one
either. It is the profile of a device sitting at the edge of its VRAM: 7.4 GiB of
a 7.8 GiB usable budget, with the allocator (and, on WDDM, the driver's system-memory
fallback) doing the work instead of the SMs. `torch.cuda.max_memory_allocated()`
for the same shape is **NOT MEASURED** (§0.2) — `profile` reports it and is the
first thing to run when the box returns.

---

## 5. Encoder invocation ratio — the size of the caching prize

```
.venv/bin/python scripts/profiling/a2_ppo_update.py dates \
  --panel synthetic --n-tickers 504 --seeds 0,1,2 --outdir <out>
.venv/bin/python scripts/profiling/a2_ppo_update.py dates \
  --panel real --seeds 0,1,2 --outdir <out>
```

`TCNEncoder.forward` is invoked once per model forward; each invocation encodes
`B × N` `(date, stock)` sequences. Per update:

- rollout: `(256 + 1) × 16 = 4112` transitions, each `N` sequences
- update: `4 epochs × 4096 = 16,384` transitions, each `N` sequences
- **20,496 transition-encodings per update = 10,329,984 stock-sequences at N=504**

Counted invocations agree exactly: the instrumented run at `n_steps=32`,
`n_minibatches=4` reported `encoder_calls_per_update` 49 (= 33 rollout + 16 update)
and `encoder_sequences_per_update` 1,298,304 (= 33·16·504 + 16·128·504).

| | synthetic, N=504 | real panel, N=143 |
|---|---|---|
| encoder sequences / update | 10,329,984 | 2,930,928 |
| distinct dates touched / update (seeds 0/1/2) | 1537 / 1585 / 1681 | 1414 |
| distinct `(date, stock)` pairs / update | 774,648 / 798,840 / 847,224 | 202,202 |
| **redundancy ratio** | **13.3 / 12.9 / 12.2** (mean **12.8×**) | **14.5×** |
| whole panel encoded once | 1972 × 504 = 993,888 | 1972 × 143 = 281,996 |

Two different prizes, and they are very different sizes:

- **Within one update**, with a trainable encoder, deduplicating `(date, stock)`
  saves **12.8×** of encoder work.
- **Across a run**, with a frozen pre-trained encoder cached over the whole panel,
  the encoder is computed **once**: 993,888 sequences against
  488.28 × 10,329,984 = 5.044 G — a **5,075× reduction**, and the per-update TCN
  cost goes to zero.

The `(date, stock)` embedding is a pure function of the panel — nothing in
`_forward_shared` mixes portfolio, cash or action into `self.encoder(features)`
(`actor_critic.py:190`). The cache is sound. Note that only the *encoder* is
cacheable: `CrossStockAttention` and both FiLM blocks sit downstream of it and are
also pure functions of the panel, but the actor and critic heads take `portfolio`,
`t_frac` and the NAV summary, so they are not.

---

## 6. Achieved FLOPS

### 6.1 Corrected analytic count

```
.venv/bin/python scripts/profiling/a2_ppo_update.py analytic --n-tickers 504 --lookback 60
```

Counting 2 FLOP per MAC; causal padding keeps every conv output length at `L`.

| | MACs | FLOPs |
|---|---|---|
| TCN block 0 (15→64 + 64→64 + 1×1 downsample) | 0.968 M | |
| TCN block 1 (64→64 ×2) | 1.475 M | |
| TCN block 2 (64→64 ×2) | 1.475 M | |
| **per stock-sequence** | **3.917 M** | **7.834 M** |

| Level | Value |
|---|---|
| sequences per gradient step (128 × 504) | 64,512 |
| TCN, forward | 505.4 GFLOP |
| cross-stock attention (504×504, d=64), forward | 10.44 GFLOP |
| actor + critic heads, forward | 0.55 GFLOP |
| **forward per gradient step** | **516.3 GFLOP** |
| **fwd+bwd per gradient step (×3)** | **1549.0 GFLOP** |
| per update (×128) | 198.3 TFLOP |
| **2M env steps (×488.28)** | **96.8 PFLOP** |
| of which the TCN encoder | **94.8 PFLOP — 97.9%** |
| 8M env steps, the configured `total_steps` | 387.3 PFLOP |

The plan's 3.3 MFLOP/sequence is close to the *MAC* count (3.92 M) — it counts a
multiply-accumulate as one FLOP and drops attention. The convention matters here:
**40 PFLOP → 96.8 PFLOP is the difference between a reachable gate and an
unreachable one.**

The auxiliary-head variant (`model=mlp_regime_aux train=ppo_aux`) does not change
this budget: `ReturnPredictionHead` is a 64→32→1 per-stock MLP on the same `z`
(`heads.py:108-113`), which adds ~0.01% of the forward and no encoder work.

`torch.profiler`'s `with_flops=True` is not usable as a cross-check: it does not
instrument `aten::conv1d` / `aten::_mps_convolution`, and reported only 25.3 GFLOP
per gradient step at N=504 / L=20 — the linear layers alone.

### 6.2 Achieved fraction

**On the 4060: NOT MEASURED** (§0.2). What is measured is the ceiling this
workload can reach on that part, which is the more useful half of the question:

| Reference rate | measured | 96.8 PFLOP takes |
|---|---|---|
| `Conv1d(15→64,k=3)` at the production shape, FP32 | 1.304 TFLOPS | 20.6 h |
| `Conv1d(64→64,k=3)` at the production shape, FP32 | 2.929 TFLOPS | 9.2 h |
| FP32 matmul peak | 9.05 TFLOPS | 3.0 h |
| TF32 matmul peak | 15.34 TFLOPS | 1.75 h |
| BF16 matmul peak | 30.96 TFLOPS | 0.87 h |

**94.1% of the TCN's MACs are 64→64 convolutions** (block 0's `conv2` plus all of
blocks 1 and 2); only 4.4% is the 15-channel input conv and 1.5% the 1×1
downsample. So **2.93 TFLOPS is a generous upper bound on what the current graph
can sustain in FP32** — 32% of peak, and 9.2 hours for 2M steps *if nothing else
cost anything*. Under bf16 autocast (which production enables on CUDA,
`ppo.py:203`) the conv rate is **NOT MEASURED**; `a2_hw_probe.py` now measures
`in_ch_{15,16,64}_bf16` and `a2_all.sh` runs it.

---

## 7. Scaling curves

### 7.1 4060 — NOT MEASURED (§0.2)

`a2_all.sh` step 4 produces this table:
`scale --tickers 163,300,504,645 --lookbacks 20,30,60 --iters 5`.

### 7.2 M4-MPS, uninstrumented, production minibatch 128

```
.venv/bin/python scripts/profiling/a2_ppo_update.py scale \
  --tickers 163,300,504,645 --lookbacks 60,30,20 --iters 5 --no-amp --outdir <out>
```

Seconds per gradient step (128 gradient steps = one update):

| N | L=60 | L=30 | L=20 | minibatch `features` at L=60 |
|---|---|---|---|---|
| 163 | 1.816 / 1.823 | 0.942 | 0.643 | 71.6 MiB |
| 300 | **7.447** | 1.813 | 1.246 | 131.8 MiB |
| 504 | **87.72** | 3.624 | 2.438 | 221.5 MiB |
| 645 | **OOM** — `MPS backend out of memory (MPS allocated 26.99 GiB, max allowed 30.19 GiB)` | **23.42** | 2.731 | 283.4 MiB |

Sorted by the activation working set `N × L` rather than by either axis, the
same nine points collapse onto one curve — which is the point:

| N × L | cell | s / gradient step | µs per (ticker·day) |
|---|---|---|---|
| 9,000 | 300 × 30 | 1.813 | 201 |
| 9,780 | 163 × 60 | 1.816 | 186 |
| 10,080 | 504 × 20 | 2.438 | 242 |
| 12,900 | 645 × 20 | 2.731 | 212 |
| 15,120 | 504 × 30 | 3.624 | 240 |
| 18,000 | 300 × 60 | 7.447 | **414** |
| 19,350 | 645 × 30 | 23.42 | **1,210** |
| 30,240 | 504 × 60 | 87.72 | **2,901** |
| 38,700 | 645 × 60 | OOM | — |

Below `N·L ≈ 15,000` the cost is flat at **~210 µs per (ticker × lookback-day)**
and cleanly linear. Above it the same unit costs 2×, then 6×, then 14×, then the allocator
gives up. Ticker count and lookback are interchangeable here: **300 × 30 and
163 × 60 cost the same to within 0.2%.**

Three things this settles.

**Cost is not linear in ticker count — there is a cliff.** At L=20 the scaling is
close to linear (163→504 is 3.09× the tickers and 3.79× the time). At L=60 it is
not: 163→300 is 1.84× the tickers and **4.1× the time**; 163→504 is 3.09× the
tickers and **48× the time**.

The cliff is a working-set cliff. The TCN materialises `[B·N, 64, L]` activations and keeps them for
backward: at N=504, L=60, minibatch 128 that is 64,512 × 64 × 60 × 4 B = **945 MiB
per saved tensor in fp32**, and the three blocks keep roughly fifteen of them live
(two convs, two LayerNorms, two GELUs, dropout and the residual per block) — about
**13.8 GiB**, against 24 GiB of unified memory shared with everything else. On the
4060 the same tensors are bf16 under autocast, so 472 MiB each, ~6.9 GiB, plus the
232 MiB fp32 input minibatch, the parameters and cuDNN workspace: **≈7.4 GiB
against a 7.8 GiB budget** — which is what §4.4's `memory.used` of 7762–7766 MiB
actually measures. **The two machines hit the same wall for the same reason, and
the arithmetic predicts the 4060's measured VRAM to within 5%.**

**L=30 is not a 2× saving; at 504 tickers it is a 24× saving** (87.72 s → 3.62 s),
because it is the difference between fitting and not fitting. L=20 buys a further
1.49×. At 645 tickers L=30 is *itself* over the cliff (23.4 s) and only L=20
recovers the linear regime. This is the answer to the L=30 question the earlier
Mac attempt never completed — with the caveat that it is measured on M4-MPS, and
**the 4060's cliff sits at a lower `N·L` than the M4's**, because its budget is
7.8 GiB against ~20. Scaling the measured M4 knee (`N·L ≈ 15,000` at ~20 GiB
usable) by the ratio of budgets puts the 4060's knee near `N·L ≈ 6,000` — which
would place 504 × 60 (30,240) five times past it, and is consistent with the
7.76 GiB / 34 W / 1–4% memory-controller signature of §4.4. That extrapolation is
**not a measurement**; `a2_all.sh` step 4 replaces it.

For reference, M4-**CPU** at 163 / L=60 is 10.95 s per gradient step — 6× slower
than the same box's MPS path.

---

## 8. Memory ceiling

### 8.1 VRAM — partially measured

| Configuration | VRAM | Evidence |
|---|---|---|
| N=504, L=60, mb=128 | **7762–7766 MiB of 8188 used** — fits, but with ~400 MiB of headroom and at 34 W | `nvidia-smi` during the run, §4.4 |
| idle baseline (desktop, LM Studio idle, Edge) | 359 MiB | `nvidia-smi` after the run |
| the (N, minibatch) grid at which it OOMs | **NOT MEASURED** | `a2_all.sh` step 5 |

Two things follow from the measured point without further data. Activation
memory for the TCN scales as `minibatch × N × L`: N=504/L=60/mb=128 sits at
~95% of usable VRAM, so **mb=256 at 504 tickers cannot fit**, and neither can
645 tickers at mb=128 (1.28× the working set). This is consistent with the
`CLAUDE.md` note that minibatches OOM at 512 — measured there at 163 tickers,
where the working set is 3.1× smaller, i.e. the same ~7–8 GiB wall.

Note that on Windows/WDDM a CUDA process at the VRAM limit does not necessarily
raise `OutOfMemoryError`: the driver's system-memory fallback spills over PCIe
instead, which degrades by an order of magnitude and reports as 100% SM
utilisation at low power — precisely the §4.4 signature. **The ceiling sweep must
therefore report time as well as success/failure**, which `cmd_ceiling` does.

### 8.2 Host RAM — measured

| N | rollout buffer | resident during update (measured) | peak during `_batch_obs` |
|---|---|---|---|
| 163 | 2.25 GiB | — | ~6.7 GiB |
| 504 | **6.95 GiB** | **15.83 GiB RSS** (4060 box, production shape) | ~20.8 GiB |
| 645 | 8.89 GiB | ~20.3 GiB | ~26.7 GiB |

`09_…md` §4.5 says "the host-side rollout buffer is 6.9 GB at 504 names, so ≥16 GB
system RAM is required." The buffer figure is right; the requirement is not.
`ppo.py:104-114` holds `obs_buf`, the `np.stack` intermediate and the final
`torch.tensor` copy simultaneously, so **≥24 GiB is required at 504 tickers and
645 tickers will not fit in the box's 29 GiB WSL allowance.**

---

## 9. The three R3 candidate fixes, sized

Not implemented. Sizing is against §6.1's corrected 96.8 PFLOP and the §3 rates.

### (a) Panel resident on GPU; observations become date indices

**Removes:** all 41.7 GiB/update of PCIe traffic (§4.2) → **40.5 min per 2M-step
run**; the 6.95 GiB host buffer and its three-copy peak (§8.2) → host RAM
requirement drops from ≥24 GiB to ~1 GiB; and `_batch_obs`'s stack-and-copy.

**Costs:** the train panel resident in VRAM is `1972 × 504 × 15 × 4 B` = **56.9 MiB
fp32 / 28.4 MiB fp16** — against a budget where §4.4 says only ~400 MiB is free at
the production shape. Affordable. (R3's "~150 MB fp32" presumably counts all
three splits and more columns; for the training tensor the figure is 57 MiB.)

**Does not remove:** any arithmetic. §4.1 measures transfers at 0.0001% of update
wall clock. **Expected speedup on update wall clock: ~1.0×.** The win is the
40 min of PCIe across a run and the host-RAM ceiling, not the update itself.

**What must change:** `PanelTradingEnv._build_obs` would have to emit a date index
instead of `features`, which changes the observation space, `ActorCritic._forward_shared`,
`_batch_obs`, `_obs_to_tensor`, and every test that constructs an obs dict. The
env would need a handle on device-resident state, which it currently does not have.
This is the largest of the three changes and buys the least wall clock.

### (b) `torch.autocast` + pad input channels 15 → 16

**Already half-done:** `ppo.py:203-207` already wraps both rollout and update in
`torch.autocast(bfloat16)` on CUDA. The remaining item is the padding.

**Measured, FP32:** `Conv1d(15→64)` 1.304 TFLOPS vs `Conv1d(16→64)` 1.392 —
**+6.8% on that one convolution**. That convolution is 172,800 of the TCN's
3,916,800 MACs, i.e. **4.4% of the encoder**, so the whole-model effect in FP32 is
**+0.3%**. The plan's "typically 2–3×" is not supported at this shape.

**Under bf16 the tensor-core argument may hold** — a channel count not divisible by
8 forces cuDNN off its fastest NHWC kernels. That measurement is **NOT MEASURED**;
`a2_hw_probe.py` now includes `in_ch_{15,16,64}_bf16` and `a2_all.sh` runs it.
Even at the most favourable outcome the ceiling is the 4.4% of MACs that first
convolution represents, unless the padding also unlocks a faster kernel for the
64→64 blocks — which it cannot, since 64 is already aligned.

**What must change:** one line in `TCNEncoder.__init__` and a matching pad in
`forward`; `FeatureNormalizer`'s buffers would need a 16th zero channel. Small.

### (c) Cache encoder outputs

**Removes 97.9% of all arithmetic** (§6.1: 94.8 of 96.8 PFLOP). With the encoder
pre-trained, frozen and its `[T, N, 64]` output cached, PPO's per-gradient-step
cost falls from 1549 GFLOP to the attention + heads' **33 GFLOP** — a **47×**
reduction. For 2M steps: **96.8 PFLOP → 2.06 PFLOP.**

It also removes the memory cliff that §7.2 measures, because the `[B·N, 64, L]`
activations disappear: the resident state becomes a `[1972, 504, 64]` fp16 table
= **121 MiB**, and the per-minibatch payload becomes `[128, 504, 64]` fp16
= **7.9 MiB** instead of 232 MiB. Minibatch 512 becomes feasible, and 645 tickers
stop being a memory question.

Combined with (a) — which becomes nearly free once the cache exists, since the
"panel" to keep resident is now the 121 MiB embedding table — PCIe traffic per
update drops from 41.7 GiB to ~1 GiB.

**What must change:** an R4-style supervised pre-training stage for the encoder;
a cache built per walk-forward window from **train-split rows only** (a cache
built over the full panel would be a leak — A1's territory); `ActorCritic` gaining
a path that takes `z` directly instead of `features`; and the acceptance that PPO
no longer trains the representation. That last point is a modelling decision, not
a compute one, and it is exactly the decision R4 exists to inform.

---

## 10. Verdict on the R3 gate

> **Gate: 2M steps in under 2 hours on the 4060, conditional on encoder caching
> landing.**

2 hours for 96.8 PFLOP requires a sustained **13.4 TFLOPS**. That is 148% of the
4060 Laptop's measured FP32 peak (9.05 TFLOPS) and 87% of its TF32 peak. The
production graph achieves at most 2.93 TFLOPS in FP32 at its own shapes (§3), and
that is before the memory cliff of §7.2, which at 504 tickers costs another order
of magnitude on the one machine where it has been measured end to end.

**Without the cache the gate is unreachable, by a factor of at least 5 and
plausibly 20.** The floor the plan quotes — "~44 minutes at 100% utilisation" —
rests on the 40 PFLOP figure and the 15 TFLOPS figure. Both are wrong: at
96.8 PFLOP and 9.05 TFLOPS the 100%-utilisation floor is **3.0 hours**, so
*2 hours is not merely optimistic, it is below the theoretical floor.* No amount
of fixing data movement changes this; §4.2 shows the entire PCIe bill is 40 min
per run and §4.1 shows transfers are 0.0001% of an update.

**With the cache the gate is reachable, with room.** 2.06 PFLOP at even
1 TFLOPS sustained is 34 minutes; at the 2.93 TFLOPS the attention/head kernels
should comfortably beat (they are matmuls, not 15-channel convs) it is under
15 minutes. The remaining costs are the rollout (which still runs the *cached*
forward, so it is cheap), the environment (0.003% — §4.1), and PCIe (which the
cache shrinks by ~40×).

**Conditions, stated plainly:**

1. The gate holds **only** with encoder caching. R3's own conditional is correct;
   its arithmetic for the unconditional case is not.
2. The gate should be **restated against 96.8 PFLOP and 9.05 TFLOPS**, not
   40 PFLOP and 15 TFLOPS.
3. It cannot be *verified* until §0.2's four measurements exist. Nothing here
   claims the cached path has been timed on the 4060 — it has not.
4. The 8M-step `total_steps` in `configs/train/ppo_baseline.yaml` is 4× the gate's
   workload, i.e. 387 PFLOP uncached. Any affordability claim made about 2M steps
   does not transfer to the configured run.
5. **A Phase 1 A/B is not affordable at the current data state regardless**, and
   not for a compute reason: §1.1 shows 71.6% of the ticker axis is zeros, so
   whatever is measured would be measured mostly on padding. R1 (one rebuilt
   panel) has to land before any A/B is worth its wall clock.

---

## 11. Reproduction

```bash
# hardware envelope (§3)
ssh -o BatchMode=yes jashm@192.168.1.7 \
  'wsl.exe -e bash /mnt/d/trading-bot-01/scripts/win_bootstrap.sh run scripts/profiling/a2_hw_probe.py'

# everything still owed on the 4060 (§0.2), unattended, results under outputs/a2/
ssh -o BatchMode=yes jashm@192.168.1.7 \
  'wsl.exe -e bash /mnt/d/trading-bot-01/scripts/profiling/a2_all.sh'
ssh -o BatchMode=yes jashm@192.168.1.7 \
  'wsl.exe -e bash /mnt/d/trading-bot-01/scripts/profiling/a2_run.sh cat scale.json'

# analytic FLOPs (§6.1)
.venv/bin/python scripts/profiling/a2_ppo_update.py analytic --n-tickers 504 --lookback 60

# encoder redundancy (§5)
.venv/bin/python scripts/profiling/a2_ppo_update.py dates --panel synthetic --n-tickers 504 --seeds 0,1,2
.venv/bin/python scripts/profiling/a2_ppo_update.py dates --panel real --seeds 0,1,2

# phase split + exact byte accounting (§4.1, §4.2)
.venv/bin/python scripts/profiling/a2_ppo_update.py update \
  --n-tickers 504 --n-steps 32 --n-minibatches 4 --updates 1 --no-amp

# op-shape forensics (§4.3)
.venv/bin/python scripts/profiling/a2_ppo_update.py profile \
  --n-tickers 504 --lookback 20 --steps 3 --no-amp

# scaling curves (§7.2)
.venv/bin/python scripts/profiling/a2_ppo_update.py scale \
  --tickers 163,300,504,645 --lookbacks 60,30,20 --iters 5 --no-amp

# §1.1
.venv/bin/python -c "
import sys; sys.path.insert(0,'src')
import polars as pl
from trader.data.universe import active_tickers, all_tickers
act=set(active_tickers())
t=set(pl.read_parquet('data/panels/train.parquet', columns=['ticker'])['ticker'].unique().to_list())
print(len(t), len(act), len(act&t), len(t-act), len(all_tickers()))"
```

---

## 12. Raw artefacts

| File | Contents |
|---|---|
| `scripts/profiling/a2_hw_probe.py` | device properties, FP32/TF32/BF16 matmul TFLOPS, `Conv1d` TFLOPS at the production shape (FP32 and bf16), PCIe H2D/D2H pageable and pinned |
| `scripts/profiling/a2_ppo_update.py` | subcommands `update` / `profile` / `scale` / `ceiling` / `dates` / `analytic`; every one writes its JSON to `--outdir` as well as stdout |
| `scripts/profiling/a2_run.sh` | detachable launcher + poller for the box (`start` / `poll` / `log` / `cat` / `ps`) |
| `scripts/profiling/a2_all.sh` | the whole remaining 4060 suite in cost order |

No file under `src/`, `configs/` or `tests/` was modified. `git status` at the end
of this unit shows only `audit/` and `scripts/profiling/` as new.

---

## The three findings that most change what should be done next

**1. The compute budget is 2.4× larger than planned and the hardware is 40%
slower than planned, so the R3 gate is below its own theoretical floor.**
96.8 PFLOP, not 40; 9.05 TFLOPS FP32, not 15. At 100% utilisation 2M steps is
3.0 hours, so "2M steps in under 2 hours" cannot be met by any amount of
engineering on the current graph. It can only be met by removing work — which
makes encoder caching not one of three candidate fixes but the whole of the fix:
the TCN is **97.9%** of all arithmetic, and caching it cuts 96.8 PFLOP to 2.06.
Fixes (a) and (b) are worth 40 minutes per run and ~0.3% respectively.

**2. R3's diagnosis names the wrong bottleneck.** Data movement is real and
larger than stated — 34.8 GiB H2D plus 6.95 GiB of pure host→device→host round
trip per update, 41.7 GiB total, 40 min per 2M-step run — but it is 0.0001% of
update wall clock. There is no per-stock Python loop; the aten graph is clean.
What the 4060 actually shows, on `nvidia-smi` during a real production-shape
update, is a **memory-ceiling** signature: 7.76 GiB of a 7.8 GiB usable budget,
memory-controller utilisation at 1–4%, 34 W on a part that pulls three times that
when it computes. The scaling curve measured end to end (**on the M4** — the
4060's own curve is the first thing `a2_all.sh` produces) has a *cliff*, not a
slope: at 504 tickers, L=60 costs **24×** what L=30 costs, and the nine cells
collapse onto a single function of the working set `N × L` that is flat below
~15,000 and blows up above it. Anyone chasing PCIe will find 5 seconds per update;
the cliff is worth tens of seconds per gradient step. **Cutting the lookback to 30
is the cheapest intervention available — a one-line config change — and should be
sized as a first-class decision, not a footnote.**

**3. None of this is worth spending on the current panels.** `active_tickers()`
returns 504, the panels hold 163, and only 143 overlap — so `PanelTradingEnv`
allocates a `[T, 504, 15]` tensor of which **71.6% of the ticker axis is
structurally zero** and every measurement above, including the 96.8 PFLOP, is
mostly arithmetic on padding. Fixing compute before R1 rebuilds the panel would
optimise a workload that will not exist. The correct order is R1, then re-measure,
then decide between L=30 and encoder caching with real shapes.
