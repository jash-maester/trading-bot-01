# 04 — Models

## Status

**PARTIAL** — last verified against `60ba1a6` (the A0/A1/A2 audit reports,
2026-09-05). The TCN encoder, the actor/critic heads, the masked softmax and the
hetero GNN are all implemented; the GNN is **not** the default config
(`mlp_regime` is) and has never produced an interpretable result.

Known broken:

- **`graph.py:334` gives 87.6% of GNN minibatch elements the wrong adjacency**
  — the per-batch mask is taken from element 0 and tiled (A1 §6).
- **`num_sectors` defaults to 8** in three places while the universe now has
  **14** sectors; any panel rebuilt from the current universe crashes the critic
  (A1 §7.6).
- **Dropout and DropEdge never take effect during training** and *are* active
  during evaluation — the mode switching in `ppo.py` is inverted (A1 §7.1).
- The TCN as configured is **3 blocks, not 4**, dilations 1/2/4, and
  `embed_dim: 128` in the config is dead.
- **`torch.compile` no longer reaches the hot path** for `ActorCritic`
  (A1 §7.9).

Every hardware figure in this document has been rewritten against the RTX 4060
8 GB. The RTX 5090 does not exist.

---

Three components, composed bottom-up:

1. **Per-ticker time-series encoder.** Produces an embedding
   `h_i ∈ R^d` per stock from its feature window.
2. **Graph attention network** over stocks and sectors. Produces
   context-refined embeddings `z_i ∈ R^d`.
3. **Actor-critic heads** that consume `z_i` and portfolio state.

Sizes are starting points. **There is no headroom.** On the target RTX 4060
8 GB, the current graph at N=504 / L=60 / minibatch 128 sits at
**7,762–7,766 MiB of 8,188 MiB** — ~95% of usable VRAM — drawing 34 W with
memory-controller utilisation at 1–4%, which is the signature of a device at
its memory ceiling rather than one computing (A2 §4.4, §8.1). Minibatch 256 at
504 tickers cannot fit, and neither can 645 tickers at minibatch 128. Resist the
urge to go big; there is nowhere to go.

## On "the RL agent adjusts the graph weights"

The original idea was: have the RL agent's action include edge-weight
updates for a fully connected stock graph. This has two problems:

- **Action dimensionality explodes.** With N=150 stocks fully connected,
  you're looking at ~10⁴ edge weights as continuous actions per step.
  PPO won't find anything.
- **Credit assignment is nearly impossible.** The reward signal is too
  sparse to teach meaningful graph structure.

Cleaner design that preserves the original intent:

> The graph is a **learnable-attention GNN** trained end-to-end by
> backprop from the policy and value losses. After training, edge
> attentions encode learned cross-stock and cross-sector relations.
> That satisfies "the agent learns which stocks affect which" without
> turning it into an RL action.

## Per-ticker encoder

Default: **Temporal Convolutional Network** (TCN) — dilated causal 1D
convs. Simpler, faster, and more sample-efficient than a transformer
for short lookbacks (60 days) and small F.

```
input:  [B, N, L, F]     # B episodes, N tickers, L=60, F features
reshape: [B*N, F, L]
TCN:  [B*N, d, L]  →  take last step → [B*N, d]
reshape: [B, N, d]
```

**Specified:** 4 blocks, kernel 3, dilations `{1, 2, 4, 8}`, 64 channels,
LayerNorm, GELU, dropout 0.1.

**As configured and run:** `configs/model/mlp_regime.yaml:18-21` sets
`num_channels: [64, 64, 64]` — **3 blocks**, dilations `{1, 2, 4}`
(`encoders.py:273-274` truncates the dilation list to the channel list). Each
`TemporalBlock` holds `conv1` and `conv2` (`encoders.py:234-235`) plus a 1×1
`downsample` when the channel count changes (`:239`), which happens only on
block 0 (15→64). **Seven `Conv1d` layers in total.**

`embed_dim: 128` in the same config is **dead whenever `num_channels` is set**
(`encoders.py:273`) — the model width is **64**, not 128 (A2 §1).

> **DEFECT — `model.encoder` is a dead config key.** All five
> `configs/model/*.yaml` declare `encoder: tcn`, and `training/runner.py:177-197`
> hardcodes the TCN. `encoder: gru` would be silently ignored (A0 §3.1). The
> Transformer alternative below is not reachable from config.

Alternative behind a config flag: small Transformer with RoPE, same
`d=64`, 2 heads, 2 layers. Useful for ablation.

**Weight sharing**: one encoder shared across all tickers. Sector
identity is injected later in the GNN.

## Hetero graph

Nodes: `N` stock nodes + `S` sector nodes. **S is 14**, not 6–8
(`universe.py:236-252`, `SECTOR_IDS` runs 1..14).

> **DEFECT — `num_sectors` defaults to 8 in three places and nothing sets it
> from the universe.** `heads.py:181` (`CriticHead.__init__`),
> `actor_critic.py:30` (`ModelConfig.num_sectors`) and `graph.py:82`
> (`GraphConfig.num_sectors`) all default to 8; `configs/model/gnn_v1.yaml:13`
> and `gnn_intra_only.yaml:13` hardcode 8 with a comment claiming it "matches
> SECTOR_IDS in universe.py", which it has not since `a5d8b79`.
> `training/runner.py:178-196` never sets it.
>
> `CriticHead.forward` (`heads.py:218-224`) does `scatter_add_` at index
> `sector_id − 1`, so any `sector_id > 8` is out of bounds. Verified directly
> (A1 §7.6): `sector_ids=[1, 2, 9]` raises
> `RuntimeError: index 8 is out of bounds for dimension 1 with size 8`. The
> largest live id, `sector_id = 14` (telecom), gives **`index 13 is out of
> bounds`**. In `HeteroGNN` the same id does not raise but silently points at
> the *next batch element's* sector-0 node (`graph.py:293-302` offsets by
> `b*S`), which is worse.
>
> This has not fired only because the panels on disk are stale and top out at
> `sector_id == 8` (A1 §7.7). **It fires the moment R1 rebuilds a panel.**

Edges (three relations):
- `stock → stock (same sector)` — dense intra-sector.
- `stock ↔ sector` — membership.
- `sector ↔ sector` — dense inter-sector.

Edge construction:
- Intra-sector: all pairs `(i, j)` where `sector(i) == sector(j)`.
- Inter-sector: all pairs of distinct sector nodes.
- Membership: `(stock_i, sector(i))`.

Initial edge features (optional but helpful as a prior):
- `corr_60d(i, j)` for stock-stock edges, clipped to `[-1, 1]`, updated
  weekly from the training panel.
- Learned embedding for sector-sector edges.

### Hetero GAT

Use `torch_geometric.nn.HeteroConv` wrapping `GATv2Conv` per relation.

```python
HeteroConv({
    ("stock", "same_sector", "stock"): GATv2Conv(d, d, heads=2, edge_dim=1),
    ("stock", "in", "sector"):         GATv2Conv((d, d), d, heads=2, add_self_loops=False),
    ("sector", "contains", "stock"):   GATv2Conv((d, d), d, heads=2, add_self_loops=False),
    ("sector", "relates_to", "sector"): GATv2Conv(d, d, heads=2),
})
```

2–3 HeteroConv layers with residual connections, LayerNorm, GELU.
Output: refined stock embeddings `z_i ∈ R^d`.

Masking untradeable stocks: zero out their features before the GNN and
exclude them from attention by setting the attention logit to `-inf`
via a custom edge filter. Their `z_i` is irrelevant because their
action logits will be masked anyway, but filtering keeps their noise
out of neighbors' attention.

> **DEFECT — `graph.py:334` builds the graph topology from batch element 0 and
> tiles it across the batch.** Node *features* are masked per element
> (`graph.py:326-327`, correct). Graph *edges* are not: `graph.py:332-338` takes
> `sector_ids[0]` **and `tradeable_mask[0]`**, builds one edge set, and tiles it.
> The comment's premise — "the universe is fixed, so sector assignments are the
> same for every batch element" — is true of `sector_ids` (measured: 0 differing
> pairs across 1,971 consecutive days) and **false of `tradeable_mask`**, which
> is a function of the date. The two are handled by the same line.
>
> Two failure modes (A1 §6.1): a name tradeable in element *b* but not in
> element 0 gets **no edges at all** and is isolated; a name untradeable in *b*
> but tradeable in element 0 **keeps its edges** while its feature vector has
> already been zeroed, so a zero vector is attended over and dilutes every
> neighbour in its sector.
>
> Measured incidence at the real PPO minibatch size of 128 (A1 §6.3):
>
> | | `data/panels` | `data/panels_kite` |
> |---|---:|---:|
> | batch elements whose tradeable set differs from `mask[0]` | **87.60%** | **94.27%** |
> | mean Hamming distance to `mask[0]` | 9.74 names | 17.54 names |
> | — tradeable but edgeless | 4.65 | 8.48 |
> | — untradeable but still edged | 5.09 | 9.07 |
>
> The embedding error is O(1) and propagates to correctly-masked neighbours. In
> PPO the same transition's `log_prob` swings 0.68 nats depending on which
> sample lands at index 0 — an importance ratio of **1.98** against
> `clip_coef = 0.2` (A1 §6.4). `ppo.py:209-217` documents "GNN `clip_frac`
> collapses to 1.000 from the very first update" and blames dropout; forcing
> `eval()` removed that term and left this one. **Every `gnn_v1` /
> `gnn_intra_only` result is uninterpretable.**
>
> No test catches it: every mask in `tests/unit/test_graph.py` is constant along
> the batch dimension (`:39`, `:210-211`, `:236-237`).

DropEdge (p=0.1) during training for regularization.

> **DEFECT — DropEdge and every dropout are dead during training and live
> during evaluation.** `ppo.py:218` calls `self.model.eval()` at the top of
> every update iteration, before the rollout; `ppo.py:445` calls
> `self.model.train()` only **after** the whole update loop. Nothing switches
> back in between, so the gradient updates at `ppo.py:342-398` also run in eval
> mode. `tcn.dropout: 0.1`, `graph.dropout: 0.1` and `graph.drop_edge_prob: 0.1`
> **never take effect**. Meanwhile `_evaluate_split` (`runner.py:397-440`) runs
> after `train()` restored train mode and never calls `eval()`, so **val and
> test metrics are measured with dropout active** — measured spread over 8
> identical forwards: 1.6e-02 with dropout on, exactly 0 with it off (A1 §7.1).
> Live trading (`paper_run.py:209`) and the shuffled-ticker arm
> (`walk_forward.py:492`) *do* call `eval()`, so the shuffle check and the real
> arm are evaluated under different stochasticity — the comparison is confounded
> by construction.

## Actor-critic heads

Inputs:
- `z ∈ R^{N x d}` — GNN output.
- `portfolio ∈ R^{N+1}` — current weights including cash.
- Scalar state: `nav_scaled, cash_scaled, t_frac`.

### Actor (policy) head

```
p_i = MLP([z_i, portfolio_i])       # [N, 1] raw logits
p_cash = MLP([mean_pool(z), cash_scaled])  # scalar logit for cash
logits = concat([p_cash, p_1, ..., p_N])     # shape [N+1]
```

PPO outputs a **Gaussian over these logits** (mean = head output, diag
std = learned parameter per dim, initialized small). The env applies
the masked softmax. This keeps the policy family standard and stable.

### Critic (value) head

```
g = mean_pool(z) concat portfolio_summary_stats
V = MLP(g) -> scalar
```

Portfolio summary stats: mean weight, max weight, gini of weights,
sector exposure entropy, turnover from last action.

## Parameter budget (starting point)

- Encoder: ~50k params
- Hetero GAT (2 layers, d=64, 2 heads): ~60k params
- Heads: ~30k params
- Total: ~150k params

**The parameter count is not the price.** The observation is. Measured and
derived by A2 §6.1 (`scripts/profiling/a2_ppo_update.py analytic --n-tickers 504
--lookback 60`), counting 2 FLOP per MAC:

| Level | Value |
|---|---|
| per stock-sequence (7 `Conv1d` layers, L=60, F=15, d=64) | **7.83 MFLOP** |
| sequences per gradient step (minibatch 128 × 504 names) | 64,512 |
| forward per gradient step (TCN 505.4 + attention 10.4 + heads 0.55 GFLOP) | 516.3 GFLOP |
| **fwd+bwd per gradient step** (×3) | **1,549 GFLOP** |
| per update (×128 gradient steps) | 198.3 TFLOP |
| **2M env steps** (×488.28 updates) | **96.8 PFLOP** |
| of which the TCN encoder | **94.8 PFLOP — 97.9%** |
| 8M env steps, the configured `total_steps` | 387.3 PFLOP |

Against the RTX 4060 8 GB's **measured** envelope (A2 §3): FP32 matmul peak
**9.05 TFLOPS** — the 15 TFLOPS figure in earlier planning is the card's **TF32**
number (15.34), not FP32. At the production `Conv1d` shapes the graph sustains
1.30 TFLOPS on the 15-channel input conv and 2.93 TFLOPS on the 64→64 inner
convs, which are 94.1% of the MACs.

So 2M steps is **9.2 hours at 2.93 TFLOPS**, and **3.0 hours even at a
100%-utilisation FP32 peak that this graph cannot reach** (A2 §6.2). Scaling `d`
to 128 or GNN depth to 3 multiplies a budget that is already the binding
constraint. Do not scale anything until encoder caching lands — caching the TCN
cuts 96.8 PFLOP to 2.06 (A2 §9c, §10).

**Not measured on the 4060:** wall clock per gradient step, achieved FLOPS,
kernel table, scaling curves and the OOM ceiling — the training box refused SSH
partway through A2 (A2 §0.2). `scripts/profiling/a2_all.sh` runs all of them
unattended.

## Alternative architectures to have ready for ablation

- **No-graph baseline**: encoder → mean pool → MLP policy. Must run
  before the GNN so the GNN's contribution is measurable.
- **Shared-sector GNN**: only the intra-sector edges, no inter-sector.
- **Fully connected no-hierarchy**: all stocks, no sector nodes.
- Report all four side-by-side in MLflow.

## What the "second agent" would look like (v2, not v1)

If after v1 the data shows clear under-specialization (e.g. allocation
is reasonable but timing is bad), you can add a **hierarchical
two-level policy**:

- Level 1 (slow): sector allocation, runs weekly, outputs a sector
  budget vector.
- Level 2 (fast): within-sector name picking, daily, constrained by
  level 1.

Implement as an options framework or a feudal-RL style manager. Do not
start here.

## Acceptance criteria for Phase 3

The old first criterion — *"forward pass on `[B=8, N=150, L=60, F=15]` under
20 ms on the 5090"* — is deleted. The machine does not exist and the shape is
not the one that runs (production is `B=128, N=504`). Its replacement is stated
at the production shape and against measured 4060 numbers:

| Criterion | State |
|---|---|
| One gradient step at the production shape costs **1,549 GFLOP**, of which the TCN is 97.9% (A2 §6.1) | **MEASURED** (analytic, reproducible via `a2_ppo_update.py analytic`) |
| Wall clock per gradient step on the 4060 at `N=504, L=60, minibatch 128`, uninstrumented, reported alongside `torch.cuda.max_memory_allocated()` | **NOT MEASURED** (A2 §0.2). Run `scripts/profiling/a2_all.sh` on the box. Until this exists, no timing claim about the 4060 is admissible |
| Peak VRAM at that shape stays under 8,188 MiB with headroom | **MEASURED and marginal** — 7,762–7,766 MiB, ~95% of the card (A2 §4.4). The OOM ceiling grid is NOT MEASURED |
| Gradients flow end-to-end (random loss backward changes every unfrozen parameter) | **UNVERIFIED** — no audit checked this test exists |
| Masked softmax: masked names have output probability < 1e-8 | **UNVERIFIED** — same |
| Attention weights on untradeable neighbours are ≈ 0 | **FAILS in spirit** — the unit test passes, but it only ever uses batch-constant masks, which is exactly the case `graph.py:334` gets right (A1 §6.5) |
| Param count matches the config's declared budget ± 5% | **UNVERIFIED**; note the effective width is 64, not the configured `embed_dim: 128` |

A criterion this phase should have had: **a GNN forward with a mask that varies
along the batch dimension matches the same elements run at `B=1`.** That one
test is the whole of the `graph.py:334` defect.
