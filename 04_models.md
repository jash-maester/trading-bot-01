# 04 — Models

Three components, composed bottom-up:

1. **Per-ticker time-series encoder.** Produces an embedding
   `h_i ∈ R^d` per stock from its feature window.
2. **Graph attention network** over stocks and sectors. Produces
   context-refined embeddings `z_i ∈ R^d`.
3. **Actor-critic heads** that consume `z_i` and portfolio state.

Sizes are starting points. With a 5090 you have room; resist the urge
to go big before the baseline works.

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

Architecture: 4 blocks, kernel 3, dilations `{1, 2, 4, 8}`, 64
channels, LayerNorm, GELU, dropout 0.1.

Alternative behind a config flag: small Transformer with RoPE, same
`d=64`, 2 heads, 2 layers. Useful for ablation.

**Weight sharing**: one encoder shared across all tickers. Sector
identity is injected later in the GNN.

## Hetero graph

Nodes: `N` stock nodes + `S` sector nodes (S ≈ 6–8).

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

DropEdge (p=0.1) during training for regularization.

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

Trivial for the 5090. Leave room to scale `d` to 128 and GNN depth to 3
once the baseline beats the baselines.

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

- Forward pass on a batch of `[B=8, N=150, L=60, F=15]` runs under 20 ms
  on the 5090.
- Gradients flow end-to-end (test: random loss backward changes every
  parameter except explicitly frozen ones).
- Masked softmax tests: masked names have output probability < 1e-8.
- Unit test: attention weights on untradeable neighbors are ≈ 0.
- Param count matches the config's declared budget ± 5%.
