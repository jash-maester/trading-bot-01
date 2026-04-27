"""Heterogeneous Graph Attention Network for cross-stock and cross-sector reasoning.

Architecture (``relations="all"``):

- **N stock nodes** — embeddings from the TCN encoder.
- **S sector nodes** — learned embeddings (one per sector).
- **Three relation types**:

  =========================================  =============================================
  Relation                                   Direction
  =========================================  =============================================
  ``stock --same_sector--> stock``           Intra-sector, bidirectional
  ``stock --in--> sector``                   Membership (stock to its sector)
  ``sector --contains--> stock``             Membership reverse
  ``sector --relates_to--> sector``          Inter-sector, bidirectional
  =========================================  =============================================

- 2+ :class:`~torch_geometric.nn.HeteroConv` layers with residual connections,
  LayerNorm, GELU activation.
- DropEdge regularisation (``drop_edge_prob``) applied during training.

With ``relations="intra_only"`` only the stock↔stock same-sector edges are built
and no sector node embeddings are used — useful as an ablation midpoint between
the full GNN (``gnn_v1``) and the no-graph MLP baseline (``mlp_baseline``).
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch_geometric.nn import GATv2Conv, HeteroConv

from trader.models.actor_critic import ModelConfig
from trader.models.encoders import CrossStockAttention, TCNEncoder
from trader.models.heads import ActorHead, CriticHead

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------

#: Edge-index dictionary: maps (src_node_type, relation, dst_node_type) → LongTensor [2, E].
_EdgeDict = dict[tuple[str, str, str], torch.Tensor]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class GNNConfig:
    """Configuration for :class:`HeteroGNN`.

    Parameters
    ----------
    embed_dim:
        Embedding dimension for both stock and sector nodes.
        Overridden in :class:`GNNActorCritic` to match the TCN encoder's
        actual output dimension.
    num_sectors:
        Total number of sector IDs in the universe (matches ``SECTOR_IDS``
        in ``universe.py``; IDs are 1-indexed, sector embeddings are 0-indexed).
    num_layers:
        Number of :class:`~torch_geometric.nn.HeteroConv` layers.
    num_heads:
        Attention heads per :class:`~torch_geometric.nn.GATv2Conv`.
    dropout:
        Dropout probability inside each GATv2Conv and on residual outputs.
    drop_edge_prob:
        DropEdge probability applied once per forward pass during training.
    relations:
        ``"all"``        — all four relation types (intra-sector, membership,
                           inter-sector).
        ``"intra_only"`` — only intra-sector stock↔stock edges.
    """

    embed_dim: int = 64
    num_sectors: int = 8
    num_layers: int = 2
    num_heads: int = 2
    dropout: float = 0.1
    drop_edge_prob: float = 0.1
    relations: str = "all"


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def build_sector_edges(
    sector_ids: torch.Tensor,              # [N] int, 1-indexed (0 = unknown)
    tradeable_mask: torch.Tensor | None,   # [N] bool — None means all tradeable
    relations: str = "all",
) -> _EdgeDict:
    """Build heterogeneous edge indices for a single graph of N stock nodes.

    Edges are only built between *tradeable* stocks (when a mask is supplied).
    Stocks with ``sector_id == 0`` (unknown / unclassified) are excluded from
    all edges.

    Parameters
    ----------
    sector_ids:
        Per-stock sector assignment; 1-indexed, ``0`` = unclassified.
    tradeable_mask:
        Boolean mask of shape ``[N]``; ``False`` entries are excluded.
        Pass ``None`` to treat every stock as tradeable.
    relations:
        ``"all"``        — intra-sector + membership + inter-sector edges.
        ``"intra_only"`` — only stock↔stock same-sector edges.

    Returns
    -------
    dict
        Maps ``(src_type, relation, dst_type)`` → ``LongTensor [2, E]``.
    """
    N = sector_ids.shape[0]
    device = sector_ids.device

    if tradeable_mask is None:
        tradeable = torch.ones(N, dtype=torch.bool, device=device)
    else:
        tradeable = tradeable_mask.bool().to(device)

    # ── Intra-sector edges (stock → stock, same sector) ──────────────────────
    src_parts: list[torch.Tensor] = []
    dst_parts: list[torch.Tensor] = []

    unique_sectors = sector_ids[sector_ids > 0].unique()  # type: ignore[no-untyped-call]
    for s in unique_sectors:
        in_s_all = (sector_ids == s).nonzero(as_tuple=True)[0]  # all in sector s
        in_s = in_s_all[tradeable[in_s_all]]                    # tradeable subset
        if in_s.shape[0] < 2:
            continue
        pairs = torch.combinations(in_s, r=2)   # [C(k,2), 2]
        i_idx, j_idx = pairs[:, 0], pairs[:, 1]
        # Bidirectional
        src_parts.append(torch.cat([i_idx, j_idx]))
        dst_parts.append(torch.cat([j_idx, i_idx]))

    if src_parts:
        intra_ei = torch.stack([torch.cat(src_parts), torch.cat(dst_parts)])
    else:
        intra_ei = torch.zeros(2, 0, dtype=torch.long, device=device)

    if relations == "intra_only":
        return {("stock", "same_sector", "stock"): intra_ei}

    # ── Membership edges (stock ↔ sector) ─────────────────────────────────────
    has_sector = (sector_ids > 0) & tradeable
    stock_idx = has_sector.nonzero(as_tuple=True)[0]   # [M]
    sec_idx_0 = sector_ids[stock_idx] - 1              # 0-indexed sector [M]

    member_ei  = torch.stack([stock_idx, sec_idx_0])   # stock → sector [2, M]
    reverse_ei = torch.stack([sec_idx_0, stock_idx])   # sector → stock [2, M]

    # ── Inter-sector edges (sector → sector, bidirectional) ───────────────────
    active_secs = sec_idx_0.unique()  # type: ignore[no-untyped-call]
    if active_secs.shape[0] >= 2:
        pairs_s = torch.combinations(active_secs, r=2)
        s1, s2 = pairs_s[:, 0], pairs_s[:, 1]
        inter_ei = torch.stack([
            torch.cat([s1, s2]),
            torch.cat([s2, s1]),
        ])
    else:
        inter_ei = torch.zeros(2, 0, dtype=torch.long, device=device)

    return {
        ("stock", "same_sector", "stock"):   intra_ei,
        ("stock", "in", "sector"):           member_ei,
        ("sector", "contains", "stock"):     reverse_ei,
        ("sector", "relates_to", "sector"):  inter_ei,
    }


# ---------------------------------------------------------------------------
# HeteroGNN
# ---------------------------------------------------------------------------


class HeteroGNN(nn.Module):
    """Multi-layer heterogeneous graph attention over stocks and sectors.

    Input:  stock embeddings ``[B, N, d]``, sector IDs ``[B, N]``.
    Output: refined stock embeddings ``[B, N, d]``.

    **Batching strategy**: the graph topology is identical for every batch
    element (same universe, same sector mapping). The B subgraphs are tiled
    into a single disconnected graph of ``B*N`` stock nodes (and ``B*S``
    sector nodes for ``relations="all"``), so a single :class:`HeteroConv`
    call handles the entire batch.
    """

    def __init__(self, cfg: GNNConfig) -> None:
        super().__init__()
        d = cfg.embed_dim
        h = cfg.num_heads
        self.cfg = cfg
        self.num_sectors = cfg.num_sectors
        self._relations = cfg.relations
        self.drop_edge_prob = cfg.drop_edge_prob

        # Learnable sector-node embedding (one per sector, 0-indexed).
        # PyTorch's default `nn.Embedding` init is N(0, 1) — far too large
        # relative to LayerNorm-ed stock embeddings (~unit norm). Without
        # rescaling, the membership / inter-sector edges initially propagate
        # ~8× the stock-embedding magnitude into every connected stock,
        # which forces the policy to spend hundreds of updates un-learning
        # this random sector influence.  N(0, 0.1) puts sector nodes on
        # the same scale as the encoder output from the start.
        self.sector_emb: nn.Embedding = nn.Embedding(cfg.num_sectors, d)
        nn.init.normal_(self.sector_emb.weight, mean=0.0, std=0.1)

        # Per-layer modules
        self.layers:       nn.ModuleList = nn.ModuleList()
        self.stock_norms:  nn.ModuleList = nn.ModuleList()
        self.sector_norms: nn.ModuleList = nn.ModuleList()

        for _ in range(cfg.num_layers):
            conv_dict: dict[tuple[str, str, str], nn.Module] = {
                ("stock", "same_sector", "stock"): GATv2Conv(
                    d, d, heads=h, concat=False,
                    dropout=cfg.dropout, add_self_loops=True,
                ),
            }
            if cfg.relations == "all":
                conv_dict.update({
                    ("stock", "in", "sector"): GATv2Conv(
                        (d, d), d, heads=h, concat=False,
                        dropout=cfg.dropout, add_self_loops=False,
                    ),
                    ("sector", "contains", "stock"): GATv2Conv(
                        (d, d), d, heads=h, concat=False,
                        dropout=cfg.dropout, add_self_loops=False,
                    ),
                    ("sector", "relates_to", "sector"): GATv2Conv(
                        d, d, heads=h, concat=False,
                        dropout=cfg.dropout, add_self_loops=True,
                    ),
                })
            self.layers.append(HeteroConv(conv_dict, aggr="sum"))
            self.stock_norms.append(nn.LayerNorm(d))
            self.sector_norms.append(nn.LayerNorm(d))

        self.drop: nn.Dropout = nn.Dropout(cfg.dropout)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _apply_drop_edge(self, edge_dict: _EdgeDict) -> _EdgeDict:
        """Randomly drop a fraction of edges (DropEdge regularisation).

        Applied once per forward pass, *before* tiling for the batch, so
        every batch element shares the same dropped mask (standard practice).
        No-op during evaluation (``self.training == False``).
        """
        if not self.training or self.drop_edge_prob <= 0.0:
            return edge_dict
        result: _EdgeDict = {}
        for rel, ei in edge_dict.items():
            if ei.shape[1] == 0:
                result[rel] = ei
                continue
            keep = torch.rand(ei.shape[1], device=ei.device) > self.drop_edge_prob
            result[rel] = ei[:, keep]
        return result

    def _tile_edges(
        self,
        base_dict: _EdgeDict,
        B: int,
        N: int,
        S: int,
    ) -> _EdgeDict:
        """Tile edge indices for B disconnected subgraphs.

        Stock nodes for batch *b* occupy indices ``[b*N, (b+1)*N)``.
        Sector nodes for batch *b* occupy indices ``[b*S, (b+1)*S)``.
        """
        tiled: _EdgeDict = {}
        for rel, ei in base_dict.items():
            if ei.shape[1] == 0:
                tiled[rel] = ei
                continue
            src_type, _, dst_type = rel
            src_step = N if src_type == "stock" else S
            dst_step = N if dst_type == "stock" else S
            parts: list[torch.Tensor] = []
            for b in range(B):
                offset = torch.tensor(
                    [[b * src_step], [b * dst_step]],
                    dtype=torch.long, device=ei.device,
                )
                parts.append(ei + offset)
            tiled[rel] = torch.cat(parts, dim=1)
        return tiled

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        stock_x: torch.Tensor,                      # [B, N, d]
        sector_ids: torch.Tensor,                   # [B, N] int, 1-indexed (0=unknown)
        tradeable_mask: torch.Tensor | None = None,  # [B, N] bool
    ) -> torch.Tensor:                              # [B, N, d]
        """Refine per-ticker embeddings via heterogeneous graph attention.

        Untradeable stocks are **zeroed before** the GNN *and* excluded from
        all edges, so their (potentially noisy or stale) features cannot leak
        into their neighbours' attention.
        """
        B, N, d = stock_x.shape
        device = stock_x.device
        S = self.num_sectors

        # ── Zero untradeable stock features ───────────────────────────────────
        if tradeable_mask is not None:
            stock_x = stock_x * tradeable_mask.unsqueeze(-1).float().to(device)

        # ── Build edge structure from first batch element ──────────────────────
        # The universe is fixed, so sector assignments are the same for every
        # batch element; we use index [0] and tile the result.
        sec_ids_0 = sector_ids[0]                              # [N]
        mask_0: torch.Tensor | None = (
            tradeable_mask[0] if tradeable_mask is not None else None
        )
        base_edges = build_sector_edges(sec_ids_0, mask_0, self._relations)
        base_edges = self._apply_drop_edge(base_edges)
        batched_edges = self._tile_edges(base_edges, B, N, S)

        # ── Flatten nodes ─────────────────────────────────────────────────────
        stock_out: torch.Tensor = stock_x.reshape(B * N, d)

        # Sector embeddings: [S, d] tiled to [B*S, d]
        sec_idx = torch.arange(S, device=device)
        sector_out: torch.Tensor = cast(
            torch.Tensor,
            self.sector_emb(sec_idx).repeat(B, 1),
        )

        # ── GNN layers with residual + LayerNorm + GELU ───────────────────────
        for conv, s_norm, sec_norm in zip(
            self.layers, self.stock_norms, self.sector_norms
        ):
            if self._relations == "intra_only":
                x_dict: dict[str, torch.Tensor] = {"stock": stock_out}
            else:
                x_dict = {"stock": stock_out, "sector": sector_out}

            # HeteroConv returns dict[str, Tensor]; typed as Any by stubs.
            new_x: dict[str, torch.Tensor] = conv(x_dict, batched_edges)

            if "stock" in new_x:
                stock_out = cast(
                    torch.Tensor,
                    s_norm(F.gelu(new_x["stock"]) + stock_out),
                )
                stock_out = cast(torch.Tensor, self.drop(stock_out))

            if "sector" in new_x:
                sector_out = cast(
                    torch.Tensor,
                    sec_norm(F.gelu(new_x["sector"]) + sector_out),
                )
                sector_out = cast(torch.Tensor, self.drop(sector_out))

        return stock_out.reshape(B, N, d)  # [B, N, d]


# ---------------------------------------------------------------------------
# GNN Actor-Critic
# ---------------------------------------------------------------------------


class GNNActorCritic(nn.Module):
    """M6 actor-critic: TCN encoder → HeteroGNN → actor / critic heads.

    Drop-in replacement for :class:`~trader.models.actor_critic.ActorCritic`
    when ``use_graph: true`` is set in the Hydra model config.  The GNN
    ``embed_dim`` is automatically aligned with the encoder's actual output
    dimension (``TCNEncoder.out_dim``), regardless of what the config declares.
    """

    def __init__(
        self,
        model_cfg: ModelConfig,
        gnn_cfg: GNNConfig,
        feat_mean: torch.Tensor | None = None,
        feat_std: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.encoder = TCNEncoder(
            in_features=model_cfg.in_features,
            embed_dim=model_cfg.embed_dim,
            num_channels=model_cfg.num_channels,
            kernel_size=model_cfg.kernel_size,
            dropout=model_cfg.dropout,
            feat_mean=feat_mean,
            feat_std=feat_std,
        )
        d = self.encoder.out_dim
        # Align GNN embed_dim with encoder output; silently overrides config value.
        self.gnn = HeteroGNN(replace(gnn_cfg, embed_dim=d))
        # Cross-stock self-attention AFTER the GNN.  GNN gives sector-aware
        # context (within sector, sector-mediated cross-sector); cross-attn
        # gives all-pairs context.  The two are complementary:
        # GNN provides structured priors, cross-attn provides flexibility.
        self.cross_attn: CrossStockAttention | None = (
            CrossStockAttention(
                embed_dim=d,
                num_heads=model_cfg.cross_attn_heads,
                dropout=model_cfg.dropout,
            )
            if model_cfg.use_cross_attn
            else None
        )
        self.actor  = ActorHead(d, hidden=model_cfg.head_hidden)
        # Critic uses the same num_sectors as the GNN (single source of truth).
        self.critic = CriticHead(
            d, num_sectors=gnn_cfg.num_sectors, hidden=model_cfg.head_hidden,
        )
        self.log_std: nn.Parameter = nn.Parameter(
            torch.full((model_cfg.n_tickers + 1,), model_cfg.log_std_init)
        )

    def forward(
        self,
        obs: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(action_mean [B, N+1], value [B], log_std [N+1])``."""
        features   = obs["features"].float().permute(0, 2, 1, 3)  # [B, L, N, F] → [B, N, L, F]
        portfolio  = obs["portfolio"].float()    # [B, N+1]
        t_frac     = obs["t_frac"].float()       # [B]
        sector_ids = obs["sector_ids"].long()    # [B, N]

        mask_raw = obs.get("mask")
        tradeable: torch.Tensor | None = (
            mask_raw.bool() if mask_raw is not None else None
        )

        if t_frac.dim() == 0:
            t_frac = t_frac.unsqueeze(0)

        # Critic-only state summary (with backward-compatible zeros).
        zero_b = torch.zeros_like(t_frac)
        recent_return = obs.get("recent_return_1d", zero_b).float()
        recent_vol = obs.get("recent_vol_20d", zero_b).float()
        nav_log_progress = obs.get("nav_log_progress", zero_b).float()
        if recent_return.dim() == 0:
            recent_return = recent_return.unsqueeze(0)
        if recent_vol.dim() == 0:
            recent_vol = recent_vol.unsqueeze(0)
        if nav_log_progress.dim() == 0:
            nav_log_progress = nav_log_progress.unsqueeze(0)

        h = self.encoder(features)                       # [B, N, d]
        z = self.gnn(h, sector_ids, tradeable)           # [B, N, d]
        if self.cross_attn is not None:
            z = self.cross_attn(z, tradeable_mask=tradeable)  # [B, N, d]
        action_mean = self.actor(z, portfolio)           # [B, N+1]
        value = self.critic(
            z, portfolio, t_frac, sector_ids,
            recent_return, recent_vol, nav_log_progress,
        )                                                # [B]
        return action_mean, value, self.log_std

    def get_action_and_value(
        self,
        obs: dict[str, torch.Tensor],
        action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample or evaluate action; return ``(action, log_prob, entropy, value)``."""
        mean, value, log_std = self.forward(obs)
        std  = log_std.exp().expand_as(mean)
        dist = Normal(mean, std)
        if action is None:
            action = dist.sample()                       # type: ignore[no-untyped-call]
        log_prob = dist.log_prob(action).sum(-1)         # type: ignore[no-untyped-call]
        entropy  = dist.entropy().sum(-1)                # type: ignore[no-untyped-call]
        return action, log_prob, entropy, value

    def get_value(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Return value estimate only (used for rollout bootstrap)."""
        _, value, _ = self.forward(obs)
        return value
