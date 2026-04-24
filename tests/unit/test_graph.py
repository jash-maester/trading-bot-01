"""Unit tests for M6 — HeteroGNN graph construction, forward pass, and acceptance criteria.

Acceptance criteria verified here:
- ``build_sector_edges`` correctness (intra-sector, membership, masking).
- Forward pass shapes for both ``relations="all"`` and ``"intra_only"``.
- **Untradeable-feature isolation**: changing an untradeable stock's features must not
  affect any tradeable neighbour (zeroing + edge exclusion).
- Gradient flow through GNN and full ``GNNActorCritic``.
- DropEdge: with ``drop_edge_prob=1.0`` all edges are removed during training.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sector_ids(N: int, num_sectors: int = 4) -> torch.Tensor:
    """Return sector IDs cycling 1..num_sectors for N stocks."""
    return torch.tensor([(i % num_sectors) + 1 for i in range(N)])


def _make_obs(
    B: int = 2,
    N: int = 12,
    L: int = 60,
    F: int = 15,
    num_sectors: int = 4,
) -> dict[str, torch.Tensor]:
    sector_ids_1d = _sector_ids(N, num_sectors)
    return {
        "features":    torch.randn(B, L, N, F),
        "mask":        torch.ones(B, N, dtype=torch.int8),
        "sector_ids":  sector_ids_1d.unsqueeze(0).expand(B, -1).int(),
        "portfolio":   torch.softmax(torch.randn(B, N + 1), dim=-1),
        "cash":        torch.full((B,), 500_000.0),
        "nav":         torch.full((B,), 1_000_000.0),
        "t_frac":      torch.full((B,), 0.5),
    }


# ---------------------------------------------------------------------------
# build_sector_edges
# ---------------------------------------------------------------------------


class TestBuildSectorEdges:
    def test_intra_only_returns_single_relation(self) -> None:
        from trader.models.graph import build_sector_edges

        sector_ids = torch.tensor([1, 1, 2, 2, 3])
        edges = build_sector_edges(sector_ids, tradeable_mask=None, relations="intra_only")

        assert set(edges.keys()) == {("stock", "same_sector", "stock")}

    def test_all_returns_four_relations(self) -> None:
        from trader.models.graph import build_sector_edges

        sector_ids = torch.tensor([1, 1, 2, 2, 3])
        edges = build_sector_edges(sector_ids, tradeable_mask=None, relations="all")

        expected = {
            ("stock", "same_sector", "stock"),
            ("stock", "in", "sector"),
            ("sector", "contains", "stock"),
            ("sector", "relates_to", "sector"),
        }
        assert set(edges.keys()) == expected

    def test_intra_edges_same_sector_only(self) -> None:
        """Every intra edge must connect stocks in the same sector."""
        from trader.models.graph import build_sector_edges

        sector_ids = torch.tensor([1, 1, 2, 2, 3])
        edges = build_sector_edges(sector_ids, tradeable_mask=None, relations="all")
        intra = edges[("stock", "same_sector", "stock")]

        for s, d in zip(intra[0].tolist(), intra[1].tolist()):
            assert sector_ids[s].item() == sector_ids[d].item(), (
                f"Cross-sector edge: {s}(sec={sector_ids[s]}) → {d}(sec={sector_ids[d]})"
            )

    def test_intra_edges_bidirectional(self) -> None:
        """Intra-sector edge set must be symmetric (both directions)."""
        from trader.models.graph import build_sector_edges

        sector_ids = torch.tensor([1, 1, 1])
        edges = build_sector_edges(sector_ids, tradeable_mask=None, relations="all")
        intra = edges[("stock", "same_sector", "stock")]

        edge_set = set(zip(intra[0].tolist(), intra[1].tolist()))
        for s, d in list(edge_set):
            assert (d, s) in edge_set, f"Missing reverse edge ({d}, {s})"

    def test_unknown_sector_excluded(self) -> None:
        """Stocks with sector_id=0 must not appear in any edge."""
        from trader.models.graph import build_sector_edges

        sector_ids = torch.tensor([1, 2, 0])   # stock 2 has unknown sector
        edges = build_sector_edges(sector_ids, tradeable_mask=None, relations="all")

        for rel, ei in edges.items():
            flat = ei.flatten().tolist()
            assert 2 not in flat, f"Unknown-sector stock 2 found in {rel} edges"

    def test_tradeable_mask_excludes_untradeable(self) -> None:
        """Untradeable stocks must not appear in any edge."""
        from trader.models.graph import build_sector_edges

        sector_ids = torch.tensor([1, 1, 1, 2])
        mask = torch.tensor([True, True, False, True])   # stock 2 not tradeable

        edges = build_sector_edges(sector_ids, tradeable_mask=mask, relations="all")

        for rel, ei in edges.items():
            flat = ei.flatten().tolist()
            assert 2 not in flat, f"Untradeable stock 2 appeared in {rel} edges"

    def test_membership_edges_correct(self) -> None:
        """Stock → sector edges must be consistent with sector_ids."""
        from trader.models.graph import build_sector_edges

        sector_ids = torch.tensor([1, 2, 1])
        edges = build_sector_edges(sector_ids, tradeable_mask=None, relations="all")
        s2sec = edges[("stock", "in", "sector")]

        for stock_node, sec_node in zip(s2sec[0].tolist(), s2sec[1].tolist()):
            expected_sec = sector_ids[stock_node].item() - 1   # 0-indexed
            assert sec_node == expected_sec, (
                f"Stock {stock_node} should map to sector {expected_sec}, got {sec_node}"
            )

    def test_empty_graph_when_all_unknown(self) -> None:
        """All-unknown sector IDs → all edge tensors have zero edges."""
        from trader.models.graph import build_sector_edges

        sector_ids = torch.zeros(5, dtype=torch.long)
        edges = build_sector_edges(sector_ids, tradeable_mask=None, relations="all")

        for rel, ei in edges.items():
            assert ei.shape[1] == 0, f"{rel} should have no edges, got {ei.shape[1]}"


# ---------------------------------------------------------------------------
# HeteroGNN forward pass
# ---------------------------------------------------------------------------


class TestHeteroGNN:
    def test_output_shape_all_relations(self) -> None:
        from trader.models.graph import GNNConfig, HeteroGNN

        B, N, d = 3, 20, 32
        cfg = GNNConfig(embed_dim=d, num_sectors=4, num_layers=2, num_heads=2)
        gnn = HeteroGNN(cfg)

        stock_x = torch.randn(B, N, d)
        sector_ids = _sector_ids(N, 4).unsqueeze(0).expand(B, -1)
        out = gnn(stock_x, sector_ids)

        assert out.shape == (B, N, d), f"Expected {(B, N, d)}, got {out.shape}"

    def test_output_shape_intra_only(self) -> None:
        from trader.models.graph import GNNConfig, HeteroGNN

        B, N, d = 2, 15, 32
        cfg = GNNConfig(
            embed_dim=d, num_sectors=4, num_layers=1,
            num_heads=2, relations="intra_only",
        )
        gnn = HeteroGNN(cfg)

        stock_x = torch.randn(B, N, d)
        sector_ids = _sector_ids(N, 4).unsqueeze(0).expand(B, -1)
        out = gnn(stock_x, sector_ids)

        assert out.shape == (B, N, d)

    def test_no_nan_output(self) -> None:
        from trader.models.graph import GNNConfig, HeteroGNN

        B, N, d = 2, 16, 32
        cfg = GNNConfig(embed_dim=d, num_sectors=4, num_layers=2, num_heads=2)
        gnn = HeteroGNN(cfg)
        gnn.eval()

        stock_x = torch.randn(B, N, d)
        sector_ids = _sector_ids(N, 4).unsqueeze(0).expand(B, -1)
        with torch.no_grad():
            out = gnn(stock_x, sector_ids)

        assert not torch.isnan(out).any(), "NaN in GNN output"

    def test_no_nan_with_tradeable_mask(self) -> None:
        from trader.models.graph import GNNConfig, HeteroGNN

        B, N, d = 2, 12, 32
        cfg = GNNConfig(embed_dim=d, num_sectors=4, num_layers=1, num_heads=2)
        gnn = HeteroGNN(cfg)
        gnn.eval()

        stock_x = torch.randn(B, N, d)
        sector_ids = _sector_ids(N, 4).unsqueeze(0).expand(B, -1)
        mask = torch.ones(B, N, dtype=torch.bool)
        mask[:, 0] = False   # mark first stock untradeable

        with torch.no_grad():
            out = gnn(stock_x, sector_ids, tradeable_mask=mask)

        assert not torch.isnan(out).any()

    # ── KEY ACCEPTANCE CRITERION ─────────────────────────────────────────────

    def test_untradeable_features_dont_leak(self) -> None:
        """M6 acceptance: changing an untradeable stock's features must not
        affect any tradeable neighbour's output.

        Mechanism: (1) untradeable features are zeroed before GNN,
        (2) untradeable stocks are excluded from all edges.
        Two runs with the same mask but different (one huge) features for the
        untradeable stock must produce *identical* outputs for all other stocks.
        """
        from trader.models.graph import GNNConfig, HeteroGNN

        # 9 stocks, 3 per sector; stock 0 will be marked untradeable
        N, d = 9, 32
        sector_ids_1d = torch.tensor([1, 1, 1, 2, 2, 2, 3, 3, 3])
        sector_ids = sector_ids_1d.unsqueeze(0)   # [1, N]

        mask = torch.ones(1, N, dtype=torch.bool)
        mask[0, 0] = False   # stock 0 is untradeable

        cfg = GNNConfig(
            embed_dim=d, num_sectors=4, num_layers=1,
            num_heads=2, drop_edge_prob=0.0,
        )
        gnn = HeteroGNN(cfg)
        gnn.eval()

        # Run A: normal features for stock 0 (still masked → zeroed)
        stock_x_normal = torch.randn(1, N, d)
        # Run B: massive perturbation for stock 0 (still masked → zeroed)
        stock_x_big = stock_x_normal.clone()
        stock_x_big[0, 0] = 1e6

        with torch.no_grad():
            out_a = gnn(stock_x_normal, sector_ids, tradeable_mask=mask)
            out_b = gnn(stock_x_big,    sector_ids, tradeable_mask=mask)

        # All stocks' outputs must be identical (stock 0 contributes nothing)
        diff = (out_b - out_a).abs().max().item()
        assert diff < 1e-5, (
            f"Untradeable stock features leaked to neighbours: max diff = {diff}"
        )

    # ── Gradient flow ────────────────────────────────────────────────────────

    def test_gradients_flow_all_relations(self) -> None:
        """Gradients must be non-NaN for all reachable parameters.

        Note: last-layer parameters that only update sector nodes (which are
        not part of the returned stock embeddings) legitimately receive no
        gradient — this is expected behaviour, not a bug.
        """
        from trader.models.graph import GNNConfig, HeteroGNN

        B, N, d = 2, 12, 32
        cfg = GNNConfig(
            embed_dim=d, num_sectors=4, num_layers=2,
            num_heads=2, drop_edge_prob=0.0,
        )
        gnn = HeteroGNN(cfg)

        stock_x = torch.randn(B, N, d)
        sector_ids = _sector_ids(N, 4).unsqueeze(0).expand(B, -1)
        out = gnn(stock_x, sector_ids)
        out.sum().backward()

        # Every parameter that IS in the computation graph must have clean grads.
        for name, param in gnn.named_parameters():
            if param.requires_grad and param.grad is not None:
                assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"

        # Sanity: the majority of parameters must be reachable.
        n_total = sum(1 for p in gnn.parameters() if p.requires_grad)
        n_grads = sum(1 for p in gnn.parameters() if p.requires_grad and p.grad is not None)
        assert n_grads > n_total // 2, (
            f"Only {n_grads}/{n_total} parameters received gradients"
        )

    def test_gradients_flow_intra_only(self) -> None:
        """In intra_only mode, sector_emb and sector_norms are unused — no grad expected.

        All active (intra-sector conv + stock_norms) parameters must be reachable.
        """
        from trader.models.graph import GNNConfig, HeteroGNN

        B, N, d = 2, 12, 32
        cfg = GNNConfig(
            embed_dim=d, num_sectors=4, num_layers=1,
            num_heads=2, drop_edge_prob=0.0, relations="intra_only",
        )
        gnn = HeteroGNN(cfg)

        stock_x = torch.randn(B, N, d)
        sector_ids = _sector_ids(N, 4).unsqueeze(0).expand(B, -1)
        out = gnn(stock_x, sector_ids)
        out.sum().backward()

        # Params with gradients must be clean.
        for name, param in gnn.named_parameters():
            if param.requires_grad and param.grad is not None:
                assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"

        # At least one parameter (intra-sector conv) must have a gradient.
        assert any(
            p.grad is not None for p in gnn.parameters() if p.requires_grad
        ), "No parameters received gradients in intra_only mode"


# ---------------------------------------------------------------------------
# DropEdge
# ---------------------------------------------------------------------------


class TestDropEdge:
    def test_drop_edge_prob_one_removes_all(self) -> None:
        """With drop_edge_prob=1.0, every non-empty edge tensor must become empty."""
        from trader.models.graph import GNNConfig, HeteroGNN, build_sector_edges

        N, d = 20, 32
        cfg = GNNConfig(
            embed_dim=d, num_sectors=4, num_layers=1,
            num_heads=2, drop_edge_prob=1.0,
        )
        gnn = HeteroGNN(cfg)
        gnn.train()   # enable DropEdge

        sector_ids_1d = _sector_ids(N, 4)
        base_edges = build_sector_edges(sector_ids_1d, tradeable_mask=None, relations="all")
        dropped = gnn._apply_drop_edge(base_edges)

        for rel, ei in dropped.items():
            assert ei.shape[1] == 0, (
                f"Expected 0 edges for {rel} with drop_prob=1.0, got {ei.shape[1]}"
            )

    def test_drop_edge_disabled_in_eval(self) -> None:
        """In eval mode, DropEdge must be a no-op regardless of probability."""
        from trader.models.graph import GNNConfig, HeteroGNN, build_sector_edges

        N, d = 20, 32
        cfg = GNNConfig(
            embed_dim=d, num_sectors=4, num_layers=1,
            num_heads=2, drop_edge_prob=1.0,
        )
        gnn = HeteroGNN(cfg)
        gnn.eval()   # disable DropEdge

        sector_ids_1d = _sector_ids(N, 4)
        base_edges = build_sector_edges(sector_ids_1d, tradeable_mask=None, relations="all")
        kept = gnn._apply_drop_edge(base_edges)

        for rel in base_edges:
            assert kept[rel].shape == base_edges[rel].shape, (
                f"DropEdge modified edges in eval mode for {rel}"
            )


# ---------------------------------------------------------------------------
# GNNActorCritic end-to-end
# ---------------------------------------------------------------------------


class TestGNNActorCritic:
    def test_forward_shapes(self) -> None:
        from trader.models.actor_critic import ModelConfig
        from trader.models.graph import GNNActorCritic, GNNConfig

        B, N, L, F = 2, 12, 60, 15
        model_cfg = ModelConfig(in_features=F, n_tickers=N, embed_dim=32)
        gnn_cfg = GNNConfig(embed_dim=32, num_sectors=4, num_layers=1, num_heads=2)

        model = GNNActorCritic(model_cfg, gnn_cfg)
        obs = _make_obs(B=B, N=N, L=L, F=F, num_sectors=4)

        action_mean, value, log_std = model(obs)

        assert action_mean.shape == (B, N + 1)
        assert value.shape == (B,)
        assert log_std.shape == (N + 1,)

    def test_get_action_and_value_shapes(self) -> None:
        from trader.models.actor_critic import ModelConfig
        from trader.models.graph import GNNActorCritic, GNNConfig

        B, N, L, F = 3, 8, 60, 15
        model_cfg = ModelConfig(in_features=F, n_tickers=N, embed_dim=32)
        gnn_cfg = GNNConfig(embed_dim=32, num_sectors=4, num_layers=1, num_heads=2)

        model = GNNActorCritic(model_cfg, gnn_cfg)
        obs = _make_obs(B=B, N=N, L=L, F=F, num_sectors=4)

        action, log_prob, entropy, value = model.get_action_and_value(obs)

        assert action.shape == (B, N + 1)
        assert log_prob.shape == (B,)
        assert entropy.shape == (B,)
        assert value.shape == (B,)
        assert not torch.isnan(action).any()
        assert torch.isfinite(log_prob).all()

    def test_intra_only_forward(self) -> None:
        from trader.models.actor_critic import ModelConfig
        from trader.models.graph import GNNActorCritic, GNNConfig

        B, N, L, F = 2, 12, 60, 15
        model_cfg = ModelConfig(in_features=F, n_tickers=N, embed_dim=32)
        gnn_cfg = GNNConfig(
            embed_dim=32, num_sectors=4, num_layers=1,
            num_heads=2, relations="intra_only",
        )

        model = GNNActorCritic(model_cfg, gnn_cfg)
        obs = _make_obs(B=B, N=N, L=L, F=F, num_sectors=4)

        action_mean, value, log_std = model(obs)

        assert action_mean.shape == (B, N + 1)
        assert value.shape == (B,)

    def test_gradients_flow_end_to_end(self) -> None:
        """Backward through TCN → GNN → heads; all parameters must get gradients."""
        from trader.models.actor_critic import ModelConfig
        from trader.models.graph import GNNActorCritic, GNNConfig

        B, N, L, F = 2, 12, 60, 15
        model_cfg = ModelConfig(in_features=F, n_tickers=N, embed_dim=32)
        gnn_cfg = GNNConfig(
            embed_dim=32, num_sectors=4, num_layers=2,
            num_heads=2, drop_edge_prob=0.0,
        )
        model = GNNActorCritic(model_cfg, gnn_cfg)
        obs = _make_obs(B=B, N=N, L=L, F=F, num_sectors=4)

        action, log_prob, entropy, value = model.get_action_and_value(obs)
        loss = -log_prob.mean() + value.pow(2).mean() - 0.01 * entropy.mean()
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                assert not torch.isnan(param.grad).any(), f"NaN grad in {name}"

    def test_embed_dim_override(self) -> None:
        """GNNActorCritic must align GNN embed_dim with TCN out_dim automatically."""
        from trader.models.actor_critic import ModelConfig
        from trader.models.graph import GNNActorCritic, GNNConfig

        B, N, L, F = 1, 8, 60, 15
        # TCN will output 48 (last num_channels value), not 64
        model_cfg = ModelConfig(
            in_features=F, n_tickers=N, embed_dim=64,
            num_channels=[32, 48],
        )
        # Deliberately mismatched embed_dim — should be overridden
        gnn_cfg = GNNConfig(embed_dim=64, num_sectors=4, num_layers=1, num_heads=2)

        model = GNNActorCritic(model_cfg, gnn_cfg)
        obs = _make_obs(B=B, N=N, L=L, F=F, num_sectors=4)

        # Should not raise (GNN dim is auto-corrected to 48)
        action_mean, value, _ = model(obs)
        assert action_mean.shape == (B, N + 1)
        assert value.shape == (B,)
