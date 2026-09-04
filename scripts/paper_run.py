#!/usr/bin/env python
"""Walk a panel day by day through the paper broker and write it to the ledger.

This is the bridge between the thing that was trained and the thing that will
one day place real orders.  It reuses :class:`~trader.env.panel_env.PanelTradingEnv`
as the *observation* source — the policy must see exactly the tensor it was
trained on — but the portfolio it reports on is the broker's, not the
environment's.  The two differ, on purpose:

* the broker settles sale proceeds T+1, so a rebalance that funds itself out of
  today's sales has its buy legs rejected;
* the broker charges DP once per scrip per sell day and accrues capital-gains
  tax per financial year.

A gap between the two NAV curves is therefore expected and is the number worth
looking at: it is what the backtest was quietly assuming away.

Usage (the ``+`` is Hydra's "this key is not in the config yet" prefix — these
knobs are run-time choices, not configuration, so they are deliberately not in
``configs/config.yaml``)::

    uv run python scripts/paper_run.py                        # equal-weight baseline
    uv run python scripts/paper_run.py +agent=momentum_top5
    uv run python scripts/paper_run.py +checkpoint=checkpoints/mlp_regime_seed42/model_000100.pt
    uv run python scripts/paper_run.py +panel=val +persist=false
    uv run python scripts/paper_run.py broker.initial_cash=100000

Config comes from ``configs/broker/paper.yaml`` (``initial_cash``,
``settlement_days``, ``slippage_model``, ``slippage_pct``) and
``configs/env/panel_daily.yaml`` (``min_trade_value``, ``lookback_days``).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import hydra
from loguru import logger
from omegaconf import DictConfig

# Large enough that PanelTradingEnv clamps the episode to the whole panel, which
# also pins the random start index to `lookback` — a paper run must walk the
# panel from one end to the other, not from a random day.
_WHOLE_PANEL = 10**9


def _resolve_panel(cfg: DictConfig, panels_root: Path) -> Path:
    """``panel=test`` names a split; anything with a separator is a path."""
    raw = str(cfg.get("panel", "test"))
    candidate = Path(raw)
    if candidate.suffix or candidate.parent != Path("."):
        return candidate
    return panels_root / f"{raw}.parquet"


def _price_tables(panel_path: Path, universe: list[str]) -> dict[str, Any]:
    """Per-date open / close / impact inputs, keyed the way the broker wants.

    The impact inputs are divided by the *close*, matching
    ``PanelTradingEnv.step`` exactly rather than approximately — a broker that
    divided by the open instead would fill at a slightly different price and
    the paper/backtest comparison would be measuring the wrong thing.
    """
    import polars as pl

    wanted = ["date", "ticker", "open", "close", "atr_14", "dollar_volume_20"]
    raw = pl.read_parquet(panel_path)
    frame = (
        raw.select([c for c in wanted if c in raw.columns])
        .filter(pl.col("ticker").is_in(universe))
        .sort(["date", "ticker"])
    )
    opens: dict[Any, dict[str, float]] = {}
    closes: dict[Any, dict[str, float]] = {}
    impact: dict[Any, dict[str, Any]] = {}

    from trader.broker.paper_broker import ImpactInputs

    has_impact = {"atr_14", "dollar_volume_20"} <= set(frame.columns)
    for row in frame.iter_rows(named=True):
        day, ticker = row["date"], row["ticker"]
        open_px, close_px = float(row["open"] or 0.0), float(row["close"] or 0.0)
        if open_px > 0.0:
            opens.setdefault(day, {})[ticker] = open_px
        if close_px > 0.0:
            closes.setdefault(day, {})[ticker] = close_px
        if has_impact and close_px > 0.0:
            impact.setdefault(day, {})[ticker] = ImpactInputs(
                atr_fraction=float(row["atr_14"] or 0.0) / close_px,
                adv_shares=max(float(row["dollar_volume_20"] or 0.0) / close_px, 1.0),
            )
    return {"opens": opens, "closes": closes, "impact": impact}


def _build_policy(
    cfg: DictConfig, panels_root: Path, universe: list[str]
) -> tuple[Any, str]:
    """Return ``(act(obs) -> logits, label)``, from a checkpoint or a baseline.

    The label becomes ``strategy_runs.strategy_id``, so it has to name the
    thing that actually traded — a baseline run labelled with the configured
    model name would be indistinguishable in the ledger from a real one.

    NOTE: the checkpoint branch reconstructs the model the same way
    ``trader.training.runner.train_one_run`` does.  That construction should be
    a shared ``build_model(cfg, ...)`` in ``runner.py``; it is duplicated here
    only because this change is not allowed to touch ``src/trader/training``.
    Any new model knob has to be added in both places until it is extracted.
    """
    checkpoint = cfg.get("checkpoint")
    if not checkpoint:
        from trader.env.baselines import (
            BuyAndHoldIndex,
            EqualWeightRebalanced,
            MomentumTopK,
            RandomPolicy,
            SixtyFortyCash,
        )

        agents = {
            "equal_weight": EqualWeightRebalanced,
            "buy_and_hold": BuyAndHoldIndex,
            "momentum_top5": MomentumTopK,
            "sixty_forty": SixtyFortyCash,
            "random": RandomPolicy,
        }
        name = str(cfg.get("agent", "equal_weight"))
        if name not in agents:
            raise SystemExit(f"unknown agent {name!r}; expected one of {sorted(agents)}")
        agent = agents[name]()
        agent.reset()
        logger.info(f"Policy: baseline {name} (no checkpoint given)")
        return agent.act, f"baseline:{name}"

    import torch

    from trader.data.feature_stats import compute_feature_stats, load_stats, stats_to_tensors
    from trader.data.features import FEATURE_COLS
    from trader.data.regime_features import (
        REGIME_DIM,
        compute_regime_stats,
        load_regime_stats,
        regime_stats_to_tensors,
    )
    from trader.models.actor_critic import ActorCritic, ModelConfig
    from trader.utils.seeding import get_device

    train_panel = panels_root / "train.parquet"
    stats_path = panels_root / "train.feature_stats.json"
    regime_path = panels_root / "train.regime_stats.json"
    feat_stats = load_stats(stats_path) if stats_path.exists() else compute_feature_stats(
        train_panel, FEATURE_COLS
    )
    regime_stats = (
        load_regime_stats(regime_path) if regime_path.exists() else compute_regime_stats(
            train_panel
        )
    )
    feat_mean, feat_std = stats_to_tensors(feat_stats, FEATURE_COLS)
    regime_mean, regime_std = regime_stats_to_tensors(regime_stats)

    channels_raw = cfg.model.tcn.get("num_channels")
    model_cfg = ModelConfig(
        in_features=len(FEATURE_COLS),
        n_tickers=len(universe),
        embed_dim=int(cfg.model.embed_dim),
        num_channels=[int(c) for c in channels_raw] if channels_raw is not None else None,
        kernel_size=int(cfg.model.tcn.kernel_size),
        dropout=float(cfg.model.tcn.dropout),
        use_cross_attn=bool(cfg.model.get("use_cross_attn", True)),
        cross_attn_heads=int(cfg.model.get("cross_attn_heads", 4)),
        regime_dim=REGIME_DIM,
        regime_film_encoder=bool(cfg.model.get("regime_film_encoder", False)),
        regime_film_attn=bool(cfg.model.get("regime_film_attn", False)),
        regime_in_critic=bool(cfg.model.get("regime_in_critic", False)),
        regime_film_hidden=int(cfg.model.get("regime_film_hidden", 32)),
        use_aux_return_head=bool(cfg.model.get("use_aux_return_head", False)),
        aux_return_hidden=int(cfg.model.get("aux_return_hidden", 32)),
    )
    if bool(cfg.model.get("use_graph", False)):
        from trader.models.graph import GNNActorCritic, GNNConfig

        graph = cfg.model.graph
        model: Any = GNNActorCritic(
            model_cfg,
            GNNConfig(
                num_sectors=int(graph.get("num_sectors", 8)),
                num_layers=int(graph.get("layers", 2)),
                num_heads=int(graph.get("num_heads", 2)),
                dropout=float(graph.get("dropout", 0.1)),
                drop_edge_prob=float(graph.get("drop_edge_prob", 0.1)),
                relations=str(graph.get("relations", "all")),
            ),
            feat_mean=feat_mean,
            feat_std=feat_std,
        )
    else:
        model = ActorCritic(
            model_cfg,
            feat_mean=feat_mean,
            feat_std=feat_std,
            regime_mean=regime_mean,
            regime_std=regime_std,
        )

    device = get_device()
    state = torch.load(Path(checkpoint), map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"] if "model_state" in state else state)
    model.to(device).eval()
    deterministic = bool(cfg.get("deterministic", True))
    logger.info(
        f"Policy: checkpoint {checkpoint} on {device} "
        f"({'mean action' if deterministic else 'sampled'})"
    )

    def act(obs: dict[str, Any]) -> Any:
        with torch.no_grad():
            batch = {k: torch.as_tensor(v, device=device).unsqueeze(0) for k, v in obs.items()}
            if deterministic:
                # Paper trading wants the policy's best guess, not a draw from
                # its exploration distribution — two runs of the same
                # checkpoint on the same panel must produce the same orders.
                action = model(batch)[0]
            else:
                action, _, _, _ = model.get_action_and_value(batch)
            return action.squeeze(0).cpu().numpy()

    return act, f"{cfg.model.get('name', 'policy')}:{Path(checkpoint).stem}"


def _open_session_db(cfg: DictConfig) -> Any:
    """A SQLAlchemy session, or None when persistence is off or unreachable."""
    if not bool(cfg.get("persist", True)):
        logger.warning("persist=false — running in memory, nothing will be written")
        return None
    try:
        from sqlalchemy.orm import Session

        from trader.db.engine import get_engine

        engine = get_engine()
        with engine.connect():
            pass
        return Session(engine)
    except Exception as exc:  # noqa: BLE001 — a paper run is still useful offline
        logger.warning(f"Postgres unreachable ({type(exc).__name__}); running in memory")
        return None


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    import hydra.utils
    import numpy as np

    from trader.broker.paper_broker import (
        PaperBroker,
        PaperBrokerConfig,
        weights_from_logits,
    )
    from trader.data.features import FEATURE_COLS
    from trader.data.universe import active_tickers
    from trader.env.panel_env import PanelTradingEnv
    from trader.utils.seeding import seed_everything

    orig_cwd = Path(hydra.utils.get_original_cwd())
    panels_root = orig_cwd / "data" / "panels"
    panel_path = _resolve_panel(cfg, panels_root)
    if not panel_path.exists():
        raise SystemExit(f"panel not found: {panel_path} — run scripts/build_features.py")

    seed_everything(int(cfg.seed))
    universe = active_tickers()
    max_weight = float(cfg.env.get("max_weight_per_name", 0.10))

    broker_cfg = PaperBrokerConfig(
        initial_cash=float(cfg.broker.get("initial_cash", 100_000)),
        settlement_days=int(cfg.broker.get("settlement_days", 1)),
        max_weight_per_name=max_weight,
        min_trade_value=float(cfg.env.get("min_trade_value", 500)),
        slippage_model=str(cfg.broker.get("slippage_model", "percentage")),  # type: ignore[arg-type]
        slippage_pct=float(cfg.broker.get("slippage_pct", 0.001)),
    )

    env = PanelTradingEnv(
        panel_path=panel_path,
        universe=universe,
        feature_columns=FEATURE_COLS,
        lookback=int(cfg.env.lookback_days),
        episode_length=_WHOLE_PANEL,
        initial_cash=broker_cfg.initial_cash,
        max_weight_per_name=max_weight,
        seed=int(cfg.seed),
    )
    tables = _price_tables(panel_path, universe)
    act, label = _build_policy(cfg, panels_root, universe)

    session = _open_session_db(cfg)
    broker = PaperBroker(
        broker_cfg,
        session=session,
        strategy_id=str(cfg.get("strategy_id") or label),
    )
    run_id = broker.start_run(notes=f"panel={panel_path.name}")
    logger.info(
        f"Paper run {run_id}: {panel_path.name}  ₹{broker_cfg.initial_cash:,.0f} start, "
        f"T+{broker_cfg.settlement_days} settlement, cap {max_weight:.0%}, "
        f"min trade ₹{broker_cfg.min_trade_value:,.0f}"
    )

    obs, _ = env.reset(seed=int(cfg.seed))
    done = False
    n_rejected = 0
    try:
        while not done:
            logits = np.asarray(act(obs), dtype=np.float64)
            broker.submit_target_weights(
                weights_from_logits(logits, universe, obs["mask"].astype(bool), max_weight)
            )
            obs, _, terminated, truncated, info = env.step(logits)
            done = bool(terminated or truncated)

            day = info["date"]
            result = broker.open_session(
                day, tables["opens"].get(day, {}), impact=tables["impact"].get(day)
            )
            n_rejected += len(result.rejected)
            snapshot = broker.mark_to_market(day, tables["closes"].get(day, {}))
            if len(broker.snapshots) % 50 == 0:
                logger.info(
                    f"{day}  NAV ₹{snapshot.nav:,.0f}  cash ₹{snapshot.settled_cash:,.0f} "
                    f"(+₹{snapshot.unsettled_cash:,.0f} unsettled)  "
                    f"fees ₹{snapshot.fees_paid:,.0f}"
                )
        broker.end_run()
        if session is not None:
            session.commit()
    finally:
        if session is not None:
            session.close()

    _report(broker, run_id, n_rejected, float(info["nav"]))


def _report(broker: Any, run_id: int, n_rejected: int, env_nav: float) -> None:
    """Print the run, and the gap between the broker's NAV and the env's."""
    from trader.env.tax import financial_year_label

    initial = broker.config.initial_cash
    logger.info("─" * 72)
    logger.info(f"run {run_id}  {len(broker.snapshots)} sessions")
    logger.info(f"  NAV            ₹{broker.nav:,.2f}  ({broker.nav / initial - 1:+.2%})")
    logger.info(f"  settled cash   ₹{broker.cash:,.2f}")
    logger.info(f"  unsettled      ₹{broker.unsettled_cash:,.2f}")
    logger.info(f"  equity         ₹{broker.equity_value:,.2f}")
    logger.info(f"  realised P&L   ₹{broker.realised_pnl:,.2f}")
    logger.info(f"  unrealised P&L ₹{broker.unrealised_pnl:,.2f}")
    logger.info(f"  charges paid   ₹{broker.fees_paid:,.2f}")
    orders = broker.get_orders()
    logger.info(
        f"  orders         {len(orders)} ({sum(1 for o in orders if o.status == 'FILLED')}"
        f" filled, {n_rejected} rejected)"
    )
    if n_rejected:
        logger.warning(
            f"  {n_rejected} orders were rejected — usually T+1: a rebalance cannot "
            "spend the same day's sale proceeds."
        )
    for liability in broker.tax_liabilities():
        logger.info(
            f"  tax {financial_year_label(liability.financial_year)}  "
            f"STCG ₹{liability.short_term_gain:,.2f} → ₹{liability.stcg_tax:,.2f}   "
            f"LTCG ₹{liability.long_term_gain:,.2f} → ₹{liability.ltcg_tax:,.2f}   "
            f"total (incl. cess) ₹{liability.total:,.2f}"
        )
    logger.info(
        f"  backtest NAV   ₹{env_nav:,.2f} — the broker is ₹{broker.nav - env_nav:,.2f} "
        "apart; T+1, DP-per-scrip and the dust filter account for the gap"
    )
    logger.info("─" * 72)
    logger.info(f"Inspect it with: uv run python scripts/ledger.py show --run-id {run_id}")


if __name__ == "__main__":
    main()
