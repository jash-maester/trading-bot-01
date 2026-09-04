#!/usr/bin/env python
"""Read the paper/live ledger: what a run did, how it performed, how two compare.

``scripts/paper_run.py`` writes ``ledger.strategy_runs`` → ``orders`` → ``fills``
→ ``lots`` → ``positions`` → ``portfolio_snapshots`` → ``pnl_daily`` and then
exits.  This is the other half: the read side, so that a run can be inspected
without opening psql and without re-running anything.

It computes no metrics of its own.  ``trader.training.quantstats_report``
already turns a daily return series into 46 QuantStats metrics plus an HTML
tearsheet, and it is what training runs are scored with — so a paper run scored
by a second, subtly different implementation would not be comparable to them.
The only thing this module does is turn the NAV column of
``portfolio_snapshots`` into the log-return series that module expects.

Postgres credentials come from the environment, as everywhere else here::

    set -a && . ./.env && set +a
    uv run python scripts/ledger.py show
    uv run python scripts/ledger.py show --run-id 12
    uv run python scripts/ledger.py report --run-id 12 --out reports/run12.html
    uv run python scripts/ledger.py compare --run-id 12 --run-id 13
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import typer

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Inspect and report on runs recorded in the ledger schema.",
)

# The console table.  The tearsheet carries all 46; these are the ones worth
# reading in a terminal, in the order you actually read them.
CONSOLE_METRICS: tuple[tuple[str, str, str], ...] = (
    ("cagr", "CAGR", "pct"),
    ("sharpe", "Sharpe", "num"),
    ("sortino", "Sortino", "num"),
    ("calmar", "Calmar", "num"),
    ("max_drawdown", "Max drawdown", "pct"),
    ("volatility", "Volatility (ann)", "pct"),
    ("ulcer_index", "Ulcer index", "num"),
    ("value_at_risk", "VaR (daily)", "pct"),
    ("win_rate", "Win rate", "pct"),
    ("profit_factor", "Profit factor", "num"),
    ("tail_ratio", "Tail ratio", "num"),
    ("kelly_criterion", "Kelly", "num"),
)


def _session() -> Any:
    from sqlalchemy.orm import Session

    from trader.db.engine import get_engine

    try:
        engine = get_engine()
        with engine.connect():
            pass
    except Exception as exc:  # noqa: BLE001 — a CLI should explain, not traceback
        typer.echo(f"cannot reach Postgres: {type(exc).__name__}: {exc}")
        typer.echo("source the credentials first:  set -a && . ./.env && set +a")
        raise typer.Exit(code=1) from exc
    return Session(engine)


def _fmt(value: float | None, kind: str) -> str:
    if value is None or not math.isfinite(value):
        return "—"
    if kind == "pct":
        return f"{value:>9.2%}"
    return f"{value:>9.3f}"


def _run_or_exit(session: Any, run_id: int) -> Any:
    from trader.broker import schema

    run = session.get(schema.StrategyRun, run_id)
    if run is None:
        typer.echo(f"no run {run_id} in ledger.strategy_runs")
        raise typer.Exit(code=1)
    return run


def _snapshots(session: Any, run_id: int) -> list[Any]:
    from sqlalchemy import select

    from trader.broker import schema

    return list(
        session.scalars(
            select(schema.PortfolioSnapshot)
            .where(schema.PortfolioSnapshot.strategy_run_id == run_id)
            .order_by(schema.PortfolioSnapshot.ts)
        )
    )


def _log_returns(snapshots: list[Any]) -> list[float]:
    """Daily log returns of the NAV column.

    Log, not simple, because that is the series
    ``trader.training.quantstats_report`` takes — it converts back to simple
    returns at its own boundary, and doing the conversion twice would compound
    an error into every metric.
    """
    navs = [float(s.total_value) for s in snapshots]
    return [
        math.log(max(b, 1e-12) / max(a, 1e-12))
        for a, b in zip(navs[:-1], navs[1:], strict=True)
    ]


def _counts(session: Any, run_id: int) -> dict[str, int]:
    from sqlalchemy import func, select

    from trader.broker import schema

    orders = select(schema.Order).where(schema.Order.strategy_run_id == run_id).subquery()
    return {
        "orders": int(session.scalar(select(func.count()).select_from(orders)) or 0),
        "filled": int(
            session.scalar(
                select(func.count())
                .select_from(orders)
                .where(orders.c.status == "FILLED")
            )
            or 0
        ),
        "rejected": int(
            session.scalar(
                select(func.count())
                .select_from(orders)
                .where(orders.c.status == "REJECTED")
            )
            or 0
        ),
        "fills": int(
            session.scalar(
                select(func.count())
                .select_from(schema.Fill)
                .join(orders, schema.Fill.order_id == orders.c.id)
            )
            or 0
        ),
        "lots": int(
            session.scalar(
                select(func.count())
                .select_from(schema.Lot)
                .where(schema.Lot.strategy_run_id == run_id)
            )
            or 0
        ),
    }


@app.command()
def show(run_id: int | None = typer.Option(None, help="Run to detail; omit to list all")) -> None:
    """List runs, or show one run's positions, orders and accrued tax."""
    from sqlalchemy import select

    from trader.broker import schema

    with _session() as session:
        if run_id is None:
            runs = list(
                session.scalars(
                    select(schema.StrategyRun).order_by(schema.StrategyRun.id.desc())
                )
            )
            if not runs:
                typer.echo("ledger.strategy_runs is empty — run scripts/paper_run.py")
                return
            typer.echo(
                f"{'id':>5}  {'mode':<10} {'strategy':<24} {'started':<19} "
                f"{'sessions':>8} {'initial':>13} {'final NAV':>14}"
            )
            for run in runs:
                snapshots = _snapshots(session, run.id)
                final = float(snapshots[-1].total_value) if snapshots else float("nan")
                typer.echo(
                    f"{run.id:>5}  {run.mode:<10} {run.strategy_id[:24]:<24} "
                    f"{run.started_at:%Y-%m-%d %H:%M:%S}  {len(snapshots):>8} "
                    f"{float(run.initial_cash):>13,.0f} {final:>14,.2f}"
                )
            return

        run = _run_or_exit(session, run_id)
        snapshots = _snapshots(session, run_id)
        counts = _counts(session, run_id)
        initial = float(run.initial_cash)

        typer.echo(f"run {run.id}  strategy={run.strategy_id}  mode={run.mode}")
        typer.echo(f"  config hash    {run.config_hash}")
        typer.echo(f"  started        {run.started_at}")
        typer.echo(f"  ended          {run.ended_at or '(still open)'}")
        typer.echo(f"  sessions       {len(snapshots)}")
        typer.echo(f"  initial cash   ₹{initial:,.2f}")
        if not snapshots:
            typer.echo("  no snapshots — the run wrote no sessions")
            return

        last = snapshots[-1]
        nav = float(last.total_value)
        typer.echo(f"  final NAV      ₹{nav:,.2f}  ({nav / initial - 1:+.2%})")
        typer.echo(f"  cash           ₹{float(last.cash):,.2f}")
        typer.echo(f"  equity         ₹{float(last.equity_value):,.2f}")
        typer.echo(f"  realised P&L   ₹{float(last.realized_pnl):,.2f}")
        typer.echo(f"  unrealised P&L ₹{float(last.unrealized_pnl):,.2f}")
        typer.echo(f"  charges paid   ₹{float(last.fees_paid):,.2f}")
        typer.echo(
            f"  orders         {counts['orders']} ({counts['filled']} filled, "
            f"{counts['rejected']} rejected, {counts['fills']} fills, "
            f"{counts['lots']} lots)"
        )

        # Tax is not a ledger table — it is an annual liability the broker
        # accrues and parks in the snapshot's metrics_json.  See paper_broker.
        metrics = last.metrics_json or {}
        by_fy = metrics.get("tax_by_fy") or {}
        if by_fy:
            typer.echo(f"  tax accrued    ₹{float(metrics.get('tax_accrued', 0.0)):,.2f}")
            for label, row in sorted(by_fy.items()):
                typer.echo(
                    f"    {label}  STCG ₹{row['short_term_gain']:>12,.2f}  "
                    f"LTCG ₹{row['long_term_gain']:>12,.2f}  "
                    f"tax ₹{row['tax']:>10,.2f}"
                )

        positions = list(
            session.scalars(
                select(schema.Position)
                .where(schema.Position.strategy_run_id == run_id)
                .where(schema.Position.quantity != 0)
                .order_by(schema.Position.symbol)
            )
        )
        if positions:
            typer.echo(f"\n  {len(positions)} open positions")
            typer.echo(
                f"    {'symbol':<16} {'qty':>8} {'avg':>12} {'mark':>12} {'value':>14}"
            )
            for p in positions:
                mark = float(p.last_mark or 0.0)
                typer.echo(
                    f"    {p.symbol:<16} {p.quantity:>8} {float(p.avg_price):>12,.2f} "
                    f"{mark:>12,.2f} {p.quantity * mark:>14,.2f}"
                )


@app.command()
def report(
    run_id: int = typer.Option(..., help="Run to score"),
    out: Path | None = typer.Option(None, help="Where to write the HTML tearsheet"),
    benchmark: int | None = typer.Option(None, help="Run id to plot the tearsheet against"),
) -> None:
    """Score a run with QuantStats and write its HTML tearsheet."""
    from trader.training.quantstats_report import aggregate_quantstats, save_tearsheet

    with _session() as session:
        run = _run_or_exit(session, run_id)
        returns = _log_returns(_snapshots(session, run_id))
        bench = _log_returns(_snapshots(session, benchmark)) if benchmark else None

    if len(returns) < 2:
        typer.echo(f"run {run_id} has fewer than two sessions — nothing to score")
        raise typer.Exit(code=1)

    # One "episode": a paper run is a single continuous track record, not a
    # sample of independent windows the way a training evaluation is.
    metrics = aggregate_quantstats([returns])
    typer.echo(f"run {run_id}  {run.strategy_id}  {len(returns) + 1} sessions")
    for key, label, kind in CONSOLE_METRICS:
        typer.echo(f"  {label:<18}{_fmt(metrics.get(key), kind)}")
    typer.echo(f"  ({len(metrics)} metrics computed; the tearsheet carries all of them)")

    out_path = out or Path("reports") / f"run{run_id}_tearsheet.html"
    written = save_tearsheet(
        returns,
        out_path,
        title=f"{run.strategy_id} (ledger run {run_id})",
        benchmark_log_returns=bench,
    )
    if written is None:
        # Tearsheet generation is best-effort upstream too — the metrics above
        # are the deliverable, the HTML is a convenience.
        typer.echo("tearsheet could not be rendered (matplotlib/quantstats); metrics stand")
    else:
        typer.echo(f"tearsheet: {written}")


@app.command()
def compare(
    run_id: list[int] = typer.Option(..., "--run-id", help="Repeat for each run to compare"),
) -> None:
    """Put two or more runs side by side on the same QuantStats metrics."""
    from trader.training.quantstats_report import aggregate_quantstats

    if len(run_id) < 2:
        typer.echo("compare needs at least two --run-id values")
        raise typer.Exit(code=1)

    columns: list[tuple[int, str, dict[str, float], float, float]] = []
    with _session() as session:
        for rid in run_id:
            run = _run_or_exit(session, rid)
            snapshots = _snapshots(session, rid)
            returns = _log_returns(snapshots)
            if len(returns) < 2:
                typer.echo(f"run {rid} has fewer than two sessions — skipped")
                continue
            initial = float(run.initial_cash)
            columns.append(
                (
                    rid,
                    run.strategy_id,
                    aggregate_quantstats([returns]),
                    float(snapshots[-1].total_value) / initial - 1.0,
                    float(snapshots[-1].fees_paid),
                )
            )
    if len(columns) < 2:
        typer.echo("not enough scoreable runs")
        raise typer.Exit(code=1)

    width = 14
    header = "  " + f"{'metric':<18}" + "".join(f"{f'run {c[0]}':>{width}}" for c in columns)
    typer.echo(header)
    typer.echo("  " + "-" * (len(header) - 2))
    typer.echo(
        "  "
        + f"{'strategy':<18}"
        + "".join(f"{c[1][: width - 1]:>{width}}" for c in columns)
    )
    typer.echo(
        "  " + f"{'total return':<18}" + "".join(f"{c[3]:>{width}.2%}" for c in columns)
    )
    typer.echo(
        "  " + f"{'charges paid':<18}" + "".join(f"{c[4]:>{width},.0f}" for c in columns)
    )
    for key, label, kind in CONSOLE_METRICS:
        cells = "".join(f"{_fmt(c[2].get(key), kind):>{width}}" for c in columns)
        typer.echo("  " + f"{label:<18}" + cells)


if __name__ == "__main__":
    app()
