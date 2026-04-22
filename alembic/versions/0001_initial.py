"""Initial schema: market.* and ledger.*

Revision ID: 0001
Revises:
Create Date: 2026-04-22 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # ------------------------------------------------------------------ schemas
    op.execute(sa.text("CREATE SCHEMA IF NOT EXISTS market"))
    op.execute(sa.text("CREATE SCHEMA IF NOT EXISTS ledger"))

    # --------------------------------------------------------------- market.*
    op.execute(
        sa.text("""
    CREATE TABLE market.instruments (
        id          BIGSERIAL PRIMARY KEY,
        symbol      TEXT NOT NULL UNIQUE,
        name        TEXT NOT NULL,
        sector      TEXT NOT NULL,
        sector_id   INTEGER NOT NULL,
        isin        VARCHAR(12) UNIQUE,
        listed_on   DATE,
        delisted_on DATE,
        lot_size    INTEGER NOT NULL DEFAULT 1,
        tick_size   NUMERIC(10,4),
        created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
        updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """)
    )

    op.execute(
        sa.text("""
    CREATE TABLE market.universe_snapshots (
        id             BIGSERIAL PRIMARY KEY,
        universe_name  TEXT NOT NULL,
        effective_date DATE NOT NULL,
        symbols        TEXT[] NOT NULL,
        config_path    TEXT,
        created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
        UNIQUE(universe_name, effective_date)
    )
    """)
    )

    op.execute(
        sa.text("""
    CREATE TABLE market.corporate_actions (
        id          BIGSERIAL PRIMARY KEY,
        symbol      TEXT NOT NULL,
        action_date DATE NOT NULL,
        action_type TEXT NOT NULL
            CHECK (action_type IN ('SPLIT','BONUS','DIVIDEND','MERGER','SUSPENSION','OTHER')),
        ratio       NUMERIC(18,6),
        value       NUMERIC(18,4),
        notes       TEXT,
        UNIQUE(symbol, action_date, action_type)
    )
    """)
    )

    op.execute(
        sa.text("""
    CREATE TABLE market.trading_calendar (
        date           DATE PRIMARY KEY,
        is_trading_day BOOLEAN NOT NULL DEFAULT TRUE,
        notes          TEXT
    )
    """)
    )

    op.execute(
        sa.text("""
    CREATE TABLE market.dataset_versions (
        id            BIGSERIAL PRIMARY KEY,
        name          TEXT NOT NULL,
        split         TEXT NOT NULL,
        file_path     TEXT NOT NULL,
        sha256        TEXT NOT NULL,
        num_dates     INTEGER,
        num_tickers   INTEGER,
        num_features  INTEGER,
        universe_name TEXT,
        start_date    DATE,
        end_date      DATE,
        created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """)
    )

    # --------------------------------------------------------------- ledger.*
    op.execute(
        sa.text("""
    CREATE TABLE ledger.strategy_runs (
        id           BIGSERIAL PRIMARY KEY,
        strategy_id  TEXT NOT NULL,
        started_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
        ended_at     TIMESTAMPTZ,
        config_hash  TEXT NOT NULL,
        mlflow_run_id TEXT,
        initial_cash NUMERIC(18,2) NOT NULL,
        mode         TEXT NOT NULL CHECK (mode IN ('paper','live-stub','live')),
        notes        TEXT
    )
    """)
    )

    op.execute(
        sa.text("""
    CREATE TABLE ledger.orders (
        id               BIGSERIAL PRIMARY KEY,
        strategy_run_id  BIGINT REFERENCES ledger.strategy_runs(id) NOT NULL,
        client_order_id  TEXT UNIQUE NOT NULL,
        ts_submitted     TIMESTAMPTZ NOT NULL,
        symbol           TEXT NOT NULL,
        side             TEXT NOT NULL CHECK (side IN ('BUY','SELL')),
        quantity         INTEGER NOT NULL CHECK (quantity > 0),
        order_type       TEXT NOT NULL,
        limit_price      NUMERIC(18,4),
        status           TEXT NOT NULL
            CHECK (status IN ('PENDING','FILLED','PARTIAL','CANCELLED','REJECTED')),
        reject_reason    TEXT,
        tag              TEXT
    )
    """)
    )
    op.execute(
        sa.text("CREATE INDEX ix_orders_run_ts ON ledger.orders (strategy_run_id, ts_submitted)")
    )

    op.execute(
        sa.text("""
    CREATE TABLE ledger.fills (
        id        BIGSERIAL PRIMARY KEY,
        order_id  BIGINT REFERENCES ledger.orders(id) NOT NULL,
        ts        TIMESTAMPTZ NOT NULL,
        quantity  INTEGER NOT NULL CHECK (quantity > 0),
        price     NUMERIC(18,4) NOT NULL,
        fees      NUMERIC(18,4) NOT NULL DEFAULT 0,
        slippage  NUMERIC(18,6) NOT NULL DEFAULT 0
    )
    """)
    )
    op.execute(sa.text("CREATE INDEX ix_fills_order_id ON ledger.fills (order_id)"))

    op.execute(
        sa.text("""
    CREATE TABLE ledger.positions (
        id               BIGSERIAL PRIMARY KEY,
        strategy_run_id  BIGINT REFERENCES ledger.strategy_runs(id) NOT NULL,
        symbol           TEXT NOT NULL,
        quantity         INTEGER NOT NULL,
        avg_price        NUMERIC(18,4) NOT NULL,
        last_mark        NUMERIC(18,4),
        updated_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
        UNIQUE(strategy_run_id, symbol)
    )
    """)
    )

    op.execute(
        sa.text("""
    CREATE TABLE ledger.portfolio_snapshots (
        id               BIGSERIAL PRIMARY KEY,
        strategy_run_id  BIGINT REFERENCES ledger.strategy_runs(id) NOT NULL,
        ts               TIMESTAMPTZ NOT NULL,
        cash             NUMERIC(18,2) NOT NULL,
        equity_value     NUMERIC(18,2) NOT NULL,
        total_value      NUMERIC(18,2) NOT NULL,
        realized_pnl     NUMERIC(18,2) NOT NULL,
        unrealized_pnl   NUMERIC(18,2) NOT NULL,
        fees_paid        NUMERIC(18,2) NOT NULL,
        turnover         NUMERIC(18,2) NOT NULL,
        metrics_json     JSONB NOT NULL DEFAULT '{}'::jsonb
    )
    """)
    )
    op.execute(
        sa.text(
            "CREATE INDEX ix_portfolio_snapshots_run_ts"
            " ON ledger.portfolio_snapshots (strategy_run_id, ts)"
        )
    )

    op.execute(
        sa.text("""
    CREATE TABLE ledger.pnl_daily (
        strategy_run_id  BIGINT REFERENCES ledger.strategy_runs(id) NOT NULL,
        date             DATE NOT NULL,
        realized         NUMERIC(18,2) NOT NULL,
        unrealized       NUMERIC(18,2) NOT NULL,
        fees             NUMERIC(18,2) NOT NULL,
        total            NUMERIC(18,2) NOT NULL,
        PRIMARY KEY (strategy_run_id, date)
    )
    """)
    )


def downgrade() -> None:
    # ledger tables (FK order: children first)
    op.execute(sa.text("DROP TABLE IF EXISTS ledger.pnl_daily"))
    op.execute(sa.text("DROP TABLE IF EXISTS ledger.portfolio_snapshots"))
    op.execute(sa.text("DROP TABLE IF EXISTS ledger.positions"))
    op.execute(sa.text("DROP TABLE IF EXISTS ledger.fills"))
    op.execute(sa.text("DROP TABLE IF EXISTS ledger.orders"))
    op.execute(sa.text("DROP TABLE IF EXISTS ledger.strategy_runs"))
    # market tables
    op.execute(sa.text("DROP TABLE IF EXISTS market.dataset_versions"))
    op.execute(sa.text("DROP TABLE IF EXISTS market.trading_calendar"))
    op.execute(sa.text("DROP TABLE IF EXISTS market.corporate_actions"))
    op.execute(sa.text("DROP TABLE IF EXISTS market.universe_snapshots"))
    op.execute(sa.text("DROP TABLE IF EXISTS market.instruments"))
