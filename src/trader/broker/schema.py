from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import Any

import sqlalchemy as sa
from sqlalchemy import (
    BigInteger,
    Date,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from trader.db.base import Base


class StrategyRun(Base):
    __tablename__ = "strategy_runs"
    __table_args__ = (
        sa.CheckConstraint("mode IN ('paper','live-stub','live')", name="ck_strategy_run_mode"),
        {"schema": "ledger"},
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    strategy_id: Mapped[str] = mapped_column(Text, nullable=False)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=sa.text("now()")
    )
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    config_hash: Mapped[str] = mapped_column(Text, nullable=False)
    mlflow_run_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    initial_cash: Mapped[Decimal] = mapped_column(Numeric(18, 2), nullable=False)
    mode: Mapped[str] = mapped_column(Text, nullable=False)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)


class Order(Base):
    __tablename__ = "orders"
    __table_args__ = (
        sa.CheckConstraint("side IN ('BUY','SELL')", name="ck_order_side"),
        sa.CheckConstraint(
            "status IN ('PENDING','FILLED','PARTIAL','CANCELLED','REJECTED')",
            name="ck_order_status",
        ),
        sa.CheckConstraint("quantity > 0", name="ck_order_quantity_positive"),
        Index("ix_orders_run_ts", "strategy_run_id", "ts_submitted"),
        {"schema": "ledger"},
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    strategy_run_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("ledger.strategy_runs.id"), nullable=False
    )
    client_order_id: Mapped[str] = mapped_column(Text, unique=True, nullable=False)
    ts_submitted: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    symbol: Mapped[str] = mapped_column(Text, nullable=False)
    side: Mapped[str] = mapped_column(Text, nullable=False)
    quantity: Mapped[int] = mapped_column(Integer, nullable=False)
    order_type: Mapped[str] = mapped_column(Text, nullable=False)
    limit_price: Mapped[Decimal | None] = mapped_column(Numeric(18, 4), nullable=True)
    status: Mapped[str] = mapped_column(Text, nullable=False)
    reject_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    tag: Mapped[str | None] = mapped_column(Text, nullable=True)


class Fill(Base):
    __tablename__ = "fills"
    __table_args__ = (
        sa.CheckConstraint("quantity > 0", name="ck_fill_quantity_positive"),
        Index("ix_fills_order_id", "order_id"),
        {"schema": "ledger"},
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    order_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("ledger.orders.id"), nullable=False
    )
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    quantity: Mapped[int] = mapped_column(Integer, nullable=False)
    price: Mapped[Decimal] = mapped_column(Numeric(18, 4), nullable=False)
    fees: Mapped[Decimal] = mapped_column(Numeric(18, 4), nullable=False, server_default="0")
    slippage: Mapped[Decimal] = mapped_column(Numeric(18, 6), nullable=False, server_default="0")


class Position(Base):
    __tablename__ = "positions"
    __table_args__ = (
        UniqueConstraint("strategy_run_id", "symbol"),
        {"schema": "ledger"},
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    strategy_run_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("ledger.strategy_runs.id"), nullable=False
    )
    symbol: Mapped[str] = mapped_column(Text, nullable=False)
    quantity: Mapped[int] = mapped_column(Integer, nullable=False)
    avg_price: Mapped[Decimal] = mapped_column(Numeric(18, 4), nullable=False)
    last_mark: Mapped[Decimal | None] = mapped_column(Numeric(18, 4), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=sa.text("now()")
    )


class PortfolioSnapshot(Base):
    __tablename__ = "portfolio_snapshots"
    __table_args__ = (
        Index("ix_portfolio_snapshots_run_ts", "strategy_run_id", "ts"),
        {"schema": "ledger"},
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    strategy_run_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("ledger.strategy_runs.id"), nullable=False
    )
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    cash: Mapped[Decimal] = mapped_column(Numeric(18, 2), nullable=False)
    equity_value: Mapped[Decimal] = mapped_column(Numeric(18, 2), nullable=False)
    total_value: Mapped[Decimal] = mapped_column(Numeric(18, 2), nullable=False)
    realized_pnl: Mapped[Decimal] = mapped_column(Numeric(18, 2), nullable=False)
    unrealized_pnl: Mapped[Decimal] = mapped_column(Numeric(18, 2), nullable=False)
    fees_paid: Mapped[Decimal] = mapped_column(Numeric(18, 2), nullable=False)
    turnover: Mapped[Decimal] = mapped_column(Numeric(18, 2), nullable=False)
    metrics_json: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")
    )


class PnlDaily(Base):
    __tablename__ = "pnl_daily"
    __table_args__ = {"schema": "ledger"}

    strategy_run_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("ledger.strategy_runs.id"), primary_key=True
    )
    date: Mapped[date] = mapped_column(Date, primary_key=True)
    realized: Mapped[Decimal] = mapped_column(Numeric(18, 2), nullable=False)
    unrealized: Mapped[Decimal] = mapped_column(Numeric(18, 2), nullable=False)
    fees: Mapped[Decimal] = mapped_column(Numeric(18, 2), nullable=False)
    total: Mapped[Decimal] = mapped_column(Numeric(18, 2), nullable=False)
