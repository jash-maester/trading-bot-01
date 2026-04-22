from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal

import sqlalchemy as sa
from sqlalchemy import (
    BigInteger,
    Boolean,
    Date,
    DateTime,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import Mapped, mapped_column

from trader.db.base import Base


class Instrument(Base):
    __tablename__ = "instruments"
    __table_args__ = {"schema": "market"}

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(Text, unique=True, nullable=False)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    sector: Mapped[str] = mapped_column(Text, nullable=False)
    sector_id: Mapped[int] = mapped_column(Integer, nullable=False)
    isin: Mapped[str | None] = mapped_column(String(12), unique=True, nullable=True)
    listed_on: Mapped[date | None] = mapped_column(Date, nullable=True)
    delisted_on: Mapped[date | None] = mapped_column(Date, nullable=True)
    lot_size: Mapped[int] = mapped_column(Integer, nullable=False, server_default="1")
    tick_size: Mapped[Decimal | None] = mapped_column(Numeric(10, 4), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=sa.text("now()")
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=sa.text("now()")
    )


class UniverseSnapshot(Base):
    __tablename__ = "universe_snapshots"
    __table_args__ = (
        UniqueConstraint("universe_name", "effective_date"),
        {"schema": "market"},
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    universe_name: Mapped[str] = mapped_column(Text, nullable=False)
    effective_date: Mapped[date] = mapped_column(Date, nullable=False)
    symbols: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False)
    config_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=sa.text("now()")
    )


class CorporateAction(Base):
    __tablename__ = "corporate_actions"
    __table_args__ = (
        UniqueConstraint("symbol", "action_date", "action_type"),
        sa.CheckConstraint(
            "action_type IN ('SPLIT','BONUS','DIVIDEND','MERGER','SUSPENSION','OTHER')",
            name="ck_corporate_action_type",
        ),
        {"schema": "market"},
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(Text, nullable=False)
    action_date: Mapped[date] = mapped_column(Date, nullable=False)
    action_type: Mapped[str] = mapped_column(Text, nullable=False)
    ratio: Mapped[Decimal | None] = mapped_column(Numeric(18, 6), nullable=True)
    value: Mapped[Decimal | None] = mapped_column(Numeric(18, 4), nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)


class TradingCalendarDay(Base):
    __tablename__ = "trading_calendar"
    __table_args__ = {"schema": "market"}

    date: Mapped[date] = mapped_column(Date, primary_key=True)
    is_trading_day: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="true")
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)


class DatasetVersion(Base):
    __tablename__ = "dataset_versions"
    __table_args__ = {"schema": "market"}

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    split: Mapped[str] = mapped_column(Text, nullable=False)
    file_path: Mapped[str] = mapped_column(Text, nullable=False)
    sha256: Mapped[str] = mapped_column(Text, nullable=False)
    num_dates: Mapped[int | None] = mapped_column(Integer, nullable=True)
    num_tickers: Mapped[int | None] = mapped_column(Integer, nullable=True)
    num_features: Mapped[int | None] = mapped_column(Integer, nullable=True)
    universe_name: Mapped[str | None] = mapped_column(Text, nullable=True)
    start_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    end_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=sa.text("now()")
    )
