"""Lot-level holding records for capital-gains tax

``ledger.positions`` keeps only quantity + avg_price, which cannot distinguish
an eleven-month holding from a thirteen-month one — and that boundary decides
whether a sale is taxed at 20% (§111A) or 12.5% (§112A).  This table stores
each purchase tranche so sales can be matched FIFO against them.

Revision ID: 0002
Revises: 0001
Create Date: 2026-09-04 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        sa.text("""
    CREATE TABLE ledger.lots (
        id                    BIGSERIAL PRIMARY KEY,
        strategy_run_id       BIGINT REFERENCES ledger.strategy_runs(id) NOT NULL,
        symbol                TEXT NOT NULL,
        buy_date              DATE NOT NULL,
        quantity              INTEGER NOT NULL
            CONSTRAINT ck_lot_quantity_positive CHECK (quantity > 0),
        remaining_quantity    INTEGER NOT NULL
            CONSTRAINT ck_lot_remaining_within_quantity
            CHECK (remaining_quantity >= 0 AND remaining_quantity <= quantity),
        cost_basis_per_share  NUMERIC(18,4) NOT NULL,
        created_at            TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """)
    )
    # FIFO matching reads open lots for one (run, symbol) oldest-first.
    op.execute(
        sa.text(
            "CREATE INDEX ix_lots_run_symbol_buy_date"
            " ON ledger.lots (strategy_run_id, symbol, buy_date)"
        )
    )


def downgrade() -> None:
    op.execute(sa.text("DROP INDEX IF EXISTS ledger.ix_lots_run_symbol_buy_date"))
    op.execute(sa.text("DROP TABLE IF EXISTS ledger.lots"))
