-- Bootstrap SQL — run once on first container start.
-- The real schema is managed by Alembic (see src/trader/db/migrations/).
-- This file intentionally left minimal.

CREATE SCHEMA IF NOT EXISTS market;
CREATE SCHEMA IF NOT EXISTS ledger;
