"""Run Alembic migrations to bring the database to head revision."""

import sys

from alembic.config import Config

from alembic import command


def main() -> None:
    cfg = Config("alembic.ini")
    try:
        command.upgrade(cfg, "head")
        print("Database schema is up to date.")
    except Exception as exc:
        print(f"Migration failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
