import os

from sqlalchemy import Engine, create_engine


def build_url(
    user: str | None = None,
    password: str | None = None,
    host: str | None = None,
    port: str | None = None,
    db: str | None = None,
) -> str:
    u = user or os.getenv("POSTGRES_USER", "trader")
    p = password or os.getenv("POSTGRES_PASSWORD", "trader")
    h = host or os.getenv("POSTGRES_HOST", "localhost")
    r = port or os.getenv("POSTGRES_PORT", "5432")
    d = db or os.getenv("POSTGRES_DB", "trader")
    return f"postgresql+psycopg://{u}:{p}@{h}:{r}/{d}"


def get_engine(url: str | None = None) -> Engine:
    return create_engine(url or build_url(), pool_pre_ping=True)
