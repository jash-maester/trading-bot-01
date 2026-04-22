#!/usr/bin/env python
"""Build the trading universe and persist to DB + local parquet snapshot.

Usage:
    python scripts/build_universe.py data=universe_v1
"""
from __future__ import annotations

import os
from datetime import date
from pathlib import Path

import hydra
from loguru import logger
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    import hydra.utils
    import polars as pl

    from trader.data.universe import all_tickers, sector_id_of, sector_of

    orig_cwd = Path(hydra.utils.get_original_cwd())
    tickers = all_tickers()
    effective = date.fromisoformat(str(cfg.data.start_date))

    logger.info(f"Universe: {len(tickers)} tickers  effective_date={effective}")

    # ------------------------------------------------------------------ snapshot
    snapshot_path = orig_cwd / "data" / "raw" / "universe_v1.parquet"
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "ticker": tickers,
            "sector": [sector_of(t) or "unknown" for t in tickers],
            "sector_id": [sector_id_of(t) for t in tickers],
        }
    ).write_parquet(snapshot_path)
    logger.info(f"Parquet snapshot: {snapshot_path}")

    # ------------------------------------------------------------------ DB (optional)
    try:
        from sqlalchemy import create_engine
        from sqlalchemy.orm import Session

        from trader.db.market_models import UniverseSnapshot

        url = (
            f"postgresql+psycopg://"
            f"{os.getenv('POSTGRES_USER', 'trader')}:"
            f"{os.getenv('POSTGRES_PASSWORD', 'trader')}@"
            f"{os.getenv('POSTGRES_HOST', 'localhost')}:"
            f"{os.getenv('POSTGRES_PORT', '5432')}/"
            f"{os.getenv('POSTGRES_DB', 'trader')}"
        )
        engine = create_engine(url, pool_pre_ping=True)
        with Session(engine) as session:
            snap = UniverseSnapshot(
                universe_name=str(cfg.data.universe),
                effective_date=effective,
                symbols=tickers,
                config_path=str(snapshot_path),
            )
            session.add(snap)
            session.commit()
        logger.info(f"DB row written — universe_name={cfg.data.universe}")
    except Exception as exc:
        logger.warning(f"DB write skipped (run make db-up && make db-migrate first): {exc}")


if __name__ == "__main__":
    main()
