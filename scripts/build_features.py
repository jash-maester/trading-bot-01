#!/usr/bin/env python
"""Build aligned feature panels for train / val / test splits.

Usage:
    python scripts/build_features.py data=universe_v1
"""
from __future__ import annotations

import hashlib
import os
from datetime import date
from pathlib import Path

import hydra
from loguru import logger
from omegaconf import DictConfig

# Split boundaries (inclusive). 1-month purge gaps between segments.
_TRAIN_END = date(2021, 12, 31)
_VAL_START = date(2022, 2, 1)   # Jan 2022 purged
_VAL_END = date(2022, 12, 31)
_TEST_START = date(2023, 2, 1)  # Jan 2023 purged
# _TEST_END: no upper bound — use all available data


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    import hydra.utils
    import polars as pl

    from trader.data.alignment import align_panel
    from trader.data.calendar import build_calendar
    from trader.data.features import FEATURE_COLS, compute_features
    from trader.data.storage import OhlcvStore
    from trader.data.universe import all_tickers, sector_id_of

    orig_cwd = Path(hydra.utils.get_original_cwd())
    parquet_root = orig_cwd / str(cfg.data.parquet_root)
    panels_root = orig_cwd / "data" / "panels"
    panels_root.mkdir(parents=True, exist_ok=True)

    store = OhlcvStore(root=parquet_root)
    tickers = all_tickers()
    sector_ids = {t: sector_id_of(t) for t in tickers}

    logger.info(f"Loading OHLCV for {len(tickers)} tickers ...")
    ohlcv = store.load(tickers=tickers)

    if ohlcv.is_empty():
        logger.error("No OHLCV data found. Run scripts/fetch_data.py first.")
        return

    logger.info(
        f"Loaded {len(ohlcv):,} rows  {ohlcv['date'].min()} → {ohlcv['date'].max()}"
    )

    # Try loading NIFTY 50 index for beta computation (optional)
    index_rets: pl.DataFrame | None = _load_index_rets(store)

    calendar = build_calendar(ohlcv)
    logger.info(f"Calendar: {calendar[0]} .. {calendar[-1]}  ({len(calendar)} days)")

    logger.info("Aligning panel ...")
    panel = align_panel(ohlcv, tickers, calendar, sector_ids=sector_ids)

    logger.info("Computing features ...")
    panel = compute_features(panel, index_rets=index_rets)

    n_feat = len([c for c in FEATURE_COLS if c in panel.columns])
    logger.info(
        f"Panel: {panel['date'].n_unique()} days"
        f" × {panel['ticker'].n_unique()} tickers"
        f" × {n_feat} features"
        f" | tradeable: {panel['is_tradeable'].sum():,}"
    )

    universe_name = str(cfg.data.get("universe", "nifty50_sectoral"))

    _write_split(panel, panels_root, "train", None, _TRAIN_END, universe_name)
    _write_split(panel, panels_root, "val", _VAL_START, _VAL_END, universe_name)
    _write_split(panel, panels_root, "test", _TEST_START, None, universe_name)

    logger.info("Done.")


# ── helpers ────────────────────────────────────────────────────────────────────


def _load_index_rets(store: object) -> object | None:
    """Try to load ^NSEI (NIFTY 50) log returns from the OHLCV store."""
    import polars as pl

    from trader.data.storage import OhlcvStore

    if not isinstance(store, OhlcvStore):
        return None
    try:
        nsei = store.load(tickers=["^NSEI"])
        if not nsei.is_empty():
            nsei = nsei.sort("date").with_columns(
                (pl.col("adj_close") / pl.col("adj_close").shift(1))
                .log()
                .alias("index_return")
            )
            return nsei.select(["date", "index_return"]).drop_nulls()
    except Exception:
        pass
    return None


def _write_split(
    panel: object,
    out_dir: Path,
    name: str,
    start: date | None,
    end: date | None,
    universe_name: str,
) -> None:
    import polars as pl

    if not isinstance(panel, pl.DataFrame):
        return

    mask = pl.lit(value=True)
    if start is not None:
        mask = mask & (pl.col("date") >= start)
    if end is not None:
        mask = mask & (pl.col("date") <= end)

    split = panel.filter(mask).sort(["date", "ticker"])

    path = out_dir / f"{name}.parquet"
    split.write_parquet(path)

    sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    (out_dir / f"{name}.sha256").write_text(sha256 + "\n")

    logger.info(
        f"  {name}.parquet: {len(split):,} rows"
        f" ({split['date'].n_unique()} days)"
        f"  SHA256={sha256[:16]}..."
    )

    _write_db_row(
        name=name,
        split_df=split,
        file_path=str(path),
        sha256=sha256,
        universe_name=universe_name,
    )


def _write_db_row(
    name: str,
    split_df: object,
    file_path: str,
    sha256: str,
    universe_name: str,
) -> None:
    import polars as pl

    if not isinstance(split_df, pl.DataFrame):
        return

    try:
        from sqlalchemy import create_engine
        from sqlalchemy.orm import Session

        from trader.db.market_models import DatasetVersion

        url = (
            f"postgresql+psycopg://"
            f"{os.getenv('POSTGRES_USER', 'trader')}:"
            f"{os.getenv('POSTGRES_PASSWORD', 'trader')}@"
            f"{os.getenv('POSTGRES_HOST', 'localhost')}:"
            f"{os.getenv('POSTGRES_PORT', '5432')}/"
            f"{os.getenv('POSTGRES_DB', 'trader')}"
        )
        engine = create_engine(url, pool_pre_ping=True)

        date_col = split_df["date"]
        start_d = date_col.min()
        end_d = date_col.max()
        n_features = len(
            [c for c in split_df.columns if c not in ("date", "ticker", "is_tradeable")]
        )

        with Session(engine) as session:
            dv = DatasetVersion(
                name=f"{universe_name}_{name}",
                split=name,
                file_path=file_path,
                sha256=sha256,
                num_dates=split_df["date"].n_unique(),
                num_tickers=split_df["ticker"].n_unique(),
                num_features=n_features,
                universe_name=universe_name,
                start_date=start_d,
                end_date=end_d,
            )
            session.add(dv)
            session.commit()
        logger.info(f"  DB row written for split={name}")
    except Exception as exc:
        logger.warning(f"  DB write skipped (run make db-up && make db-migrate first): {exc}")


if __name__ == "__main__":
    main()
