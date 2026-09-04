#!/usr/bin/env python
"""Build aligned feature panels for train / val / test splits.

Usage:
    python scripts/build_features.py data=universe_v1   # yfinance → data/panels
    python scripts/build_features.py data=kite_v1       # Kite     → data/panels_kite

Everything that varies between those two datasets — the OHLCV store, the output
directory, the split boundaries — comes from the ``data`` config group, defaulting
to the values the original yfinance panel was built with. That is on purpose: two
near-identical copies of this script would drift, and the differences between the
datasets are exactly what the comparison in ``reports/kite_vs_yfinance.md``
measures, so they belong in config where they can be read off.
"""
from __future__ import annotations

import hashlib
import os
from datetime import date, datetime
from pathlib import Path

import hydra
from loguru import logger
from omegaconf import DictConfig

# Split boundaries (inclusive), used when the data config does not override them.
# 1-month purge gaps between segments, cut from the START of each later segment so
# that no 60-day feature window straddles a boundary.
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
    from trader.data.corporate_actions import (
        DEFAULT_MOVE_THRESHOLD,
        detect_extreme_moves,
        format_triage_report,
        mask_corporate_events,
    )
    from trader.data.features import FEATURE_COLS, compute_features
    from trader.data.storage import OhlcvStore
    from trader.data.universe import all_tickers, sector_id_of

    orig_cwd = Path(hydra.utils.get_original_cwd())
    parquet_root = orig_cwd / str(cfg.data.parquet_root)
    panels_root = orig_cwd / str(cfg.data.get("panels_root", "data/panels"))
    panels_root.mkdir(parents=True, exist_ok=True)

    store = OhlcvStore(root=parquet_root)
    tickers = all_tickers()
    sector_ids = {t: sector_id_of(t) for t in tickers}

    # OhlcvStore.load defaults to scanning from 2010 when no start is given, which
    # would silently amputate the five extra years the Kite fetch went and got.
    # Pass the configured start explicitly so the panel spans what was fetched.
    start = datetime.fromisoformat(str(cfg.data.start_date))
    logger.info(f"Loading OHLCV for {len(tickers)} tickers from {start.date()} ...")
    ohlcv = store.load(tickers=tickers, start=start)

    if ohlcv.is_empty():
        logger.error("No OHLCV data found. Run scripts/fetch_data.py first.")
        return

    logger.info(
        f"Loaded {len(ohlcv):,} rows  {ohlcv['date'].min()} → {ohlcv['date'].max()}"
    )

    # Try loading NIFTY 50 index for beta computation (optional). Same start as
    # the panel: a shorter index series leaves beta null on the uncovered dates,
    # and compute_features turns a null feature into is_tradeable=False — so a
    # 2010-limited index would quietly delete 2005-2009 from the panel.
    index_rets: pl.DataFrame | None = _load_index_rets(store, start)

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

    # ── corporate actions ────────────────────────────────────────────────────
    # Detection is diagnostic and always runs: the triage list is how a *new*
    # demerger gets noticed at all. Masking only ever touches dates that are
    # explicitly listed in KNOWN_CORPORATE_EVENTS, so an unexplained -45% day stays
    # in the data until a human confirms what it was.
    threshold = float(cfg.data.get("extreme_move_threshold", DEFAULT_MOVE_THRESHOLD))
    candidates = detect_extreme_moves(panel, threshold)
    for line in format_triage_report(candidates, threshold):
        logger.info(line)
    _write_triage(candidates, panels_root)

    if bool(cfg.data.get("mask_corporate_actions", True)):
        panel, ca_audit = mask_corporate_events(panel)
        for row in ca_audit.iter_rows(named=True):
            state = (
                f"masked {row['rows_masked']:,} rows {row['mask_start']} → {row['mask_end']}"
                if row["matched"]
                else "outside this panel's date range — not masked"
            )
            logger.info(f"  {row['kind']:9s} {row['ticker']:18s} {row['event_date']}  {state}")
        logger.info(f"After masking | tradeable: {panel['is_tradeable'].sum():,}")
    else:
        logger.warning("mask_corporate_actions=false — known demergers left in the data")

    universe_name = str(cfg.data.get("universe", "nifty50_sectoral"))
    train_end = _split_date(cfg, "train_end", _TRAIN_END)
    val_start = _split_date(cfg, "val_start", _VAL_START)
    val_end = _split_date(cfg, "val_end", _VAL_END)
    test_start = _split_date(cfg, "test_start", _TEST_START)

    logger.info(
        f"Splits — train: … → {train_end} | val: {val_start} → {val_end} "
        f"| test: {test_start} → …"
    )
    _write_split(panel, panels_root, "train", None, train_end, universe_name)
    _write_split(panel, panels_root, "val", val_start, val_end, universe_name)
    _write_split(panel, panels_root, "test", test_start, None, universe_name)

    logger.info("Done.")


# ── helpers ────────────────────────────────────────────────────────────────────


def _split_date(cfg: DictConfig, key: str, fallback: date) -> date:
    """Read a split boundary from the data config, falling back to the built-in."""
    raw = cfg.data.get(key)
    return fallback if raw is None else date.fromisoformat(str(raw))


def _write_triage(candidates: object, out_dir: Path) -> None:
    """Persist the extreme-move candidates next to the panels.

    A log line scrolls away; the next person to ask "was that -45% day real?" needs
    the table, not the transcript of the run that produced it.
    """
    import polars as pl

    if not isinstance(candidates, pl.DataFrame):
        return
    path = out_dir / "corporate_action_triage.parquet"
    candidates.write_parquet(path)
    logger.info(f"  triage candidates → {path} ({len(candidates)} rows)")


def _load_index_rets(store: object, start: datetime) -> object | None:
    """Try to load ^NSEI (NIFTY 50) log returns from the OHLCV store."""
    import polars as pl

    from trader.data.storage import OhlcvStore

    if not isinstance(store, OhlcvStore):
        return None
    try:
        nsei = store.load(tickers=["^NSEI"], start=start)
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
