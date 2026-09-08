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
# 3-month purge gaps between segments, cut from the START of each later segment so
# that no feature window straddles a boundary.
#
# It used to be 1 month. That is 17–23 NSE trading days (measured over
# data/kite_ohlcv, 2014–2026), against `realized_vol_60d` and `beta_nifty_60d`,
# which look back 60. Every boundary therefore had ~40 contaminated rows on each
# side. 3 calendar months is the shortest gap that clears 60 trading days
# (62 over Jan–Mar 2024 and Jan–Mar 2025); 4 gives margin — the worst 3-month
# span in the last decade, Feb–Apr 2014, is only 58.
#
# NOTE: the panels currently on disk under data/panels were built with the old
# 1-month gap and are contaminated at both boundaries. They need a rebuild.
_TRAIN_END = date(2021, 12, 31)
_VAL_START = date(2022, 4, 1)   # Jan–Mar 2022 purged
_VAL_END = date(2022, 12, 31)
_TEST_START = date(2023, 4, 1)  # Jan–Mar 2023 purged
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
    from trader.data.features import FEATURE_COLS, compute_features, resolve_panels_root
    from trader.data.storage import OhlcvStore
    from trader.data.universe import all_tickers, sector_id_of

    orig_cwd = Path(hydra.utils.get_original_cwd())
    parquet_root = orig_cwd / str(cfg.data.parquet_root)
    panels_root = resolve_panels_root(cfg, orig_cwd)
    panels_root.mkdir(parents=True, exist_ok=True)

    store = OhlcvStore(root=parquet_root)

    # Where the universe and its sector ids come from. Absent or "sectoral" is
    # the historical behaviour, unchanged: the 504 hand-curated names and
    # `sector_id_of`. "store" is for the point-in-time rebuild, where the
    # universe is every name the bhavcopy store holds and sector ids come from
    # NSE's own classification.
    #
    # These MUST travel together. `sector_id_of` returns 0 for anything outside
    # SECTOR_MAP, and 0 is the phantom sector `CLAUDE.md` records; a wide
    # universe scored with the narrow sector function would put hundreds of
    # names into it at once. So selecting "store" without an industry file is
    # refused rather than defaulted.
    source = str(cfg.data.get("universe_source", "sectoral"))
    if source == "sectoral":
        tickers = all_tickers()
        sector_ids = {t: sector_id_of(t) for t in tickers}
    elif source == "store":
        industry_file = cfg.data.get("industry_file", None)
        if not industry_file:
            logger.error(
                "data.universe_source='store' needs data.industry_file. Without "
                "it every name outside SECTOR_MAP would take sector_id 0 — the "
                "phantom sector in CLAUDE.md — and there would be hundreds of "
                "them. Run scripts/fetch_industries.py."
            )
            raise SystemExit(1)
        from trader.data.nse_industry import UNKNOWN_INDUSTRY_ID, industry_ids

        ind = pl.read_parquet(orig_cwd / str(industry_file))
        # The store also holds ^NSEI, which `_load_index_rets` reads separately
        # for beta. It is a benchmark, not something to hold: leaving it in the
        # universe would add a column to the action space that can never be
        # traded, and hand it an "unknown industry" id on the way.
        tickers = [t for t in store.tickers() if not t.startswith("^")]
        sector_ids = industry_ids(ind, tickers)
        n_unknown = sum(1 for v in sector_ids.values() if v == UNKNOWN_INDUSTRY_ID)
        logger.info(
            f"universe from {parquet_root}: {len(tickers):,} tickers, "
            f"{n_unknown:,} in the unknown industry bucket "
            f"({UNKNOWN_INDUSTRY_ID})"
        )
    else:
        logger.error(f"data.universe_source must be 'sectoral' or 'store', got {source!r}")
        raise SystemExit(1)

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

    # NIFTY 50 index returns for beta. NOT optional: compute_features raises
    # MissingBenchmarkError rather than let beta_nifty_60d collapse to a constant
    # 1.0, which is what silently happened to data/panels (data/ohlcv has no
    # ^NSEI at all). Same start as the panel: a shorter index series leaves beta
    # null on the uncovered dates, and compute_features turns a null feature into
    # is_tradeable=False — so a 2010-limited index would delete 2005-2009 from
    # the panel. That is now logged as a warning by _add_beta, not left to be
    # noticed in a row count.
    index_rets: pl.DataFrame | None = _load_index_rets(store, start)
    if index_rets is None:
        logger.error(
            f"No benchmark index returns found in {parquet_root}. "
            "beta_nifty_60d cannot be computed and will NOT be faked as 1.0. "
            "Fetch the index first: scripts/fetch_data.py with "
            "data.fetch_index=true (data=kite_v1 already sets it)."
        )
        return

    calendar = build_calendar(ohlcv)
    logger.info(f"Calendar: {calendar[0]} .. {calendar[-1]}  ({len(calendar)} days)")

    logger.info("Aligning panel ...")
    panel = align_panel(ohlcv, tickers, calendar, sector_ids=sector_ids)

    logger.info("Computing features ...")
    panel = compute_features(panel, index_rets=index_rets)

    # ── point-in-time universe ───────────────────────────────────────────────
    # `align_panel` marks a name untradeable outside its listing span, and
    # `compute_features` marks it untradeable during feature warm-up. Neither
    # asks whether it was worth trading. Without this third belt the panel is
    # still a fixed universe -- every name that ever cleared the bar, tradeable
    # on every day it was listed -- which is the selection bias the rebuild
    # exists to remove.
    #
    # Eligibility is decided on a monthly grid and carried forward, because that
    # is when the allocator actually reselects. A name admitted at a rebalance
    # stays in the book until the next one, exactly as it would in the env.
    pit_cfg = cfg.data.get("pit_universe", None)
    if pit_cfg is not None:
        from trader.data.pit_universe import (
            LiquidityRule,
            apply_monthly_mask,
            universe_schedule,
        )

        rule = LiquidityRule(
            min_median_turnover=float(pit_cfg.get("min_median_turnover", 5e7)),
            lookback_days=int(pit_cfg.get("lookback_days", 365)),
            min_sessions=int(pit_cfg.get("min_sessions", 100)),
            min_price=float(pit_cfg.get("min_price", 5.0)),
            max_names=(int(pit_cfg["max_names"]) if pit_cfg.get("max_names") else None),
        )
        logger.info(f"Point-in-time universe: {rule.describe()}")
        # The rule reads raw turnover, which lives on the OHLCV rows rather than
        # on the aligned panel (alignment fills absent names with zeros, and a
        # zero-turnover row would look like an illiquid name rather than an
        # absent one).
        liq = ohlcv.select(["date", "ticker", "close", "turnover"]).with_columns(
            pl.lit("EQ").alias("series")
        )
        month_starts = sorted({date(d.year, d.month, 1) for d in calendar})
        sched = universe_schedule(liq, month_starts, rule)
        sizes = {d: len(v) for d, v in sched.items()}
        union = {t for names in sched.values() for t in names}
        logger.info(
            f"  eligible per month: min {min(sizes.values())}, "
            f"max {max(sizes.values())}, "
            f"union {len(union):,} names over {len(sizes)} months"
        )
        if not union or max(sizes.values()) == 0:
            logger.error(
                "The point-in-time rule admitted NOTHING on any month. A panel "
                "with no tradeable row is not an empty result, it is a backtest "
                "that silently does nothing and still prints a CAGR. Most likely "
                f"the bars are shorter than the rule needs (min_sessions="
                f"{rule.min_sessions} over {rule.lookback_days}d) or the "
                "turnover bar is set above the whole market."
            )
            raise SystemExit(1)
        # Unknown industries matter among names that can actually be TRADED, not
        # among every column: the union carries years of delisted names that no
        # current constituent list can classify and that the rule never selects.
        from trader.data.nse_industry import UNKNOWN_INDUSTRY_ID as _UNK

        ever = union
        unk_tradeable = sum(1 for t in ever if sector_ids.get(t) == _UNK)
        logger.info(
            f"  of the {len(ever):,} ever-eligible names, {unk_tradeable:,} "
            f"({unk_tradeable / max(len(ever), 1):.1%}) have no NSE industry"
        )
        panel = apply_monthly_mask(panel, sched)

    n_feat = len([c for c in FEATURE_COLS if c in panel.columns])
    n_tradeable = int(panel["is_tradeable"].sum())
    logger.info(
        f"Panel: {panel['date'].n_unique()} days"
        f" × {panel['ticker'].n_unique()} tickers"
        f" × {n_feat} features"
        f" | tradeable: {n_tradeable:,}"
    )
    if n_tradeable == 0:
        logger.error(
            "No row in this panel is tradeable. Every downstream backtest would "
            "hold cash for the whole span and report it as a result. Refusing to "
            "write."
        )
        raise SystemExit(1)

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
    _write_full_panel(panel, panels_root)

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


def _write_full_panel(panel: object, out_dir: Path) -> None:
    """Write the un-split panel as ``full.parquet``.

    The three split parquets do NOT tile the history: the purge months between
    them are dropped entirely, so ``concat(train, val, test)`` has month-long
    holes at each boundary. `scripts/walk_forward.py` rebuilt its "full
    historical panel" exactly that way, and its own windows — which carry their
    own purge structure and land nowhere near the build-time boundaries — were
    silently sliced against a calendar with two months missing. On the panel in
    data/panels that truncated 4 of 12 walk-forward segments.

    Purge gaps belong to the *split*, not to the data, so the un-split panel is
    the correct source for any downstream re-splitting. No DB row: this is not a
    split, it is the material the splits are cut from.
    """
    import polars as pl

    if not isinstance(panel, pl.DataFrame):
        return

    full = panel.sort(["date", "ticker"])
    path = out_dir / "full.parquet"
    full.write_parquet(path)
    sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    (out_dir / "full.sha256").write_text(sha256 + "\n")
    logger.info(
        f"  full.parquet: {len(full):,} rows"
        f" ({full['date'].n_unique()} days, no purge holes)"
        f"  SHA256={sha256[:16]}..."
    )


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
