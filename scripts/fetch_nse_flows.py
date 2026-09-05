#!/usr/bin/env python
"""Fetch NSE delivery / FII-DII flow / bulk-block deal data, and report coverage.

R8's data-layer entrypoint. Reads ``configs/data/ext_features.yaml`` (every key
in that file is read here — see ``--help`` for the overrides), writes one parquet
per source under ``out_root``, and optionally joins them onto a panel to produce
the ext features and their coverage report.

    # what would be fetched, and how long it takes, without fetching
    uv run python scripts/fetch_nse_flows.py --start 2026-09-01 --end 2026-09-05 --dry-run

    # fetch (throttled to ~1 req/s; a re-run reads the on-disk cache and is free)
    uv run python scripts/fetch_nse_flows.py --start 2026-09-01 --end 2026-09-05

    # parse from cache only, never touch the network
    uv run python scripts/fetch_nse_flows.py --start 2026-09-01 --end 2026-09-05 --offline

    # build the ext features on a panel and print per-feature coverage
    uv run python scripts/fetch_nse_flows.py --start 2024-01-01 --end 2024-12-31 \
        --offline --panel data/panels/val.parquet

Nothing here writes into ``data/panels*``: the ext group is opt-in and additive,
and no existing panel, config, or feature changes because this script ran.

Flows note: NSE's live endpoint publishes the latest day only, so ``--refresh``
caches today's figures as a dated JSON snapshot (``fiidii_<date>.json``) and
writes the per-date deal snapshots.  It does NOT append to the hand-maintained
``fiidii_backfill.csv`` ledger.  When a date appears in both, the dated JSON
snapshot is authoritative — ``pivot_flows`` applies that precedence explicitly
and logs the override rather than averaging the two.  History accumulates by
running this daily; there is no verified backfill endpoint (see the module
docstring of ``trader.data.sources.nse_flows``).
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:  # allow `python scripts/...` without install
    sys.path.insert(0, str(REPO_ROOT / "src"))

import polars as pl  # noqa: E402
from loguru import logger  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402

from trader.data.features_ext import (  # noqa: E402
    compute_ext_features,
    ext_feature_coverage,
    format_coverage_report,
    required_purge_months,
)
from trader.data.sources.nse_flows import (  # noqa: E402
    DealsSource,
    DeliverySource,
    FlowsSource,
    NSEClient,
)

DEFAULT_CONFIG = REPO_ROOT / "configs" / "data" / "ext_features.yaml"
ALL_SOURCES = ("delivery", "flows", "deals")


def _parse_date(text: str) -> date:
    return datetime.strptime(text, "%Y-%m-%d").date()


def _load_config(path: Path) -> DictConfig:
    cfg = OmegaConf.load(path)
    if not isinstance(cfg, DictConfig):
        raise TypeError(f"{path} did not parse to a mapping")
    return cfg


def _business_days(start: date, end: date) -> int:
    days = 0
    cursor = start
    while cursor <= end:
        if cursor.weekday() < 5:
            days += 1
        cursor = date.fromordinal(cursor.toordinal() + 1)
    return days


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--start", type=_parse_date, required=True, help="YYYY-MM-DD, inclusive")
    parser.add_argument("--end", type=_parse_date, required=True, help="YYYY-MM-DD, inclusive")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--sources",
        default=None,
        help=f"comma-separated subset of {','.join(ALL_SOURCES)} (default: the config's list)",
    )
    parser.add_argument("--cache-root", type=Path, default=None, help="overrides cache_root")
    parser.add_argument("--out-root", type=Path, default=None, help="overrides out_root")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="parse the on-disk cache only; never open a socket",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="also pull today's FII/DII figures and bulk/block snapshot (latest day only)",
    )
    parser.add_argument("--panel", type=Path, default=None, help="panel parquet to featurise")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the request plan and the time estimate, then stop",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = _load_config(args.config)

    cache_root = Path(args.cache_root or str(cfg.cache_root))
    out_root = Path(args.out_root or str(cfg.out_root))
    sources = (
        [s.strip() for s in args.sources.split(",") if s.strip()]
        if args.sources
        else [str(s) for s in cfg.sources]
    )
    unknown = set(sources) - set(ALL_SOURCES)
    if unknown:
        raise SystemExit(f"unknown source(s) {sorted(unknown)}; known: {list(ALL_SOURCES)}")
    if args.start > args.end:
        raise SystemExit(f"--start {args.start} is after --end {args.end}")

    throttle = cfg.throttle
    min_interval_s = float(throttle.min_interval_s)
    sessions = _business_days(args.start, args.end)
    # One request per session for delivery; flows and deals are latest-day only.
    planned = (sessions if "delivery" in sources else 0) + (2 if args.refresh else 0)
    eta_minutes = planned * min_interval_s / 60.0

    logger.info(
        f"range {args.start} .. {args.end} = {sessions} weekday sessions; "
        f"sources={sources}; cache={cache_root}; offline={args.offline}"
    )
    logger.info(
        f"at most {planned} requests at {min_interval_s:.1f}s spacing "
        f"=> ~{eta_minutes:.1f} min if nothing is cached; cached days cost nothing"
    )
    logger.info(
        "purge reminder: with this group enabled the walk-forward purge must be "
        f">= {required_purge_months()} months (delivery_pct_z_60d looks back 61 sessions)"
    )
    if args.dry_run:
        return 0

    client = NSEClient(
        min_interval_s=min_interval_s,
        max_retries=int(throttle.max_retries),
        timeout_s=float(throttle.timeout_s),
    )
    out_root.mkdir(parents=True, exist_ok=True)
    frames: dict[str, pl.DataFrame] = {}

    if "delivery" in sources:
        delivery_source = DeliverySource(
            cache_root=cache_root,
            client=client,
            offline=args.offline,
            backend=str(cfg.delivery.backend),
        )
        frames["delivery"] = delivery_source.fetch(args.start, args.end)

    if "flows" in sources:
        flows_source = FlowsSource(cache_root=cache_root, client=client, offline=args.offline)
        if args.refresh and not args.offline:
            latest = flows_source.refresh_latest()
            logger.info(f"flows: refreshed {latest.height} category rows from the live API")
        frames["flows"] = flows_source.fetch(args.start, args.end)

    deals_source: DealsSource | None = None
    if "deals" in sources:
        deals_source = DealsSource(cache_root=cache_root, client=client, offline=args.offline)
        if args.refresh and not args.offline:
            written = deals_source.snapshot_latest()
            logger.info(f"deals: snapshotted {[d.isoformat() for d in written]}")
        frames["deals"] = deals_source.fetch(args.start, args.end)

    for name, frame in frames.items():
        path = out_root / f"{name}.parquet"
        frame.write_parquet(path)
        n_dates = frame["date"].n_unique() if frame.height else 0
        logger.info(f"{name}: {frame.height} rows over {n_dates} dates -> {path}")

    if args.panel is not None:
        _report_panel_coverage(args, frames, deals_source, out_root)
    return 0


def _report_panel_coverage(
    args: Any,
    frames: dict[str, pl.DataFrame],
    deals_source: DealsSource | None,
    out_root: Path,
) -> None:
    panel = pl.read_parquet(args.panel)
    observed = deals_source.observed_dates() if deals_source is not None else None
    featured = compute_ext_features(
        panel,
        frames.get("delivery"),
        frames.get("flows"),
        frames.get("deals"),
        observed_deal_dates=observed,
    )
    coverage = ext_feature_coverage(featured)
    report = format_coverage_report(coverage)
    print(report)  # noqa: T201 — this report is the point of the invocation

    stem = Path(args.panel).stem
    featured_path = out_root / f"panel_ext_{stem}.parquet"
    featured.write_parquet(featured_path)
    coverage_path = out_root / f"coverage_{stem}.txt"
    coverage_path.write_text(report + "\n")
    logger.info(f"wrote {featured_path} and {coverage_path}")

    dead = coverage.filter(pl.col("dead"))
    if dead.height:
        logger.error(
            f"DEAD channels (zero variance): {dead['feature'].to_list()} — "
            "do not train on these; beta_nifty_60d was exactly this failure (B7)"
        )


if __name__ == "__main__":
    raise SystemExit(main())
