#!/usr/bin/env python
"""Cut a panel split restricted to names already trading well before the backtest.

This is the P4 sensitivity arm of `11_cost_defect_and_fix_plan.md`.  It writes a
split; it runs **no** backtest and touches no GPU.

What it is for
--------------
The 645 tickers in `data/panels_kite/full.parquet` come from a **2026**
instrument dump.  Two separate things follow from that, and only one of them is
testable:

* *Look-ahead* — trading a name before it listed — is already prevented.
  `align_panel` marks every pre-listing date `is_tradeable=False`
  (`alignment.py:53-74`) and `allocate` intersects candidates with that mask
  (`allocator/deterministic.py:150`,
  `cand = mask & np.isfinite(r_hat) & ...`).  `tests/unit/test_listing.py` asserts it.
* *Selection* — the set was picked with 2026 knowledge of which listings
  mattered — is not prevented.  283 of the 645 names first traded after 2014.

This script bounds the second.  Restricting the universe to names that were
already trading before a cut well ahead of the backtest removes every name
whose post-hoc selection could be doing the work, and it removes them from
**both** arms (equal-weight reads the same `obs["mask"]`, `baselines.py:99`), so
the allocator-minus-equal-weight comparison stays like-for-like on a smaller
opportunity set.

What it does **not** fix is delisting: every one of the 645 names is still
tradeable on the panel's last date, so the data contains zero deaths and the
restricted arm inherits that in full.  See `audit/P4_survivorship.md`.

How the restriction actually binds
----------------------------------
Dropping a ticker's rows from the split is enough.  `scripts/run_allocator.py`
builds its env universe from `active_tickers()` (`run_allocator.py:291`), not
from the panel, and `PanelTradingEnv` leaves a universe ticker with no panel
rows as an all-zero column — so its mask is `False` on every date and both arms
drop it.  The restriction is therefore enforced by exactly the belt the unit
test covers.

Usage
-----
    uv run python scripts/survivorship_arm.py --dry-run          # census only
    uv run python scripts/survivorship_arm.py                    # write the split
    uv run python scripts/survivorship_arm.py --source oos_r4_v2 --out oos_r4_v2_pre2015

Defaults: restrict `oos` to names with >= 252 tradeable days strictly before
2014-12-31, judged on `full.parquet` (the split itself starts in 2016 and holds
no pre-2014 history, so eligibility must be read off the full panel).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import date, datetime
from pathlib import Path

import polars as pl
from loguru import logger

from trader.data.listing import eligible_at, first_tradeable_dates
from trader.data.universe import active_tickers

_ELIGIBILITY_COLS = ["date", "ticker", "is_tradeable"]


def _parse_date(text: str) -> date:
    return datetime.strptime(text, "%Y-%m-%d").date()


def _census(elig_panel: pl.DataFrame, asof: date) -> list[tuple[str, int]]:
    """``[(bucket, n_tickers)]`` by first-tradeable year, split around ``asof``.

    The earliest bucket is *not* a listing cohort: it is everything already
    trading when the panel's calendar opens.  Labelled so it cannot be misread.
    """
    firsts = first_tradeable_dates(elig_panel)
    if not firsts:
        return []
    start_year = min(d.year for d in firsts.values())
    buckets = {
        f"already trading at panel start ({start_year})": 0,
        f"listed {start_year + 1}..{asof.year}": 0,
        f"listed after {asof.year}": 0,
    }
    keys = list(buckets)
    for first in firsts.values():
        if first.year == start_year:
            buckets[keys[0]] += 1
        elif first.year <= asof.year:
            buckets[keys[1]] += 1
        else:
            buckets[keys[2]] += 1
    return list(buckets.items())


def _cohort_diagnostics(src: pl.DataFrame, keep: set[str]) -> list[str]:
    """Lines describing what the restriction removes, in breadth and in return.

    Two numbers decide how a restricted-arm result should be read, and both are
    panel statistics — no backtest, no env, no cost model:

    * **Breadth.** Tradeable names per date, before and after.  ``k`` is a
      count, not a fraction, so shrinking the cross-section makes the allocator
      *more* selective at the same ``k``.  A restricted arm is therefore not a
      clean one-factor change and the write-up must say so.

      Reported **twice**, and the second line is the one to quote.  The panel
      carries 645 tickers but the env's universe is ``active_tickers()`` = 504
      (`run_allocator.py`), so 141 panel names are never traded by either arm
      and a panel-wide breadth figure understates ``k/N`` by ~2.5 percentage
      points.  Both arms — allocator and ``EqualWeightRebalanced`` — see the
      env-universe number, not the panel-wide one.
    * **Cohort return.** The equal-weight cross-sectional daily log return of
      the kept and dropped cohorts, annualised.  If the dropped names earned
      more, the restriction lowers both arms — the allocator *and* its
      equal-weight benchmark — so a fall in the allocator's absolute CAGR is
      expected and is not by itself evidence that the edge was selection.
    """
    out: list[str] = []
    trd = src.filter(pl.col("is_tradeable")).with_columns(
        pl.col("ticker").is_in(list(keep)).alias("_kept")
    )
    per_date = trd.group_by("date").agg(
        pl.len().alias("n_all"), pl.col("_kept").sum().alias("n_kept")
    )
    out.append(
        f"breadth (panel-wide, 645 tickers): tradeable names/date "
        f"{per_date['n_all'].mean():.1f} "
        f"(min {per_date['n_all'].min()}, max {per_date['n_all'].max()}) -> "
        f"{per_date['n_kept'].mean():.1f} "
        f"(min {per_date['n_kept'].min()}, max {per_date['n_kept'].max()})"
    )

    # The width both arms actually trade.  `run_allocator.py` builds the env
    # universe from `active_tickers()`, so anything outside it is an all-zero
    # column that neither the allocator nor equal-weight can hold.
    env_universe = set(active_tickers())
    env_trd = trd.filter(pl.col("ticker").is_in(list(env_universe)))
    env_per_date = env_trd.group_by("date").agg(
        pl.len().alias("n_all"), pl.col("_kept").sum().alias("n_kept")
    )
    if env_per_date.height:
        mean_all = float(env_per_date["n_all"].mean() or 0.0)
        mean_kept = float(env_per_date["n_kept"].mean() or 0.0)
        out.append(
            f"breadth (env universe, active_tickers()={len(env_universe)}): "
            f"tradeable names/date {mean_all:.1f} "
            f"(min {env_per_date['n_all'].min()}, max {env_per_date['n_all'].max()}) -> "
            f"{mean_kept:.1f} "
            f"(min {env_per_date['n_kept'].min()}, "
            f"max {env_per_date['n_kept'].max()}); "
            f"k=30 is {100.0 * 30.0 / max(mean_all, 1.0):.2f}% -> "
            f"{100.0 * 30.0 / max(mean_kept, 1.0):.2f}% of the cross-section; "
            f"names removed {mean_all - mean_kept:.1f}/date"
        )
    # Sector composition: the restriction is not uniform across sectors, so
    # `max_sector_weight` binds on a different mix and not merely a smaller one.
    if "sector_id" in src.columns:
        sec = (
            env_trd.group_by("sector_id")
            .agg(
                pl.col("ticker").n_unique().alias("n_all"),
                pl.col("ticker").filter(pl.col("_kept")).n_unique().alias("n_kept"),
            )
            .sort("sector_id")
        )
        parts = [
            f"{int(r['sector_id'])}: {int(r['n_all'])}->{int(r['n_kept'])}"
            f" ({100.0 * int(r['n_kept']) / max(int(r['n_all']), 1):.0f}%)"
            for r in sec.iter_rows(named=True)
        ]
        out.append("sector retention (env universe): " + ", ".join(parts))
    # CLAUDE.md's phantom-sector trap: an active ticker with no rows in the
    # split becomes an all-zero column, i.e. sector_id == 0, an id absent from
    # SECTOR_IDS.  Harmless for the allocator (`sector_ids > 0`) and for
    # equal-weight (mask only), fatal for a graph model.
    dropped_active = len(env_universe - set(src["ticker"].unique().to_list()) - keep)
    out.append(
        f"phantom sectors: {len(env_universe - keep)} of {len(env_universe)} "
        f"active tickers have no rows in the restricted split and become "
        f"all-zero columns with sector_id == 0 ({dropped_active} of them are "
        f"already absent from the source split). Never hand this split to a "
        f"graph model (CLAUDE.md, phantom sectors)."
    )
    if "log_return_1d" not in src.columns:
        out.append("cohort return: skipped, no log_return_1d column")
        return out
    daily = (
        trd.group_by(["date", "_kept"])
        .agg(pl.col("log_return_1d").mean().alias("r"))
        .sort("date")
    )
    for kept_flag, label in ((True, "kept   "), (False, "dropped")):
        col = daily.filter(pl.col("_kept") == kept_flag)["r"]
        if col.len() == 0:
            continue
        ann = float(col.mean() or 0.0) * 252.0
        vol = float(col.std() or 0.0) * 252.0**0.5
        out.append(
            f"cohort return ({label}): equal-weight log return {ann:+.4f}/yr "
            f"(~{100.0 * math.expm1(ann):+.1f}% CAGR), "
            f"ann vol {vol:.4f}, over {col.len()} dates"
        )
    return out


def _tail_exposure(
    src: pl.DataFrame, pred_path: Path, k: int, horizon: int
) -> list[str]:
    """How often the top-``k`` by ``r_hat`` land in the left tail of realised return.

    This is the quantity that sets the SIGN of the survivorship residual in the
    allocator-vs-equal-weight comparison, and it is not the one the obvious
    argument names.  "The allocator holds 30 names and equal-weight holds 365,
    so the allocator is more exposed to the tail" is a statement about
    *variance*; concentration alone produces no bias in a difference of means.
    What decides the sign is the rate at which the names survivorship truncates
    — the left tail of realised return — would have been PICKED, against the
    rate at which they sit in the benchmark.  Equal-weight's rate is its
    population share by construction; the allocator's is measured here.

    Deliberately a **proxy and not a bound**: the tail this measures is realised
    forward return inside the panel, and the names survivorship actually removed
    are not in the panel at all (`--dry-run` prints "0 deaths").  A rate above
    the population share says the selection leans into the left tail on the data
    that exists, so the residual runs upward for the allocator; it says nothing
    about how large the residual is.  The magnitude stays UNVERIFIED, and no
    amount of arithmetic on this panel can change that.
    """
    if "log_return_1d" not in src.columns:
        return ["tail exposure: skipped, no log_return_1d column"]
    fwd = (
        src.sort(["ticker", "date"])
        .with_columns(
            pl.col("log_return_1d")
            .rolling_sum(horizon)
            .shift(-horizon)
            .over("ticker")
            .alias("_fwd")
        )
        .filter(pl.col("is_tradeable") & pl.col("_fwd").is_not_null())
        .select("date", "ticker", "_fwd")
    )
    preds = pl.read_parquet(pred_path, columns=["date", "ticker", f"r_hat_{horizon}d"])
    joined = fwd.join(preds, on=["date", "ticker"], how="inner").drop_nulls()
    if joined.height == 0:
        return ["tail exposure: skipped, signal and panel do not overlap"]
    ranked = joined.with_columns(
        pl.col(f"r_hat_{horizon}d").rank("ordinal", descending=True).over("date").alias("_r"),
        pl.col("_fwd").rank("ordinal").over("date").alias("_fr"),
        pl.len().over("date").alias("_n"),
    ).filter(pl.col("_n") >= k * 2)
    picked = ranked.filter(pl.col("_r") <= k)
    n_dates = picked["date"].n_unique()
    out = [
        f"tail exposure ({n_dates} dates, top-{k} by r_hat_{horizon}d vs realised "
        f"forward {horizon}d return):"
    ]
    for q, label in ((0.10, "bottom decile"), (0.05, "bottom 5%")):
        share = float(
            (picked["_fr"].cast(pl.Float64) <= q * picked["_n"].cast(pl.Float64)).mean()
            or 0.0
        )
        out.append(
            f"  in the {label}: {100.0 * share:.2f}% of picks "
            f"(a rate-blind pick would be {100.0 * q:.0f}%) -> "
            f"{share / q:.2f}x the population share"
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--panels-root", default="data/panels_kite")
    ap.add_argument("--source", default="oos", help="split to restrict")
    ap.add_argument(
        "--eligibility-source",
        default="full",
        help="split that supplies pre-asof history (must reach back before --asof)",
    )
    ap.add_argument("--asof", default="2014-12-31", type=_parse_date)
    ap.add_argument(
        "--min-history-days",
        default=252,
        type=int,
        help="tradeable days required strictly before --asof (>=1)",
    )
    ap.add_argument("--out", default=None, help="split name to write (default: <source>_pre<year>)")
    ap.add_argument("--force", action="store_true", help="overwrite an existing split")
    ap.add_argument(
        "--tail-exposure",
        default=None,
        help=(
            "signal dir (e.g. data/signal/r4_v1) — also report how often the "
            "top-k by r_hat land in the left tail of realised forward return, "
            "which is what sets the SIGN of the survivorship residual"
        ),
    )
    ap.add_argument("--tail-k", type=int, default=30)
    ap.add_argument("--tail-horizon", type=int, default=20)
    ap.add_argument(
        "--allow-pre-asof",
        action="store_true",
        help="permit a source split that begins on or before --asof",
    )
    ap.add_argument("--dry-run", action="store_true", help="print the census, write nothing")
    args = ap.parse_args()

    if args.min_history_days < 1:
        raise SystemExit(
            f"--min-history-days must be >= 1, got {args.min_history_days}; 0 would "
            f"admit a name with no tradeable history before the cut at all."
        )

    root = Path(args.panels_root)
    asof: date = args.asof
    out_name = args.out or f"{args.source}_pre{asof.year + 1}"
    out_path = root / f"{out_name}.parquet"

    src_path = root / f"{args.source}.parquet"
    elig_path = root / f"{args.eligibility_source}.parquet"
    for path in (src_path, elig_path):
        if not path.exists():
            raise SystemExit(f"{path} not found — build panels first.")

    # Refuse to clobber before doing the expensive read, not after.
    if out_path.exists() and not args.force and not args.dry_run:
        raise SystemExit(
            f"{out_path} already exists. Re-run with --force to overwrite, or pass "
            f"a different --out. Refusing silently to replace a split another "
            f"result may already cite."
        )

    # ── eligibility, judged on the panel that actually holds the history ──────
    elig_panel = pl.read_parquet(elig_path, columns=_ELIGIBILITY_COLS)
    elig_lo = elig_panel["date"].min()
    if not isinstance(elig_lo, date) or elig_lo >= asof:
        raise SystemExit(
            f"{elig_path} starts {elig_lo}, on or after --asof {asof}: it holds no "
            f"pre-cut history, so every name would be judged ineligible. Point "
            f"--eligibility-source at the full panel."
        )
    eligible = eligible_at(elig_panel, asof, args.min_history_days)
    n_panel_tickers = elig_panel["ticker"].n_unique()

    logger.info(
        f"Eligibility from {elig_path} ({elig_lo}..{elig_panel['date'].max()}): "
        f"{len(eligible)} of {n_panel_tickers} tickers have >= "
        f"{args.min_history_days} tradeable days strictly before {asof}."
    )
    for label, count in _census(elig_panel, asof):
        logger.info(f"  {label}: {count}")

    # The delisting half of the bias, stated every time this script runs so it
    # cannot be quietly dropped from a write-up: how many names are still
    # tradeable on the panel's very last date.  A real 2005-2026 Indian equity
    # universe delists names; a 2026 instrument dump cannot contain any.
    last_day = elig_panel["date"].max()
    survivors = (
        elig_panel.lazy()
        .filter(pl.col("is_tradeable"))
        .group_by("ticker")
        .agg(pl.col("date").max().alias("last"))
        .filter(pl.col("last") == pl.lit(last_day, dtype=pl.Date))
        .collect()
        .height
    )
    logger.info(
        f"  UNRECOVERABLE: {survivors} of {n_panel_tickers} tickers are still "
        f"tradeable on {last_day} ({n_panel_tickers - survivors} deaths in "
        f"{elig_lo.year}..{last_day.year if isinstance(last_day, date) else '?'}). "
        f"Restricting the listing side does not fix this."
    )

    # The env trades `active_tickers()`, a subset of the panel's tickers, so
    # report the count that actually reaches the allocator as well.
    try:
        from trader.data.universe import active_tickers

        active = set(active_tickers())
        logger.info(
            f"  of active_tickers() ({len(active)}): {len(active & set(eligible))} eligible"
        )
    except Exception as exc:                                  # pragma: no cover
        logger.warning(f"could not intersect with active_tickers(): {exc}")

    # ── the source split ─────────────────────────────────────────────────────
    src = pl.read_parquet(src_path)
    src_lo, src_hi = src["date"].min(), src["date"].max()
    if isinstance(src_lo, date) and src_lo <= asof and not args.allow_pre_asof:
        raise SystemExit(
            f"{src_path} begins {src_lo}, on or before --asof {asof}. The "
            f"restriction is only meaningful when the whole backtest sits after "
            f"the cut. Move --asof earlier, or pass --allow-pre-asof if you "
            f"intend the partial version and will say so in the write-up."
        )

    keep = sorted(set(eligible) & set(src["ticker"].unique().to_list()))
    restricted = src.filter(pl.col("ticker").is_in(keep)).sort(["date", "ticker"])
    if restricted.height == 0:
        raise SystemExit("Restriction left no rows — check --asof and --min-history-days.")

    dropped = src["ticker"].n_unique() - len(keep)
    logger.info(
        f"{args.source} ({src_lo}..{src_hi}): {src['ticker'].n_unique()} tickers -> "
        f"{len(keep)} kept, {dropped} dropped."
    )
    logger.info(
        f"  rows {src.height:,} -> {restricted.height:,}; "
        f"tradeable {src.filter(pl.col('is_tradeable')).height:,} -> "
        f"{restricted.filter(pl.col('is_tradeable')).height:,}"
    )
    for line in _cohort_diagnostics(src, set(keep)):
        logger.info(f"  {line}")

    if args.tail_exposure:
        pred_path = Path(args.tail_exposure) / "predictions.parquet"
        if not pred_path.exists():
            raise SystemExit(f"{pred_path} not found — --tail-exposure needs R4 output.")
        for line in _tail_exposure(src, pred_path, args.tail_k, args.tail_horizon):
            logger.info(f"  {line}")

    if args.dry_run:
        logger.info("--dry-run: nothing written.")
        return

    restricted.write_parquet(out_path)
    digest = hashlib.sha256(out_path.read_bytes()).hexdigest()
    (root / f"{out_name}.sha256").write_text(digest + "\n")
    manifest = {
        "out": out_name,
        "source": args.source,
        "eligibility_source": args.eligibility_source,
        "asof": asof.isoformat(),
        "min_history_days": args.min_history_days,
        "tickers_kept": len(keep),
        "tickers_dropped": dropped,
        "rows": restricted.height,
        "dates": restricted["date"].n_unique(),
        "span": [str(restricted["date"].min()), str(restricted["date"].max())],
        "sha256": digest,
        "tickers": keep,
    }
    (root / f"{out_name}.manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    logger.info(
        f"{out_path}: {restricted.height:,} rows, {len(keep)} tickers, "
        f"{restricted['date'].n_unique()} dates"
    )
    logger.info(f"SHA256={digest[:16]}...  manifest={root / f'{out_name}.manifest.json'}")
    logger.info(
        "Written only. Backtesting it is the orchestrator's serialised grid run: "
        f"`uv run python scripts/run_allocator.py +split={out_name} ...`"
    )


if __name__ == "__main__":
    main()
