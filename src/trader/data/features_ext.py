"""R8 — the opt-in NSE feature group: delivery, market flows, disclosed deals.

All 15 columns in :data:`trader.data.features.FEATURE_COLS` derive from
``adj_close`` and ``volume``. `10_architecture_revamp.md` §4 ranks richer
features as the lever most likely to move the cross-sectional signal, and §6
gates that lever behind R4's rank-IC gate: the pipeline must first prove it can
*measure* a signal before new inputs join the experiment.

So this module is **additive and opt-in**. It never touches ``FEATURE_COLS``,
the default panel, or any existing feature's values: :func:`compute_ext_features`
returns the input panel with :data:`EXT_FEATURE_COLS` appended and every
pre-existing column byte-identical. R4 is runnable with and without it.

--------------------------------------------------------------------------------
Timing discipline
--------------------------------------------------------------------------------

Every feature here is **strictly backward-looking through t-1**, one step
stricter than :mod:`trader.data.features` (whose rolling windows close on day
``t``) and matching :mod:`trader.data.regime_features`.

That is not conservatism for its own sake. All three sources are *published
after the close of the day they describe*: the MTO file for trade date ``d``
appears in NSE's archive that evening, ``fiidiiTradeReact`` reports ``d`` after
the session, and bulk/block disclosures are end-of-day filings. A feature that
read day ``t``'s delivery on day ``t`` would be using information that did not
exist when the allocator had to act. Concretely, every column starts from a
``.shift(1).over("ticker")`` (or ``.shift(1)`` on the market-level series over
the panel's own trading dates).

--------------------------------------------------------------------------------
The six features, and why each one
--------------------------------------------------------------------------------

``delivery_pct_20d``
    Mean deliverable-to-traded ratio over the 20 sessions ending t-1. The level
    of delivery separates accumulation from intraday churn: a stock whose volume
    is 70% delivered is being *bought*, one at 25% is being traded around. It is
    a slow, level-like variable, so a 20-day mean is the natural summary.

``delivery_pct_z_60d``
    ``delivery_pct`` at t-1, z-scored against its own trailing 60 sessions. The
    level differs structurally across stocks (index heavyweights sit lower
    because of derivative-linked churn), so the *level* is largely a fixed
    per-stock effect that a cross-sectional model cannot use. The change
    relative to the stock's own history is the part that carries news. Null —
    never zero — when the trailing standard deviation is degenerate.

``fii_net_5d`` / ``dii_net_5d``
    Sum of net foreign / domestic institutional cash-market flow, ₹ crore, over
    the 5 sessions ending t-1. Market-level, broadcast to every ticker exactly
    as the regime vector is. Kept as two separate columns rather than a
    difference because they are not opposites: both can be net buyers on a day
    of heavy primary issuance, and that state differs from the tug-of-war state.

``bulk_deal_flag_5d``
    Fraction of the 5 sessions ending t-1 on which this ticker had at least one
    disclosed bulk or block deal. Preferred over a 0/1 "any deal in 5 days"
    indicator: the fraction is strictly more informative, degrades gracefully,
    and costs nothing extra. Sparse by nature — most cells are a real 0.0, which
    is *not* the same as the null emitted for a day whose deal file was never
    captured.

``bulk_deal_net_20d``
    Net disclosed deal value (buy minus sell) over the 20 sessions ending t-1,
    divided by the ticker's own traded value over those sessions. The raw rupee
    figure is not comparable across the cross-section — a ₹100 cr block is
    routine in RELIANCE and enormous in a mid-cap — so it is normalised by
    ``sum(adj_close x volume)`` over the same trailing window, computed *here*
    from shifted panel columns rather than borrowing ``dollar_volume_20`` (which
    closes on day t and would leak same-day information into a t-1 feature).
    Null when the panel carries no ``adj_close``/``volume``, or when trailing
    traded value is zero.

--------------------------------------------------------------------------------
Nulls, coverage, and the purge
--------------------------------------------------------------------------------

**Nothing is ever zero-filled.** B7 (`09_revamp_and_audit.md`) is the reason:
``beta_nifty_60d`` was a fabricated constant 1.0 across 276,005 training rows
and the dead channel acted as a fixed bias for months. A missing delivery day, a
day whose deal file was never captured, a flow series with a hole — all produce
nulls. :func:`ext_feature_coverage` reports the fraction of real values per
feature, together with mean/std so a dead channel is visible immediately.

The one place a *zero* is legitimate is the deals group, and only on an observed
date: NSE published the complete list of that day's disclosed deals, and this
ticker was not on it. That is a measured absence. A date with no snapshot on
disk is unknown and stays null — see ``observed_deal_dates``.

:data:`EXT_FEATURE_LOOKBACK_DAYS` mirrors ``features.FEATURE_LOOKBACK_DAYS`` in
shape so the walk-forward purge guard can be derived from both. Note
``delivery_pct_z_60d`` is **61** trading days — one longer than anything in the
base group — so the guard must widen when this group is enabled. See
:func:`combined_max_lookback_days` and :func:`required_purge_months`;
``trader.training.walk_forward.assert_purge_clears_feature_lookback`` currently
reads ``features.MAX_FEATURE_LOOKBACK_DAYS`` alone and is owned by another
stream, so wiring it is reported, not applied.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import date
from typing import Final

import polars as pl

from trader.data.features import FEATURE_LOOKBACK_DAYS, MAX_FEATURE_LOOKBACK_DAYS

# ── Public feature list ──────────────────────────────────────────────────────

EXT_FEATURE_COLS: list[str] = [
    "delivery_pct_20d",
    "delivery_pct_z_60d",
    "fii_net_5d",
    "dii_net_5d",
    "bulk_deal_flag_5d",
    "bulk_deal_net_20d",
]

# Longest backward-looking window each ext feature depends on, in TRADING days.
# Same shape and same purpose as `features.FEATURE_LOOKBACK_DAYS`: the single
# source of truth for "how far back does this feature see", from which the
# walk-forward purge is derived. Each entry is `1 + window` because every
# feature starts from a shift(1) (see "Timing discipline" above).
EXT_FEATURE_LOOKBACK_DAYS: dict[str, int] = {
    "delivery_pct_20d": 21,      # shift(1) + 20-session mean
    "delivery_pct_z_60d": 61,    # shift(1) + 60-session mean/std  <- widens the purge
    "fii_net_5d": 6,             # shift(1) + 5-session sum
    "dii_net_5d": 6,
    "bulk_deal_flag_5d": 6,
    "bulk_deal_net_20d": 21,     # shift(1) + 20-session sum, and its 20-session scaler
}

MAX_EXT_FEATURE_LOOKBACK_DAYS: int = max(EXT_FEATURE_LOOKBACK_DAYS.values())

# ── Window parameters ────────────────────────────────────────────────────────
# Named rather than inlined so `test_features_ext.py` can scan this module's
# source for window literals exceeding the declared lookback, exactly as
# `test_alignment_features.py` does for features.py.
#
# WARM-UP IS SET BY `min_samples`, NOT BY THE DECLARED LOOKBACK.  The two are
# different numbers and it matters when reading a value:
#
#   feature               declared lookback   actual warm-up (min_samples)
#   delivery_pct_20d              21                    15
#   delivery_pct_z_60d            61                    40
#
# So `delivery_pct_z_60d` is NOT necessarily a 60-observation z-score — early
# in a ticker's history it can be a **40-observation** one, and it carries no
# marker saying which.  EXT_FEATURE_LOOKBACK_DAYS is a conservative UPPER bound
# used to size the walk-forward purge; it is safe in that direction (the purge
# is wider than it needs to be) and it is not a statement about how many
# observations went into any particular value.

_DELIVERY_MEAN_WIN: Final[int] = 20
_DELIVERY_MEAN_MIN: Final[int] = 15   # 75% of the window must be real
_DELIVERY_Z_WIN: Final[int] = 60
_DELIVERY_Z_MIN: Final[int] = 40
_FLOW_WIN: Final[int] = 5             # market-level series: require the full window
_DEAL_FLAG_WIN: Final[int] = 5
_DEAL_NET_WIN: Final[int] = 20
_TURNOVER_WIN: Final[int] = 20
_TURNOVER_MIN: Final[int] = 10
_STD_FLOOR: Final[float] = 1e-12

_TRADING_DAYS_PER_MONTH: Final[int] = 21  # mirrors walk_forward._TRADING_DAYS_PER_MONTH


# ── Purge helpers ────────────────────────────────────────────────────────────


def combined_max_lookback_days(*, include_ext: bool = True) -> int:
    """Longest feature lookback across the base group and, optionally, this one.

    ``trader.training.walk_forward.assert_purge_clears_feature_lookback`` derives
    the minimum purge from ``features.MAX_FEATURE_LOOKBACK_DAYS``. When the ext
    group is enabled the true bound is this function's value instead.
    """
    if not include_ext:
        return MAX_FEATURE_LOOKBACK_DAYS
    return max(MAX_FEATURE_LOOKBACK_DAYS, MAX_EXT_FEATURE_LOOKBACK_DAYS)


def required_purge_months(*, include_ext: bool = True) -> int:
    """Smallest ``walk.purge_months`` that clears :func:`combined_max_lookback_days`."""
    days = combined_max_lookback_days(include_ext=include_ext)
    return -(-days // _TRADING_DAYS_PER_MONTH)


def assert_disjoint_from_base() -> None:
    """Raise if an ext feature name collides with a base ``FEATURE_COLS`` name.

    A collision would make the panel's column silently ambiguous, and the join
    in :func:`compute_ext_features` would rename rather than fail.
    """
    clash = set(EXT_FEATURE_COLS) & set(FEATURE_LOOKBACK_DAYS)
    if clash:
        raise ValueError(
            f"EXT_FEATURE_COLS collides with trader.data.features.FEATURE_COLS on "
            f"{sorted(clash)}; ext features must be strictly additive."
        )


# ── Input validation ─────────────────────────────────────────────────────────


def _require_columns(frame: pl.DataFrame, columns: Sequence[str], what: str) -> None:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValueError(f"{what} is missing column(s) {missing}; has {frame.columns}")


def _require_unique(frame: pl.DataFrame, keys: Sequence[str], what: str) -> None:
    if frame.height and frame.select(list(keys)).is_duplicated().any():
        dupes = (
            frame.filter(frame.select(list(keys)).is_duplicated())
            .select(list(keys))
            .unique()
            .head(5)
        )
        raise ValueError(
            f"{what} has duplicate {list(keys)} rows, which would multiply panel rows "
            f"on join. First few:\n{dupes}"
        )


# ── The feature computation ──────────────────────────────────────────────────


def compute_ext_features(
    panel: pl.DataFrame,
    delivery: pl.DataFrame | None = None,
    flows: pl.DataFrame | None = None,
    deals: pl.DataFrame | None = None,
    *,
    observed_deal_dates: Sequence[date] | None = None,
) -> pl.DataFrame:
    """Append :data:`EXT_FEATURE_COLS` to ``panel``, leaving every other column alone.

    Args:
        panel: the aligned panel; needs ``date`` and ``ticker``. ``adj_close``
            and ``volume`` are used, if present, to scale ``bulk_deal_net_20d``.
        delivery: :data:`trader.data.sources.nse_flows.DELIVERY_SCHEMA` rows —
            one per (date, ticker). ``None`` leaves the delivery features null.
        flows: :data:`trader.data.sources.nse_flows.FLOWS_SCHEMA` rows — one per
            date, market-level. ``None`` leaves the flow features null.
        deals: :data:`trader.data.sources.nse_flows.DEALS_SCHEMA` rows — one per
            (date, ticker) that had a disclosed deal. ``None`` leaves the deal
            features null.
        observed_deal_dates: the dates for which a deals file was actually
            captured. On such a date a ticker with no deal row is a real 0.0; on
            any other date the deal features are null. Defaults to the distinct
            dates present in ``deals``, which is right except for the (never yet
            observed) case of a session with no disclosed deal in the entire
            market — that day would be treated as uncaptured, i.e. null, which is
            the conservative direction.

    Returns:
        A frame with the same rows, in the same order, as ``panel``, plus one
        Float64 column per entry of :data:`EXT_FEATURE_COLS`. Missing inputs
        produce nulls; nothing is zero-filled.
    """
    assert_disjoint_from_base()
    _require_columns(panel, ["date", "ticker"], "panel")
    clash = set(EXT_FEATURE_COLS) & set(panel.columns)
    if clash:
        raise ValueError(f"panel already carries ext feature column(s) {sorted(clash)}")

    work = panel.with_row_index("_ext_row")
    work = _add_delivery_features(work, delivery)
    work = _add_flow_features(work, flows)
    work = _add_deal_features(work, deals, observed_deal_dates)

    ordered = work.sort("_ext_row").drop("_ext_row")
    return ordered.select(
        [*panel.columns, *[pl.col(c).cast(pl.Float64) for c in EXT_FEATURE_COLS]]
    )


def _null_columns(frame: pl.DataFrame, names: Sequence[str]) -> pl.DataFrame:
    return frame.with_columns(
        [pl.lit(None, dtype=pl.Float64).alias(name) for name in names]
    )


def _add_delivery_features(work: pl.DataFrame, delivery: pl.DataFrame | None) -> pl.DataFrame:
    names = ["delivery_pct_20d", "delivery_pct_z_60d"]
    if delivery is None or delivery.height == 0:
        return _null_columns(work, names)

    _require_columns(delivery, ["date", "ticker", "delivery_pct"], "delivery")
    _require_unique(delivery, ["date", "ticker"], "delivery")

    joined = work.join(
        delivery.select("date", "ticker", pl.col("delivery_pct").cast(pl.Float64)),
        on=["date", "ticker"],
        how="left",
    ).sort(["ticker", "date"])

    lag = pl.col("delivery_pct").shift(1).over("ticker")
    mean_60 = lag.rolling_mean(
        window_size=_DELIVERY_Z_WIN, min_samples=_DELIVERY_Z_MIN
    ).over("ticker")
    std_60 = lag.rolling_std(
        window_size=_DELIVERY_Z_WIN, min_samples=_DELIVERY_Z_MIN
    ).over("ticker")

    return joined.with_columns(
        lag.rolling_mean(window_size=_DELIVERY_MEAN_WIN, min_samples=_DELIVERY_MEAN_MIN)
        .over("ticker")
        .alias("delivery_pct_20d"),
        # A degenerate trailing std means the z-score is undefined, not zero.
        pl.when(std_60 > _STD_FLOOR)
        .then((lag - mean_60) / std_60)
        .otherwise(None)
        .alias("delivery_pct_z_60d"),
    ).drop("delivery_pct")


def _add_flow_features(work: pl.DataFrame, flows: pl.DataFrame | None) -> pl.DataFrame:
    names = ["fii_net_5d", "dii_net_5d"]
    if flows is None or flows.height == 0:
        return _null_columns(work, names)

    _require_columns(flows, ["date", "fii_net_cr", "dii_net_cr"], "flows")
    _require_unique(flows, ["date"], "flows")

    # The market-level series is rolled over the PANEL's trading dates, so that
    # shift(1) means "the previous session in this panel" and a date the panel
    # does not contain cannot silently enter a window.
    calendar = (
        work.select("date")
        .unique()
        .sort("date")
        .join(
            flows.select(
                "date",
                pl.col("fii_net_cr").cast(pl.Float64),
                pl.col("dii_net_cr").cast(pl.Float64),
            ),
            on="date",
            how="left",
        )
    )
    calendar = calendar.with_columns(
        pl.col("fii_net_cr")
        .shift(1)
        .rolling_sum(window_size=_FLOW_WIN, min_samples=_FLOW_WIN)
        .alias("fii_net_5d"),
        pl.col("dii_net_cr")
        .shift(1)
        .rolling_sum(window_size=_FLOW_WIN, min_samples=_FLOW_WIN)
        .alias("dii_net_5d"),
    ).select("date", *names)

    return work.join(calendar, on="date", how="left")


def _add_deal_features(
    work: pl.DataFrame,
    deals: pl.DataFrame | None,
    observed_deal_dates: Sequence[date] | None,
) -> pl.DataFrame:
    names = ["bulk_deal_flag_5d", "bulk_deal_net_20d"]
    if deals is None or deals.height == 0:
        if observed_deal_dates is None:
            return _null_columns(work, names)
        deals = pl.DataFrame(
            schema={"date": pl.Date(), "ticker": pl.Utf8(), "net_value_inr": pl.Float64()}
        )

    _require_columns(deals, ["date", "ticker", "net_value_inr"], "deals")
    _require_unique(deals, ["date", "ticker"], "deals")

    observed = (
        sorted(set(observed_deal_dates))
        if observed_deal_dates is not None
        else sorted(set(deals["date"].to_list()))
    )

    joined = work.join(
        deals.select("date", "ticker", pl.col("net_value_inr").cast(pl.Float64)),
        on=["date", "ticker"],
        how="left",
    ).sort(["ticker", "date"])

    # On a captured date, "no deal row" is a measured zero. On an uncaptured
    # date it is unknown — null, never zero (B7).
    is_observed = pl.col("date").is_in(observed)
    joined = joined.with_columns(
        pl.when(is_observed)
        .then(pl.col("net_value_inr").is_not_null().cast(pl.Float64))
        .otherwise(None)
        .alias("_deal_flag"),
        pl.when(is_observed)
        .then(pl.col("net_value_inr").fill_null(0.0))
        .otherwise(None)
        .alias("_deal_net"),
    )

    flag_5d = (
        pl.col("_deal_flag")
        .shift(1)
        .rolling_mean(window_size=_DEAL_FLAG_WIN, min_samples=_DEAL_FLAG_WIN)
        .over("ticker")
    )
    net_20d = (
        pl.col("_deal_net")
        .shift(1)
        .rolling_sum(window_size=_DEAL_NET_WIN, min_samples=_DEAL_NET_WIN)
        .over("ticker")
    )

    if {"adj_close", "volume"} <= set(joined.columns):
        turnover = pl.col("adj_close").cast(pl.Float64) * pl.col("volume").cast(pl.Float64)
        traded_value_20d = (
            turnover.shift(1)
            .rolling_sum(window_size=_TURNOVER_WIN, min_samples=_TURNOVER_MIN)
            .over("ticker")
        )
        scaled = (
            pl.when(traded_value_20d > 0.0).then(net_20d / traded_value_20d).otherwise(None)
        )
    else:
        # No price/volume to normalise against: an unnormalised rupee figure is
        # not comparable across the cross-section, so emit null rather than a
        # number that means something different for every ticker.
        scaled = pl.lit(None, dtype=pl.Float64)

    return joined.with_columns(
        flag_5d.alias("bulk_deal_flag_5d"),
        scaled.alias("bulk_deal_net_20d"),
    ).drop("net_value_inr", "_deal_flag", "_deal_net")


# ── Coverage / liveness report ───────────────────────────────────────────────

COVERAGE_SCHEMA: Final[tuple[str, ...]] = (
    "feature",
    "n_cells",
    "n_present",
    "n_missing",
    "coverage",
    "mean",
    "std",
    "dead",
    "status",
)


def _as_float(value: object) -> float | None:
    """Narrow a polars aggregate (which is typed as a union of scalars) to float."""
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def ext_feature_coverage(
    frame: pl.DataFrame,
    *,
    features: Sequence[str] | None = None,
    tradeable_only: bool = True,
) -> pl.DataFrame:
    """Per-feature coverage and liveness over a panel's (date, ticker) cells.

    Columns: ``feature``, ``n_cells`` (candidate cells), ``n_present`` (cells
    with a real value), ``coverage`` (= ``n_present / n_cells``, null when there
    are no cells), ``mean``, ``std``, and ``dead`` (``std`` below 1e-12 — the
    ``beta_nifty_60d`` failure mode, where a constant channel acted as a fixed
    bias into every convolution).

    ``tradeable_only`` restricts the denominator to rows where ``is_tradeable``
    is true, which is the population the model actually sees.
    """
    names = list(features) if features is not None else list(EXT_FEATURE_COLS)
    rows = frame
    if tradeable_only and "is_tradeable" in frame.columns:
        rows = frame.filter(pl.col("is_tradeable"))
    n_cells = rows.height

    records: list[dict[str, object]] = []
    for name in names:
        if name not in rows.columns:
            raise ValueError(f"{name!r} is not a column of the frame; has {rows.columns}")
        column = rows[name].cast(pl.Float64)
        # NaN-AWARE.  `is_not_null()` alone counts a NaN as present, and the
        # std of an all-NaN column is NaN, which is neither None nor < the
        # floor — so an entirely NaN channel reported coverage 1.0000 / "ok".
        # That is precisely the beta_nifty_60d failure this function exists to
        # catch (constant 1.0 across all 276,005 tradeable rows, stored stats
        # recording mean 0.0, a dead channel acting as a fixed bias into every
        # convolution). A poisoned channel must not be able to look healthy here.
        present = column.is_not_null() & column.is_not_nan()
        n_present = int(present.sum())
        finite = column.filter(present)
        n_nan = int((column.is_null() | column.is_nan()).sum())
        mean = _as_float(finite.mean()) if n_present else None
        std = _as_float(finite.std()) if n_present > 1 else None
        std_is_nan = std is not None and math.isnan(std)
        records.append(
            {
                "feature": name,
                "n_cells": n_cells,
                "n_present": n_present,
                "n_missing": n_nan,
                "coverage": (n_present / n_cells) if n_cells else None,
                "mean": mean,
                "std": None if std_is_nan else std,
                # `dead` is true for no-variance AND for a channel with no
                # usable variance at all (all-NaN, or an unusable std).
                "dead": bool(
                    (std is not None and not std_is_nan and std < _STD_FLOOR)
                    or std_is_nan
                    or (n_cells > 0 and n_present == 0)
                ),
                "status": (
                    "EMPTY (all null/NaN)"
                    if n_cells > 0 and n_present == 0
                    else "DEAD (unusable std)"
                    if std_is_nan
                    else "DEAD (no variance)"
                    if std is not None and std < _STD_FLOOR
                    else "ok"
                ),
            }
        )
    return pl.DataFrame(
        records,
        schema={
            "feature": pl.Utf8(),
            "n_cells": pl.Int64(),
            "n_present": pl.Int64(),
            "n_missing": pl.Int64(),
            "coverage": pl.Float64(),
            "mean": pl.Float64(),
            "std": pl.Float64(),
            "dead": pl.Boolean(),
            "status": pl.Utf8(),
        },
    )


def format_coverage_report(coverage: pl.DataFrame) -> str:
    """Render :func:`ext_feature_coverage` output as a fixed-width text table."""
    header = (
        f"{'feature':<22}{'cells':>10}{'present':>10}{'coverage':>10}"
        f"{'mean':>14}{'std':>14}  status"
    )
    lines = [header, "-" * len(header)]
    for row in coverage.iter_rows(named=True):
        cov = row["coverage"]
        mean = row["mean"]
        std = row["std"]
        status = row["status"]
        lines.append(
            f"{row['feature']:<22}{row['n_cells']:>10}{row['n_present']:>10}"
            f"{'  n/a' if cov is None else f'{cov:>10.4f}'}"
            f"{'  n/a' if mean is None else f'{mean:>14.6g}'}"
            f"{'  n/a' if std is None else f'{std:>14.6g}'}  {status}"
        )
    return "\n".join(lines)


__all__: Final[Sequence[str]] = (
    "COVERAGE_SCHEMA",
    "EXT_FEATURE_COLS",
    "EXT_FEATURE_LOOKBACK_DAYS",
    "MAX_EXT_FEATURE_LOOKBACK_DAYS",
    "assert_disjoint_from_base",
    "combined_max_lookback_days",
    "compute_ext_features",
    "ext_feature_coverage",
    "format_coverage_report",
    "required_purge_months",
)
