"""First-tradeable dates and point-in-time eligibility, read off a panel.

Why this exists
---------------
CLAUDE.md lists survivorship under *Known-dangerous ground*:
``market.universe_snapshots`` is empty and has no read site, so the universe is
not point-in-time.  That entry is true but imprecise about *which* of the two
distinct problems is live:

**(a) mechanical look-ahead — trading a name before it listed.**  Already
prevented, by two independent belts.  ``align_panel`` marks every date outside a
ticker's ``[first_date, last_date]`` span ``is_tradeable=False``
(``alignment.py:53-74``), and ``allocate`` intersects its candidate set with that
mask — ``cand = mask & np.isfinite(r_hat) & ...``
(``allocator/deterministic.py:150``; the line moves as that file changes, so it
is quoted here rather than cited by number alone).  ``tests/unit/test_listing.py``
asserts the second belt directly.

**(b) selection — a universe chosen in 2026.**  Not prevented and not
preventable from the data on hand.  The 645 names in ``data/panels_kite`` come
from a 2026 instrument dump, so the set embeds 2026 knowledge of which listings
mattered, and every name in it survived to 2026.

This module addresses the tractable half of (b): it lets a caller restrict the
universe to names that were *already trading well before* a backtest starts, so
the edge can be re-measured without any name whose post-hoc selection could be
doing the work.  It does **not** address delisting — see
``audit/P4_survivorship.md``, which states that as an unrecoverable limit.

Both functions are pure: a polars frame in, plain Python out.  No I/O, no
network, no config.
"""
from __future__ import annotations

from datetime import date
from typing import cast

import polars as pl

_REQUIRED = ("date", "ticker", "is_tradeable")


def _validate(panel: pl.DataFrame) -> None:
    """Raise unless ``panel`` carries the three columns these functions read."""
    missing = [c for c in _REQUIRED if c not in panel.columns]
    if missing:
        raise ValueError(
            f"panel is missing column(s) {missing}; "
            f"listing eligibility needs {list(_REQUIRED)}"
        )


def first_tradeable_dates(panel: pl.DataFrame) -> dict[str, date]:
    """``{ticker: first date on which it was tradeable}``.

    A ticker with **no** tradeable row anywhere in ``panel`` is absent from the
    result rather than mapped to a sentinel: "never traded here" and "traded
    from day one" are different facts and a caller that conflates them would
    silently admit a dead column to the universe.

    The date is read off the ``is_tradeable`` mask, not off price presence.
    That is deliberate — ``features.py`` forces ``is_tradeable=False`` on
    warm-up rows where a rolling feature is still null (``features.py:212-221``),
    so the mask is strictly more conservative than the first price bar, and the
    mask is what ``allocate`` actually gates on.
    """
    _validate(panel)
    grouped = (
        panel.lazy()
        .filter(pl.col("is_tradeable"))
        .group_by("ticker")
        .agg(pl.col("date").min().alias("first_tradeable"))
        .collect()
    )
    tickers = cast(list[str], grouped["ticker"].to_list())
    firsts = cast(list[date], grouped["first_tradeable"].to_list())
    return dict(zip(tickers, firsts, strict=True))


def eligible_at(
    panel: pl.DataFrame,
    asof: date,
    min_history_days: int,
) -> list[str]:
    """Tickers with ``>= min_history_days`` tradeable days **strictly before** ``asof``.

    Parameters
    ----------
    panel:
        Any frame carrying ``date``, ``ticker`` and ``is_tradeable``.  It must
        be the panel that *contains the history*: asking a split that begins in
        2016 which names were eligible in 2014 correctly returns nothing,
        because that frame holds no pre-2014 rows.  Pass ``full.parquet``.
    asof:
        The cut.  Rows dated ``asof`` itself do **not** count.  Strictness is
        the point: a name whose first tradeable day *is* the cut has zero days
        of prior history, and admitting it would reintroduce exactly the
        look-ahead this function exists to exclude.
    min_history_days:
        Distinct tradeable **dates** required before ``asof``.  Counted as
        distinct dates, not rows, so a panel that ever carries a duplicated
        ``(date, ticker)`` cannot inflate a name into eligibility.

        Must be ``>= 1``.  Zero is rejected rather than treated as "no
        constraint": "at least 0 days of history" is satisfied by a name with
        no history at all, which is the one answer this function must never
        give.  Use ``1`` for "has ever traded before the cut".

    Returns
    -------
    Sorted list of tickers.  Sorted so a caller writing a split file gets a
    deterministic ticker order and therefore a stable file hash.
    """
    _validate(panel)
    if min_history_days < 1:
        raise ValueError(
            f"min_history_days must be >= 1, got {min_history_days}; "
            "0 would admit a name with no tradeable history at all"
        )
    counts = (
        panel.lazy()
        .filter(pl.col("is_tradeable") & (pl.col("date") < pl.lit(asof, dtype=pl.Date)))
        .group_by("ticker")
        .agg(pl.col("date").n_unique().alias("n_days"))
        .filter(pl.col("n_days") >= min_history_days)
        .collect()
    )
    return sorted(cast(list[str], counts["ticker"].to_list()))


def listing_year_counts(panel: pl.DataFrame) -> dict[int, int]:
    """``{year: how many tickers first became tradeable in it}``.

    The earliest year is not a listing cohort — it is everything that was
    already trading when the panel's calendar opens, which for
    ``data/panels_kite/full.parquet`` is 2005-01-03.  Read it as "already
    listed at panel start", never as "listed in 2005".
    """
    firsts = first_tradeable_dates(panel)
    out: dict[int, int] = {}
    for first in firsts.values():
        out[first.year] = out.get(first.year, 0) + 1
    return dict(sorted(out.items()))
