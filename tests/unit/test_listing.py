"""Unit tests for listing eligibility, and for the no-look-ahead claim it rests on.

Two things are being tested here and they are not the same thing.

``trader.data.listing`` is new code and is tested the ordinary way.

The load-bearing test is :func:`test_allocate_never_selects_a_name_before_it_lists`.
It asserts a property of code this stream does **not** own — ``allocate`` is
imported read-only — because the entire P4 argument rests on it: if the
allocator could pick a name on a date the panel says it was not tradeable, the
survivorship question would be a live look-ahead *leak* rather than a
*selection* bias, and the restricted-universe arm would be the wrong response.
The test is built so the mask is the only thing standing in the way: the
pre-listing name carries the best ``r_hat`` on the grid, a finite positive
``vol`` and a valid sector, so deleting ``mask &`` from the candidate line makes
it the top pick and the assertion fails.  (Verified by running the assertion
against a copy of ``deterministic.py`` with ``mask &`` removed — it picks the
unlisted name at weight 0.10 on every pre-listing date.)
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from trader.allocator import AllocatorParams, allocate
from trader.data.listing import eligible_at, first_tradeable_dates, listing_year_counts

# ── fixtures ──────────────────────────────────────────────────────────────────

_START = date(2020, 1, 1)


def _panel(spans: dict[str, tuple[int, int]], n_days: int = 10) -> pl.DataFrame:
    """Dense (date × ticker) frame; ``spans[t] = (first_idx, last_idx)`` inclusive.

    Mirrors what ``align_panel`` produces: every ticker has a row on every
    date, and ``is_tradeable`` is what distinguishes a live row from a
    pre-listing or post-delisting one.
    """
    dates = [_START + timedelta(days=i) for i in range(n_days)]
    rows: list[dict[str, object]] = []
    for ticker, (lo, hi) in spans.items():
        for i, d in enumerate(dates):
            rows.append({"date": d, "ticker": ticker, "is_tradeable": lo <= i <= hi})
    return pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Date)).sort(
        ["date", "ticker"]
    )


# ── first_tradeable_dates ─────────────────────────────────────────────────────


def test_first_tradeable_date_is_the_first_masked_in_day() -> None:
    """A ticker that lists mid-panel reports its listing day, not the panel's start."""
    panel = _panel({"OLD": (0, 9), "NEW": (4, 9)})
    got = first_tradeable_dates(panel)
    assert got == {"OLD": _START, "NEW": _START + timedelta(days=4)}


def test_ticker_with_no_tradeable_row_is_absent_not_sentinel() -> None:
    """"Never traded" must not be reported as "traded from day one"."""
    panel = _panel({"OLD": (0, 9), "DEAD": (5, 4)})   # empty span → all False
    got = first_tradeable_dates(panel)
    assert "DEAD" not in got
    assert got == {"OLD": _START}


def test_missing_column_raises_rather_than_returning_empty() -> None:
    """A frame without ``is_tradeable`` is a caller bug, not an empty universe."""
    panel = _panel({"A": (0, 9)}).drop("is_tradeable")
    with pytest.raises(ValueError, match="is_tradeable"):
        first_tradeable_dates(panel)
    with pytest.raises(ValueError, match="is_tradeable"):
        eligible_at(panel, _START, 1)


def test_listing_year_counts_buckets_by_first_tradeable_year() -> None:
    panel = pl.concat(
        [
            _panel({"A": (0, 9), "B": (3, 9)}, n_days=10),
            _panel({"C": (0, 9)}, n_days=10).with_columns(
                pl.col("date").dt.offset_by("2y")
            ),
        ]
    )
    assert listing_year_counts(panel) == {2020: 2, 2022: 1}


# ── eligible_at ───────────────────────────────────────────────────────────────


def test_eligible_at_counts_only_history_strictly_before_the_cut() -> None:
    """A name whose first tradeable day *is* the cut has zero prior history.

    This is the semantics the restricted arm depends on.  With ``<=`` instead of
    ``<`` in ``eligible_at``, ``AT_CUT`` below becomes eligible on one day of
    "history" that is the cut itself.
    """
    panel = _panel({"EARLY": (0, 9), "AT_CUT": (5, 9), "LATE": (6, 9)})
    cut = _START + timedelta(days=5)
    assert eligible_at(panel, cut, min_history_days=1) == ["EARLY"]
    assert "AT_CUT" not in eligible_at(panel, cut, min_history_days=1)


def test_eligible_at_threshold_is_inclusive_at_the_boundary() -> None:
    """``>= min_history_days``: exactly n days of history qualifies, n-1 does not."""
    panel = _panel({"THREE": (0, 9), "TWO": (1, 9)})
    cut = _START + timedelta(days=3)          # THREE has 3 days before, TWO has 2
    assert eligible_at(panel, cut, min_history_days=3) == ["THREE"]
    assert eligible_at(panel, cut, min_history_days=2) == ["THREE", "TWO"]
    assert eligible_at(panel, cut, min_history_days=4) == []


def test_eligible_at_counts_distinct_dates_not_rows() -> None:
    """A duplicated (date, ticker) must not inflate a name into eligibility."""
    panel = _panel({"DUP": (0, 9)})
    doubled = pl.concat([panel, panel]).sort(["date", "ticker"])
    cut = _START + timedelta(days=2)
    assert eligible_at(doubled, cut, min_history_days=3) == []
    assert eligible_at(doubled, cut, min_history_days=2) == ["DUP"]


def test_eligible_at_rejects_zero_min_history() -> None:
    """"At least 0 days of history" is satisfied by a name with none at all."""
    panel = _panel({"A": (0, 9)})
    with pytest.raises(ValueError, match="min_history_days must be >= 1"):
        eligible_at(panel, _START, 0)


def test_eligible_at_returns_sorted_for_a_stable_split_hash() -> None:
    panel = _panel({"ZEE": (0, 9), "ACC": (0, 9), "MMM": (0, 9)})
    cut = _START + timedelta(days=5)
    assert eligible_at(panel, cut, min_history_days=1) == ["ACC", "MMM", "ZEE"]


def test_eligible_at_on_a_split_with_no_prior_history_returns_nothing() -> None:
    """The trap the docstring names: ask a 2016-onwards split about 2014."""
    panel = _panel({"A": (0, 9), "B": (0, 9)})
    assert eligible_at(panel, _START - timedelta(days=1), min_history_days=1) == []


# ── the claim P4 rests on: allocate() cannot trade a name before it lists ──────


def test_allocate_never_selects_a_name_before_it_lists() -> None:
    """A name that has NEVER been tradeable can never acquire weight.

    That is the exact claim, and it is narrower than "``allocate`` returns 0.0
    for any masked-out name", which this test used to say and which is **false**
    — see ``test_a_held_name_that_becomes_untradeable_is_not_liquidated_in_one
    _step`` below.  The mask gates *candidacy*, so a name that never listed can
    never be bought; it does not gate *holdings*, so a name already held is not
    forced to zero when its mask goes False.  P4 needs only the first, because a
    pre-listing name is never held.

    Construction: ``UNLISTED`` becomes tradeable on day 5 of 10 and is handed
    the highest ``r_hat`` on the grid throughout, plus the lowest ``vol`` (so
    inverse-vol weighting wants it most), a valid sector and ``k`` large enough
    that it is never crowded out on rank.  Before day 5 the mask is the only
    thing excluding it, and the book starts in all cash so it has never been
    held.
    """
    n_days, n = 10, 6
    listing_day = 5
    tickers = ["UNLISTED"] + [f"OLD{i}" for i in range(n - 1)]
    spans = {"UNLISTED": (listing_day, n_days - 1)}
    for t in tickers[1:]:
        spans[t] = (0, n_days - 1)
    panel = _panel(spans, n_days=n_days)

    # The panel and the arrays must agree on ticker order.
    order = sorted(spans)
    idx_unlisted = order.index("UNLISTED")

    r_hat = np.full(n, 0.01)
    r_hat[idx_unlisted] = 10.0                # best signal on the grid, by far
    vol = np.full(n, 0.30)
    vol[idx_unlisted] = 0.05                  # and the most attractive vol
    sector_ids = np.arange(1, n + 1, dtype=np.int64)
    params = AllocatorParams(
        k=n, max_name_weight=0.10, max_sector_weight=1.0, turnover_budget=2.0
    )

    mask_by_date = {
        (row["date"], row["ticker"]): bool(row["is_tradeable"])
        for row in panel.iter_rows(named=True)
    }
    dates = sorted({d for d, _ in mask_by_date})

    current_w = np.zeros(n + 1)
    current_w[0] = 1.0
    seen_after = 0.0
    for d in dates:
        mask = np.array([mask_by_date[(d, t)] for t in order], dtype=bool)
        w = allocate(r_hat, vol, mask, sector_ids, current_w, params)
        assert w.shape == (n + 1,)
        if d < dates[listing_day]:
            assert w[1 + idx_unlisted] == 0.0, (
                f"{d}: allocate() gave UNLISTED weight {w[1 + idx_unlisted]!r} "
                f"on a date the panel marks is_tradeable=False — that is a "
                f"look-ahead leak, not a selection bias"
            )
        else:
            seen_after = max(seen_after, float(w[1 + idx_unlisted]))
        current_w = w

    # Negative control on the test itself: once listed, this name IS picked, so
    # the assertion above is excluding it on the mask and not on something else
    # (an unsatisfiable cap, a k that never reaches it, a NaN).
    assert seen_after > 0.0, "UNLISTED was never selected even after listing"


def test_allocate_holds_cash_when_every_name_is_pre_listing() -> None:
    """The degenerate day: nothing has listed yet, so the book is all cash."""
    n = 4
    r_hat = np.linspace(0.05, 0.20, n)
    vol = np.full(n, 0.25)
    mask = np.zeros(n, dtype=bool)
    sector_ids = np.arange(1, n + 1, dtype=np.int64)
    current_w = np.zeros(n + 1)
    current_w[0] = 1.0
    w = allocate(r_hat, vol, mask, sector_ids, current_w, AllocatorParams(k=n))
    assert w[0] == pytest.approx(1.0)
    assert np.all(w[1:] == 0.0)


def test_a_held_name_that_becomes_untradeable_is_not_liquidated_in_one_step() -> None:
    """The counterexample to the over-broad claim, so nobody re-establishes it.

    ``mask`` decides which names may be BOUGHT.  It does not force a held name
    to zero: ``turnover_budget`` scales the exit like every other trade, so an
    untradeable holding drains over several rebalances rather than in one.  With
    ``no_trade_band`` above its weight it does not drain at all.

    Harmless on this panel — it contains zero delistings, so no name ever goes
    permanently untradeable — and live the moment a point-in-time universe lands
    (`audit/P4_survivorship.md` §4), because a delisting is exactly this case.
    """
    n = 5
    vol = np.full(n, 0.25)
    sector_ids = np.arange(1, n + 1, dtype=np.int64)
    r_hat = np.linspace(0.05, 0.01, n)
    held = np.zeros(n + 1)
    held[1:] = 0.2                                   # fully invested, equal weight

    # (a) every name goes untradeable at once: the book cannot be liquidated in
    #     one rebalance, and the turnover budget is exactly what stops it.
    params = AllocatorParams()
    assert params.turnover_budget == 0.30 and params.no_trade_band == 0.0
    w = allocate(r_hat, vol, np.zeros(n, dtype=bool), sector_ids, held, params)
    assert w[0] == pytest.approx(0.30), w
    np.testing.assert_allclose(w[1:], 0.14, atol=1e-12)
    # and it takes several more rebalances to drain
    w2 = allocate(r_hat, vol, np.zeros(n, dtype=bool), sector_ids, w, params)
    assert float(w2[1:].sum()) > 0.0

    # (b) one name of five goes untradeable: it keeps most of its weight.
    mask = np.ones(n, dtype=bool)
    mask[0] = False
    w3 = allocate(r_hat, vol, mask, sector_ids, held, params)
    assert w3[1] > 0.0, "a held untradeable name was liquidated in one step"

    # (c) with a band wider than its weight it is pinned indefinitely.
    banded = AllocatorParams(no_trade_band=0.5)
    w4 = allocate(r_hat, vol, mask, sector_ids, held, banded)
    assert w4[1] == pytest.approx(held[1]), "the band did not pin the stranded name"
