"""Unit tests for demerger detection and masking.

Two properties carry the module. First, masking must actually break the *feature*
chain, not just blank one row — a -76% return sits inside every 60-day rolling
window for the following three months. Second, detection must not fire on the
alignment stage's sentinel rows, or the triage list fills with artefacts and the
real events get skipped past.
"""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl

from trader.data.corporate_actions import (
    DEFAULT_CONTAMINATION_DAYS,
    CorporateEvent,
    detect_extreme_moves,
    format_triage_report,
    mask_corporate_events,
)

EVENT_DATE = date(2020, 6, 1)


def _panel(
    n_days: int = 200,
    drop_at: int | None = 100,
    drop_factor: float = 0.24,  # a 1:1 demerger-scale -76% move
    tickers: tuple[str, ...] = ("AAA.NS",),
) -> pl.DataFrame:
    """A dense two-column panel: consecutive weekdays, flat price, one cliff."""
    rows: list[dict[str, object]] = []
    for ticker in tickers:
        cursor = date(2020, 1, 6)
        price = 100.0
        for i in range(n_days):
            while cursor.weekday() >= 5:
                cursor += timedelta(days=1)
            if drop_at is not None and i == drop_at:
                price *= drop_factor
            rows.append(
                {
                    "date": cursor,
                    "ticker": ticker,
                    "adj_close": price,
                    "is_tradeable": True,
                }
            )
            cursor += timedelta(days=1)
    return pl.DataFrame(rows, schema={
        "date": pl.Date,
        "ticker": pl.Utf8,
        "adj_close": pl.Float64,
        "is_tradeable": pl.Boolean,
    })


def _event_date_of(panel: pl.DataFrame, index: int) -> date:
    return panel["date"].unique().sort().to_list()[index]


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


def test_detects_the_cliff() -> None:
    panel = _panel()
    found = detect_extreme_moves(panel, 0.35)
    assert len(found) == 1
    assert found["date"][0] == _event_date_of(panel, 100)
    assert found["pct_move"][0] == -0.76


def test_threshold_is_respected() -> None:
    panel = _panel(drop_factor=0.9)  # -10%
    assert detect_extreme_moves(panel, 0.35).is_empty()
    assert len(detect_extreme_moves(panel, 0.05)) == 1


def test_detects_upward_moves_too() -> None:
    """A demerger is a fall, but a bad tick or a stitch error can be a spike."""
    panel = _panel(drop_factor=2.5)
    found = detect_extreme_moves(panel, 0.35)
    assert len(found) == 1 and found["pct_move"][0] > 0


def test_non_tradeable_rows_never_produce_candidates() -> None:
    """Pre-listing sentinels are zeros; a 0 → 100 "move" is alignment, not a price event."""
    panel = _panel(drop_at=None)
    panel = panel.with_columns(
        pl.when(pl.col("date") < _event_date_of(panel, 20))
        .then(pl.lit(value=False))
        .otherwise(pl.col("is_tradeable"))
        .alias("is_tradeable"),
        pl.when(pl.col("date") < _event_date_of(panel, 20))
        .then(pl.lit(0.0))
        .otherwise(pl.col("adj_close"))
        .alias("adj_close"),
    )
    assert detect_extreme_moves(panel, 0.35).is_empty()


def test_gap_is_measured_between_consecutive_tradeable_sessions() -> None:
    """A suspension in the middle must not manufacture a move at the resumption."""
    panel = _panel(drop_at=None)
    dates = panel["date"].unique().sort().to_list()
    suspended = set(dates[50:60])
    panel = panel.with_columns(
        pl.when(pl.col("date").is_in(list(suspended)))
        .then(pl.lit(value=False))
        .otherwise(pl.col("is_tradeable"))
        .alias("is_tradeable")
    )
    assert detect_extreme_moves(panel, 0.35).is_empty()


def test_known_events_are_flagged_as_known() -> None:
    """Real NIITLTD numbers, so the flag is exercised against the shipped list."""
    panel = pl.DataFrame(
        {
            "date": [date(2023, 6, 7), date(2023, 6, 8)],
            "ticker": ["NIITLTD.NS", "NIITLTD.NS"],
            "adj_close": [398.58, 95.12],
            "is_tradeable": [True, True],
        },
        schema={
            "date": pl.Date,
            "ticker": pl.Utf8,
            "adj_close": pl.Float64,
            "is_tradeable": pl.Boolean,
        },
    )
    found = detect_extreme_moves(panel, 0.35)
    assert len(found) == 1
    assert bool(found["known"][0])


def test_empty_input_returns_typed_empty_frame() -> None:
    empty = detect_extreme_moves(pl.DataFrame(), 0.35)
    assert empty.is_empty()
    assert "pct_move" in empty.columns  # still joins/concats like a real result


def test_triage_report_separates_known_from_unexplained() -> None:
    panel = _panel()
    text = "\n".join(format_triage_report(detect_extreme_moves(panel, 0.35), 0.35))
    assert "TRIAGE" in text
    assert "1 unexplained" in text


# ---------------------------------------------------------------------------
# Masking
# ---------------------------------------------------------------------------


def _mask_event(panel: pl.DataFrame, days: int | None = None) -> CorporateEvent:
    return CorporateEvent(
        ticker="AAA.NS",
        event_date=_event_date_of(panel, 100),
        kind="demerger",
        note="synthetic",
        contamination_days=days,
    )


def test_mask_covers_the_event_day_and_the_feature_window() -> None:
    """One masked row is not enough: realized_vol_60d keeps the fake return for 60 sessions."""
    panel = _panel()
    masked, audit = mask_corporate_events(panel, (_mask_event(panel),))

    assert int(audit["rows_masked"][0]) == DEFAULT_CONTAMINATION_DAYS + 1
    still = masked.filter(pl.col("is_tradeable"))["date"].to_list()
    dates = panel["date"].unique().sort().to_list()
    assert dates[100] not in still
    assert dates[100 + DEFAULT_CONTAMINATION_DAYS] not in still
    # The session just past the window is clean again — this is a mask, not a delete.
    assert dates[100 + DEFAULT_CONTAMINATION_DAYS + 1] in still


def test_history_before_the_event_survives() -> None:
    """The whole reason masking replaces blacklisting: 20 years of good data stays."""
    panel = _panel()
    masked, _ = mask_corporate_events(panel, (_mask_event(panel),))
    before = masked.filter(pl.col("date") < _event_date_of(panel, 100))
    assert before["is_tradeable"].all()
    assert len(before) == 100


def test_masking_changes_only_the_flag_never_the_prices() -> None:
    panel = _panel()
    masked, _ = mask_corporate_events(panel, (_mask_event(panel),))
    assert masked["adj_close"].to_list() == panel["adj_close"].to_list()


def test_mask_window_is_counted_in_trading_days() -> None:
    """60 calendar days is ~41 sessions — a third of the window would stay dirty."""
    panel = _panel()
    _, audit = mask_corporate_events(panel, (_mask_event(panel, days=10),))
    dates = panel["date"].unique().sort().to_list()
    assert audit["mask_start"][0] == dates[100]
    assert audit["mask_end"][0] == dates[110]
    assert int(audit["rows_masked"][0]) == 11


def test_mask_only_touches_the_named_ticker() -> None:
    panel = _panel(tickers=("AAA.NS", "BBB.NS"))
    masked, _ = mask_corporate_events(panel, (_mask_event(panel, days=5),))
    other = masked.filter(pl.col("ticker") == "BBB.NS")
    assert other["is_tradeable"].all()


def test_event_outside_the_panel_is_reported_not_raised() -> None:
    """The same event list is shared by panels covering different date ranges."""
    panel = _panel()
    event = CorporateEvent("AAA.NS", date(2099, 1, 1), "demerger", "future")
    masked, audit = mask_corporate_events(panel, (event,))
    assert not bool(audit["matched"][0])
    assert int(audit["rows_masked"][0]) == 0
    assert masked["is_tradeable"].all()


def test_event_on_a_non_trading_day_snaps_to_the_next_session() -> None:
    """An announcement date is not a bar; the first bar after it carries the jump."""
    panel = _panel()
    dates = panel["date"].unique().sort().to_list()
    saturday = dates[100] - timedelta(days=1)
    while saturday in dates:
        saturday -= timedelta(days=1)
    event = CorporateEvent("AAA.NS", saturday, "demerger", "announcement date", 0)
    _, audit = mask_corporate_events(panel, (event,))
    assert bool(audit["matched"][0])
    assert audit["mask_start"][0] >= saturday


def test_mask_window_clamps_at_the_end_of_the_panel() -> None:
    panel = _panel()
    event = CorporateEvent("AAA.NS", _event_date_of(panel, 195), "demerger", "late")
    _, audit = mask_corporate_events(panel, (event,))
    assert audit["mask_end"][0] == _event_date_of(panel, 199)


def test_no_events_is_a_no_op() -> None:
    panel = _panel()
    masked, audit = mask_corporate_events(panel, ())
    assert masked.equals(panel)
    assert audit.is_empty()
