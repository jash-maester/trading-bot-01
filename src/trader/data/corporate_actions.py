"""Detect and neutralise corporate actions that back-adjustment does not handle.

The gap this closes
-------------------
``yfinance``'s ``auto_adjust=True`` back-adjusts for splits, bonuses and dividends.
Kite adjusts for splits and bonuses. **Neither adjusts for a demerger.** When a
company spins a division out into a separately listed entity, the parent's market
cap genuinely falls by the value of what left, and every price feed records that
fall as an ordinary close-to-close move. It is not one: a shareholder who held
through it lost nothing — they received shares in the new entity.

Two confirmed instances in this universe:

* ``NIITLTD`` 2023-06-08, **-76.13%** — NIIT Learning Systems demerged 1:1.
* ``MASTEK``  2015-06-12, **-66.00%** — Majesco demerged 1:1.

A -76% one-day return is not a rounding error. It is the single largest move in the
whole panel, it is fictional, and left in place it teaches the policy that this
particular configuration of RSI, volatility and volume precedes catastrophe.

Why masking rather than deleting the ticker
-------------------------------------------
The pre-existing remedy was ``BLACKLISTED_TICKERS`` — dropping the entire symbol
forever. That is a bad trade: it removes a NIFTY-50 name for two decades to avoid
one bad day, and it is silent, so the universe shrinks without anything saying why.
Masking sets ``is_tradeable=False`` on the affected rows instead. The environment
already treats that flag as the source of truth (it gates position sizing and is
the weight in the equal-weight benchmark return), so a masked row cannot enter the
reward, the benchmark, or a held position. Everything before and after the event
survives.

Why the mask is a *window*, not a single day
--------------------------------------------
Masking only the event date breaks the return chain at t, but every feature in
``FEATURE_COLS`` is a backward-looking rolling statistic: ``realized_vol_60d`` and
``beta_nifty_60d`` keep the fake -76% in their window for the next 60 trading days,
``realized_vol_20d`` and ``z_close_20`` for 20, and MACD's EMAs decay rather than
expire. So the default mask covers the event day plus
:data:`DEFAULT_CONTAMINATION_DAYS` trading days after it, which is the longest
rolling window in the feature set. Roughly 60 rows per event — against ~5,300 rows
per ticker, a rounding error, and the alternative is a feature vector that is
quietly wrong for three months.

The known-event list is data
----------------------------
:data:`KNOWN_CORPORATE_EVENTS` is a plain tuple of records. Adding next year's
demerger is one line there; it must never require touching the masking logic. Use
:func:`detect_extreme_moves` to find candidates in a rebuilt panel, triage them by
hand (large moves are frequently *real* — YESBANK's RBI moratorium and the
Adani/Hindenburg selloff both belong in the data), and promote the genuine
corporate actions into the list.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Final

import polars as pl
from loguru import logger

# A single-day equity move this large is almost never an ordinary market move on a
# NIFTY-scale name. It is a threshold for *triage*, not for automatic correction —
# see the module docstring on why real crashes must stay in the data.
DEFAULT_MOVE_THRESHOLD: Final = 0.35

# Longest backward-looking window in trader.data.features.FEATURE_COLS
# (realized_vol_60d, beta_nifty_60d). Any row within this many trading days *after*
# the event still has the fake return inside at least one of its feature windows.
DEFAULT_CONTAMINATION_DAYS: Final = 60


@dataclass(frozen=True)
class CorporateEvent:
    """One known price discontinuity that no feed's back-adjustment removes.

    ``ticker`` is the project ticker (``NIITLTD.NS``). ``event_date`` is the first
    session that trades ex-entitlement, i.e. the date whose *close-to-previous-close*
    return is the fictional one.
    """

    ticker: str
    event_date: date
    kind: str
    note: str
    contamination_days: int | None = None
    """Override the default mask length. None → :data:`DEFAULT_CONTAMINATION_DAYS`."""


KNOWN_CORPORATE_EVENTS: Final[tuple[CorporateEvent, ...]] = (
    CorporateEvent(
        ticker="NIITLTD.NS",
        event_date=date(2023, 6, 8),
        kind="demerger",
        note=(
            "NIIT Learning Systems (now NSE: NIITMTS) demerged from NIIT Ltd, 1:1 "
            "allotment. Observed -76.13% in the yfinance panel; holders lost nothing."
        ),
    ),
    CorporateEvent(
        ticker="MASTEK.NS",
        event_date=date(2015, 6, 12),
        kind="demerger",
        note=(
            "Majesco demerged from Mastek, 1:1 allotment. Observed -66.00% in the "
            "yfinance panel; holders lost nothing."
        ),
    ),
    CorporateEvent(
        ticker="TATAMOTORS.NS",
        event_date=date(2025, 10, 14),
        kind="demerger",
        note=(
            "Tata Motors split into passenger (TMPV, which retains instrument token "
            "884737 and all history) and commercial (TMCV, a new listing from "
            "2025-11-12) vehicles. Only relevant to the Kite panel, where "
            "TATAMOTORS.NS resolves through the TMPV alias; the yfinance dataset "
            "never had this ticker at all. Date is the ex-demerger session."
        ),
    ),
    # ── Kite feed artefacts ──────────────────────────────────────────────────
    # Not corporate actions: places where the *feed* is discontinuous. Same
    # treatment, because the model cannot tell the difference between a fictional
    # return caused by a demerger and one caused by an archive seam.
    CorporateEvent(
        ticker="MASTEK.NS",
        event_date=date(2015, 1, 1),
        kind="feed_seam",
        note=(
            "Kite archive boundary: +194.96% (135.29 → 399.05) with no corporate "
            "action. Kite's pre-2015-01-01 history for Mastek is already adjusted for "
            "the June-2015 Majesco demerger while its 2015+ bars are not, so the two "
            "halves are on different scales and the seam shows up as a 2.9x jump. "
            "Confirmed against yfinance, whose series is continuous across this date."
        ),
    ),
    CorporateEvent(
        ticker="CGPOWER.NS",
        event_date=date(2015, 1, 1),
        kind="feed_seam",
        note=(
            "Kite archive boundary: +190.67% (64.18 → 186.55) with no corporate "
            "action — same 2015-01-01 seam as MASTEK, here against the March-2016 "
            "consumer-products demerger. yfinance is continuous across this date."
        ),
    ),
    CorporateEvent(
        ticker="ABBOTINDIA.NS",
        event_date=date(2010, 1, 8),
        kind="feed_seam",
        note=(
            "Kite serves four placeholder bars at the very start of this instrument's "
            "history (2010-01-04..07: an unchanging 270/270/260/265 OHLC on ZERO "
            "volume), then the first real bar at 790 — a fictional +198.11%. The "
            "zero volume is what identifies these as placeholders rather than trades."
        ),
    ),
)


def detect_extreme_moves(
    frame: pl.DataFrame,
    threshold: float = DEFAULT_MOVE_THRESHOLD,
    price_col: str = "adj_close",
) -> pl.DataFrame:
    """Flag every single-day move beyond ``threshold`` on a tradeable row.

    Consecutive *tradeable* rows only. A move measured across a suspension or a
    pre-listing sentinel row is an artefact of the alignment stage, not a price
    event, and including those would bury the handful of rows worth looking at.

    Returns one row per candidate with ``ticker, date, prev_close, close, pct_move``,
    a ``known`` flag saying whether :data:`KNOWN_CORPORATE_EVENTS` already explains
    it — so the triage question is only ever "what is new here" — and ``gap_days``,
    the calendar distance back to the previous tradeable session. ``gap_days`` is
    load-bearing: a "+38% move" measured across a four-month hole in the feed is a
    data-quality bug, not a corporate action, and masking it would hide the hole.
    """
    if frame.is_empty():
        return _empty_candidates()

    work = frame.select(
        [c for c in ("date", "ticker", price_col, "is_tradeable") if c in frame.columns]
    ).sort(["ticker", "date"])

    if "is_tradeable" not in work.columns:
        work = work.with_columns(pl.lit(value=True).alias("is_tradeable"))

    # Restrict to tradeable rows *before* shifting, so `prev` is the previous
    # tradeable session rather than the previous calendar row.
    work = work.filter(pl.col("is_tradeable") & (pl.col(price_col) > 0.0))
    work = work.with_columns(
        [
            pl.col(price_col).shift(1).over("ticker").alias("prev_close"),
            pl.col("date").shift(1).over("ticker").alias("prev_date"),
        ]
    ).drop_nulls("prev_close")

    work = work.with_columns(
        ((pl.col(price_col) / pl.col("prev_close")) - 1.0).alias("pct_move")
    )
    candidates = work.filter(pl.col("pct_move").abs() >= threshold)

    known_keys = {(e.ticker, e.event_date) for e in KNOWN_CORPORATE_EVENTS}
    return (
        candidates.select(
            pl.col("ticker"),
            pl.col("date"),
            pl.col("prev_date"),
            pl.col("prev_close"),
            pl.col(price_col).alias("close"),
            pl.col("pct_move"),
        )
        .with_columns(
            pl.struct(["ticker", "date"])
            .map_elements(
                lambda s: (str(s["ticker"]), s["date"]) in known_keys,
                return_dtype=pl.Boolean,
            )
            .alias("known"),
            (pl.col("date") - pl.col("prev_date")).dt.total_days().alias("gap_days"),
        )
        .sort("pct_move")
    )


def format_triage_report(candidates: pl.DataFrame, threshold: float) -> list[str]:
    """Render the candidate table as log lines, unexplained moves first.

    Deliberately not a pass/fail: a large move is a *question*. The report exists so
    that answering it is a five-minute job rather than a re-derivation.
    """
    if candidates.is_empty():
        return [f"No single-day moves beyond ±{threshold:.0%} — nothing to triage."]

    n_known = int(candidates["known"].sum())
    lines = [
        f"Corporate-action triage: {len(candidates)} move(s) beyond ±{threshold:.0%} "
        f"({n_known} already explained by KNOWN_CORPORATE_EVENTS, "
        f"{len(candidates) - n_known} unexplained)"
    ]
    ordered = candidates.sort(["known", "pct_move"])
    for row in ordered.iter_rows(named=True):
        tag = "known " if row["known"] else "TRIAGE"
        gap = int(row.get("gap_days") or 1)
        # Anything past a long weekend means the previous bar is not "yesterday",
        # so the percentage is a stitch across missing data rather than a move.
        suffix = f"  ⚠ measured across a {gap}-day gap since {row['prev_date']}" if gap > 5 else ""
        lines.append(
            f"  [{tag}] {row['ticker']:18s} {row['date']}  "
            f"{row['pct_move']:+7.2%}  {row['prev_close']:.2f} → {row['close']:.2f}{suffix}"
        )
    if len(candidates) - n_known:
        lines.append(
            "  Unexplained moves are not automatically corrected. Genuine crashes "
            "(YESBANK's 2020 moratorium, the 2023 Adani selloff) belong in the data; "
            "only add an entry to KNOWN_CORPORATE_EVENTS once the corporate action is "
            "confirmed."
        )
    return lines


def mask_corporate_events(
    panel: pl.DataFrame,
    events: tuple[CorporateEvent, ...] = KNOWN_CORPORATE_EVENTS,
    default_contamination_days: int = DEFAULT_CONTAMINATION_DAYS,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Set ``is_tradeable=False`` over each event's contamination window.

    Only the flag is touched. Prices are left exactly as the feed reported them, so
    the panel stays an honest record of what traded and the mask stays the single
    source of truth about what may be *acted on* — the same contract the alignment
    stage already established for pre-listing and suspended rows.

    An event whose ticker is absent from the panel, or whose date falls outside it,
    is reported as ``matched=False`` rather than raising: the same event list is
    shared by panels covering different date ranges (2005-2026 Kite, 2014-2024
    yfinance), and an event predating a panel is expected, not an error.

    Returns ``(masked_panel, audit)`` where ``audit`` has one row per event with the
    resolved window and how many rows it actually changed.
    """
    if panel.is_empty() or not events:
        return panel, _empty_audit()

    calendar = panel["date"].unique().sort().to_list()
    audit_rows: list[dict[str, object]] = []
    mask_expr: pl.Expr | None = None

    for event in events:
        window = (
            event.contamination_days
            if event.contamination_days is not None
            else default_contamination_days
        )
        first, last, matched = _resolve_window(calendar, event.event_date, window)
        n_rows = 0
        if matched:
            hit = (
                (pl.col("ticker") == event.ticker)
                & (pl.col("date") >= first)
                & (pl.col("date") <= last)
            )
            n_rows = int(
                panel.filter(hit & pl.col("is_tradeable")).height
                if "is_tradeable" in panel.columns
                else panel.filter(hit).height
            )
            mask_expr = hit if mask_expr is None else (mask_expr | hit)

        audit_rows.append(
            {
                "ticker": event.ticker,
                "event_date": event.event_date,
                "kind": event.kind,
                "matched": matched,
                "mask_start": first,
                "mask_end": last,
                "rows_masked": n_rows,
                "note": event.note,
            }
        )

    audit = pl.DataFrame(audit_rows, schema=_AUDIT_SCHEMA)

    if mask_expr is None:
        logger.warning(
            "mask_corporate_events: none of the known events fall inside this panel; "
            "nothing was masked."
        )
        return panel, audit

    masked = panel.with_columns(
        pl.when(mask_expr).then(pl.lit(value=False)).otherwise(pl.col("is_tradeable"))
        .alias("is_tradeable")
    )
    total = int(audit["rows_masked"].sum())
    logger.info(
        f"Masked {total:,} tradeable rows across "
        f"{int(audit['matched'].sum())} corporate event(s)"
    )
    return masked, audit


# ── private helpers ───────────────────────────────────────────────────────────


_AUDIT_SCHEMA: Final[dict[str, pl.DataType]] = {
    "ticker": pl.Utf8(),
    "event_date": pl.Date(),
    "kind": pl.Utf8(),
    "matched": pl.Boolean(),
    "mask_start": pl.Date(),
    "mask_end": pl.Date(),
    "rows_masked": pl.Int64(),
    "note": pl.Utf8(),
}


def _empty_audit() -> pl.DataFrame:
    return pl.DataFrame(schema=_AUDIT_SCHEMA)


def _empty_candidates() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "ticker": pl.Utf8(),
            "date": pl.Date(),
            "prev_date": pl.Date(),
            "prev_close": pl.Float64(),
            "close": pl.Float64(),
            "pct_move": pl.Float64(),
            "known": pl.Boolean(),
            "gap_days": pl.Int64(),
        }
    )


def _resolve_window(
    calendar: list[date], event_date: date, contamination_days: int
) -> tuple[date | None, date | None, bool]:
    """Map an event date onto ``contamination_days`` *trading* days of the calendar.

    Counting in trading days rather than calendar days matters: 60 calendar days is
    about 41 sessions, which leaves the last third of a 60-session feature window
    still carrying the fictional return.

    If the event date is not itself a trading day (an announcement date, or a panel
    built from a coarser calendar) the window starts at the first session on or
    after it — that is the first bar whose return can carry the discontinuity.
    """
    start_idx = _first_index_at_or_after(calendar, event_date)
    if start_idx is None:
        return None, None, False
    end_idx = min(start_idx + contamination_days, len(calendar) - 1)
    return calendar[start_idx], calendar[end_idx], True


def _first_index_at_or_after(calendar: list[date], target: date) -> int | None:
    from bisect import bisect_left

    idx = bisect_left(calendar, target)
    return idx if idx < len(calendar) else None
