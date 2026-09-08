"""Point-in-time universe selection from full-market bars.

`CLAUDE.md` lists survivorship under *Known-dangerous ground*, and
`trader.data.listing` is precise about which half of it was already handled:
mechanical look-ahead (trading a name before it listed) is prevented by two
independent belts, while **selection** — a universe chosen in 2026 — is not.

This module addresses selection. Measured 2026-09-08, the size of the problem:

* of the names clearing ₹5 crore of median daily turnover, the **645 panelled
  names cover 60.0%** on average over 2011–2020 — measured by
  `scripts/pit_universe_report.py`;
* at 2016-01-01, 99 eligible names are missing: **72 still trade** and were
  simply never selected, while **27 had stopped trading altogether** and no
  list drawn today can contain them.

**A figure quoted earlier in this project was conflated and is corrected here.**
Coverage against `active_tickers()` reads 49.5%, but roughly ten points of that
gap is `INACTIVE_SECTORS` — five sectors held out purely to keep the observation
width at 504 rather than 645 (`universe.py`), which is a compute decision and
not survivorship. Only coverage against `all_tickers()` isolates selection.

The fix is to stop choosing the universe at all, and derive it: on each
rebalance date, the eligible set is every name that a liquidity rule would have
admitted **using only sessions strictly before that date**. A name that later
delisted is in the 2019 universe because it was liquid in 2019, and a name that
IPO'd in 2023 is absent from 2019 because nobody could have bought it.

WHY A LIQUIDITY RULE AND NOT "EVERY LISTED NAME". NSE lists ~2,200 EQ/BE
securities; most are untradeable at any size. Admitting them would replace a
survivorship bias with a liquidity fantasy — a backtest that buys names nobody
could exit. The bar is a parameter, stated in the artefact, not a hidden
constant.

`listing.eligible_at` answers a different and complementary question — has this
name traded long *enough* — and both belts should be applied. This one is about
whether it was worth trading; that one is about whether there is history to
model it with.

Pure: frames in, plain Python out. No I/O, no network, no config.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Final

import polars as pl

_REQUIRED: Final[tuple[str, ...]] = ("date", "ticker", "series", "close", "turnover")


@dataclass(frozen=True)
class LiquidityRule:
    """What it takes to be in the universe on a given day.

    Defaults are the bar used to measure the survivorship gap on 2026-09-08 and
    are deliberately permissive: ₹5 crore of median daily turnover is roughly
    1/300th of RELIANCE's, and any name clearing it is enormously liquid
    relative to a ₹1 lakh book. The bar exists to exclude the untradeable tail,
    not to pick winners.
    """

    #: Median daily turnover over the lookback, in rupees.
    min_median_turnover: float = 5e7
    #: Calendar days of history examined. 365 ≈ one year of sessions.
    lookback_days: int = 365
    #: Sessions the name must actually have traded within that window. Guards
    #: against a name that traded twice at huge size and looks liquid by median
    #: of a two-element sample.
    min_sessions: int = 100
    #: Last close must clear this. A sub-₹5 stock cannot be sized sensibly
    #: against a flat ₹15.34 demat debit per sell.
    #:
    #: **Set this to 0 when the bars are BACK-ADJUSTED, which they are anywhere
    #: downstream of `nse_bhavcopy.back_adjust`.** A nominal floor applied to an
    #: adjusted series does not mean what it says: a stock at ₹50 in 2012 that
    #: later split 10:1 shows as ₹5 in the adjusted history and fails the floor
    #: for a price it never traded at. Worse, the bias has a direction — splits
    #: follow price appreciation, so adjusted early prices are systematically
    #: low for exactly the companies that went on to do well, and the floor
    #: would preferentially drop future winners out of the early universe.
    #: The turnover bar already excludes the untradeable tail.
    min_price: float = 5.0
    #: Cash-market series. EQ is normal rolling settlement; BE is
    #: trade-to-trade. Everything else is a different instrument.
    series: tuple[str, ...] = ("EQ", "BE")
    #: Keep only the N most liquid names that clear the bar, by trailing median
    #: turnover. ``None`` keeps all of them.
    #:
    #: This is what makes a point-in-time universe affordable to MODEL, not just
    #: to measure. The eligible set drifts between roughly 600 and 750 names,
    #: and a variable-width universe means a variable-width action space; worse,
    #: 750 names against the 504/L=30/mb=64 configuration measured on an 8 GiB
    #: card would not fit — cost scales linearly in ticker count (`CLAUDE.md`,
    #: Compute).
    #:
    #: Capping by liquidity is also the more honest rule. "Every name above a
    #: turnover floor" is not how anyone selects a book; "the most liquid N"
    #: is, and it keeps the width constant so a rebuilt panel is directly
    #: comparable with the fixed-504 one it is being tested against.
    max_names: int | None = None

    def __post_init__(self) -> None:
        if self.min_median_turnover < 0:
            raise ValueError("min_median_turnover must be >= 0")
        if self.lookback_days < 1:
            raise ValueError("lookback_days must be >= 1")
        if self.min_sessions < 1:
            raise ValueError(
                f"min_sessions must be >= 1, got {self.min_sessions}; 0 would "
                "admit a name with no sessions in the window at all"
            )
        if not self.series:
            raise ValueError("series must name at least one cash-market series")
        if self.max_names is not None and self.max_names < 1:
            raise ValueError(f"max_names must be >= 1 when set, got {self.max_names}")

    def describe(self) -> str:
        cap = f", top {self.max_names} by turnover" if self.max_names else ""
        return (
            f"median turnover >= Rs {self.min_median_turnover:,.0f} over "
            f"{self.lookback_days}d, >= {self.min_sessions} sessions, "
            f"close >= Rs {self.min_price:g}, series {'/'.join(self.series)}{cap}"
        )


def _validate(bars: pl.DataFrame) -> None:
    missing = [c for c in _REQUIRED if c not in bars.columns]
    if missing:
        raise ValueError(
            f"bars is missing column(s) {missing}; point-in-time selection "
            f"needs {list(_REQUIRED)}"
        )


def eligible_on(
    bars: pl.DataFrame, asof: date, rule: LiquidityRule | None = None
) -> list[str]:
    """Tickers the rule admits on ``asof``, from sessions strictly before it.

    ``asof`` itself is excluded. That strictness is the whole point: a decision
    taken on the morning of ``asof`` cannot use that day's turnover, and a rule
    that included it would be reading the tape it is about to trade into.

    Returns a sorted list, so a caller writing a universe file gets a
    deterministic order and therefore a stable hash.
    """
    r = rule or LiquidityRule()
    _validate(bars)
    lo = asof - timedelta(days=r.lookback_days)
    window = bars.lazy().filter(
        (pl.col("date") >= lo)
        & (pl.col("date") < asof)          # strictly before — no lookahead
        & pl.col("series").is_in(list(r.series))
        & pl.col("turnover").is_not_null()
    )
    agg = (
        window.group_by("ticker")
        .agg(
            pl.col("turnover").median().alias("med_turnover"),
            pl.col("date").n_unique().alias("sessions"),
            # The most recent close inside the window, for the price floor.
            pl.col("close").sort_by("date").last().alias("last_close"),
        )
        .filter(
            (pl.col("med_turnover") >= r.min_median_turnover)
            & (pl.col("sessions") >= r.min_sessions)
            & (pl.col("last_close") >= r.min_price)
        )
        .select(["ticker", "med_turnover"])
        .collect()
    )
    if r.max_names is not None and agg.height > r.max_names:
        # Ties broken by ticker so the selection is deterministic: two names on
        # identical turnover must not swap places between runs and change the
        # panel hash.
        agg = agg.sort(["med_turnover", "ticker"], descending=[True, False]).head(
            r.max_names
        )
    return sorted(agg["ticker"].to_list())


def universe_schedule(
    bars: pl.DataFrame,
    rebalance_dates: list[date],
    rule: LiquidityRule | None = None,
) -> dict[date, list[str]]:
    """``{rebalance date: eligible tickers}`` — the universe, as it was.

    One entry per date a selection is actually made. Between rebalances the
    universe does not change, because nothing acts on it.
    """
    r = rule or LiquidityRule()
    _validate(bars)
    return {d: eligible_on(bars, d, r) for d in sorted(rebalance_dates)}


def coverage_report(
    bars: pl.DataFrame,
    rebalance_dates: list[date],
    current_universe: list[str],
    rule: LiquidityRule | None = None,
) -> pl.DataFrame:
    """How much of the point-in-time universe a fixed list actually covers.

    This is the measurement that sized the problem, made reproducible so the
    number in `audit/` regenerates rather than being quoted (`CLAUDE.md` rule 3).

    Columns: ``date``, ``n_eligible``, ``n_covered``, ``coverage``,
    ``n_missing``. ``coverage`` below 1.0 is names that were liquid and
    tradeable on that date and are simply absent from ``current_universe``.
    """
    r = rule or LiquidityRule()
    _validate(bars)
    have = set(current_universe)
    rows: list[dict[str, object]] = []
    for d in sorted(rebalance_dates):
        elig = set(eligible_on(bars, d, r))
        covered = elig & have
        rows.append({
            "date": d,
            "n_eligible": len(elig),
            "n_covered": len(covered),
            "coverage": len(covered) / len(elig) if elig else float("nan"),
            "n_missing": len(elig - have),
        })
    return pl.DataFrame(rows)


def survivorship_gap(
    bars: pl.DataFrame,
    asof: date,
    current_universe: list[str],
    rule: LiquidityRule | None = None,
    *,
    still_trading_after: date | None = None,
) -> pl.DataFrame:
    """Names eligible on ``asof`` that a fixed universe leaves out, and why.

    Splits the gap into the two mechanisms, which have different remedies:

    * ``vanished`` — the name stopped trading before ``still_trading_after``.
      This is classic survivorship and cannot be fixed by widening a list drawn
      today, because the name is not on any list drawn today.
    * ``omitted`` — the name still trades and was simply not selected. This is
      selection bias and *is* fixable by widening the universe.

    Conflating the two overstates how much a bigger ticker list would help.
    """
    r = rule or LiquidityRule()
    _validate(bars)
    have = set(current_universe)
    missing = sorted(set(eligible_on(bars, asof, r)) - have)
    if not missing:
        return pl.DataFrame(
            schema={"ticker": pl.Utf8(), "last_seen": pl.Date(), "reason": pl.Utf8()}
        )
    cutoff = still_trading_after or bars["date"].max()
    last = (
        bars.lazy()
        .filter(pl.col("ticker").is_in(missing))
        .group_by("ticker")
        .agg(pl.col("date").max().alias("last_seen"))
        .collect()
    )
    return (
        last.with_columns(
            pl.when(pl.col("last_seen") < cutoff)
            .then(pl.lit("vanished"))
            .otherwise(pl.lit("omitted"))
            .alias("reason")
        )
        .sort(["reason", "ticker"])
    )


def apply_monthly_mask(
    panel: pl.DataFrame, schedule: dict[date, list[str]]
) -> pl.DataFrame:
    """AND ``is_tradeable`` with the month's eligible set, carried forward.

    ``schedule`` is keyed by month start, as :func:`universe_schedule` returns
    it. Every panel row takes the decision made at the start of its own month,
    so a name admitted at a rebalance stays tradeable until the next one —
    deciding daily would drop a name mid-hold on a liquidity wobble, which is
    not how the strategy behaves.

    A month with no entry in ``schedule`` makes every row in it untradeable.
    That is deliberate: a missing decision is not permission.

    Lives here rather than inline in `scripts/build_features.py` because it is
    the step that makes the whole rebuild mean anything, and a step that cannot
    be tested is a step that gets quietly wrong.
    """
    if "is_tradeable" not in panel.columns:
        raise ValueError("panel is missing is_tradeable")
    if "date" not in panel.columns or "ticker" not in panel.columns:
        raise ValueError("panel is missing date/ticker")

    elig = pl.DataFrame(
        {
            "_month": [m for m, names in schedule.items() for _ in names],
            "ticker": [t for names in schedule.values() for t in names],
        },
        schema={"_month": pl.Date(), "ticker": pl.Utf8()},
    ).with_columns(pl.lit(True).alias("_pit"))

    return (
        panel.with_columns(pl.col("date").dt.truncate("1mo").alias("_month"))
        .join(elig, on=["_month", "ticker"], how="left")
        .with_columns(
            (pl.col("is_tradeable") & pl.col("_pit").fill_null(False)).alias(
                "is_tradeable"
            )
        )
        .drop(["_month", "_pit"])
    )
