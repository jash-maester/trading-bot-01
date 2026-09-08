"""Cross-sectional features from NSE quarterly results.

`13_fundamentals_and_news.md` section 3b. Two design decisions dominate this
module and both come from that section.

**Level and change are kept separate, deliberately.** A fundamental *level* is
near-constant inside a 20-day horizon, so it behaves as a stock fixed effect:
the model can learn "these particular tickers did well in 2016-2024" rather than
"profitable companies outperform", which memorises names and will not transfer.
The *change* — a margin inflection, a growth surprise — is an event, is higher
frequency, and is far more plausible at our horizon. Every level feature here
has a change counterpart so the two can be measured apart.

**Nothing is computed from a figure before it was public.** Every row carries
``visible_from`` from `nse_fundamentals`, and the panel join uses that column,
never ``period_to``. Measured lag: median 40 days, p10 24, p90 81 — so a fixed
lag would be wrong for most companies in both directions.

WHAT THIS SOURCE CANNOT GIVE. Quarterly results filings carry the P&L only.
There is no balance sheet, so return on equity, book-to-price and debt-to-equity
are **not buildable here** and their absence is a property of the data, not an
oversight. Shares outstanding are recoverable as
``paid_up_equity / face_value`` — verified against INFY, which yields 4.144bn
shares against a reported ~4.14bn — which is what makes earnings yield possible.
"""
from __future__ import annotations

from typing import Final

import numpy as np
import polars as pl

#: Level features: what the company looks like now.
LEVEL_COLS: Final[tuple[str, ...]] = (
    "f_net_margin",
    "f_pbt_margin",
    "f_employee_cost_ratio",
    "f_finance_cost_ratio",
    "f_tax_rate",
    "f_earnings_yield",
)

#: Change features: what just moved. These are the ones expected to carry signal
#: at a 5-20 day horizon; see the module docstring.
CHANGE_COLS: Final[tuple[str, ...]] = (
    "f_revenue_growth_yoy",
    "f_profit_growth_yoy",
    "f_eps_growth_yoy",
    "f_net_margin_change_yoy",
    "f_revenue_surprise",
    "f_profit_surprise",
)

FUNDAMENTAL_COLS: Final[tuple[str, ...]] = LEVEL_COLS + CHANGE_COLS

_EPS: Final[float] = 1e-9


def _safe_ratio(num: str, den: str, alias: str) -> pl.Expr:
    """``num/den``, null where the denominator is absent or ~zero.

    Null rather than 0.0 on purpose. A zero-denominator row is *unknown*, and
    filling it with a number would put a fabricated value into a
    cross-sectional rank — the same class of error as a dead feature channel
    reading a constant.
    """
    return (
        pl.when(pl.col(den).is_null() | (pl.col(den).abs() < _EPS))
        .then(None)
        .otherwise(pl.col(num) / pl.col(den))
        .alias(alias)
    )


def _growth(col: str, base: pl.Expr, alias: str) -> pl.Expr:
    """``(x - base) / |base|`` — growth that keeps its meaning through zero.

    The obvious ``x/base - 1`` inverts whenever the base is negative: a company
    going from a ₹50 crore loss to a ₹10 crore loss has improved, but reads
    -1.2 and ranks among the worst in the cross-section. Dividing by ``|base|``
    alone is not enough either — that form scores a -50 to +50 turnaround as
    exactly 0.0, indistinguishable from a company that did not move at all.

    Taking the difference first and scaling by ``|base|`` fixes both: the sign
    is the direction of travel, and the magnitude is the move as a multiple of
    where the company started. For a positive base it is identical to the
    familiar ratio, so nothing changes for the ~85% of rows that are profitable.
    """
    return (
        pl.when(base.is_null() | (base.abs() < _EPS))
        .then(None)
        .otherwise((pl.col(col) - base) / base.abs())
        .alias(alias)
    )


def build_quarterly_features(figures: pl.DataFrame) -> pl.DataFrame:
    """Per (ticker, quarter) fundamental features, keyed by ``visible_from``.

    ``figures`` is the output of `scripts/fetch_xbrl_figures.py`: one row per
    company-quarter with P&L fields plus ``visible_from``.

    Year-on-year comparisons use the filing four quarters earlier, matched on
    position within each ticker's own sorted history rather than on a date
    arithmetic — a company that skipped a filing would otherwise be compared
    against the wrong quarter silently.
    """
    required = {"ticker", "period_to", "visible_from", "revenue", "net_profit"}
    missing = required - set(figures.columns)
    if missing:
        raise ValueError(f"figures is missing {sorted(missing)}")

    df = (
        figures.filter(pl.col("visible_from").is_not_null())
        .sort(["ticker", "period_to"])
        .unique(subset=["ticker", "period_to"], keep="last", maintain_order=True)
    )

    # Shares outstanding from the P&L: paid-up equity / face value. Verified
    # against INFY (20.72bn / 5 = 4.144bn shares).
    df = df.with_columns(_safe_ratio("paid_up_equity", "face_value", "f_shares"))

    lag4 = pl.col("revenue").shift(4).over("ticker")
    lag4_profit = pl.col("net_profit").shift(4).over("ticker")
    lag4_eps = pl.col("eps_basic").shift(4).over("ticker")
    lag1_rev = pl.col("revenue").shift(1).over("ticker")
    lag1_profit = pl.col("net_profit").shift(1).over("ticker")

    df = df.with_columns(
        [
            # ── levels ──
            _safe_ratio("net_profit", "revenue", "f_net_margin"),
            _safe_ratio("pbt", "revenue", "f_pbt_margin"),
            _safe_ratio("employee_cost", "revenue", "f_employee_cost_ratio"),
            _safe_ratio("finance_costs", "revenue", "f_finance_cost_ratio"),
            _safe_ratio("tax_expense", "pbt", "f_tax_rate"),
            # ── changes, year on year (same quarter last year, so no seasonality) ──
            _growth("revenue", lag4, "f_revenue_growth_yoy"),
            _growth("net_profit", lag4_profit, "f_profit_growth_yoy"),
            _growth("eps_basic", lag4_eps, "f_eps_growth_yoy"),
            # ── surprise: this quarter against the LAST one, which is the
            #    sequential move an announcement actually reveals ──
            _growth("revenue", lag1_rev, "f_revenue_surprise"),
            _growth("net_profit", lag1_profit, "f_profit_surprise"),
        ]
    )
    # Margin change needs the margin to exist first, hence a second pass.
    df = df.with_columns(
        (pl.col("f_net_margin") - pl.col("f_net_margin").shift(4).over("ticker"))
        .alias("f_net_margin_change_yoy")
    )
    return df


def attach_earnings_yield(
    features: pl.DataFrame, panel: pl.DataFrame
) -> pl.DataFrame:
    """Trailing-four-quarter earnings over market capitalisation.

    Price is taken on ``visible_from`` — the first date the figure could be
    used — so the yield is formed from an earnings number and a price an
    investor could both have had on the same morning.
    """
    ttm = (
        pl.col("net_profit").rolling_sum(window_size=4).over("ticker").alias("f_ttm_profit")
    )
    out = features.sort(["ticker", "period_to"]).with_columns(ttm)
    px = panel.select(
        pl.col("ticker"), pl.col("date").alias("visible_from"), pl.col("close")
    )
    out = out.join(px, on=["ticker", "visible_from"], how="left")
    return out.with_columns(
        pl.when(
            pl.col("close").is_null()
            | pl.col("f_shares").is_null()
            | (pl.col("close") * pl.col("f_shares") < _EPS)
        )
        .then(None)
        .otherwise(pl.col("f_ttm_profit") / (pl.col("close") * pl.col("f_shares")))
        .alias("f_earnings_yield")
    )


def as_of_panel(
    features: pl.DataFrame,
    dates: list[object],
    tickers: list[str],
    cols: tuple[str, ...],
) -> dict[str, np.ndarray]:
    """Forward-fill each feature onto the panel grid from ``visible_from``.

    A quarterly figure stays current until the next one is published, so it is
    carried forward — but only from the date it became public, never before.
    Returns ``{col: [T, N] float64}`` with NaN where nothing has been published
    yet for that ticker.
    """
    d_idx = {d: i for i, d in enumerate(dates)}
    t_idx = {t: i for i, t in enumerate(tickers)}
    out: dict[str, np.ndarray] = {
        c: np.full((len(dates), len(tickers)), np.nan, dtype=np.float64) for c in cols
    }

    rel = features.filter(
        pl.col("visible_from").is_in(list(d_idx)) & pl.col("ticker").is_in(list(t_idx))
    ).sort("visible_from")
    for row in rel.iter_rows(named=True):
        i, j = d_idx[row["visible_from"]], t_idx[row["ticker"]]
        for c in cols:
            v = row.get(c)
            if v is not None:
                out[c][i, j] = float(v)
    # Carry forward down the date axis, per ticker. The running-maximum trick
    # propagates each observation's row index downward, so a gather reproduces
    # the fill without a Python loop over ~4,000 dates x 500 tickers x 12 cols.
    # Dates before a ticker's first filing keep index 0, which is NaN there, so
    # nothing is invented ahead of the first announcement.
    rows_ix = np.arange(len(dates))[:, None]
    cols_ix = np.arange(len(tickers))[None, :]
    for c in cols:
        arr = out[c]
        seen = np.isfinite(arr)
        src = np.where(seen, rows_ix, 0)
        np.maximum.accumulate(src, axis=0, out=src)
        out[c] = arr[src, cols_ix]
    return out


#: The change features that survived the standalone IC pass 6/6 windows
#: positive at both horizons (`audit/F2_FUNDAMENTAL_IC.md` §2). Anything
#: scoring a company on its fundamentals should use these, and the list lives
#: here rather than in a script so two consumers cannot drift apart.
#:
#: The LEVEL features are deliberately absent: not one was significant, and net
#: margin flips sign at W5 and stays flipped, which is a regime, not a factor.
SIGNAL_COLS: Final[tuple[str, ...]] = (
    "f_profit_growth_yoy",
    "f_eps_growth_yoy",
    "f_net_margin_change_yoy",
)


def rank01(x: np.ndarray) -> np.ndarray:
    """Cross-sectional rank in [0, 1], NaN-preserving.

    Ranks rather than raw values because growth rates have unbounded tails: one
    company recovering from a near-zero base would otherwise dominate any
    average taken across features.
    """
    out = np.full_like(x, np.nan, dtype=np.float64)
    v = np.isfinite(x)
    if v.sum() < 2:
        return out
    order = np.argsort(np.argsort(x[v], kind="stable"), kind="stable")
    out[v] = order.astype(np.float64) / (v.sum() - 1.0)
    return out


def company_score(
    features: pl.DataFrame,
    dates: list[object],
    tickers: list[str],
    cols: tuple[str, ...] = SIGNAL_COLS,
) -> np.ndarray:
    """One fundamental score per (date, ticker): the mean of feature ranks.

    NaN where nothing has been filed for that ticker yet. An all-NaN row is the
    normal case rather than a fault -- most names have not filed on most days.
    """
    import warnings

    grids = as_of_panel(features, dates, tickers, cols)
    stacked = np.stack(
        [np.stack([rank01(grids[c][i]) for i in range(len(dates))]) for c in cols]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out: np.ndarray = np.nanmean(stacked, axis=0)
    return out
