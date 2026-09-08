"""Splits and bonuses from NSE's corporate-actions feed.

WHY THIS EXISTS, AND WHAT IT REPLACES
-------------------------------------
`nse_bhavcopy` originally derived adjustments from the bhavcopy's own
``PREVCLOSE``, on the stated belief that NSE publishes it already adjusted for
overnight actions. **That belief was wrong.** ``PREVCLOSE`` is the raw previous
close, so ``prev_close / previous close`` is 1.0 by construction on every
ordinary day *and on every ex-date*, and the detector built on it found nothing.
Verified 2026-09-08 against three known splits:

===============  ==========  ===========  ============  ==================
ticker           date        close        prev_close    action
===============  ==========  ===========  ============  ==================
NESTLEIND        2024-01-05     2,666.40     27,116.40  1:10 split
HDFCBANK         2019-09-19     1,101.05      2,187.75  1:2 split
IRCTC            2021-10-28       913.50      4,130.15  1:5 split
===============  ==========  ===========  ============  ==================

In each case ``prev_close`` is the *pre-split* close carried forward unchanged.
The measurement that looked like proof of a precise detector — 23,693 of 23,693
ratios exactly 1.0 — was proof of an inert one. The two are indistinguishable
unless you test against a known action, which is now what
``tests/unit/test_nse_corporate_actions.py`` does.

WHAT THE FEED GIVES
-------------------
``/api/corporates-corporateActions`` returns one record per action with an
``exDate``, an ``isin``, a ``series`` and a free-text ``subject``::

    Face Value Split (Sub-Division) - From Rs 5/- Per Share To Re 1/- Per Share
    Bonus 3:1

Both forms state the ratio outright, so the factor is read rather than guessed
from a price jump — which matters, because a heuristic on price cannot tell a
1:2 split from a stock that halved.

**Dividends are deliberately ignored.** Kite's bars are adjusted for splits and
bonuses but not ordinary dividends (`configs/data/kite_v1.yaml`), so adjusting
for them here would make the two price sources disagree in a way no downstream
comparison could see.
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any, Final

import polars as pl
from loguru import logger

CA_URL: Final[str] = (
    "https://www.nseindia.com/api/corporates-corporateActions?index=equities"
    "&from_date={from_date}&to_date={to_date}"
)

CA_SCHEMA: Final[dict[str, pl.DataType]] = {
    "ex_date": pl.Date(),
    "symbol": pl.Utf8(),
    "ticker": pl.Utf8(),
    "isin": pl.Utf8(),
    "series": pl.Utf8(),
    "kind": pl.Utf8(),          # "split" | "bonus" | "consolidation"
    #: Multiply prices BEFORE ``ex_date`` by this to put them on the post-action
    #: scale. A 1:10 split gives 0.1; a 1:1 bonus gives 0.5.
    "price_factor": pl.Float64(),
    "subject": pl.Utf8(),
}

#: "From Rs 10/- Per Share To Re 1/- Per Share", and the many spellings of it.
_SPLIT_RE: Final[re.Pattern[str]] = re.compile(
    r"from\s+(?:rs\.?|re\.?|inr)?\s*([\d.]+)\s*/?-?\s*.*?"
    r"to\s+(?:rs\.?|re\.?|inr)?\s*([\d.]+)\s*/?-?",
    re.I | re.S,
)
#: "Bonus 3:1" — A new shares for every B held.
_BONUS_RE: Final[re.Pattern[str]] = re.compile(r"bonus\s*(\d+)\s*:\s*(\d+)", re.I)


class CorporateActionParseError(ValueError):
    """The payload is not an NSE corporate-actions response."""


def price_factor_for(subject: str) -> tuple[str, float] | None:
    """``(kind, price_factor)`` for a subject line, or None if not an adjustment.

    ``price_factor`` multiplies prices dated BEFORE the ex-date to put them on
    the post-action scale.

    * **Face-value split** from ``old`` to ``new``: each share becomes
      ``old/new`` shares, so the price divides by that. Factor ``new/old``.
      Rs 10 → Re 1 gives 0.1.
    * **Consolidation** (reverse split) uses the same arithmetic in the other
      direction: Re 1 → Rs 10 is ten shares becoming one, factor 10.0.
    * **Bonus A:B**: A new shares for every B held, so B shares become A+B.
      Factor ``B/(A+B)``. Bonus 1:1 gives 0.5; bonus 3:1 gives 0.25.

    Dividends and everything else return None — see the module docstring.
    """
    s = (subject or "").strip()
    if not s:
        return None
    if re.search(r"bonus", s, re.I):
        m = _BONUS_RE.search(s)
        if not m:
            return None
        a, b = int(m.group(1)), int(m.group(2))
        if a <= 0 or b <= 0:
            return None
        return "bonus", b / (a + b)
    if re.search(r"split|sub-?division|consolidat", s, re.I):
        m = _SPLIT_RE.search(s)
        if not m:
            return None
        old, new = float(m.group(1)), float(m.group(2))
        if old <= 0 or new <= 0 or old == new:
            return None
        # The same arithmetic covers both directions. A SPLIT cuts the face
        # value (Rs 10 -> Re 1, factor 0.1) and a CONSOLIDATION raises it
        # (Re 1 -> Rs 10, factor 10.0, ten shares becoming one). Rejecting the
        # second because "new >= old looks wrong" would leave a tenfold price
        # error in the history of every reverse split.
        kind = "split" if new < old else "consolidation"
        return kind, new / old
    return None


def parse_corporate_actions(
    text: str, *, name: str = "<corporate-actions>"
) -> pl.DataFrame:
    """Parse the API's JSON into :data:`CA_SCHEMA`, splits and bonuses only."""
    from trader.data.sources.nse_flows import nse_symbol_to_ticker

    try:
        rows: Any = json.loads(text)
    except json.JSONDecodeError as exc:
        raise CorporateActionParseError(f"{name}: not JSON: {exc}") from exc
    if not isinstance(rows, list):
        raise CorporateActionParseError(f"{name}: expected a list, got {type(rows).__name__}")

    out: dict[str, list[object]] = {c: [] for c in CA_SCHEMA}
    for r in rows:
        if not isinstance(r, dict):
            continue
        series = (r.get("series") or "").strip()
        if series not in ("EQ", "BE"):
            continue
        parsed = price_factor_for(r.get("subject") or "")
        if parsed is None:
            continue
        kind, factor = parsed
        raw_date = (r.get("exDate") or "").strip()
        try:
            ex = datetime.strptime(raw_date, "%d-%b-%Y").date()
        except ValueError:
            continue
        sym = (r.get("symbol") or "").strip()
        if not sym:
            continue
        out["ex_date"].append(ex)
        out["symbol"].append(sym)
        out["ticker"].append(nse_symbol_to_ticker(sym))
        out["isin"].append((r.get("isin") or "").strip() or None)
        out["series"].append(series)
        out["kind"].append(kind)
        out["price_factor"].append(factor)
        out["subject"].append((r.get("subject") or "").strip())
    return pl.DataFrame(out, schema=CA_SCHEMA).unique(
        subset=["ticker", "ex_date", "kind", "price_factor"], keep="first"
    ).sort(["ticker", "ex_date"])


def back_adjust_with_actions(
    bars: pl.DataFrame, actions: pl.DataFrame
) -> pl.DataFrame:
    """Put prices on the latest scale using a real corporate-actions feed.

    Every bar dated **strictly before** an ex-date is multiplied by that
    action's ``price_factor``; volume is divided by it so price × volume, and
    therefore turnover, is unchanged. Actions compound.

    Adjusting on the LATEST scale matches Kite's ``auto_adjust=True``
    convention. Mixing conventions between two price sources is the silent
    failure this whole module exists to prevent, and getting it wrong once
    already cost a rebuild.
    """
    need = {"date", "ticker", "close"}
    missing = need - set(bars.columns)
    if missing:
        raise ValueError(f"bars is missing {sorted(missing)}")
    if actions.is_empty():
        logger.warning(
            "no corporate actions supplied — prices are left RAW, so every "
            "split will read as a crash"
        )
        return bars

    price_cols = [
        c for c in ("open", "high", "low", "close", "last", "prev_close")
        if c in bars.columns
    ]
    # Cumulative factor for each bar: the product of every action strictly
    # after it. Done as a per-ticker join-and-scan rather than a cross join,
    # which would be 8M x 3k rows.
    acts = (
        actions.select(["ticker", "ex_date", "price_factor"])
        .sort(["ticker", "ex_date"], descending=[False, True])
        .with_columns(
            pl.col("price_factor").cum_prod().over("ticker").alias("_cum")
        )
        .sort(["ticker", "ex_date"])
    )
    # STRICTLY after. A bar dated ON the ex-date is already trading post-action,
    # so it must not be scaled — matching it to its own action would halve the
    # very first correct price. Joining on `date + 1 day` with a forward
    # strategy makes "the next ex-date at or after tomorrow" mean "the next
    # ex-date strictly after today", which is the intended relation.
    # polars cannot verify sortedness through a `by` group and says so on every
    # call; both frames are sorted immediately above.
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Sortedness of columns")
        out = _asof_join(bars, acts)
    out = out.with_columns(
        [(pl.col(c) * pl.col("_cum")).alias(c) for c in price_cols]
        + ([(pl.col("volume") / pl.col("_cum")).alias("volume")]
           if "volume" in out.columns else [])
    )
    return out.drop([c for c in ("_cum", "ex_date") if c in out.columns])


def _asof_join(bars: pl.DataFrame, acts: pl.DataFrame) -> pl.DataFrame:
    """Attach each bar the compounded factor of every action strictly after it."""
    return (
        bars.sort(["ticker", "date"])
        .with_columns((pl.col("date") + pl.duration(days=1)).alias("_probe"))
        .sort(["ticker", "_probe"])
        .join_asof(
            acts.select(["ticker", "ex_date", "_cum"]),
            left_on="_probe",
            right_on="ex_date",
            by="ticker",
            strategy="forward",
        )
        .with_columns(pl.col("_cum").fill_null(1.0).alias("_cum"))
        .drop("_probe")
    )
