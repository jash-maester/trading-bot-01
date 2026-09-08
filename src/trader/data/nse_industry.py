"""NSE's own industry classification, for a universe wider than SECTOR_MAP.

`trader.data.universe.SECTOR_MAP` covers the 504 hand-curated names, and
`sector_id_of` returns **0** for anything outside it. That zero is the phantom
sector `CLAUDE.md` records: an id absent from ``SECTOR_IDS`` that became a ninth
graph node nobody intended. It is harmless today only because ``all_tickers()``
returns exactly the names that have a sector.

A point-in-time universe breaks that guarantee. `scripts/fetch_bhavcopy.py`
yields ~2,200 EQ/BE securities and the liquidity rule admits roughly 600–750 of
them, most outside SECTOR_MAP — so building a panel on it with ``sector_id_of``
would put hundreds of names into the phantom bucket at once.

NSE publishes the classification. ``ind_niftytotalmarket_list.csv`` carries 755
constituents across 22 industries and is a superset of the NIFTY 500, smallcap
250 and microcap 250 lists (verified 2026-09-08: their union is exactly the 755).

WHY A SEPARATE ID TABLE RATHER THAN A MAPPING ONTO THE EXISTING 14. Collapsing
22 into 14 requires arbitrary joins — Construction into Construction Materials,
Realty and Textiles into nothing at all — and each one destroys information for
no benefit. The bhav panel is a separate `data=` config writing to a separate
panels root, so it can carry its own ids without touching the Kite pipeline or
any test that depends on it.

**Unknown is 99, not 0.** A name NSE does not classify — typically one that
delisted before the current constituent list was drawn — gets an explicit,
documented bucket. Zero is reserved as invalid precisely so that an unmapped
name is a loud fact rather than a silent default.
"""
from __future__ import annotations

from typing import Final

import polars as pl

#: NSE's macro-economic sector classification, as published in the index
#: constituent files. Ids are assigned alphabetically and are stable: appending
#: a new industry must take the next free number, never renumber these.
NSE_INDUSTRY_IDS: Final[dict[str, int]] = {
    "Automobile and Auto Components": 1,
    "Capital Goods": 2,
    "Chemicals": 3,
    "Construction": 4,
    "Construction Materials": 5,
    "Consumer Durables": 6,
    "Consumer Services": 7,
    "Diversified": 8,
    "Fast Moving Consumer Goods": 9,
    "Financial Services": 10,
    "Forest Materials": 11,
    "Healthcare": 12,
    "Information Technology": 13,
    "Media Entertainment & Publication": 14,
    "Metals & Mining": 15,
    "Oil Gas & Consumable Fuels": 16,
    "Power": 17,
    "Realty": 18,
    "Services": 19,
    "Telecommunication": 20,
    "Textiles": 21,
    "Utilities": 22,
}

#: An explicit bucket for a name NSE does not classify — usually one that
#: delisted before the current constituent list was drawn. Deliberately NOT 0:
#: zero is what `sector_id_of` returns by accident, and the whole point here is
#: that "unclassified" should be a stated fact rather than a default.
UNKNOWN_INDUSTRY_ID: Final[int] = 99

INDUSTRY_URL: Final[str] = (
    "https://nsearchives.nseindia.com/content/indices/ind_niftytotalmarket_list.csv"
)

INDUSTRY_SCHEMA: Final[dict[str, pl.DataType]] = {
    "symbol": pl.Utf8(),
    "ticker": pl.Utf8(),
    "isin": pl.Utf8(),
    "industry": pl.Utf8(),
    "industry_id": pl.Int64(),
}


class IndustryParseError(ValueError):
    """The payload is not an NSE index constituent list."""


def parse_industry_list(text: str, *, name: str = "<industries>") -> pl.DataFrame:
    """Parse an ``ind_*list.csv`` into :data:`INDUSTRY_SCHEMA`.

    An industry NSE publishes that is absent from :data:`NSE_INDUSTRY_IDS`
    raises rather than falling through to unknown. A new NSE industry is a real
    change to the taxonomy and should be a deliberate edit here, not a silent
    reclassification of every name in it.
    """
    import csv
    import io

    from trader.data.sources.nse_flows import nse_symbol_to_ticker

    head = text.lstrip()[:200]
    if "Symbol" not in head or "Industry" not in head:
        raise IndustryParseError(
            f"{name}: no Symbol/Industry header in the first 200 chars; "
            f"got {head[:80]!r}"
        )
    rows = [
        {(k or "").strip(): (v or "").strip() for k, v in r.items()}
        for r in csv.DictReader(io.StringIO(text))
    ]
    out: dict[str, list[object]] = {c: [] for c in INDUSTRY_SCHEMA}
    unseen: set[str] = set()
    for r in rows:
        sym, ind = r.get("Symbol", ""), r.get("Industry", "")
        if not sym:
            continue
        if ind and ind not in NSE_INDUSTRY_IDS:
            unseen.add(ind)
            continue
        out["symbol"].append(sym)
        out["ticker"].append(nse_symbol_to_ticker(sym))
        out["isin"].append(r.get("ISIN Code") or None)
        out["industry"].append(ind or None)
        out["industry_id"].append(
            NSE_INDUSTRY_IDS.get(ind, UNKNOWN_INDUSTRY_ID) if ind else UNKNOWN_INDUSTRY_ID
        )
    if unseen:
        raise IndustryParseError(
            f"{name}: NSE published industry/industries {sorted(unseen)} that are "
            "not in NSE_INDUSTRY_IDS. Add them there with the next free id — "
            "falling through to unknown would silently reclassify every name in "
            "them."
        )
    if not out["symbol"]:
        raise IndustryParseError(f"{name}: parsed zero constituents")
    return pl.DataFrame(out, schema=INDUSTRY_SCHEMA)


def industry_ids(
    frame: pl.DataFrame, tickers: list[str]
) -> dict[str, int]:
    """``{ticker: industry_id}`` for every ticker asked for.

    Names absent from ``frame`` map to :data:`UNKNOWN_INDUSTRY_ID`. Every ticker
    asked for appears in the result, so a caller building a panel cannot end up
    with a name that has no id at all — which is how a `None` becomes a 0
    becomes a phantom sector node.
    """
    if "ticker" not in frame.columns or "industry_id" not in frame.columns:
        raise ValueError("frame must carry `ticker` and `industry_id`")
    known = dict(
        zip(
            frame["ticker"].to_list(),
            [int(v) for v in frame["industry_id"].to_list()],
            strict=True,
        )
    )
    return {t: known.get(t, UNKNOWN_INDUSTRY_ID) for t in tickers}
