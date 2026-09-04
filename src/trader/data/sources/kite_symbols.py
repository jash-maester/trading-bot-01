"""Project ticker → Kite tradingsymbol resolution, with an explicit rename ledger.

Why this module exists
----------------------
The project speaks yfinance (``RELIANCE.NS``); Kite speaks bare NSE tradingsymbols
(``RELIANCE``). Stripping ``.NS`` covers ~98% of the universe, and the naive
"strip and look up" that used to live in :mod:`trader.data.sources.zerodha_source`
handled the rest by logging a warning and dropping the ticker.

That failure mode is the problem. NSE tradingsymbols are *not* stable identifiers:

* Companies rename. LTIMindtree became LTM Limited on 2026-02-27 and the
  tradingsymbol went ``LTIM`` → ``LTM``. Same ISIN, same instrument token,
  same continuous price history — a different string.
* Companies demerge and the *surviving* entity is re-tickered. Tata Motors split
  into passenger and commercial vehicles in 2025; the original listing (instrument
  token 884737, history back to 2005-01-03) is now ``TMPV``, while ``TMCV`` is a
  brand-new listing from 2025-11-12.
* NSE moves illiquid or surveillance-flagged names into a restricted series and
  suffixes the tradingsymbol: ``STLTECH`` trades as ``STLTECH-BE`` (trade-to-trade).
  The suffix is a *market segment*, not a different company.
* And occasionally a name really is gone — ``MCDHOLDING`` is suspended from both
  NSE and BSE with no successor listing.

A silent drop makes all four indistinguishable, and the panel simply comes back
with 160 tickers instead of 163 while every log line says "skipping". The point of
this module is that each of those four outcomes gets a *different, named* status,
so a rename can never quietly shrink the universe: an unresolved symbol is either
listed in :data:`DELISTED_SYMBOLS` with a reason, or it is a loud failure.

The maps below are deliberately data, not logic. Adding next year's rename is a
one-line edit here — it must never require touching the resolution algorithm.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

NSE_SUFFIX: Final = ".NS"

# NSE series suffixes. A tradingsymbol carrying one of these is the *same company*
# in a restricted trading segment, so it is a legitimate resolution — but the
# suffix is recorded in the report because a move to ``-BE`` (trade-to-trade, no
# intraday netting) or ``-BZ`` (surveillance) is a real liquidity signal about
# a name we are about to size positions in.
SERIES_SUFFIXES: Final[tuple[str, ...]] = (
    "-BE",  # trade-to-trade: delivery-only, no intraday netting
    "-BZ",  # surveillance / restricted
    "-BT",  # trade-for-trade (transition)
    "-SM",  # SME platform
    "-ST",  # SME trade-to-trade
    "-IL",  # institutional
    "-GB",  # gold bonds / non-equity carve-out
)

# Project symbol (bare, no ``.NS``) → current Kite tradingsymbol.
#
# Every entry needs a date and a reason. "It didn't resolve so I guessed" is how a
# panel silently acquires another company's price history: the check that matters
# is that the instrument token and the pre-rename history are continuous, not that
# the new name looks plausible.
KITE_RENAMES: Final[dict[str, str]] = {
    # LTIMindtree Ltd → LTM Ltd, tradingsymbol changed 2026-02-27. Instrument
    # token unchanged; history is continuous across the rename.
    "LTIM": "LTM",
    # Tata Motors' 2025 demerger. The ORIGINAL listing (token 884737, daily bars
    # back to 2005-01-03) survives as the passenger-vehicle entity TMPV; the
    # commercial-vehicle entity TMCV is a separate new listing that first traded
    # 2025-11-12. Mapping TATAMOTORS → TMPV keeps 20 years of history; mapping it
    # to TMCV (whose Kite `name` is, confusingly, still "TATA MOTORS") would hand
    # back ten months of an unrelated series. The demerger day itself is a price
    # discontinuity and is handled by trader.data.corporate_actions, not here.
    "TATAMOTORS": "TMPV",
}

# Index pseudo-tickers. The project inherited yfinance's Yahoo index names
# (``^NSEI`` for NIFTY 50) and ``build_features`` still looks for exactly that
# string when computing ``beta_nifty_60d``. Kite carries indices in a separate
# ``INDICES`` segment under human-readable tradingsymbols with spaces in them, so
# they are kept apart from KITE_RENAMES: an index is not a tradable instrument and
# must never be resolvable while a caller is enumerating a *cash-equity* universe.
INDEX_ALIASES: Final[dict[str, str]] = {
    "^NSEI": "NIFTY 50",
    "^NSEBANK": "NIFTY BANK",
}

# Project symbol (bare) → why it can never resolve. Presence here converts a hard
# failure into a *documented* one; it does not make the ticker disappear from the
# report. Anything not in this map that fails to resolve is an unknown, which is
# exactly the case that must stay loud.
DELISTED_SYMBOLS: Final[dict[str, str]] = {
    "MCDHOLDING": (
        "McDowell Holdings — suspended from trading on both NSE and BSE; no "
        "successor listing exists. Verified against the NSE instrument dump 2026-09-04."
    ),
}


class ResolutionStatus(StrEnum):
    """How a project ticker was matched against the Kite instrument dump.

    A ``StrEnum`` so a status can be logged, sorted and written into a Parquet
    report column without an explicit ``.value`` at every call site.
    """

    DIRECT = "direct"
    """Bare symbol is present in the dump verbatim."""

    ALIAS = "alias"
    """Matched through :data:`KITE_RENAMES` — a documented rename."""

    SERIES = "series"
    """Matched by appending an NSE series suffix (see :data:`SERIES_SUFFIXES`)."""

    DELISTED = "delisted"
    """Known-gone, with a reason in :data:`DELISTED_SYMBOLS`. Never fetchable."""

    UNRESOLVED = "unresolved"
    """Not in the dump and not explained. This is the status that must be noisy."""


@dataclass(frozen=True)
class SymbolResolution:
    """One ticker's outcome. ``kite_symbol`` is None iff nothing is fetchable."""

    ticker: str
    """Project ticker as the universe spells it, e.g. ``LTIM.NS``."""

    kite_symbol: str | None
    """Kite tradingsymbol to fetch, or None for DELISTED / UNRESOLVED."""

    status: ResolutionStatus
    note: str

    @property
    def is_fetchable(self) -> bool:
        return self.kite_symbol is not None


@dataclass(frozen=True)
class ResolutionReport:
    """Every ticker's outcome, grouped so a caller can act per-status.

    Kept as one object rather than a bare list because the interesting question is
    never "what happened to ticker X" but "did this universe survive intact".
    """

    resolutions: tuple[SymbolResolution, ...]

    def by_status(self, status: ResolutionStatus) -> tuple[SymbolResolution, ...]:
        return tuple(r for r in self.resolutions if r.status is status)

    @property
    def fetchable(self) -> tuple[SymbolResolution, ...]:
        return tuple(r for r in self.resolutions if r.is_fetchable)

    @property
    def symbol_map(self) -> dict[str, str]:
        """Project ticker → Kite tradingsymbol, fetchable entries only."""
        return {r.ticker: r.kite_symbol for r in self.resolutions if r.kite_symbol is not None}

    @property
    def counts(self) -> dict[str, int]:
        return {s.value: len(self.by_status(s)) for s in ResolutionStatus}

    def format_lines(self) -> list[str]:
        """Human-readable summary: totals first, then every non-DIRECT ticker.

        DIRECT resolutions are counted but not listed — 160 lines of "worked as
        expected" is how the three lines that matter get scrolled past.
        """
        counts = self.counts
        lines = [
            f"Symbol resolution: {len(self.resolutions)} tickers → "
            f"{len(self.fetchable)} fetchable  "
            + "  ".join(f"{k}={v}" for k, v in counts.items() if v)
        ]
        for status in (
            ResolutionStatus.ALIAS,
            ResolutionStatus.SERIES,
            ResolutionStatus.DELISTED,
            ResolutionStatus.UNRESOLVED,
        ):
            for res in self.by_status(status):
                target = res.kite_symbol or "—"
                lines.append(f"  [{status.value:10s}] {res.ticker:18s} → {target:14s} {res.note}")
        return lines


def to_kite_symbol(ticker: str) -> str:
    """``RELIANCE.NS`` → ``RELIANCE``. Already-bare symbols pass through."""
    symbol = ticker.strip().upper()
    if symbol.endswith(NSE_SUFFIX):
        symbol = symbol[: -len(NSE_SUFFIX)]
    return symbol


def to_project_ticker(tradingsymbol: str) -> str:
    """``RELIANCE`` → ``RELIANCE.NS``. Idempotent, so it is safe to re-apply."""
    symbol = tradingsymbol.strip().upper()
    return symbol if symbol.endswith(NSE_SUFFIX) else f"{symbol}{NSE_SUFFIX}"


def resolve_symbol(ticker: str, known_symbols: Iterable[str] | Mapping[str, object]) -> (
    SymbolResolution
):
    """Resolve one project ticker against the set of live Kite tradingsymbols.

    Order matters. The delisted check runs *first* so that a name we know is gone
    can never be rescued by a coincidental series-suffix match against an unrelated
    listing — ``FOO`` being dead does not stop some ``FOO-SM`` SME listing from
    existing. After that: exact hit, documented rename, then series suffix, which
    is the order of decreasing confidence.
    """
    bare = to_kite_symbol(ticker)
    known = set(known_symbols)

    reason = DELISTED_SYMBOLS.get(bare)
    if reason is not None:
        return SymbolResolution(ticker, None, ResolutionStatus.DELISTED, reason)

    if bare in known:
        return SymbolResolution(ticker, bare, ResolutionStatus.DIRECT, "exact tradingsymbol match")

    alias = KITE_RENAMES.get(bare)
    if alias is not None:
        if alias in known:
            return SymbolResolution(
                ticker, alias, ResolutionStatus.ALIAS, f"renamed {bare} → {alias}"
            )
        # A rename map that has itself gone stale is worse than no rename map: it
        # looks authoritative. Say so instead of falling through to a suffix guess.
        return SymbolResolution(
            ticker,
            None,
            ResolutionStatus.UNRESOLVED,
            f"KITE_RENAMES maps {bare} → {alias}, but {alias} is not in the instrument "
            "dump either; the alias map is stale",
        )

    index_alias = INDEX_ALIASES.get(bare)
    if index_alias is not None and index_alias in known:
        return SymbolResolution(
            ticker,
            index_alias,
            ResolutionStatus.ALIAS,
            f"index alias {bare} → {index_alias!r} (NSE INDICES segment, not tradable)",
        )

    for suffix in SERIES_SUFFIXES:
        candidate = f"{bare}{suffix}"
        if candidate in known:
            return SymbolResolution(
                ticker,
                candidate,
                ResolutionStatus.SERIES,
                f"trades in the {suffix.lstrip('-')} series — restricted segment, "
                "expect thinner liquidity",
            )

    return SymbolResolution(
        ticker,
        None,
        ResolutionStatus.UNRESOLVED,
        "no NSE tradingsymbol matches; add it to KITE_RENAMES (renamed) or "
        "DELISTED_SYMBOLS (gone) — do not leave it silent",
    )


def resolve_universe(
    tickers: Iterable[str], known_symbols: Iterable[str] | Mapping[str, object]
) -> ResolutionReport:
    """Resolve a whole universe in one pass, preserving input order."""
    known = set(known_symbols)
    return ResolutionReport(tuple(resolve_symbol(t, known) for t in tickers))
