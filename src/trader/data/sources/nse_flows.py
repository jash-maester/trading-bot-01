"""NSE public-file sources for the R8 opt-in feature group: delivery, flows, deals.

Three free, structured, timestamped NSE publications, none of which is derived
from ``adj_close`` or ``volume`` (`10_architecture_revamp.md` §4 ranks richer
features as the lever most likely to move the cross-sectional signal):

1. **Delivery position** — per symbol per day, deliverable quantity over traded
   quantity. High delivery means positions taken, not intraday churn.
2. **FII/DII cash-market flows** — one row per *day*, market-wide. Becomes a
   market-level feature broadcast to every stock, like the regime vector.
3. **Bulk and block deals** — per symbol per day, large disclosed trades.
   Sparse: most (date, ticker) cells legitimately have no deal.

Nothing here changes ``FEATURE_COLS``, the default panel, or any existing
feature. R4 is runnable identically with and without this group.

--------------------------------------------------------------------------------
URL patterns and what was verified
--------------------------------------------------------------------------------

Every status code below was observed in this session (2026-09-05) from the Mac
mini, with the session warm-up described under "Session" and ~1 req/s
throttling. Anything not personally observed is marked **UNVERIFIED**.

**Delivery, primary — MTO** (``DeliverySource``, backend ``mto``)::

    https://nsearchives.nseindia.com/archives/equities/mto/MTO_<DDMMYYYY>.DAT

    MTO_04092026.DAT  200  2,040,488 B header, 3,375 records   (current layout)
    MTO_02012015.DAT  200  56,719 B                            (settlement layout)
    MTO_01042008.DAT  200  43,688 B                            (settlement layout)
    MTO_02012007.DAT  200  34,641 B
    MTO_03012006.DAT  200  28,738 B
    MTO_01042005.DAT  200  27,058 B   <- earliest date verified available

    Fields per record (7 comma-separated, though the header line names only 6 —
    the series column is unnamed in NSE's own header):
        record type (always 20), serial no, symbol, series, quantity traded,
        deliverable quantity (gross, client level), % deliverable to traded.

**Delivery, fallback — security-wise bhavcopy** (backend ``bhavdata``)::

    https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_<DDMMYYYY>.csv

    04092026  200  397,073 B   (8 of ~3,300 rows are in the fixture)
    03012022  200  251,873 B
    01042021  200  224,405 B
    01072020  200  215,792 B   <- earliest date verified available
    02012019  UNVERIFIED — the request timed out three times; not shown to 404.

    Delivery fields ``DELIV_QTY`` / ``DELIV_PER`` carry a literal ``-`` for
    series where NSE does not publish delivery (BE / trade-to-trade). That is a
    missing value and is parsed to null, never to zero.

**FII/DII flows** (``FlowsSource``)::

    https://www.nseindia.com/api/fiidiiTradeReact          200, 218 B

    Returns the LATEST published day only — a two-element JSON array, one object
    per category (``DII``, ``FII/FPI``), each with ``date`` (``04-Sep-2026``),
    ``buyValue``, ``sellValue``, ``netValue``, all ₹ crore as strings. There is
    no dated variant of this endpoint that this session could reach, so history
    comes from a local append-only CSV ledger in ``FlowsSource.BACKFILL_COLUMNS``
    (schema documented on the class); the daily API call appends to it.
    A dated historical flows endpoint is **UNVERIFIED** — none was found.

**Bulk and block deals** (``DealsSource``)::

    https://nsearchives.nseindia.com/content/equities/bulk.csv    200, 14,691 B
    https://nsearchives.nseindia.com/content/equities/block.csv   200,    278 B

    Both files hold the LATEST trading day only — verified: every one of the 156
    rows in bulk.csv and both rows in block.csv carried ``04-SEP-2026``. History
    is therefore accumulated by snapshotting daily into the per-date cache under
    ``data/raw/nse/deals/``.

    The dated JSON endpoints
    ``https://www.nseindia.com/api/historical/bulk-deals?from=&to=`` and
    ``.../block-deals`` returned **503 Service Unavailable** in this session and
    are **UNVERIFIED**. ``DealsSource`` does not call them.

**Session.** ``https://www.nseindia.com`` answered **403** to this client and
still set the cookie NSE's archive hosts want (``AKA_A2``), after which every
archive URL above returned 200. So the warm-up is required and its status code
must NOT be treated as failure. Non-trading days are expected to 404; a 404 is
cached as a ``.missing`` marker so a re-run costs nothing.

--------------------------------------------------------------------------------
Output conventions
--------------------------------------------------------------------------------

* Symbols in NSE files are bare (``RELIANCE``). Every ``fetch`` maps them to the
  project's ``.NS`` convention via :func:`nse_symbol_to_ticker`, which also
  applies ``KITE_RENAMES`` — a 2015 MTO file says ``LTIM`` and ``TATAMOTORS``
  where the current universe says ``LTM.NS`` and ``TMPV.NS``, and a symbol-keyed
  join without that step silently drops history (``CLAUDE.md``, "Symbol
  renames").
* Caching is per file under ``data/raw/nse/<source>/``; a re-run re-parses from
  disk and issues no request.
* Missing means null. Nothing in this module ever substitutes 0 for an absent
  observation (B7: a fabricated constant hid for months).
"""
from __future__ import annotations

import csv
import io
import json
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any, Final

import polars as pl
from loguru import logger

from trader.data.sources.kite_symbols import KITE_RENAMES, SERIES_SUFFIXES

# ── Constants ────────────────────────────────────────────────────────────────

NSE_HOME: Final[str] = "https://www.nseindia.com"
NSE_ARCHIVES: Final[str] = "https://nsearchives.nseindia.com"

MTO_URL: Final[str] = NSE_ARCHIVES + "/archives/equities/mto/MTO_{ddmmyyyy}.DAT"
BHAVDATA_URL: Final[str] = (
    NSE_ARCHIVES + "/products/content/sec_bhavdata_full_{ddmmyyyy}.csv"
)
BULK_URL: Final[str] = NSE_ARCHIVES + "/content/equities/bulk.csv"
BLOCK_URL: Final[str] = NSE_ARCHIVES + "/content/equities/block.csv"
FIIDII_URL: Final[str] = NSE_HOME + "/api/fiidiiTradeReact"

DEFAULT_CACHE_ROOT: Final[str] = "data/raw/nse"

# Earliest date each source was verified reachable in this session. Requesting
# earlier than this is allowed; it is simply not something anyone checked.
MTO_VERIFIED_FROM: Final[date] = date(2005, 4, 1)
BHAVDATA_VERIFIED_FROM: Final[date] = date(2020, 7, 1)

# Cash-market series worth keeping. EQ is the normal rolling-settlement series;
# BE is trade-to-trade (STLTECH trades as STLTECH-BE). Everything else NSE puts
# in an MTO file — N1/N3 odd-lot and non-equity series — is not the instrument
# the panel prices, and folding it in would double-count a symbol on some days.
CASH_SERIES: Final[tuple[str, ...]] = ("EQ", "BE")

# Imported, NOT re-declared. The local copy this replaced had already drifted:
# it was missing "-BT", so a -BT tradingsymbol did not have its suffix stripped
# and silently failed to join the universe. A tuple is immutable, so importing
# it cannot mutate kite_symbols' list -- the reason the copy was made in the
# first place was not a real hazard.
_SERIES_SUFFIXES: Final[tuple[str, ...]] = SERIES_SUFFIXES

_BROWSER_HEADERS: Final[dict[str, str]] = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "*/*",
    "Referer": NSE_HOME + "/",
}


# ── Errors ───────────────────────────────────────────────────────────────────


class NSEFetchError(RuntimeError):
    """A request to NSE failed after every retry."""


class NSENotFound(NSEFetchError):
    """NSE returned 404 — usually a non-trading day, not a fault."""


class NotACSVError(ValueError):
    """NSE served something other than the CSV this URL is supposed to return.

    Distinct from a parse failure on a real CSV: this one is safe to SKIP for a
    day and carry on, because it says nothing about the other 1,612 days.
    """


class MTOLayoutError(ValueError):
    """An MTO file did not match a layout this parser knows.

    Raised loudly and by name rather than guessed at: an MTO record whose column
    order shifted would silently redefine ``delivery_pct`` for every row after
    the change.
    """


# ── HTTP client ──────────────────────────────────────────────────────────────


class NSEClient:
    """Throttled, retrying, cookie-warmed HTTP client for NSE's public files.

    ``requests`` is imported lazily so that importing this module (and therefore
    :mod:`trader.data.features_ext`) never needs the network stack. Tests are
    offline and construct no client.
    """

    def __init__(
        self,
        *,
        min_interval_s: float = 1.0,
        max_retries: int = 3,
        timeout_s: float = 20.0,
        session: Any | None = None,
    ) -> None:
        self.min_interval_s = min_interval_s
        self.max_retries = max_retries
        self.timeout_s = timeout_s
        self._session: Any | None = session
        self._warmed = session is not None
        self._last_request_at = 0.0

    # -- session -------------------------------------------------------------

    def _ensure_session(self) -> Any:
        if self._session is None:
            import requests

            session = requests.Session()
            session.headers.update(_BROWSER_HEADERS)
            self._session = session
        if not self._warmed:
            self._warm_up()
        return self._session

    def _warm_up(self) -> None:
        """Hit the homepage so the archive hosts accept the session cookie.

        The homepage answers 403 to this client and sets the cookie anyway
        (observed 2026-09-05: ``home 403 368 {'AKA_A2': 'A'}``, after which every
        archive URL returned 200), so the status code is deliberately ignored.
        """
        self._warmed = True
        try:
            self._sleep_for_throttle()
            self._session.get(NSE_HOME, timeout=self.timeout_s)  # type: ignore[union-attr]
        except Exception as exc:  # noqa: BLE001 — warm-up is best-effort
            logger.warning(f"NSE warm-up request failed ({type(exc).__name__}): {exc}")

    def _sleep_for_throttle(self) -> None:
        wait = self.min_interval_s - (time.monotonic() - self._last_request_at)
        if wait > 0:
            time.sleep(wait)
        self._last_request_at = time.monotonic()

    # -- fetch ---------------------------------------------------------------

    def get_text(self, url: str) -> str:
        """GET ``url`` and return its body, retrying with exponential backoff.

        Raises:
            NSENotFound: NSE answered 404 (typically a non-trading day).
            NSEFetchError: every attempt failed, or a non-404 error status.
        """
        session = self._ensure_session()
        last_error: str = "no attempt made"
        for attempt in range(self.max_retries):
            self._sleep_for_throttle()
            try:
                response = session.get(url, timeout=self.timeout_s)
            except Exception as exc:  # noqa: BLE001 — any transport error retries
                last_error = f"{type(exc).__name__}: {exc}"
                logger.debug(f"NSE GET {url} attempt {attempt + 1} failed: {last_error}")
                time.sleep(2.0 * (attempt + 1))
                continue
            status = int(response.status_code)
            if status == 404:
                raise NSENotFound(f"404 for {url}")
            if status == 200:
                return str(response.text)
            last_error = f"HTTP {status}"
            logger.debug(f"NSE GET {url} attempt {attempt + 1}: {last_error}")
            time.sleep(2.0 * (attempt + 1))
        raise NSEFetchError(f"GET {url} failed after {self.max_retries} attempts: {last_error}")


# ── Symbol mapping ───────────────────────────────────────────────────────────


def nse_symbol_to_ticker(symbol: str) -> str:
    """Map a bare NSE symbol to the project's ``.NS`` ticker convention.

    Strips a series suffix (``STLTECH-BE`` → ``STLTECH``) and applies the
    documented rename ledger (``LTIM`` → ``LTM``, ``TATAMOTORS`` → ``TMPV``), so
    that a 2015 file joins onto the current universe instead of dropping.
    """
    bare = symbol.strip().upper()
    for suffix in _SERIES_SUFFIXES:
        if bare.endswith(suffix):
            bare = bare[: -len(suffix)]
            break
    return f"{KITE_RENAMES.get(bare, bare)}.NS"


def _ddmmyyyy(day: date) -> str:
    return day.strftime("%d%m%Y")


def _trading_days(start: date, end: date) -> Iterator[date]:
    """Weekdays in ``[start, end]``. NSE holidays simply 404 and are cached."""
    day = start
    while day <= end:
        if day.weekday() < 5:
            yield day
        day += timedelta(days=1)


# ── On-disk cache ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class _FileCache:
    """One directory of raw NSE files, plus ``.missing`` markers for 404s."""

    root: Path

    def path(self, name: str) -> Path:
        return self.root / name

    def read(self, name: str) -> str | None:
        path = self.path(name)
        if path.exists():
            return path.read_text(encoding="utf-8", errors="replace")
        return None

    def is_missing(self, name: str) -> bool:
        return self.path(name + ".missing").exists()

    MANIFEST_NAME: Final[str] = "fetch_manifest.jsonl"

    def write(self, name: str, text: str, *, url: str | None = None) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.path(name).write_text(text, encoding="utf-8")
        self.record(name, text, url=url)

    def record(self, name: str, text: str, *, url: str | None = None) -> None:
        """Append a provenance line for a cached file.

        Every live-network measurement this module produced previously left no
        durable trace: the HTTP statuses, byte counts, row counts and the
        "earliest date verified available" claim were all unreproducible from
        the repo the moment the session ended, because nothing was committed
        and `data/raw/nse/` did not exist.  CLAUDE.md's verification standard
        requires a durable artefact behind a claim, so each write appends
        ``{name, url, bytes, sha256, fetched_at}`` to ``fetch_manifest.jsonl``
        next to the cached files.  The manifest is append-only and cheap, and it
        is what a later reader cites instead of a number in a chat log.
        """
        import hashlib
        from datetime import UTC, datetime

        payload = text.encode("utf-8")
        line = json.dumps(
            {
                "name": name,
                "url": url,
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "fetched_at": datetime.now(UTC).isoformat(),
            },
            sort_keys=True,
        )
        with self.path(self.MANIFEST_NAME).open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    def manifest(self) -> list[dict[str, Any]]:
        """Every recorded fetch, oldest first; empty when nothing was fetched."""
        path = self.path(self.MANIFEST_NAME)
        if not path.exists():
            return []
        out: list[dict[str, Any]] = []
        for raw in path.read_text(encoding="utf-8").splitlines():
            if raw.strip():
                out.append(json.loads(raw))
        return out

    def mark_missing(self, name: str) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.path(name + ".missing").write_text("", encoding="utf-8")


class _CachedSource:
    """Shared cache + client plumbing for the three sources."""

    subdir: str = ""

    def __init__(
        self,
        *,
        cache_root: str | Path = DEFAULT_CACHE_ROOT,
        client: NSEClient | None = None,
        offline: bool = False,
    ) -> None:
        self.cache = _FileCache(Path(cache_root) / self.subdir)
        self.offline = offline
        self._client = client

    @property
    def client(self) -> NSEClient:
        if self._client is None:
            self._client = NSEClient()
        return self._client

    def _load(self, name: str, url: str) -> str | None:
        """Return a cached file, else fetch and cache it. None means absent."""
        cached = self.cache.read(name)
        if cached is not None:
            return cached
        if self.cache.is_missing(name):
            return None
        if self.offline:
            logger.debug(f"offline: {name} not in {self.cache.root}")
            return None
        try:
            text = self.client.get_text(url)
        except NSENotFound:
            self.cache.mark_missing(name)
            return None
        except NSEFetchError as exc:
            logger.warning(f"fetch failed, not cached: {exc}")
            return None
        self.cache.write(name, text)
        return text


# ── 1. Delivery ──────────────────────────────────────────────────────────────


class MTOLayout(StrEnum):
    """Header layouts observed in real MTO files."""

    SETTLEMENT = "mto_settlement_header"
    """Line 3 carries ``Settlement No`` and ``Settlement Date``. Seen 2005–2015."""

    NO_SETTLEMENT = "mto_no_settlement_header"
    """Line 3 carries only trade date and settlement type. Seen 2026."""


# The column-header line of every MTO file this parser accepts, normalised by
# `_normalise_header`. NSE names six columns and then writes seven fields per
# record — the series column has no name. Both real layouts share this line.
_MTO_COLUMN_HEADER: Final[str] = (
    "recordtype,srno,nameofsecurity,quantitytraded,"
    "deliverablequantity(grossacrossclientlevel),"
    "%ofdeliverablequantitytotradedquantity"
)
_MTO_RECORD_FIELDS: Final[int] = 7

DELIVERY_SCHEMA: Final[dict[str, pl.DataType]] = {
    "date": pl.Date(),
    "ticker": pl.Utf8(),
    "series": pl.Utf8(),
    "traded_qty": pl.Int64(),
    "deliverable_qty": pl.Int64(),
    "delivery_pct": pl.Float64(),
    # Turnover fields, kept from 2026-09-07. `sec_bhavdata_full` has carried
    # them all along and this parser was discarding them.
    #
    # They matter because Kite's historical bars are OHLCV with no turnover, so
    # nothing else in this repo knows the rupee VALUE traded. `avg_price` is
    # NSE's own VWAP, and turnover/qty reproduces it — which is genuinely
    # absent from OHLC: a stock that closed at its high after trading near its
    # low all day is a different animal from one that traded near its high
    # throughout, and only these columns separate them.
    #
    # Null, never zero, on rows where NSE publishes a literal "-". The MTO
    # backend cannot supply them at all and leaves them null, which is why a
    # consumer must check `source` before assuming coverage.
    "turnover": pl.Float64(),        # rupees, converted from TURNOVER_LACS
    "avg_price": pl.Float64(),       # NSE's AVG_PRICE, i.e. VWAP
    "n_trades": pl.Int64(),
    "source": pl.Utf8(),
}

#: NSE reports turnover in lakhs (10^5 rupees). Stored in rupees so a consumer
#: never has to remember the unit.
_LACS_TO_RUPEES: Final[float] = 1e5


def _normalise_header(line: str) -> str:
    return line.replace(" ", "").strip().lower()


def _parse_angle_fields(line: str) -> dict[str, str]:
    """Parse ``Trade Date <02-JAN-2015>,Settlement Type <N>`` into a dict."""
    out: dict[str, str] = {}
    for part in line.split(","):
        if "<" not in part or not part.rstrip().endswith(">"):
            continue
        key, _, rest = part.partition("<")
        out[_normalise_header(key)] = rest.rstrip().rstrip(">").strip()
    return out


def detect_mto_layout(text: str, *, name: str = "<mto>") -> MTOLayout:
    """Classify an MTO file's header, or raise :class:`MTOLayoutError`.

    The two real layouts differ only in line 3: pre-2020s files carry settlement
    number and settlement date, current files do not. A file whose *column
    header* or record arity differs from either is refused by name rather than
    guessed at.
    """
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) < 4:
        raise MTOLayoutError(f"{name}: only {len(lines)} non-empty lines; not an MTO file")
    header_line = _normalise_header(lines[3])
    if header_line != _MTO_COLUMN_HEADER:
        raise MTOLayoutError(
            f"{name}: unknown MTO column layout. Expected the header "
            f"{_MTO_COLUMN_HEADER!r} but found {header_line!r}. "
            "Layouts known to this parser: "
            f"{MTOLayout.SETTLEMENT.value}, {MTOLayout.NO_SETTLEMENT.value}. "
            "Refusing to guess which column is which — add the new layout "
            "explicitly in trader.data.sources.nse_flows."
        )
    angle = _parse_angle_fields(lines[2])
    if "tradedate" not in angle:
        raise MTOLayoutError(
            f"{name}: line 3 has no 'Trade Date <...>' field: {lines[2]!r}"
        )
    if "settlementno" in angle or "settlementdate" in angle:
        return MTOLayout.SETTLEMENT
    return MTOLayout.NO_SETTLEMENT


def parse_mto(text: str, *, name: str = "<mto>") -> pl.DataFrame:
    """Parse an MTO ``.DAT`` file into :data:`DELIVERY_SCHEMA` (all series).

    ``delivery_pct`` is recomputed as ``deliverable_qty / traded_qty`` (a
    fraction in ``[0, 1]``, not NSE's percentage) and cross-checked against the
    published percentage; a disagreement past 1e-3 is logged. A zero traded
    quantity yields a null, never a zero.

    Raises:
        MTOLayoutError: the file does not match a known layout.
    """
    layout = detect_mto_layout(text, name=name)
    lines = [ln for ln in text.splitlines() if ln.strip()]
    trade_date_raw = _parse_angle_fields(lines[2])["tradedate"]
    trade_date = datetime.strptime(trade_date_raw, "%d-%b-%Y").date()

    dates: list[date] = []
    tickers: list[str] = []
    series: list[str] = []
    traded: list[int] = []
    delivered: list[int] = []
    pct: list[float | None] = []

    for raw in lines[4:]:
        fields = [f.strip() for f in raw.split(",")]
        if fields[0] != "20":
            continue
        if len(fields) != _MTO_RECORD_FIELDS:
            raise MTOLayoutError(
                f"{name}: layout {layout.value} promises {_MTO_RECORD_FIELDS} fields "
                f"per record but this record has {len(fields)}: {raw!r}. "
                "Refusing to guess the column order."
            )
        _, _, symbol, ser, qty_s, deliv_s, pct_s = fields
        qty = int(qty_s)
        deliv = int(deliv_s)
        if qty > 0:
            computed = deliv / qty
            published = float(pct_s) / 100.0
            if abs(computed - published) > 1e-3:
                logger.warning(
                    f"{name}: {symbol} {ser} delivery pct mismatch — published "
                    f"{published:.6f} vs computed {computed:.6f}"
                )
            pct.append(computed)
        else:
            # Traded nothing: the ratio is undefined. Null, never zero.
            pct.append(None)
        dates.append(trade_date)
        tickers.append(nse_symbol_to_ticker(symbol))
        series.append(ser)
        traded.append(qty)
        delivered.append(deliv)

    return pl.DataFrame(
        {
            "date": dates,
            "ticker": tickers,
            "series": series,
            "traded_qty": traded,
            "deliverable_qty": delivered,
            "delivery_pct": pct,
            # The MTO file carries quantities only — no turnover, no VWAP, no
            # trade count. Null rather than 0, so a consumer can tell "this
            # backend cannot know" from "there was no trading"; `source` says
            # which backend produced the row.
            "turnover": [None] * len(dates),
            "avg_price": [None] * len(dates),
            "n_trades": [None] * len(dates),
            "source": ["mto"] * len(dates),
        },
        schema=DELIVERY_SCHEMA,
    )


def parse_sec_bhavdata(text: str, *, name: str = "<bhavdata>") -> pl.DataFrame:
    """Parse a ``sec_bhavdata_full_*.csv`` into :data:`DELIVERY_SCHEMA`.

    The fallback delivery backend. Header names and values both carry leading
    spaces in NSE's file, and ``DELIV_QTY`` / ``DELIV_PER`` are a literal ``-``
    on series where NSE publishes no delivery — parsed to null, never zero.
    """
    # Reject a payload that is not the CSV we asked for BEFORE handing it to
    # the csv module, which otherwise produces a baffling error a long way from
    # the cause. Measured 2026-09-07: NSE served a ZIP/XLSX body for
    # 2022-08-08 (magic PK\x03\x04, containing [Content_Types].xml) at the
    # normal .csv URL. One such day in 551 killed a 1,613-day backfill.
    head = text.lstrip()[:400]
    if text.startswith("PK\x03\x04") or "[Content_Types].xml" in head:
        raise NotACSVError(
            f"{name}: NSE served a ZIP/XLSX body at the .csv URL, not a CSV"
        )
    if "SYMBOL" not in head:
        raise NotACSVError(
            f"{name}: body does not carry a SYMBOL header in its first 400 "
            f"chars; got {head[:80]!r}"
        )

    reader = csv.DictReader(io.StringIO(text))
    rows = [{(k or "").strip(): (v or "").strip() for k, v in row.items()} for row in reader]

    dates: list[date] = []
    tickers: list[str] = []
    series: list[str] = []
    traded: list[int | None] = []
    delivered: list[int | None] = []
    pct: list[float | None] = []
    turnover: list[float | None] = []
    avg_price: list[float | None] = []
    n_trades: list[int | None] = []

    for row in rows:
        missing = {"SYMBOL", "SERIES", "DATE1", "TTL_TRD_QNTY", "DELIV_QTY"} - set(row)
        if missing:
            raise ValueError(f"{name}: bhavdata row missing columns {sorted(missing)}")
        qty = _int_or_none(row["TTL_TRD_QNTY"])
        deliv = _int_or_none(row["DELIV_QTY"])
        dates.append(datetime.strptime(row["DATE1"], "%d-%b-%Y").date())
        tickers.append(nse_symbol_to_ticker(row["SYMBOL"]))
        series.append(row["SERIES"])
        traded.append(qty)
        delivered.append(deliv)
        pct.append(deliv / qty if (qty is not None and deliv is not None and qty > 0) else None)
        # Turnover columns are NOT in the required set above: they are absent
        # from older layouts, and a file that predates them should still yield
        # delivery data rather than raising.
        lacs = _float_or_none(row.get("TURNOVER_LACS", ""))
        turnover.append(lacs * _LACS_TO_RUPEES if lacs is not None else None)
        avg_price.append(_float_or_none(row.get("AVG_PRICE", "")))
        n_trades.append(_int_or_none(row.get("NO_OF_TRADES", "")))

    return pl.DataFrame(
        {
            "date": dates,
            "ticker": tickers,
            "series": series,
            "traded_qty": traded,
            "deliverable_qty": delivered,
            "delivery_pct": pct,
            "turnover": turnover,
            "avg_price": avg_price,
            "n_trades": n_trades,
            "source": ["bhavdata"] * len(dates),
        },
        schema=DELIVERY_SCHEMA,
    )


def _int_or_none(value: str) -> int | None:
    text = value.strip().replace(",", "")
    if text in {"", "-", "NA", "N/A"}:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


class DeliverySource(_CachedSource):
    """Daily security-wise delivery position.

    Output of :meth:`fetch` — one row per (date, ticker), cash series only:

    ==================  =========  ==================================================
    column              dtype      meaning
    ==================  =========  ==================================================
    ``date``            Date       NSE trade date, read from the file, not the URL
    ``ticker``          str        project ``.NS`` ticker, renames applied
    ``series``          str        ``EQ`` or ``BE``
    ``traded_qty``      Int64      total quantity traded
    ``deliverable_qty`` Int64      quantity marked for delivery (gross, client level)
    ``delivery_pct``    Float64    ``deliverable_qty / traded_qty`` in [0, 1], or null
    ``source``          str        ``mto`` or ``bhavdata``
    ==================  =========  ==================================================

    A symbol quoted in several series on one day (``SBIN`` EQ / N1 / N3) is
    reduced to one row, preferring ``EQ``.
    """

    subdir = "mto"

    def __init__(
        self,
        *,
        cache_root: str | Path = DEFAULT_CACHE_ROOT,
        client: NSEClient | None = None,
        offline: bool = False,
        backend: str = "mto",
    ) -> None:
        super().__init__(cache_root=cache_root, client=client, offline=offline)
        if backend not in {"mto", "bhavdata"}:
            raise ValueError(f"backend must be 'mto' or 'bhavdata', got {backend!r}")
        self.backend = backend
        self.subdir = backend
        self.cache = _FileCache(Path(cache_root) / backend)

    def fetch(self, start: date, end: date) -> pl.DataFrame:
        frames: list[pl.DataFrame] = []
        skipped: list[date] = []
        for day in _trading_days(start, end):
            try:
                frame = self.fetch_day(day)
            except (NotACSVError, MTOLayoutError) as exc:
                # One malformed day must not kill a multi-hour backfill. NSE
                # served a ZIP at the .csv URL for 2022-08-08 and aborted a
                # 1,613-day run 554 days in. Skip it, name it, keep going --
                # and report the count at the end so a silent hole in the
                # history is impossible.
                logger.warning(f"{day}: skipped, {exc}")
                skipped.append(day)
                continue
            if frame is not None:
                frames.append(frame)
        if skipped:
            logger.warning(
                f"{len(skipped)} day(s) skipped as unparseable: "
                + ", ".join(str(d) for d in skipped[:10])
                + (" ..." if len(skipped) > 10 else "")
            )
        if not frames:
            return pl.DataFrame(schema=DELIVERY_SCHEMA)
        return self._dedupe(pl.concat(frames)).sort(["date", "ticker"])

    def fetch_day(self, day: date) -> pl.DataFrame | None:
        """Return one day's delivery rows, or None if NSE published no file."""
        stamp = _ddmmyyyy(day)
        if self.backend == "mto":
            name, url = f"MTO_{stamp}.DAT", MTO_URL.format(ddmmyyyy=stamp)
            parse = parse_mto
        else:
            name = f"sec_bhavdata_full_{stamp}.csv"
            url = BHAVDATA_URL.format(ddmmyyyy=stamp)
            parse = parse_sec_bhavdata
        text = self._load(name, url)
        if text is None:
            return None
        return parse(text, name=name)

    @staticmethod
    def _dedupe(frame: pl.DataFrame) -> pl.DataFrame:
        """Keep cash series only, one row per (date, ticker), EQ winning."""
        rank = pl.when(pl.col("series") == "EQ").then(0).otherwise(1)
        return (
            frame.filter(pl.col("series").is_in(list(CASH_SERIES)))
            .with_columns(rank.alias("_rank"))
            .sort(["date", "ticker", "_rank"])
            .unique(subset=["date", "ticker"], keep="first", maintain_order=True)
            .drop("_rank")
        )


# ── 2. FII / DII flows ───────────────────────────────────────────────────────


FLOWS_SCHEMA: Final[dict[str, pl.DataType]] = {
    "date": pl.Date(),
    "fii_buy_cr": pl.Float64(),
    "fii_sell_cr": pl.Float64(),
    "fii_net_cr": pl.Float64(),
    "dii_buy_cr": pl.Float64(),
    "dii_sell_cr": pl.Float64(),
    "dii_net_cr": pl.Float64(),
}


def _normalise_category(raw: str) -> str | None:
    text = raw.strip().upper().replace(" ", "")
    if text.startswith("FII") or text.startswith("FPI"):
        return "fii"
    if text.startswith("DII"):
        return "dii"
    return None


def parse_fiidii_json(text: str) -> pl.DataFrame:
    """Parse the ``fiidiiTradeReact`` payload (latest day) into long rows.

    Columns: ``date``, ``category`` (``fii``/``dii``), ``buy_value_cr``,
    ``sell_value_cr``, ``net_value_cr`` — the same shape as the backfill ledger,
    so the two concatenate.
    """
    payload: Any = json.loads(text)
    if not isinstance(payload, list):
        raise ValueError(f"fiidii payload is {type(payload).__name__}, expected a list")
    rows: list[dict[str, Any]] = []
    for item in payload:
        category = _normalise_category(str(item.get("category", "")))
        if category is None:
            logger.warning(f"fiidii: unknown category {item.get('category')!r}, skipped")
            continue
        rows.append(
            {
                "date": datetime.strptime(str(item["date"]).strip(), "%d-%b-%Y").date(),
                "category": category,
                "buy_value_cr": _float_or_none(str(item.get("buyValue", ""))),
                "sell_value_cr": _float_or_none(str(item.get("sellValue", ""))),
                "net_value_cr": _float_or_none(str(item.get("netValue", ""))),
                "source": "snapshot",
            }
        )
    return pl.DataFrame(rows, schema=_FLOWS_LONG_SCHEMA)


def _float_or_none(value: str) -> float | None:
    """NSE writes a literal ``-`` where a figure does not exist. Null, never 0."""
    text = value.strip().replace(",", "")
    if text in {"", "-", "NA", "N/A"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


_FLOWS_LONG_SCHEMA: Final[dict[str, pl.DataType]] = {
    "date": pl.Date(),
    "category": pl.Utf8(),
    "buy_value_cr": pl.Float64(),
    "sell_value_cr": pl.Float64(),
    "net_value_cr": pl.Float64(),
    # Which artefact this row came from. Load-bearing: `pivot_flows` resolves a
    # duplicated (date, category) by SOURCE PRECEDENCE (snapshot > ledger)
    # rather than by averaging, and without this column it cannot tell them
    # apart. "snapshot" = the dated fiidiiTradeReact JSON; "ledger" =
    # fiidii_backfill.csv.
    "source": pl.Utf8(),
}


def parse_flows_backfill(text: str) -> pl.DataFrame:
    """Parse the local flows ledger (:data:`FlowsSource.BACKFILL_COLUMNS`).

    Dates are ISO; ``category`` accepts any spelling NSE has used (``FII``,
    ``FII/FPI``, ``DII``) and is normalised to ``fii`` / ``dii``.
    """
    reader = csv.DictReader(io.StringIO(text))
    rows: list[dict[str, Any]] = []
    for raw in reader:
        row = {(k or "").strip(): (v or "").strip() for k, v in raw.items()}
        missing = set(FlowsSource.BACKFILL_COLUMNS) - set(row)
        if missing:
            raise ValueError(f"flows backfill missing columns {sorted(missing)}")
        category = _normalise_category(row["category"])
        if category is None:
            logger.warning(f"flows backfill: unknown category {row['category']!r}, skipped")
            continue
        rows.append(
            {
                "date": date.fromisoformat(row["date"]),
                "category": category,
                "buy_value_cr": _float_or_none(row["buy_value_cr"]),
                "sell_value_cr": _float_or_none(row["sell_value_cr"]),
                "net_value_cr": _float_or_none(row["net_value_cr"]),
                "source": "ledger",
            }
        )
    return pl.DataFrame(rows, schema=_FLOWS_LONG_SCHEMA)


_FLOW_VALUE_COLUMNS: Final[tuple[str, ...]] = (
    "buy_value_cr", "sell_value_cr", "net_value_cr",
)


def _resolve_flow_duplicates(long: pl.DataFrame) -> pl.DataFrame:
    """One row per (date, category), by precedence rather than by averaging.

    ``source`` (when present) ranks the artefacts: ``snapshot`` (the dated API
    response) beats ``ledger`` (``fiidii_backfill.csv``) beats anything else.
    Rows that disagree are logged as an override so a silent change of value is
    impossible; rows that agree collapse without noise.
    """
    frame = long
    if "source" not in frame.columns:
        frame = frame.with_columns(pl.lit("unknown").alias("source"))
    rank = pl.when(pl.col("source") == "snapshot").then(0).when(
        pl.col("source") == "ledger"
    ).then(1).otherwise(2)
    frame = frame.with_columns(rank.alias("_rank"))

    dupes = (
        frame.group_by(["date", "category"])
        .agg(
            pl.col("net_value_cr").n_unique().alias("_n_net"),
            pl.col("net_value_cr").alias("_nets"),
            pl.col("source").alias("_sources"),
        )
        .filter(pl.col("_n_net") > 1)
    )
    for row in dupes.iter_rows(named=True):
        logger.warning(
            f"conflicting flow observations for {row['date']} {row['category']}: "
            f"net_value_cr {row['_nets']} from sources {row['_sources']}; "
            f"taking the highest-precedence source (snapshot > ledger). "
            f"These are NOT averaged."
        )
    keep = (
        frame.sort(["date", "category", "_rank"])
        .group_by(["date", "category"], maintain_order=True)
        .first()
    )
    return keep.select(["date", "category", *_FLOW_VALUE_COLUMNS])


def pivot_flows(long: pl.DataFrame) -> pl.DataFrame:
    """Turn long ``(date, category, ...)`` flow rows into :data:`FLOWS_SCHEMA`.

    A date with only one of the two categories keeps nulls for the other; the
    absent side is unknown, not zero.

    Duplicate ``(date, category)`` rows are **not averaged**.  The same date can
    arrive from two artefacts — the dated JSON API snapshot and the
    ``fiidii_backfill.csv`` ledger — and this used to return their mean, so a
    ledger value of -100.0 against a snapshot value of -3111.94 silently became
    -1605.97: a number that was never observed anywhere.  Precedence is
    explicit instead: **the dated API snapshot wins over the ledger**, because
    it is the primary source and the ledger is a hand-maintained backfill.  An
    override is logged. Identical duplicates are collapsed silently.
    """
    if long.height == 0:
        return pl.DataFrame(schema=FLOWS_SCHEMA)
    out = _resolve_flow_duplicates(long)
    frames = {
        cat: out.filter(pl.col("category") == cat)
        .drop("category")
        .rename(
            {
                "buy_value_cr": f"{cat}_buy_cr",
                "sell_value_cr": f"{cat}_sell_cr",
                "net_value_cr": f"{cat}_net_cr",
            }
        )
        for cat in ("fii", "dii")
    }
    merged = frames["fii"].join(frames["dii"], on="date", how="full", coalesce=True)
    for column, dtype in FLOWS_SCHEMA.items():
        if column not in merged.columns:
            merged = merged.with_columns(pl.lit(None, dtype=dtype).alias(column))
    return merged.select(list(FLOWS_SCHEMA)).sort("date")


class FlowsSource(_CachedSource):
    """Market-wide FII/DII cash-market flows, one row per day.

    ``fetch`` returns :data:`FLOWS_SCHEMA`: ``date`` plus buy/sell/net in ₹ crore
    for each of FII and DII. There is no per-stock dimension — this becomes a
    market-level feature broadcast to every ticker, like the regime vector.

    History comes from a local append-only ledger, because the only live
    endpoint reachable in this session returns the latest day only. The ledger
    lives at ``<cache_root>/flows/fiidii_backfill.csv`` with columns
    :data:`BACKFILL_COLUMNS`; :meth:`refresh_latest` appends today's API row to
    it, so running the fetch script daily accumulates history.
    """

    subdir = "flows"
    BACKFILL_COLUMNS: Final[tuple[str, ...]] = (
        "date",
        "category",
        "buy_value_cr",
        "sell_value_cr",
        "net_value_cr",
    )
    LEDGER_NAME: Final[str] = "fiidii_backfill.csv"

    def fetch(self, start: date, end: date) -> pl.DataFrame:
        ledger = self.cache.read(self.LEDGER_NAME)
        long = (
            parse_flows_backfill(ledger)
            if ledger is not None
            else pl.DataFrame(schema=_FLOWS_LONG_SCHEMA)
        )
        for name in sorted(p.name for p in self.cache.root.glob("fiidii_*.json")):
            text = self.cache.read(name)
            if text is not None:
                long = pl.concat([long, parse_fiidii_json(text)])
        wide = pivot_flows(long)
        if wide.height == 0:
            return wide
        return wide.filter(pl.col("date").is_between(start, end)).sort("date")

    def refresh_latest(self) -> pl.DataFrame:
        """Fetch the latest published day and cache it as a dated JSON file.

        It writes ``fiidii_<date>.json`` ONLY.  It does **not** append to
        ``fiidii_backfill.csv``; that ledger is hand-maintained and this method
        never touches it.

        Which artefact is authoritative for a date present in both: the dated
        JSON snapshot.  ``pivot_flows`` resolves the collision by source
        precedence (``snapshot`` > ``ledger``) and logs the override; the two
        are never averaged.
        """
        if self.offline:
            return pl.DataFrame(schema=_FLOWS_LONG_SCHEMA)
        text = self.client.get_text(FIIDII_URL)
        parsed = parse_fiidii_json(text)
        if parsed.height:
            day = parsed["date"].min()
            assert isinstance(day, date)
            self.cache.write(f"fiidii_{day.isoformat()}.json", text)
        return parsed


# ── 3. Bulk and block deals ──────────────────────────────────────────────────


DEAL_ROW_SCHEMA: Final[dict[str, pl.DataType]] = {
    "date": pl.Date(),
    "ticker": pl.Utf8(),
    "deal_type": pl.Utf8(),
    "client_name": pl.Utf8(),
    "side": pl.Utf8(),
    "quantity": pl.Int64(),
    "price": pl.Float64(),
    "value_inr": pl.Float64(),
}

DEALS_SCHEMA: Final[dict[str, pl.DataType]] = {
    "date": pl.Date(),
    "ticker": pl.Utf8(),
    "n_deals": pl.Int64(),
    "buy_value_inr": pl.Float64(),
    "sell_value_inr": pl.Float64(),
    "net_value_inr": pl.Float64(),
}

_DEAL_HEADER_ALIASES: Final[dict[str, str]] = {
    "date": "date",
    "symbol": "symbol",
    "clientname": "client_name",
    "buy/sell": "side",
    "quantitytraded": "quantity",
    "tradeprice/wght.avg.price": "price",
    "tradeprice/wghtavgprice": "price",
}


def filter_deals_text(text: str, day: date) -> str:
    """The same CSV, keeping only the rows whose trade date is ``day``.

    Operates on the ORIGINAL text — same header, same columns, same quoting —
    rather than re-serialising a parsed frame, so a cached per-date file is
    byte-compatible with what :func:`parse_deals_csv` expects and nothing is
    lost in a round trip.  A row whose date will not parse is kept: dropping it
    here would silently discard data that the parser is entitled to complain
    about itself.
    """
    reader = csv.reader(io.StringIO(text))
    try:
        header = next(reader)
    except StopIteration:
        return text
    keys = [_DEAL_HEADER_ALIASES.get(_normalise_header(h), "") for h in header]
    try:
        date_col = keys.index("date")
    except ValueError:
        return text
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(header)
    for record in reader:
        if not record or all(not cell.strip() for cell in record):
            continue
        if len(record) <= date_col:
            continue
        try:
            parsed = datetime.strptime(record[date_col].strip(), "%d-%b-%Y").date()
        except ValueError:
            writer.writerow(record)
            continue
        if parsed == day:
            writer.writerow(record)
    return buf.getvalue()


def parse_deals_csv(text: str, *, deal_type: str) -> pl.DataFrame:
    """Parse ``bulk.csv`` / ``block.csv`` into per-deal rows.

    Returns :data:`DEAL_ROW_SCHEMA`. Client names contain commas and are quoted,
    so the stdlib CSV reader is used rather than a split. ``value_inr`` is
    ``quantity × price``; a row missing either is dropped with a warning rather
    than valued at zero.
    """
    reader = csv.reader(io.StringIO(text))
    try:
        header = next(reader)
    except StopIteration:
        return pl.DataFrame(schema=DEAL_ROW_SCHEMA)
    keys = [_DEAL_HEADER_ALIASES.get(_normalise_header(h), "") for h in header]
    for required in ("date", "symbol", "side", "quantity", "price"):
        if required not in keys:
            raise ValueError(
                f"{deal_type} deals file: header {header!r} has no column mapping to "
                f"{required!r}; known aliases {sorted(_DEAL_HEADER_ALIASES)}"
            )

    rows: list[dict[str, Any]] = []
    for record in reader:
        if not record or all(not cell.strip() for cell in record):
            continue
        row = {k: v.strip() for k, v in zip(keys, record, strict=False) if k}
        quantity = _int_or_none(row.get("quantity", ""))
        price = _float_or_none(row.get("price", ""))
        if quantity is None or price is None:
            logger.warning(f"{deal_type} deal row has no quantity/price, dropped: {record!r}")
            continue
        side = row.get("side", "").strip().upper()
        rows.append(
            {
                "date": datetime.strptime(row["date"], "%d-%b-%Y").date(),
                "ticker": nse_symbol_to_ticker(row["symbol"]),
                "deal_type": deal_type,
                "client_name": row.get("client_name", ""),
                "side": "BUY" if side.startswith("B") else "SELL",
                "quantity": quantity,
                "price": price,
                "value_inr": float(quantity) * price,
            }
        )
    return pl.DataFrame(rows, schema=DEAL_ROW_SCHEMA)


def aggregate_deals(rows: pl.DataFrame) -> pl.DataFrame:
    """Reduce per-deal rows to one row per (date, ticker) — :data:`DEALS_SCHEMA`.

    ``net_value_inr`` is buy value minus sell value, so a crossed deal (the same
    block reported once on each side) nets to zero while ``n_deals`` stays 2.
    """
    if rows.height == 0:
        return pl.DataFrame(schema=DEALS_SCHEMA)
    buy = pl.when(pl.col("side") == "BUY").then(pl.col("value_inr")).otherwise(0.0)
    sell = pl.when(pl.col("side") == "SELL").then(pl.col("value_inr")).otherwise(0.0)
    return (
        rows.with_columns(buy.alias("_buy"), sell.alias("_sell"))
        .group_by(["date", "ticker"])
        .agg(
            pl.len().cast(pl.Int64).alias("n_deals"),
            pl.col("_buy").sum().alias("buy_value_inr"),
            pl.col("_sell").sum().alias("sell_value_inr"),
        )
        .with_columns(
            (pl.col("buy_value_inr") - pl.col("sell_value_inr")).alias("net_value_inr")
        )
        .select(list(DEALS_SCHEMA))
        .sort(["date", "ticker"])
    )


class DealsSource(_CachedSource):
    """Bulk and block deals, aggregated per (date, ticker).

    ``fetch`` returns :data:`DEALS_SCHEMA`. NSE's ``bulk.csv`` / ``block.csv``
    hold the latest trading day only (verified: all 156 bulk rows on 2026-09-05
    carried ``04-SEP-2026``), so :meth:`snapshot_latest` writes today's files
    into the per-date cache and history accumulates by running the fetch script
    daily. Dates present in the cache are *observed*; dates absent are unknown,
    and :func:`trader.data.features_ext.compute_ext_features` emits null for
    them rather than treating "no deal row" as "no deal".
    """

    subdir = "deals"

    def observed_dates(self) -> list[date]:
        """Dates for which a deals snapshot exists on disk."""
        days: set[date] = set()
        for path in self.cache.root.glob("*_*.csv"):
            stamp = path.stem.split("_", 1)[1]
            try:
                days.add(date.fromisoformat(stamp))
            except ValueError:
                logger.warning(f"deals cache: unparseable filename {path.name}")
        return sorted(days)

    def fetch(self, start: date, end: date) -> pl.DataFrame:
        rows: list[pl.DataFrame] = []
        for day in self.observed_dates():
            if not (start <= day <= end):
                continue
            for deal_type in ("bulk", "block"):
                text = self.cache.read(f"{deal_type}_{day.isoformat()}.csv")
                if text is not None:
                    # Filter to the date the FILENAME names.  The NSE endpoint
                    # returns a multi-date window, and `snapshot_latest` used to
                    # cache the whole response under every date it contained —
                    # so a 2-date response cached under 2 names made every deal
                    # count twice (n_deals and buy_value_inr both doubled).
                    # Filtering here is the backstop; snapshot_latest also now
                    # writes only the matching rows.
                    rows.append(
                        parse_deals_csv(text, deal_type=deal_type)
                        .filter(pl.col("date") == day)
                    )
        if not rows:
            return pl.DataFrame(schema=DEALS_SCHEMA)
        return aggregate_deals(pl.concat(rows))

    def snapshot_latest(self) -> list[date]:
        """Download today's bulk/block files into the per-date cache.

        Returns the trade dates written. Idempotent: a date already cached is
        left alone, so re-running on the same day costs two requests and no
        change.
        """
        if self.offline:
            return []
        written: set[date] = set()
        for deal_type, url in (("bulk", BULK_URL), ("block", BLOCK_URL)):
            try:
                text = self.client.get_text(url)
            except NSEFetchError as exc:
                logger.warning(f"{deal_type} deals snapshot failed: {exc}")
                continue
            parsed = parse_deals_csv(text, deal_type=deal_type)
            for day in sorted(set(parsed["date"].to_list())):
                name = f"{deal_type}_{day.isoformat()}.csv"
                if self.cache.read(name) is None:
                    # Re-serialise ONLY this day's rows.  Writing the full
                    # multi-date response under each date's filename made a
                    # file's contents disagree with its name, and every
                    # subsequent fetch double-counted.
                    self.cache.write(name, filter_deals_text(text, day))
                written.add(day)
        return sorted(written)


__all__: Final[Sequence[str]] = (
    "BHAVDATA_VERIFIED_FROM",
    "CASH_SERIES",
    "DEALS_SCHEMA",
    "DEAL_ROW_SCHEMA",
    "DELIVERY_SCHEMA",
    "FLOWS_SCHEMA",
    "MTO_VERIFIED_FROM",
    "DealsSource",
    "DeliverySource",
    "FlowsSource",
    "MTOLayout",
    "MTOLayoutError",
    "NSEClient",
    "NSEFetchError",
    "NSENotFound",
    "aggregate_deals",
    "detect_mto_layout",
    "nse_symbol_to_ticker",
    "parse_deals_csv",
    "parse_fiidii_json",
    "parse_flows_backfill",
    "parse_mto",
    "parse_sec_bhavdata",
    "pivot_flows",
)
