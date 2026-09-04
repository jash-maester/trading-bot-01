"""Zerodha Kite Connect adapter for historical OHLCV bars — strictly READ ONLY.

Why this module exists
----------------------
yfinance is the training-time source of truth: it is free, it back-adjusts prices
for splits and bonuses, and it goes back far enough to train on. What it is not is
a broker feed — it is an unofficial scrape with no SLA, missing recent bars, and
occasional silent revisions. Kite is the feed the live account actually trades
against, so having the same panel reachable from the broker's own history API is
what lets us reconcile "what the model was trained on" against "what the broker
believes happened".

Scope is deliberately narrowed to data. This class touches only ``instruments()``
and ``historical_data()`` — the two read endpoints. It exposes no way to place,
modify or cancel an order, and it must stay that way: order execution belongs to
the local paper broker and its dummy ledger. If you ever need an order path, add a
separate broker module so that "reads data" and "moves money" can never be
confused at a call site.

Contract with the rest of the pipeline
--------------------------------------
The frame returned by :meth:`ZerodhaSource.fetch_ohlcv` is byte-for-byte
schema-compatible with :class:`~trader.data.sources.yfinance_source.YFinanceSource`
(see :data:`OUTPUT_SCHEMA`), because both feed the same alignment / feature code.
A divergence here does not raise — it silently becomes train/serve skew.

Three deliberate choices that are easy to get wrong:

* **Date range is half-open ``[start, end)``**, matching ``yf.download``. Kite's
  ``to_date`` is inclusive, so we request ``end - 1 day``. Same arguments to
  either source therefore yield the same rows.
* **Timestamps are naive IST.** Kite stamps candles ``+05:30``; the panel is naive.
  We convert *into* IST and then drop the tzinfo, so an intraday bar keeps its NSE
  wall-clock time. Converting to UTC instead would push the 09:15 open back to the
  previous calendar day and quietly corrupt every date-keyed join.
* **Prices are NOT back-adjusted.** Kite returns raw traded prices; yfinance is
  called with ``auto_adjust=True``. Do not mix the two in one panel — see
  :meth:`ZerodhaSource.fetch_corporate_actions`.

Credentials come from ``KITE_API_KEY`` / ``KITE_ACCESS_TOKEN`` in the environment.
Nothing in this codebase calls ``load_dotenv()``, so the caller must export them
(e.g. ``set -a; source .env; set +a``) — a populated ``.env`` alone does nothing.
Access tokens expire daily and are minted from a single-use ``KITE_REQUEST_TOKEN``
plus ``KITE_API_SECRET`` via :meth:`ZerodhaSource.mint_access_token`, which is a
deliberately separate, explicit call: a data fetch must never spend that one-shot
token behind the caller's back. Secrets are never logged — not the values, not
their lengths.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Final, Literal

import polars as pl
from loguru import logger

from trader.data.sources.kite_symbols import (
    ResolutionReport,
    ResolutionStatus,
    resolve_symbol,
    resolve_universe,
    to_kite_symbol,
    to_project_ticker,
)

# NSE never observes DST and the exchange has never changed offset, so a fixed
# offset is safer here than a tz-database lookup that can go missing on a slim image.
IST: Final = timezone(timedelta(hours=5, minutes=30))

# The project speaks yfinance ("RELIANCE.NS"); Kite speaks bare tradingsymbols on
# an exchange ("RELIANCE" @ NSE). Every crossing of that boundary goes through
# trader.data.sources.kite_symbols, so renames, series suffixes and delistings are
# handled in exactly one place.
INDICES_SEGMENT: Final = "INDICES"

_INTERVAL_TO_KITE: Final[dict[str, str]] = {
    "1d": "day",
    "1h": "60minute",
    "15m": "15minute",
}

# Kite caps the span a single historical request may cover (docs: "Historical
# candle limits"). Exceeding it is a hard 400, not a truncation, so long ranges
# must be chunked and stitched.
DEFAULT_MAX_DAYS_PER_REQUEST: Final[dict[str, int]] = {
    "day": 2000,
    "60minute": 400,
    "15minute": 200,
}

# Historical API is throttled at 3 requests/second.
DEFAULT_MIN_REQUEST_INTERVAL: Final = 1.0 / 3.0

# kiteconnect may not be installed, so its exception classes cannot be imported to
# use in `except` clauses. Classify by class name / HTTP code instead.
_RETRYABLE_EXC_NAMES: Final = frozenset({"NetworkException", "DataException", "GeneralException"})
_AUTH_EXC_NAMES: Final = frozenset({"TokenException", "PermissionException"})
_RETRYABLE_STATUS: Final = frozenset({429, 500, 502, 503, 504})

# Exact schema of YFinanceSource.fetch_ohlcv output, including column order.
OUTPUT_SCHEMA: Final[dict[str, Any]] = {
    "date": pl.Datetime("ns"),
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Int64,
    "adj_close": pl.Float64,
    "ticker": pl.Utf8,
    "source": pl.Utf8,
}

_INSTALL_HINT = (
    'install the optional extra: `uv sync --extra zerodha` (or `uv pip install "kiteconnect>=4.2"`)'
)


class ZerodhaUnavailableError(NotImplementedError):
    """The Kite-backed source cannot serve data in this environment.

    Subclasses :class:`NotImplementedError` on purpose. Without the optional SDK or
    without credentials this class is exactly what it has always been — an
    interface with no working implementation — and callers (and the M2 contract
    test) that treat ``ZerodhaSource`` as "not available here" stay correct. It is
    raised before any network call, so an unconfigured process can never reach out.
    """


class KiteConnectNotInstalledError(ZerodhaUnavailableError, ImportError):
    """``kiteconnect`` is an optional dependency and is not installed."""


class ZerodhaCredentialsError(ZerodhaUnavailableError):
    """``KITE_API_KEY`` / ``KITE_ACCESS_TOKEN`` are absent from the environment."""


class ZerodhaPermissionError(RuntimeError):
    """The app's Kite plan does not include the endpoint that was called.

    Distinct from :class:`ZerodhaAuthError` because the remedy is completely
    different: an expired token needs a re-login, whereas this needs a Kite
    Connect subscription (market data — historical candles, quote, ohlc, ltp —
    is not available on the free Personal tier).  Verified 2026-09-04: a
    Personal-tier app returns ``PermissionException`` for every market-data
    call while ``profile``/``instruments``/``holdings`` succeed, so reporting
    it as "token expired" sends you round a login loop that cannot fix it.
    """


class ZerodhaAuthError(RuntimeError):
    """Kite rejected the session (expired/invalid access token).

    Deliberately *not* a :class:`ZerodhaUnavailableError`: this surfaces mid-run and
    is not retryable per ticker — retrying 150 symbols against a dead token would
    turn one loud failure into an empty panel.
    """


def _empty_frame() -> pl.DataFrame:
    """Zero rows, full schema — so an empty result still joins like a real one."""
    return pl.DataFrame(schema=OUTPUT_SCHEMA)


def _import_kiteconnect() -> Any:
    """Lazy import — importing this module must never require the optional SDK."""
    try:
        import kiteconnect
    except ImportError as exc:
        raise KiteConnectNotInstalledError(
            "ZerodhaSource needs the optional 'kiteconnect' package; " + _INSTALL_HINT
        ) from exc
    return kiteconnect


def _require_credentials(*pairs: tuple[str, str]) -> None:
    """Fail before any network call if an env var is unset — by name, never by value."""
    missing = [name for name, value in pairs if not value]
    if missing:
        raise ZerodhaCredentialsError(
            f"ZerodhaSource is missing credentials: {', '.join(missing)}. "
            "Export them into the environment (nothing in this codebase calls "
            "load_dotenv), e.g. `set -a; source .env; set +a`. An access token is "
            "minted once per trading day — see ZerodhaSource.mint_access_token."
        )


class ZerodhaSource:
    """Read-only historical bar source backed by Kite Connect.

    Implements :class:`~trader.data.sources.base.MarketDataSource`. Construction is
    free of side effects: no import of ``kiteconnect``, no credential lookup and no
    network access happen until a fetch method is actually called.
    """

    def __init__(
        self,
        api_key: str | None = None,
        access_token: str | None = None,
        *,
        cache_root: str | Path = "data/raw/zerodha",
        exchange: str = "NSE",
        max_retries: int = 3,
        min_request_interval: float = DEFAULT_MIN_REQUEST_INTERVAL,
        backoff_base: float = 1.0,
        max_days_per_request: dict[str, int] | None = None,
        include_indices: bool = False,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        client: Any | None = None,
    ) -> None:
        """Explicit ``api_key``/``access_token`` win over the environment.

        ``sleep``/``clock`` are injectable so tests can exercise throttling and
        backoff without spending wall-clock time; ``client`` accepts a pre-built
        (or fake) KiteConnect instance and bypasses credential handling entirely.

        ``include_indices`` opts the instrument map into NSE's ``INDICES`` segment
        so ``^NSEI`` resolves to the NIFTY 50 series. It is off by default because
        an index is not tradable: a caller enumerating a cash-equity universe must
        not be able to accidentally size a position in one.
        """
        self._api_key = api_key
        self._access_token = access_token
        self._cache = Path(cache_root)
        self._exchange = exchange
        self._include_indices = include_indices
        self._max_retries = max(1, max_retries)
        self._min_interval = min_request_interval
        self._backoff_base = backoff_base
        self._max_days = dict(max_days_per_request or DEFAULT_MAX_DAYS_PER_REQUEST)
        self._sleep = sleep
        self._clock = clock
        self._client: Any | None = client
        self._token_cache: dict[str, int] | None = None
        self._last_request_at: float | None = None

    def __repr__(self) -> str:
        # Never render secrets: presence only, so a repr in a traceback or log is safe.
        has_key = bool(self._api_key or os.environ.get("KITE_API_KEY"))
        has_token = bool(self._access_token or os.environ.get("KITE_ACCESS_TOKEN"))
        return (
            f"ZerodhaSource(exchange={self._exchange!r}, cache_root={str(self._cache)!r}, "
            f"api_key_set={has_key}, access_token_set={has_token})"
        )

    # ------------------------------------------------------------------ symbols

    @staticmethod
    def to_kite_symbol(ticker: str) -> str:
        """``RELIANCE.NS`` → ``RELIANCE``. Delegates to :mod:`kite_symbols`."""
        return to_kite_symbol(ticker)

    @staticmethod
    def to_project_ticker(tradingsymbol: str) -> str:
        """``RELIANCE`` → ``RELIANCE.NS``. Delegates to :mod:`kite_symbols`."""
        return to_project_ticker(tradingsymbol)

    def resolve(self, tickers: list[str]) -> ResolutionReport:
        """Classify every ticker against the live instrument dump, without fetching.

        Exposed separately from ``fetch_ohlcv`` because "which of my 163 names does
        the broker still know about" is a question worth answering — and logging —
        *before* spending a thousand history requests on the answer.
        """
        return resolve_universe(tickers, self._load_instruments())

    # ------------------------------------------------------------------ public

    def fetch_ohlcv(
        self,
        tickers: list[str],
        start: datetime,
        end: datetime,
        interval: Literal["1d", "1h", "15m"] = "1d",
    ) -> pl.DataFrame:
        """Fetch bars for ``tickers`` over the half-open range ``[start, end)``.

        Tickers that cannot be resolved, or whose history fails after retries, are
        skipped with a log line rather than aborting the batch — but a ticker is
        never returned *partially*, since a hole in the middle of a series looks
        like a real gap to the alignment stage.
        """
        kite_interval = _INTERVAL_TO_KITE.get(interval)
        if kite_interval is None:  # unreachable via the Literal, cheap guard for dynamic callers
            logger.warning(f"ZerodhaSource: unsupported interval {interval!r}")
            return _empty_frame()

        # yfinance treats `end` as exclusive; Kite's to_date is inclusive.
        last_day = end.date() - timedelta(days=1)
        if last_day < start.date():
            logger.warning(f"ZerodhaSource: empty range {start.date()} → {end.date()}")
            return _empty_frame()

        frames: list[pl.DataFrame] = []
        for ticker in tickers:
            frame = self._fetch_one(ticker, start.date(), last_day, kite_interval, interval)
            if frame is not None and not frame.is_empty():
                frames.append(frame)

        if not frames:
            return _empty_frame()

        combined = pl.concat(frames)
        # Chunk boundaries and Kite's inclusive to_date can hand back the same candle
        # twice; dedupe on the panel key. Row order mirrors the yfinance source
        # (per-ticker blocks, chronological within a ticker).
        return combined.sort(["ticker", "date"]).unique(
            subset=["date", "ticker"], keep="first", maintain_order=True
        )

    def fetch_corporate_actions(self, tickers: list[str]) -> pl.DataFrame:
        """Not available: Kite Connect exposes no corporate-actions endpoint.

        This is a genuine gap in the API, not a missing implementation — there is no
        splits/bonus/dividend resource to call. Splits and bonuses must keep coming
        from the yfinance path (``auto_adjust=True``), and this method stays loud on
        purpose: a plausible-looking empty frame here would let an unadjusted Kite
        series be stitched onto a back-adjusted yfinance series. That produces a
        panel with a fake 50% overnight return at every past split, which trains
        fine, backtests fine, and is wrong.
        """
        raise NotImplementedError(
            "Kite Connect has no corporate-actions endpoint. Use YFinanceSource."
            f" (requested: {len(tickers)} tickers)"
        )

    # ------------------------------------------------------------------ auth

    @staticmethod
    def mint_access_token(
        api_key: str | None = None,
        api_secret: str | None = None,
        request_token: str | None = None,
    ) -> str:
        """Exchange a request token for a one-day access token. Never called implicitly.

        Kite access tokens expire every morning, and the request token that mints one
        is single-use and short-lived — which is why ``.env`` carries
        ``KITE_REQUEST_TOKEN`` rather than an access token. Doing this exchange lazily
        inside a data fetch would burn that one-shot token on a cache miss, so it is a
        separate, explicit call: run it, then export the result as
        ``KITE_ACCESS_TOKEN`` for the rest of the session. The token is returned, never
        logged.
        """
        kiteconnect = _import_kiteconnect()
        key = api_key or os.environ.get("KITE_API_KEY", "")
        secret = api_secret or os.environ.get("KITE_API_SECRET", "")
        token = request_token or os.environ.get("KITE_REQUEST_TOKEN", "")
        _require_credentials(
            ("KITE_API_KEY", key), ("KITE_API_SECRET", secret), ("KITE_REQUEST_TOKEN", token)
        )

        client = kiteconnect.KiteConnect(api_key=key)
        session: dict[str, Any] = client.generate_session(token, api_secret=secret)
        logger.info("Minted a Kite access token; export it as KITE_ACCESS_TOKEN (expires daily)")
        return str(session["access_token"])

    # ------------------------------------------------------------------ client

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client

        kiteconnect = _import_kiteconnect()
        api_key = self._api_key or os.environ.get("KITE_API_KEY", "")
        access_token = self._access_token or os.environ.get("KITE_ACCESS_TOKEN", "")
        _require_credentials(("KITE_API_KEY", api_key), ("KITE_ACCESS_TOKEN", access_token))

        client = kiteconnect.KiteConnect(api_key=api_key)
        client.set_access_token(access_token)
        self._client = client
        logger.debug(f"ZerodhaSource: Kite client ready for exchange {self._exchange}")
        return client

    # ------------------------------------------------------------------ instruments

    def _instrument_cache_path(self) -> Path:
        # Dated filename: the dump changes at most once a day, and a stale token map
        # silently fetches the wrong instrument, so expiry is part of the path. The
        # indices flag is in the path too — the two variants hold different symbol
        # sets, and reusing one for the other would make ^NSEI resolve or not
        # depending on which caller happened to warm the cache first.
        scope = "eq+idx" if self._include_indices else "eq"
        return (
            self._cache / f"instruments_{self._exchange}_{scope}_{date.today().isoformat()}.parquet"
        )

    def _load_instruments(self) -> dict[str, int]:
        """tradingsymbol → instrument_token, memoised in memory and on disk.

        The NSE dump is a multi-MB CSV of every tradable instrument; fetching it per
        ticker would blow the rate limit before the first candle arrives.
        """
        if self._token_cache is not None:
            return self._token_cache

        path = self._instrument_cache_path()
        if path.exists():
            try:
                cached = pl.read_parquet(path)
                self._token_cache = {
                    str(sym): int(tok)
                    for sym, tok in zip(
                        cached["tradingsymbol"].to_list(),
                        cached["instrument_token"].to_list(),
                        strict=True,
                    )
                }
                logger.debug(f"Instrument cache hit: {path} ({len(self._token_cache)} symbols)")
                return self._token_cache
            except Exception as exc:  # corrupt/partial cache must not be fatal
                logger.warning(f"Ignoring unreadable instrument cache {path}: {exc}")

        client = self._get_client()
        self._throttle()
        rows: list[dict[str, Any]] = list(client.instruments(self._exchange))

        # Cash equities only, unless indices were explicitly requested. The NSE dump
        # also carries derivatives and the indices, and indices are typed "EQ" too —
        # they are separated by `segment`, so both fields are needed to keep
        # "NIFTY 50" from shadowing a tradable symbol.
        allowed_segments: tuple[str | None, ...] = (None, self._exchange)
        if self._include_indices:
            allowed_segments += (INDICES_SEGMENT,)

        tokens: dict[str, int] = {}
        for row in rows:
            if row.get("instrument_type") not in (None, "EQ"):
                continue
            if row.get("segment") not in allowed_segments:
                continue
            symbol = str(row.get("tradingsymbol", "")).upper()
            token = row.get("instrument_token")
            if symbol and token is not None:
                tokens[symbol] = int(token)

        self._token_cache = tokens
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            pl.DataFrame(
                {
                    "tradingsymbol": list(tokens.keys()),
                    "instrument_token": list(tokens.values()),
                },
                schema={"tradingsymbol": pl.Utf8, "instrument_token": pl.Int64},
            ).write_parquet(path)
            logger.debug(f"Cached {len(tokens)} instruments to {path}")
        except Exception as exc:  # a read-only cache dir is not a reason to fail the fetch
            logger.warning(f"Could not write instrument cache {path}: {exc}")
        return tokens

    def _resolve_token(self, ticker: str) -> int | None:
        """Instrument token for a project ticker, or None with a reason in the log.

        Routed through :func:`resolve_symbol` rather than a bare dict lookup so that
        a rename (LTIM → LTM) or a series move (STLTECH → STLTECH-BE) resolves
        instead of silently deleting a ticker from the universe, and so a genuine
        delisting is logged as a *known* loss rather than an anonymous miss.
        """
        instruments = self._load_instruments()
        res = resolve_symbol(ticker, instruments)
        if res.kite_symbol is None:
            level = "warning" if res.status is ResolutionStatus.DELISTED else "error"
            logger.log(
                level.upper(),
                f"{ticker}: {res.status.value} on {self._exchange} — {res.note}",
            )
            return None
        if res.status is not ResolutionStatus.DIRECT:
            logger.info(f"{ticker}: resolved via {res.status.value} → {res.kite_symbol}")
        return instruments.get(res.kite_symbol)

    # ------------------------------------------------------------------ fetch

    def _fetch_one(
        self,
        ticker: str,
        first_day: date,
        last_day: date,
        kite_interval: str,
        interval: str,
    ) -> pl.DataFrame | None:
        # _resolve_token has already logged *why* (rename gone stale, delisted,
        # unknown); repeating a generic "no token" line here would only bury it.
        token = self._resolve_token(ticker)
        if token is None:
            return None

        candles: list[dict[str, Any]] = []
        for chunk_start, chunk_end in self._chunk_ranges(first_day, last_day, kite_interval):
            try:
                candles.extend(
                    self._historical_with_retry(token, chunk_start, chunk_end, kite_interval)
                )
            except (ZerodhaAuthError, ZerodhaPermissionError, ZerodhaUnavailableError):
                # Session/plan/config problems are global, not per-ticker. A missing
                # market-data subscription fails identically for all 163 symbols, so
                # swallowing it here yields an empty panel plus 163 error logs instead
                # of one actionable failure.
                raise
            except Exception as exc:
                # Drop the whole ticker: a stitched series missing one chunk reads as
                # a genuine trading gap downstream, which is worse than no ticker.
                logger.error(f"Zerodha {ticker} {chunk_start}→{chunk_end} failed: {exc}")
                return None

        if not candles:
            logger.warning(f"No Zerodha data for {ticker} {first_day}→{last_day}")
            return None
        return self._to_polars(candles, ticker, interval)

    def _chunk_ranges(
        self, first_day: date, last_day: date, kite_interval: str
    ) -> list[tuple[date, date]]:
        """Split an inclusive range into contiguous, non-overlapping Kite-sized chunks."""
        max_days = self._max_days.get(kite_interval, 100)
        chunks: list[tuple[date, date]] = []
        cursor = first_day
        while cursor <= last_day:
            chunk_end = min(cursor + timedelta(days=max_days - 1), last_day)
            chunks.append((cursor, chunk_end))
            cursor = chunk_end + timedelta(days=1)
        return chunks

    def _throttle(self) -> None:
        """Space requests by at least ``min_request_interval`` (Kite allows ~3/s)."""
        if self._min_interval <= 0:
            self._last_request_at = self._clock()
            return
        now = self._clock()
        if self._last_request_at is not None:
            wait = self._min_interval - (now - self._last_request_at)
            if wait > 0:
                self._sleep(wait)
                now = self._clock()
        self._last_request_at = now

    def _historical_with_retry(
        self, token: int, from_date: date, to_date: date, kite_interval: str
    ) -> list[dict[str, Any]]:
        client = self._get_client()
        last_exc: Exception | None = None

        for attempt in range(self._max_retries):
            self._throttle()
            try:
                raw: Any = client.historical_data(
                    instrument_token=token,
                    from_date=from_date,
                    to_date=to_date,
                    interval=kite_interval,
                    continuous=False,
                    oi=False,
                )
                return list(raw)
            except Exception as exc:
                if type(exc).__name__ == "PermissionException":
                    raise ZerodhaPermissionError(
                        "Kite refused this call: the app's plan does not include "
                        "market data. Historical candles, quote, ohlc and ltp all "
                        "require the Kite Connect subscription (Rs 500/month); the "
                        "free Personal tier exposes only account and instrument "
                        "endpoints. Re-logging in will NOT fix this."
                    ) from exc
                if _is_auth_error(exc):
                    raise ZerodhaAuthError(
                        "Kite rejected the session — KITE_ACCESS_TOKEN is expired or invalid. "
                        "Access tokens are valid for a single trading day; re-run the login flow."
                    ) from exc
                if not _is_retryable(exc):
                    raise
                last_exc = exc
                wait = self._backoff_base * 2**attempt
                logger.warning(
                    f"Zerodha token {token} attempt {attempt + 1}/{self._max_retries} "
                    f"failed: {exc}; retrying in {wait}s"
                )
                if attempt < self._max_retries - 1:
                    self._sleep(wait)

        raise RuntimeError(
            f"Zerodha historical_data exhausted {self._max_retries} retries: {last_exc}"
        )

    # ------------------------------------------------------------------ shaping

    def _to_polars(self, candles: list[dict[str, Any]], ticker: str, interval: str) -> pl.DataFrame:
        rows: list[dict[str, Any]] = []
        for candle in candles:
            stamp = _to_naive_ist(candle.get("date"))
            if stamp is None:
                continue
            close = _as_float(candle.get("close"))
            rows.append(
                {
                    "date": stamp,
                    "open": _as_float(candle.get("open")),
                    "high": _as_float(candle.get("high")),
                    "low": _as_float(candle.get("low")),
                    "close": close,
                    "volume": _as_int(candle.get("volume")),
                    # Kite serves unadjusted prices and has no adjustment factor, so
                    # adj_close mirrors close. It exists to keep the schema aligned —
                    # it is NOT interchangeable with yfinance's back-adjusted column.
                    "adj_close": close,
                    "ticker": ticker,
                    "source": "zerodha",
                }
            )

        frame = pl.DataFrame(rows, schema=OUTPUT_SCHEMA)
        if interval == "1d":
            # Daily bars must key on midnight to join the trading calendar, exactly
            # as yfinance's DatetimeIndex does.
            frame = frame.with_columns(pl.col("date").dt.truncate("1d"))
        return frame


# ---------------------------------------------------------------------- helpers


def _is_retryable(exc: BaseException) -> bool:
    """Transient? Class name, HTTP code, then message — kiteconnect may be absent."""
    if type(exc).__name__ in _RETRYABLE_EXC_NAMES:
        return True
    code = getattr(exc, "code", None)
    if isinstance(code, int) and code in _RETRYABLE_STATUS:
        return True
    text = str(exc).lower()
    return "429" in text or "too many requests" in text


def _is_auth_error(exc: BaseException) -> bool:
    return type(exc).__name__ in _AUTH_EXC_NAMES


def _to_naive_ist(value: Any) -> datetime | None:
    """Normalise a Kite timestamp to a naive IST datetime.

    Converting into IST before dropping tzinfo (rather than a bare ``replace``)
    keeps the NSE wall-clock reading correct even if the SDK, a proxy or a fixture
    hands back UTC.
    """
    if isinstance(value, datetime):
        if value.tzinfo is not None:
            return value.astimezone(IST).replace(tzinfo=None)
        return value
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day)
    if isinstance(value, str):
        try:
            return _to_naive_ist(datetime.fromisoformat(value))
        except ValueError:
            logger.warning(f"Unparseable Zerodha timestamp: {value!r}")
            return None
    return None


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
