"""Unit tests for the Zerodha Kite historical-data adapter.

Everything here runs with no network and without ``kiteconnect`` installed: the
Kite client is a hand-rolled fake, and the two places the real SDK would be
imported are exercised by injecting a stub module into ``sys.modules``.

The load-bearing test is `test_schema_matches_yfinance_source`: both sources feed
the same alignment/feature code, so a schema drift here would not raise anywhere —
it would just become train/serve skew.
"""
from __future__ import annotations

import sys
import types
from datetime import UTC, date, datetime, timedelta
from typing import Any

import polars as pl
import pytest

from trader.data.sources.zerodha_source import (
    IST,
    OUTPUT_SCHEMA,
    KiteConnectNotInstalledError,
    ZerodhaAuthError,
    ZerodhaCredentialsError,
    ZerodhaSource,
    ZerodhaUnavailableError,
)

MODULE = "trader.data.sources.zerodha_source"


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeKiteError(Exception):
    """Stand-in for kiteconnect.exceptions.KiteException (which carries `.code`)."""

    def __init__(self, message: str, code: int | None = None) -> None:
        super().__init__(message)
        self.code = code


def _named_error(class_name: str, message: str = "boom", code: int | None = None) -> Exception:
    """Build an exception whose *class name* matches a kiteconnect exception.

    The adapter classifies errors by name because it cannot import the SDK's
    exception classes when the SDK is not installed — so tests must too.
    """
    cls = type(class_name, (FakeKiteError,), {})
    return cls(message, code)


class FakeKite:
    """Minimal stand-in for KiteConnect exposing only the two read endpoints."""

    def __init__(
        self,
        instruments: list[dict[str, Any]] | None = None,
        errors: list[Exception | None] | None = None,
        candles_per_request: int | None = None,
    ) -> None:
        self._instruments = instruments if instruments is not None else _default_instruments()
        self._errors = list(errors or [])
        self._candles_per_request = candles_per_request
        self.instrument_calls: list[str] = []
        self.historical_calls: list[dict[str, Any]] = []

    def instruments(self, exchange: str) -> list[dict[str, Any]]:
        self.instrument_calls.append(exchange)
        return self._instruments

    def historical_data(
        self,
        instrument_token: int,
        from_date: date,
        to_date: date,
        interval: str,
        continuous: bool = False,
        oi: bool = False,
    ) -> list[dict[str, Any]]:
        self.historical_calls.append(
            {
                "instrument_token": instrument_token,
                "from_date": from_date,
                "to_date": to_date,
                "interval": interval,
            }
        )
        if self._errors:
            err = self._errors.pop(0)
            if err is not None:
                raise err
        return _candles(from_date, to_date, limit=self._candles_per_request)


def _default_instruments() -> list[dict[str, Any]]:
    return [
        {
            "instrument_token": 738561,
            "tradingsymbol": "RELIANCE",
            "instrument_type": "EQ",
            "segment": "NSE",
        },
        {
            "instrument_token": 2953217,
            "tradingsymbol": "TCS",
            "instrument_type": "EQ",
            "segment": "NSE",
        },
        # Indices ride along in the same NSE dump with instrument_type "EQ"; they
        # must not shadow a cash symbol.
        {
            "instrument_token": 256265,
            "tradingsymbol": "NIFTY 50",
            "instrument_type": "EQ",
            "segment": "INDICES",
        },
        {
            "instrument_token": 12345,
            "tradingsymbol": "RELIANCE24DECFUT",
            "instrument_type": "FUT",
            "segment": "NFO-FUT",
        },
    ]


def _candles(from_date: date, to_date: date, limit: int | None = None) -> list[dict[str, Any]]:
    """Weekday daily candles stamped at IST midnight, as Kite returns them."""
    out: list[dict[str, Any]] = []
    cursor = from_date
    while cursor <= to_date:
        if cursor.weekday() < 5:
            out.append(
                {
                    "date": datetime(cursor.year, cursor.month, cursor.day, tzinfo=IST),
                    "open": 100.0,
                    "high": 105.0,
                    "low": 99.0,
                    "close": 104.0,
                    "volume": 1_000,
                }
            )
            if limit is not None and len(out) >= limit:
                break
        cursor += timedelta(days=1)
    return out


class FakeClock:
    """Monotonic clock that only advances when the injected sleep is called."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def _source(tmp_path: Any, client: FakeKite | None = None, **kwargs: Any) -> ZerodhaSource:
    """A source wired to a fake client, a tmp cache and a no-op clock by default."""
    clock = FakeClock()
    kwargs.setdefault("min_request_interval", 0.0)
    kwargs.setdefault("sleep", clock.sleep)
    kwargs.setdefault("clock", clock)
    return ZerodhaSource(
        cache_root=tmp_path,
        client=client if client is not None else FakeKite(),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Import hygiene / optional dependency
# ---------------------------------------------------------------------------


def test_import_is_side_effect_free() -> None:
    """Importing the module must not pull in `kiteconnect` or touch the network.

    Runs in a subprocess on purpose. Asserting on this process's `sys.modules`
    is order-dependent: any earlier test that reaches the client-construction
    path imports kiteconnect for the whole session, and this assertion then
    fails for a reason that has nothing to do with the module under test.
    A fresh interpreter is the only honest way to test an import side effect.
    """
    import subprocess
    import sys

    code = (
        "import sys\n"
        "import trader.data.sources.zerodha_source as z\n"
        "assert 'kiteconnect' not in sys.modules, 'module imported kiteconnect at load time'\n"
        "assert z.ZerodhaSource is not None\n"
        "print('clean')\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=60
    )
    assert proc.returncode == 0, f"subprocess failed: {proc.stderr}"
    assert "clean" in proc.stdout

def test_missing_kiteconnect_raises_actionable_error(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A None entry in sys.modules makes `import kiteconnect` raise, so this test is
    # correct whether or not the optional package happens to be installed.
    monkeypatch.setitem(sys.modules, "kiteconnect", None)
    source = ZerodhaSource(cache_root=tmp_path)

    with pytest.raises(KiteConnectNotInstalledError) as excinfo:
        source.fetch_ohlcv(["RELIANCE.NS"], datetime(2024, 1, 1), datetime(2024, 1, 31))

    message = str(excinfo.value)
    assert "kiteconnect" in message
    assert "uv" in message, "error must tell the caller how to install it"
    # Both bases matter: ImportError is the honest type for a missing optional dep,
    # NotImplementedError keeps the "source unusable here" contract callers rely on.
    assert isinstance(excinfo.value, ImportError)
    assert isinstance(excinfo.value, NotImplementedError)


def test_missing_credentials_error_names_vars_without_values(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    stub = types.ModuleType("kiteconnect")
    stub.KiteConnect = FakeKite  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "kiteconnect", stub)
    monkeypatch.delenv("KITE_API_KEY", raising=False)
    monkeypatch.delenv("KITE_ACCESS_TOKEN", raising=False)

    source = ZerodhaSource(cache_root=tmp_path)
    with pytest.raises(ZerodhaCredentialsError) as excinfo:
        source.fetch_ohlcv(["RELIANCE.NS"], datetime(2024, 1, 1), datetime(2024, 1, 31))

    assert "KITE_API_KEY" in str(excinfo.value)
    assert "KITE_ACCESS_TOKEN" in str(excinfo.value)
    assert isinstance(excinfo.value, ZerodhaUnavailableError)


def test_mint_access_token_requires_secret_and_request_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub = types.ModuleType("kiteconnect")
    stub.KiteConnect = FakeKite  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "kiteconnect", stub)
    monkeypatch.setenv("KITE_API_KEY", "env-key")
    monkeypatch.delenv("KITE_API_SECRET", raising=False)
    monkeypatch.delenv("KITE_REQUEST_TOKEN", raising=False)

    with pytest.raises(ZerodhaCredentialsError) as excinfo:
        ZerodhaSource.mint_access_token()

    assert "KITE_API_SECRET" in str(excinfo.value)
    assert "KITE_REQUEST_TOKEN" in str(excinfo.value)
    assert "env-key" not in str(excinfo.value), "credential values must never be echoed"


def test_mint_access_token_exchanges_request_token(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    class LoginKite:
        def __init__(self, api_key: str) -> None:
            seen["api_key"] = api_key

        def generate_session(self, request_token: str, api_secret: str) -> dict[str, Any]:
            seen["request_token"] = request_token
            seen["api_secret"] = api_secret
            return {"access_token": "fresh-access-token", "user_id": "AB1234"}

    stub = types.ModuleType("kiteconnect")
    stub.KiteConnect = LoginKite  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "kiteconnect", stub)
    monkeypatch.setenv("KITE_API_KEY", "env-key")
    monkeypatch.setenv("KITE_API_SECRET", "env-secret")
    monkeypatch.setenv("KITE_REQUEST_TOKEN", "env-request-token")

    assert ZerodhaSource.mint_access_token() == "fresh-access-token"
    assert seen == {
        "api_key": "env-key",
        "request_token": "env-request-token",
        "api_secret": "env-secret",
    }


def test_fetch_never_mints_a_token(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """A fetch must not spend the single-use request token behind the caller's back."""

    class NoLoginKite(FakeKite):
        def generate_session(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            raise AssertionError("fetch_ohlcv must never call generate_session")

    stub = types.ModuleType("kiteconnect")
    stub.KiteConnect = lambda api_key: NoLoginKite()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "kiteconnect", stub)
    monkeypatch.setenv("KITE_API_KEY", "env-key")
    monkeypatch.setenv("KITE_API_SECRET", "env-secret")
    monkeypatch.setenv("KITE_REQUEST_TOKEN", "env-request-token")
    monkeypatch.delenv("KITE_ACCESS_TOKEN", raising=False)

    source = ZerodhaSource(cache_root=tmp_path, min_request_interval=0.0)
    with pytest.raises(ZerodhaCredentialsError) as excinfo:
        source.fetch_ohlcv(["RELIANCE.NS"], datetime(2024, 1, 1), datetime(2024, 1, 31))
    assert "KITE_ACCESS_TOKEN" in str(excinfo.value)


def test_secrets_never_appear_in_repr(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KITE_API_KEY", "key-should-never-be-printed")
    monkeypatch.setenv("KITE_ACCESS_TOKEN", "token-should-never-be-printed")
    text = repr(ZerodhaSource())
    assert "should-never-be-printed" not in text
    assert "api_key_set=True" in text and "access_token_set=True" in text


def test_client_built_from_environment(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    class RecordingKite(FakeKite):
        def __init__(self, api_key: str) -> None:
            super().__init__()
            seen["api_key"] = api_key

        def set_access_token(self, access_token: str) -> None:
            seen["access_token"] = access_token

    stub = types.ModuleType("kiteconnect")
    stub.KiteConnect = RecordingKite  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "kiteconnect", stub)
    monkeypatch.setenv("KITE_API_KEY", "env-key")
    monkeypatch.setenv("KITE_ACCESS_TOKEN", "env-token")

    source = ZerodhaSource(cache_root=tmp_path, min_request_interval=0.0)
    df = source.fetch_ohlcv(["RELIANCE.NS"], datetime(2024, 1, 1), datetime(2024, 1, 8))

    assert seen == {"api_key": "env-key", "access_token": "env-token"}
    assert not df.is_empty()


# ---------------------------------------------------------------------------
# Symbol mapping
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("ticker", "expected"),
    [
        ("RELIANCE.NS", "RELIANCE"),
        ("TCS.NS", "TCS"),
        ("reliance.ns", "RELIANCE"),
        ("RELIANCE", "RELIANCE"),  # already bare → unchanged
        ("M&M.NS", "M&M"),
        ("BAJAJ-AUTO.NS", "BAJAJ-AUTO"),
    ],
)
def test_to_kite_symbol(ticker: str, expected: str) -> None:
    assert ZerodhaSource.to_kite_symbol(ticker) == expected


@pytest.mark.parametrize(
    ("symbol", "expected"),
    [
        ("RELIANCE", "RELIANCE.NS"),
        ("TCS", "TCS.NS"),
        ("RELIANCE.NS", "RELIANCE.NS"),  # idempotent
        ("m&m", "M&M.NS"),
    ],
)
def test_to_project_ticker(symbol: str, expected: str) -> None:
    assert ZerodhaSource.to_project_ticker(symbol) == expected


def test_symbol_mapping_round_trips() -> None:
    for ticker in ("RELIANCE.NS", "TCS.NS", "BAJAJ-AUTO.NS"):
        assert ZerodhaSource.to_project_ticker(ZerodhaSource.to_kite_symbol(ticker)) == ticker


# ---------------------------------------------------------------------------
# Instrument token resolution + caching
# ---------------------------------------------------------------------------


def test_resolves_instrument_token_from_dump(tmp_path: Any) -> None:
    client = FakeKite()
    source = _source(tmp_path, client)

    assert source._resolve_token("RELIANCE.NS") == 738561
    assert source._resolve_token("TCS.NS") == 2953217
    # Derivatives and indices must not be resolvable as cash equities.
    assert source._resolve_token("NIFTY 50.NS") is None
    assert source._resolve_token("RELIANCE24DECFUT.NS") is None
    assert source._resolve_token("NOTLISTED.NS") is None


def test_instrument_dump_fetched_once_per_instance(tmp_path: Any) -> None:
    client = FakeKite()
    source = _source(tmp_path, client)

    for _ in range(3):
        source._resolve_token("RELIANCE.NS")
        source._resolve_token("TCS.NS")

    assert client.instrument_calls == ["NSE"], "instrument dump must be memoised in memory"


def test_instrument_cache_persists_across_instances(tmp_path: Any) -> None:
    first = FakeKite()
    _source(tmp_path, first)._resolve_token("RELIANCE.NS")
    assert len(first.instrument_calls) == 1

    second = FakeKite()
    assert _source(tmp_path, second)._resolve_token("RELIANCE.NS") == 738561
    assert second.instrument_calls == [], "a same-day dump on disk must be reused"


def test_unresolvable_ticker_is_skipped_not_fatal(tmp_path: Any) -> None:
    source = _source(tmp_path, FakeKite())
    df = source.fetch_ohlcv(
        ["NOTLISTED.NS", "RELIANCE.NS"], datetime(2024, 1, 1), datetime(2024, 1, 15)
    )
    assert set(df["ticker"].to_list()) == {"RELIANCE.NS"}


# ---------------------------------------------------------------------------
# Date-range chunking
# ---------------------------------------------------------------------------


def test_chunk_ranges_three_chunks_contiguous(tmp_path: Any) -> None:
    source = _source(tmp_path, max_days_per_request={"day": 400})
    first_day, last_day = date(2020, 1, 1), date(2022, 12, 31)  # 1096 days → 400+400+296

    chunks = source._chunk_ranges(first_day, last_day, "day")

    assert len(chunks) == 3
    assert chunks[0][0] == first_day
    assert chunks[-1][1] == last_day
    for (_, prev_end), (next_start, _) in zip(chunks, chunks[1:], strict=False):
        assert next_start == prev_end + timedelta(days=1), "chunks must not overlap or gap"
    for chunk_start, chunk_end in chunks:
        span = (chunk_end - chunk_start).days + 1
        assert 1 <= span <= 400, f"chunk of {span} days exceeds Kite's cap"
    assert sum((e - s).days + 1 for s, e in chunks) == (last_day - first_day).days + 1


def test_chunk_boundaries_exact(tmp_path: Any) -> None:
    source = _source(tmp_path, max_days_per_request={"day": 400})
    chunks = source._chunk_ranges(date(2020, 1, 1), date(2022, 12, 31), "day")
    assert chunks == [
        (date(2020, 1, 1), date(2021, 2, 3)),
        (date(2021, 2, 4), date(2022, 3, 10)),
        (date(2022, 3, 11), date(2022, 12, 31)),
    ]


def test_range_shorter_than_cap_is_one_chunk(tmp_path: Any) -> None:
    source = _source(tmp_path, max_days_per_request={"day": 400})
    assert source._chunk_ranges(date(2024, 1, 1), date(2024, 1, 1), "day") == [
        (date(2024, 1, 1), date(2024, 1, 1))
    ]


def test_fetch_issues_one_request_per_chunk_and_stitches(tmp_path: Any) -> None:
    client = FakeKite()
    source = _source(tmp_path, client, max_days_per_request={"day": 400})

    df = source.fetch_ohlcv(["RELIANCE.NS"], datetime(2020, 1, 1), datetime(2023, 1, 1))

    assert len(client.historical_calls) == 3
    assert [c["interval"] for c in client.historical_calls] == ["day"] * 3
    assert client.historical_calls[0]["from_date"] == date(2020, 1, 1)
    assert client.historical_calls[-1]["to_date"] == date(2022, 12, 31)
    assert df["date"].min() == datetime(2020, 1, 1)
    assert df["date"].max() == datetime(2022, 12, 30)  # last weekday in range


@pytest.mark.parametrize(
    ("interval", "kite_interval"),
    [("1d", "day"), ("1h", "60minute"), ("15m", "15minute")],
)
def test_interval_mapping(tmp_path: Any, interval: str, kite_interval: str) -> None:
    client = FakeKite()
    source = _source(tmp_path, client)
    source.fetch_ohlcv(["RELIANCE.NS"], datetime(2024, 1, 1), datetime(2024, 1, 5), interval)  # type: ignore[arg-type]
    assert client.historical_calls[0]["interval"] == kite_interval


def test_end_date_is_exclusive_like_yfinance(tmp_path: Any) -> None:
    client = FakeKite()
    source = _source(tmp_path, client)
    df = source.fetch_ohlcv(["RELIANCE.NS"], datetime(2024, 1, 1), datetime(2024, 1, 10))

    assert client.historical_calls[0]["to_date"] == date(2024, 1, 9)
    assert df["date"].max() == datetime(2024, 1, 9)


def test_empty_range_returns_typed_empty_frame(tmp_path: Any) -> None:
    client = FakeKite()
    source = _source(tmp_path, client)
    df = source.fetch_ohlcv(["RELIANCE.NS"], datetime(2024, 1, 5), datetime(2024, 1, 5))

    assert df.is_empty()
    assert dict(df.schema) == OUTPUT_SCHEMA
    assert client.historical_calls == []


# ---------------------------------------------------------------------------
# Throttling and retries
# ---------------------------------------------------------------------------


def test_throttle_spaces_requests(tmp_path: Any) -> None:
    clock = FakeClock()
    client = FakeKite()
    source = ZerodhaSource(
        cache_root=tmp_path,
        client=client,
        min_request_interval=1.0 / 3.0,
        sleep=clock.sleep,
        clock=clock,
        max_days_per_request={"day": 400},
    )

    source.fetch_ohlcv(["RELIANCE.NS"], datetime(2020, 1, 1), datetime(2023, 1, 1))

    assert len(client.historical_calls) == 3
    # 4 requests in total (1 instrument dump + 3 candle chunks); the first goes
    # straight through and each subsequent one waits out the remaining interval.
    assert len(clock.sleeps) == 3
    assert all(abs(s - 1.0 / 3.0) < 1e-9 for s in clock.sleeps)


def test_retries_transient_network_error_with_backoff(tmp_path: Any) -> None:
    clock = FakeClock()
    client = FakeKite(
        errors=[_named_error("NetworkException"), _named_error("NetworkException"), None]
    )
    source = ZerodhaSource(
        cache_root=tmp_path,
        client=client,
        min_request_interval=0.0,
        backoff_base=1.0,
        sleep=clock.sleep,
        clock=clock,
    )

    df = source.fetch_ohlcv(["RELIANCE.NS"], datetime(2024, 1, 1), datetime(2024, 1, 15))

    assert len(client.historical_calls) == 3
    assert clock.sleeps == [1.0, 2.0], "exponential backoff, and no real sleeping"
    assert not df.is_empty()


def test_retries_on_429(tmp_path: Any) -> None:
    clock = FakeClock()
    client = FakeKite(errors=[_named_error("KiteException", "Too many requests", code=429), None])
    source = ZerodhaSource(
        cache_root=tmp_path, client=client, min_request_interval=0.0, sleep=clock.sleep, clock=clock
    )

    df = source.fetch_ohlcv(["RELIANCE.NS"], datetime(2024, 1, 1), datetime(2024, 1, 15))

    assert len(client.historical_calls) == 2
    assert not df.is_empty()


def test_exhausted_retries_drops_ticker(tmp_path: Any) -> None:
    clock = FakeClock()
    client = FakeKite(errors=[_named_error("NetworkException")] * 3)
    source = ZerodhaSource(
        cache_root=tmp_path,
        client=client,
        min_request_interval=0.0,
        max_retries=3,
        sleep=clock.sleep,
        clock=clock,
    )

    df = source.fetch_ohlcv(["RELIANCE.NS"], datetime(2024, 1, 1), datetime(2024, 1, 15))

    assert len(client.historical_calls) == 3
    assert df.is_empty() and dict(df.schema) == OUTPUT_SCHEMA


def test_non_retryable_error_is_not_retried(tmp_path: Any) -> None:
    clock = FakeClock()
    client = FakeKite(errors=[_named_error("InputException", "invalid token")])
    source = ZerodhaSource(
        cache_root=tmp_path, client=client, min_request_interval=0.0, sleep=clock.sleep, clock=clock
    )

    df = source.fetch_ohlcv(["RELIANCE.NS"], datetime(2024, 1, 1), datetime(2024, 1, 15))

    assert len(client.historical_calls) == 1
    assert clock.sleeps == []
    assert df.is_empty()


def test_auth_error_propagates(tmp_path: Any) -> None:
    """An expired token must fail loudly, not silently empty every ticker."""
    client = FakeKite(errors=[_named_error("TokenException", "token expired")])
    source = _source(tmp_path, client)

    with pytest.raises(ZerodhaAuthError):
        source.fetch_ohlcv(
            ["RELIANCE.NS", "TCS.NS"], datetime(2024, 1, 1), datetime(2024, 1, 15)
        )
    assert len(client.historical_calls) == 1, "must not march on to the next ticker"


# ---------------------------------------------------------------------------
# Output shaping: schema, tz, ordering
# ---------------------------------------------------------------------------


def _yfinance_reference_frame() -> pl.DataFrame:
    """Run the real YFinanceSource shaping over a synthetic yf.download payload."""
    import pandas as pd

    from trader.data.sources.yfinance_source import YFinanceSource

    index = pd.DatetimeIndex(
        [datetime(2024, 1, 1), datetime(2024, 1, 2), datetime(2024, 1, 3)], name="Date"
    )
    columns = pd.MultiIndex.from_product(
        [["Close", "High", "Low", "Open", "Volume"], ["RELIANCE.NS"]]
    )
    raw = pd.DataFrame(
        [
            [104.0, 105.0, 99.0, 100.0, 1000],
            [105.0, 106.0, 100.0, 104.0, 1100],
            [106.0, 107.0, 101.0, 105.0, 1200],
        ],
        index=index,
        columns=columns,
    )
    return YFinanceSource()._to_polars(raw, "RELIANCE.NS")


def test_schema_matches_yfinance_source(tmp_path: Any) -> None:
    """Both sources feed the same panel; divergence here is silent train/serve skew."""
    reference = _yfinance_reference_frame()
    zerodha = _source(tmp_path, FakeKite()).fetch_ohlcv(
        ["RELIANCE.NS"], datetime(2024, 1, 1), datetime(2024, 1, 15)
    )

    assert zerodha.columns == reference.columns
    assert dict(zerodha.schema) == dict(reference.schema)
    assert dict(zerodha.schema) == OUTPUT_SCHEMA


def test_source_column_marks_provenance(tmp_path: Any) -> None:
    df = _source(tmp_path, FakeKite()).fetch_ohlcv(
        ["RELIANCE.NS"], datetime(2024, 1, 1), datetime(2024, 1, 15)
    )
    assert set(df["source"].to_list()) == {"zerodha"}
    assert set(df["ticker"].to_list()) == {"RELIANCE.NS"}, "output keeps the .NS convention"


def test_daily_timestamps_normalised_to_naive_midnight(tmp_path: Any) -> None:
    class MiddayKite(FakeKite):
        def historical_data(self, **kwargs: Any) -> list[dict[str, Any]]:  # type: ignore[override]
            self.historical_calls.append(kwargs)
            return [
                {
                    "date": datetime(2024, 1, 2, 9, 15, tzinfo=IST),
                    "open": 100.0,
                    "high": 105.0,
                    "low": 99.0,
                    "close": 104.0,
                    "volume": 1000,
                }
            ]

    df = _source(tmp_path, MiddayKite()).fetch_ohlcv(
        ["RELIANCE.NS"], datetime(2024, 1, 1), datetime(2024, 1, 15)
    )

    stamp = df["date"].to_list()[0]
    assert stamp == datetime(2024, 1, 2)
    assert stamp.tzinfo is None
    assert df.schema["date"] == pl.Datetime("ns")


def test_utc_timestamps_converted_to_ist_wall_clock(tmp_path: Any) -> None:
    """UTC in → IST wall clock out; a naive UTC drop would move bars a day earlier."""

    class UtcKite(FakeKite):
        def historical_data(self, **kwargs: Any) -> list[dict[str, Any]]:  # type: ignore[override]
            self.historical_calls.append(kwargs)
            return [
                {
                    # 03:45Z == 09:15 IST, the NSE open on 2 Jan.
                    "date": datetime(2024, 1, 2, 3, 45, tzinfo=UTC),
                    "open": 100.0,
                    "high": 105.0,
                    "low": 99.0,
                    "close": 104.0,
                    "volume": 1000,
                }
            ]

    df = _source(tmp_path, UtcKite()).fetch_ohlcv(
        ["RELIANCE.NS"], datetime(2024, 1, 1), datetime(2024, 1, 15), interval="15m"
    )

    stamp = df["date"].to_list()[0]
    assert stamp == datetime(2024, 1, 2, 9, 15)
    assert stamp.tzinfo is None


def test_rows_sorted_and_deduplicated(tmp_path: Any) -> None:
    """Overlapping chunk boundaries must not duplicate a (date, ticker) row."""

    class OverlappingKite(FakeKite):
        def historical_data(self, **kwargs: Any) -> list[dict[str, Any]]:  # type: ignore[override]
            self.historical_calls.append(kwargs)
            # Every chunk returns the same three candles, out of order.
            candles = _candles(date(2024, 1, 1), date(2024, 1, 3))
            return list(reversed(candles))

    source = _source(tmp_path, OverlappingKite(), max_days_per_request={"day": 5})
    df = source.fetch_ohlcv(
        ["TCS.NS", "RELIANCE.NS"], datetime(2024, 1, 1), datetime(2024, 1, 16)
    )

    keys = df.select(["date", "ticker"])
    assert keys.n_unique() == len(df), "duplicate (date, ticker) rows leaked through"
    assert df.equals(df.sort(["ticker", "date"])), "row order must match the yfinance source"


def test_adj_close_mirrors_close(tmp_path: Any) -> None:
    """Kite is unadjusted and has no factor; adj_close exists only to hold the schema."""
    df = _source(tmp_path, FakeKite()).fetch_ohlcv(
        ["RELIANCE.NS"], datetime(2024, 1, 1), datetime(2024, 1, 15)
    )
    assert df["adj_close"].to_list() == df["close"].to_list()


# ---------------------------------------------------------------------------
# Corporate actions stay unimplemented on purpose
# ---------------------------------------------------------------------------


def test_fetch_corporate_actions_still_raises() -> None:
    source = ZerodhaSource()
    with pytest.raises(NotImplementedError) as excinfo:
        source.fetch_corporate_actions(["RELIANCE.NS"])
    assert "YFinanceSource" in str(excinfo.value)


def test_still_satisfies_market_data_source_protocol() -> None:
    from trader.data.sources.base import MarketDataSource

    assert isinstance(ZerodhaSource(), MarketDataSource)
