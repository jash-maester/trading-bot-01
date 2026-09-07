"""Offline tests for the NSE flow/delivery/deal parsers.

Every test reads a fixture from ``tests/fixtures/nse/`` (see its README for
provenance). Nothing here opens a socket: sources are constructed with
``offline=True`` or with a stub session, so a network outage cannot turn a
parser bug into a skipped test.
"""
from __future__ import annotations

import shutil
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from trader.data.sources.nse_flows import (
    DealsSource,
    DeliverySource,
    FlowsSource,
    MTOLayout,
    MTOLayoutError,
    NSEClient,
    NSEFetchError,
    NSENotFound,
    aggregate_deals,
    detect_mto_layout,
    nse_symbol_to_ticker,
    parse_deals_csv,
    parse_fiidii_json,
    parse_flows_backfill,
    parse_mto,
    parse_sec_bhavdata,
    pivot_flows,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "nse"


def _text(name: str) -> str:
    return (FIXTURES / name).read_text()


# ── Symbol mapping ───────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("symbol", "expected"),
    [
        ("RELIANCE", "RELIANCE.NS"),
        ("  infy ", "INFY.NS"),
        # Renames: a 2015 MTO file spells these the old way, and the current
        # universe spells them the new way. Without the ledger the join drops.
        ("LTIM", "LTM.NS"),
        ("TATAMOTORS", "TMPV.NS"),
        # Series suffix, as NSE writes it for trade-to-trade names.
        ("STLTECH-BE", "STLTECH.NS"),
    ],
)
def test_nse_symbol_to_ticker(symbol: str, expected: str) -> None:
    assert nse_symbol_to_ticker(symbol) == expected


# ── MTO: the two real layouts ────────────────────────────────────────────────


def test_mto_2015_layout_is_the_settlement_variant() -> None:
    text = _text("MTO_02012015.DAT")
    assert detect_mto_layout(text) is MTOLayout.SETTLEMENT

    frame = parse_mto(text, name="MTO_02012015.DAT")
    assert frame.height == 8
    # Trade date comes from the file's own header, not from the filename.
    assert frame["date"].unique().to_list() == [date(2015, 1, 2)]

    reliance = frame.filter(pl.col("ticker") == "RELIANCE.NS").row(0, named=True)
    assert reliance["traded_qty"] == 1675827
    assert reliance["deliverable_qty"] == 1048543
    assert reliance["delivery_pct"] == pytest.approx(1048543 / 1675827)
    # Published percentage was 62.57; the recomputed fraction must agree.
    assert reliance["delivery_pct"] == pytest.approx(0.6257, abs=1e-4)

    # The 2015 file still calls these by their pre-rename symbols.
    assert "TMPV.NS" in frame["ticker"].to_list()


def test_mto_2026_layout_has_no_settlement_fields() -> None:
    text = _text("MTO_04092026.DAT")
    assert detect_mto_layout(text) is MTOLayout.NO_SETTLEMENT

    frame = parse_mto(text, name="MTO_04092026.DAT")
    assert frame["date"].unique().to_list() == [date(2026, 9, 4)]
    tickers = set(frame["ticker"].to_list())
    assert {"LTM.NS", "TMPV.NS", "RELIANCE.NS"} <= tickers


def test_mto_unknown_layout_fails_loudly_and_names_the_layouts() -> None:
    with pytest.raises(MTOLayoutError) as excinfo:
        parse_mto(_text("MTO_unknown_layout.DAT"), name="MTO_unknown_layout.DAT")
    message = str(excinfo.value)
    assert "MTO_unknown_layout.DAT" in message
    assert "unknown MTO column layout" in message
    # It must say which layouts it *does* know, so the fix is obvious.
    assert MTOLayout.SETTLEMENT.value in message
    assert MTOLayout.NO_SETTLEMENT.value in message


def test_mto_record_arity_change_is_refused() -> None:
    """A file with the known header but a short record must not be guessed at."""
    good = _text("MTO_04092026.DAT").splitlines()
    broken = "\n".join([*good[:4], "20,1,RELIANCE,EQ,13031534"])
    with pytest.raises(MTOLayoutError, match="fields per record"):
        parse_mto(broken, name="broken.DAT")


def test_mto_zero_traded_quantity_is_null_not_zero() -> None:
    good = _text("MTO_04092026.DAT").splitlines()
    with_zero = "\n".join([*good[:4], "20,1,NOTRADE,EQ,0,0,0.00"])
    frame = parse_mto(with_zero, name="zero.DAT")
    assert frame["delivery_pct"].to_list() == [None]


# ── Bhavdata fallback ────────────────────────────────────────────────────────


def test_bhavdata_dash_is_null_not_zero() -> None:
    frame = parse_sec_bhavdata(_text("sec_bhavdata_full_04092026.csv"))
    stltech = frame.filter(pl.col("ticker") == "STLTECH.NS").row(0, named=True)
    # NSE publishes "-" for a BE-series delivery figure. That is missing, not 0.
    assert stltech["series"] == "BE"
    assert stltech["deliverable_qty"] is None
    assert stltech["delivery_pct"] is None

    reliance = frame.filter(pl.col("ticker") == "RELIANCE.NS").row(0, named=True)
    assert reliance["delivery_pct"] == pytest.approx(8600946 / 13031534)


def test_bhavdata_and_mto_agree_on_the_same_day() -> None:
    """Two independent NSE publications of the same figure must match."""
    mto = parse_mto(_text("MTO_04092026.DAT")).filter(pl.col("series") == "EQ")
    bhav = parse_sec_bhavdata(_text("sec_bhavdata_full_04092026.csv")).filter(
        pl.col("series") == "EQ"
    )
    merged = mto.join(bhav, on=["date", "ticker"], how="inner", suffix="_bhav")
    assert merged.height >= 6
    for row in merged.iter_rows(named=True):
        assert row["traded_qty"] == row["traded_qty_bhav"], row["ticker"]
        assert row["delivery_pct"] == pytest.approx(row["delivery_pct_bhav"]), row["ticker"]


# ── Delivery dedupe ──────────────────────────────────────────────────────────


def test_delivery_dedupe_keeps_one_cash_row_per_symbol() -> None:
    frame = parse_mto(_text("MTO_02012015.DAT"))
    # SBIN appears three times in the fixture: EQ, N1, N3.
    assert frame.filter(pl.col("ticker") == "SBIN.NS").height == 3

    deduped = DeliverySource._dedupe(frame)
    sbin = deduped.filter(pl.col("ticker") == "SBIN.NS")
    assert sbin.height == 1
    assert sbin["series"].to_list() == ["EQ"]
    assert sbin["traded_qty"].to_list() == [9935094]
    assert deduped.select(["date", "ticker"]).is_duplicated().any() is False


def test_delivery_dedupe_drops_non_cash_series() -> None:
    frame = parse_mto(_text("MTO_04092026.DAT"))
    deduped = DeliverySource._dedupe(frame)
    # 0MOFSL27 is an N3-series instrument, not the cash equity the panel prices.
    assert "0MOFSL27.NS" not in set(deduped["ticker"].to_list())


# ── Delivery source: cache is read offline ───────────────────────────────────


def test_delivery_source_reads_the_cache_with_no_network(tmp_path: Path) -> None:
    cache = tmp_path / "mto"
    cache.mkdir(parents=True)
    shutil.copy(FIXTURES / "MTO_04092026.DAT", cache / "MTO_04092026.DAT")

    source = DeliverySource(cache_root=tmp_path, offline=True)
    frame = source.fetch(date(2026, 9, 4), date(2026, 9, 4))
    assert frame.height > 0
    assert set(frame.columns) == {
        "date",
        "ticker",
        "series",
        "traded_qty",
        "deliverable_qty",
        "delivery_pct",
        "turnover",
        "avg_price",
        "n_trades",
        "source",
    }
    assert frame["source"].unique().to_list() == ["mto"]


def test_delivery_source_missing_day_returns_no_rows(tmp_path: Path) -> None:
    source = DeliverySource(cache_root=tmp_path, offline=True)
    # Nothing cached and no network: an absent day yields an empty frame with
    # the right schema, never a fabricated row.
    frame = source.fetch(date(2026, 9, 1), date(2026, 9, 4))
    assert frame.height == 0
    assert "delivery_pct" in frame.columns


def test_delivery_source_bhavdata_backend(tmp_path: Path) -> None:
    cache = tmp_path / "bhavdata"
    cache.mkdir(parents=True)
    shutil.copy(
        FIXTURES / "sec_bhavdata_full_04092026.csv",
        cache / "sec_bhavdata_full_04092026.csv",
    )
    source = DeliverySource(cache_root=tmp_path, offline=True, backend="bhavdata")
    frame = source.fetch(date(2026, 9, 4), date(2026, 9, 4))
    assert frame["source"].unique().to_list() == ["bhavdata"]
    # STLTECH is BE series, so it survives the cash-series filter with a null pct.
    stltech = frame.filter(pl.col("ticker") == "STLTECH.NS")
    assert stltech.height == 1
    assert stltech["delivery_pct"].to_list() == [None]


def test_delivery_source_rejects_unknown_backend(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="backend must be"):
        DeliverySource(cache_root=tmp_path, backend="guess")


# ── FII / DII flows ──────────────────────────────────────────────────────────


def test_parse_fiidii_json() -> None:
    long = parse_fiidii_json(_text("fiidii_latest.json"))
    assert set(long["category"].to_list()) == {"fii", "dii"}
    wide = pivot_flows(long)
    assert wide.height == 1
    row = wide.row(0, named=True)
    assert row["date"] == date(2026, 9, 4)
    assert row["fii_net_cr"] == pytest.approx(-3111.94)
    assert row["dii_net_cr"] == pytest.approx(8930.12)
    assert row["fii_buy_cr"] == pytest.approx(13857.58)
    assert row["dii_sell_cr"] == pytest.approx(10324.07)


def test_parse_flows_backfill_normalises_category_spellings() -> None:
    long = parse_flows_backfill(_text("flows_backfill.csv"))
    # The ledger uses both "FII" and "FII/FPI"; both mean the same series.
    assert set(long["category"].to_list()) == {"fii", "dii"}
    wide = pivot_flows(long)
    assert wide["date"].to_list() == [date(2026, 9, 1), date(2026, 9, 2), date(2026, 9, 3)]
    assert wide["fii_net_cr"].to_list() == pytest.approx([-1499.75, 1000.0, -3000.0])
    assert wide["dii_net_cr"].to_list() == pytest.approx([3000.0, -500.0, 5000.0])


def test_pivot_flows_leaves_a_one_sided_day_null() -> None:
    long = pl.DataFrame(
        {
            "date": [date(2026, 9, 1)],
            "category": ["fii"],
            "buy_value_cr": [1.0],
            "sell_value_cr": [2.0],
            "net_value_cr": [-1.0],
        }
    )
    wide = pivot_flows(long)
    assert wide["dii_net_cr"].to_list() == [None]


def test_flows_source_merges_ledger_and_daily_snapshots(tmp_path: Path) -> None:
    cache = tmp_path / "flows"
    cache.mkdir(parents=True)
    shutil.copy(FIXTURES / "flows_backfill.csv", cache / FlowsSource.LEDGER_NAME)
    shutil.copy(FIXTURES / "fiidii_latest.json", cache / "fiidii_2026-09-04.json")

    source = FlowsSource(cache_root=tmp_path, offline=True)
    frame = source.fetch(date(2026, 9, 1), date(2026, 9, 4))
    assert frame["date"].to_list() == [
        date(2026, 9, 1),
        date(2026, 9, 2),
        date(2026, 9, 3),
        date(2026, 9, 4),
    ]
    assert frame["fii_net_cr"].to_list()[-1] == pytest.approx(-3111.94)


# ── Bulk / block deals ───────────────────────────────────────────────────────


def test_parse_bulk_deals_handles_quoted_client_names() -> None:
    rows = parse_deals_csv(_text("bulk.csv"), deal_type="bulk")
    assert rows.height == 5
    graviton = rows.filter(pl.col("client_name").str.contains("GRAVITON")).row(0, named=True)
    # The client name contains a comma inside quotes; a naive split breaks here.
    assert graviton["client_name"] == "GRAVITON RESEARCH CAPITAL LLP, FUND A"
    assert graviton["side"] == "SELL"
    assert graviton["ticker"] == "ANTELOPUS.NS"
    assert graviton["value_inr"] == pytest.approx(217937 * 1000.63)
    assert set(rows["deal_type"].to_list()) == {"bulk"}
    assert rows["date"].unique().to_list() == [date(2026, 9, 4)]


def test_parse_block_deals() -> None:
    rows = parse_deals_csv(_text("block.csv"), deal_type="block")
    assert rows.height == 2
    assert set(rows["ticker"].to_list()) == {"NXST.NS"}
    assert sorted(rows["side"].to_list()) == ["BUY", "SELL"]


def test_deal_header_change_is_refused() -> None:
    with pytest.raises(ValueError, match="no column mapping"):
        parse_deals_csv("Date,Symbol,Something Else\n04-SEP-2026,X,1\n", deal_type="bulk")


def test_aggregate_deals_nets_a_crossed_trade_to_zero() -> None:
    rows = parse_deals_csv(_text("bulk.csv"), deal_type="bulk")
    agg = aggregate_deals(rows)
    antelopus = agg.filter(pl.col("ticker") == "ANTELOPUS.NS").row(0, named=True)
    # Same quantity bought and sold at the same price: two deals, zero net.
    assert antelopus["n_deals"] == 2
    assert antelopus["net_value_inr"] == pytest.approx(0.0)
    assert antelopus["buy_value_inr"] == pytest.approx(217937 * 1000.63)

    ltm = agg.filter(pl.col("ticker") == "LTM.NS").row(0, named=True)
    assert ltm["n_deals"] == 1
    assert ltm["net_value_inr"] == pytest.approx(600000 * 4554.00)


def test_deals_source_reads_per_date_snapshots(tmp_path: Path) -> None:
    cache = tmp_path / "deals"
    cache.mkdir(parents=True)
    shutil.copy(FIXTURES / "bulk.csv", cache / "bulk_2026-09-04.csv")
    shutil.copy(FIXTURES / "block.csv", cache / "block_2026-09-04.csv")

    source = DealsSource(cache_root=tmp_path, offline=True)
    assert source.observed_dates() == [date(2026, 9, 4)]

    frame = source.fetch(date(2026, 9, 1), date(2026, 9, 30))
    assert set(frame.columns) == {
        "date",
        "ticker",
        "n_deals",
        "buy_value_inr",
        "sell_value_inr",
        "net_value_inr",
    }
    # Bulk (4 distinct symbols) plus block (NXST) on one day.
    assert set(frame["ticker"].to_list()) == {
        "ABH.NS",
        "AHIMSA.NS",
        "ANTELOPUS.NS",
        "LTM.NS",
        "NXST.NS",
    }
    assert frame.select(["date", "ticker"]).is_duplicated().any() is False


def test_a_two_date_deals_file_is_not_double_counted(tmp_path: Path) -> None:
    """The doubling regression.

    The NSE bulk/block endpoint returns a multi-date window.  ``snapshot_latest``
    cached the WHOLE response under every date it contained, and ``fetch`` then
    parsed each cached file in full — so a 2-date response cached under 2 names
    counted every deal twice, in both ``n_deals`` and ``buy_value_inr``.
    """
    cache = tmp_path / "deals"
    cache.mkdir(parents=True)
    one_day = (FIXTURES / "bulk.csv").read_text()
    lines = one_day.strip().split("\n")
    header, body = lines[0], lines[1:]
    # Same deals, restamped to a second date, appended to the same file.
    second = [line.replace("04-SEP-2026", "03-SEP-2026") for line in body]
    assert second != body, "fixture date spelling changed; this test is inert"
    two_days = "\n".join([header, *body, *second]) + "\n"
    (cache / "bulk_2026-09-04.csv").write_text(two_days)
    (cache / "bulk_2026-09-03.csv").write_text(two_days)

    source = DealsSource(cache_root=tmp_path, offline=True)
    frame = source.fetch(date(2026, 9, 1), date(2026, 9, 30)).sort(["date", "ticker"])

    single = parse_deals_csv(one_day, deal_type="bulk")
    expected_n = int(single.height)
    expected_value = float(
        single.filter(pl.col("side") == "BUY")["value_inr"].sum()
    )
    for day in (date(2026, 9, 3), date(2026, 9, 4)):
        got = frame.filter(pl.col("date") == day)
        assert int(got["n_deals"].sum()) == expected_n, (
            f"{day}: n_deals {int(got['n_deals'].sum())} != {expected_n}; "
            f"the multi-date file was counted more than once"
        )
        assert float(got["buy_value_inr"].sum()) == pytest.approx(expected_value), day
    assert frame.select(["date", "ticker"]).is_duplicated().any() is False


def test_snapshot_latest_writes_only_the_rows_for_the_date_it_names(
    tmp_path: Path,
) -> None:
    """A cached file must contain only the date its filename claims."""
    one_day = (FIXTURES / "bulk.csv").read_text()
    lines = one_day.strip().split("\n")
    header, body = lines[0], lines[1:]
    second = [line.replace("04-SEP-2026", "03-SEP-2026") for line in body]
    two_days = "\n".join([header, *body, *second]) + "\n"

    class _Stub:
        def get_text(self, url: str) -> str:
            return two_days if "bulk" in url.lower() else header + "\n"

    source = DealsSource(cache_root=tmp_path, offline=False, client=_Stub())  # type: ignore[arg-type]
    written = source.snapshot_latest()
    assert date(2026, 9, 3) in written and date(2026, 9, 4) in written
    for day in (date(2026, 9, 3), date(2026, 9, 4)):
        text = (tmp_path / "deals" / f"bulk_{day.isoformat()}.csv").read_text()
        got = parse_deals_csv(text, deal_type="bulk")
        assert set(got["date"].to_list()) == {day}, (
            f"bulk_{day}.csv contains {sorted(set(got['date'].to_list()))}"
        )


def test_flows_source_snapshot_overrides_the_ledger_for_a_shared_date(
    tmp_path: Path,
) -> None:
    """End-to-end: the same date in BOTH artefacts resolves to the snapshot.

    This is the real path — `pivot_flows` can only apply precedence if the
    parsers tag their rows, so this fails if either `parse_flows_backfill` or
    `parse_fiidii_json` stops setting `source`.
    """
    cache = tmp_path / "flows"
    cache.mkdir(parents=True)
    ledger = (
        "date,category,buy_value_cr,sell_value_cr,net_value_cr\n"
        "2026-09-04,FII,10.0,110.0,-100.0\n"
        "2026-09-04,DII,1.0,2.0,-1.0\n"
    )
    (cache / FlowsSource.LEDGER_NAME).write_text(ledger)
    shutil.copy(FIXTURES / "fiidii_latest.json", cache / "fiidii_2026-09-04.json")

    snapshot_net = {
        r["category"]: r["net_value_cr"]
        for r in parse_fiidii_json(
            (FIXTURES / "fiidii_latest.json").read_text()
        ).iter_rows(named=True)
    }
    frame = FlowsSource(cache_root=tmp_path, offline=True).fetch(
        date(2026, 9, 4), date(2026, 9, 4)
    )
    assert frame.height == 1
    got = frame["fii_net_cr"].to_list()[0]
    assert got == pytest.approx(snapshot_net["fii"]), (
        f"ledger -100.0 and snapshot {snapshot_net['fii']} resolved to {got}; "
        f"the snapshot must win and the two must never be averaged"
    )
    assert got != pytest.approx((-100.0 + snapshot_net["fii"]) / 2.0)


def test_every_cache_write_records_a_provenance_line(tmp_path: Path) -> None:
    """A fetched file must leave a durable, citable trace.

    Every live-network number this module produced was previously
    unreproducible: nothing was committed and `data/raw/nse/` did not exist, so
    the HTTP statuses, byte counts and row counts cited a chat log and not the
    repo. The manifest is what a later reader cites instead.
    """
    one_day = (FIXTURES / "bulk.csv").read_text()

    class _Stub:
        def get_text(self, url: str) -> str:
            return one_day if "bulk" in url.lower() else one_day.split("\n")[0] + "\n"

    source = DealsSource(cache_root=tmp_path, offline=False, client=_Stub())  # type: ignore[arg-type]
    source.snapshot_latest()

    entries = source.cache.manifest()
    assert entries, "a live fetch wrote no manifest line"
    for entry in entries:
        assert entry["bytes"] > 0
        assert len(entry["sha256"]) == 64
        assert entry["fetched_at"]
        assert entry["name"]
    # The recorded hash must be the hash of what is actually on disk.
    import hashlib

    for entry in entries:
        path = source.cache.path(entry["name"])
        if path.exists():
            got = hashlib.sha256(path.read_bytes()).hexdigest()
            assert got == entry["sha256"], f"{entry['name']} manifest hash is stale"


def test_conflicting_flow_rows_are_resolved_by_precedence_not_averaged() -> None:
    """The ledger and the dated JSON snapshot can both carry one date.

    Averaging them returns a number that was never observed anywhere: -100.0
    and -3111.94 silently became -1605.97.  Precedence is explicit — the dated
    API snapshot wins over the hand-maintained ledger.
    """
    long = pl.DataFrame(
        {
            "date": [date(2026, 9, 4)] * 2,
            "category": ["fii", "fii"],
            "buy_value_cr": [10.0, 20.0],
            "sell_value_cr": [110.0, 30.0],
            "net_value_cr": [-100.0, -3111.94],
            "source": ["ledger", "snapshot"],
        }
    )
    wide = pivot_flows(long)
    assert wide.height == 1
    net = wide["fii_net_cr"].to_list()[0]
    assert net == pytest.approx(-3111.94), f"snapshot must win, got {net}"
    assert net != pytest.approx(-1605.97), "the two sources were averaged"


def test_identical_duplicate_flow_rows_collapse_without_conflict() -> None:
    long = pl.DataFrame(
        {
            "date": [date(2026, 9, 4)] * 2,
            "category": ["dii", "dii"],
            "buy_value_cr": [5.0, 5.0],
            "sell_value_cr": [3.0, 3.0],
            "net_value_cr": [2.0, 2.0],
            "source": ["ledger", "snapshot"],
        }
    )
    wide = pivot_flows(long)
    assert wide.height == 1
    assert wide["dii_net_cr"].to_list()[0] == pytest.approx(2.0)


def test_deals_source_ignores_dates_outside_the_range(tmp_path: Path) -> None:
    cache = tmp_path / "deals"
    cache.mkdir(parents=True)
    shutil.copy(FIXTURES / "bulk.csv", cache / "bulk_2026-09-04.csv")
    source = DealsSource(cache_root=tmp_path, offline=True)
    assert source.fetch(date(2026, 1, 1), date(2026, 1, 31)).height == 0


# ── HTTP client behaviour (stubbed session, no sockets) ──────────────────────


class _StubResponse:
    def __init__(self, status: int, text: str) -> None:
        self.status_code = status
        self.text = text


class _StubSession:
    """Replays a scripted list of responses/exceptions and records the URLs."""

    def __init__(self, script: list[Any]) -> None:
        self.script = list(script)
        self.urls: list[str] = []
        self.headers: dict[str, str] = {}

    def get(self, url: str, timeout: float | None = None) -> _StubResponse:
        self.urls.append(url)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        assert isinstance(item, _StubResponse)
        return item


@pytest.fixture(autouse=True)
def _no_sleeping(monkeypatch: pytest.MonkeyPatch) -> None:
    """Backoff and throttle sleeps must not slow the suite."""
    monkeypatch.setattr("trader.data.sources.nse_flows.time.sleep", lambda _s: None)


def test_client_returns_body_on_200() -> None:
    session = _StubSession([_StubResponse(200, "hello")])
    client = NSEClient(session=session, min_interval_s=0.0)
    assert client.get_text("https://example.test/f") == "hello"
    assert session.urls == ["https://example.test/f"]


def test_client_raises_not_found_on_404_without_retrying() -> None:
    session = _StubSession([_StubResponse(404, "")])
    client = NSEClient(session=session, min_interval_s=0.0)
    with pytest.raises(NSENotFound):
        client.get_text("https://example.test/holiday")
    assert len(session.urls) == 1


def test_client_retries_a_transport_error_then_succeeds() -> None:
    session = _StubSession([TimeoutError("boom"), _StubResponse(503, ""), _StubResponse(200, "ok")])
    client = NSEClient(session=session, min_interval_s=0.0, max_retries=3)
    assert client.get_text("https://example.test/f") == "ok"
    assert len(session.urls) == 3


def test_client_gives_up_after_max_retries() -> None:
    session = _StubSession([_StubResponse(503, "") for _ in range(3)])
    client = NSEClient(session=session, min_interval_s=0.0, max_retries=3)
    with pytest.raises(NSEFetchError, match="after 3 attempts"):
        client.get_text("https://example.test/f")


def test_cached_source_marks_a_404_day_missing_and_never_refetches(tmp_path: Path) -> None:
    session = _StubSession([_StubResponse(404, "")])
    client = NSEClient(session=session, min_interval_s=0.0)
    source = DeliverySource(cache_root=tmp_path, client=client)

    assert source.fetch_day(date(2026, 1, 26)) is None
    assert (tmp_path / "mto" / "MTO_26012026.DAT.missing").exists()
    # Second call must not issue a request — the stub would raise IndexError.
    assert source.fetch_day(date(2026, 1, 26)) is None
    assert len(session.urls) == 1


def test_cached_source_writes_then_reuses_the_file(tmp_path: Path) -> None:
    body = _text("MTO_04092026.DAT")
    session = _StubSession([_StubResponse(200, body)])
    client = NSEClient(session=session, min_interval_s=0.0)
    source = DeliverySource(cache_root=tmp_path, client=client)

    first = source.fetch_day(date(2026, 9, 4))
    assert first is not None and first.height == 8
    assert (tmp_path / "mto" / "MTO_04092026.DAT").exists()

    second = source.fetch_day(date(2026, 9, 4))
    assert second is not None and second.height == 8
    assert len(session.urls) == 1  # re-run is free


# ── turnover columns (added 2026-09-07) ──────────────────────────────────────


def test_bhavdata_parses_turnover_vwap_and_trade_count() -> None:
    """`sec_bhavdata_full` carried these all along and the parser discarded them.

    They matter because Kite's historical bars are OHLCV with no turnover, so
    nothing else in this repo knows the rupee value traded, and `avg_price` is
    NSE's own VWAP — genuinely absent from OHLC.
    """
    from trader.data.sources.nse_flows import parse_sec_bhavdata

    text = (
        "SYMBOL, SERIES, DATE1, PREV_CLOSE, OPEN_PRICE, HIGH_PRICE, LOW_PRICE, "
        "LAST_PRICE, CLOSE_PRICE, AVG_PRICE, TTL_TRD_QNTY, TURNOVER_LACS, "
        "NO_OF_TRADES, DELIV_QTY, DELIV_PER\n"
        "20MICRONS, EQ, 04-Sep-2026, 217.82, 218.91, 224.90, 216.50, 222.95, "
        "222.95, 221.03, 174812, 386.39, 4145, 82174, 47.01\n"
    )
    df = parse_sec_bhavdata(text, name="sec_bhavdata_full_04092026.csv")
    row = df.row(0, named=True)
    assert row["avg_price"] == pytest.approx(221.03)
    assert row["n_trades"] == 4145
    # Stored in RUPEES, not the lakhs NSE reports, so no consumer has to
    # remember the unit.
    assert row["turnover"] == pytest.approx(386.39 * 1e5)
    # turnover / qty reproduces NSE's own VWAP to within rounding, which is the
    # check that the unit conversion is right rather than merely consistent.
    assert row["turnover"] / row["traded_qty"] == pytest.approx(row["avg_price"], rel=0.01)


def test_a_dash_in_a_turnover_field_is_null_not_zero() -> None:
    """Zero turnover means "traded nothing"; a dash means "NSE published none"."""
    from trader.data.sources.nse_flows import parse_sec_bhavdata

    text = (
        "SYMBOL, SERIES, DATE1, AVG_PRICE, TTL_TRD_QNTY, TURNOVER_LACS, "
        "NO_OF_TRADES, DELIV_QTY\n"
        "FOO, BE, 04-Sep-2026, -, 100, -, -, -\n"
    )
    row = parse_sec_bhavdata(text, name="x.csv").row(0, named=True)
    assert row["turnover"] is None
    assert row["avg_price"] is None
    assert row["n_trades"] is None


def test_an_older_layout_without_turnover_columns_still_parses() -> None:
    """A file predating these columns must yield delivery data, not raise."""
    from trader.data.sources.nse_flows import parse_sec_bhavdata

    text = (
        "SYMBOL, SERIES, DATE1, TTL_TRD_QNTY, DELIV_QTY\n"
        "FOO, EQ, 04-Sep-2026, 100, 40\n"
    )
    row = parse_sec_bhavdata(text, name="old.csv").row(0, named=True)
    assert row["delivery_pct"] == pytest.approx(0.4)
    assert row["turnover"] is None


def test_the_mto_backend_reports_null_turnover_not_zero() -> None:
    """MTO carries quantities only. `source` is how a consumer tells which."""
    from trader.data.sources.nse_flows import DELIVERY_SCHEMA

    assert {"turnover", "avg_price", "n_trades"} <= set(DELIVERY_SCHEMA)
