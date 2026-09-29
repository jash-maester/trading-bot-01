"""dashboard/data.py on tiny synthetic fixtures. No Streamlit, no real repo files."""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

_spec = importlib.util.spec_from_file_location(
    "dashboard_data", Path(__file__).resolve().parents[2] / "dashboard" / "data.py"
)
D = importlib.util.module_from_spec(_spec)
sys.modules["dashboard_data"] = D  # dataclasses resolve their module by name
_spec.loader.exec_module(D)

SIG, EW = D.SIGNAL_BOOK, D.EW_BOOK


def null(i: int) -> str:
    return f"null_signal_k30_b0.01_rvolstop_s{i}_monthly_20d"


def line(d: str, rets: dict[str, float], run_ts: str = "2026-09-25T15:00:00Z") -> str:
    return json.dumps(
        {"run_ts": run_ts, "date": d, "books": {b: {"nav": 1.0, "ret": r} for b, r in rets.items()}}
    )


def write_record(tmp_path: Path, lines: list[str]) -> Path:
    p = tmp_path / "record.jsonl"
    p.write_text("\n".join(lines) + "\n")
    return p


# ── missing / empty ──────────────────────────────────────────────────────────
def test_missing_record(tmp_path):
    r = D.load_record(tmp_path / "nope.jsonl")
    assert not r.ok and r.value == [] and "does not exist" in r.error


def test_empty_record(tmp_path):
    r = D.load_record(write_record(tmp_path, [""]))
    assert not r.ok and r.value == []
    h = D.headline([])
    assert h["sessions"] == 0 and h["rank"] is None and h["signal"]["rupees"] == 0.0
    assert D.cumulative_series([]).height == 0
    assert D.null_band(D.cumulative_series([])).height == 0
    assert D.session_table([]).height == 0


def test_bad_lines_skipped(tmp_path):
    p = write_record(tmp_path, [line("2026-09-25", {SIG: 0.01}), "{not json", '{"date":"x"}'])
    r = D.load_record(p)
    assert r.ok and len(r.value) == 1 and r.extra["bad_lines"] == 2


def test_missing_json_and_health(tmp_path):
    assert not D.read_json(tmp_path / "x.json").ok
    (tmp_path / "e.json").write_text("")
    assert "empty" in D.read_json(tmp_path / "e.json").error
    assert not D.merge_health(tmp_path / "a.json", tmp_path / "b.json").ok


NOW = datetime(2026, 9, 29, 15, 0, tzinfo=UTC)


def _health(p: Path, at: str, checks: dict[str, str]) -> Path:
    p.write_text(
        json.dumps(
            {
                "checked_at": at,
                "overall": D.worst(checks.values()),
                "checks": [{"check": k, "status": v, "detail": k} for k, v in checks.items()],
            }
        )
    )
    return p


def test_merge_keeps_host_scheduler_fail(tmp_path):
    host = _health(
        tmp_path / "host.json",
        "2026-09-29T10:00:00+00:00",
        {"nse_access": "OK", "scheduler": "FAIL"},
    )
    dash = _health(tmp_path / "dash.json", "2026-09-29T11:00:00+00:00", {"nse_access": "OK"})
    h = D.merge_health(host, dash, now=NOW)
    assert h.ok and h.value["overall"] == "FAIL"
    by = {c["check"]: c for c in h.value["checks"]}
    assert by["scheduler"]["_source"] == str(host)
    assert by["nse_access"]["_source"] == str(dash)
    assert h.value["checked_at"] == "2026-09-29T11:00:00+00:00"
    assert "determinism" in h.value["missing"] and "scheduler" not in h.value["missing"]


def test_merge_newest_check_wins_and_staleness(tmp_path):
    a = _health(tmp_path / "a.json", "2026-09-27T10:00:00+00:00", {"last_run": "FAIL"})
    b = _health(tmp_path / "b.json", "2026-09-27T11:00:00+00:00", {"last_run": "OK"})
    h = D.merge_health(a, b, tmp_path / "missing.json", now=NOW)
    assert h.value["overall"] == "OK"
    assert h.value["stale"] and h.value["age_hours"] == pytest.approx(52.0)
    fresh = D.merge_health(a, b, now=datetime(2026, 9, 27, 12, 0, tzinfo=UTC))
    assert not fresh.value["stale"]


def test_live_checks(tmp_path):
    (tmp_path / "2026-09-29.status").write_text(
        "=== run 2026-09-29T14:40:54Z ===\na start\na DONE data=2026-09-29\n"
    )
    rec = [{"_date": date(2026, 9, 29), "date": "2026-09-29", "books": {}}]
    by = {c["check"]: c["status"] for c in D.live_checks(tmp_path, rec, now=NOW)}
    assert by == {"run_started": "OK", "last_run": "OK", "record_age": "OK"}
    later = datetime(2026, 10, 5, 15, 0, tzinfo=UTC)
    by = {c["check"]: c["status"] for c in D.live_checks(tmp_path, rec, now=later)}
    assert by["run_started"] == "FAIL" and by["record_age"] == "WARN"
    assert {c["check"]: c["status"] for c in D.live_checks(tmp_path / "x", [], now=NOW)} == {
        "run_started": "FAIL",
        "record_age": "WARN",
    }


def test_probe_gate_counts_attempts_and_locks(tmp_path):
    stamp, lock = tmp_path / "probe_stamp", tmp_path / "probe.lock"
    assert D.probe_gate(stamp, lock, NOW, 600, 390)[0]
    assert D.acquire_probe(stamp, lock, NOW, 390)
    assert not D.acquire_probe(stamp, lock, NOW, 390)  # second session: lock held
    ok, _, why = D.probe_gate(stamp, lock, NOW, 600, 390)
    assert not ok and "running" in why
    D.release_probe(lock)  # the check died: no health.json, but the stamp stays
    ok, age, why = D.probe_gate(stamp, lock, NOW, 600, 390)
    assert not ok and why == "rate limit" and age == 0
    assert D.probe_gate(stamp, lock, NOW + timedelta(seconds=601), 600, 390)[0]


# ── P&L arithmetic ───────────────────────────────────────────────────────────
def test_pnl_from_ret_not_nav(tmp_path):
    rets = [0.01, -0.02, 0.005]
    lines = [
        line("2026-09-09", {SIG: 0.5}),  # freeze date: excluded
        *[line(f"2026-09-2{i}", {SIG: r, EW: r / 2}) for i, r in enumerate(rets)],
    ]
    rec = D.load_record(write_record(tmp_path, lines)).value
    h = D.headline(rec)
    cum = sum(rets)
    assert h["sessions"] == 3
    assert h["signal"]["cum_log"] == pytest.approx(cum)
    assert h["signal"]["rupees"] == pytest.approx(1_000_000 * (math.exp(cum) - 1))
    assert h["signal"]["pct"] == pytest.approx(math.exp(cum) - 1)
    assert h["equal_weight"]["cum_log"] == pytest.approx(cum / 2)
    s = D.cumulative_series(rec).filter(pl.col("book") == SIG).sort("date")
    assert s["cum_log"].to_list() == pytest.approx([0.01, -0.01, -0.005])


def test_missing_book_on_a_day_counts_zero_and_is_reported(tmp_path):
    rec = D.load_record(
        write_record(tmp_path, [line("2026-09-25", {SIG: 0.01}), line("2026-09-28", {EW: 0.02})])
    ).value
    assert D.cum_log(rec, SIG) == pytest.approx(0.01)
    assert sorted(D.missing_pairs(rec)) == [("2026-09-25", EW), ("2026-09-28", SIG)]


def test_elapsed_counts_from_first_recorded_session(tmp_path):
    # restart 2: record starts 2026-09-25 though the freeze is 2026-09-09
    lines = [line(d, {SIG: 0.0}) for d in ("2026-09-25", "2026-09-28", "2026-09-29")]
    h = D.headline(D.load_record(write_record(tmp_path, lines)).value)
    assert h["first_date"] == "2026-09-25"
    assert h["months_elapsed"] == pytest.approx(4 / 30.4375)
    assert h["trading_months"] == pytest.approx(3 / 21)


def test_freeze_mismatch():
    assert D.freeze_mismatch({"freeze_date": "2026-09-09"}) is None
    assert "differs" in D.freeze_mismatch({"freeze_date": "2027-01-01"})
    assert "no freeze_date" in D.freeze_mismatch({})
    assert D.freeze_mismatch({"freeze_date": "x"}) is not None


# ── random band and rank ────────────────────────────────────────────────────
def test_null_books_only_volstop_sorted():
    books = {null(10), null(2), "null_signal_k30_b0.01_rnone_s1_monthly_20d", SIG, EW}
    assert D.null_books(books) == [null(2), null(10)]


def test_quantile_matches_linear():
    xs = list(range(1, 11))
    assert D.quantile(xs, 0.5) == pytest.approx(5.5)
    assert D.quantile(xs, 0.1) == pytest.approx(1.9)
    assert D.quantile(xs, 0.9) == pytest.approx(9.1)
    assert math.isnan(D.quantile([], 0.5))


def test_band_and_rank(tmp_path):
    # 20 random books with cum ret 0.001 * i; signal at 0.0155 beats i <= 15
    rets = {null(i): 0.001 * i for i in range(1, 21)}
    rets[SIG] = 0.0155
    rec = D.load_record(write_record(tmp_path, [line("2026-09-25", rets)])).value
    h = D.headline(rec)
    assert h["n_null"] == 20
    assert h["rank"] == 1 + 5  # s16..s20 are strictly greater
    assert h["null_median"]["cum_log"] == pytest.approx(0.0105)
    band = D.null_band(D.cumulative_series(rec))
    assert band.height == 1
    exp10 = 1_000_000 * (math.exp(D.quantile([0.001 * i for i in range(1, 21)], 0.1)) - 1)
    assert band["p10"][0] == pytest.approx(exp10)
    assert band["p10"][0] < band["p50"][0] < band["p90"][0]


def test_rank_ties_count_as_not_better():
    assert D.rank_among(0.01, [0.01, 0.02, 0.0]) == 2


# ── replay, status, memory ──────────────────────────────────────────────────
def test_latest_dir_fallback(tmp_path):
    paper = tmp_path / "paper"
    (paper / "latest").mkdir(parents=True)  # partial: no summary
    assert not D.latest_dir(paper).ok
    snap = paper / "snapshots" / "2026-09-25"
    snap.mkdir(parents=True)
    (snap / "summary.json").write_text("{}")
    got = D.latest_dir(paper)
    assert got.ok and got.value == snap and "note" in got.extra
    (paper / "latest" / "summary.json").write_text("{}")
    assert D.latest_dir(paper).value == paper / "latest"


def test_warmup_navs_cut_at_freeze(tmp_path):
    d = pl.DataFrame(
        {
            "date": [date(2026, 9, 8), date(2026, 9, 9), date(2026, 9, 10)],
            "nav": [1e6, 1.01e6, 1.02e6],
        }
    )
    d.write_parquet(tmp_path / f"nav_{SIG}_full.parquet")
    (tmp_path / "nav_garbage_full.parquet").write_text("not parquet")
    w = D.warmup_navs(tmp_path)
    assert w.height == 2 and w["growth"].to_list() == pytest.approx([1.0, 1.01])
    assert D.warmup_navs(tmp_path / "empty").height == 0


def test_status_blocks(tmp_path):
    (tmp_path / "2026-09-25.status").write_text(
        "=== run 2026-09-25T14:00:00Z ===\na start\na FAIL run_allocator\n"
        "=== run 2026-09-25T15:00:00Z ===\na start\na DONE data=2026-09-25\n"
    )
    (tmp_path / "2026-09-29.status").write_text("=== run 2026-09-29T14:40:54Z ===\na predict OK\n")
    b = D.status_blocks(tmp_path, n=5)
    assert [x["outcome"] for x in b] == ["FAIL", "DONE", "INCOMPLETE"]
    assert b[-1]["started"] == "2026-09-29T14:40:54Z"
    assert D.status_blocks(tmp_path / "none") == []


def test_memory_samples(tmp_path):
    assert not D.memory_samples(tmp_path).ok
    (tmp_path / "mem_2026-09-29.csv").write_text("ts,used_mb,avail_mb\n2026-09-29T14:40:54Z,1,2\n")
    m = D.memory_samples(tmp_path)
    assert m.ok and m.value.height == 1


def test_experiments_missing_and_parsed(tmp_path):
    rows = D.experiments(tmp_path)
    assert all(r["verdict"] == "MISSING" for r in rows)
    (tmp_path / "audit").mkdir()
    (tmp_path / "audit/R_2026-09-29.md").write_text(
        "## E3 — x\n\n**Pass (all):** criterion text\n\n## E4 — y\n\n**Pass (both):** c\n\n"
        "## Results\n\n### E4 — FAIL (today)\n"
    )
    by = {r["id"]: r for r in D.experiments(tmp_path)}
    assert by["E4"]["verdict"] == "FAIL" and "criterion" not in by["E4"]["criterion"]
    assert by["E3"]["verdict"] == "PENDING" and "criterion text" in by["E3"]["criterion"]


def test_experiments_prose_criteria_and_measurement(tmp_path):
    (tmp_path / "audit").mkdir()
    (tmp_path / "audit/R5_VERDICT_PIT.md").write_text(
        "## Verdict: R5 NOT PASSED\n\nThe criterion is a paired interval.\n"
    )
    (tmp_path / "audit/R_OVERNIGHT_2026-09-25.md").write_text(
        "## E0 — x\n\nCriteria: rank corr >= 0.8\n\n## Results\n\n"
        "### E0 — NOT AT PARITY (t)\n\n### E0b — pre-registered t, before running\n\n"
        "**Reading, fixed now:** seed floor rule\n\n"
        "### E0b — the platform is not the cause (t)\n"
    )
    by = {r["id"]: r for r in D.experiments(tmp_path)}
    assert by["R5"]["verdict"] == "FAIL" and "paired interval" in by["R5"]["criterion"]
    assert by["E0b"]["verdict"] == "MEASUREMENT" and "seed floor rule" in by["E0b"]["criterion"]
    assert by["E0"]["verdict"] == "FAIL" and "stands" in by["E0"]["note"]
