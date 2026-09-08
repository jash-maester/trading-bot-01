"""Which names a run may trade, and saying so.

Both `run_baselines.py` and `run_allocator.py` defaulted to `active_tickers()`.
Pointed at the rebuilt point-in-time panel, either would have measured the OLD
fixed 504 names on the new bars and printed a table that looks entirely normal
while answering a question nobody asked. That nearly wasted Phase 1 of the
survivorship rebuild, so the logic lives in one place and is pinned here.
"""
from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from trader.data.universe import active_tickers, resolve_traded_universe


def _panel(tmp_path, tickers: list[str]):  # noqa: ANN001, ANN202
    p = tmp_path / "panel.parquet"
    pl.DataFrame({
        "date": [date(2020, 1, 1)] * len(tickers),
        "ticker": tickers,
    }).write_parquet(p)
    return p


def test_the_default_is_the_fixed_list(tmp_path) -> None:  # noqa: ANN001
    """Every number already recorded must reproduce, so this cannot change."""
    p = _panel(tmp_path, ["ZZZ.NS", "AAA.NS"])
    names, prov = resolve_traded_universe(p, from_panel=False)
    assert names == active_tickers()
    assert "active_tickers()" in prov


def test_from_panel_takes_every_column(tmp_path) -> None:
    """A point-in-time panel carries names outside the fixed list on purpose.

    Per-date eligibility lives in `is_tradeable`, not in the ticker list, so
    the universe must be the panel's full width or the env cannot see the
    names the rule admits.
    """
    p = _panel(tmp_path, ["ZZZ.NS", "AAA.NS", "MID.NS"])
    names, prov = resolve_traded_universe(p, from_panel=True)
    assert names == ["AAA.NS", "MID.NS", "ZZZ.NS"], "not sorted or not deduped"
    assert str(p) in prov and "3 tickers" in prov


def test_the_two_modes_actually_differ(tmp_path) -> None:
    """A flag that changes nothing is worse than no flag: it reads as covered."""
    p = _panel(tmp_path, ["NOTAREALNAME1.NS", "NOTAREALNAME2.NS"])
    from_panel, _ = resolve_traded_universe(p, from_panel=True)
    fixed, _ = resolve_traded_universe(p, from_panel=False)
    assert from_panel != fixed


def test_provenance_is_returned_so_a_run_can_state_it(tmp_path) -> None:
    """A run that does not say which universe it used cannot be compared."""
    p = _panel(tmp_path, ["AAA.NS"])
    for flag in (True, False):
        _, prov = resolve_traded_universe(p, from_panel=flag)
        assert prov and "tickers" in prov


def test_an_empty_panel_is_refused(tmp_path) -> None:
    """Zero tickers would build an env that can hold nothing and report 0% return."""
    p = tmp_path / "empty.parquet"
    pl.DataFrame(schema={"date": pl.Date, "ticker": pl.Utf8}).write_parquet(p)
    with pytest.raises(ValueError, match="no tickers"):
        resolve_traded_universe(p, from_panel=True)
