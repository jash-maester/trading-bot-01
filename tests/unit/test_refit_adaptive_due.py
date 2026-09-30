"""scripts/refit_adaptive.py: when a refit is due."""
import importlib.util
import sys
from datetime import date
from pathlib import Path

_P = Path(__file__).resolve().parents[2] / "scripts" / "refit_adaptive.py"
_s = importlib.util.spec_from_file_location("refit_adaptive", _P)
ra = importlib.util.module_from_spec(_s)
sys.modules["refit_adaptive"] = ra
_s.loader.exec_module(ra)


def reg(*fits):
    return [{"fit_date": f, "status": "OK"} for f in fits]


def test_due_on_eve_of_rebalance_only():
    r = reg("2026-09-30")                                # served the 1 Oct rebalance
    assert not ra.due(date(2026, 10, 1), r)              # mid-period
    assert not ra.due(date(2026, 10, 14), r)
    assert ra.due(date(2026, 10, 15), r)                 # Thu; Fri 16 Oct opens a new half
    assert not ra.due(date(2026, 10, 16), reg("2026-09-30", "2026-10-15"))
    assert ra.due(date(2026, 10, 30), reg("2026-10-15"))  # Fri; Mon 2 Nov opens November


def test_catch_up_after_a_missed_refit_and_first_run():
    assert ra.due(date(2026, 10, 20), reg("2026-09-30"))  # the 15 Oct refit never ran
    assert ra.due(date(2026, 10, 7), [])
    assert ra.due(date(2026, 10, 7), [{"fit_date": "2026-09-30", "status": "FAIL"}])
