#!/usr/bin/env python
"""Recompute an R4 run's gate verdict from its ``summary.json`` — no retraining.

The gate is a pure function of the per-window OOS metrics, and ``summary.json``
carries every one of them.  When the rule changes (``12_gate_decision.md``), a
finished run can be re-judged in seconds instead of hours, and the GPU stays
free.

    uv run python scripts/regate.py data/signal/r4_v2            # print only
    uv run python scripts/regate.py data/signal/r4_v2 --write    # rewrite gate.json

``--write`` keeps the previous ``gate.json`` beside the new one as
``gate.pre_regate.json`` the first time, updates ``summary.json``'s ``gate``
block, and stamps both with the rule name and the summary they came from.
Nothing else in the artefact directory is touched.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import shutil
from pathlib import Path
from typing import Any

from loguru import logger

from trader.training.supervised import (
    GATE_RULE,
    HorizonMetrics,
    format_verdict,
    gate_json_payload,
    gate_verdict,
)

_HM_FIELDS = {f.name for f in dataclasses.fields(HorizonMetrics)} - {"daily"}


def _metrics(d: dict[str, Any]) -> HorizonMetrics:
    """A ``HorizonMetrics`` from its ``to_dict`` form (``daily`` is not stored)."""
    kw = {k: v for k, v in d.items() if k in _HM_FIELDS}
    for k in ("mean_ic", "std_ic", "icir", "t_stat", "hit_rate", "ci_lo", "ci_hi",
              "decile_spread", "ic_acf_lag1", "n_eff"):
        if kw.get(k) is None:
            kw[k] = float("nan")
    return HorizonMetrics(**kw)


def _window_name(w: dict[str, Any]) -> str:
    win = w["window"]
    return str(win["name"]) if isinstance(win, dict) else str(win)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("signal_dir", type=Path, help="data/signal/<tag>")
    ap.add_argument("--write", action="store_true", help="rewrite gate.json and summary.json")
    args = ap.parse_args()

    summary_path = args.signal_dir / "summary.json"
    if not summary_path.exists():
        raise SystemExit(f"{summary_path} not found — nothing to re-gate.")
    summary = json.loads(summary_path.read_text())

    per_window: dict[str, dict[int, HorizonMetrics]] = {
        _window_name(w): {int(h): _metrics(m) for h, m in w["test"].items()}
        for w in summary["windows"]
    }
    pooled = {int(h): _metrics(m) for h, m in summary["pooled_test"].items()}

    previous = summary.get("gate", {})
    gate = gate_verdict(per_window)
    payload = gate_json_payload(gate, pooled, n_windows=len(per_window))
    payload["regated_from"] = str(summary_path)

    logger.info(
        f"{summary.get('tag', args.signal_dir.name)}: previous verdict "
        f"{'PASS' if previous.get('passed') else 'FAIL'} "
        f"(rule {previous.get('rule', 'every-window/v1')}) -> "
        f"{'PASS' if gate.passed else 'FAIL'} (rule {GATE_RULE})"
    )
    print(format_verdict(gate))

    if not args.write:
        logger.info("dry run — pass --write to update gate.json and summary.json")
        return

    gate_path = args.signal_dir / "gate.json"
    backup = args.signal_dir / "gate.pre_regate.json"
    if gate_path.exists() and not backup.exists():
        shutil.copy2(gate_path, backup)
        logger.info(f"previous gate kept at {backup}")
    gate_path.write_text(json.dumps(payload, indent=2) + "\n")
    summary["gate"] = gate.to_dict()
    summary["gate_regated_from"] = str(summary_path)
    summary_path.write_text(json.dumps(summary, indent=2))
    logger.info(f"wrote {gate_path} and updated {summary_path}")


if __name__ == "__main__":
    main()
