"""P3 adaptive shadow book: one refit on the newest data (audit/P3_ADAPTIVE_SHADOW.md).

    uv run python scripts/refit_adaptive.py                 # data through the store's last session
    uv run python scripts/refit_adaptive.py --data-date 2026-09-30
    uv run python scripts/refit_adaptive.py --dry           # print the window, run nothing
    uv run python scripts/refit_adaptive.py --if-due        # nightly host job: refit only when due

Runs on the HOST (Apple GPU, ~18 min, ~3.5 GB), not in the Docker VM (~60 min
at 7.4 GB). Steps:

1. build a feature panel from the bhavcopy store through the data date
   (data/panels_refit/, rebuilt each time; same pipeline as every panel);
2. train ONE window anchored at the end of the data: train 5 y, purge 3 m,
   val 12 m, purge 3 m, test 1 m ending on the data date. Hyper-parameters are
   `train=r4_pit`, untouched;
3. append the outcome to audit/paper/adaptive/refits.jsonl whatever the gate
   says. The nightly loop (scripts/adaptive_signal.py) uses the newest OK
   refit whose data ends on or before each signal date.

Idempotent: a data date that already has an OK refit is skipped.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

REG = Path("audit/paper/adaptive/refits.jsonl")
PANELS = Path("data/panels_refit")
LOGS = Path("logs/retrain")
MONTHS = 60 + 3 + 12 + 3 + 1  # train 5y + purge + val + purge + test 1m
PANEL_WARMUP_DAYS = 400  # feature warm-up before the first training row


def _add_months(d: date, n: int) -> date:
    y, m = divmod(d.month - 1 + n, 12)
    y, m = d.year + y, m + 1
    for day in (d.day, 30, 29, 28):
        try:
            return date(y, m, day)
        except ValueError:
            continue
    raise ValueError(d)


def _registry() -> list[dict]:
    if not REG.exists():
        return []
    return [json.loads(x) for x in REG.read_text().splitlines() if x.strip()]


def _append(rec: dict) -> None:
    REG.parent.mkdir(parents=True, exist_ok=True)
    with REG.open("a") as fh:
        fh.write(json.dumps(rec) + "\n")


def _store_last_session() -> date:
    import polars as pl  # noqa: PLC0415

    return (
        pl.scan_parquet("data/ext/bhavcopy.parquet").select(pl.col("date").max()).collect().item()
    )


def _run(cmd: list[str], log: Path) -> int:
    with log.open("w") as fh:
        return subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT).returncode


def _next_weekday(d: date) -> date:
    d += timedelta(days=1)
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d


def _serves(d: date) -> tuple[int, int]:
    """The semimonthly rebalance period a refit on data through `d` is for."""
    n = _next_weekday(d)
    return (n.year, n.month * 2 + (n.day >= 16))


def due(data_date: date, registry: list[dict]) -> bool:
    """Refit on the eve of a rebalance (the next weekday opens a new half-month),
    or as a catch-up if the newest OK refit was made for an earlier period."""
    ok = [date.fromisoformat(r["fit_date"]) for r in registry if r.get("status") == "OK"]
    return not ok or _serves(max(ok)) != _serves(data_date)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-date", type=date.fromisoformat, default=None)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--if-due", action="store_true",
                    help="refit only on the eve of a semimonthly rebalance, or to catch up")
    a = ap.parse_args()

    dd = a.data_date or _store_last_session()
    if a.if_due and not due(dd, _registry()):
        print(f"data {dd}: no refit due (next rebalance period already served)")
        return
    tag = f"adaptive_{dd:%Y%m%d}"
    if any(r.get("fit_date") == dd.isoformat() and r.get("status") == "OK" for r in _registry()):
        print(f"{tag}: already refitted -- nothing to do")
        return

    start = _add_months(dd + timedelta(days=1), -MONTHS)
    from trader.training.walk_forward import compute_windows  # noqa: PLC0415

    wins = compute_windows(
        data_start=start,
        data_end=dd,
        train_years=5,
        val_months=12,
        test_months=1,
        purge_months=3,
        step_months=12,
        n_windows=1,
    )
    if len(wins) != 1 or wins[0].test_end != dd:
        raise SystemExit(f"window arithmetic: expected one window ending {dd}, got {wins}")
    w = wins[0]
    print(
        f"{tag}: train {w.train_start}..{w.train_end}  val {w.val_start}..{w.val_end}  "
        f"test {w.test_start}..{w.test_end}"
    )
    if a.dry:
        return

    LOGS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rec = {
        "fit_date": dd.isoformat(),
        "tag": tag,
        "started": datetime.now(UTC).isoformat(timespec="seconds"),
        "window": {
            k: str(getattr(w, k))
            for k in ("train_start", "train_end", "val_start", "val_end", "test_start", "test_end")
        },
    }

    # 1. panel through the data date (split dates only label files; unused here)
    import shutil  # noqa: PLC0415

    shutil.rmtree(PANELS, ignore_errors=True)
    pstart = start - timedelta(days=PANEL_WARMUP_DAYS)
    rc = _run(
        [
            sys.executable,
            "scripts/build_features.py",
            "data=bhav_v1",
            f"data.panels_root={PANELS}",
            f"data.start_date={pstart}",
            f"data.end_date={dd}",
            f"+data.train_end={w.val_end}",
            f"+data.val_start={w.test_start}",
            f"+data.val_end={dd}",
            f"+data.test_start={dd}",
        ],
        LOGS / f"{tag}.panel.log",
    )
    if rc != 0:
        _append(
            {
                **rec,
                "status": "FAIL",
                "error": f"build_features exited {rc}",
                "wall_s": round(time.time() - t0),
            }
        )
        raise SystemExit(f"build_features failed; see {LOGS / (tag + '.panel.log')}")

    # 2. train one end-anchored window
    log = LOGS / f"{tag}.train.log"
    rc = _run(
        [
            sys.executable,
            "scripts/train_signal.py",
            "data=bhav_v1",
            "train=r4_pit",
            "model=signal",
            f"data.panels_root={PANELS}",
            f"train.tag={tag}",
            "walk.n_windows=1",
            "walk.test_months=1",
            f"walk.data_start={start}",
            f"walk.data_end={dd}",
        ],
        log,
    )
    wall = round(time.time() - t0)
    if rc != 0:
        _append({**rec, "status": "FAIL", "error": f"train_signal exited {rc}", "wall_s": wall})
        raise SystemExit(f"train_signal failed; see {log}")

    # 3. register, whatever the gate says
    sig = Path("data/signal") / tag
    gate = json.loads((sig / "gate.json").read_text()) if (sig / "gate.json").exists() else {}
    index = json.loads((sig / "index.json").read_text()) if (sig / "index.json").exists() else {}
    runs = re.findall(r"runs/([0-9a-f]{32})", log.read_text())
    # prediction needs signal_model.pt, index.json and summary.json only; the
    # embeddings and per-window copies are ~466 MB a refit, 24 refits a year
    (sig / "embeddings.npy").unlink(missing_ok=True)
    shutil.rmtree(sig / "windows", ignore_errors=True)
    _append(
        {
            **rec,
            "status": "OK",
            "wall_s": wall,
            "finished": datetime.now(UTC).isoformat(timespec="seconds"),
            "gate_verdict": gate.get("verdict"),
            "encoder_sha": index.get("encoder_state_sha256"),
            "mlflow_runs": runs,
            "signal_dir": str(sig),
        }
    )
    print(f"{tag}: OK in {wall} s; gate {gate.get('verdict')!r} (recorded, not used)")


if __name__ == "__main__":
    main()
